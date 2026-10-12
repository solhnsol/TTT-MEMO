"""Direct completion versus QA from the same reproduced K=4 fast weights.
No streaming-retention claim: write KV and hidden states are discarded, and
only the native updated prime FFN crosses to a fresh recall context.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
nvrtc = ROOT / '.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib'
os.environ['LD_LIBRARY_PATH'] = str(nvrtc) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import importlib.util
import json
import sys
import time
sys.path.insert(0, str(ROOT / 'scripts/exp01_gold_evidence_qa'))
sys.path.insert(0, str(ROOT / 'scripts/baseline_125m/common'))
sys.path.insert(0, str(ROOT / 'e2e'))
from model_runtime import load_tokenizer, snapshot_fast_weights, _hash_pytree
from runtime_helpers import describe_gpu

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf
from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
from ttt.model.sharding import ModelSharding
from ttt.model.transformer import MetaModel, BlockCollectionSplit
from ttt.model.data import Batch
from ttt.utils.jax_utils import eval_shape_and_sharding, initialize_distibuted, set_random_seed
from ttt.utils.filter_utils import get_filter_spec


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

binding = import_file('binding_metrics', ROOT/'scripts/exp02a1_binding_specificity/metrics.py')
config3b = import_file('baseline_3b', ROOT/'scripts/baseline_3b/common/baseline_config.py')


def clone_state(s):
    return jax.tree.map(lambda x: x, s)


def load():
    # Mirrors the existing Exp01 loader, retaining state_all needed by native SGD.
    cfg = config3b.compose_3b_128k_config()
    initialize_distibuted(cfg.backend)
    cfg.model.seq_len = cfg.training.seq_length
    key = set_random_seed(cfg.training.model_seed)
    mesh = ModelSharding(cfg).mesh
    shard = ModelSharding(cfg)
    @eqx.filter_jit
    def create():
        m, s = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        s = jax.device_put(s, jax.NamedSharding(mesh, P()))
        return shard.shard_params(m), s
    abstract = eval_shape_and_sharding(lambda: create()[0].weights())
    ckpt = Checkpointer(config=cfg, for_saving=False)
    assert ckpt.checkpoint_exists()
    restored = ckpt.load_checkpoint(step=cfg.training.resume_step, targets={'model_weights': abstract}, restore=cfg.training.load_part)
    fresh, state = create()
    model, missing = unify_dict_with_eqx_module(restored['model_weights'], fresh)
    assert not missing, missing
    del fresh, restored
    state = state.set(model.step_index, jnp.asarray(jnp.iinfo(jnp.int32).max-100, dtype=jnp.int32))
    prefix_state, suffix_state = BlockCollectionSplit.split_state(state.substate(model.language_model.model.h.blocks), cfg.model.suffix_len)
    collection = BlockCollectionSplit(cfg.model, block_collection=model.language_model.model.h.blocks,
                                      prime_storage=model.language_model.model.h.prime_storage, key=jax.random.PRNGKey(0))
    model = eqx.tree_at(lambda m:m.language_model.model.h, model, collection)
    model = jax.tree.map(lambda x:x.astype(model.state_dtype), model)
    return cfg, mesh, model, state, prefix_state, suffix_state


def encode(tokenizer, text, bos, eos, chunk, candidate_span=None):
    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = [bos]+enc.ids
    assert len(ids) <= chunk
    # Batch target_tokens already shifted; candidate token at i is scored at logits i-1.
    selected = list(range(1, len(ids))) if candidate_span is None else [i+1 for i,(a,b) in enumerate(enc.offsets) if b>a and a<candidate_span[1] and b>candidate_span[0]]
    assert selected and min(selected)>0
    inputs = ids+[eos]*(chunk-len(ids))
    targets = inputs[1:]+[eos]
    mask = [False]*chunk
    for i in selected:mask[i-1]=True
    return Batch(input_ids=jnp.asarray(inputs,dtype=jnp.int32), target_tokens=jnp.asarray(targets,dtype=jnp.int32),loss_masks=jnp.asarray(mask)), {'text':text,'n_real_tokens':len(ids),'n_loss_tokens':len(selected),'candidate_token_ids':[ids[i] for i in selected],'selected_logit_positions':[i-1 for i in selected]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=4,choices=(1,4,8));ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    t0=time.monotonic()
    reference=ROOT/'experiments/exp02a4_ttt_e2e_binding/update_budget_k4_20261012'
    old_cfg=json.loads((reference/'config.json').read_text())
    old_records={r['pair_id']:r for r in map(json.loads,(reference/'samples.jsonl').read_text().splitlines())}
    reference_hashes={f:hashlib.sha256((reference/f).read_bytes()).hexdigest() for f in ('dataset.jsonl','config.json','samples.jsonl','summary.json')}
    path=ROOT/'experiments/exp02a4_ttt_e2e_binding/calibration_20261012/dataset_matched.jsonl'
    # Reuse the archived calibration dataset unchanged; no paraphrase evaluation.
    pairs=[json.loads(l) for l in path.read_text().splitlines()]
    pairs=[p for p in pairs if p['info_type']=='exact' and p['smoke']][:args.n]
    assert len(pairs)==args.n
    (args.out/'dataset.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    print('Loading official 3B checkpoint',flush=True)
    cfg,mesh,base,state_all,state_prefix,state_suffix=load()
    tok=load_tokenizer();chunk=int(cfg.model.mini_batch_size);bos=int(cfg.model.bos_token_id);eos=int(cfg.model.eos_token_id)
    fast_initial_hash,_=_hash_pytree(base.inner_parameters())
    inner_filter=get_filter_spec(base,cfg.training.spec_inner,'inner parameters')
    outer=eqx.partition(base,inner_filter)[1]
    frozen_hash,_=_hash_pytree(outer)
    n_inner=sum(x.size for x in jax.tree.leaves(base.inner_parameters()) if eqx.is_inexact_array(x))
    settings={'experiment':'exp02a4_completion_vs_qa','n':args.n,'checkpoint':str(config3b.CHECKPOINT_DIR),'config':OmegaConf.to_container(cfg,resolve=True),
              'gpu':describe_gpu().__dict__,'source_dataset':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'prompt':'lexical QA identical; direct Fact: wrapper plus entity-specific write sentence prefix',
              'initial_fast_hash':fast_initial_hash,'frozen_hash':frozen_hash,'n_inner_parameters':n_inner,
              'write':'Fact: wrapper plus original two-fact block; four independent-KV native chunk updates with fast weights carried forward; all real write tokens except BOS are labeled; padding masked',
              'readout':'fresh Context/Question/Answer base-LM prompt; no write KV or hidden states; direct source eviction, not filler retention',
              'calibration_reference':'calibration_20261012 matched_T3 lexical_near','updates_per_mapping':4,'native_chunk_tokens':chunk,'source_exposures':4,
              'candidate_scoring':'joint tokenization; causal shift; candidate-only logp; EOS excluded; mean primary, sum diagnostic'}
    assert fast_initial_hash==old_cfg['initial_fast_hash'] and frozen_hash==old_cfg['frozen_hash']
    assert (args.out/'dataset.jsonl').read_bytes()==(reference/'dataset.jsonl').read_bytes()
    settings.update(fast_weights_retrained=True,fast_weight_checkpoint_available=False,reference=str(reference),reference_sha256=reference_hashes,write_kv_reset_each_update=True,optimizer_note='native SGD lr/clip unchanged; optimizer initialized once per mapping and returned state carried across updates',same_fact_reexposed_every_update=True)
    (args.out/'config.json').write_text(json.dumps(settings,indent=2,default=str))
    # Fresh suffix state: every cache array and chunk counter must be zero.
    assert all(bool(jnp.all(x==0)) for x in jax.tree.leaves(state_suffix))

    @eqx.filter_jit
    def prefix(m,seq):
        x=m.language_model.wte_call(seq.input_ids)
        return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,x,clone_state(state_prefix),seq).last_hidden_state

    @eqx.filter_jit
    def update(m,seq,pref,opt_state):
        return MetaModel.inner_loop_step(m,opt_state,(clone_state(state_all),clone_state(state_suffix)),seq,pref)

    @eqx.filter_jit
    def score(m,seq):
        pref=prefix(m,seq)
        out=m.language_model.suffix_call(prefix_outputs=pref,state=clone_state(state_suffix),seq=seq)
        lp=jax.nn.log_softmax(out.logits.astype(jnp.float32),axis=-1)
        token_lp=jnp.take_along_axis(lp,seq.target_tokens[:,None],axis=-1)[:,0]
        total=jnp.sum(jnp.where(seq.loss_masks,token_lp,0))
        return total,total/jnp.sum(seq.loss_masks),token_lp

    @eqx.filter_jit
    def deltas(m, reference):
        # Full frozen parameter check, not just selected projections.
        current_outer=eqx.partition(m,inner_filter)[1]
        diffs=[jnp.max(jnp.abs(a-b)) for a,b in zip(jax.tree.leaves(current_outer),jax.tree.leaves(eqx.partition(reference,inner_filter)[1])) if eqx.is_inexact_array(a)]
        inner_diffs=[jnp.max(jnp.abs(a-b)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(reference.inner_parameters())) if eqx.is_inexact_array(a)]
        return jnp.max(jnp.stack(diffs)),jnp.max(jnp.stack(inner_diffs))

    def evaluate(m,pair,source=None):
        matrices={};detail={}
        for style in ('lexical_near','direct_completion'):
            matrices[style]={e:{} for e in ('e1','e2')};detail[style]={}
            for e in ('e1','e2'):
                q=pair['questions'][e]['lexical_near']
                context='' if source is None else pair['write_block_'+source]
                prompt='Read the context and answer the question with a short answer.\n\nContext:\n'+context+'\n\nQuestion:\n'+q+'\n\nAnswer:'
                if style=='direct_completion':
                    assert source is None
                    prompt='Fact:\n'+pair['write_template'].split('{payload}')[0].format(entity=pair[e]).rstrip()
                    assert prompt=='Fact:\nThe passphrase assigned to '+pair[e]+' is'
                if source is None:
                    assert pair['v1'] not in prompt and pair['v2'] not in prompt
                    assert all(s not in prompt for s in pair['sentences_A']+pair['sentences_B'])
                detail[style][e]={}
                for v in ('v1','v2'):
                    # Single leading space matches native base-LM answer continuation.
                    text=prompt+' '+pair[v]
                    start=len(prompt)+1
                    seq,meta=encode(tok,text,bos,eos,chunk,(start,len(text)))
                    assert tok.decode(meta['candidate_token_ids'],skip_special_tokens=False).strip()==pair[v]
                    total,mean,token_lp=score(m,seq)
                    total=float(total);mean=float(mean)
                    selected=np.asarray(token_lp)[meta['selected_logit_positions']]
                    assert np.isfinite(selected).all()
                    assert abs(float(selected.sum())-total)<1e-4
                    assert abs(float(selected.mean())-mean)<1e-4
                    matrices[style][e][v]={'mean':mean,'sum':total}
                    detail[style][e][v]={**meta,'sum_logp':total,'mean_logp':mean,'token_logp':selected.tolist()}
                assert detail[style][e]['v1']['n_loss_tokens']==detail[style][e]['v2']['n_loss_tokens']
        return matrices,detail

    records=[]
    with mesh:
        for pair in pairs:
            print('Pair '+pair['pair_id'],flush=True)
            previous=old_records[pair['pair_id']]
            matrix={};details={};writes={};reproduction={}
            matrix['static'],details['static']=evaluate(base,pair)
            for mapping in ('A','B'):
                write='Fact:\n'+pair['write_block_'+mapping]
                seq,meta=encode(tok,write,bos,eos,chunk)
                pref=prefix(base,seq)
                adapted=base
                opt_state=base.inner_optimizer(clone_state(state_all)).init(base.inner_parameters())
                history=[]
                for step in range(1,5):
                    result=update(adapted,seq,pref,opt_state)
                    jax.block_until_ready(result.metrics)
                    frozen_step,fast_step=map(float,deltas(result.new_model,adapted))
                    frozen_delta,fast_delta=map(float,deltas(result.new_model,base))
                    assert frozen_step==frozen_delta==0 and fast_step>0
                    adapted=result.new_model
                    opt_state=result.new_optimizer_state
                    loss_pre=float(result.metrics[MetaModel.MetricType.loss])
                    loss_error=loss_pre-previous['write'][mapping]['trajectory'][step-1]['loss_pre']
                    _,post,_=score(adapted,seq)
                    loss_post=-float(post)
                    history.append({'step':step,'loss_pre':loss_pre,'loss_post':loss_post,'archived_loss_pre_difference':loss_error,'fast_step_max_abs_delta':fast_step,'fast_cumulative_max_abs_delta':fast_delta,'frozen_max_abs_delta':frozen_delta,'same_fact_reexposed':True,'fresh_write_kv':True})
                    del result
                    print('  update '+mapping+' '+str(step)+' loss '+str(loss_pre),flush=True)
                writes[mapping]={**meta,'native_updates':4,'trajectory':history,'total_loss_tokens':4*meta['n_loss_tokens']}
                del seq,pref,opt_state
                # BOTH probes use this exact same adapted object; readout states are discarded.
                matrix['ttt_'+mapping],details['ttt_'+mapping]=evaluate(adapted,pair)
                frozen_after,fast_after=map(float,deltas(adapted,base))
                assert frozen_after==0 and fast_after==fast_delta
                err=max(abs(matrix['ttt_'+mapping]['lexical_near'][e][v][f]-previous['matrices']['ttt_'+mapping]['lexical_near'][e][v][f]) for e in ('e1','e2') for v in ('v1','v2') for f in ('mean','sum'))
                print('  archived QA max score error '+str(err),flush=True)
                reproduction[mapping]={'max_QA_score_abs_error':err,'same_fast_state_both_probes':True,'fast_state_unchanged_after_readout':True}
                del adapted
            metrics={}
            for style in ('lexical_near','direct_completion'):
                metrics[style]={}
                for field in ('mean','sum'):
                    def mat(c):return {e:{v:matrix[c][style][e][v][field] for v in ('v1','v2')} for e in ('e1','e2')}
                    ma,mb=mat('ttt_A'),mat('ttt_B');static=mat('static')
                    metrics[style][field]={'ttt':binding.b_swap(ma,mb),'flips':binding.flip_diagnostics(ma,mb),'static':binding.b_swap(static,static),'static_flips':binding.flip_diagnostics(static,static),
                        'margins':{m:{e:matrix['ttt_'+m][style][e]['v1' if (m=='A')==(e=='e1') else 'v2'][field]-matrix['ttt_'+m][style][e]['v2' if (m=='A')==(e=='e1') else 'v1'][field] for e in ('e1','e2')} for m in ('A','B')}}
            rec={'pair_id':pair['pair_id'],'write':writes,'matrices':matrix,'candidate_details':details,'metrics':metrics,'reproduction':reproduction}
            records.append(rec)
            with (args.out/'samples.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            print(json.dumps({'pair':pair['pair_id'],'metrics':metrics}),flush=True)
        assert _hash_pytree(base.inner_parameters())[0]==fast_initial_hash
        assert _hash_pytree(eqx.partition(base,inner_filter)[1])[0]==frozen_hash
    summary={'n':len(records),'invariants_passed':True,'fast_weights_retrained':True,'elapsed_seconds':time.monotonic()-t0,'jax_memory_stats':jax.devices()[0].memory_stats(),'total_inner_updates':8*len(records),'total_write_loss_tokens':sum(w['total_loss_tokens'] for r in records for w in r['write'].values()),'results':{}}
    for style in ('lexical_near','direct_completion'):
        summary['results'][style]={}
        for field in ('mean','sum'):
            summary['results'][style][field]={}
            for label,bkey,fkey in [('source_free_no_update','static','static_flips'),('source_free_K4','ttt','flips')]:
                ms=[r['metrics'][style][field] for r in records];fs=[m[fkey] for m in ms]
                summary['results'][style][field][label]={'A_correct':sum(f['e1_prefers_gold_A']+f['e2_prefers_gold_A'] for f in fs),'B_correct':sum(f['e1_prefers_gold_B']+f['e2_prefers_gold_B'] for f in fs),'n_answers_per_mapping':2*len(records),'both_flip_count':sum(f['both_flipped'] for f in fs),'single_entity_flip_count':sum(f['e1_flipped']+f['e2_flipped'] for f in fs),'mean_B_A':float(np.mean([m[bkey]['B_A'] for m in ms])),'mean_B_B':float(np.mean([m[bkey]['B_B'] for m in ms])),'mean_B_swap':float(np.mean([m[bkey]['B_swap'] for m in ms]))}
    assert all(hashlib.sha256((reference/f).read_bytes()).hexdigest()==h for f,h in reference_hashes.items())
    summary['reference_sha256']=reference_hashes
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=='__main__':main()
