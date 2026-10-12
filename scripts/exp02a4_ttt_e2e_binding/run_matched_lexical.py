"""Length-matched lexical-only native TTT-E2E 3B binding pilot.
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
    settings={'experiment':'exp02a4_matched_lexical_source_free','n':args.n,'checkpoint':str(config3b.CHECKPOINT_DIR),'config':OmegaConf.to_container(cfg,resolve=True),
              'gpu':describe_gpu().__dict__,'source_dataset':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'prompt':'calibration T3 lexical, identical; source-free context empty',
              'initial_fast_hash':fast_initial_hash,'frozen_hash':frozen_hash,'n_inner_parameters':n_inner,
              'write':'Fact: wrapper plus original two-fact block; one native chunk update; all real write tokens except BOS are labeled; padding masked',
              'readout':'fresh Context/Question/Answer base-LM prompt; no write KV or hidden states; direct source eviction, not filler retention',
              'calibration_reference':'calibration_20261012 matched_T3 lexical_near','updates_per_mapping':1,'native_chunk_tokens':chunk,'source_exposures':1,
              'candidate_scoring':'joint tokenization; causal shift; candidate-only logp; EOS excluded; mean primary, sum diagnostic'}
    (args.out/'config.json').write_text(json.dumps(settings,indent=2,default=str))
    # Fresh suffix state: every cache array and chunk counter must be zero.
    assert all(bool(jnp.all(x==0)) for x in jax.tree.leaves(state_suffix))

    @eqx.filter_jit
    def prefix(m,seq):
        x=m.language_model.wte_call(seq.input_ids)
        return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,x,clone_state(state_prefix),seq).last_hidden_state

    @eqx.filter_jit
    def update(m,seq,pref):
        optimizer=m.inner_optimizer(clone_state(state_all))
        opt_state=optimizer.init(m.inner_parameters())
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
        for style in ('lexical_near',):
            matrices[style]={e:{} for e in ('e1','e2')};detail[style]={}
            for e in ('e1','e2'):
                q=pair['questions'][e][style]
                context='' if source is None else pair['write_block_'+source]
                prompt='Read the context and answer the question with a short answer.\n\nContext:\n'+context+'\n\nQuestion:\n'+q+'\n\nAnswer:'
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
            matrix={};details={};writes={}
            print('  no-update source-free scoring',flush=True)
            matrix['static'],details['static']=evaluate(base,pair)
            for mapping in ('A','B'):
                print('  source-present '+mapping,flush=True)
                matrix['source_present_'+mapping],details['source_present_'+mapping]=evaluate(base,pair,source=mapping)
                # Same immutable model and state for every pair/mapping.
                write='Fact:\n'+pair['write_block_'+mapping]
                assert all(q not in write for qs in pair['questions'].values() for q in qs.values())
                assert tuple(map(float,deltas(base,base)))==(0.0,0.0)
                seq,meta=encode(tok,write,bos,eos,chunk)
                pref=prefix(base,seq)
                t_write=time.monotonic()
                print('  native update '+mapping,flush=True)
                result=update(base,seq,pref)
                jax.block_until_ready(result.metrics)
                frozen_delta,fast_delta=map(float,deltas(result.new_model,base))
                assert frozen_delta==0 and fast_delta>0,(frozen_delta,fast_delta)
                writes[mapping]={**meta,'native_updates':1,'loss_pre':float(result.metrics[MetaModel.MetricType.loss]),'fast_max_abs_delta':fast_delta,'frozen_max_abs_delta':frozen_delta,'seconds':time.monotonic()-t_write}
                adapted=result.new_model
                # Discard write KV, optimizer, source tensors and source prefix hidden states.
                del result,seq,pref
                print('  source-evicted readout '+mapping,flush=True)
                matrix['ttt_'+mapping],details['ttt_'+mapping]=evaluate(adapted,pair)
                frozen_after,fast_after=map(float,deltas(adapted,base))
                assert frozen_after==0 and fast_after==fast_delta
                del adapted
            metrics={}
            for style in ('lexical_near',):
                metrics[style]={}
                for field in ('mean','sum'):
                    def mat(c):return {e:{v:matrix[c][style][e][v][field] for v in ('v1','v2')} for e in ('e1','e2')}
                    ma,mb=mat('ttt_A'),mat('ttt_B')
                    spa,spb=mat('source_present_A'),mat('source_present_B')
                    metrics[style][field]={'ttt':binding.b_swap(ma,mb),'flips':binding.flip_diagnostics(ma,mb),
                        'source_present':binding.b_swap(spa,spb),'source_present_flips':binding.flip_diagnostics(spa,spb),
                        'static_B_A':binding.binding_score_from_matrix(mat('static'),'A'),
                        'static':binding.b_swap(mat('static'),mat('static')),
                        'static_flips':binding.flip_diagnostics(mat('static'),mat('static'))}
                    assert metrics[style][field]['source_present_flips']['both_flipped']==1, 'lexical calibration failed'
            rec={'pair_id':pair['pair_id'],'write':writes,'matrices':matrix,'candidate_details':details,'metrics':metrics}
            records.append(rec)
            with (args.out/'samples.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            print(json.dumps({'pair':pair['pair_id'],'metrics':metrics}),flush=True)
        # Original initialization retained unchanged throughout.
        assert _hash_pytree(base.inner_parameters())[0]==fast_initial_hash
        assert _hash_pytree(eqx.partition(base,inner_filter)[1])[0]==frozen_hash
    summary={'n':len(records),'invariants_passed':True,'elapsed_seconds':time.monotonic()-t0,
             'jax_memory_stats':jax.devices()[0].memory_stats(),'n_inner_parameters':n_inner,
             'total_inner_updates':2*len(records),'total_write_loss_tokens':sum(w['n_loss_tokens'] for r in records for w in r['write'].values()),
             'source_eviction':'fresh recall context + zero initial KV/chunk counter; no filler retention measured','results':{}}
    for style in ('lexical_near',):
        summary['results'][style]={}
        for field in ('mean','sum'):
            ms=[r['metrics'][style][field] for r in records]
            bs=[m['ttt']['B_swap'] for m in ms]
            summary['results'][style][field]={'mean_B_swap':float(np.mean(bs)),'n_positive_B_swap':sum(b>0 for b in bs),
                'both_entities_flip_rate':float(np.mean([m['flips']['both_flipped'] for m in ms])),
                'mapping_A_both_correct_rate':float(np.mean([m['flips']['e1_prefers_gold_A'] and m['flips']['e2_prefers_gold_A'] for m in ms])),
                'mapping_B_both_correct_rate':float(np.mean([m['flips']['e1_prefers_gold_B'] and m['flips']['e2_prefers_gold_B'] for m in ms])),
                'source_present_B_swap':float(np.mean([m['source_present']['B_swap'] for m in ms])),
                'source_present_both_flip_rate':float(np.mean([m['source_present_flips']['both_flipped'] for m in ms])),
                'static_mean_B_A':float(np.mean([m['static_B_A'] for m in ms]))}
            conditions={}
            for label,bkey,fkey in [('source_present_no_update','source_present','source_present_flips'),('source_free_no_update','static','static_flips'),('source_free_native_ttt','ttt','flips')]:
                fs=[m[fkey] for m in ms]
                conditions[label]={'A_correct':sum(f['e1_prefers_gold_A']+f['e2_prefers_gold_A'] for f in fs),
                    'B_correct':sum(f['e1_prefers_gold_B']+f['e2_prefers_gold_B'] for f in fs),
                    'n_answers_per_mapping':2*len(records),
                    'A_accuracy':float(np.mean([f['e1_prefers_gold_A']+f['e2_prefers_gold_A'] for f in fs]))/2,
                    'B_accuracy':float(np.mean([f['e1_prefers_gold_B']+f['e2_prefers_gold_B'] for f in fs]))/2,
                    'both_flip_count':sum(f['both_flipped'] for f in fs),
                    'both_flip_rate':float(np.mean([f['both_flipped'] for f in fs])),
                    'mean_B_swap':float(np.mean([m[bkey]['B_swap'] for m in ms]))}
            summary['results'][style][field]['conditions']=conditions
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=='__main__':main()
