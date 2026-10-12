"""Source-present-only 2x2 diagnostic: original/matched values x T3/T2.
Original/T3 uses archived scores; all other cells run no-update native readout.
Primary gate is matched/T3: 100% per-mapping and both flip, both styles.
T2 is a fixed diagnostic, not selected by optimizing over prompt variants.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import run_pilot as native
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

ROOT=native.ROOT


def prompt_for(pair,mapping,style,entity,template):
    ctx=pair['write_block_'+mapping];q=pair['questions'][entity][style]
    if template=='T3':
        return 'Read the context and answer the question with a short answer.\n\nContext:\n'+ctx+'\n\nQuestion:\n'+q+'\n\nAnswer:'
    assert template=='T2'
    return ctx+'\n\nQ: '+q+'\nA:'


def matched_pair(original):
    p=json.loads(json.dumps(original))
    # Determined using tokenizer only, before new model scores. Same entities,
    # question forms, sentence template, entity order, and number of words.
    replacements={'exact-01':('willow ember pebble','willow ember stone'),
                  'exact-03':('pine coral dune','pine coral lake')}
    old,new=replacements.get(p['pair_id'],(None,None))
    def replace(x):
        if isinstance(x,str):return x.replace(old,new) if old is not None else x
        if isinstance(x,list):return [replace(v) for v in x]
        if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
        return x
    p=replace(p)
    p.pop('tokenizer',None)  # Original stats belong to Qwen tokenizer, not Llama3.
    p['calibration_change']={'old_value':old,'new_value':new,'reason':'tokenizer-only one-word change; no score-based selection'}
    return p


def evaluate_metrics(matrix):
    result={}
    for style in ('lexical_near','paraphrase'):
        result[style]={}
        for field in ('mean','sum'):
            ma={e:{v:matrix['A'][style][e][v][field] for v in ('v1','v2')} for e in ('e1','e2')}
            mb={e:{v:matrix['B'][style][e][v][field] for v in ('v1','v2')} for e in ('e1','e2')}
            flips=native.binding.flip_diagnostics(ma,mb)
            result[style][field]={'binding':native.binding.b_swap(ma,mb),'flips':flips}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    source=ROOT/'experiments/exp02a4_ttt_e2e_binding/pilot_20261012'
    pairs=[json.loads(l) for l in (source/'dataset.jsonl').read_text().splitlines()]
    archived={r['pair_id']:r for r in [json.loads(l) for l in (source/'samples.jsonl').read_text().splitlines()]}
    tok=native.load_tokenizer()
    variants={'original':pairs,'matched':[matched_pair(p) for p in pairs]}
    # Tokenization preflight on the complete scoring text, not standalone values.
    audit=[]
    for variant,ps in variants.items():
        for p in ps:
            counts=[]
            for template in ('T3','T2'):
                for mapping in ('A','B'):
                    for style in ('lexical_near','paraphrase'):
                        for e in ('e1','e2'):
                            prompt=prompt_for(p,mapping,style,e,template)
                            ns=[]
                            for v in ('v1','v2'):
                                text=prompt+' '+p[v];offset=len(prompt)+1
                                enc=tok.encode(text,add_special_tokens=False)
                                ids=[t for t,(a,b) in zip(enc.ids,enc.offsets) if b>a and a<len(text) and b>offset]
                                assert tok.decode(ids,skip_special_tokens=False).strip()==p[v]
                                ns.append(len(ids))
                            if variant=='matched':assert ns[0]==ns[1],(p['pair_id'],ns)
                            counts.append({'template':template,'mapping':mapping,'style':style,'entity':e,'lengths':ns})
            audit.append({'variant':variant,'pair_id':p['pair_id'],'value_1':p['v1'],'value_2':p['v2'],'counts':counts})
    (args.out/'dataset_original.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    (args.out/'dataset_matched.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in variants['matched']))
    (args.out/'tokenization_audit.json').write_text(json.dumps(audit,indent=2))
    cfg,mesh,base,state_all,state_prefix,state_suffix=native.load()
    initial,_=native._hash_pytree(base)
    original_config=json.loads((source/'config.json').read_text())
    assert native._hash_pytree(base.inner_parameters())[0]==original_config['initial_fast_hash']
    settings={'experiment':'exp02a4_source_present_calibration','conditions':['original_T3_archived','original_T2','matched_T3','matched_T2'],
              'primary':'matched_T3; mean-token logp','pass_gate':'all 8/8 answers per mapping and 4/4 both-entity flips in both styles',
              'diagnostic_prompt':'T2 fixed before evaluation; no prompt search','native_updates':0,'n_pairs':4,
              'checkpoint':str(native.config3b.CHECKPOINT_DIR),'baseline_samples_sha256':hashlib.sha256((source/'samples.jsonl').read_bytes()).hexdigest(),
              'base_hash_initial':initial,'score':'same native scorer formula/encoder, shifted candidate-only CE; EOS excluded; mean and sum from identical token scores',
              'matched_changes':{p['pair_id']:p['calibration_change'] for p in variants['matched']}}
    (args.out/'config.json').write_text(json.dumps(settings,indent=2))
    chunk=int(cfg.model.mini_batch_size);bos=int(cfg.model.bos_token_id);eos=int(cfg.model.eos_token_id)
    assert all(bool(jnp.all(x==0)) for x in jax.tree.leaves(state_suffix))

    @eqx.filter_jit
    def score(m,seq):
        embed=m.language_model.wte_call(seq.input_ids)
        pref=m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,embed,native.clone_state(state_prefix),seq).last_hidden_state
        out=m.language_model.suffix_call(prefix_outputs=pref,state=native.clone_state(state_suffix),seq=seq)
        lp=jax.nn.log_softmax(out.logits.astype(jnp.float32),axis=-1)
        token_lp=jnp.take_along_axis(lp,seq.target_tokens[:,None],axis=-1)[:,0]
        total=jnp.sum(jnp.where(seq.loss_masks,token_lp,0))
        return total,total/jnp.sum(seq.loss_masks),token_lp

    records=[]
    with mesh:
        for variant,ps in variants.items():
            for template in ('T3','T2'):
                for p in ps:
                    name=variant+'_'+template
                    matrices={};details={}
                    if name=='original_T3':
                        old=archived[p['pair_id']]
                        matrices={m:old['matrices']['source_present_'+m] for m in ('A','B')}
                        details={m:old['candidate_details']['source_present_'+m] for m in ('A','B')}
                    else:
                        for m in ('A','B'):
                            matrices[m]={};details[m]={}
                            for style in ('lexical_near','paraphrase'):
                                matrices[m][style]={};details[m][style]={}
                                for e in ('e1','e2'):
                                    prompt=prompt_for(p,m,style,e,template)
                                    matrices[m][style][e]={};details[m][style][e]={}
                                    for v in ('v1','v2'):
                                        text=prompt+' '+p[v]
                                        seq,meta=native.encode(tok,text,bos,eos,chunk,(len(prompt)+1,len(text)))
                                        assert tok.decode(meta['candidate_token_ids'],skip_special_tokens=False).strip()==p[v]
                                        total,avg,lp=score(base,seq)
                                        total,avg=float(total),float(avg)
                                        token_lp=np.asarray(lp)[meta['selected_logit_positions']]
                                        assert np.isfinite(token_lp).all()
                                        assert abs(float(token_lp.sum())-total)<1e-4 and abs(float(token_lp.mean())-avg)<1e-4
                                        matrices[m][style][e][v]={'mean':avg,'sum':total}
                                        details[m][style][e][v]={**meta,'sum_logp':total,'mean_logp':avg,'token_logp':token_lp.tolist()}
                    record={'cell':name,'pair_id':p['pair_id'],'matrices':matrices,'details':details,'metrics':evaluate_metrics(matrices)}
                    records.append(record)
                    with (args.out/'samples.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print(json.dumps({'cell':name,'pair':p['pair_id'],'metrics':record['metrics']}),flush=True)
        assert native._hash_pytree(base)[0]==initial
    summary={'n_pairs':4,'native_updates':0,'weight_hash_unchanged':True,'elapsed_seconds':time.monotonic()-start,'jax_memory_stats':jax.devices()[0].memory_stats(),'results':{}}
    for cell in ('original_T3','original_T2','matched_T3','matched_T2'):
        rs=[r for r in records if r['cell']==cell];assert len(rs)==4
        summary['results'][cell]={}
        for style in ('lexical_near','paraphrase'):
            summary['results'][cell][style]={}
            for field in ('mean','sum'):
                ms=[r['metrics'][style][field] for r in rs]
                f=[m['flips'] for m in ms]
                a=sum(x['e1_prefers_gold_A']+x['e2_prefers_gold_A'] for x in f)
                b=sum(x['e1_prefers_gold_B']+x['e2_prefers_gold_B'] for x in f)
                flip=sum(x['both_flipped'] for x in f)
                summary['results'][cell][style][field]={'A_correct':a,'B_correct':b,'n_answers_per_mapping':8,'A_accuracy':a/8,'B_accuracy':b/8,
                    'both_flip_count':flip,'both_flip_rate':flip/4,'mean_B_swap':float(np.mean([m['binding']['B_swap'] for m in ms])),
                    'gate_passed':a==8 and b==8 and flip==4}
    summary['primary_gate_passed']=all(summary['results']['matched_T3'][s]['mean']['gate_passed'] for s in ('lexical_near','paraphrase'))
    summary['diagnostic_T2_gate_passed']=all(summary['results']['matched_T2'][s]['mean']['gate_passed'] for s in ('lexical_near','paraphrase'))
    summary['source_free_added']=False
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
