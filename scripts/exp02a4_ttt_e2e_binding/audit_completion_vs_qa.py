"""Independent CPU-only audit/aggregation from saved completion/QA scores."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/exp02a4_ttt_e2e_binding/completion_vs_qa_20261012'
REF=ROOT/'experiments/exp02a4_ttt_e2e_binding/update_budget_k4_20261012'
def read(p):return list(map(json.loads,(p/'samples.jsonl').read_text().splitlines()))
def aggregate(records,style,field,updated):
    rows=[]
    for r in records:
        margins={};ok={}
        for m in ('A','B'):
            mat=r['matrices']['ttt_'+m if updated else 'static'][style]
            margins[m]={};ok[m]={}
            for e in ('e1','e2'):
                gold='v1' if (m=='A')==(e=='e1') else 'v2';other='v2' if gold=='v1' else 'v1'
                margins[m][e]=mat[e][gold][field]-mat[e][other][field];ok[m][e]=margins[m][e]>0
        bs={m:statistics.mean(margins[m].values()) for m in ('A','B')}
        flips={e:ok['A'][e] and ok['B'][e] for e in ('e1','e2')}
        rows.append({'pair_id':r['pair_id'],'margins':margins,'success':ok,'flips':flips,'both_flip':all(flips.values()),'B_A':bs['A'],'B_B':bs['B'],'B_swap':statistics.mean(bs.values())})
    return {'A_correct':sum(sum(r['success']['A'].values()) for r in rows),'B_correct':sum(sum(r['success']['B'].values()) for r in rows),'both_flip_count':sum(r['both_flip'] for r in rows),'single_entity_flip_count':sum(sum(r['flips'].values()) for r in rows),'mean_margin_A':statistics.mean(r['B_A'] for r in rows),'mean_margin_B':statistics.mean(r['B_B'] for r in rows),'mean_B_swap':statistics.mean(r['B_swap'] for r in rows),'samples':rows}
rs=read(OUT);previous=read(REF);cfg=json.loads((OUT/'config.json').read_text());dataset=list(map(json.loads,(OUT/'dataset.jsonl').read_text().splitlines()))
assert len(rs)==len(previous)==len(dataset)==4
assert (OUT/'dataset.jsonl').read_bytes()==(REF/'dataset.jsonl').read_bytes()
for f,h in cfg['reference_sha256'].items():assert hashlib.sha256((REF/f).read_bytes()).hexdigest()==h
max_error=0
for r,old,pair in zip(rs,previous,dataset):
    assert r['pair_id']==old['pair_id']==pair['pair_id']
    for m in ('A','B'):
        assert r['write'][m]['text']==old['write'][m]['text']
        for step,history in enumerate(r['write'][m]['trajectory']):
            assert history['fast_step_max_abs_delta']>0 and history['frozen_max_abs_delta']==0
            assert history['same_fact_reexposed'] and history['fresh_write_kv']
            assert abs(history['archived_loss_pre_difference']-(history['loss_pre']-old['write'][m]['trajectory'][step]['loss_pre']))<1e-9
        assert r['reproduction'][m]['same_fast_state_both_probes'] and r['reproduction'][m]['fast_state_unchanged_after_readout']
    for condition,d in r['candidate_details'].items():
        for style,ents in d.items():
            for e,scores in ents.items():
                assert scores['v1']['n_loss_tokens']==scores['v2']['n_loss_tokens']
                for v,s in scores.items():
                    n=s['n_loss_tokens'];assert len(s['token_logp'])==n
                    assert abs(sum(s['token_logp'])-s['sum_logp'])<1e-4
                    assert abs(statistics.mean(s['token_logp'])-s['mean_logp'])<1e-4
                    assert r['matrices'][condition][style][e][v]=={'mean':s['mean_logp'],'sum':s['sum_logp']}
                    prompt=s['text'][:-(len(pair[v])+1)]
                    assert pair['v1'] not in prompt and pair['v2'] not in prompt
                    if style=='direct_completion':assert prompt=='Fact:\nThe passphrase assigned to '+pair[e]+' is'
                    else:
                        assert 'Context:\n\n\nQuestion:' in prompt
                        oldc=condition if condition=='static' else condition
                        err=max(abs(s[f+'_logp']-old['candidate_details'][oldc][style][e][v][f+'_logp']) for f in ('mean','sum'))
                        max_error=max(max_error,err)
out={'invariants_passed':True,'max_archived_QA_score_abs_error':max_error,'fast_weights_retrained':True,'results':{style:{field:{label:aggregate(rs,style,field,updated) for label,updated in [('no_update',False),('K4',True)]} for field in ('mean','sum')} for style in ('direct_completion','lexical_near')}}
(OUT/'independent_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({s:{f:{k:{a:b for a,b in v.items() if a!='samples'} for k,v in c.items()} for f,c in d.items()} for s,d in out['results'].items()},indent=2))
