"""CPU-only independent aggregation of stored K=1/K=4 candidate scores."""
import hashlib, json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/exp02a4_ttt_e2e_binding/matched_lexical_source_free_20261012'
NEW=ROOT/'experiments/exp02a4_ttt_e2e_binding/update_budget_k4_20261012'
def read(p):return [json.loads(l) for l in (p/'samples.jsonl').read_text().splitlines()]
def aggregate(records,field):
    rows=[];nll=[]
    for r in records:
        mats=[r['matrices']['ttt_'+m]['lexical_near'] for m in ('A','B')]
        ok=[];margins=[]
        for i,mat in enumerate(mats):
            success=[];mg=[]
            for j,e in enumerate(('e1','e2')):
                gold='v1' if i==j else 'v2';other='v2' if gold=='v1' else 'v1'
                g=mat[e][gold][field];o=mat[e][other][field]
                success.append(g>o);mg.append(g-o);nll.append(-g)
            ok.append(success);margins.append(mg)
        rows.append({'pair_id':r['pair_id'],'A_success':ok[0],'B_success':ok[1],'e1_flip':ok[0][0] and ok[1][0],'e2_flip':ok[0][1] and ok[1][1],'both_flip':all(ok[0]+ok[1]),'B_swap':sum(sum(m) for m in margins)/4})
    return {'A_correct':sum(sum(r['A_success']) for r in rows),'B_correct':sum(sum(r['B_success']) for r in rows),'n_per_mapping':2*len(rows),'both_flip_count':sum(r['both_flip'] for r in rows),'mean_B_swap':statistics.mean(r['B_swap'] for r in rows),'gold_answer_nll':statistics.mean(nll),'samples':rows}
a,b=read(OLD),read(NEW)
assert len(a)==len(b)==4
cfg=json.loads((NEW/'config.json').read_text())
for f,h in cfg['reference_sha256'].items():assert hashlib.sha256((OLD/f).read_bytes()).hexdigest()==h
assert (NEW/'dataset.jsonl').read_bytes()==(OLD/'dataset.jsonl').read_bytes()
for old,r in zip(a,b):
    assert old['pair_id']==r['pair_id']
    for c in ('static','source_present_A','source_present_B'):assert old['matrices'][c]==r['matrices'][c] and old['candidate_details'][c]==r['candidate_details'][c]
    for w in r['write'].values():
        assert w['native_updates']==4 and len(w['trajectory'])==4
        assert len({x['write_text_sha256'] for x in w['trajectory']})==1
        for x in w['trajectory']:assert x['same_fact_reexposed'] and x['fresh_write_kv'] and x['frozen_max_abs_delta']==0 and x['fast_step_max_abs_delta']>0
    for condition,d in r['candidate_details'].items():
        for e,scores in d['lexical_near'].items():
            assert scores['v1']['n_loss_tokens']==scores['v2']['n_loss_tokens']
            for v,s in scores.items():
                assert len(s['token_logp'])==s['n_loss_tokens']
                assert abs(sum(s['token_logp'])-s['sum_logp'])<1e-4
                assert abs(statistics.mean(s['token_logp'])-s['mean_logp'])<1e-4
                assert 'Context:\n\n\nQuestion:' in s['text'] if condition.startswith('ttt') else True
out={'invariants_passed':True,'comparison':{f:{'K1':aggregate(a,f),'K4':aggregate(b,f)} for f in ('mean','sum')},'write_loss_mean_trajectory':[statistics.mean([r['write'][m]['trajectory'][0]['loss_pre'] if step==0 else r['write'][m]['trajectory'][step-1]['loss_post'] for r in b for m in ('A','B')]) for step in range(5)],'total_native_updates':32,'total_write_loss_tokens':sum(w['total_loss_tokens'] for r in b for w in r['write'].values())}
(NEW/'independent_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
