"""Independent saved-score audit of matched lexical native TTT pilot."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
root=Path(__file__).resolve().parents[2];out=args.out.resolve()
rows=[json.loads(l) for l in (out/'samples.jsonl').read_text().splitlines()]
s=json.loads((out/'summary.json').read_text());assert len(rows)==4
cal=root/'experiments/exp02a4_ttt_e2e_binding/calibration_20261012'
reference={r['pair_id']:r for r in [json.loads(l) for l in (cal/'samples.jsonl').read_text().splitlines()] if r['cell']=='matched_T3'}
audit={'n':4,'per_pair':[],'calibration_max_abs_score_diff':0.0,'checks':[]}
lines=['# Length-matched lexical-only TTT-E2E 3B source-free binding','',
'Calibration dataset and T3 lexical prompt reused unchanged. Paraphrase excluded. Mean-token logp primary; sum diagnostic. Native chunk=1024; one SGD update per mapping, lr=1, clip=1; real write tokens labeled, padding excluded.', '',
'| pair | native B_swap mean | native A correct /2 | native B correct /2 | both entities flip |','|---|---:|---:|---:|---:|']
for r in rows:
    assert set(r['metrics'])=={'lexical_near'}
    record={'pair':r['pair_id'],'mean':{},'sum':{}}
    for field in ['mean','sum']:
        for label,ka,kb in [('source_present_no_update','source_present_A','source_present_B'),('source_free_no_update','static','static'),('source_free_native_ttt','ttt_A','ttt_B')]:
            a=r['matrices'][ka]['lexical_near'];b=r['matrices'][kb]['lexical_near']
            pa=[a[e]['v1'][field]-a[e]['v2'][field] for e in ['e1','e2']]
            pb=[b[e]['v1'][field]-b[e]['v2'][field] for e in ['e1','e2']]
            assert all(x!=0 for x in pa+pb), 'tie requires explicit report'
            aa=[pa[0]>0,pa[1]<0];bb=[pb[0]<0,pb[1]>0]
            bs=(pa[0]-pa[1]+pb[1]-pb[0])/4
            record[field][label]={'A_correct':sum(aa),'B_correct':sum(bb),'both_flip':all(aa+bb),'B_swap':bs}
        for mapping in ['A','B']:
            cur=r['matrices']['source_present_'+mapping]['lexical_near']
            prev=reference[r['pair_id']]['matrices'][mapping]['lexical_near']
            for e in ['e1','e2']:
                for v in ['v1','v2']:
                    delta=abs(cur[e][v][field]-prev[e][v][field])
                    audit['calibration_max_abs_score_diff']=max(audit['calibration_max_abs_score_diff'],delta)
    for condition,styles in r['candidate_details'].items():
        assert set(styles)=={'lexical_near'}
        for entity in styles['lexical_near'].values():
            assert entity['v1']['n_loss_tokens']==entity['v2']['n_loss_tokens']
            for d in entity.values():
                assert abs(sum(d['token_logp'])-d['sum_logp'])<1e-4
                assert abs(mean(d['token_logp'])-d['mean_logp'])<1e-4
    for w in r['write'].values():
        assert w['native_updates']==1 and w['fast_max_abs_delta']>0 and w['frozen_max_abs_delta']==0
    for condition in record['mean']:
        assert record['mean'][condition]['A_correct']==record['sum'][condition]['A_correct']
        assert record['mean'][condition]['B_correct']==record['sum'][condition]['B_correct']
        assert record['mean'][condition]['both_flip']==record['sum'][condition]['both_flip']
    audit['per_pair'].append(record)
    m=record['mean']['source_free_native_ttt'];lines.append(f"| {r['pair_id']} | {m['B_swap']:.6f} | {m['A_correct']} | {m['B_correct']} | {int(m['both_flip'])} |")
for field in ['mean','sum']:
    for condition,expected in s['results']['lexical_near'][field]['conditions'].items():
        rr=[r[field][condition] for r in audit['per_pair']]
        assert sum(r['A_correct'] for r in rr)==expected['A_correct']
        assert sum(r['B_correct'] for r in rr)==expected['B_correct']
        assert sum(r['both_flip'] for r in rr)==expected['both_flip_count']
        assert abs(mean(r['B_swap'] for r in rr)-expected['mean_B_swap'])<1e-9
config=json.loads((out/'config.json').read_text())
assert hashlib.sha256((cal/'dataset_matched.jsonl').read_bytes()).hexdigest()==config['source_sha256']
assert (out/'dataset.jsonl').read_bytes()==(cal/'dataset_matched.jsonl').read_bytes()
audit['checks']=['independent causal token-score sum/mean reconstruction','matched candidate lengths and identical mean/sum decisions','independent condition-wise accuracy, flip and binding-swap reconstruction','dataset byte-identical to calibration; source hash unchanged','actual fast changes and zero full frozen differences in every write','lexical only; source-present comparison to calibration recorded']
(out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
lines += ['', '## Condition-wise results','',json.dumps(s['results']['lexical_near'],indent=2),'',
f"Calibration max absolute candidate score difference: {audit['calibration_max_abs_score_diff']:.9g}.",
f"Native updates total: {s['total_inner_updates']}; loss tokens total: {s['total_write_loss_tokens']}; elapsed including restore/compilation: {s['elapsed_seconds']:.2f}s; JAX peak: {s['jax_memory_stats']['peak_bytes_in_use']/2**30:.3f} GiB.", '',
'Fast weights and full frozen weights were initialized identically across mappings/examples. Each native update changes only prime FFN; all frozen deltas=0. Write KV, optimizer, source tensors and prefix hidden states are dropped before readout. Every candidate starts with initial zero KV/chunk counter; no source-bearing state crosses readout. Initial fast/frozen hashes remain unchanged.', '',
'Limitations: four fixed exact pairs and one lexical question family; candidate discrimination, not free generation. Single native chunk exposure/update is a limited write budget. Source is completely evicted into a fresh context rather than moved through a measured 8K+ stream. Calibration permits interpretation of this lexical readout but does not establish broader QA or memory competence.', '',
'Next suggested action: evaluate a single fixed budget of four native write chunks/exposures per mapping, retaining the native SGD/clip rule and the same source/KV eviction; record token/compute budgets without a hyperparameter search.']
(out/'summary.md').write_text('\n'.join(lines)+'\n')
paths=[root/'scripts/exp02a4_ttt_e2e_binding/run_matched_lexical.py',Path(__file__),*list(out.glob('*.json')),*list(out.glob('*.jsonl')),out/'summary.md']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name!='sha256_manifest.json'}
(out/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Independent audit passed; calibration score diff',audit['calibration_max_abs_score_diff'])
