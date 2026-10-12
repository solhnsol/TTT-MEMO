"""Saved-score-only calibration audit; no model imports."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
out=args.out.resolve();root=Path(__file__).resolve().parents[2]
s=json.loads((out/'summary.json').read_text());rs=[json.loads(l) for l in (out/'samples.jsonl').read_text().splitlines()]
assert len(rs)==16
mismatches=[];errors=[]
for r in rs:
 for mapping in ['A','B']:
  for style in ['lexical_near','paraphrase']:
   for entity in ['e1','e2']:
    gold='v1' if ((mapping=='A')==(entity=='e1')) else 'v2';other='v2' if gold=='v1' else 'v1'
    data=r['details'][mapping][style][entity]
    for v in ['v1','v2']:
     d=data[v]
     assert abs(sum(d['token_logp'])-d['sum_logp'])<1e-4
     assert abs(mean(d['token_logp'])-d['mean_logp'])<1e-4
    mean_ok=data[gold]['mean_logp']>data[other]['mean_logp']
    sum_ok=data[gold]['sum_logp']>data[other]['sum_logp']
    case={'cell':r['cell'],'pair':r['pair_id'],'mapping':mapping,'style':style,'entity':entity,'gold':gold,
          'mean_correct':mean_ok,'sum_correct':sum_ok,'scores':data}
    if mean_ok!=sum_ok:mismatches.append(case)
    if not mean_ok:errors.append(case)
    if r['cell'].startswith('matched'):assert mean_ok==sum_ok
(out/'scoring_disagreements.json').write_text(json.dumps(mismatches,indent=2)+'\n')
(out/'mean_scoring_errors.json').write_text(json.dumps(errors,indent=2)+'\n')
lines=['# TTT-E2E 3B source-present readout calibration','',
'Four original exact pairs. No inner updates and no source-free runs. All parameters hash-identical before/after inference. Original/T3 reuses archived scores; original/T2, matched/T3 and matched/T2 are newly evaluated on the same checkpoint.','',
'| values / prompt | question | mean A correct /8 | mean B correct /8 | mean both flip /4 | sum A correct /8 | sum B correct /8 | sum both flip /4 |','|---|---|---:|---:|---:|---:|---:|---:|']
for cell,styles in s['results'].items():
 for style,result in styles.items():
  a,b=result['mean'],result['sum'];lines.append(f"| {cell} | {style} | {a['A_correct']} | {a['B_correct']} | {a['both_flip_count']} | {b['A_correct']} | {b['B_correct']} | {b['both_flip_count']} |")
lines+=['','## Attribution','',
'1. Tokenization: original candidate lengths under Llama3 were 3/3, 5/4, 4/4, 4/3. The original Qwen-matched pairs are not Llama3 matched. Tokenizer-only edits (pebble -> stone, dune -> lake) yield 3/3, 4/4, 4/4, 3/3; verified in every complete scoring prompt before any new scores.', '',
'2. Scoring: mean and sum are computed from the same saved candidate token logps; causal shift, offsets, decoded candidate span and EOS exclusion are verified. Different candidate denominators can reverse the winner without a token-score bug. With matched lengths, mean/sum decisions coincide. Archived original/T3 lexical A/B was 8/8 and 7/8 under mean versus 8/8 and 8/8 under sum; paraphrase 6/8 and 6/8 versus 7/8 and 8/8. Detailed disagreements are retained.', '',
'3. Prompt: T3 is the existing instruction/Context/Question/Answer prompt. T2 is the existing minimal context/Q/A prompt. Questions, entities, mapping order, leading answer space, scoring mask and model are held fixed within original_T3 vs original_T2 and matched_T3 vs matched_T2 comparisons. Two formats were fixed before evaluation; T2 is diagnostic, not an optimized prompt choice.', '',
'4. Limitations: changing words to match token lengths also changes lexical priors and the local fact surface. Therefore new matched-versus-original differences cannot be attributed solely to token count. Question-style and prompt-format effects remain conditional on these four pairs. This tests forced candidate discrimination, not free-generation QA.', '',
f"Primary matched/T3 gate: {s['primary_gate_passed']}; diagnostic matched/T2 gate: {s['diagnostic_T2_gate_passed']}. Gate requires 8/8 answers for each mapping and 4/4 both-entity flips for BOTH styles under mean-token logp.",
'No source-free experiment was added.','',f"Elapsed: {s['elapsed_seconds']:.2f}s; inner updates: 0.",
'Next action: if primary fails, improve source-present readout and confirm on held-out length-matched pairs before any additional memory experiment; if primary passes, freeze the calibrated protocol and confirm on held-out pairs before expanding memory evaluation.']
(out/'summary.md').write_text('\n'.join(lines)+'\n')
config=json.loads((out/'config.json').read_text());baseline=root/'experiments/exp02a4_ttt_e2e_binding/pilot_20261012/samples.jsonl'
assert hashlib.sha256(baseline.read_bytes()).hexdigest()==config['baseline_samples_sha256']
paths=[root/'scripts/exp02a4_ttt_e2e_binding/calibrate_readout.py',Path(__file__),root/'scripts/exp02a4_ttt_e2e_binding/run_pilot.py',*list(out.glob('*.json')), *list(out.glob('*.jsonl')),out/'summary.md']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name!='sha256_manifest.json'}
(out/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved candidate-score audit passed; matched mean/sum decisions identical; baseline unchanged.')
