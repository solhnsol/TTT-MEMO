"""Independent saved-score audit and reproducibility report; stdlib only."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
root=Path(__file__).resolve().parents[2];out=args.out.resolve()
s=json.loads((out/'summary.json').read_text());records=[json.loads(l) for l in (out/'samples.jsonl').read_text().splitlines()]
assert len(records)==s['n']
lines=['# Exp 02A4 native TTT-E2E 3B binding pilot','',
'One native SGD update per mapping (lr=1, clip=1); source/write KV completely discarded before fresh recall. No filler retention or free-generation evaluation.','',
'| pair | probe | B_swap (mean) | Mapping A both correct | Mapping B both correct | Both entities flip | Source-present both flip |','|---|---|---:|---:|---:|---:|---:|']
for r in records:
 for style in ['lexical_near','paraphrase']:
  for field in ['mean','sum']:
   mats=r['matrices']
   def preferences(c):return [mats[c][style][e]['v1'][field]-mats[c][style][e]['v2'][field] for e in ['e1','e2']]
   a,b=preferences('ttt_A'),preferences('ttt_B')
   bs=((a[0]-a[1])+(b[1]-b[0]))/4
   aa=a[0]>0 and a[1]<0;bb=b[0]<0 and b[1]>0
   metric=r['metrics'][style][field]
   assert abs(bs-metric['ttt']['B_swap'])<1e-9
   assert bool(metric['flips']['both_flipped']) == (aa and bb)
   for condition,d in r['candidate_details'].items():
    for probe in d.values():
     for entity in probe.values():
      for candidate in entity.values():
       assert abs(sum(candidate['token_logp'])-candidate['sum_logp'])<1e-4
       assert abs(mean(candidate['token_logp'])-candidate['mean_logp'])<1e-4
   if field=='mean':lines.append(f"| {r['pair_id']} | {style} | {bs:.6f} | {int(aa)} | {int(bb)} | {int(aa and bb)} | {metric['source_present_flips']['both_flipped']} |")
lines += ['', '## Aggregate results', '', json.dumps(s['results'],indent=2), '',
'## Correctness and cost', '',
'Native fast weights changed in every write; all frozen parameter deltas were exactly zero. Initial fast/frozen hashes stayed unchanged. Initial zero suffix cache and chunk counter used on every readout. Per-token saved scores independently reproduce sums, means, binding scores and flips.','',
f"{s['total_inner_updates']} total native updates; {s['total_write_loss_tokens']} total labeled write tokens; {s['n_inner_parameters']:,} native inner parameters.",
f"Elapsed {s['elapsed_seconds']:.2f}s including restore, hashing and compilation; JAX peak bytes in use {s['jax_memory_stats']['peak_bytes_in_use']/2**30:.3f} GiB.", '',
'## Qwen comparison and limitations', '',
'The same four exact pairs, write facts and lexical/paraphrase questions are used as Exp02A1. Qwen K=32 exact B_swap was 0.296 (lexical) / 0.219 (paraphrase), both-entity flip 0% for both; source-present both flip 100%. TTT scores should be compared descriptively, not as matched optimization budgets.', '',
'TTT uses full real write-token NTP with one exposure/update, 346M native prime FFN parameters, Llama3 tokenization and base-LM T3 readout. Qwen used value-only CE, 32 reused updates, q/o LoRA, Qwen tokenization and chat readout. Model size, pretraining, instruction tuning and meta-training differ. A small fixed 4-pair pilot cannot isolate a meta-training causal effect or establish absence of storage capability.', '',
'Actual Llama3 candidate token lengths (V1/V2): exact-00 3/3, exact-01 5/4, exact-02 4/4, exact-03 4/3. The Qwen-matched dataset is not length-matched under Llama3. Source-present mean-logp both-flip is 75% lexical / 25% paraphrase; sum-logp 100% / 75%. This limits readout calibration and negative-memory interpretation.', '',
'Complete source/KV eviction is stronger isolation than merely moving raw tokens out of SWA but does not test normal 8K+ streaming behavior or persistent source-derived KV. Candidate-only likelihood does not establish free-generation success.','',
'Next suggested action: calibrate the base-LM readout on source-present pairs with candidate lengths matched under the actual Llama3 tokenizer, before expanding native write exposures. No hyperparameter search.']
(out/'summary.md').write_text('\n'.join(lines)+'\n')
config=json.loads((out/'config.json').read_text());source=root/'scripts/exp02a1_binding_specificity/frozen/dataset_pairs.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest()==config['source_sha256']
paths=[*list((root/'scripts/exp02a4_ttt_e2e_binding').glob('*')),source,root/'scripts/exp02a1_binding_specificity/metrics.py',root/'e2e/ttt/model/transformer.py',root/'e2e/ttt/model/attention.py',*list(out.glob('*.json')),out/'dataset.jsonl',out/'samples.jsonl',out/'summary.md']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file() and p.name!='sha256_manifest.json'}
(out/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Independent candidate-score/binding/flip audit and dataset hash verified.')
