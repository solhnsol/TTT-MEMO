#!/bin/bash
# JOB 1308m = QWEN3-4B-BASE E2E-MATCHED SWA NTP TTT WITHOUT OUTER LOOP: stage 0 stability + stage 1 ISO feasibility (EXPLORATORY, no pre-registered rule; not queued until Hans' sentence; GPU one job at a time; Hans 11/17 'e2e 맞춤 qwen 1308z 앞에 올려'; runs after job 1308 and before job 1308z by file name: 1308_ < 1308m < 1308z in every sort order). Design (the E2E rule on Qwen): fast weights = the full gate / up / down MLP matrices of the last 9 of 36 layers (25%, as the 8 of 32 prime-FFN layers of E2E ext-3b-e2e-128K, fp32), plain SGD with the E2E inner rule (lr 1, global-norm clip 1.0, one step per 1024-token chunk), window = the exp09 detached rolling KV of 7 chunks + the current chunk = 8192 (as E2E sliding_window_size 8192); the outer loop (meta-learned initial fast weights and backbone of E2E) cannot be ported. Update size reference (relative norm = update norm / ||W_fast||): E2E per step 0.094%, cumulative path 87.5 / 725.9 = 12.05%, net displacement 16 -> 128 exposures 1.0%; Qwen last-9-layer MLP ||W|| = 649.1 (672,399,360 parameters). STAGE 0 (stability): fullmlp NATS with lr 1.0 and clip 1.0 (the E2E rule) on world seeds 161 and 162, reads at exposures 16 / 32 / 64 / 128, canary; the job continues with stage 1 even when stage 0 fails (nothing is excluded). STAGE 1 (ISO feasibility): ISO-STD (fact sentences only, no context, rows packed per segment (0,16] (16,32] (32,64] (64,128], cell_iso_ladder.py) with fullmlp fast weights, SGD direction rescaled to a cumulative update norm of 2.0% (12.98), 6.0% (38.9) and 12.05% (78.2) of ||W_fast|| (the E2E path = 12.05%) spread proportionally to the exposures of the segment (synthetic norm reference files with the constant per-chunk norm 12.98 / 128 and norm multipliers 1 / 3 / 6.025), and the pure E2E rule (norm matching off, lr 1.0, clip 1.0); seeds 161 and 162; ONE model load for all four configurations. Output experiments/exp09_qwen_ttt/e2e_match/{nats_lr1,iso,ref_e2e}; SWEEP lines per exposure (out-of-window binding margin, W0 gain, top-1, in-window margin, canary, update norm, clip share) with the stage-0 NATS of the same seeds as the reference, summaries summary_iso_e<exposure>.json. Interpretation reference (different model: no direct comparison): E2E 3B ISO-STD out-of-window margin -0.07, top-1 0.516 at exposure 128. ESTIMATE ~50 min: stage 0 2 seeds x ~7 min (NATS 128 chunks + W0 + 4 reads) + 2 model loads ~4 min = ~18 min; stage 1 4 configurations x 2 seeds x ~3.6 min (49 packed steps, 4 reads, canary) + 1 load ~2 min = ~31 min.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT   # the torch env of exp09 (the job runner sets the JAX environment)
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; M=experiments/exp09_qwen_ttt/cband_main; O=experiments/exp09_qwen_ttt/e2e_match
SEEDS=161,162
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
mkdir -p $O
cat > experiments/exp09_qwen_ttt/e2e_match_pinned_sha256.txt <<'PIN'
c59d311e55e1f30f750a6d2c18d7af6e4736d3530065e8d41e2b4ff422bedfdd  scripts/exp09_qwen_ttt/cell_iso_ladder.py
cea4d9b87ccf3941b17e5d770277526d8486d5b129e94edeb80a69da12bcc669  scripts/exp09_qwen_ttt/cband_iso_analyze.py
8dc1d071ab86797cc92e5e81bc20bbce9662af265abee58b10f733316b6f0c54  scripts/exp09_qwen_ttt/cband_iso_sweep_summary.py
f380f84cf5f46838b22d8f5baba7bb90c4c707054307e9d5a1c41ae351f3dc6e  scripts/exp09_qwen_ttt/cell_band_ladder.py
d7f0e9997cc4d8f155f6ec9f806f128bbfc810face51f1dbae0d436d689d2aaf  scripts/exp09_qwen_ttt/bandmask.py
1f8a2c3671f23d6c317aaf7423fc8d4ea5cc2479efaa05df73fc2145e0d3cb33  scripts/exp09_qwen_ttt/qwen_ladder.py
f384119a6421871492b7037a1226a4e99245aa278cf2c6809ee8a6c13339311e  scripts/exp09_qwen_ttt/qwen_runtime.py
870fc4d1c0186975ab14c0aa7aa3cd3d4181970d808f8199a4431c5fe18796cc  scripts/exp09_qwen_ttt/inner.py
9d2c2be43883134d771823306044236495cf151934da3aa6693694dc90164d9e  scripts/exp09_qwen_ttt/readout.py
dbb188c6c901955e82cb1d3790a4eb9efda1648373f27893e577387e757e3232  scripts/exp09_qwen_ttt/world_text.py
PIN
sha256sum -c experiments/exp09_qwen_ttt/e2e_match_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
for s in ${SEEDS//,/ }; do [ -f $M/cband_NATS_s$s.json ] || { echo "missing reference $M/cband_NATS_s$s.json: exit 14"; exit 14; }; done
CUDA_VISIBLE_DEVICES= $PYQ $E/cell_iso_ladder.py --selftest | tail -1
CUDA_VISIBLE_DEVICES= $PYQ $E/cband_iso_sweep_summary.py --selftest | tail -1
# STAGE 0: fullmlp NATS with the E2E inner rule (lr 1, clip 1); a failure does not stop stage 1
set +e
for s in ${SEEDS//,/ }; do
  $PYQ $E/cell_band_ladder.py --stage run --world-seed $s --arms NATS --reads 16,32,64,128 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/nats_lr1
  echo "STAGE0 seed $s rc=$?"
done
set -e
# STAGE 1: synthetic norm reference (constant per-chunk norm 12.98 / 128 = 2.0% of ||W_fast|| = 649.1 in total, real meta of the seed) and the W0 of the stage-0 fullmlp run
$PYQ - $M $O $SEEDS <<'PYE'
import json,sys,shutil,os
M,O,seeds=sys.argv[1],sys.argv[2],sys.argv[3].split(',');X0=0.02*649.09;os.makedirs(f'{O}/ref_e2e',exist_ok=True)
for s in seeds:
    real=json.load(open(f'{M}/cband_NATS_s{s}.json'));json.dump(dict(meta=real['meta'],log=[dict(update_norm=X0/128)]*128),open(f'{O}/ref_e2e/cband_NATS_s{s}.json','w'))
    w0=f'{O}/nats_lr1/cband_W0_s{s}.json'
    if not os.path.exists(w0):w0=f'{M}/cband_W0_s{s}.json';print('stage-0 W0 missing for seed',s,': the LoRA-run W0 of job 1306 is used (same base model, reads identical up to dtype rounding)')
    shutil.copy(w0,f'{O}/ref_e2e/cband_W0_s{s}.json')
print('synthetic references written, total target norm',round(X0,2))
PYE
$PYQ $E/cell_iso_ladder.py --seed-list $SEEDS --ref-dir $O/ref_e2e --reads 16,32,64,128 --clip 1.0 --fast fullmlp --lr 1.0 --out $O/iso --sweep "e2e2pct:;e2e6pct:norm_mult=3;e2e12pct:norm_mult=6.025;pure_lr1:iso_norm=off,lr=1.0"
for e in 16 32 64 128; do
  echo "== exposure $e"
  $PYQ $E/cband_iso_sweep_summary.py --dir $O/iso --main $O/nats_lr1 --seeds $SEEDS --expo $e --json $O/summary_iso_e$e.json || echo "summary exposure $e failed (stage 0 files missing?)"
done
echo "outputs: $O/summary_iso_e{16,32,64,128}.json"
