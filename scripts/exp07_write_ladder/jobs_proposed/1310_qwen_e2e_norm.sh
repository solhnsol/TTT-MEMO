#!/bin/bash
# JOB 1310 = QWEN3-4B-BASE E2E-SIZE NATS / BAND ARMS WITH A FIXED PER-CHUNK UPDATE NORM (EXPLORATORY, no pre-registered rule; not queued until Hans' sentence; GPU one job at a time; Hans 11/18 '12%도 시도는 해 보되 2%나 6%도 같이 해 보고, qwen은 쓰기 학습이 안 된 모델이라 발산하지 않는 파라미터를 찾았다고 명시'): Qwen3-4B-Base, fast weights = the full gate / up / down MLP matrices of the last 9 of 36 layers (fullmlp, fp32, ||W_fast|| = 649.09), SWA structure of exp09 (detached rolling KV of 7 chunks + the current chunk = 8192, one inner step per 1024-token chunk), arms NATS, NATMASK1024, NATMASK256 (cell_band_ladder.py arms), world seeds 161 and 162, reads at exposures 16 / 32 / 64 / 128 (out-of-window + in-window), the step of EVERY chunk is a normalised-gradient step of a FIXED global norm (scripts/exp09_qwen_ttt/cell_band_norm.py): THREE SIZES = cumulative relative path over 128 chunks of 2.0% (0.10142 per chunk, 12.98 in total), 6.0% (0.30426 per chunk, 38.95) and 12.05% (0.61110 per chunk, 78.22 = the E2E NATS path 87.48 / 725.93 on the E2E prime FFN, measured on seeds 71..78). The same sizes were the ISO sizes of job 1308m (ISO 12.05% diverged, 2% and 6% did not). For the post-hoc judgement of divergence (no rule applied) every run keeps the per-chunk log (ce, grad_norm, update_norm, raw_update_norm) and the canary NLL after every read exposure (reads[<e>][canary]) plus the end-of-run canary; cband_norm_summary.py prints per size, arm and seed the largest chunk loss, the last loss, the largest gradient norm, non-finite flags, the canary changes, the out-of-window margin / gain over W0 / top-1 at every exposure and the in-window margin, with a descriptive 'DIV?' flag (largest loss > 10 nat, final canary change > +1.0 nat / token, non-finite; NOT a rule). W0 reads reused from experiments/exp09_qwen_ttt/e2e_match/nats_lr1 (fullmlp W0 of job 1308m, stream sha asserted by the runner of 1308m; same base model). CHECKS: pinned files (exit 12), W0 references exist (exit 14), CPU selftests of the runner and the summary. Output experiments/exp09_qwen_ttt/e2e_norm/{n2pct,n6pct,n12pct}/ and summary_e2e_norm.json. ESTIMATE ~1.9 h: per size and seed 3 arms x ~5 min + model load ~2 min (job 1304s fullmlp 272 s, job 1306 arms 261-424 s, canary after every read adds ~10 s per read) = ~17 min; 3 sizes x 2 seeds = ~1.7 h + CPU checks ~5 min; up to ~2.5 h when the server is loaded.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT   # the torch env of exp09 (the job runner sets the JAX environment)
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; R=experiments/exp09_qwen_ttt/e2e_match/nats_lr1; O=experiments/exp09_qwen_ttt/e2e_norm
SEEDS="161 162"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
mkdir -p $O
cat > experiments/exp09_qwen_ttt/e2e_norm_pinned_sha256.txt <<'PIN'
497cd5ca573352c86cac2b8945a32273c009decd897780d94f2e6882f7cca4cf  scripts/exp09_qwen_ttt/cell_band_norm.py
f942a0cd5671290c121e08956715088826da30f6f536c47dda06b50ab711f772  scripts/exp09_qwen_ttt/cband_norm_summary.py
cea4d9b87ccf3941b17e5d770277526d8486d5b129e94edeb80a69da12bcc669  scripts/exp09_qwen_ttt/cband_iso_analyze.py
f380f84cf5f46838b22d8f5baba7bb90c4c707054307e9d5a1c41ae351f3dc6e  scripts/exp09_qwen_ttt/cell_band_ladder.py
d7f0e9997cc4d8f155f6ec9f806f128bbfc810face51f1dbae0d436d689d2aaf  scripts/exp09_qwen_ttt/bandmask.py
1f8a2c3671f23d6c317aaf7423fc8d4ea5cc2479efaa05df73fc2145e0d3cb33  scripts/exp09_qwen_ttt/qwen_ladder.py
f384119a6421871492b7037a1226a4e99245aa278cf2c6809ee8a6c13339311e  scripts/exp09_qwen_ttt/qwen_runtime.py
870fc4d1c0186975ab14c0aa7aa3cd3d4181970d808f8199a4431c5fe18796cc  scripts/exp09_qwen_ttt/inner.py
9d2c2be43883134d771823306044236495cf151934da3aa6693694dc90164d9e  scripts/exp09_qwen_ttt/readout.py
dbb188c6c901955e82cb1d3790a4eb9efda1648373f27893e577387e757e3232  scripts/exp09_qwen_ttt/world_text.py
PIN
sha256sum -c experiments/exp09_qwen_ttt/e2e_norm_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
for s in $SEEDS; do [ -f $R/cband_W0_s$s.json ] || { echo "missing W0 reference $R/cband_W0_s$s.json: exit 14"; exit 14; }; done
CUDA_VISIBLE_DEVICES= $PYQ $E/cell_band_norm.py --selftest | tail -1
CUDA_VISIBLE_DEVICES= $PYQ $E/cband_norm_summary.py --selftest | tail -1
# size name : per-chunk update norm (cumulative relative path x 649.09 / 128)
for SZ in n2pct:0.10142 n6pct:0.30426 n12pct:0.61110; do
  NAME=${SZ%%:*}; T=${SZ##*:}; mkdir -p $O/$NAME
  for s in $SEEDS; do
    cp -n $R/cband_W0_s$s.json $O/$NAME/cband_W0_s$s.json
    set +e
    $PYQ $E/cell_band_norm.py --step-norm $T --world-seed $s --arms NATS,NATMASK1024,NATMASK256 --reads 16,32,64,128 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/$NAME
    echo "RUN $NAME seed $s rc=$?"
    set -e
  done
done
$PYQ $E/cband_norm_summary.py --dir $O --seeds 161,162 --json experiments/exp09_qwen_ttt/summary_e2e_norm.json
echo "outputs: experiments/exp09_qwen_ttt/summary_e2e_norm.json"
