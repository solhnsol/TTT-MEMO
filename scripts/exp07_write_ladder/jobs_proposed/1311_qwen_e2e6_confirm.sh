#!/bin/bash
# JOB 1311 = QWEN3-4B-BASE 6% BAND CONFIRMATION WITH THE ISO ESTABLISHMENT GATE (CONFIRMATORY P1; rules = docstring of scripts/exp09_qwen_ttt/cband_confirm_e2e6.py, its sha256 is frozen with a signed tag in the private registry BEFORE the first seed is run; GPU one job at a time; not queued until Hans' sentence; Hans decision card 11/18 11:37 KST '1306과 같은 규칙'). Design: Qwen3-4B-Base, fullmlp (full gate / up / down MLP matrices of the last 9 of 36 layers, fp32, ||W_fast|| = 649.09), SWA structure of exp09 (detached rolling KV of 7 chunks + the current chunk), the step of EVERY chunk is a normalised-gradient step of the fixed norm 0.30426 (cumulative relative path 6.0% of ||W_fast|| over 128 chunks; cell_band_norm.py), arms NATS, NATMASK1024, NATMASK256 and ISO-STD (cell_iso_ladder.py, norm matched to the NATS increment of the same seed = the same 6.0% path), NEW world seeds 171..178 (never used with Qwen before), reads at exposures 16 / 32 / 64 / 128 (out + in-window), canary after every read exposure (band arms) and at the end. After every seed the registered run gate (no non-finite log value, last-8-step mean ce <= 4.0, canary change <= +1.0, reads present, NATMASK1024 differs from NATS at chunk 1) is checked; a failing seed is rerun ONCE (all four arms, --rerun); a seed that fails again is left to the analyzer to EXCLUDE. Analysis: establishment gate ISO - NATS (one-sided 95% t lower bound > 0, else 'TEST NOT ESTABLISHED', P1 not judged), then P1 (band 1024 - NATS: lower bound > 0 AND mean >= 0.5 AND exact sign-flip p < .05 for H0 mu <= 0.5 and mu <= 0), reported: band 256, in-window cost, top-1, mean rank, exposure curves, canary, divergence record -> experiments/exp09_qwen_ttt/cband_e2e6_result.json. W0 reads of each seed are computed by the first arm run of the seed (same base model). CHECKS: pinned files (exit 12), CPU selftests of the runners and the analyzer. ESTIMATE ~3.3 h: per seed NATS ~5 min + band arms 2 x ~5.6 min + ISO ~3.5 min + W0 reads ~2 min + model loads ~3 min = ~25 min x 8 seeds; up to ~4.5 h when the server is loaded; a rerun seed adds ~25 min.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT   # the torch env of exp09 (the job runner sets the JAX environment)
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/cband_e2e6
T=0.30426
SEEDS="171 172 173 174 175 176 177 178"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
mkdir -p $O
cat > experiments/exp09_qwen_ttt/cband_e2e6_pinned_sha256.txt <<'PIN'
75042b3c9d24c52f2ac21707e7d68dfc6c9d0b5643bd27ec40e73d33556e2892  scripts/exp09_qwen_ttt/cband_confirm_e2e6.py
d852967ea3d6fe31653831a5cbd20f08138fe1e3acfe51b8cb39981cbac05528  scripts/exp09_qwen_ttt/cband_confirm_analyze.py
cea4d9b87ccf3941b17e5d770277526d8486d5b129e94edeb80a69da12bcc669  scripts/exp09_qwen_ttt/cband_iso_analyze.py
497cd5ca573352c86cac2b8945a32273c009decd897780d94f2e6882f7cca4cf  scripts/exp09_qwen_ttt/cell_band_norm.py
c59d311e55e1f30f750a6d2c18d7af6e4736d3530065e8d41e2b4ff422bedfdd  scripts/exp09_qwen_ttt/cell_iso_ladder.py
f380f84cf5f46838b22d8f5baba7bb90c4c707054307e9d5a1c41ae351f3dc6e  scripts/exp09_qwen_ttt/cell_band_ladder.py
d7f0e9997cc4d8f155f6ec9f806f128bbfc810face51f1dbae0d436d689d2aaf  scripts/exp09_qwen_ttt/bandmask.py
1f8a2c3671f23d6c317aaf7423fc8d4ea5cc2479efaa05df73fc2145e0d3cb33  scripts/exp09_qwen_ttt/qwen_ladder.py
f384119a6421871492b7037a1226a4e99245aa278cf2c6809ee8a6c13339311e  scripts/exp09_qwen_ttt/qwen_runtime.py
870fc4d1c0186975ab14c0aa7aa3cd3d4181970d808f8199a4431c5fe18796cc  scripts/exp09_qwen_ttt/inner.py
9d2c2be43883134d771823306044236495cf151934da3aa6693694dc90164d9e  scripts/exp09_qwen_ttt/readout.py
dbb188c6c901955e82cb1d3790a4eb9efda1648373f27893e577387e757e3232  scripts/exp09_qwen_ttt/world_text.py
PIN
sha256sum -c experiments/exp09_qwen_ttt/cband_e2e6_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
CUDA_VISIBLE_DEVICES= $PYQ $E/cband_confirm_e2e6.py --selftest | tail -1
CUDA_VISIBLE_DEVICES= $PYQ $E/cell_band_norm.py --selftest | tail -1
CUDA_VISIBLE_DEVICES= $PYQ $E/cell_iso_ladder.py --selftest | tail -1
runseed() {  # seed [--rerun]
  s=$1; shift
  set +e
  $PYQ $E/cell_band_norm.py --step-norm $T --world-seed $s --arms NATS,NATMASK1024,NATMASK256 --reads 16,32,64,128 --fast fullmlp --lr 1.0 --clip 1.0 --out $O "$@"
  echo "RUN bands seed $s rc=$?"
  $PYQ $E/cell_iso_ladder.py --seed-list $s --ref-dir $O --reads 16,32,64,128 --clip 1.0 --fast fullmlp --lr 1.0 --out $O "$@"
  echo "RUN iso seed $s rc=$?"
  set -e
}
for s in $SEEDS; do
  runseed $s
  if ! $PYQ $E/cband_confirm_e2e6.py --dir $O --gate-seed $s; then
    echo "seed $s failed the run gate: rerun once"
    runseed $s --rerun
    $PYQ $E/cband_confirm_e2e6.py --dir $O --gate-seed $s || echo "seed $s fails the gate again: left to the analyzer (excluded)"
  fi
done
$PYQ $E/cband_confirm_e2e6.py --dir $O --seeds 171,172,173,174,175,176,177,178 --json experiments/exp09_qwen_ttt/cband_e2e6_result.json
echo "outputs: experiments/exp09_qwen_ttt/cband_e2e6_result.json"
