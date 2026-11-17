#!/bin/bash
# JOB 1306 = QWEN3-4B-BASE BAND CONFIRMATION (P1 confirmatory; rules = docstring of scripts/exp09_qwen_ttt/cband_confirm_analyze.py, its sha256 is frozen with a signed tag in the private registry BEFORE the first seed is run; GPU one job at a time; not queued until Hans' sentence). Design: LoRA r16 on the MLPs of the last 9 layers, SGD lr 3e-2, clip 1.0 (Hans 11/17), world seeds 161..168, arms NATS, NATMASK1024, NATMASK256 (secondary), K0 = 0, reads at exposures 16/32/64/128. After every seed the registered gate (finite log, reads present, NATMASK1024 differs from NATS at chunk 1) is checked; a failing seed is rerun ONCE (--rerun, all arms); a seed that fails again is left to the analyzer to exclude. ESTIMATE ~23 min per seed (3 arms x ~7 min, W0 read once), about 3.0-3.3 h for 8 seeds (2 arms: ~2.1 h). Output experiments/exp09_qwen_ttt/cband_main/, result experiments/exp09_qwen_ttt/cband_main_result.json.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT   # the torch env of exp09 (the job runner sets the JAX environment)
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/cband_main
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
mkdir -p $O
cat > experiments/exp09_qwen_ttt/cband_main_pinned_sha256.txt <<'PIN'
d852967ea3d6fe31653831a5cbd20f08138fe1e3acfe51b8cb39981cbac05528  scripts/exp09_qwen_ttt/cband_confirm_analyze.py
f380f84cf5f46838b22d8f5baba7bb90c4c707054307e9d5a1c41ae351f3dc6e  scripts/exp09_qwen_ttt/cell_band_ladder.py
d7f0e9997cc4d8f155f6ec9f806f128bbfc810face51f1dbae0d436d689d2aaf  scripts/exp09_qwen_ttt/bandmask.py
1f8a2c3671f23d6c317aaf7423fc8d4ea5cc2479efaa05df73fc2145e0d3cb33  scripts/exp09_qwen_ttt/qwen_ladder.py
f384119a6421871492b7037a1226a4e99245aa278cf2c6809ee8a6c13339311e  scripts/exp09_qwen_ttt/qwen_runtime.py
870fc4d1c0186975ab14c0aa7aa3cd3d4181970d808f8199a4431c5fe18796cc  scripts/exp09_qwen_ttt/inner.py
9d2c2be43883134d771823306044236495cf151934da3aa6693694dc90164d9e  scripts/exp09_qwen_ttt/readout.py
dbb188c6c901955e82cb1d3790a4eb9efda1648373f27893e577387e757e3232  scripts/exp09_qwen_ttt/world_text.py
9352edb46f58439e249e884482ae3641d8c2f4ffe859189d3ba8d4e8fdeedcd5  scripts/exp09_qwen_ttt/test_bandmask.py
460e4c3b7a58bd06c2524bfd0b624b92f8d0b89096c27af5c4cab43fed35ff8a  scripts/exp09_qwen_ttt/test_bandmask_vs_jax.py
PIN
sha256sum -c experiments/exp09_qwen_ttt/cband_main_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
for t in test_bandmask.py test_bandmask_vs_jax.py; do CUDA_VISIBLE_DEVICES= $PYQ $E/$t | tail -1; done
CUDA_VISIBLE_DEVICES= $PYQ $E/cband_confirm_analyze.py --selftest | tail -1
SEEDS="161 162 163 164 165 166 167 168"
gate() { $PYQ -c "import sys;sys.path.insert(0,'$E');import cband_confirm_analyze as C;g=C.gate('$O',$1);print('GATE seed $1:',g or 'ok');sys.exit(1 if g else 0)"; }
for s in $SEEDS; do
  $PYQ $E/cell_band_ladder.py --stage run --world-seed $s --arms NATS,NATMASK1024,NATMASK256 --reads 16,32,64,128 --fast lora --lr 3e-2 --clip 1.0 --out $O
  if ! gate $s; then echo "seed $s failed the gate: rerun once"; $PYQ $E/cell_band_ladder.py --stage run --world-seed $s --arms NATS,NATMASK1024,NATMASK256 --reads 16,32,64,128 --fast lora --lr 3e-2 --clip 1.0 --out $O --rerun; gate $s || echo "seed $s fails the gate again: left to the analyzer (excluded)"; fi
done
$PYQ $E/cband_confirm_analyze.py --dir $O --seeds 161,162,163,164,165,166,167,168 --json experiments/exp09_qwen_ttt/cband_main_result.json > /dev/null
echo "outputs: experiments/exp09_qwen_ttt/cband_main_result.json"
