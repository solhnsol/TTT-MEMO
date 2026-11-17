#!/bin/bash
# JOB 1309 = QWEN3-4B-BASE ISO-STD (EXPLORATORY; not queued until Hans' sentence; GPU one job at a time; Hans 11/17 'qwen iso 해보자'): does Qwen learn the binding (out-of-window top-1 and margin) when it sees only the fact sentences without context? Same model / fast weights / optimizer / world / reads as job 1306 (Qwen3-4B-Base, LoRA r16 on the MLPs of the last 9 layers, SGD lr 3e-2, clip 1.0, world seeds 161..168, reads at exposures 16 / 32 / 64 / 128, 16 candidates), ISO-STD of E55 (e55_iso.py) ported by scripts/exp09_qwen_ttt/cell_iso_ladder.py: every exposure sentence ('\\n\\n' + sentence) is a separate row, rows of a round shuffled, segments (0,16], (16,32], (32,64], (64,128] packed on their own into <= 1024-token steps with per-row position restart (block-diagonal attention), one clipped SGD step per packed step rescaled after the clip so that the segment update norm equals the NATS increment of the same seed (cband_main/cband_NATS_s<seed>.json), no window context in the write. 52 packed steps per seed (7 / 7 / 13 / 25 steps per segment, checked on seeds 161 and 168). W0 reads and the NATS / NATMASK1024 / NATMASK256 values are REUSED from job 1306 (same world sha, checked). CHECKS: pinned files (exit 12), the NATS reference files of all 8 seeds exist (exit 14), CPU selftests (rows, packing, norm matching, schema), the stream sha of every seed equals that of the NATS file (assert in the run). Output experiments/exp09_qwen_ttt/cband_iso/ (cband_ISO_s<seed>.json, W0 copies), analysis cband_iso_analyze.py -> experiments/exp09_qwen_ttt/cband_iso_result.json (descriptive, no pass / fail rule). ESTIMATE ~35 min: model load ~2 min once + per seed 52 steps ~40 s + 4 reads ~40 s each + canary ~15 s = ~3.7-4.2 min x 8 seeds (jobs 1304s / 1306: arm run 242-272 s with 128 steps and 2-4 reads).
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT   # the torch env of exp09 (the job runner sets the JAX environment)
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; M=experiments/exp09_qwen_ttt/cband_main; O=experiments/exp09_qwen_ttt/cband_iso
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
mkdir -p $O
cat > experiments/exp09_qwen_ttt/cband_iso_pinned_sha256.txt <<'PIN'
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
sha256sum -c experiments/exp09_qwen_ttt/cband_iso_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
for s in 161 162 163 164 165 166 167 168; do for f in $M/cband_NATS_s$s.json $M/cband_W0_s$s.json; do [ -f $f ] || { echo "missing reference $f: exit 14"; exit 14; }; done; done
CUDA_VISIBLE_DEVICES= $PYQ $E/cell_iso_ladder.py --selftest | tail -1
CUDA_VISIBLE_DEVICES= $PYQ $E/cband_iso_analyze.py --selftest | tail -1
$PYQ $E/cell_iso_ladder.py --seed-list 161,162,163,164,165,166,167,168 --ref-dir $M --out $O --fast lora --lr 3e-2 --clip 1.0 --reads 16,32,64,128
$PYQ $E/cband_iso_analyze.py --iso $O --main $M --json experiments/exp09_qwen_ttt/cband_iso_result.json > /dev/null
echo "outputs: experiments/exp09_qwen_ttt/cband_iso_result.json"
