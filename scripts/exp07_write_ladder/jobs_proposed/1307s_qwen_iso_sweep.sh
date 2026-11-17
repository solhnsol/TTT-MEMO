#!/bin/bash
# JOB 1307s = QWEN3-4B-BASE ISO WRITE-FEASIBILITY SWEEP (EXPLORATORY, no pre-registered rule; not queued until Hans' sentence; GPU one job at a time; Hans 11/17 'iso로 하이퍼파라미터 튜닝을 해보면서 바인딩이 써지는 게 가능은 한지, 바인딩 마진 기준'; runs after job 1307 and before job 1308 / 1309 by file name (1307_ < 1307s_ < 1308_ < 1309_ in every sort order)): can the binding be written at all by the ISO-STD write (fact sentences only, no context; scripts/exp09_qwen_ttt/cell_iso_ladder.py) in Qwen3-4B-Base, judged by the out-of-window binding margin (summed span log-prob gold minus best wrong, 16 candidates, mean over the 16 facts) and the out-of-window top-1 (chance 0.0625) at exposure 128, with the canary? World seeds 161 and 162 (162 optional: edit SEEDS), ONE read at exposure 128 (out + in-window), one segment (0,128] of 2048 rows (~50 packed steps), configurations of one process share one model load and start from the same initial weights; W0 reads reused from job 1306. AXES (a configuration name = the directory under experiments/exp09_qwen_ttt/iso_sweep/<group>/): group lora (LoRA r16 last 9 layers, as 1306): m1 = norm matching ON x1 (update norm = the NATS cumulative norm of the seed, ISO-STD as in 1309), m3 / m10 = matching x3 / x10 (larger writes), p3e-2 / p1e-1 / p3e-1 = pure ISO (norm matching OFF, raw clipped SGD) at lr 3e-2 / 1e-1 / 3e-1, mom / adam = momentum 0.9 / Adam direction with the matching x1 (E55 ISO-MOM / ISO-ADAM); group fullmlp (full MLP matrices of the last 9 layers): fm_m1 matched, fm_p1e-2 / fm_p3e-2 pure ISO lr 1e-2 / 3e-2; group l18 (LoRA r16 on the last 18 layers): matched and pure lr 1e-1; group r64 (LoRA r64, alpha 128, last 9 layers): matched and pure lr 1e-1. Each configuration prints a SWEEP line (out margin and gain over W0, top-1, in margin, canary change, update norm, clip share) next to the NATS reference of job 1306 (same seeds). ESTIMATE: per configuration and seed ~1.8 min (~50 steps ~40 s, one read ~40 s, canary ~15 s); 15 configurations x 2 seeds = 30 runs ~54 min + 4 model loads ~8 min = ~62 min (seed 161 only: ~35 min).
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT   # the torch env of exp09 (the job runner sets the JAX environment)
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; M=experiments/exp09_qwen_ttt/cband_main; O=experiments/exp09_qwen_ttt/iso_sweep
SEEDS=161,162
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
mkdir -p $O
cat > experiments/exp09_qwen_ttt/iso_sweep_pinned_sha256.txt <<'PIN'
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
sha256sum -c experiments/exp09_qwen_ttt/iso_sweep_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
for s in ${SEEDS//,/ }; do for f in $M/cband_NATS_s$s.json $M/cband_W0_s$s.json; do [ -f $f ] || { echo "missing reference $f: exit 14"; exit 14; }; done; done
CUDA_VISIBLE_DEVICES= $PYQ $E/cell_iso_ladder.py --selftest | tail -1
CUDA_VISIBLE_DEVICES= $PYQ $E/cband_iso_sweep_summary.py --selftest | tail -1
RUN="$PYQ $E/cell_iso_ladder.py --seed-list $SEEDS --ref-dir $M --reads 128 --clip 1.0"
$RUN --fast lora --lr 3e-2 --out $O/lora --sweep "m1:;m3:norm_mult=3;m10:norm_mult=10;p3e-2:iso_norm=off,lr=0.03;p1e-1:iso_norm=off,lr=0.1;p3e-1:iso_norm=off,lr=0.3;mom:iso_opt=mom;adam:iso_opt=adam"
$PYQ $E/cband_iso_sweep_summary.py --dir $O/lora --main $M --seeds $SEEDS --json $O/summary_lora.json
$RUN --fast fullmlp --lr 1e-2 --out $O/fullmlp --sweep "fm_m1:;fm_p1e-2:iso_norm=off,lr=0.01;fm_p3e-2:iso_norm=off,lr=0.03"
$PYQ $E/cband_iso_sweep_summary.py --dir $O/fullmlp --main $M --seeds $SEEDS --json $O/summary_fullmlp.json
$RUN --fast lora --lr 3e-2 --lora-layers 18 --out $O/l18 --sweep "l18_m1:;l18_p1e-1:iso_norm=off,lr=0.1"
$PYQ $E/cband_iso_sweep_summary.py --dir $O/l18 --main $M --seeds $SEEDS --json $O/summary_l18.json
$RUN --fast lora --lr 3e-2 --lora-r 64 --lora-alpha 128 --out $O/r64 --sweep "r64_m1:;r64_p1e-1:iso_norm=off,lr=0.1"
$PYQ $E/cband_iso_sweep_summary.py --dir $O/r64 --main $M --seeds $SEEDS --json $O/summary_r64.json
echo "outputs: $O/summary_{lora,fullmlp,l18,r64}.json"
