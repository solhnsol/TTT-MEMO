#!/bin/bash
# JOB 1304s = QWEN3-4B WRITE-SIZE SWEEP SMOKE (technical; EXPLORATORY; not queued until Hans' sentence; GPU one job at a time; Hans 11/16 '스모크로 둘 다 스윕'): where does the NTP write of Qwen3-4B-Base (SWA-structure write of exp09, E57 cell v3_16_1 with the Qwen tokenizer, world seed 161) become measurable? Arm NATS only (no mask), reads at exposures 64 and 128 (out-of-window and in-window), W0 reads, pg11 canary. Two fast-weight families in PRIORITY ORDER (the order of the runs; each run prints a SWEEP line when it finishes so the first results can be read while the rest runs): (1) fullmlp = full MLP matrices of the last 9 layers as fast weights (closest to the prime FFN of TTT-E2E, larger write capacity in the exp09 multi-pass results), lr 1e-3, 1e-2, 1e-4, 3e-3, 3e-4; (2) LoRA r16 on the MLPs of the last 9 layers (the 1301s setting), lr 1e-2, 1e-1, 3e-2, 3e-3, 3e-1; clip 1.0 for all. Interleaved so that the two most informative runs of each family come first: fm 1e-3, fm 1e-2, lora 1e-2, lora 1e-1, then the rest. Criterion printed per run (technical, no pass / fail): margin gain over W0 at exposure 128 out-of-window AND in-window >= 1 nat = 'WRITES'; canary NLL rise > 0.05 nat / token = 'CANARY'. ESTIMATE ~5 min per run (write 128 chunks ~2-3 min + 2 reads + W0 read), 10 runs ~50 min. Output experiments/exp09_qwen_ttt/lr_sweep/<cfg>/ and the table lr_sweep/summary.json.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT   # the torch env of exp09 (the job runner sets the JAX environment)
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/lr_sweep
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for t in test_bandmask.py test_bandmask_vs_jax.py; do CUDA_VISIBLE_DEVICES= $PYQ $E/$t | tail -1; done
CUDA_VISIBLE_DEVICES= $PYQ $E/cband_sweep_summary.py --selftest | tail -1
mkdir -p $O
run() { # cfg fast lr
  [ -f $O/$1/cband_NATS_s161.json ] && [ -f $O/$1/cband_W0_s161.json ] || $PYQ $E/cell_band_ladder.py --stage run --world-seed 161 --arms NATS --reads 64,128 --fast $2 --lr $3 --clip 1.0 --out $O/$1
  $PYQ $E/cband_sweep_summary.py --dir $O --cfgs $1
}
run fm_lr1e-3 fullmlp 1e-3
run fm_lr1e-2 fullmlp 1e-2
run lora_lr1e-2 lora 1e-2
run lora_lr1e-1 lora 1e-1
run fm_lr1e-4 fullmlp 1e-4
run lora_lr3e-2 lora 3e-2
run fm_lr3e-3 fullmlp 3e-3
run lora_lr3e-3 lora 3e-3
run fm_lr3e-4 fullmlp 3e-4
run lora_lr3e-1 lora 3e-1
$PYQ $E/cband_sweep_summary.py --dir $O --json $O/summary.json
echo "outputs: $O/summary.json"
