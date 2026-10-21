#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 30 min, range 15-60 min (ESTIMATE, nothing of this path has run). Two official-trainer launches of the 125M outer loop (meta training, batch 8 x 8192 tokens, 5 steps each, dummy data, no checkpoint saved, no eval): per launch about 1 min checkpoint load + about 5-10 min JAX compile of the meta gradient (not measured; the second launch reuses the compile cache of the same graph only if XLA accepts the changed mask input, otherwise it compiles again) + 5 steps x about 3-10 s. CPU steps (unit test, needs CPU jax in the venv) about 1 min. The per-step time x 8 gives the estimate for one real step (batch 64 x 8192).
# What it does (design / core request 2026-10-21: can the outer loop be continued from the 125M checkpoint, and what does a step cost): runs the official trainer ttt.train (e2e/ unmodified) with training.load_part=params from checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc, training.dummy_dataset=true (random tokens 0..19: only timing and mechanics, NOT learning), global batch 8, accum_steps 4, 1 GPU, ilr_warmup_steps 0 (no inner-lr warmup restart), outer lr 1e-4 with 1 warmup step, wandb off, break_step 4 (steps 0..4 run, then the loop stops before the final checkpoint save and eval). Launch A: official data path. Launch B: MASK_FRAC=0.25, i.e. outer_smoke.py wraps lm_dataset._to_batch so that loss_masks = default AND a random mask removing 25% of the targets: those positions leave the inner write loss and the outer loss of every chunk (the loss-mask hook, outer_mask.py). Compare the logged loss and the step time of A and B; the logged line "#Trainable params" and the restore messages show whether the checkpoint was restored.
# GPU memory: per-device batch 8 with accum_steps 4 (micro-batch 2 x 8192 tokens); if it runs out of memory set ACCUM=8 (micro-batch 1). Nothing else may run on the GPU at the same time.
# PINNED INSTALL, NO FETCH: checks the 3 files out of the local repository at commit $COMMIT (exit 4 if absent: git fetch origin claude/project-thread-rwhdac first), verifies sha256 (exit 5). NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. Needs the official e2e/ checkout of the repository (submodule at the pinned commit a4fc4788ace3), the lab venv with the e2e dependencies, and checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc.
# CPU-only dry run (separate directory, no GPU, no checkpoint needed): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_o150 FETCH_HEAD && rmdir /tmp/dry_o150/e2e && ln -s "$(pwd)/e2e" /tmp/dry_o150/e2e && cd /tmp/dry_o150 && PY=python3 CPU_ONLY=1 bash scripts/exp08_125m/jobs_proposed/150_outer_smoke.sh   (afterwards: git worktree remove --force /tmp/dry_o150). It runs the unit test against the official loss.py and lists the paths the GPU step would use.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
ROOT=$(pwd);D=scripts/exp08_125m
COMMIT=bb211c88307848c0c7f2be31d77262189cb874ab
FILES="$D/outer_mask.py $D/outer_mask_test.py $D/outer_smoke.py"
ACCUM="${ACCUM:-4}"
CK=$ROOT/checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc
OUTD="${OUTD:-$ROOT/experiments/exp08_125m/outer_smoke}"   # own folder; the runner exports OUT=exp07_write_ladder, which is NOT used here
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH"; exit 5; }
  done <<'MANIFEST'
f037c4e01729 outer_mask.py
a0f00678f876 outer_mask_test.py
8eb69aa54c89 outer_smoke.py

MANIFEST
  echo "installed code verified against the manifest"
fi
[ -f $ROOT/e2e/ttt/model/loss.py ] || { echo "official e2e checkout missing (git submodule update --init e2e)"; exit 6; }
JAX_PLATFORMS=cpu $PY $D/outer_mask_test.py --e2e $ROOT/e2e
echo "checkpoint: $CK $([ -d $CK ] && echo present || echo MISSING); output: $OUTD"
[ -n "$CPU_ONLY" ] && exit 0
[ -d $CK ] || { echo "checkpoint missing: exit 7"; exit 7; }
mkdir -p $OUTD
ARGS="+deploy=interactive +experiment=125m/pretrain/pretrain-125m-e2e checkpoint.resume_checkpoint_dir=$CK training.load_part=params training.dummy_dataset=true deploy_paths.data.dclm_filter_8k=unused deploy_paths.data.books3=unused training.log_wandb=false training.wandb_entity=none training.wandb_project=none training.wandb_key=none backend.num_devices=1 training.global_batch_size=8 training.accum_steps=$ACCUM training.ilr_warmup_steps=0 training.optimizer_outer.lr=1e-4 training.optimizer_outer.lr_warmup_steps=1 training.optimizer_outer.lr_decay_steps=6 training.total_steps=6 training.break_step=4 training.checkpoint_path=$OUTD/ckpt training.exp_dir=$OUTD/logs"
ts(){ tr '\r' '\n' | while IFS= read -r l; do echo "$(date +%H:%M:%S) $l"; done; }
echo "=== launch A: official data path (no extra mask) ==="
E2E_ROOT=$ROOT/e2e MASK_FRAC=0 $PY -u $D/outer_smoke.py $ARGS training.exp_name=smokeA 2>&1 | ts | tee $OUTD/launch_A.log | tail -40
echo "=== launch B: 25% random loss mask ==="
E2E_ROOT=$ROOT/e2e MASK_FRAC=0.25 $PY -u $D/outer_smoke.py $ARGS training.exp_name=smokeB 2>&1 | ts | tee $OUTD/launch_B.log | tail -40
echo "per-step times: read the timestamps of the 'Outer Loop Training' lines in $OUTD/launch_{A,B}.log (step 0 includes the compile); the checkpoint directories under $OUTD/ckpt can be deleted."
