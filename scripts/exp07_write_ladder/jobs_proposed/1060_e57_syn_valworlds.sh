#!/bin/bash
# [Hans 11/06 02:07 KST: this job prepares the TUNE worlds only (s62 s71 s72); the hold-out worlds s65 s73 s74 are prepared by job 1100 after job 1090.]
# JOB 1060 = E57 SYNTHETIC VALIDATION WORLDS, PREPARATION ONLY (no arms; GPM thread, Hans 11/05 'a b c 승인' via coordinator): six worlds of three densities (facts per 1024-token chunk, 128 chunks each): tune s62 (v3_16_1, density 16), s71 (v3_4_1, 4), s72 (v3_32_1, 32; new cell registered in e57_longcell.py); hold-out s65 (v3_16_1), s73 (v3_4_1), s74 (v3_32_1). Per world: (1) window-gain pass with band 128 only (e57_syn_w0pass.py --bands 128, own output directory, then copied to e57_syn_s<seed>/syn_s<seed>_nll128.npz; the band-256 files of job 1050 are not touched); (2) for the four seeds without a NAT norm file (71 72 73 74) the plain NAT write (arm NATSGD of e57_syn_mrun.py, --ntg-nat auto, no inline reads) -> e57_synq/E55_synqnat_NATSGD_s<seed>.saved.json (s62 and s65 use E55_v3a_NAT_s<seed>.saved.json). s64 is not run.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_longcell.py e57_syn_w0pass.py e57_syn_mrun.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_w0pass.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1
SQ=$OUT/e57_synq; DUMMY=$SQ/syn_s64_m30_masks.npz
[ -f "$DUMMY" ] || { echo "input missing: $DUMMY (exit 7)"; exit 7; }
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
for W in ${WORLDS:-62:v3_16_1 71:v3_4_1 72:v3_32_1}; do
  SEED=${W%%:*}; CELL=${W##*:}; DIR=$OUT/e57_syn_s$SEED; mkdir -p "$DIR"
  if [ ! -f "$DIR/syn_s${SEED}_nll128.npz" ]; then
    $PY $S/e57_syn_w0pass.py --seed $SEED --cell $CELL --out "$OUT/e57_syn_b128_s$SEED" --bands 128
    cp "$OUT/e57_syn_b128_s$SEED/syn_s${SEED}_nll.npz" "$DIR/syn_s${SEED}_nll128.npz"; cp "$OUT/e57_syn_b128_s$SEED/syn_s${SEED}_nll.json" "$DIR/syn_s${SEED}_nll128.json"
  fi
  case $SEED in 71|72|73|74)
    [ -f "$SQ/E55_synqnat_NATSGD_s$SEED.saved.json" ] || $PY $S/e57_syn_mrun.py --seed $SEED --cell $CELL --out $SQ --tag synqnat --m30 $DUMMY --ntg-nat auto --arms NATSGD --save-apps 128 --no-inline;;
  esac
done
echo "outputs (worlds ${WORLDS:-62 71 72}): $OUT/e57_syn_s<seed>/syn_s<seed>_nll128.npz; NAT norms $SQ/E55_synqnat_NATSGD_s<seed>.saved.json for the new seeds (s62, s65: E55_v3a_NAT)"
