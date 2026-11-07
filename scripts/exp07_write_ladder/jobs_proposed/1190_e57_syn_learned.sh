#!/bin/bash
# JOB 1190 = E57 LEARNED-SELECTOR WRITE ARMS on ONE hold-out world (GPM thread, Hans 11/07 '그래 한번 보자'): arms M3TLBW16N05 (model b: logistic regression on the W0 last-prefix-layer hidden state trained on the TRUE deciding rows of s62/s71/s72) and M3TLAW16N05 (model a: trained on g128>2 rows); per chunk the top ceil(5%) eligible rows by score get loss weight 16, all rows ALLQ256 (M3 structure), update norm N05. Masks: e57_synq/syn_s<seed>_learned.npz (e57_selector_syn_cpu.py --masks-only, keys qtLB / qtLA). Baselines (ALLQ256, M3 w=8, TOP5, ABS2) are NOT rerun: their per-token reads are merged from e57_syn_rules_s<seed>/ (same seed, same stream, W0 checked). Usage: SEED=65 CELL=v3_16_1 PY=/abs/python bash 1190_e57_syn_learned.sh
set -eo pipefail
export PYTHONUNBUFFERED=1
: "${SEED:?set SEED}"; : "${CELL:?set CELL}"
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
ARMS=M3TLBW16N05,M3TLAW16N05
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_selector_syn_cpu.py e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_mrun.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=$OUT/e57_syn_learned_s$SEED; mkdir -p "$TT"
if [ -f "$OUT/E55_v3a_NAT_s$SEED.saved.json" ]; then NTG=$OUT/E55_v3a_NAT_s$SEED.saved.json; else NTG=$SQ/E55_synqnat_NATSGD_s$SEED.saved.json; fi
for f in $NTG $OUT/e57_syn_s$SEED/syn_s${SEED}_nll128.npz $SQ/syn_s${SEED}_learned.npz $OUT/e57_syn_rules_s$SEED/e57_syn_tokread_s$SEED.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
$PY -I - <<PYE
import numpy as np
d=np.load("$SQ/syn_s${SEED}_learned.npz");ks=[k for k in d.files if k.startswith("qtLB_")];assert len(ks)==120 and all(("qtLA_"+k[5:]) in d.files for k in ks),len(ks);print("learned masks ok",len(ks))
PYE
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e57_syn_mrun.py --seed $SEED --cell $CELL --out $SQ --tag synqrl --m30 $SQ/syn_s${SEED}_learned.npz --ntg-nat $NTG --arms $ARMS --save-apps 16,32,64,96,128
$PY $S/e57_syn_tokread.py --stage read --seed $SEED --cell $CELL --out "$TT" --tag synqrl --arms $ARMS
$PY -I - <<PYE
import json
O="$OUT";a=json.load(open(O+"/e57_syn_rules_s$SEED/e57_syn_tokread_s$SEED.json"));b=json.load(open("$TT/e57_syn_tokread_s$SEED.json"))
assert a["tokids"]==b["tokids"];mx=max(abs(sum(x)-sum(y)) for x,y in zip(a["W0"],b["W0"]));assert mx<1e-3,mx
a["arms"].update(b["arms"]);a["W0_vs_rules_max_abs_diff_sum"]=mx;json.dump(a,open("$TT/e57_syn_tokread_s${SEED}_merged.json","w"));print("merged arms",sorted(a["arms"]))
PYE
$PY $S/e57_curve_report.py --kind syn --seed $SEED --tag synqrl --tokread $TT/e57_syn_tokread_s${SEED}_merged.json --out $TT/e57_curve_learned_s$SEED.json
echo "outputs: $TT/e57_curve_learned_s$SEED.json (merged tokread $TT/e57_syn_tokread_s${SEED}_merged.json)"
