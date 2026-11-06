#!/bin/bash
# JOB 1121 = HOLD-OUT world s65 (v3_16_1) with the chosen rule ABS tau 2 only (Hans 11/06 10:29 KST, '\ABS2로 가자'): arms M3TA2W16N05 via job 1070's script (TAUS=2). Run once; no other rule is put on the hold-out worlds.
set -eo pipefail
PY=$(realpath -s "$PY"); export PY
cd "$(dirname "$0")/../../.." 2>/dev/null || true
SEED=65 CELL=v3_16_1 TAUS=2 ARMS=M3TA2W16N05 bash scripts/exp07_write_ladder/jobs_proposed/1070_e57_syn_rules.sh
O=${EXP07_OUT:-experiments/exp07_write_ladder}; for a in M3W1N05 M3W8N05 M3T05W16N05; do [ -e $O/e57_synq/E55_synqrl_${a}_s65.read.json ] || ln -s E55_synqrp_${a}_s65.read.json $O/e57_synq/E55_synqrl_${a}_s65.read.json; done
$PY - <<PYE
import json,os,pathlib
O=pathlib.Path(os.environ.get("EXP07_OUT","experiments/exp07_write_ladder"));a=json.load(open(O/"e57_syn_repro_s65/e57_syn_tokread_s65.json"));b=json.load(open(O/"e57_syn_rules_s65/e57_syn_tokread_s65.json"))
assert a["tokids"]==b["tokids"];mx=max(abs(sum(x)-sum(y)) for x,y in zip(a["W0"],b["W0"]));assert mx<1e-3,mx
(O/"e57_syn_rules_s65/e57_syn_tokread_s65_ta2only.json").write_text(json.dumps(b));a["arms"].update(b["arms"]);a["W0_repro_vs_rules_max_abs_diff_sum"]=mx;json.dump(a,open(O/"e57_syn_rules_s65/e57_syn_tokread_s65.json","w"));print("merged arms",sorted(a["arms"]))
PYE
