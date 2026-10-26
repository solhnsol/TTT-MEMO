#!/bin/bash
# E57 pod chain (RunPod A100, volume at /workspace): mask stage -> arms (smoke 12 first) -> reads -> summary. Every stage boundary and error goes to /workspace/exp07_out/pod_logs/stages.log (read-only log viewer on port 8000). Run: nohup bash /workspace/run_pod.sh > /workspace/exp07_out/pod_logs/chain.out 2>&1 &
set -u
mkdir -p /home/solhnsol/workspace && ln -sfn /workspace/TTT-MEMO /home/solhnsol/workspace/TTT-MEMO
source /workspace/rp_env.sh
export PYTHONUNBUFFERED=1
cd /workspace/TTT-MEMO
S=scripts/exp07_write_ladder; B=/workspace/exp07_out/e57_news; O=/workspace/exp07_out/e57_pod; L=/workspace/exp07_out/pod_logs; X=/workspace/exp07_out
mkdir -p "$O" "$L"
stage(){ local name=$1; shift; echo "$(date -u +%T) STAGE_START $name" >> $L/stages.log; "$@" > $L/$name.log 2>&1; local rc=$?; echo "$(date -u +%T) STAGE_END $name rc=$rc" >> $L/stages.log; if [ $rc -ne 0 ]; then echo "$(date -u +%T) CHAIN_FAILED at $name (rc=$rc); last lines:" >> $L/stages.log; tail -5 $L/$name.log >> $L/stages.log; exit $rc; fi; }
stage gpu nvidia-smi
stage selftest $PY $S/e57_mask_stage.py --selftest
stage mask $PY $S/e57_mask_stage.py --bundle $B --out $O --coef $X/e55_sentgrid_ext_s61_coef.npz --oracle-all
if [ -f $L/HOLD ]; then echo "$(date -u +%T) HOLD after the mask stage (remove $L/HOLD to continue)" >> $L/stages.log; while [ -f $L/HOLD ]; do sleep 15; done; echo "$(date -u +%T) HOLD released" >> $L/stages.log; fi
ARGS="--bundle $B --out $O --cycles 1 --npos 1 --canary-npos 1 --arms C1,LRN,RC1,RLRN,ORCA,RORCA --pod-masks $O/e57_pod_masks.npz --ntg-from $X/E57_main_NAT.arms.json --guard 4"
stage arms $PY $S/e57_pod_run.py $ARGS --stage arms
stage reads $PY $S/e57_pod_run.py $ARGS --stage reads
stage summary $PY $S/e57_pod_summary.py --dir $O
echo "$(date -u +%T) CHAIN_DONE" >> $L/stages.log
