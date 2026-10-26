#!/bin/bash
# E57 T1/T2 scoring on a RunPod A100 pod (volume at /workspace). NOT APPROVED (Hans must approve the pod separately). Start command of the pod; same log method as pod_run_chain.sh: every stage boundary and error goes to /workspace/exp07_out/pod_logs/stages.log (read-only log viewer on port 8000); the pod is deleted by the person who runs it at CHAIN_DONE.
# Needs on the volume (copy before the pod starts, from the lab server): $X/e57_t1t2_states/state_main{NAT,NATADAMH,ORC,RAND}_279.npz (4 x 1,384,121,068 bytes), $X/e57_t1t2_v1.json (frozen file), $X/e57_t1t2_states_manifest.json, scripts/exp07_write_ladder/{e57_t1t2_run.py,jobs_proposed/276_e57_t1t2_score_oracle_cycle1.sh} in /workspace/TTT-MEMO (the rest of the code and the bundle / bank_v1.json are already there from the E57 pod chain).
set -u
mkdir -p /home/solhnsol/workspace && ln -sfn /workspace/TTT-MEMO /home/solhnsol/workspace/TTT-MEMO
source /workspace/rp_env.sh
export PYTHONUNBUFFERED=1
cd /workspace/TTT-MEMO
X=/workspace/exp07_out; L=$X/pod_logs; mkdir -p "$L"
J=scripts/exp07_write_ladder/jobs_proposed/276_e57_t1t2_score_oracle_cycle1.sh
stage(){ local name=$1; shift; echo "$(date -u +%T) STAGE_START $name" >> $L/stages.log; "$@" > $L/$name.log 2>&1; local rc=$?; echo "$(date -u +%T) STAGE_END $name rc=$rc" >> $L/stages.log; if [ $rc -ne 0 ]; then echo "$(date -u +%T) CHAIN_FAILED at $name (rc=$rc); last lines:" >> $L/stages.log; tail -5 $L/$name.log >> $L/stages.log; exit $rc; fi; }
export OUT=$X BUNDLE=$X/e57_news FROZEN=$X/e57_t1t2_v1.json MANIFEST=$X/e57_t1t2_states_manifest.json STATE_DIR=$X/e57_t1t2_states TOUT=$X/e57_t1t2_score NO_INSTALL=1
stage gpu nvidia-smi
stage cpucheck env CPU_ONLY=1 bash $J
stage reads bash $J
echo "$(date -u +%T) CHAIN_DONE" >> $L/stages.log
