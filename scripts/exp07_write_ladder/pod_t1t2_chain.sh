#!/bin/bash
# E57 T1/T2 scoring on a RunPod A100 pod (volume at /workspace). NOT APPROVED (Hans must approve the pod separately). Start command of the pod; same log method as pod_run_chain.sh: every stage boundary and error goes to /workspace/exp07_out/pod_logs/stages.log (read-only log viewer on port 8000); the pod is deleted by the person who runs it at CHAIN_DONE.
# Needs on the volume, all under /workspace/t1t2_in/ (uploaded from the lab server; nothing of the running E57 chain is touched): exp07_out/e57_t1t2_states/state_main{NAT,NATADAMH,ORC,RAND}_279.npz (4 x 1,384,121,068 bytes), exp07_out/e57_t1t2_v1.json (frozen file), exp07_out/e57_t1t2_states_manifest.json, TTT-MEMO/scripts/exp07_write_ladder/{e57_t1t2_run.py,pod_t1t2_chain.sh,jobs_proposed/276_e57_t1t2_score_oracle_cycle1.sh}. The rest of the code (/workspace/TTT-MEMO) and the bundle / bank_v1.json (/workspace/exp07_out/e57_news) are used read-only; the 276 code in t1t2_in is copied over a FRESH copy of the code tree only when the chain starts (see CODE below).
set -u
mkdir -p /home/solhnsol/workspace && ln -sfn /workspace/TTT-MEMO /home/solhnsol/workspace/TTT-MEMO
source /workspace/rp_env.sh
export PYTHONUNBUFFERED=1
I=/workspace/t1t2_in
cd /workspace/TTT-MEMO
cp -n $I/TTT-MEMO/scripts/exp07_write_ladder/e57_t1t2_run.py scripts/exp07_write_ladder/ ; cp -n $I/TTT-MEMO/scripts/exp07_write_ladder/jobs_proposed/276_e57_t1t2_score_oracle_cycle1.sh scripts/exp07_write_ladder/jobs_proposed/   # CODE: only NEW files are added to the tree (cp -n never overwrites)
I=/workspace/t1t2_in; X=/workspace/exp07_out; L=$I/pod_logs; mkdir -p "$L"
J=scripts/exp07_write_ladder/jobs_proposed/276_e57_t1t2_score_oracle_cycle1.sh
stage(){ local name=$1; shift; echo "$(date -u +%T) STAGE_START $name" >> $L/stages.log; "$@" > $L/$name.log 2>&1; local rc=$?; echo "$(date -u +%T) STAGE_END $name rc=$rc" >> $L/stages.log; if [ $rc -ne 0 ]; then echo "$(date -u +%T) CHAIN_FAILED at $name (rc=$rc); last lines:" >> $L/stages.log; tail -5 $L/$name.log >> $L/stages.log; exit $rc; fi; }
export OUT=$X BUNDLE=$X/e57_news FROZEN=$I/exp07_out/e57_t1t2_v1.json MANIFEST=$I/exp07_out/e57_t1t2_states_manifest.json STATE_DIR=$I/exp07_out/e57_t1t2_states TOUT=$I/score_out NO_INSTALL=1
stage gpu nvidia-smi
stage cpucheck env CPU_ONLY=1 bash $J
stage reads bash $J
echo "$(date -u +%T) CHAIN_DONE" >> $L/stages.log
