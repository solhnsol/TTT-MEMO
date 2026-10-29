#!/bin/bash
# E57 suffix-band gain (job 313) on a RunPod A100 pod (volume at /workspace). NOT APPROVED by the pod rule until Hans names the pod action; start command of the pod, same log method as pod_t1t2_chain.sh: stage boundaries and errors go to /workspace/exp07_out/pod_logs_suffixband/stages.log. Needs on the volume: /workspace/TTT-MEMO with the code of jobs 289/290 (e57_closure_masks.py d520326, e57_main_run.py 7b74b02), the bundle and bank at /workspace/exp07_out/e57_news, optional /workspace/exp07_out/e57_windowless/E57_windowless_W0.npz (control), and the two NEW files e57_suffixband_run.py and jobs_proposed/313_e57_news_suffixband_w0.sh (this repo, branch claude/project-thread-uo9xqo) copied into scripts/exp07_write_ladder/. Pod: host RAM >= 128 GB (minRAMPerGPU 128). Partial results: E57_suffixband_W0.partial.npz every 16 chunks.
set -u
mkdir -p /home/solhnsol/workspace && ln -sfn /workspace/TTT-MEMO /home/solhnsol/workspace/TTT-MEMO
source /workspace/rp_env.sh
export PYTHONUNBUFFERED=1
X=/workspace/exp07_out; L=$X/pod_logs_suffixband; mkdir -p "$L"
cd /workspace/TTT-MEMO
J=scripts/exp07_write_ladder/jobs_proposed/313_e57_news_suffixband_w0.sh
stage(){ local name=$1; shift; echo "$(date -u +%T) STAGE_START $name" >> $L/stages.log; "$@" > $L/$name.log 2>&1; local rc=$?; echo "$(date -u +%T) STAGE_END $name rc=$rc" >> $L/stages.log; if [ $rc -ne 0 ]; then echo "$(date -u +%T) CHAIN_FAILED at $name (rc=$rc); last lines:" >> $L/stages.log; tail -5 $L/$name.log >> $L/stages.log; exit $rc; fi; }
export OUT=$X BUNDLE=$X/e57_news MOUT=$X/e57_suffixband REF=$X/e57_windowless/E57_windowless_W0.npz
stage gpu nvidia-smi
stage disk df -h /workspace
stage mem free -g
stage cpucheck env CPU_ONLY=1 bash $J
stage run bash $J
echo "$(date -u +%T) CHAIN_DONE" >> $L/stages.log
