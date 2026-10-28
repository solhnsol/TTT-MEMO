#!/bin/bash
# E57 job 287a (news window-less NLL, W0, no write) on a RunPod pod, volume at /workspace. Start command of the pod:  bash -c "echo <base64 of this file> | base64 -d > /tmp/chain.sh; bash /tmp/chain.sh <check|run>"
# check = CPU pod: overlay the pinned code, verify hashes, CPU selftests and --check-only (no GPU step). run = A100 pod: the same checks, then the GPU job.
# Logs: /workspace/exp07_out/pod_logs_287a/ (read-only HTTP on port 8000, nothing else is served). End markers: CHAIN_DONE / CHAIN_FAILED. After the end marker the script waits GRACE seconds and exits, which stops the pod (safety net; the thread deletes the pod at the marker).
set -u
MODE=${1:-check}; GRACE=${GRACE:-1200}; RUN_TIMEOUT=${RUN_TIMEOUT:-7200}
RAW=https://raw.githubusercontent.com/solhnsol/TTT-MEMO
X=/workspace/exp07_out; L=$X/pod_logs_287a; R=$L/results; mkdir -p "$L" "$R"
PYS=$(command -v python3 || ls /workspace/uv-python/*/bin/python3 | head -1)
(cd "$L" && nohup "$PYS" -m http.server 8000 --bind 0.0.0.0 > /tmp/http.log 2>&1 &)
log(){ echo "$(date -u +%T) $*" >> $L/stages.log; }
end(){ log "$1"; sleep $GRACE; exit ${2:-0}; }
stage(){ local name=$1; shift; log "STAGE_START $name"; "$@" > $L/$name.log 2>&1; local rc=$?; log "STAGE_END $name rc=$rc"; if [ $rc -ne 0 ]; then log "last lines of $name:"; tail -8 $L/$name.log >> $L/stages.log; end "CHAIN_FAILED at $name (rc=$rc)" $rc; fi; }
log "START mode=$MODE"
mkdir -p /home/solhnsol/workspace && ln -sfn /workspace/TTT-MEMO /home/solhnsol/workspace/TTT-MEMO
source /workspace/rp_env.sh
export PYTHONUNBUFFERED=1
cd /workspace/TTT-MEMO
D=scripts/exp07_write_ladder; J=$D/jobs_proposed/287a_e57_news_windowless_nll_w0.sh; BK=/workspace/code_backup_287a
# pinned overlay (commit path) -> same path in the volume tree; replaced files are backed up once
overlay(){
  local c=$1 f=$2 t=/tmp/ov.$$; mkdir -p "$(dirname $f)"
  curl -fsSL --retry 4 --retry-delay 3 -o $t "$RAW/$c/$f" || return 1
  [ -s $t ] || return 1
  if [ -f "$f" ] && [ ! -f "$BK/$f" ]; then mkdir -p "$BK/$(dirname $f)"; cp -p "$f" "$BK/$f"; fi
  cp $t "$f"; rm -f $t
}
fetch_all(){ (
  set -e
  overlay 7b74b024a1cd4117ec77e4a9884ad0f61074b415 $D/e57_main_run.py
  overlay 7b74b024a1cd4117ec77e4a9884ad0f61074b415 $D/e57_main_report.py
  overlay c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 $D/e57_reads.py
  overlay c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 $D/e57_trial_run.py
  overlay c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 $D/e57/e57_bank.py
  overlay 57c50b5ce5dee7e3efea0ba036c79ab4eb6e6f02 $D/e57_canary_run.py
  overlay 5c835409e9a32b77829fdb20e4022a6547f06705 $D/e56/e56_text.py
  overlay 5c835409e9a32b77829fdb20e4022a6547f06705 $D/e57/e57_common.py
  overlay 2d756f2d8e54a8132619de3a5817b650399e1140 $D/e57_windowless_run.py
  overlay f65fe7f89fe0c2b18e10d3c9f1219643a3e85624 $J
  [ "$(sha256sum $J | cut -d' ' -f1)" = 7c6c2cb53dd7ae85be2d4a41472f94807795850bc47c6d74a96c2de4b1d1ec1c ]
) }
envinfo(){ date -u; uname -a; df -h /workspace | tail -1; nvidia-smi -L || echo "no gpu"; $PY -V; ls -la $X/e57_news $X/e57_news/e57_bundle $X/e57_news/e57_bank; ls $X/e57_windowless 2>&1 | head; echo PY=$PY; }
stage envinfo envinfo
stage overlay fetch_all
export OUT=$X BUNDLE=$X/e57_news MOUT=$X/e57_windowless NO_INSTALL=1
stage cpucheck env CPU_ONLY=1 bash $J
if [ "$MODE" != run ]; then end "CHAIN_DONE (check only)" 0; fi
stage gpu nvidia-smi
stage run timeout $RUN_TIMEOUT bash $J
cp -p $MOUT/E57_windowless_W0.json $MOUT/E57_windowless_W0.npz "$R"/ 2>>$L/stages.log
(cd "$R" && sha256sum * > sha256.txt)
ls -la "$R" >> $L/stages.log
end "CHAIN_DONE" 0
