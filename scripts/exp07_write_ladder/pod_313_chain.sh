#!/bin/bash
# E57 job 313 (news full-chunk value dump, W0, no write; about 15 MB output) on a RunPod pod, volume at /workspace. Start command:  bash -c "echo <base64 of this file> | base64 -d > /tmp/chain.sh; bash /tmp/chain.sh <check|run|serve>"
# check = CPU pod: overlay the pinned code, verify hashes, CPU selftests and --check-only, list inputs (no GPU step). run = A100 pod: the same checks, then the GPU job. serve = CPU pod: only the read-only HTTP server for the finished output (output stays on the volume after the GPU pod is deleted).
# Logs: /workspace/exp07_out/pod_logs_313/ (read-only HTTP on port 8000); the dump is linked as pod_logs_313/dump -> $X/e57_valuedump_full (nothing else is served). End markers: CHAIN_DONE / CHAIN_FAILED. After the marker the script waits GRACE seconds and exits, which stops the pod (safety net; the thread deletes the pod at the marker).
set -u
MODE=${1:-check}; GRACE=${GRACE:-1200}; RUN_TIMEOUT=${RUN_TIMEOUT:-7200}
RAW=https://raw.githubusercontent.com/solhnsol/TTT-MEMO
X=/workspace/exp07_out; L=$X/pod_logs_313; mkdir -p "$L"; ln -sfn $X/e57_valuedump_full $L/dump
PYS=$(command -v python3 || ls /workspace/uv-python/*/bin/python3 | head -1)
(cd "$L" && nohup "$PYS" -m http.server 8000 --bind 0.0.0.0 > /tmp/http.log 2>&1 &)
log(){ echo "$(date -u +%T) $*" >> $L/stages.log; }
end(){ log "$1"; sleep $GRACE; exit ${2:-0}; }
stage(){ local name=$1; shift; log "STAGE_START $name"; "$@" > $L/$name.log 2>&1; local rc=$?; log "STAGE_END $name rc=$rc"; if [ $rc -ne 0 ]; then log "last lines of $name:"; tail -8 $L/$name.log >> $L/stages.log; end "CHAIN_FAILED at $name (rc=$rc)" $rc; fi; }
log "START mode=$MODE"
if [ "$MODE" = serve ]; then (cd $X/e57_valuedump_full && find . -type f -exec sha256sum {} + > $L/sha256_all.txt) 2>>$L/stages.log; du -sh $X/e57_valuedump_full >> $L/stages.log; end "SERVE_READY (waiting $GRACE s)" 0; fi
# per-job code tree (scripts/ is a private copy, everything else is linked to the shared tree), so jobs running at the same time on the volume cannot overwrite each other's pinned files
C=/workspace/code_313
if [ ! -d $C/scripts ]; then mkdir -p $C; for e in /workspace/TTT-MEMO/* /workspace/TTT-MEMO/.[!.]*; do b=$(basename $e); [ "$b" = scripts ] || ln -sfn $e $C/$b; done; cp -a /workspace/TTT-MEMO/scripts $C/scripts; fi
du -sh $C/scripts >> $L/stages.log
mkdir -p /home/solhnsol/workspace && ln -sfn $C /home/solhnsol/workspace/TTT-MEMO
source /workspace/rp_env.sh
export PYTHONUNBUFFERED=1
cd $C
D=scripts/exp07_write_ladder; J=$D/jobs_proposed/313_e57_news_valuedump_full.sh; BK=/workspace/code_backup_313
overlay(){
  local c=$1 f=$2 t=/tmp/ov.$$; mkdir -p "$(dirname $f)"
  curl -fsSL --retry 4 --retry-delay 3 -o $t "$RAW/$c/$f" || return 1
  [ -s $t ] || return 1
  if [ -f "$f" ] && [ ! -f "$BK/$f" ]; then mkdir -p "$BK/$(dirname $f)"; cp -p "$f" "$BK/$f"; fi
  cp $t "$f"; rm -f $t
}
fetch_all(){ (
  set -e
  overlay 02fdc66bfb92d505486163179b573bb2f536e7db $D/e57_main_run.py
  overlay 02fdc66bfb92d505486163179b573bb2f536e7db $D/e57_main_report.py
  overlay 02fdc66bfb92d505486163179b573bb2f536e7db $D/e57_pod_run.py
  overlay 02fdc66bfb92d505486163179b573bb2f536e7db $D/e57_orca2_masks.py
  overlay 02fdc66bfb92d505486163179b573bb2f536e7db $D/e57_mask_stage.py
  overlay 02fdc66bfb92d505486163179b573bb2f536e7db $D/e57_attdump.py
  overlay 02fdc66bfb92d505486163179b573bb2f536e7db $D/e57_pod_summary.py
  overlay 02fdc66bfb92d505486163179b573bb2f536e7db $D/e57/e57_orc2.py
  overlay c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 $D/e57_reads.py
  overlay c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 $D/e57_trial_run.py
  overlay 57c50b5ce5dee7e3efea0ba036c79ab4eb6e6f02 $D/e57_canary_run.py
  overlay 5c835409e9a32b77829fdb20e4022a6547f06705 $D/e56/e56_text.py
  overlay 5c835409e9a32b77829fdb20e4022a6547f06705 $D/e57/e57_common.py
  overlay c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 $D/e57/e57_bank.py
  overlay c605ec573ae39eaae3e1a1563bbca0870c5b4e97 $D/e57_closure_run.py
  overlay c605ec573ae39eaae3e1a1563bbca0870c5b4e97 $D/e57_closure_masks.py
  overlay c605ec573ae39eaae3e1a1563bbca0870c5b4e97 $D/e57_qband_masks.py
  overlay c605ec573ae39eaae3e1a1563bbca0870c5b4e97 $D/e57_qband_run.py
  overlay 63803e02133718a03ecef9c32973532962fee925 $D/e57_keyq_run.py
  overlay 63803e02133718a03ecef9c32973532962fee925 $D/e57_keyq_masks.py
  overlay fc45473aaf91d09ecf5fbe1df24b579cd13f698e $D/e57_attdump.py
  overlay 2703daa42eb3a158e90b2b36d732d014479bfc54 $D/e57_attdump2.py
  overlay 0cf9f36ebd7973ce2da466bf9503743d6a09678f $D/e57_attdump3.py
  overlay 95e3161d67745a093ce46522014c3e5648a8d45a $D/e57_valuedump.py
  overlay c2c4feb6d53e8b82c168faa2ccb06dd26fa5f991 $D/e57_valuedump_full.py
  overlay 8cb78a4ed12760148c3cc0d3f50e7a8253ffb96f $J
  [ "$(sha256sum $J | cut -d' ' -f1)" = 21999b24f0830ac723803a40475ab8807fdcc0e4998eea9cf0d0037244ed38a3 ]
) }
envinfo(){ date -u; uname -a; df -h /workspace | tail -1; nvidia-smi -L || echo "no gpu"; $PY -V; ls -la $X/e57_news/e57_bundle $X/e57_news/e57_bank | head -20; ls -la /workspace/TTT-MEMO/checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 | head -5; ls -la /workspace/TTT-MEMO/data/tokenizer/llama3/tokenizer.json; echo "out dir:"; ls $X/e57_valuedump_full 2>&1 | head -3; echo "job-278 dump (cross-check input):"; ls $X/e57_pod/e57_pod_masks.npz $X/e57_windowless/E57_windowless_W0.npz 2>&1 | head -10; echo PY=$PY; }
stage envinfo envinfo
stage overlay fetch_all
inputs(){ ls -la $X/e57_news/e57_bundle $X/e57_news/e57_bank | head -8; ls -la $X/e57_sharp_heads.npz; if [ ! -f $X/e57_valuedump_rows.npz ] && [ -n "${ROWS_URL:-}" ]; then curl -fsSL --retry 4 --retry-delay 3 -o $X/e57_valuedump_rows.npz "$ROWS_URL"; fi; [ -f $X/e57_valuedump_rows.npz ] || { echo "MISSING $X/e57_valuedump_rows.npz (rows file, sha256 11a493bea00d90315e1bc99c8e3519ed330f6f899af28803e24f9dd3b3023b62)"; return 1; }; [ "$(sha256sum $X/e57_valuedump_rows.npz | cut -d' ' -f1)" = 11a493bea00d90315e1bc99c8e3519ed330f6f899af28803e24f9dd3b3023b62 ] || { echo "rows file hash differs"; return 1; }; [ -f $X/e57_sharp_heads.npz ] || { echo "MISSING sharp heads"; return 1; }; }
stage inputs inputs
export OUT=$X BUNDLE=$X/e57_news MOUT=$X/e57_valuedump_full NO_INSTALL=1
stage cpucheck env CPU_ONLY=1 bash $J
if [ "$MODE" != run ]; then end "CHAIN_DONE (check only)" 0; fi
stage gpu nvidia-smi
# run with a cost cap (RUN_TIMEOUT) and a no-progress watchdog: the job is stopped if its log has not been written for STALE seconds (default 75 min; the first layers of the record stage are silent for up to 40 min)
runwd(){ STALE=${STALE:-4500}; timeout $RUN_TIMEOUT bash $J & JP=$!; while kill -0 $JP 2>/dev/null; do sleep 60; AGE=$(( $(date +%s) - $(stat -c %Y $L/run.log 2>/dev/null || echo 0) )); if [ $AGE -gt $STALE ]; then echo "WATCHDOG: no log progress for $AGE s; stopping the job"; pkill -TERM -P $JP; kill -TERM $JP; sleep 20; kill -KILL $JP 2>/dev/null; wait $JP; return 124; fi; done; wait $JP; }
stage run runwd
(cd $MOUT && sha256sum * > $L/sha256_all.txt) 2>>$L/stages.log
ls -la $MOUT >> $L/stages.log
du -sh $MOUT >> $L/stages.log
end "CHAIN_DONE" 0
