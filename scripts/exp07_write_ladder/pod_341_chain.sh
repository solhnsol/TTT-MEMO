#!/bin/bash
# E57 job 341 (news full-chunk value dump, W0, no write; about 15 MB output) on a RunPod pod, volume at /workspace. Start command:  bash -c "echo <base64 of this file> | base64 -d > /tmp/chain.sh; bash /tmp/chain.sh <check|run|serve>"
# check = CPU pod: overlay the pinned code, verify hashes, CPU selftests and --check-only, list inputs (no GPU step). run = A100 pod: the same checks, then the GPU job. serve = CPU pod: only the read-only HTTP server for the finished output (output stays on the volume after the GPU pod is deleted).
# Logs: /workspace/exp07_out/pod_logs_341/ (read-only HTTP on port 8000); the dump is linked as pod_logs_341/dump -> $X/e57_syn_s63 (nothing else is served). End markers: CHAIN_DONE / CHAIN_FAILED. After the marker the script waits GRACE seconds and exits, which stops the pod (safety net; the thread deletes the pod at the marker).
set -u
MODE=${1:-check}; GRACE=${GRACE:-1200}; RUN_TIMEOUT=${RUN_TIMEOUT:-18000}
RAW=https://raw.githubusercontent.com/solhnsol/TTT-MEMO
X=/workspace/exp07_out; L=$X/pod_logs_341; mkdir -p "$L"; ln -sfn $X/e57_syn_s63 $L/dump
PYS=$(command -v python3 || ls /workspace/uv-python/*/bin/python3 | head -1)
(cd "$L" && nohup "$PYS" -m http.server 8000 --bind 0.0.0.0 > /tmp/http.log 2>&1 &)
log(){ echo "$(date -u +%T) $*" >> $L/stages.log; }
end(){ log "$1"; sleep $GRACE; exit ${2:-0}; }
stage(){ local name=$1; shift; log "STAGE_START $name"; "$@" > $L/$name.log 2>&1; local rc=$?; log "STAGE_END $name rc=$rc"; if [ $rc -ne 0 ]; then log "last lines of $name:"; tail -8 $L/$name.log >> $L/stages.log; end "CHAIN_FAILED at $name (rc=$rc)" $rc; fi; }
log "START mode=$MODE"
if [ "$MODE" = serve ]; then (cd $X/e57_syn_s63 && find . -type f -exec sha256sum {} + > $L/sha256_all.txt) 2>>$L/stages.log; du -sh $X/e57_syn_s63 >> $L/stages.log; end "SERVE_READY (waiting $GRACE s)" 0; fi
# per-job code tree (scripts/ is a private copy, everything else is linked to the shared tree), so jobs running at the same time on the volume cannot overwrite each other's pinned files
C=/workspace/code_341
if [ ! -d $C/scripts ]; then mkdir -p $C; for e in /workspace/TTT-MEMO/* /workspace/TTT-MEMO/.[!.]*; do b=$(basename $e); [ "$b" = scripts ] || ln -sfn $e $C/$b; done; cp -a /workspace/TTT-MEMO/scripts $C/scripts; fi
du -sh $C/scripts >> $L/stages.log
mkdir -p /home/solhnsol/workspace && ln -sfn $C /home/solhnsol/workspace/TTT-MEMO
source /workspace/rp_env.sh
export PYTHONUNBUFFERED=1
cd $C
D=scripts/exp07_write_ladder; J=$D/jobs_proposed/339_e57_syn_pass_base.sh; BK=/workspace/code_backup_341
overlay(){
  local c=$1 f=$2 t=/tmp/ov.$$; mkdir -p "$(dirname $f)"
  curl -fsSL --retry 4 --retry-delay 3 -o $t "$RAW/$c/$f" || return 1
  [ -s $t ] || return 1
  if [ -f "$f" ] && [ ! -f "$BK/$f" ]; then mkdir -p "$BK/$(dirname $f)"; cp -p "$f" "$BK/$f"; fi
  cp $t "$f"; rm -f $t
}
fetch_all(){ (
  set -e
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_run.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_core.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_gpu.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_swap.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_read.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_analyze.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_metrics.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_ic_report.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_iso.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_diag.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_votegrid.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_uv.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/s3.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/rope_fly.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_sentdump.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_sentdump_labels.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_sentsel.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_sentgrid.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_selmask.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_gates.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_bgbooks.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_bgbooks_sha.json
  overlay 13904abb21ef295590784448002d6163141bad2d $D/e55_rev.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/../exp04_what_ttt_retains/gen_facts.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/../exp04_what_ttt_retains/gen_world_ctl.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/../exp04_what_ttt_retains/frame_bank.py
  overlay 13904abb21ef295590784448002d6163141bad2d $D/../exp04_what_ttt_retains/frames300.txt
  overlay 63803e02133718a03ecef9c32973532962fee925 $D/e57_closure_run.py
  overlay 63803e02133718a03ecef9c32973532962fee925 $D/e57_closure_masks.py
  overlay 63803e02133718a03ecef9c32973532962fee925 $D/e57_qband_masks.py
  overlay 63803e02133718a03ecef9c32973532962fee925 $D/e57_qband_run.py
  overlay 63803e02133718a03ecef9c32973532962fee925 $D/e57_keyq_run.py
  overlay 63803e02133718a03ecef9c32973532962fee925 $D/e57_keyq_masks.py
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
  overlay fc45473aaf91d09ecf5fbe1df24b579cd13f698e $D/e57_attdump.py
  overlay 2703daa42eb3a158e90b2b36d732d014479bfc54 $D/e57_attdump2.py
  overlay 0cf9f36ebd7973ce2da466bf9503743d6a09678f $D/e57_attdump3.py
  overlay 95e3161d67745a093ce46522014c3e5648a8d45a $D/e57_valuedump.py
  overlay 17e7e50167eaa8db156d1c29140293595f90ae26 $D/e57_syn_pass.py
  overlay 17e7e50167eaa8db156d1c29140293595f90ae26 $D/e57_valuedump_full.py
  overlay 9fe47c0bf1f44cfe5eaba3802cec3c7269acf05e $J
  [ "$(sha256sum $J | cut -d' ' -f1)" = caebe834453d3256b05cc004c5b55b6a14425bceec0eac40f33701e2e84bb342 ]
) }
envinfo(){ date -u; uname -a; df -h /workspace | tail -1; nvidia-smi -L || echo "no gpu"; $PY -V; ls -la $X/e57_news/e57_bundle $X/e57_news/e57_bank | head -20; ls -la /workspace/TTT-MEMO/checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 | head -5; ls -la /workspace/TTT-MEMO/data/tokenizer/llama3/tokenizer.json; echo "out dir:"; ls $X/e57_syn_s63 2>&1 | head -3; echo "job-278 dump (cross-check input):"; ls $X/e57_pod/e57_pod_masks.npz $X/e57_windowless/E57_windowless_W0.npz 2>&1 | head -10; echo PY=$PY; }
stage envinfo envinfo
stage overlay fetch_all
inputs(){ df -h /workspace | tail -1; ls $X | head -40; ls -la /workspace/TTT-MEMO/checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 | head -3; [ $(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9) -ge 8 ] || { echo volume free space below 8 GB; return 1; }; dd if=/dev/zero of=$X/.wtest bs=1M count=512 conv=fsync > /dev/null 2>&1 || { echo volume write test FAILED; return 1; }; rm -f $X/.wtest; echo write-test-ok; }
stage inputs inputs
export OUT=$X MOUT=$X/e57_syn_s63 SEEDS=63 NO_INSTALL=1
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
