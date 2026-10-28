#!/bin/bash
# E57 job 291 (news closure masks, W0, no write; about 20 MB output) on a RunPod pod, volume at /workspace. Start command:  bash -c "echo <base64 of this file> | base64 -d > /tmp/chain.sh; bash /tmp/chain.sh <check|run|serve>"
# check = CPU pod: overlay the pinned code, verify hashes, CPU selftests and --check-only, list inputs (no GPU step). run = A100 pod: the same checks, then the GPU job. serve = CPU pod: only the read-only HTTP server for the finished output (output stays on the volume after the GPU pod is deleted).
# Logs: /workspace/exp07_out/pod_logs_291/ (read-only HTTP on port 8000); the dump is linked as pod_logs_291/dump -> $X/e57_qband (nothing else is served). End markers: CHAIN_DONE / CHAIN_FAILED. After the marker the script waits GRACE seconds and exits, which stops the pod (safety net; the thread deletes the pod at the marker).
set -u
MODE=${1:-check}; GRACE=${GRACE:-1200}; RUN_TIMEOUT=${RUN_TIMEOUT:-7200}
RAW=https://raw.githubusercontent.com/solhnsol/TTT-MEMO
X=/workspace/exp07_out; L=$X/pod_logs_291; mkdir -p "$L"; ln -sfn $X/e57_qband $L/dump
PYS=$(command -v python3 || ls /workspace/uv-python/*/bin/python3 | head -1)
(cd "$L" && nohup "$PYS" -m http.server 8000 --bind 0.0.0.0 > /tmp/http.log 2>&1 &)
log(){ echo "$(date -u +%T) $*" >> $L/stages.log; }
end(){ log "$1"; sleep $GRACE; exit ${2:-0}; }
stage(){ local name=$1; shift; log "STAGE_START $name"; "$@" > $L/$name.log 2>&1; local rc=$?; log "STAGE_END $name rc=$rc"; if [ $rc -ne 0 ]; then log "last lines of $name:"; tail -8 $L/$name.log >> $L/stages.log; end "CHAIN_FAILED at $name (rc=$rc)" $rc; fi; }
log "START mode=$MODE"
if [ "$MODE" = serve ]; then (cd $X/e57_qband && sha256sum * > $L/sha256_closure.txt) 2>>$L/stages.log; ls -la $X/e57_qband >> $L/stages.log; for f in $X/e57_pod/e57_pod_masks.npz $X/e57_windowless/E57_windowless_W0.npz; do if [ -f $f ]; then echo "INPUT $(sha256sum $f)" >> $L/stages.log; else echo "INPUT MISSING $f" >> $L/stages.log; fi; done; ls $X/e57_news/e57_bundle $X/e57_news/e57_bank 2>&1 | head -5 >> $L/stages.log; end "SERVE_READY (waiting $GRACE s)" 0; fi
mkdir -p /home/solhnsol/workspace && ln -sfn /workspace/TTT-MEMO /home/solhnsol/workspace/TTT-MEMO
source /workspace/rp_env.sh
export PYTHONUNBUFFERED=1
cd /workspace/TTT-MEMO
D=scripts/exp07_write_ladder; J=$D/jobs_proposed/291_e57_news_allq_band_write_arms.sh; BK=/workspace/code_backup_291
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
  overlay 399c3de92258f26b513eddb565cd00748ced07dc $D/e57_closure_masks.py
  overlay 399c3de92258f26b513eddb565cd00748ced07dc $D/e57_qband_masks.py
  overlay 399c3de92258f26b513eddb565cd00748ced07dc $D/e57_qband_run.py
  overlay 96c439849a8d49eac37c61c121305eb26587e24f $J
  [ "$(sha256sum $J | cut -d' ' -f1)" = 9ff9b5d3843ef2383bdee6c8c6133396fa3f78a74083517c13a0dccf362c320f ]
) }
envinfo(){ date -u; uname -a; df -h /workspace | tail -1; nvidia-smi -L || echo "no gpu"; $PY -V; ls -la $X/e57_news/e57_bundle $X/e57_news/e57_bank | head -20; ls -la /workspace/TTT-MEMO/checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 | head -5; ls -la /workspace/TTT-MEMO/data/tokenizer/llama3/tokenizer.json; echo "out dir:"; ls $X/e57_qband 2>&1 | head -3; echo "job-278 dump (cross-check input):"; ls $X/e57_pod/e57_pod_masks.npz $X/e57_windowless/E57_windowless_W0.npz 2>&1 | head -10; echo PY=$PY; }
stage envinfo envinfo
stage overlay fetch_all
inputs(){ mkdir -p $X/e57_main; [ -f $X/e57_main/E57_main_NAT.arms.json ] || cp -n $X/E57_main_NAT.arms.json $X/e57_main/E57_main_NAT.arms.json; ls -la $X/e57_main $X/E57_main_* 2>&1 | head -30; for f in $X/e57_main/E57_main_NAT.arms.json $X/e57_main/E57_main_W0.json; do [ -f $f ] && sha256sum $f || { echo "MISSING $f"; return 1; }; done; }
stage inputs inputs
export OUT=$X BUNDLE=$X/e57_news MOUT=$X/e57_qband NO_INSTALL=1
stage cpucheck env CPU_ONLY=1 bash $J
if [ "$MODE" != run ]; then end "CHAIN_DONE (check only)" 0; fi
stage gpu nvidia-smi
stage run timeout $RUN_TIMEOUT bash $J
(cd $MOUT && sha256sum * > $L/sha256_closure.txt) 2>>$L/stages.log
ls -la $MOUT >> $L/stages.log
du -sh $MOUT >> $L/stages.log
end "CHAIN_DONE" 0
