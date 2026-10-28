#!/bin/bash
# E57 job 288 (news token-level attention dump v2, W0, no write; about 7 GiB output) on a RunPod pod, volume at /workspace. Start command:  bash -c "echo <base64 of this file> | base64 -d > /tmp/chain.sh; bash /tmp/chain.sh <check|run|serve>"
# check = CPU pod: overlay the pinned code, verify hashes, CPU selftests and --check-only, list inputs (no GPU step). run = A100 pod: the same checks, then the GPU job. serve = CPU pod: only the read-only HTTP server for the finished output (output stays on the volume after the GPU pod is deleted).
# Logs: /workspace/exp07_out/pod_logs_288/ (read-only HTTP on port 8000); the dump is linked as pod_logs_288/dump -> $X/e57_attdump2 (nothing else is served). End markers: CHAIN_DONE / CHAIN_FAILED. After the marker the script waits GRACE seconds and exits, which stops the pod (safety net; the thread deletes the pod at the marker).
set -u
MODE=${1:-check}; GRACE=${GRACE:-1200}; RUN_TIMEOUT=${RUN_TIMEOUT:-10800}
RAW=https://raw.githubusercontent.com/solhnsol/TTT-MEMO
X=/workspace/exp07_out; L=$X/pod_logs_288; mkdir -p "$L"; ln -sfn $X/e57_attdump2 $L/dump
PYS=$(command -v python3 || ls /workspace/uv-python/*/bin/python3 | head -1)
(cd "$L" && nohup "$PYS" -m http.server 8000 --bind 0.0.0.0 > /tmp/http.log 2>&1 &)
log(){ echo "$(date -u +%T) $*" >> $L/stages.log; }
end(){ log "$1"; sleep $GRACE; exit ${2:-0}; }
stage(){ local name=$1; shift; log "STAGE_START $name"; "$@" > $L/$name.log 2>&1; local rc=$?; log "STAGE_END $name rc=$rc"; if [ $rc -ne 0 ]; then log "last lines of $name:"; tail -8 $L/$name.log >> $L/stages.log; end "CHAIN_FAILED at $name (rc=$rc)" $rc; fi; }
log "START mode=$MODE"
if [ "$MODE" = serve ]; then (cd $X/e57_attdump2 && sha256sum * > $L/sha256_attdump2.txt) 2>>$L/stages.log; ls -la $X/e57_attdump2 >> $L/stages.log; end "SERVE_READY (waiting $GRACE s)" 0; fi
mkdir -p /home/solhnsol/workspace && ln -sfn /workspace/TTT-MEMO /home/solhnsol/workspace/TTT-MEMO
source /workspace/rp_env.sh
export PYTHONUNBUFFERED=1
cd /workspace/TTT-MEMO
D=scripts/exp07_write_ladder; J=$D/jobs_proposed/288_e57_news_token_attention_dump_v2_w0.sh; BK=/workspace/code_backup_288
overlay(){
  local c=$1 f=$2 t=/tmp/ov.$$; mkdir -p "$(dirname $f)"
  curl -fsSL --retry 4 --retry-delay 3 -o $t "$RAW/$c/$f" || return 1
  [ -s $t ] || return 1
  if [ -f "$f" ] && [ ! -f "$BK/$f" ]; then mkdir -p "$BK/$(dirname $f)"; cp -p "$f" "$BK/$f"; fi
  cp $t "$f"; rm -f $t
}
fetch_all(){ (
  set -e
  for f in e57_main_run.py e57_main_report.py e57_attdump_run.py e57_attdump.py e57_mask_stage.py; do overlay 6bf5250047fe95da4404071bd3325de1711bb12d $D/$f; done
  for f in e57_attdump2.py e57_attdump2_run.py; do overlay 770742cec1978b0ed881db8d7ab5e34a5c0b2f38 $D/$f; done
  overlay c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 $D/e57_reads.py
  overlay c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 $D/e57_trial_run.py
  overlay c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 $D/e57/e57_bank.py
  overlay 57c50b5ce5dee7e3efea0ba036c79ab4eb6e6f02 $D/e57_canary_run.py
  overlay 5c835409e9a32b77829fdb20e4022a6547f06705 $D/e56/e56_text.py
  overlay 5c835409e9a32b77829fdb20e4022a6547f06705 $D/e57/e57_common.py
  overlay 514a35bbb6279e07ff056f2455890484afcad608 $J
  [ "$(sha256sum $J | cut -d' ' -f1)" = a79144c0458a2102f4ddea7d0399d8347f9efd19b8c08504d2c237c9f0a3c7b5 ]
) }
envinfo(){ date -u; uname -a; df -h /workspace | tail -1; nvidia-smi -L || echo "no gpu"; $PY -V; ls -la $X/e57_news/e57_bundle $X/e57_news/e57_bank | head -20; ls -la /workspace/TTT-MEMO/checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 | head -5; ls -la /workspace/TTT-MEMO/data/tokenizer/llama3/tokenizer.json; echo "dump out dir:"; ls $X/e57_attdump2 2>&1 | head -3; echo "job-278 dump (cross-check input):"; ls $X/e57_attdump 2>&1 | head -10; echo PY=$PY; }
stage envinfo envinfo
stage overlay fetch_all
export OUT=$X BUNDLE=$X/e57_news MOUT=$X/e57_attdump2 NO_INSTALL=1
stage cpucheck env CPU_ONLY=1 bash $J
if [ "$MODE" != run ]; then end "CHAIN_DONE (check only)" 0; fi
stage gpu nvidia-smi
stage run timeout $RUN_TIMEOUT bash $J
(cd $MOUT && sha256sum * > $L/sha256_attdump2.txt) 2>>$L/stages.log
ls -la $MOUT >> $L/stages.log
du -sh $MOUT >> $L/stages.log
end "CHAIN_DONE" 0
