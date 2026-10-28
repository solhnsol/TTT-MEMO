#!/bin/bash
# E57 jobs 296/297 (news RANDQ random-row control, W0 base, 2 arms; about 2.8 GiB output) on a RunPod pod, volume at /workspace. Start command:  bash -c "echo <base64 of this file> | base64 -d > /tmp/chain.sh; bash /tmp/chain.sh <check|run|serve>"
# check = CPU pod: overlay the pinned code, verify hashes, CPU selftests and --check-only, list inputs (no GPU step). run = A100 pod: the same checks, then the GPU job. serve = CPU pod: only the read-only HTTP server for the finished output (output stays on the volume after the GPU pod is deleted).
# Logs: /workspace/exp07_out/pod_logs_296_p$PP/ (read-only HTTP on port 8000); the dump is linked as pod_logs_296_p$PP/dump -> $X/e57_randq_p$PP (nothing else is served). End markers: CHAIN_DONE / CHAIN_FAILED. After the marker the script waits GRACE seconds and exits, which stops the pod (safety net; the thread deletes the pod at the marker).
set -u
MODE=${1:-check}; PP=${2:-0.26}; GRACE=${GRACE:-1200}; RUN_TIMEOUT=${RUN_TIMEOUT:-7200}
RAW=https://raw.githubusercontent.com/solhnsol/TTT-MEMO
X=/workspace/exp07_out; L=$X/pod_logs_296_p$PP; mkdir -p "$L"; ln -sfn $X/e57_randq_p$PP $L/dump
PYS=$(command -v python3 || ls /workspace/uv-python/*/bin/python3 | head -1)
(cd "$L" && nohup "$PYS" -m http.server 8000 --bind 0.0.0.0 > /tmp/http.log 2>&1 &)
log(){ echo "$(date -u +%T) $*" >> $L/stages.log; }
end(){ log "$1"; sleep $GRACE; exit ${2:-0}; }
stage(){ local name=$1; shift; log "STAGE_START $name"; "$@" > $L/$name.log 2>&1; local rc=$?; log "STAGE_END $name rc=$rc"; if [ $rc -ne 0 ]; then log "last lines of $name:"; tail -8 $L/$name.log >> $L/stages.log; end "CHAIN_FAILED at $name (rc=$rc)" $rc; fi; }
log "START mode=$MODE"
if [ "$MODE" = serve ]; then (cd $X/e57_randq_p$PP && find . -type f -exec sha256sum {} + > $L/sha256_all.txt) 2>>$L/stages.log; du -sh $X/e57_randq_p$PP >> $L/stages.log; end "SERVE_READY (waiting $GRACE s)" 0; fi
# per-job code tree (scripts/ is a private copy, everything else is linked to the shared tree), so jobs running at the same time on the volume cannot overwrite each other's pinned files
C=/workspace/code_296_p$PP
if [ ! -d $C/scripts ]; then mkdir -p $C; for e in /workspace/TTT-MEMO/* /workspace/TTT-MEMO/.[!.]*; do b=$(basename $e); [ "$b" = scripts ] || ln -sfn $e $C/$b; done; cp -a /workspace/TTT-MEMO/scripts $C/scripts; fi
du -sh $C/scripts >> $L/stages.log
mkdir -p /home/solhnsol/workspace && ln -sfn $C /home/solhnsol/workspace/TTT-MEMO
source /workspace/rp_env.sh
export PYTHONUNBUFFERED=1
cd $C
D=scripts/exp07_write_ladder; J296=$D/jobs_proposed/296_e57_news_randq_band_write_arms.sh; J297=$D/jobs_proposed/297_e57_news_randq_band_write_arms_p010.sh; case $PP in 0.10) J=$J297;; *) J=$J296;; esac; BK=/workspace/code_backup_296_p$PP
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
  overlay 7fd3cb815d770b14e1c80389658ac667c73c0efd $D/e57_closure_masks.py
  overlay 7fd3cb815d770b14e1c80389658ac667c73c0efd $D/e57_qband_masks.py
  overlay 7fd3cb815d770b14e1c80389658ac667c73c0efd $D/e57_qband_run.py
  overlay 7fd3cb815d770b14e1c80389658ac667c73c0efd $J296
  [ "$(sha256sum $J296 | cut -d' ' -f1)" = 7ea454fbfb4efb6b14110455f76e8892f268365e54511f8a4da83c978dc43e6a ]
  overlay 7fd3cb815d770b14e1c80389658ac667c73c0efd $J297
  [ "$(sha256sum $J297 | cut -d' ' -f1)" = bed6d4a0b119628d8615b6b833bd56f8d113bfa5a38a16872231c8b42410e1c2 ]
) }
envinfo(){ date -u; uname -a; df -h /workspace | tail -1; nvidia-smi -L || echo "no gpu"; $PY -V; ls -la $X/e57_news/e57_bundle $X/e57_news/e57_bank | head -20; ls -la /workspace/TTT-MEMO/checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 | head -5; ls -la /workspace/TTT-MEMO/data/tokenizer/llama3/tokenizer.json; echo "out dir:"; ls $X/e57_randq_p$PP 2>&1 | head -3; echo "job-278 dump (cross-check input):"; ls $X/e57_pod/e57_pod_masks.npz $X/e57_windowless/E57_windowless_W0.npz 2>&1 | head -10; echo PY=$PY; }
stage envinfo envinfo
stage overlay fetch_all
inputs(){ mkdir -p $X/e57_main $X/e57_band; [ -f $X/e57_band/e57_band_W0.npz ] || cp -n $X/e57_band/E57_closure_W0.npz $X/e57_band/e57_band_W0.npz; sha256sum $X/e57_band/e57_band_W0.npz $X/e57_windowless/E57_windowless_W0.npz; [ -f $X/e57_main/E57_main_NAT.arms.json ] || cp -n $X/E57_main_NAT.arms.json $X/e57_main/E57_main_NAT.arms.json; [ -f $X/e57_main/E57_main_W0.json ] || { [ -f $X/e57_pod/E57_main_W0.json ] && cp -n $X/e57_pod/E57_main_W0.json $X/e57_main/E57_main_W0.json && echo "W0 reads of job 261 are not on the volume: copied the W0 reads of the pod run (e57_pod, same values as job 261 except timings and bank path; checked on the lab copies)"; }; ls -la $X/e57_main $X/e57_pod 2>&1 | head -40; for f in $X/e57_main/E57_main_NAT.arms.json $X/e57_main/E57_main_W0.json; do [ -f $f ] && sha256sum $f || { echo "MISSING $f"; return 1; }; done; }
stage inputs inputs
export OUT=$X BUNDLE=$X/e57_news MOUT=$X/e57_randq_p$PP NO_INSTALL=1 P=$PP ARMS=RANDQ128,RANDQ512
stage cpucheck env CPU_ONLY=1 bash $J
if [ "$MODE" != run ]; then end "CHAIN_DONE (check only)" 0; fi
stage gpu nvidia-smi
stage run timeout $RUN_TIMEOUT bash $J
(cd $MOUT && sha256sum * > $L/sha256_closure.txt) 2>>$L/stages.log
ls -la $MOUT >> $L/stages.log
du -sh $MOUT >> $L/stages.log
end "CHAIN_DONE" 0
