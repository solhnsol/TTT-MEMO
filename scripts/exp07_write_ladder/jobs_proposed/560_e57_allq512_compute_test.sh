#!/bin/bash
# JOB 560 = E57 ALLQ512 COMPUTE-REDUCTION TEST (prepared; NOT STARTED until Hans confirms job name + commit + time; Hans 05:17Z 'go' on the coordinator proposal): ALLQ512 (no selection, band-512 masking on every row of chunks >= 8; arm of job 291, never run on the news stream so no chunk-279 state exists) is written for ONE cycle together with NATQ (same SELC1 engine, no mask = NATADAMH, the same-engine wall-clock control), settings of job 291 (norm_targets from E57_main_NAT.arms.json, Adam direction 0.5 x NAT norm). Per-chunk write time of both arms is in update_log[].sec of E57_main_<arm>.arms.json (printed by the summary step, first compile chunk excluded). Then the forward-only T1/T2 OUT reads of the two chunk-279 states (e57_t1t2_allq_run.py = e57_t1t2_run.py of job 410: frozen t1t2_v1.json, 150 templates, read-only) and the canary at the cycle end. No design or threshold change. ESTIMATE ONLY: load 15 min + record 5 min + 2 arms x 6.5 min + canary + t1t2 scorers 30-40 min + 2 reads = about 70-80 min. NO git checkout (shared tree, see memory rule): the working-tree files are used as they are and their sha256 are printed.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; D4=scripts/exp04_what_ttt_retains; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_qband_run.py e57_qband_masks.py e57_closure_masks.py e57_main_run.py e57_t1t2_run.py e57_t1t2_allq_run.py e55_core.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_qband_masks.py --selftest | tail -1
$PY $S/e57_qband_run.py --selftest | tail -1
$PY $S/e57_t1t2_allq_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_allq512}; ARMS=NATQ,ALLQ512
FROZEN=${FROZEN:-$BUNDLE/e57_t1t2/t1t2_v1.json}; TOUT=${TOUT:-$OUT/e57_t1t2_allq512}; MANIFEST=$TOUT/e57_t1t2_states_manifest.json
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] && [ -f "$FROZEN" ] || { echo "bundle store, bank or frozen file missing (exit 7)"; exit 7; }
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_main_ALLQ512.arms.json" ] && [ -z "$RERUN" ] && { echo "E57_main_ALLQ512.arms.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
mkdir -p "$MOUT" "$TOUT"; [ -f "$MOUT/E57_main_W0.json" ] || cp "$OUT/e57_main/E57_main_W0.json" "$MOUT/E57_main_W0.json"
$PY $S/e57_qband_masks.py --allq-only --bands 512 --out "$MOUT/e57_qband_masks.npz"
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --npos 1 --canary-npos 10 --arms $ARMS --qband-masks $MOUT/e57_qband_masks.npz --ntg-from $OUT/e57_main/E57_main_NAT.arms.json"
$PY $S/e57_qband_run.py $ARGS --stage arms
$PY $S/e57_qband_run.py $ARGS --stage reads
$PY $S/e57_pod_summary.py --dir "$MOUT" --arms W0,$ARMS || true
$PY - "$MOUT" "$MANIFEST" <<'PYE'
import json,sys,numpy as np
mo,mf=sys.argv[1:3];m={};T={}
for arm in ('NATQ','ALLQ512'):
    r=json.load(open(f'{mo}/E57_main_{arm}.arms.json'));st=r['states']['279'];m[arm]=dict(file=st['file'],hash=st['hash'])
    s=np.array([e['sec'] for e in r['update_log'] if 'sec' in e]);T[arm]=dict(n=len(s),mean_excl_first=float(s[1:].mean()),median=float(np.median(s[1:])),first=float(s[0]),total_chunks_sec=float(s.sum()))
json.dump(m,open(mf,'w'),indent=1);json.dump(T,open(mo+'/e57_allq512_chunk_time.json','w'),indent=1);print('manifest',mf);print('per-chunk write time (s):',T)
PYE
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2_run.py --cpu-check --frozen "$FROZEN" --manifest "$MANIFEST" --out "$TOUT"
$PY $S/e57_t1t2_allq_run.py --bundle "$BUNDLE" --out "$TOUT" --bank "$BUNDLE/e57_bank/bank_v1.json" --cycles 1 --npos 1 --canary-npos 1 --arms ALLQ512,NATQ --stage reads --frozen "$FROZEN" --manifest "$MANIFEST"
echo "outputs: $MOUT/E57_main_ALLQ512.arms.json $MOUT/E57_main_NATQ.arms.json $MOUT/e57_allq512_chunk_time.json $TOUT/E57_t1t2_scores.json $TOUT/E57_t1t2_score_keys.json"
