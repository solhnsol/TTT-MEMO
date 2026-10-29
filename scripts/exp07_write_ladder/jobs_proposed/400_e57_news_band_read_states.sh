#!/bin/bash
# JOB 400 = E57 NEWS BAND-512 READS OF THE WRITTEN STATES (post-hoc; Hans via the discussion thread 04:26Z "응 승인할게"; forward only, NO weight update, no write): e57_band_read_run.py reads the 42 questions of the frozen bank from the saved states at chunk 279 of the one-cycle news stream with a BAND context (every token of the read sees only the previous 512 tokens, itself included; prefix and suffix layers; neutral OUT tail and IN window), per state the question rows of e57_reads.log_rows (gold log-prob, rank, rank ratio, accuracy, every candidate log-prob) with condition OUT_B512 / IN_B512; the ordinary window reads (OUT / IN) of the same states are taken from the stored arm files (set RECOMPUTE_WINDOW=1 to recompute them). States: W0, NAT, NATADAMH, GAINQ512 p.26, RANDQ512 p.10, GAINQ256 p.26, RANDQ256 p.26, GAINQ128 p.26 (job 345b), GAINQ512x2, GAINQ512x4, NATQx2 (job 347), optional GAINQ512 p.10 (WITH_P10=1). Output $OUT/e57_band_reads/E57_bandread_<LABEL>.json + E57_bandread_control.json. Control of every run: band 8192 with the index-distance prefix mask reproduces Packed.r1 of W0 (stop above 0.05). NOT APPROVED yet (needs Hans job name + commit + time; the queue entry is held until Hans approves). ESTIMATE ONLY, unmeasured: model load about 15 min + scorers for OUT / IN (Packed, about 7 min each time they are built: ordinary, band 512, control band 8192 = about 21 min) + compiles + control about 8 min + 11 states x about 1.2 min = about 1 h (50-80 min). The GPU code path of e57_band_read_run.py has not been run before this job.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
sha256sum $D/e57_band_read_run.py $D/e57_closure_masks.py | cut -c1-16,65-
$PY $S/e57_band_read_run.py --selftest
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_band_reads}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
LISTS=$($PY - "$OUT" "${WITH_P10:-}" <<'PYE'
import json,sys,os
o=sys.argv[1];p10=sys.argv[2]
arms=[('NAT',o+'/e57_main/E57_main_NAT'),('NATADAMH',o+'/e57_main/E57_main_NATADAMH'),('GAINQ512_p026',o+'/e57_gainq_p0.26/E57_main_GAINQ512'),('RANDQ512_p010',o+'/e57_randq_p0.10/E57_main_RANDQ512'),('GAINQ256_p026',o+'/e57_gainq256_p0.26/E57_main_GAINQ256'),('RANDQ256_p026',o+'/e57_randq256_p0.26/E57_main_RANDQ256'),('GAINQ128_p026',o+'/e57_gainq_p0.26/E57_main_GAINQ128'),
 ('GAINQ512x2',o+'/e57_lrscale_p0.26/E57_main_GAINQ512x2'),('GAINQ512x4',o+'/e57_lrscale_p0.26/E57_main_GAINQ512x4'),('NATQx2',o+'/e57_lrscale_p0.26/E57_main_NATQx2')]
if p10:arms.append(('GAINQ512_p010',o+'/e57_gainq_p0.10/E57_main_GAINQ512'))
rs=['W0=@W0'];ref=['W0='+o+'/e57_main/E57_main_W0.json']
for lab,base in arms:
    st=json.load(open(base+'.arms.json'))['states']['279'];f=st['file'] if 'file' in st else st[0];h=st['hash']
    assert os.path.exists(f),('state missing',lab,f);rs.append(f'{lab}={f}:{h}');ref.append(f'{lab}={base}.json')
print(','.join(rs));print(','.join(ref))
PYE
)
RS=$(echo "$LISTS" | sed -n 1p); REF=$(echo "$LISTS" | sed -n 2p)
echo "states: $RS"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_bandread_NAT.json" ] && [ -z "$RERUN" ] && { echo "E57_bandread_NAT.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
EXTRA=""; [ -n "$RECOMPUTE_WINDOW" ] && EXTRA="--recompute-window"
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --rstates $RS --refjson $REF --bands 512 $EXTRA"
[ -n "$RERUN" ] && ARGS="$ARGS --rerun"
$PY $S/e57_band_read_run.py $ARGS --stage reads
chmod -R a+rX "$MOUT"
