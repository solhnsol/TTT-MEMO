#!/bin/bash
# JOB 575 = E57 T1/T2 v2 GAINQ128 p=.26 (prepared; NOT QUEUED until Hans confirms job name + commit; Hans 01:41 KST 'GAINQ로도 해볼까' after job 570): the GAINQ write arm on the SAME frozen v2 setup as job 570 (t1t2_v2_frozen.json sha f459c1a0..., candidates, OUT context, per-token mean, same reader e57_t1t2v2_run.py read stage; 570 arms: W0, NAT, NATQ, ALLQ128). Stream A: the existing chunk-279 state of GAINQ128 p=.26 (job 292b, hash 386e0950..., same SELC1 engine / Adam / 0.5 x NAT norm as the ALLQ128 arm of 570) is only read. Stream B20: new forward-only W0 scoring pass (natural + band-128 NLL per row), top 26% of eligible rows per chunk (chunks 8..285) by nll_band128 - nll_win, GAINQ128 written for one cycle (norm targets of the NAT arm of 570 on B20), then the reads. NOT included: the prefix-only (GAINQ-P) write arm (job 561 scored band 512 on stream A only, and a prefix-only WRITE needs a changed write path: separate job if wanted). New file e57_t1t2v2_gainq_run.py; no existing e57_*.py edited, no git checkout (working-tree hashes are printed). Output $TOUT: E57_v2B20_score_W0.npz, e57_v2B20_gainq_masks.npz, E57_v2B20_GAINQ128.arms.json, E57_t1t2v2_scores.json + E57_t1t2v2_score_keys.json (compare with e57_t1t2v2/ of job 570). ESTIMATE ONLY, unmeasured: model load 15 min + B20 record 5 min + score pass ~10 min + write ~7 min + reads (load, 4502 candidate rows ~15 min, W0 + 2 states) ~15 min = ~65-70 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_t1t2v2_gainq_run.py e57_t1t2v2_run.py e57_qband_run.py e57_closure_masks.py e57_t1t2_run.py e57_main_run.py e55_core.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_t1t2v2_gainq_run.py --selftest | tail -1
$PY $S/e57_t1t2v2_run.py --selftest | tail -1
IN=${IN:-$OUT/e57_t1t2v2_in}; TOUT=${TOUT:-$OUT/e57_t1t2v2_gainq}; P570=${P570:-$OUT/e57_t1t2v2}; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}
chk(){ [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "$1 differs from the frozen copy (exit 9)"; exit 9; }; }
A_ARMS=$OUT/e57_gainq_p0.26/E57_main_GAINQ128.arms.json
[ -f "$A_ARMS" ] && [ -f "$BANK" ] && [ -f "$IN/t1t2_v2_frozen.json" ] && [ -f "$P570/streamB20/E57_v2B20_NAT.arms.json" ] || { echo "input missing (exit 7)"; exit 7; }
chk $IN/t1t2_v2_frozen.json f459c1a08b60eeb8e1adf01f766deff1e9574328912d30eb64147801b8b04d0f
chk $IN/e57_t1t2v2d3_candidates.json 8384a05ef434b30a665a67360c2450a987d454e6637c64041b891981a8e97854
chk $IN/streamB20_stream_sentences.jsonl ee7b1dc00622a55f395e63360a96095faf397dfc87081b9563022c759e055c85
mkdir -p "$TOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2v2_run.py --cpu-check --frozen $IN/t1t2_v2_frozen.json --cands $IN/e57_t1t2v2d3_candidates.json --bank "$BANK" --sentences-b $IN/streamB20_stream_sentences.jsonl --out "$TOUT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_t1t2v2_scores.json" ] && [ -z "$RERUN" ] && { echo "E57_t1t2v2_scores.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_t1t2v2_gainq_run.py --stage score --sentences $IN/streamB20_stream_sentences.jsonl --out $TOUT
$PY $S/e57_t1t2v2_gainq_run.py --stage mask --sentences $IN/streamB20_stream_sentences.jsonl --out $TOUT --p 0.26
$PY $S/e57_t1t2v2_gainq_run.py --stage write --sentences $IN/streamB20_stream_sentences.jsonl --out $TOUT --nat-ref $P570/streamB20/E57_v2B20_NAT.arms.json
MAN=$TOUT/e57_t1t2v2_states_manifest.json
$PY - "$A_ARMS" "$TOUT" "$MAN" <<'PYE'
import json,sys
a,t,mf=sys.argv[1:4];m={}
s=json.load(open(a))['states']['279'];assert s['hash'].startswith('386e09509062'),s['hash'][:12];m['GAINQ128_A']=dict(label='GAINQ128',stream='A',file=s['file'],hash=s['hash'])
s=json.load(open(t+'/E57_v2B20_GAINQ128.arms.json'))['states']['285'];m['GAINQ128_B']=dict(label='GAINQ128',stream='B',file=s['file'],hash=s['hash'])
json.dump(m,open(mf,'w'),indent=1);print('manifest',{k:(v['stream'],v['hash'][:12]) for k,v in m.items()})
PYE
$PY $S/e57_t1t2v2_run.py --stage read --frozen $IN/t1t2_v2_frozen.json --cands $IN/e57_t1t2v2d3_candidates.json --bank "$BANK" --manifest "$MAN" --out "$TOUT"
echo "outputs: $TOUT/E57_t1t2v2_scores.json $TOUT/E57_t1t2v2_score_keys.json $TOUT/E57_v2B20_GAINQ128.arms.json $TOUT/e57_v2B20_gainq_masks.npz"
