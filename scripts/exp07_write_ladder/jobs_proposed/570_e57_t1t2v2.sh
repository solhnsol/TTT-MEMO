#!/bin/bash
# JOB 570 = E57 T1/T2 v2 (frozen reverse questions: 98 items / 30 events, sha256 f459c1a0..04d0f; Hans 14:39Z "둘다해봐": W0, NAT (plain SGD) AND NATQ (Adam, 0.5 x NAT norm), ALLQ k=128): stream A (e57_stream_sentences.jsonl, 280 chunks, ids_sha 715077df...) and stream B20 (streamB20_stream_sentences.jsonl, 286 chunks, ids_sha 13089ac9...), one cycle each. Writes: A = ALLQ128 only (NAT 5f235dd9efb9 and NATQ = NATADAMH 23fc03c9f71f states exist, hash checked); B20 = NAT (norm targets), NATQ, ALLQ128. Reads: v2 frames (question + cloze; forward_cloze NOT scored, no gold in the frozen file), candidates = gold (fixed surface first) + 20 easy (e57_t1t2v2d3_candidates.json) + nominated hard of the item, per-token mean primary, OUT context of jobs 276 / 410. No thresholds. Output $OUT/e57_t1t2v2/ (writes in e57_t1t2v2/streamA, streamB20; reads E57_t1t2v2_scores.json + E57_t1t2v2_score_keys.json). ESTIMATE ONLY: A write ~15 min + B20 write ~30 min + reads (load, 4502 candidate rows prepared ~15 min, 8 reads) ~25 min = ~70 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e ea1125ca7f76f70994e2dfe4dc56623aa7e0d407^{commit} || { echo "commit ea1125ca7f76f70994e2dfe4dc56623aa7e0d407 missing (exit 4)"; exit 4; }; git checkout ea1125ca7f76f70994e2dfe4dc56623aa7e0d407 -- $D/e57_t1t2v2_run.py; fi
sha256sum $D/e57_t1t2v2_run.py $D/e57_t1t2_run.py $D/e57_qband_run.py $D/e57_closure_masks.py | cut -c1-16,65-
for x in "3419b588f180 e57_t1t2_run.py" "c48a57e093ed e57_qband_run.py" "091506dbe7b8 e57_closure_masks.py" "d13a0ac9ee4a e57_main_run.py" "b19e986904ba e57_reads.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2: have $G, expected $1 (exit 5)"; exit 5; }; done
$PY $S/e57_t1t2v2_run.py --selftest | tail -1
IN=${IN:-$OUT/e57_t1t2v2_in}; TOUT=${TOUT:-$OUT/e57_t1t2v2}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}
chk(){ [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "$1 differs from the frozen copy (exit 9)"; exit 9; }; }
[ -f "$NATARMS" ] && [ -f "$BANK" ] && [ -f "$IN/t1t2_v2_frozen.json" ] || { echo "input missing (exit 7)"; exit 7; }
chk $IN/t1t2_v2_frozen.json f459c1a08b60eeb8e1adf01f766deff1e9574328912d30eb64147801b8b04d0f
chk $IN/e57_t1t2v2d3_candidates.json 8384a05ef434b30a665a67360c2450a987d454e6637c64041b891981a8e97854
chk $IN/e57_stream_sentences.jsonl 38ef483747a28a9334e5fba9fc7c12bc6fa2bdd92f8919a33faebe007cb4e8b6
chk $IN/streamB20_stream_sentences.jsonl ee7b1dc00622a55f395e63360a96095faf397dfc87081b9563022c759e055c85
mkdir -p "$TOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2v2_run.py --cpu-check --frozen $IN/t1t2_v2_frozen.json --cands $IN/e57_t1t2v2d3_candidates.json --bank "$BANK" --sentences-a $IN/e57_stream_sentences.jsonl --sentences-b $IN/streamB20_stream_sentences.jsonl --out "$TOUT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_t1t2v2_scores.json" ] && [ -z "$RERUN" ] && { echo "E57_t1t2v2_scores.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_t1t2v2_run.py --stage write --stream A --sentences $IN/e57_stream_sentences.jsonl --out $TOUT/streamA --nat-ref "$NATARMS" --arms ALLQ128
$PY $S/e57_t1t2v2_run.py --stage write --stream B20 --sentences $IN/streamB20_stream_sentences.jsonl --out $TOUT/streamB20 --arms NAT,NATQ,ALLQ128
MAN=$TOUT/e57_t1t2v2_states_manifest.json
$PY - "$OUT" "$TOUT" "$MAN" <<'PYE'
import json,sys
o,t,mf=sys.argv[1:4];m={}
def st(path,k):s=json.load(open(path))['states'][k];return dict(file=s['file'],hash=s['hash'])
m['NAT_A']=dict(label='NAT',stream='A',**st(o+'/e57_main/E57_main_NAT.arms.json','279'))
m['NATQ_A']=dict(label='NATQ',stream='A',**st(o+'/e57_adamb1/E57_main_NATADAMH.arms.json','279'))
m['ALLQ128_A']=dict(label='ALLQ128',stream='A',**st(t+'/streamA/E57_v2A_ALLQ128.arms.json','279'))
for a in ('NAT','NATQ','ALLQ128'):m[a+'_B']=dict(label=a,stream='B',**st(t+f'/streamB20/E57_v2B20_{a}.arms.json','285'))
assert m['NAT_A']['hash'].startswith('5f235dd9efb9') and m['NATQ_A']['hash'].startswith('23fc03c9f71f'),(m['NAT_A']['hash'][:12],m['NATQ_A']['hash'][:12])
json.dump(m,open(mf,'w'),indent=1);print('manifest',{k:(v['stream'],v['hash'][:12]) for k,v in m.items()})
PYE
$PY $S/e57_t1t2v2_run.py --stage read --frozen $IN/t1t2_v2_frozen.json --cands $IN/e57_t1t2v2d3_candidates.json --bank "$BANK" --manifest "$MAN" --out "$TOUT"
echo "outputs: $TOUT/E57_t1t2v2_scores.json $TOUT/E57_t1t2v2_score_keys.json (copy to the shared root as e57_t1t2v2_scores.json / e57_t1t2v2_score_keys.json); states in $TOUT/stream*/states/"
