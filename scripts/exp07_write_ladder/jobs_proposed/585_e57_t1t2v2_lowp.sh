#!/bin/bash
# JOB 585 = E57 T1/T2 v2 LOW MASK BUDGET GAINQ128 p=.10 vs RANDQ128 p=.10 (prepared; NOT QUEUED until Hans confirms job name + commit; Hans 14:41 KST 'ok, approve the experiment'): same frozen v2 setup, reader and norm targets as jobs 570 / 575 (comparators already measured there: W0, NAT, NATQ, ALLQ128 p=1, GAINQ128 p=.26). Stream A: the existing chunk-279 states of GAINQ128 p=.10 (e57_gainq_p0.10, hash d9157f6f...) and RANDQ128 p=.10 (e57_randq_p0.10, hash 83ef0fde...) are only read (hash checked). Stream B20: GAINQ128P10 (top 10% of the eligible rows per chunk 8..285 by nll_band128 - nll_win from the ALREADY SAVED W0 score file of job 575, e57_t1t2v2_gainq/E57_v2B20_score_W0.npz: no new scoring pass) and RANDQ128P10 (random 10% of the eligible rows, default_rng(585)) written for one cycle (SELC1 engine, Adam direction, 0.5 x the NAT norm of the 570 B20 NAT arm), end canary (gen / tail rise) recorded in the arms.json trajectory, then the reads. New file e57_t1t2v2_lowp_run.py; no existing e57_*.py edited, no git checkout. Output $TOUT: E57_t1t2v2_scores.json + E57_t1t2v2_score_keys.json (same format as 570 / 575), E57_v2B20_{GAINQ128P10,RANDQ128P10}.arms.json, e57_v2B20_lowp_masks.npz. ESTIMATE ONLY, unmeasured: model load 15 min + B20 record 5 min + 2 writes x ~7 min + reads (load + 4502 candidate rows ~15 min, W0 + 4 states) ~20 min = ~55-60 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_t1t2v2_lowp_run.py e57_t1t2v2_gainq_run.py e57_t1t2v2_run.py e57_qband_run.py e57_closure_masks.py e57_t1t2_run.py e57_main_run.py e55_core.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_t1t2v2_lowp_run.py --selftest | tail -1
$PY $S/e57_t1t2v2_run.py --selftest | tail -1
IN=${IN:-$OUT/e57_t1t2v2_in}; TOUT=${TOUT:-$OUT/e57_t1t2v2_lowp}; P570=${P570:-$OUT/e57_t1t2v2}; SCORE=${SCORE:-$OUT/e57_t1t2v2_gainq/E57_v2B20_score_W0.npz}; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}
chk(){ [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "$1 differs from the frozen copy (exit 9)"; exit 9; }; }
AG=$OUT/e57_gainq_p0.10/E57_main_GAINQ128.arms.json; AR=$OUT/e57_randq_p0.10/E57_main_RANDQ128.arms.json
[ -f "$AG" ] && [ -f "$AR" ] && [ -f "$SCORE" ] && [ -f "$BANK" ] && [ -f "$IN/t1t2_v2_frozen.json" ] && [ -f "$P570/streamB20/E57_v2B20_NAT.arms.json" ] || { echo "input missing (exit 7)"; exit 7; }
chk $IN/t1t2_v2_frozen.json f459c1a08b60eeb8e1adf01f766deff1e9574328912d30eb64147801b8b04d0f
chk $IN/e57_t1t2v2d3_candidates.json 8384a05ef434b30a665a67360c2450a987d454e6637c64041b891981a8e97854
chk $IN/streamB20_stream_sentences.jsonl ee7b1dc00622a55f395e63360a96095faf397dfc87081b9563022c759e055c85
mkdir -p "$TOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2v2_run.py --cpu-check --frozen $IN/t1t2_v2_frozen.json --cands $IN/e57_t1t2v2d3_candidates.json --bank "$BANK" --sentences-b $IN/streamB20_stream_sentences.jsonl --out "$TOUT"
$PY $S/e57_t1t2v2_lowp_run.py --stage mask --sentences $IN/streamB20_stream_sentences.jsonl --score "$SCORE" --out "$TOUT" --p 0.10
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_t1t2v2_scores.json" ] && [ -z "$RERUN" ] && { echo "E57_t1t2v2_scores.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_t1t2v2_lowp_run.py --stage write --sentences $IN/streamB20_stream_sentences.jsonl --out "$TOUT" --nat-ref $P570/streamB20/E57_v2B20_NAT.arms.json --arms GAINQ128P10,RANDQ128P10
MAN=$TOUT/e57_t1t2v2_states_manifest.json
$PY - "$AG" "$AR" "$TOUT" "$MAN" <<'PYE'
import json,sys
ag,ar,t,mf=sys.argv[1:5];m={}
for lab,f,h in (('GAINQ128P10',ag,'d9157f6fc403'),('RANDQ128P10',ar,'83ef0fde5d28')):
    s=json.load(open(f))['states']['279'];assert s['hash'].startswith(h),(lab,s['hash'][:12]);m[lab+'_A']=dict(label=lab,stream='A',file=s['file'],hash=s['hash'])
    s=json.load(open(f'{t}/E57_v2B20_{lab}.arms.json'))['states']['285'];m[lab+'_B']=dict(label=lab,stream='B',file=s['file'],hash=s['hash'])
json.dump(m,open(mf,'w'),indent=1);print('manifest',{k:(v['stream'],v['hash'][:12]) for k,v in m.items()})
PYE
$PY $S/e57_t1t2v2_run.py --stage read --frozen $IN/t1t2_v2_frozen.json --cands $IN/e57_t1t2v2d3_candidates.json --bank "$BANK" --manifest "$MAN" --out "$TOUT"
echo "outputs: $TOUT/E57_t1t2v2_scores.json $TOUT/E57_t1t2v2_score_keys.json $TOUT/E57_v2B20_GAINQ128P10.arms.json $TOUT/E57_v2B20_RANDQ128P10.arms.json"
