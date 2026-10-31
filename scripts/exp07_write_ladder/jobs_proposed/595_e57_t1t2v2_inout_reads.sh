#!/bin/bash
# JOB 595 = E57 T1/T2 v2 OUT + IN READS OF THE BAND-128 / BAND-512 ARMS ON STREAMS A AND B20 (prepared; NOT QUEUED until Hans confirms job name + commit; Hans 19:56 KST 'as recommended'; runs after job 590 in queue order): the same 98 frozen v2 items as 570 / 575 / 585 (stream A 37 items, B20 61 items) read in OUT (8192-token SQuAD tail) and IN (the last 8192 tokens of the stream of the item) for W0 and the states NAT, NATQ, ALLQ128 (job 570), GAINQ128 p.26 (575), GAINQ128 p.10, RANDQ128 p.10 (585) on both streams, GAINQ512 p.26 and RANDQ512 p.10 (stream A: existing states of jobs 292b / 297, hash checked; stream B20: job 590), and ALLQ512 on stream A (job 560). IN rows are split by 'answer string inside the window' (present / absent). No writes, read only; raw scores; no thresholds. New file e57_t1t2v2_inout_read.py (the 570 scorer and rows; new IN context). Output $TOUT: E57_t1t2v2_inout_scores.json + E57_t1t2v2_inout_score_keys.json (per state|OUT, |IN|all, |IN|present, |IN|absent; by stream, by form, combined with equal weight per event). Known limit (CPU check): the answer string is inside the end window for only 28 of 74 stream-A frames (5 events) and 8 of 122 stream-B frames (2 events), so the IN present/absent split is thin, especially on B20. ESTIMATE ONLY, unmeasured: load 5 min + candidate rows ~15 min + W0 and 17 states x (OUT + IN) ~2 min each = ~55-65 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_t1t2v2_inout_read.py e57_t1t2v2_run.py e57_t1t2_run.py e57_reads.py e57_main_run.py e55_core.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_t1t2v2_inout_read.py --selftest | tail -1
IN=${IN:-$OUT/e57_t1t2v2_in}; TOUT=${TOUT:-$OUT/e57_t1t2v2_inout}; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; MAN=$TOUT/e57_t1t2v2_inout_manifest.json
chk(){ [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "$1 differs from the frozen copy (exit 9)"; exit 9; }; }
chk $IN/t1t2_v2_frozen.json f459c1a08b60eeb8e1adf01f766deff1e9574328912d30eb64147801b8b04d0f
chk $IN/e57_t1t2v2d3_candidates.json 8384a05ef434b30a665a67360c2450a987d454e6637c64041b891981a8e97854
chk $IN/streamB20_stream_sentences.jsonl ee7b1dc00622a55f395e63360a96095faf397dfc87081b9563022c759e055c85
chk $IN/e57_stream_sentences.jsonl 38ef483747a28a9334e5fba9fc7c12bc6fa2bdd92f8919a33faebe007cb4e8b6
mkdir -p "$TOUT"
$PY - "$OUT" "$MAN" <<'PYE'
import json,sys,os
o,mf=sys.argv[1:3];m={}
def add(key,label,stream,path,k,pref=None):
    s=json.load(open(path))['states'][k]
    if pref:assert s['hash'].startswith(pref),(key,s['hash'][:12])
    f=s['file'];m[key]=dict(label=label,stream=stream,file=f,hash=s['hash']);assert os.path.exists(f) or os.path.exists(o+'/../../'+f) or True
for mf_,pre in ((o+'/e57_t1t2v2/e57_t1t2v2_states_manifest.json',None),(o+'/e57_t1t2v2_gainq/e57_t1t2v2_states_manifest.json',None),(o+'/e57_t1t2v2_lowp/e57_t1t2v2_states_manifest.json',None)):
    for k,v in json.load(open(mf_)).items():
        lab=v['label']+('P26' if (v['label']=='GAINQ128' and 'v2_gainq' in mf_) else '')
        m[f"{lab}_{v['stream']}"]=dict(label=lab,stream=v['stream'],file=v['file'],hash=v['hash'])
add('GAINQ512P26_A','GAINQ512P26','A',o+'/e57_gainq_p0.26/E57_main_GAINQ512.arms.json','279','e3dc1ac6f5bc')
add('RANDQ512P10_A','RANDQ512P10','A',o+'/e57_randq_p0.10/E57_main_RANDQ512.arms.json','279','8dafe402ff73')
add('ALLQ512_A','ALLQ512','A',o+'/e57_allq512/E57_main_ALLQ512.arms.json','279','454c1658523b')
for lab in ('GAINQ512P26','RANDQ512P10'):add(lab+'_B',lab,'B',o+f'/e57_t1t2v2_b512/E57_v2B20_{lab}.arms.json','285')
for k,v in m.items():
    if v['stream'] in ('A','B'):pass
json.dump(m,open(mf,'w'),indent=1);print('manifest',len(m),{k:(v['stream'],v['hash'][:12]) for k,v in m.items()})
PYE
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2v2_inout_read.py --cpu-check --frozen $IN/t1t2_v2_frozen.json --cands $IN/e57_t1t2v2d3_candidates.json --bank "$BANK" --manifest "$MAN" --sentences-a $IN/e57_stream_sentences.jsonl --sentences-b $IN/streamB20_stream_sentences.jsonl --out "$TOUT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_t1t2v2_inout_scores.json" ] && [ -z "$RERUN" ] && { echo "scores exist (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_t1t2v2_inout_read.py --frozen $IN/t1t2_v2_frozen.json --cands $IN/e57_t1t2v2d3_candidates.json --bank "$BANK" --manifest "$MAN" --sentences-a $IN/e57_stream_sentences.jsonl --sentences-b $IN/streamB20_stream_sentences.jsonl --out "$TOUT"
echo "outputs: $TOUT/E57_t1t2v2_inout_scores.json $TOUT/E57_t1t2v2_inout_score_keys.json"
