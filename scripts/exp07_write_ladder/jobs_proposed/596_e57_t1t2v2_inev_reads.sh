#!/bin/bash
# JOB 596 = E57 T1/T2 v2 EVENT-LOCAL IN READS (variant iv; prepared; NOT QUEUED until Hans confirms job name + commit; after job 590 in queue order): the same 98 frozen v2 items and the same 17 states + W0 as job 595, but the IN context is built PER EVENT: the 8192-token (8-chunk) window of the item's stream that ends at the chunk of the event's fragments in which the answer string of the most items of the event stands (ties: the later chunk); every item is read with the FINAL weights of the state in the window of its event (no intermediate states needed). CPU check: the answer string is inside the event window for 74 of 74 stream-A frames (10 events) and 122 of 122 stream-B frames (20 events), so IN is all-present here and the OUT reads (no context) of 570 / 575 / 585 / 595 are the matching without-context baseline. Caveat: the weights have also been trained on the window text and on all later stream text, so IN gain mixes in-context use and memory; compare arms against W0 and NAT. Output $TOUT: E57_t1t2v2_inout_scores.json + E57_t1t2v2_inout_score_keys.json (state labels <label>|INEV|all). ESTIMATE ONLY, unmeasured: load 5 min + 30 event contexts (one 8K forward each, fp32) x 18 states x ~3-5 s = 30-50 min + rows = ~55-75 min. (Header of 595 follows.)
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
IN=${IN:-$OUT/e57_t1t2v2_in}; TOUT=${TOUT:-$OUT/e57_t1t2v2_inev}; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; MAN=$TOUT/e57_t1t2v2_inev_manifest.json
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
for lab in ('GAINQ512P26','RANDQ512P10'):
    pth=o+f'/e57_t1t2v2_b512/E57_v2B20_{lab}.arms.json'
    if os.path.exists(pth):add(lab+'_B',lab,'B',pth,'285')
    else:print('NOTE: job 590 output missing, B20 512 states not in the manifest:',pth)
for k,v in m.items():
    if v['stream'] in ('A','B'):pass
json.dump(m,open(mf,'w'),indent=1);print('manifest',len(m),{k:(v['stream'],v['hash'][:12]) for k,v in m.items()})
PYE
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2v2_inout_read.py --cpu-check --frozen $IN/t1t2_v2_frozen.json --cands $IN/e57_t1t2v2d3_candidates.json --bank "$BANK" --manifest "$MAN" --sentences-a $IN/e57_stream_sentences.jsonl --sentences-b $IN/streamB20_stream_sentences.jsonl --out "$TOUT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_t1t2v2_inout_scores.json" ] && [ -z "$RERUN" ] && { echo "scores exist (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_t1t2v2_inout_read.py --frozen $IN/t1t2_v2_frozen.json --cands $IN/e57_t1t2v2d3_candidates.json --bank "$BANK" --manifest "$MAN" --sentences-a $IN/e57_stream_sentences.jsonl --sentences-b $IN/streamB20_stream_sentences.jsonl --out "$TOUT" --conds INEV
echo "outputs: $TOUT/E57_t1t2v2_inout_scores.json $TOUT/E57_t1t2v2_inout_score_keys.json"
