#!/bin/bash
# E57 v4 CPU freeze (design section 11.3; Hans approved the CPU preparation 2026-10-23 09:29Z, "다 승인할게 진행해봐"). CPU only, NO GPU, NO model, NO download. Reuses titles.pkl, clusters.pkl (and analyze_cache_f1.pkl of job 206 if present, else the text stage is redone once) and the shards in $RAW of the job-202 run (exit 7 if missing); fetch is NOT called.
# What it does: (1) draws 30 news events from the 568-class pool (events with >= 5 lead targets) by a seeded permutation (FREEZE_SEED 57101) with an automatic ethics filter ETH-v1 applied after the draw (the same permutation continues until 30 events are accepted; rejected events and reasons listed); (2) freezes the lead texts (articles.jsonl: publisher copyright, stays on the server), the fixed-order cyclic round as token ids (round_ids.npy, padded with blank-line tokens to whole 1024-token chunks) and the target / question bank with candidate tiers (strict +-1, middle c' >= 2 and |c-c'| <= 2, relaxed) and the confirmation subset (c in {3,4} with >= 10 middle candidates), with sha256 of every file (freeze_manifest.json); (3) records G1 (n, n_eff, theta, power for the main bank and the confirmation subset), G2 (rho_win of the frozen layout: CLUSTER vs fixed SPREAD, decomposed) and G3 (RoPE position check in float32 up to 3.4M positions, and the periodic K,V scheme of the GPU job).
# Output dir $OUT/e57_freeze/ ; report e57_freeze/e57_freeze_report.md. Exit 8 if the report exists unless RERUN=1 (RERUN=1 also redoes the text stage). PROXIES as in job 202 (regex facts, title-token clusters, candidate exposures c' over whole articles while c is the lead count). The sequence length of the RoPE table is assumed to be 131072 (SEQ_LEN env to override); the GPU job prints and asserts the real value.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. 0.2-0.6 h with the 206 cache (one detail text stage over about 40-60 drawn events: each shard is read once; tokenisation, bank and layouts take minutes); 0.5-1.5 h if the text stage has to be redone (equals the measure stage of job 202: use the lab log of 202 for the real number). Memory as the measure stage of job 202.
# Needs: lab python with pyarrow, numpy, tokenizers (exit 3 if missing), the repo tokenizer data/tokenizer/llama3/tokenizer.json. Writes only under $OUT.
# PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). Exit 8 if the report exists unless RERUN=1.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=849385ed7616e9781d667db9e91dba06149a232d
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py $D/e57/e57_rope.py $D/e57/e57_freeze.py"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}/e57_news}"; mkdir -p "$OUT"; : "${RAW:=$OUT/news_raw}"; : "${WORKERS:=16}"; : "${MAXCL:=3000}"
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
f09dcb7e69eb e56/e56_text.py
a0590cd53539 e57/e57_common.py
76afc723ea81 e57/e57_news.py
9bef0d97a08a e57/e57_news96.py
a39189e927f7 e57/e57_news10.py
29fd1f69fb0d e57/e57_rope.py
de41671acd28 e57/e57_freeze.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_freeze/e57_freeze_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl clusters.pkl; do [ -f "$OUT/$F" ] || { echo "$OUT/$F missing: the job-202 run must have finished the titles and cluster stages (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_news96.py --selftest
$PY $D/e57/e57_rope.py --selftest
$PY $D/e57/e57_freeze.py --selftest
$PY $D/e57/e57_freeze.py --out "$OUT" --raw "$RAW" --workers $WORKERS --max-clusters $MAXCL --seq-len ${SEQ_LEN:-131072} ${RERUN:+--rerun}
tail -120 "$REP"
