#!/bin/bash
# JOB 580 = E57 T1/T2 v2 FORWARD-pair READS (PREPARED, NOT QUEUED, NOT APPROVED; starts only after the forward items are frozen and Hans names the job / commit): the states of job 570 (manifest e57_t1t2v2/e57_t1t2v2_states_manifest.json: NAT / NATQ / ALLQ128 on stream A and stream B20, hash checked) and W0, scored on the forward items with exactly the scoring of job 570 (OUT context of jobs 276 / 410, per-token mean primary, accuracy + rank ratio; one frame per item, group T1F-question). No writes. Needs @FROZEN_SHA@ (sha256 of the frozen forward file) and @CANDS_SHA@ (sha256 of the final forward candidate file) filled in, and FROZEN / CANDS paths. ESTIMATE ONLY: load 5 min + scorer rows (about 2,000 for 92 frames) ~8 min + 7 reads ~5 min = ~20 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e ea1125ca7f76f70994e2dfe4dc56623aa7e0d407^{commit} && git cat-file -e 2de77b0aa70006772003ef01fa576884cf2cd94e^{commit} || { echo "commit missing (exit 4)"; exit 4; }; git checkout ea1125ca7f76f70994e2dfe4dc56623aa7e0d407 -- $D/e57_t1t2v2_run.py; git checkout 2de77b0aa70006772003ef01fa576884cf2cd94e -- $D/e57_t1t2v2F_run.py; fi
sha256sum $D/e57_t1t2v2_run.py $D/e57_t1t2v2F_run.py | cut -c1-16,65-
for x in "3419b588f180 e57_t1t2_run.py" "d13a0ac9ee4a e57_main_run.py" "b19e986904ba e57_reads.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2: have $G, expected $1 (exit 5)"; exit 5; }; done
$PY $S/e57_t1t2v2F_run.py --selftest | tail -1
FROZEN=${FROZEN:?FROZEN path of the frozen forward file}; CANDS=${CANDS:?CANDS path of the final forward candidate file}; FSHA=@FROZEN_SHA@; CSHA=@CANDS_SHA@
case "$FSHA$CSHA" in *@*) echo "template: fill in the two sha256 values (exit 9)"; exit 9;; esac
[ "$(sha256sum "$FROZEN" | cut -c1-64)" = "$FSHA" ] && [ "$(sha256sum "$CANDS" | cut -c1-64)" = "$CSHA" ] || { echo "frozen forward file or candidate file differs from the pinned sha256 (exit 9)"; exit 9; }
TOUT=${TOUT:-$OUT/e57_t1t2v2F}; MAN=${MAN:-$OUT/e57_t1t2v2/e57_t1t2v2_states_manifest.json}; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; HARD=${HARD:-nominated}
[ -f "$MAN" ] && [ -f "$BANK" ] || { echo "manifest or bank missing (exit 7)"; exit 7; }
mkdir -p "$TOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2v2F_run.py --cpu-check --frozen "$FROZEN" --frozen-sha $FSHA --cands "$CANDS" --bank "$BANK" --manifest "$MAN" --out "$TOUT" --hard-source $HARD
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_t1t2v2_scores.json" ] && [ -z "$RERUN" ] && { echo "E57_t1t2v2_scores.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_t1t2v2F_run.py --stage read --frozen "$FROZEN" --frozen-sha $FSHA --cands "$CANDS" --cands-sha $CSHA --bank "$BANK" --manifest "$MAN" --out "$TOUT" --hard-source $HARD
echo "outputs: $TOUT/E57_t1t2v2_scores.json $TOUT/E57_t1t2v2_score_keys.json (copy to the shared root as e57_t1t2v2F_scores.json / e57_t1t2v2F_score_keys.json)"
