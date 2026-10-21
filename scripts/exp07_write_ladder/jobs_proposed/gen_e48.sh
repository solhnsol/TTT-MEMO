#!/bin/bash
# generator (not a job): writes 129_e48_world_table.sh (CPU only), 130-134 (single-cell jobs) and 135 (bundle oracle + k6p + k3all in one process) (E48 frame-disjoint world, GPU); code pinned to the commit H (local checkout, no fetch)
cd "$(dirname "$0")/.."
H=fe966694cfd0f801a35efb726056bfbc1610ac33
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
FILES="$D/s3.py $D/s3_sets.py $D/e43_sets_test.py $D/e44_sets_test.py $D/e46_sets_test.py $D/e47_sets_test.py $D/e42_k2_tokens.py $D/k3_counts.py $D/suppress.py $D/ladder.py $D/margins_split.py $D/margins_split_test.py $D/e48_world_test.py $D/e48_world_stats.py $D4/gen_world.py $D4/frame_bank.py"
FL="";MAN="";for F in $FILES;do FL="$FL$F ";MAN="$MAN$(git show $H:$F | sha256sum | cut -c1-12) $F
";done
LAD=$(git show $H:$D/ladder.py | sha256sum | cut -c1-12)
GW=$(git show $H:$D4/gen_world.py | sha256sum | cut -c1-12)
head() { # num name time-comment
cat <<JOB
#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): $3
# E48 frame-disjoint world (PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread). World seed 101. E48_MODE=pair (default) = W1: every pair has its own R = 16 sentence frames, no frame shared between pairs (the two persons of a pair share the pair's frames); E48_MODE=sentence = W2: every exposure sentence has its own frame. Only the exposure sentence templates change (names, places, occupations, schedule, background order are the same rng stream; the stream positions shift a little because the sentence lengths differ); gen_world.py reads E48_FRAMES (this job exports it) and defaults to the old shared frames, bit-identical (e48_world_test.py --old). Result tags carry the prefix w1_ / w2_ so nothing of the old worlds is overwritten.
# PINNED INSTALL, NO FETCH: step 0 checks that the lab ladder.py is the lab version b9fd20049420 or the patched file and that the lab gen_world.py is the version the E48 patch was made on ($GW is the patched file, base 1514a13e67c7), then checks the code out of the local repository at commit \$COMMIT (exit 4 if absent: fetch by hand; exit 6 if a lab file is neither base nor patched); step 1 verifies sha256 (first 12 hex) against the manifest (exit 5). NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. suppress.py is VERIFIED ONLY (lab copy used).
# CPU-only dry run (separate directory): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e48 FETCH_HEAD && ln -s "\$(pwd)/data" /tmp/dry_e48/data && cd /tmp/dry_e48 && PY=python3 S=scripts/exp07_write_ladder CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/$1_e48_$2.sh   (afterwards: git worktree remove --force /tmp/dry_e48)
set -e
cd "\$(dirname "\$0")/../../.." 2>/dev/null || true
D=$D
D4=$D4
COMMIT=$H
FILES="$FL"
LADDER_OK="b9fd20049420 $LAD"
GW_OK="1514a13e67c7 $GW"
E48_MODE="\${E48_MODE:-pair}"
case "\$E48_MODE" in pair) TP=w1;; sentence) TP=w2;; *) echo "E48_MODE must be pair or sentence"; exit 2;; esac
if [ -z "\$NO_INSTALL" ]; then
  for X in "\$D/ladder.py:\$LADDER_OK" "\$D4/gen_world.py:\$GW_OK"; do
    GL=\$(sha256sum \${X%%:*} | cut -c1-12); case " \${X#*:} " in *" \$GL "*) ;; *) echo "lab file \${X%%:*} (\$GL) is neither the lab version the patch was made on nor the patched file: not overwritten (exit 6), the code thread must rebase"; exit 6;; esac
  done
  git cat-file -e \$COMMIT^{commit} || { echo "commit \$COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }
  git checkout \$COMMIT -- \$(for F in \$FILES; do [ "\$F" = "\$D/suppress.py" ] || echo \$F; done)
fi
if [ -z "\$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "\$HH" ] || continue
    G=\$(sha256sum \$F | cut -c1-12)
    if [ "\$F" = "\$D/suppress.py" ]; then [ "\$G" = "\$HH" ] || echo "WARNING suppress.py differs from the repo copy: have \$G, repo \$HH (the lab copy is used)"; continue; fi
    [ "\$G" = "\$HH" ] || { echo "VERSION MISMATCH \$F: have \$G, expected \$HH"; exit 5; }
  done <<'MANIFEST'
$MAN
MANIFEST
  echo "installed code verified against the manifest"
fi
export E48_FRAMES="\$E48_MODE"
# CPU (the same lines in the dry run): unit tests of the old sets and of the new world (frame disjointness, slots, one exposure per triple and round, frame field of the sentence table; the old-code identity test needs --old and is run by the code thread)
\$PY \$S/e43_sets_test.py
\$PY \$S/e44_sets_test.py
\$PY \$S/e47_sets_test.py
\$PY \$S/margins_split_test.py
\$PY \$S/e48_world_test.py
JOB
}
# ---- CPU table job
head 129 world_table "CPU only, about 5 min, no GPU, no approval needed (frame-disjoint world size / structure table)" > jobs_proposed/129_e48_world_table.sh
cat >> jobs_proposed/129_e48_world_table.sh <<JOB
unset E48_FRAMES
\$PY \$S/e48_world_stats.py --seeds 101 --passes 4 --out-dir \${E48_STATS_DIR:-/tmp/e48_world_stats}
JOB
gen() { # num name time cmd-lines
  head $1 $2 "$3" > jobs_proposed/$1_e48_$2.sh
  cat >> jobs_proposed/$1_e48_$2.sh <<JOB
[ -n "\$CPU_ONLY" ] && exit 0
$4
JOB
}
gen 130 native "about 50-60 min GPU [design thread estimate from the old L0fs_main: stream + eval of 4 passes, worlds A and B; not measured for this world]" 'for M in save eval; do
  $PY $S/ladder.py --mode $M --level L0fs --world-seed 101 --passes 4 --worlds AB --keep --tag ${TP}main
done'
for c in "131 oracle oracle 35-40" "132 k6p k6p 40" "133 k3all k3all 42" "134 sent sent 50-55"; do set -- $c
  gen $1 $2 "about $4 min GPU [design thread estimate from the old runs of the same cell: K3-all measured eval 5.5 min + save, SENT 50-55 min; worlds A and B, 4 passes, --keep]" "for M in save eval; do
  \$PY \$S/s3.py --cell $3 --world-seed 101 --passes 4 --worlds AB --keep --tag \${TP}_$3 --mode \$M
done"
done
# bundle: several cells in ONE s3.py process per mode (one model load / setup per call); per-cell result files as for single jobs; a failing cell does not delete the finished ones
head 135 bundle_oracle_k6p_k3all "about 75 min GPU [estimate from the lab timing of single-cell jobs: save call 8-10 min overhead + ~14 min training A+B per cell, eval call ~7.5 min overhead + ~5.5 min per cell: 9 + 3 x 14 + 7.5 + 3 x 5.5 = about 75 min instead of about 3 x 45; NOT measured for the bundle]; SMOKE=1 runs only a 2-cell smoke test of the bundling (canary only, 2 chunks, tags smoke_*) first" > jobs_proposed/135_e48_bundle_oracle_k6p_k3all.sh
cat >> jobs_proposed/135_e48_bundle_oracle_k6p_k3all.sh <<JOB
if [ -n "\$SMOKE" ]; then
  # untested-on-GPU bundling code: tiny smoke first (2 cells, canary only, 2 chunks), results tagged smoke_*; nothing of the real cells is touched
  \$PY \$S/s3.py --cells oracle,k3all --tags smoke_b1,smoke_b2 --world-seed 101 --passes 1 --worlds A --canary-only --smoke-chunks 2 --mode save
  exit 0
fi
[ -n "\$CPU_ONLY" ] && exit 0
for M in save eval; do
  \$PY \$S/s3.py --cells oracle,k6p,k3all --tags \${TP}_oracle,\${TP}_k6p,\${TP}_k3all --world-seed 101 --passes 4 --worlds AB --keep --mode \$M
done
JOB

