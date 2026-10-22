"""Generates the RunPod decomposition jobs 170-173 (E53 W1 gain-ladder decomposition). Usage: gen_rp_decomp.py COMMIT   (run from the repo root at that commit; writes the four job files next to this script)"""
import sys,hashlib,subprocess
from pathlib import Path
C=sys.argv[1];HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
D='scripts/exp07_write_ladder';D4='scripts/exp04_what_ttt_retains'
TRAIN=[f'{D}/{x}' for x in 's3.py s3_sets.py e43_sets_test.py e44_sets_test.py e46_sets_test.py e47_sets_test.py e42_k2_tokens.py k3_counts.py suppress.py ladder.py margins_split.py margins_split_test.py e48_world_test.py e48_world_stats.py e47_native_da.py rp_native_check.py rp_unifg_check.py'.split()]+[f'{D4}/gen_world.py',f'{D4}/frame_bank.py']
E49=[f'{D}/e49_stats.py',f'{D}/e49_candidates.py',f'{D}/e49_test.py',f'{D}/e48_world_test.py',f'{D}/ladder.py',f'{D}/suppress.py',f'{D}/margins_split.py',f'{D4}/gen_world.py',f'{D4}/frame_bank.py']
NLL=[f'{D}/e37_firstchunk_nll.py',f'{D}/e37_nll_table.py',f'{D}/ladder.py',f'{D}/suppress.py',f'{D}/margins_split.py',f'{D4}/gen_world.py',f'{D4}/frame_bank.py',f'{D}/e47_native_da.py']
def sh(f):
    return subprocess.check_output(['git','show',f'{C}:{f}'],cwd=ROOT).__len__() and hashlib.sha256(subprocess.check_output(['git','show',f'{C}:{f}'],cwd=ROOT)).hexdigest()[:12]
def pre(files,need_gb,job):
    man='\n'.join(f'{sh(f)} {f}' for f in files)
    return f'''set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D={D}
D4={D4}
COMMIT={C}
FILES="{' '.join(files)}"
: "${{PY:=python3}}"; : "${{S:=$D}}"; : "${{OUT:=${{EXP07_OUT:-experiments/exp07_write_ladder}}}}"; export EXP07_OUT="$OUT"
JOB={job}
RPE="${{RP_ENV:-/workspace/rp_env.sh}}"; [ -f "$RPE" ] && . "$RPE"   # written by jobs_proposed/rp_setup.sh on the pod (PY, OUT, XLA_FLAGS, LD_LIBRARY_PATH); on the lab server this file does not exist and the runner environment is used
# RunPod: every file of the pinned commit is checked out (the pod holds a repo clone, not the lab-patched copies); NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the manifest check.
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{{commit}} || {{ echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }}
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $F | cut -c1-12)
    [ "$G" = "$HH" ] || {{ echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }}
  done <<'MANIFEST'
{man}

MANIFEST
  echo "installed code verified against the manifest"
fi
export E48_FRAMES=pair
# environment checks (no GPU work): deterministic ops flag, GPU visible, checkpoint, tokenizer, free disk
case "$XLA_FLAGS" in *deterministic_ops=true*) ;; *) echo "XLA_FLAGS must contain --xla_gpu_deterministic_ops=true (ladder.py asserts it): exit 3"; exit 3;; esac
if [ -z "$CPU_ONLY" ]; then nvidia-smi -L || {{ echo "no GPU visible: exit 3"; exit 3; }}; fi
[ -d checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 ] || {{ echo "checkpoint checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 (the 3B TTT-E2E 128K Books checkpoint, step 1949, about 11 GB; baseline_config.CHECKPOINT_DIR) missing: exit 3"; exit 3; }}
[ -f data/tokenizer/llama3/tokenizer.json ] || {{ echo "tokenizer data/tokenizer/llama3/tokenizer.json missing: exit 3"; exit 3; }}
mkdir -p "$OUT/states"
AV=$(df -P --block-size=1G "$OUT" | awk 'NR==2{{print $4}}'); echo "free disk at $OUT: $AV GB (need >= {need_gb})"
[ "$AV" -ge {need_gb} ] || {{ echo "not enough free disk (exit 3 outside CPU_ONLY)"; [ -n "$CPU_ONLY" ] || exit 3; }}
'''
def post(job,tags):
    return f'''
# provenance: pod environment and sha256 (first 64 hex) of every result file this job wrote (states: sizes + sha256 of the pass-4 states)
{{ echo "job $JOB commit $COMMIT"; date -u; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; $PY -c "import jax;print('jax',jax.__version__,jax.devices())"; echo "XLA_FLAGS=$XLA_FLAGS"; nvidia-smi | head -4; (nvcc --version 2>/dev/null | tail -2) || true; echo "--- pip freeze"; ($PY -m pip freeze 2>/dev/null || uv pip freeze --python $PY 2>/dev/null) || true; echo "--- masked tokens per saved log (world pass n_masked_total)"; for F in "$OUT"/*_s101_[AB].saved.json; do [ "$F" -nt "$MARK" ] && $PY -c "import json,sys;L=json.load(open(sys.argv[1]))['log'];print(sys.argv[1].split('/')[-1],[(e.get('epoch'),e.get('n_masked_total')) for e in L])" "$F"; done; }} > "$OUT/rp_env_$JOB.txt" 2>&1 || true
(cd "$OUT" && find . -maxdepth 1 -type f -newer "$MARK" ! -name 'rp_sha_*' -exec sha256sum {{}} +) > "$OUT/rp_sha_$JOB.txt"
for T in {tags}; do for W in A B; do F="$OUT/states/${{T}}_s101/$W/state_004.npz"; [ -f "$F" ] && {{ ls -l "$F"; sha256sum "$F"; }}; done; done >> "$OUT/rp_sha_$JOB.txt" 2>&1 || true
echo "provenance: $OUT/rp_env_$JOB.txt, $OUT/rp_sha_$JOB.txt"
'''
def hdr(t,body):return f'#!/bin/bash\n# {t}\n'+body
COMMON_DRY='CPU-only dry run (checks the commit, the manifest, the CPU unit tests, the environment and the output names; no GPU, no model): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_rp FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_rp/data && ln -s "$(pwd)/checkpoints" /tmp/dry_rp/checkpoints && cd /tmp/dry_rp && PY=python3 CPU_ONLY=1 NO_INSTALL=1 bash scripts/exp07_write_ladder/jobs_proposed/{job}.sh   (then: git worktree remove --force /tmp/dry_rp; with NO_INSTALL=1 the checkout step is skipped because the worktree is already at the pinned commit; the manifest check still runs)'
TESTS='$PY $S/e43_sets_test.py\n$PY $S/e44_sets_test.py\n$PY $S/e47_sets_test.py\n$PY $S/margins_split_test.py\n$PY $S/e48_world_test.py\n$PY $S/rp_native_check.py --selftest | tail -1\n$PY $S/rp_unifg_check.py --selftest | tail -1\n'
def w(name,txt):(HERE/name).write_text(txt);(HERE/name).chmod(0o755);print(name,len(txt))
# ---- 170 native reproduction check
job='170_rp_w1_native_check'
w(job+'.sh',hdr('EXPECTED RUN TIME (Hans confirms before it runs): about 40 min on the RunPod A100 (save call about 9 min overhead + about 14 min training worlds A+B, eval call about 7.5 min overhead + about 5.5 min; basis: design thread E53 timing and the lab job 130 estimate 50-60 min; NOT measured on the pod), with slack about 60 min (cold JAX compile cache and first data read on a new volume).\n# RunPod reproduction check: re-runs the W1 native cell (job 130 definition: level L0fs, world seed 101, 4 passes, worlds A and B, --keep) with the NEW tag w1rp (results L0fs_w1rp_s101_*, nothing of the lab files w1main is touched or needed) and compares D_alt per pass 1-4 with the lab values (job 130: 0.017 / 0.091 / 0.177 / 0.194). Verdict PASS / FAIL (|difference| <= RP_TOL at every pass, default 0.05, the value Hans agreed in E53 section 11) is written to $OUT/rp_native_check_PASS.txt; job 171 refuses to start without PASS (FORCE=1 overrides). If the lab result files L0fs_w1main_s101_{A,B}.json + meta are also in $OUT, the pair-level max abs difference is reported too.\n# NO OVERWRITE: exit 8 if L0fs_w1rp_s101_A.json exists.\n# '+COMMON_DRY.format(job=job)+'\n',
 pre(TRAIN,40,job)+'MARK=$(mktemp)\n'+TESTS.replace('\\n','\n')+'''[ -f "$OUT/L0fs_w1rp_s101_A.json" ] && { echo "L0fs_w1rp_s101_A.json exists: not overwritten (exit 8)"; exit 8; }
[ -n "$CPU_ONLY" ] && { echo "would write L0fs_w1rp_s101_{A,B}.json, rp_native_check_PASS.txt"; exit 0; }
CK=checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc
[ -f "$OUT/rp_ckpt_sha256.txt" ] || (cd $CK && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$OUT/rp_ckpt_sha256.txt"
echo "checkpoint file list sha256 -> $OUT/rp_ckpt_sha256.txt ($(wc -l < $OUT/rp_ckpt_sha256.txt) files); compare with the same command on the lab server: sha256sum of that file = $(sha256sum $OUT/rp_ckpt_sha256.txt | cut -c1-16)"
for M in save eval; do
  $PY $S/ladder.py --mode $M --level L0fs --world-seed 101 --passes 4 --worlds AB --keep --tag w1rp
done
CELLS=L0fs:w1rp; REF="--ref-values 0.017,0.091,0.177,0.194"
if [ -f "$OUT/L0fs_w1main_s101_A.json" ] && [ -f "$OUT/meta_L0fs_w1main_s101.json" ]; then CELLS=L0fs:w1rp,L0fs:w1main; REF="--ref L0fs:w1main"; fi
$PY $S/e47_native_da.py --seed 101 --cells $CELLS --out-dir "$OUT/rp_check"
set +e
$PY $S/rp_native_check.py --new L0fs:w1rp $REF --tol ${RP_TOL:-0.05} --marker "$OUT/rp_native_check_PASS.txt"
echo "native check exit code $? (0 PASS, 1 FAIL): $(cat $OUT/rp_native_check_PASS.txt)"
set -e'''+post(job,'L0fs_w1rp')))
# ---- 171 BG + NOO
job='171_rp_w1_bg_noo'
w(job+'.sh',hdr('EXPECTED RUN TIME (Hans confirms before it runs): about 2 h on the RunPod A100, with slack about 2.6 h (design thread E53 section 8: 2 cells x about 53 min + about 20 min overhead; one call = one model load per mode; NOT measured for the bundle on the pod).\n# E53 decomposition on W1 (Hans approved the RunPod run): BG (O + all non-O positions of the window except the exposure sentences B/P/X: the background is masked, the other exposure sentences stay visible; BOS masked) and NOO (only O visible, everything else masked), W1 world seed 101, 4 passes, worlds A and B, --keep (states of passes 0-4 kept, about 13 GB per cell and world), tags w1_bg, w1_noo (results S3_w1_bg_s101_*, S3_w1_noo_s101_*; no lab file has these names). Definitions are those of s3_sets.py (CPU tested by e44_sets_test, mask union validated by e53_gap_dist).\n# Starts only after job 170 wrote PASS (exit 9 otherwise; FORCE=1 overrides). NO OVERWRITE: exit 8 if S3_w1_bg_s101_A.saved.json or S3_w1_noo_s101_A.saved.json exists. Job 172 (UNIF-G) reads S3_w1_bg_s101_{A,B}.saved.json, so it must run on the SAME volume after this job.\n# '+COMMON_DRY.format(job=job)+'\n',
 pre(TRAIN,70,job)+'MARK=$(mktemp)\n'+TESTS.replace('\\n','\n')+'''if [ -z "$FORCE" ] && ! grep -q '^PASS' "$OUT/rp_native_check_PASS.txt" 2>/dev/null; then echo "no PASS from job 170 (rp_native_check_PASS.txt): exit 9 (FORCE=1 overrides)"; exit 9; fi
for T in w1_bg w1_noo; do [ -f "$OUT/S3_${T}_s101_A.saved.json" ] && { echo "S3_${T}_s101_A.saved.json exists: not overwritten (exit 8)"; exit 8; }; done
[ -n "$CPU_ONLY" ] && { echo "would write S3_w1_bg_s101_*, S3_w1_noo_s101_* (+ states)"; exit 0; }
for M in save eval; do
  $PY $S/s3.py --cells bg,noo --tags w1_bg,w1_noo --world-seed 101 --passes 4 --worlds AB --keep --mode $M
done
$PY $S/e47_native_da.py --seed 101 --cells S3:w1_bg,S3:w1_noo --out-dir "$OUT/rp_bg_noo"'''+post(job,'S3_w1_bg S3_w1_noo')))
# ---- 172 UNIF-G
job='172_rp_w1_unifg'
w(job+'.sh',hdr('EXPECTED RUN TIME (Hans confirms before it runs): about 75 min on the RunPod A100, with slack about 1.6 h (design thread E53 section 8: 53 min + about 20 min overhead; NOT measured on the pod).\n# E53 UNIF-G on W1: O + uniformly random positions, the number per chunk equal to the per-chunk number of masked tokens of BG (s3.py --cell oraclefill --fill-size-from w1_bg --fill-no-o: O-free chunks are filled too, because BG masks the background there; the total O + fill per chunk is the n_masked of BG, asserted after the run by rp_unifg_check.py; job 88 did this in the old world without --fill-no-o). Tag w1_unifg (results S3_w1_unifg_s101_*). Reads $OUT/S3_w1_bg_s101_{A,B}.saved.json written by job 171: it must run on the same volume AFTER job 171 (exit 3 if missing). Known limit (E53 section 8): the random positions may include exposure-sentence tokens and BOS.\n# NO OVERWRITE: exit 8 if S3_w1_unifg_s101_A.saved.json exists.\n# '+COMMON_DRY.format(job=job)+'\n',
 pre(TRAIN,40,job)+'MARK=$(mktemp)\n'+TESTS.replace('\\n','\n')+'''for W in A B; do [ -f "$OUT/S3_w1_bg_s101_$W.saved.json" ] || { echo "$OUT/S3_w1_bg_s101_$W.saved.json missing (job 171 must run first on this volume): exit 3"; exit 3; }; done
[ -f "$OUT/S3_w1_unifg_s101_A.saved.json" ] && { echo "S3_w1_unifg_s101_A.saved.json exists: not overwritten (exit 8)"; exit 8; }
[ -n "$CPU_ONLY" ] && { echo "would write S3_w1_unifg_s101_* (+ states)"; exit 0; }
for M in save eval; do
  $PY $S/s3.py --cell oraclefill --fill-size-from w1_bg --fill-no-o --world-seed 101 --passes 4 --worlds AB --keep --tag w1_unifg --mode $M
done
$PY $S/e47_native_da.py --seed 101 --cells S3:w1_unifg --out-dir "$OUT/rp_unifg"
# size assertion: the TOTAL masked tokens (O + fill) per world, pass and chunk equal BG's n_masked
$PY $S/rp_unifg_check.py --bg w1_bg --unifg w1_unifg | tee "$OUT/rp_unifg_check.txt"'''+post(job,'S3_w1_unifg')))
# ---- 173 DCLM first-chunk read
job='173_rp_w1_dclm_nll'
w(job+'.sh',hdr('EXPECTED RUN TIME (Hans confirms before it runs): about 25-30 min on the RunPod A100, with slack about 40 min (same basis as job 160_w1_nll_dclm: 8 states + W0 + W0 repeat = 10 reads of 200 samples at about 1.1-2.3 min plus about 5 min load; NOT measured). NO training, NO weights written.\n# DCLM first-chunk damage read (E37c method) of the PASS-4 states of the RunPod decomposition cells, worlds A and B: re-run native (L0fs_w1rp), BG (S3_w1_bg), NOO (S3_w1_noo), UNIF-G (S3_w1_unifg), plus W0 and a W0 repeat (max |diff| must be 0). It must run BEFORE any state of these cells is deleted or the pod is terminated; run it right after job 172 on the same volume. Needs data/dclm_val_subset/val_c0_200x8193.u32le.bin on the pod. Output (tag e37w1dec): $OUT/e37_firstchunk_e37w1dec.{json,csv} and the table $OUT/e37_w1dec_nll_table.md. NO OVERWRITE: exit 8 if the json exists. Cells whose pass-4 states are absent are listed and the job stops (exit 3).\n# '+COMMON_DRY.format(job=job)+'\n',
 pre(NLL,10,job)+'''MARK=$(mktemp)
IDS=data/dclm_val_subset/val_c0_200x8193.u32le.bin
[ -f "$IDS" ] || { echo "$IDS missing: exit 3"; exit 3; }
ARGS=""
for C in w1rp_native=L0fs_w1rp w1_bg=S3_w1_bg w1_noo=S3_w1_noo w1_unifg=S3_w1_unifg; do
  N=${C%%=*}; P=${C#*=}
  for W in A B; do
    F=$OUT/states/${P}_s101/$W/state_004.npz; [ -f "$F" ] || { echo "missing $F (exit 3)"; exit 3; }
    ARGS="$ARGS --state ${N}_p4_${W}=$F"
  done
done
$PY $S/e37_nll_table.py --selftest | tail -1
[ -f "$OUT/e37_firstchunk_e37w1dec.json" ] && { echo "$OUT/e37_firstchunk_e37w1dec.json exists: not overwritten (exit 8)"; exit 8; }
[ -n "$CPU_ONLY" ] && { echo "would read: $ARGS"; exit 0; }
$PY $S/e37_firstchunk_nll.py --ids $IDS --seed 101 --tag e37w1dec $ARGS
$PY $S/e37_nll_table.py --json $OUT/e37_firstchunk_e37w1dec.json --out $OUT/e37_w1dec_nll_table.md'''+post(job,'')))
# ---- 174 30-candidate readout, passes 1-4
job='174_rp_w1_candidates'
H174=('EXPECTED RUN TIME (Hans confirms before it runs): about 40-55 min on the RunPod A100, with slack about 70 min (ESTIMATE, not measured: job 160 W1 readout basis, about 1 min per state read; here 4 cells x 4 passes x worlds A,B = 32 reads + W0 + about 5 min load + 2-3 min building rows; one call, one model load). NO training, no state written or removed.\n'
 '# E49 30-candidate readout (same definition as job 160_w1_e49_candidates_5cells: all 30 places scored as the continuation of each L3 item, packed fp32 R1 and R1-alt tails) on EVERY pass 1-4 of the RunPod decomposition cells: re-run native (L0fs_w1rp), BG (S3_w1_bg), NOO (S3_w1_noo), UNIF-G (S3_w1_unifg), worlds A and B, plus W0; result names e49_cand_s101_<cell>_p<pass>.json, summary e49_candidates_s101_w1dec.json (tag _w1dec, W0 name w1dec_W0; none of them is a lab file name). Reference cell for the change table: w1rp_native. Must run BEFORE the states are deleted or the pod is terminated, after job 172 on the same volume. NO OVERWRITE: exit 8 if the summary exists unless RERUN=1; --strict aborts on a missing state.\n# ')
B174=r'''MARK=$(mktemp)
$PY $S/e49_stats.py | tail -1
$PY $S/e49_test.py | tail -1
$PY $S/e48_world_test.py | tail -1
CARGS=""; for C in w1rp_native=L0fs_w1rp w1_bg=S3_w1_bg w1_noo=S3_w1_noo w1_unifg=S3_w1_unifg; do
  N=${C%%=*}; P=${C#*=}
  echo "$N ($P): states A $(ls $OUT/states/${P}_s101/A 2>/dev/null | wc -l), B $(ls $OUT/states/${P}_s101/B 2>/dev/null | wc -l), saved log A: $([ -f $OUT/${P}_s101_A.saved.json ] && echo yes || echo NO)"
  CARGS="$CARGS --cell $C"
done
[ -n "$CPU_ONLY" ] && exit 0
[ -f $OUT/e49_candidates_s101_w1dec.json ] && [ -z "$RERUN" ] && { echo "summary e49_candidates_s101_w1dec.json exists: not overwritten (exit 8)"; exit 8; }
$PY $S/e49_candidates.py --seed 101 --epochs 1,2,3,4 --w0 --w0-name w1dec_W0 --ref w1rp_native --tag _w1dec --strict $CARGS'''
w(job+'.sh',hdr(H174+COMMON_DRY.format(job=job)+'\n',pre(E49,10,job)+B174+post(job,'')))
