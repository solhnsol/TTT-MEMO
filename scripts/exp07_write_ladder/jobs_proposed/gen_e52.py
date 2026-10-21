#!/usr/bin/env python3
"""generator (not a job): writes 160_e52_L2a_native.sh, 161_e52_L2b_native.sh, 162_e52_L1.sh (E52 Level-2 controlled insertion, GPU, PROPOSED, NOT QUEUED). The code is pinned to --commit (checked out from the local clone, no fetch), the two world files are pinned by sha256.
Usage: gen_e52.py --commit <40-hex> --sha-l2a <sha256> --sha-l2b <sha256>   (CPU only)"""
import argparse,os,re
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);ap.add_argument('--sha-l2a',required=True);ap.add_argument('--sha-l2b',required=True);ap.add_argument('--nat-dir',default='data/natural-text/worlds/e52',help='folder holding nat_9201_L2a.json / nat_9201_L2b.json (default for $NAT_DIR in the jobs; the files are read from there, never from the repo)');a=ap.parse_args()
assert re.fullmatch(r'[0-9a-f]{40}',a.commit) and all(re.fullmatch(r'[0-9a-f]{64}',x) for x in (a.sha_l2a,a.sha_l2b)),'full 40-hex commit and 64-hex sha256 values required'
D='scripts/exp07_write_ladder'
FILES=' '.join(f'{D}/{f}' for f in ('nat_world.py','nat_filler.py','nat_pass.py','nat_ladder.py','nat_analyze.py','nat_dryrun.py','nat_signal.py','e51_nat_table.py','e52_table.py','e52_analyze.py'))
COMMON=f'''set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
NAT_DIR=${{NAT_DIR:-{a.nat_dir}}}
mkdir -p $OUT
TOK=${{TOKENIZER_JSON:-data/tokenizer/llama3/tokenizer.json}}
COMMIT={a.commit}
FILES="{FILES}"
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{{commit}} || {{ echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }}
  git checkout $COMMIT -- $FILES
  git diff --quiet $COMMIT -- $FILES || {{ echo "checked-out files differ from commit $COMMIT"; exit 4; }}
fi
sha() {{ [ "$(sha256sum "$NAT_DIR/$1" | cut -d' ' -f1)" = "$2" ] || {{ echo "world file $1 does not match the pinned sha256 $2: not run (exit 5)"; exit 5; }}; }}
sha nat_9201_L2a.json {a.sha_l2a}
sha nat_9201_L2b.json {a.sha_l2b}
'''
CPU=lambda lay:f'''$PY $S/nat_filler.py --selftest
$PY $S/nat_pass.py --selftest
$PY $S/e52_table.py --nat-dir $NAT_DIR --stream 9201 --require-real --tokenizer-file $TOK --out $OUT/e52_person_table_9201.tsv   # exit 1 on any person-level rule violation; no GPU
$PY $S/nat_dryrun.py --nat-dir $NAT_DIR --stream 9201 --layout {lay} --passes 2 --require-real --tokenizer-file $TOK   # whole pass-2 filler assembly with the real tokenizer: drift <= 2, strata equal, no pool paragraph twice, audit; exit 3 on failure
$PY $S/nat_filler.py --nat-dir $NAT_DIR --stream 9201 --layout {lay} --passes 2 --budget-only
$PY $S/nat_filler.py --nat-dir $NAT_DIR --stream 9201 --layout {lay} --passes 2 --out-dir $OUT
'''
NOTE='''# Filler assembly: the world file carries its own audited pass-2 paragraph pool ('filler_pools', 'insert.filler_pools_note': non-sealed streams 9101/9103/9104/9105, forbidden strings / duplicates / canary filtered, equal in both layouts); nat_filler.select returns it instead of selecting whole streams from $NAT_DIR (other worlds are unchanged), so
#   pass 1 = the world's own background (identical to the world file), pass 2 = pool paragraphs cut to each slot's token length (exposure units at the same token positions as pass 1, asserted; strata equal; audit repeated on the used texts). No sealed stream (9102, 9106) is read.
# Readout: R1-alt (SQuAD tail, primary) and R1 (pg1342 tail), fp32 packed, every item family per person (meta_nat_<tag>_s9201.json 'items'); states are deleted after scoring (no --keep). No --log-clip. No real-name map is read.
# Aborts before any GPU work if the per-person table (e52_table.py), the pass-2 dry run or the filler audit fails; if save mode stops at an assert (stream length, exposure drift, SQuAD-tail length), fix the code, do not relax the assert. First GPU run of the Level-2 stream.
'''
def native(num,lay,tag,t_est,extra):
    return f'''#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about {t_est} [estimate with 30% slack from the design: ~25-30 min native A+B passes 1-2 + 13-23 min model load/readout setup per call + evaluation of 5 distinct states (W0, pass 1 and 2 of worlds A and B); NOT measured on this stream]. CPU steps first (~5-10 min: table, dry run with the real tokenizer, filler audit).
# E52 Level-2 controlled insertion, layout {lay} (PROPOSED, NOT QUEUED; needs Hans's approval in the experiment-run thread). Stream 9201 = real E33 background (stream 9105 paragraphs + about 100 non-sealed paragraphs) with 60 new synthetic persons (30 swap pairs; dev2 names, places, occupations, frames, none from the sealed lists):
#   24 flip pairs (in L2a 12 pairs C1 = 4 exposures with every gap <= 4096 tokens and in different 1024-token chunks, 12 pairs C2 = 4 exposures with every gap >= 12288; reversed in L2b) + 6 C0 pairs (1 exposure in both layouts). Last-exposure position per pair equal in both layouts (max diff <= 150 tokens), total length and background equal.
# Cell: native-fs (L0fs, the base model's own write), worlds A and B, passes 1-2, tag {tag}. {extra}
{NOTE}# Output: $OUT/L0fs_{tag}_s9201_{{A,B}}.json, meta_nat_{tag}_s9201.json, e52_person_table_9201.tsv, nat_filler_audit_9201.{{md,json}}.
{COMMON}{CPU(lay)}COMMON="--nat 9201:{lay} --nat-dir $NAT_DIR --cells L0fs --passes 2 --worlds AB --tag {tag}"
$PY $S/nat_ladder.py $COMMON --mode save
$PY $S/nat_ladder.py $COMMON --mode eval
'''
j160=native(160,'L2a','e52a','1.0-1.5 h (design estimate 36-56 min per call for the GPU part + CPU steps)','Run this job first; job 161 (L2b) runs the same cell on the other layout and then writes the pair-level analysis.')
j161=native(161,'L2b','e52b','1.0-1.5 h (design estimate 36-56 min per call for the GPU part + CPU steps)','Needs the outputs of job 160 in $OUT for the final analysis step (skipped with a message when they are missing).')+'''if [ -f $OUT/L0fs_e52a_s9201_A.json ] && [ -f $OUT/L0fs_e52a_s9201_B.json ]; then
  $PY $S/e52_analyze.py --dir $OUT --nat-dir $NAT_DIR --stream 9201 --tags e52a,e52b --cell L0fs
else
  echo "analysis skipped: the L2a outputs (job 160) are not in $OUT"
fi
'''
j162=f'''#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 0.6-0.9 h [estimate with 30% slack from the design: 33-52 min; L1 doc rows ~15 packed 1024-token steps per epoch, 8 epochs x worlds A and B, evaluation at W0 and epochs 1, 2, 3, 8; NOT measured on this stream].
# E52 Level-2, offline L1 control (PROPOSED, NOT QUEUED; needs Hans's approval in the experiment-run thread): each exposure unit of the stream 9201 alone (doc rows, the same units in both layouts, so one run on layout L2a covers both), plain SGD as the synthetic L1, worlds A and B, 8 epochs, tag e52l1. Tells whether the inserted facts are readable at all by a short, context-free write.
# Output: $OUT/L1_e52l1_s9201_{{A,B}}.json, meta_nat_e52l1_s9201.json. No filler is used (no pass-2 pool). No sealed stream is read.
{COMMON}$PY $S/nat_filler.py --selftest
$PY $S/nat_pass.py --selftest
$PY $S/e52_table.py --nat-dir $NAT_DIR --stream 9201 --require-real --tokenizer-file $TOK --out $OUT/e52_person_table_9201.tsv
COMMON="--nat 9201:L2a --nat-dir $NAT_DIR --cells L1 --epochs 8 --l1-eval-epochs 1,2,3,8 --worlds AB --tag e52l1"
$PY $S/nat_ladder.py $COMMON --mode save
$PY $S/nat_ladder.py $COMMON --mode eval
'''
here=Path(__file__).resolve().parent
for n,t in (('160_e52_L2a_native.sh',j160),('161_e52_L2b_native.sh',j161),('162_e52_L1.sh',j162)):
    (here/n).write_text(t);os.chmod(here/n,0o755);print('wrote',n)
