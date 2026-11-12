#!/bin/bash
# Label-free suppression screen (PROTOCOL_stage01.md 2026-10-16 22:15 KST): STAGE=lfscreen on s101-103, then analysis.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
for sd in 101 102 103; do [ -f $OUT/ner_spans_s${sd}_A.json ] && [ -f $OUT/ner_spans_s${sd}_B.json ] || { lg "LFSCREEN missing NER spans s$sd"; exit 1; }; done
for sd in 101 102 103; do lg START lfscreen s$sd; SEED=$sd ONESTEP2=1 STAGE=lfscreen $PY $S/grad_components.py > $OUT/onestep2_lfscreen_s$sd.log 2>&1 && lg DONE lfscreen s$sd || lg FAILED lfscreen s$sd; done
$PY $S/analyze_onestep2.py lfscreen > $OUT/onestep2_lfscreen.log 2>&1; lg "ANALYZED lfscreen $(grep -o 'LF-SWAP [A-Z]*' $OUT/onestep2_lfscreen.md | head -1)"
lg LFSCREEN_DONE
