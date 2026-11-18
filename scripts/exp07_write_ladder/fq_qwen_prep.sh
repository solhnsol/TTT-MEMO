#!/bin/bash
# CPU chain between the Qwen W0 screen (GPU) and the Qwen FictionalQA runs (thread-server 11/18; new file): Qwen screens -> replacement on the Llama-3 worlds (fq_w0_select.py + the unchanged fq_w0_apply.py; the facts of a Qwen world are the facts of its Llama world) -> Qwen re-tokenisation (fq_rebuild_qwen.py of the registry branch, tokenizer Qwen/Qwen3-4B, every chunk exactly 1024 tokens) -> controls completed to 4 (fq_ctrl_add.py with the Qwen 3-token gold filter) -> adapter checks (fq_world.py checks with the Qwen tokenizer: all must be True, 4 controls). usage: fq_qwen_prep.sh <screen dir> <out root>   (default env /home/solhnsol/fqenv, parquet in fq/data)
set -eo pipefail
cd "$(dirname "$0")/../.."
SCREEN=$1; R=$2; FQ=experiments/exp07_write_ladder/fq; ENV=${FQENV:-/home/solhnsol/fqenv/bin/python}; D=scripts/exp07_write_ladder
rm -rf $R; mkdir -p $R/w0q $R/w0q_in $R/qwen $R/qwen_in $R/qwen_c
$ENV $D/fq_w0_select.py --worlds $FQ/worlds --screen $SCREEN --pool $FQ/data --out $R/w0q > $R/select.log 2>&1 || { cat $R/select.log; exit 21; }
grep -q STOP $R/w0q/fq_w0_select_report.json && { echo "reserve exhausted in a world: STOP"; cat $R/w0q/fq_w0_select_report.json; exit 22; }
for s in 1 2 3 4 5 6 7 8; do python3 - $R/w0q/fq_20${s}_cap3_VAR_w0.json $R/w0q_in/fq_20${s}_cap3_VAR.json <<'PYE'
import json,sys
w=json.load(open(sys.argv[1]));w.pop('ctrl_extra',None);w.pop('ctrl_extra_source',None)      # the dropped facts that fq_w0_select.py stores as ctrl_extra have their gold in the background of the replaced world: the rebuild must not treat them as controls (fq_ctrl_add.py chooses the controls afterwards)
json.dump(w,open(sys.argv[2],'w'),ensure_ascii=False)
PYE
done
$ENV -I $FQ/code/fq_rebuild_qwen.py $FQ/data $R/w0q_in $R/qwen 201 202 203 204 205 206 207 208 2>&1 | grep -v Warn | tee $R/rebuild.log
grep -q "CHECK FAIL" $R/rebuild.log && { echo "Qwen rebuild check failed"; exit 23; }
for s in 1 2 3 4 5 6 7 8; do cp $R/qwen/fq_20${s}_cap3_VAR.json $R/qwen_in/fq_20${s}_cap3_VAR_w0.json; done
$ENV -I $D/fq_ctrl_add.py $FQ/data $FQ/code $R/qwen_in $R/qwen_c Qwen/Qwen3-4B 2>&1 | grep -v Warn | tail -3
.venv/bin/python - $R/qwen_c <<'PYE'
import sys,glob,pathlib
sys.path.insert(0,'scripts/exp07_write_ladder');import fq_world as FW
tk=FW.load_tokenizer(sorted(pathlib.Path('/home/solhnsol/.cache/huggingface/hub').glob('models--Qwen--Qwen3-4B-Base/snapshots/*/tokenizer.json'))[0]);bad=0
for f in sorted(glob.glob(sys.argv[1]+'/fq_*_w0.json')):
    w=FW.load_world(f);gw=FW.to_gen_world(w,tk);c=FW.checks(w,gw,tk);ok=all(v is True for k,v in c.items() if k!='items_kinds') and len(gw['ctrl'])==4 and gw['fq']['n_fitted']['none']==128;print(f.split('/')[-1],'seed label',gw['seed'],'ctrl',len(gw['ctrl']),'fit',gw['fq']['n_fitted'],'checks ok',ok)
    bad+=not ok
assert bad==0,'adapter checks failed'
print('QWEN FQ PREP OK')
PYE
