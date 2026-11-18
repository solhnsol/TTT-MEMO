"""FictionalQA W0 pre-screen on Qwen3-4B-Base (GPU forward only, no update; new file; thread-server 11/18; the Qwen counterpart of scripts/exp07_write_ladder/fq_screen_run.py): for every world fq_<builder seed 201..208>_cap3_VAR.json of --fq-worlds (the Qwen3 worlds, labels 211..218) all 24 facts (16 world + 8 reserve) are read at W0 with the Qwen reader of the exp09 runs (cell_band_ladder.read_all: neutral out-of-window tail of --tail-len tokens, summed span log-prob). Candidates of fact i: its own gold first, then the golds of the facts i+1..i+15 (mod 24), as in the E2E screen (same item builder fq_screen_run.screen_items); margin = gold minus best wrong; screened out = margin > 0. Output {out}/fq_w0screen_<label>.json in the format of the E2E screens (fq_w0_select.py reads both).
  python3 fq_qwen_screen.py --fq-worlds DIR --out OUT --fast fullmlp [--labels 211,...,218] | --selftest"""
import sys,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
import numpy as np,torch
import cell_band_ladder as CB
import qwen_ladder as QL,qwen_runtime as QR,world_text as WT
import fq_screen_run as FS,fq_world as FW
def parser():
    ap=CB.parser();ap.add_argument('--fq-worlds');ap.add_argument('--labels',default=','.join(str(x) for x in range(211,219)));return ap
def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);dev,mdir,tok,bos,model,lcfg=QL.build(a);pad=tok.token_to_id('<|endoftext|>');out_ids=WT.neutral_tail(tok,bos,a.tail_len)[0]
    rd={'bf16':None,'fp32':torch.float32}[a.readout_dtype] if not a.tiny else None;inv={v:k for k,v in FW.SEED_LABEL.items()}
    for lab in [int(x) for x in a.labels.split(',')]:
        fn=out/f'fq_w0screen_{lab}.json'
        if fn.exists() and not a.rerun:print('exists',fn);continue
        t0=time.monotonic();w=json.load(open(Path(a.fq_worlds)/f'fq_{inv[lab]}_cap3_VAR.json'));items=FS.screen_items(w);rows,nt=CB.item_rows(tok,items);assert all(len(set(n))==1 for n in nt)
        S=np.asarray(CB.read_all(model,{'out':out_ids},rows,dev,a.readout_batch,pad,rd)['out'],np.float64);assert S.shape==(FS.NALL,FS.NC)
        m,t1,rk=FS.margins(S);rec=dict(label=lab,builder_seed=inv[lab],world_sha=w['sha'],tokenizer='Qwen3-4B-Base (reader of exp09)',facts=[dict(idx=i,role=items[i]['role'],fact_id=(w['facts']+w['reserve'])[i].get('id',i-FS.NF),margin=float(m[i]),top1=bool(t1[i]),rank=int(rk[i]),screened_out=bool(m[i]>0)) for i in range(FS.NALL)],scores=S.tolist(),sec=time.monotonic()-t0,candidates='own gold + golds of facts i+1..i+15 mod 24',margin_def='summed span log-prob gold minus best wrong, out-of-window neutral tail, W0, Qwen3-4B-Base')
        json.dump(rec,open(fn,'w'));print(f'world {lab} (builder {inv[lab]}): world facts margin>0 {int((m[:FS.NF]>0).sum())}/16, reserve margin>0 {int((m[FS.NF:]>0).sum())}/8, mean margin {m.mean():+.2f}, {time.monotonic()-t0:.0f}s',flush=True)
def selftest():
    sys.argv=[sys.argv[0]];import tempfile,shutil
    td=Path(tempfile.mkdtemp());wd=ROOT/'experiments/exp07_write_ladder/fq/worlds_qwen3'
    a=parser().parse_args(['--tiny','--device','cpu','--tail-len','256','--readout-batch','16','--threads','4','--lora-layers','2','--fast','fullmlp','--fq-worlds',str(wd),'--out',str(td),'--labels','211','--rerun']);torch.manual_seed(0);run(a)
    r=json.load(open(td/'fq_w0screen_211.json'));assert len(r['facts'])==24 and len(r['scores'])==24 and len(r['scores'][0])==16 and r['facts'][3]['role']=='world' and r['facts'][20]['role']=='reserve'
    shutil.rmtree(td,ignore_errors=True);print('fq_qwen_screen selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:a=parser().parse_args();torch.manual_seed(0);run(a)
