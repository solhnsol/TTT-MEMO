"""Qwen read-path diagnosis (job 1369 'qwen_readdiag'; GPU; NEW file; implementation thread 11/29; Hans 11/29 04:42 KST "1367b는 보류하고 진단부터 해봐"). Why ISO1 of 1367a binds ~0 (0.01 / 0.21) while the ISO of 1322 bound ~2 (2.08 / 1.73) with the SAME write (fact CE 7.23 -> 0.93 vs 7.24 -> 0.94, 368 loss tokens, norm 0.304): the read path differs. ISO1 (e59_qwen_ladder: the 16 read-fact sentences isolated, one fixed-norm step per chunk) on the U12 world, seed --world-seed, 64 chunks; at --reads (default 32, 64) and at W0 the R1 questions (16 x 16 candidates, 12 exposed + 4 never exposed) are read under 2 x 2 conditions:
  context SQ8 = BOS + SQuAD dev 8192 tokens (the 1367a R1)  |  NV7 = the narrative tail pg1342, 7168 tokens (world_text.neutral_tail, the R1 of 1322 / 1334)
  hook    on  = the 1367a probe-row sliding limit (key distance < 8192)  |  off = the original readout.score_rows
Second run (pipeline check): --cell v3_16_1 --world-seed 221 --diag-arm ISO reproduces the 1322 ISO world and arm under the same four read conditions. Output: raw scores per condition, matched binding / top-1 / familiarity printed per condition (cand_fact of the items, no scipy).
  python3 e59_qwen_readdiag.py --world-seed 231 --cell u12_k32 --fast fullmlp --lr 1.0 --clip 1.0 --step-norm 0.30426 --window-chunks 8 --readout-dtype fp32 --max-chunks 64 --reads 32,64 --out O"""
import sys,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'exp07_write_ladder'));sys.path.insert(0,str(HERE.parent/'exp07_write_ladder'/'e57'))
import numpy as np
def measures(S,W0,items):
    fi=[i for i,it in enumerate(items) if it['kind']=='fact'];items=[items[i] for i in fi];S=np.asarray(S,float)[fi];W0=np.asarray(W0,float)[fi];dS=S-W0;mb=[];acc=[];fam=[]
    for i,it in enumerate(items):
        f=it['fact'];g=it['gold_idx'];oth=[dS[j,items[j]['cand_fact'].index(f)] for j in range(len(items)) if j!=i and f in items[j]['cand_fact']]
        mb.append(dS[i,g]-np.mean(oth));acc.append(int(np.argmax(S[i])==g));ex=[c for c in range(len(it['cand_fact'])) if it['cand_fact'][c]>=0 and c!=g];ne=[c for c in range(len(it['cand_fact'])) if it['cand_fact'][c]<0]
        if ne:fam.append(dS[i,ex].mean()-dS[i,ne].mean())
    return dict(bind=float(np.mean(mb)),top1=float(np.mean(acc)),fam=float(np.mean(fam)) if fam else None)
def run():
    import torch,e59_qwen_ladder as QL,cell_band_ladder as CB,qwen_ladder as QLD,qwen_runtime as QR,readout as RO
    _orig_sr=RO.score_rows;E=QL.install();_hook_sr=RO.score_rows;import world_text as WT
    ap=E.parser();ap.add_argument('--diag-arm',default='ISO1');a=ap.parse_args();torch.manual_seed(0);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t0=time.monotonic()
    dev,mdir,tok,bos,model,lcfg=QLD.build(a);CH=a.chunk;reads=[int(x) for x in a.reads.split(',')];seed=a.world_seed
    tf=QR.tokenizer_file(mdir);enc,w,stream,sha,items=CB.build_world(tf,seed,a.cell,reads);rows,nt=CB.item_rows(tok,items);pad=tok.token_to_id('<|endoftext|>')
    sq8=WT.neutral_tail(tok,bos)[0];nv7=QL_orig_tail(WT,tok,bos);assert len(sq8)==8192 and len(nv7)==7168,(len(sq8),len(nv7))
    rd={'bf16':None,'fp32':torch.float32}[a.readout_dtype];init=QR.lora_state(model)
    def read4(m):
        res={}
        for hk,fn in (('on',_hook_sr),('off',_orig_sr)):
            RO.score_rows=fn;E.CB.RO.score_rows=fn
            try:
                for cn,ids in (('SQ8',sq8),('NV7',nv7)):res[f'{cn}_{hk}']=CB.read_all(m,{'x':ids},rows,dev,a.readout_batch,pad,rd)['x']
            finally:RO.score_rows=_hook_sr;E.CB.RO.score_rows=_hook_sr
        return res
    QR.load_lora_state_(model,init);W0=read4(model);RES=dict(seed=seed,reads={},W0=W0,args=vars(a),measures={});print('W0 read',round(time.monotonic()-t0),'s',flush=True)
    fix=E.Fixed(model,a.step_norm,dev);nch=min(a.max_chunks or 64,len(stream)//CH);stream=stream[:nch*CH];sep=list(w['sep'])
    for k,rec in E.run_arm(a,a.diag_arm,w,stream,range(nch),model,dev,a.step_norm,fix,CH,sep):
        if (k+1) in reads:
            R=read4(model);RES['reads'][str(k+1)]=R;RES['measures'][str(k+1)]={c:measures(R[c],W0[c],items) for c in R}
            print(f'{a.diag_arm} {a.cell} s{seed} e{k+1}:',json.dumps({c:{q:round(v,3) for q,v in d.items() if v is not None} for c,d in RES['measures'][str(k+1)].items()}),flush=True)
            json.dump(RES,open(out/f'readdiag_{a.cell}_s{seed}.json','w'),default=float)
    RES['fact_ce_first_last']=[fix.steps[0].get('ce'),fix.steps[-1].get('ce')];json.dump(RES,open(out/f'readdiag_{a.cell}_s{seed}.json','w'),default=float);print('done',round(time.monotonic()-t0),'s',flush=True)
def QL_orig_tail(WT,tok,bos):
    """the 1322 / 1334 R1 context: world_text.neutral_tail_text (pg1342), BOS + tokens, JAX_TAIL_LEN = 7168"""
    text,sha=WT.neutral_tail_text();ids=WT._pre(bos)+WT.enc(tok,text).ids;return ids[:WT.JAX_TAIL_LEN]
if __name__=='__main__':run()
