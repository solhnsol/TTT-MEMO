"""E57 label shares of the highest-window-gain rows, narrower and wider top fractions (CPU only; new file; numbers only). Same label definition as e57_gain_labels.py (A = inside an answer-string hit of a bank question in an article of its event; S = inside a subject-string hit; SENT = sentence holding an A or S hit of the same event, outside the hit; OTHER) for the 278,256 eligible rows of the chunks 8..279 (row i predicts token chunk*1024 + i + 1). Rows are ranked by the W0 band gain (band-B NLL - window NLL) for B = 512, 256, 128. For the top fractions 0.01 / 0.03 / 0.1 / 0.3 / 1 / 3 / 10 / 30 / 100 % per label: rows, share of the group's rows, share of the group's gain sum, enrichment (row share / row share among all eligible rows). Per question (B = 512): the A rows of that question, how many are in the top 1 % and in the top 0.1 %. Saves the labels per row in an npz.
  .venv/bin/python scripts/exp07_write_ladder/e57_gain_labels_top.py --bundle B --band-npz e57_band_W0.npz --band256 E57_closure_W0.npz --out O.json --npz R.npz"""
import sys,json,re,pathlib,argparse,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
import e57_gain_labels as GL
CH=1024;FR=[.0001,.0003,.001,.003,.01,.03,.10,.30,1.0]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--band-npz');ap.add_argument('--band256');ap.add_argument('--out');ap.add_argument('--npz');ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);tk=UVm.tokenizer(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits_=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(tk,ids);N=len(ids)
    strs=[tk.decode([int(t)]) for t in ids];offs=np.cumsum([0]+[len(s) for s in strs]);full=''.join(strs)
    lab=np.zeros(N,np.int8);qA=np.full(N,-1,np.int16);sent_of=np.full(N,-1,np.int64);ns=0
    for k in range(N//CH):
        for (s_,e_) in UVm.chunk_sents(ends,k,CH):sent_of[k*CH+s_:k*CH+e_]=ns;ns+=1
    sentflag=np.zeros(ns+1,bool)
    for qi,it in enumerate(items):
        q=it['q'];rxA=re.compile('|'.join(r'(?<![\w])'+re.escape(v)+r'(?![\w])' for v in GL.forms(q)),re.I);S=q.get('S') or it['event'];rxS=re.compile(r'(?<![\w])'+re.escape(S)+r'(?![\w])',re.I)
        for (s0,e0,ev,rid) in iv:
            e0=min(e0,N)
            if ev!=it['event'] or s0>=N:continue
            txt=full[offs[s0]:offs[e0]];of=offs[s0:e0+1]-offs[s0]
            for (i0,i1) in GL.hits(txt,of,rxA):
                for p in range(s0+i0,s0+i1+1):lab[p]=3;qA[p]=qi
                so=sent_of[s0+i0:s0+i1+1];sentflag[so[so>=0]]=True
            for (i0,i1) in GL.hits(txt,of,rxS):
                for p in range(s0+i0,s0+i1+1):
                    if lab[p]<2:lab[p]=2
                so=sent_of[s0+i0:s0+i1+1];sentflag[so[so>=0]]=True
    insent=(sent_of>=0)&sentflag[np.clip(sent_of,0,ns)];lab=np.where((lab==0)&insent,1,lab).astype(np.int8)
    B=np.load(a.band_npz);B2=np.load(a.band256);nbs={int(b):B['nll_band'][i] for i,b in enumerate(B['bands'])};nbs[256]=B2['nll_band'][list(B2['bands']).index(256)];nw=B['nll_win'];lm=B['lm']
    pos=[];gains={b:[] for b in (128,256,512)}
    for k in range(8,280):
        sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(UVm.chunk_sents(ends,k,CH)):sid[s_:e_]=i
        el=CM.rows_eligible(lm[k],sid)&lm[k];r=np.nonzero(el)[0];pos.append(k*CH+r+1)
        for b in gains:gains[b].append((nbs[b][k]-nw[k])[r].astype(np.float64))
    pos=np.concatenate(pos);gains={b:np.concatenate(v) for b,v in gains.items()};lb=lab[pos];qq=qA[pos];names=['OTHER','SENT','S','A'];n=len(pos);base={nm:float((lb==i).mean()) for i,nm in enumerate(names)}
    np.savez_compressed(a.npz,pos=pos,label=lb,question=qq,gain128=gains[128],gain256=gains[256],gain512=gains[512],label_names=np.asarray(names))
    out={'rows':n,'label_share_all':base,'definition':'ranking by W0 band-B NLL - window NLL; labels as e57_gain_labels.py','bands':{}}
    for b in (512,256,128):
        g=gains[b];order=np.argsort(-g,kind='stable');res={}
        for f in FR:
            m=np.zeros(n,bool);m[order[:max(1,int(round(f*n)))]]=True;gs=g[m].sum();d={}
            for i,nm in enumerate(names):
                s=m&(lb==i);d[nm]=dict(rows=int(s.sum()),row_share=float(s.sum()/m.sum()),gain_sum_share=float(g[s].sum()/gs) if gs!=0 else None,enrichment=float(s.sum()/m.sum()/base[nm]) if base[nm]>0 else None)
            res[f'{f*100:g}%']=dict(rows=int(m.sum()),mean_gain=float(g[m].mean()),min_gain=float(g[m].min()),labels=d)
        out['bands'][f'B{b}']=res
    g=gains[512];order=np.argsort(-g,kind='stable');t1=np.zeros(n,bool);t1[order[:int(round(.01*n))]]=True;t01=np.zeros(n,bool);t01[order[:int(round(.001*n))]]=True;pq={}
    for qi,it in enumerate(items):
        s=qq==qi;na=int(s.sum());pq[it['id']]=dict(event=it['event'],type=it['type'],rows_with_A=na,in_top1pct=int((s&t1).sum()),frac_top1pct=float((s&t1).sum()/na) if na else None,in_top0_1pct=int((s&t01).sum()),frac_top0_1pct=float((s&t01).sum()/na) if na else None)
    out['per_question_B512']=pq;json.dump(out,open(a.out,'w'),indent=1)
    print('rows',n,{k:round(v,4) for k,v in base.items()})
    for bk,res in out['bands'].items():
        print('==',bk)
        for fk,v in res.items():
            L=v['labels'];print(f"  top {fk:>7s} rows {v['rows']:6d} mean gain {v['mean_gain']:+.2f} | "+' | '.join(f"{nm} {L[nm]['row_share']:.3f} (x{L[nm]['enrichment']:.2f}, gain {L[nm]['gain_sum_share'] if L[nm]['gain_sum_share'] is None else round(L[nm]['gain_sum_share'],3)})" for nm in ('A','S','SENT','OTHER')))
if __name__=='__main__':main()
