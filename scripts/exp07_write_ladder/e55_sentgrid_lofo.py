"""E55 32.12b leakage checks of the learned head weighting (242 part 2) on the seed-61 sentence dump (CPU only). The 242 result (leave-one-CHUNK-out: hidden@R0.90 = .324 = the fact share, i.e. a perfect ranking) could be inflated because the same frame / the same fact sits in the train and the test chunks. Here the cross-validation is GROUPED so that no group is shared between train and test:
  frame   : folds of sentence frames (e55_sentframe.py labels; fact sentences grouped by frame, K folds, background sentences spread by a fixed hash)
  fact    : folds of fact ids (no fact, hence no name / place, shared between train and test)
  chunk   : leave-one-chunk-out (the 242 setting, reference)
Controls (same frame folds): PERMUTED labels (training labels shuffled: must fall to chance), features from NON-FACT query rows only / FACT query rows only (does the ranking need the write-chunk sentences that repeat the facts? the query-row labels q_fact select rows for this analysis only; the label-free selection would use all rows).
  python3 e55_sentgrid_lofo.py --dump D --labels L --frames F --out report.md   |   --selftest
LABELS ARE USED IN TRAINING in every row; the evaluation is out-of-fold for every sentence."""
import sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e55_sentgrid_ext as X
import e55_sentgrid as G
K=10
def feats_rows(d,agg,group,rowsel):
    """like X.features but the query sentences are restricted to rowsel (bool [nQ])"""
    AL=np.asarray(d['AL'],np.float32);nL=AL.shape[0]
    if group=='g3':AL=AL[2*nL//3:]
    cand=np.nonzero(np.asarray(d['w_len'])>=X.MIN_TOK)[0];A=AL[:,:,rowsel,:]
    F=(A.sum(2) if agg=='sum' else A.max(2) if A.shape[2] else np.zeros(A.shape[:2]+(A.shape[3],),np.float32))[:,:,cand]
    return np.log(F.reshape(-1,len(cand)).T+1e-4),cand
def folds_for(mode,chunks_d,labs,frames,seed=0):
    """-> {chunk: fold id per candidate sentence}"""
    rng=np.random.default_rng(seed);fold={}
    uf=sorted({int(f) for k in chunks_d for f in frames[k]['w_frame'] if f>=0});um=sorted({int(f) for k in chunks_d for f in labs[k]['w_fact'] if f>=0})
    fmap={f:i%K for i,f in enumerate(rng.permutation(uf))};tmap={f:i%K for i,f in enumerate(rng.permutation(um))}
    for ci,k in enumerate(sorted(chunks_d)):
        cand=np.nonzero(np.asarray(chunks_d[k]['w_len'])>=X.MIN_TOK)[0];bgf=rng.integers(0,K,len(cand));fr=frames[k]['w_frame'][cand];ft=np.asarray(labs[k]['w_fact'])[cand]
        if mode=='frame':fold[k]=np.where(fr>=0,[fmap.get(int(f),0) for f in fr],bgf)
        elif mode=='fact':fold[k]=np.where(ft>=0,[tmap.get(int(f),0) for f in ft],bgf)
        else:fold[k]=np.full(len(cand),ci)
    return fold
def grouped_scores(chunks_d,labs,frames,mode='frame',agg='sum',group='all',lam=100.0,rows='all',permute=False,seed=0):
    FE={};
    for k,d in chunks_d.items():
        nQ=len(d['q_len']);qf=np.asarray(labs[k]['q_fact']);sel=np.ones(nQ,bool) if rows=='all' else (qf<0 if rows=='nonfact' else qf>=0);FE[k]=feats_rows(d,agg,group,sel)
    fold=folds_for(mode,chunks_d,labs,frames,seed);ks=sorted(chunks_d)
    Xall=np.concatenate([FE[k][0] for k in ks]);yall=np.concatenate([(np.asarray(labs[k]['w_fact'])[FE[k][1]]>=0).astype(float) for k in ks]);fall=np.concatenate([fold[k] for k in ks])
    rng=np.random.default_rng(seed+7);pred=np.zeros(len(yall))
    for f in np.unique(fall):
        tr=fall!=f;ytr=yall[tr].copy()
        if permute:rng.shuffle(ytr)
        m=X.fit_lr(Xall[tr],ytr,lam);pred[~tr]=X.predict_lr(m,Xall[~tr])
    scores={};o=0
    for k in ks:
        n=len(FE[k][1]);s=np.full(len(chunks_d[k]['w_len']),-np.inf);s[FE[k][1]]=pred[o:o+n];scores[k]=s;o+=n
    return scores
def report(chunks,labs,frames,out=None):
    L=['# E55 32.12b grouped cross-validation and controls of the learned head weighting (LABELS USED IN TRAINING; evaluation out-of-fold)','',f'Frame / fact folds: K = {K} disjoint groups, no frame (fact) shared between train and test. Reference rows: leave-one-chunk-out (242). Hide rule and summaries as in 242; the bar is hidden@R0.90 of candidate 1 (0.596), chance 0.84; a perfect ranking gives about 0.324.','']
    for st in sorted(chunks):
        cd=chunks[st];L+=[f'## state {st}','','| evaluation | agg | rows | recall@0.45 | recall@0.35 | hidden@R0.90 | hidden@R0.80 | precision@0.35 |','|---|---|---|---|---|---|---|---|']
        ref=X.summarize(X.curve({k:X.score_cand1(d) for k,d in cd.items()},cd,labs));L.append(f"| cand1 rank curve (no labels) | - | all | {ref['recall@0.45']:.3f} | {ref['recall@0.35']:.3f} | {ref['hidden@R0.90']:.3f} | {ref['hidden@R0.80']:.3f} | {ref['precision@0.35']:.3f} |")
        for mode,agg,rows,perm,name in (('chunk','sum','all',False,'leave-one-chunk-out (242)'),('frame','sum','all',False,'grouped by FRAME'),('frame','max','all',False,'grouped by FRAME'),('fact','sum','all',False,'grouped by FACT'),('fact','max','all',False,'grouped by FACT'),('frame','sum','nonfact',False,'grouped by FRAME, query rows without fact label only'),('frame','sum','fact',False,'grouped by FRAME, fact query rows only'),('frame','sum','all',True,'grouped by FRAME, PERMUTED training labels (control)')):
            s=X.summarize(X.curve(grouped_scores(cd,labs,frames,mode,agg,'all',100.0,rows,perm),cd,labs))
            L.append(f"| {name} | {agg} | {rows} | {s['recall@0.45']:.3f} | {s['recall@0.35']:.3f} | {s['hidden@R0.90']:.3f} | {s['hidden@R0.80']:.3f} | {s['precision@0.35']:.3f} |")
        L.append('')
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def load_frames(path):
    Z=np.load(path);fr={}
    for key in Z.files:
        c,nm=key.split('/');fr.setdefault(int(c[1:]),{})[nm]=Z[key]
    return fr
def selftest():
    cd={};labs={};frames={}
    for k in (1,2,3,4):
        d,l=X.synth(k);nW=len(d['w_len']);fact=np.asarray(l['w_fact'])>=0;l['w_fact']=np.where(fact,np.arange(nW)%12,-1).astype(np.int16);l['q_fact']=np.where(np.arange(len(d['q_len']))%2==0,3,-1).astype(np.int16)
        cd[k]=d;labs[k]=l;frames[k]={'w_frame':np.where(fact,(np.arange(nW)*7+k)%40,-1).astype(np.int16)}
    for mode in ('frame','fact','chunk'):
        s=X.summarize(X.curve(grouped_scores(cd,labs,frames,mode,'sum','all',10.0),cd,labs));assert s['recall@0.35']>0.85,(mode,s)
    sp=X.summarize(X.curve(grouped_scores(cd,labs,frames,'frame','sum','all',10.0,permute=True),cd,labs));assert sp['recall@0.35']<0.7,sp
    sn=X.summarize(X.curve(grouped_scores(cd,labs,frames,'frame','sum','all',10.0,rows='nonfact'),cd,labs));assert sn['recall@0.35']>0.5,sn
    f=folds_for('frame',cd,labs,frames);assert max(int(v.max()) for v in f.values())<K
    t=report({'S':cd},labs,frames);assert 'PERMUTED' in t;print('e55_sentgrid_lofo selftest ok: permuted-label recall@0.35 %.2f'%sp['recall@0.35'])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--dump');ap.add_argument('--labels');ap.add_argument('--frames');ap.add_argument('--out');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    chunks,lab=G.load(a.dump,a.labels);print(report(chunks,lab,load_frames(a.frames),a.out))
