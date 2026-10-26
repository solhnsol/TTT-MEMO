"""E55 32.12 token-selection extensions on the sentence dump (CPU only, numpy / scipy; no GPU, no model). Goal: a LOWER mask fraction than candidate 1 (V6 tok heads50% g3 thr0.1 trigq>=1; seed 61: recall .90 at mask ~.61) at the same recall.
  python3 e55_sentgrid_ext.py --dump e55_sentdump_s61.npz --labels e55_sentdump_labels_s61.npz --out report.md [--save-coef coef.npz]   |   --apply-coef coef.npz --dump D --labels L --out R   |   --selftest
Every method gives a SCORE per window sentence (candidates = >= 4 tokens); the hide rule is the same for all: per chunk, hide the highest-scoring candidate sentences until a target mask fraction of the window tokens is reached (ties broken at random, fixed seed). Curves are pooled over the chunks (token-weighted); metrics: recall, precision, accuracy, F1, mask fraction. Labels (fact sentences) enter ONLY the metrics, except method (2), which uses them for training and is marked 'LABELS USED IN TRAINING'.
  (1) surprisal: candidate-1 votes counted only from query tokens with high write cross-entropy (top q %), or weighted by the cross-entropy of the voting token / voting query sentence.
  (2) AT2-style learned head weighting: logistic regression (L2) on the per-(layer, head) sentence mass summed (or maximised) over the query sentences; leave-one-chunk-out inside the dump (train on the other chunks of the same state); frozen coefficients can be saved and applied to other seeds' dumps (--apply-coef).
  (3) neighbour expansion / pooling of the candidate-1 vote score over adjacent window sentences (weighted sum, max-pool) and two natural-point post-filters (dilation by one sentence, gap filling).
Reference: candidate 1 at its natural point (all sentences with >= 1 vote) and as a rank curve; chance (random sentences: recall = mask fraction)."""
import sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e55_sentsel as SEL
import e55_sentgrid as G
MIN_TOK=SEL.MIN_TOK
GRID=np.round(np.arange(0.05,0.951,0.025),4)
CAND1=dict(hgroup='g3',hpct=50,thr=0.1)
CAND1_CFG={'name':'V6 tok heads50% g3 thr0.1 trigq>=1','v6':True,'sel':'tok','hgroup':'g3','hpct':50,'thr':0.1,'trig':'q','k':1,'group':'g3','heads':'sharp'}
REFS=(0.61,0.45,0.35);RECALL_AT=(0.90,0.80)

# ---------------- scores (no labels) ----------------
def q_of_token(d):
    qs=np.full(np.asarray(d['T_max']).shape[-1],-1,np.int64)
    for i,(a,n) in enumerate(zip(np.asarray(d['q_start']),np.asarray(d['q_len']))):qs[a:a+n]=i
    return qs
def vote_matrix(d,qmask=None,weight=None,hgroup='g3',hpct=50,thr=0.1):
    """candidate-1 token votes: V[q,w] bool (query sentence q has a vote for window sentence w), ev[w] number of (head, query token) vote events, wev[w] the same weighted by weight[token]"""
    w_len=np.asarray(d['w_len']);nW=len(w_len);nQ=len(d['q_len']);sel=SEL.head_rank(d,hgroup,hpct);li,hi=np.nonzero(sel)
    Tm=np.asarray(d['T_max'],np.float32)[li,hi];Tw=np.asarray(d['T_wsid'])[li,hi].astype(np.int64);qs=q_of_token(d)
    ok=(Tm>=thr)&(qs[None,:]>=0)&(Tw>=0)
    if qmask is not None:ok&=np.asarray(qmask,bool)[None,:]
    r,t=np.nonzero(ok);V=np.zeros((nQ,nW),bool);V[qs[t],Tw[r,t]]=True
    ev=np.bincount(Tw[r,t],minlength=nW).astype(np.float64)
    wev=np.bincount(Tw[r,t],weights=np.asarray(weight,np.float64)[t],minlength=nW) if weight is not None else ev.copy()
    return V,ev,wev
def with_cand(score,d):
    s=np.asarray(score,np.float64).copy();s[np.asarray(d['w_len'])<MIN_TOK]=-np.inf;return s
def score_cand1(d,**kw):
    V,ev,_=vote_matrix(d,**CAND1,**kw);return with_cand(V.sum(0)+1e-3*ev/max(ev.max(),1),d)
def top_ce_mask(d,pct):
    ce=np.nan_to_num(np.asarray(d['q_tce'],np.float32),nan=-1e9);qs=q_of_token(d)>=0
    if not qs.any():return qs
    thr=np.percentile(ce[qs],100-pct);return qs&(ce>=thr)
def score_surprisal_top(d,pct):
    V,ev,_=vote_matrix(d,**CAND1,qmask=top_ce_mask(d,pct));return with_cand(V.sum(0)+1e-3*ev/max(ev.max(),1),d)
def score_surprisal_tokw(d):
    ce=np.nan_to_num(np.asarray(d['q_tce'],np.float32),nan=0.0);V,ev,wev=vote_matrix(d,**CAND1,weight=ce);return with_cand(wev,d)
def score_surprisal_sentw(d):
    V,ev,_=vote_matrix(d,**CAND1);qc=np.nan_to_num(np.asarray(d['q_ce'],np.float64),nan=0.0);return with_cand((V*qc[:,None]).sum(0)+1e-6*ev,d)
def shift(x,k):
    y=np.zeros_like(x)
    if k>0:y[k:]=x[:-k]
    else:y[:k]=x[-k:]
    return y
def score_neighbour(d,a=None,maxpool=False):
    V,ev,_=vote_matrix(d,**CAND1);c=V.sum(0).astype(np.float64)
    s=np.maximum(np.maximum(shift(c,1),shift(c,-1)),c) if maxpool else c+a*(shift(c,1)+shift(c,-1))
    return with_cand(s+1e-3*ev/max(ev.max(),1),d)
def natural_mask(d):
    s=score_cand1(d);return np.isfinite(s)&(s>=1)
def post_dilate(m):return m|shift(m,1)|shift(m,-1)
def post_gapfill(m):return m|(shift(m,1)&shift(m,-1))

# ---------------- hide rule, counts, metrics ----------------
def hide_to_fraction(score,w_len,frac,rng):
    """hide the highest-scoring candidate sentences (finite score) until frac of the window tokens is reached"""
    wl=np.asarray(w_len,np.int64);W=wl.sum();cand=np.nonzero(np.isfinite(score))[0]
    if len(cand)==0:return np.zeros(len(wl),bool)
    key=np.lexsort((rng.random(len(cand)),-score[cand]));order=cand[key];cum=np.cumsum(wl[order])
    n=int(np.searchsorted(cum,frac*W,side='left'))+1;m=np.zeros(len(wl),bool);m[order[:n]]=True;return m
def pooled_counts(masks,chunks_d,labs):
    c=np.zeros(6)
    for k,d in chunks_d.items():c+=G.counts(masks[k],d,labs[k])
    return c
def curve(scores,chunks_d,labs,seed=0):
    """scores: {chunk: score[nW]} -> list of (frac_target, metrics dict) over GRID, pooled over the chunks"""
    out=[]
    for f in GRID:
        rng=np.random.default_rng(seed);masks={k:hide_to_fraction(scores[k],chunks_d[k]['w_len'],f,rng) for k in chunks_d}
        out.append((float(f),G.metrics(pooled_counts(masks,chunks_d,labs))))
    return out
def summarize(cv):
    h=np.array([m['hidden'] for _,m in cv]);r=np.array([m['recall'] for _,m in cv]);p=np.array([m['precision'] for _,m in cv]);o=np.argsort(h);h,r,p=h[o],r[o],p[o]
    row={f'recall@{x:.2f}':float(np.interp(x,h,r)) for x in REFS}
    for tgt in RECALL_AT:
        i=np.nonzero(r>=tgt)[0];row[f'hidden@R{tgt:.2f}']=float(np.interp(tgt,r[:i[0]+1][-2:],h[:i[0]+1][-2:])) if len(i) and i[0]>0 else (float(h[i[0]]) if len(i) else float('nan'))
    row['mean_recall_minus_chance']=float(np.mean(r-h));row['precision@0.35']=float(np.interp(0.35,h,p));return row

# ---------------- (2) learned head weighting ----------------
def features(d,agg='sum',group='all'):
    """[nCand, L*H] log head-sentence mass (candidate sentences only) and the candidate index"""
    AL=np.asarray(d['AL'],np.float32);nL=AL.shape[0]
    if group=='g3':AL=AL[2*nL//3:]
    cand=np.nonzero(np.asarray(d['w_len'])>=MIN_TOK)[0];X=(AL.sum(2) if agg=='sum' else AL.max(2))[:,:,cand]
    X=np.log(X.reshape(-1,len(cand)).T+1e-4);return X,cand
def fit_lr(X,y,lam):
    from scipy.optimize import minimize
    mu=X.mean(0);sd=X.std(0)+1e-6;Z=(X-mu)/sd;n,p=Z.shape;pos=max(y.mean(),1e-3);w0=np.zeros(p+1);w0[-1]=np.log(pos/(1-pos))
    def f(w):
        z=Z@w[:-1]+w[-1];l=np.logaddexp(0,z)-y*z;pr=1/(1+np.exp(-z));g=Z.T@(pr-y)/n+lam*w[:-1]/n;return l.mean()+0.5*lam*(w[:-1]**2).sum()/n,np.r_[g,(pr-y).mean()]
    w=minimize(f,w0,jac=True,method='L-BFGS-B',options={'maxiter':300}).x;return {'w':w[:-1],'b':w[-1],'mu':mu,'sd':sd}
def predict_lr(m,X):return ((X-m['mu'])/m['sd'])@m['w']+m['b']
def lr_scores_loco(chunks_d,labs,agg,group,lam):
    feats={k:features(d,agg,group) for k,d in chunks_d.items()};scores={}
    for k in chunks_d:
        tr=[j for j in chunks_d if j!=k];X=np.concatenate([feats[j][0] for j in tr]);y=np.concatenate([(np.asarray(labs[j]['w_fact'])[feats[j][1]]>=0).astype(np.float64) for j in tr])
        m=fit_lr(X,y,lam);s=np.full(len(chunks_d[k]['w_len']),-np.inf);s[feats[k][1]]=predict_lr(m,feats[k][0]);scores[k]=s
    return scores
def lr_scores_frozen(chunks_d,model,agg,group):
    out={}
    for k,d in chunks_d.items():
        X,c=features(d,agg,group);s=np.full(len(d['w_len']),-np.inf);s[c]=predict_lr(model,X);out[k]=s
    return out

# ---------------- driver ----------------
def methods(chunks_d,labs,lams=(10.0,100.0,1000.0)):
    M=[('cand1 rank curve (votes, reference)',lambda d:score_cand1(d),False)]
    for q in (20,30,50,70):M.append((f'(1) cand1 votes from the top {q}% CE query tokens only',(lambda q:lambda d:score_surprisal_top(d,q))(q),False))
    M+= [('(1) cand1 votes weighted by token CE',score_surprisal_tokw,False),('(1) cand1 votes weighted by query-sentence CE',score_surprisal_sentw,False)]
    for a in (0.25,0.5,1.0):M.append((f'(3) cand1 votes + {a} x neighbour votes',(lambda a:lambda d:score_neighbour(d,a=a))(a),False))
    M.append(('(3) cand1 votes max-pooled over +-1 sentence',lambda d:score_neighbour(d,maxpool=True),False))
    res=[]
    for name,fn,lab in M:res.append((name,lab,curve({k:fn(d) for k,d in chunks_d.items()},chunks_d,labs)))
    for agg in ('sum','max'):
        for group in ('g3','all'):
            for lam in lams:res.append((f'(2) logistic regression on head x sentence mass ({agg} over queries, layers {group}, L2 {lam:g}), leave-one-chunk-out',True,curve(lr_scores_loco(chunks_d,labs,agg,group,lam),chunks_d,labs)))
    rng=np.random.default_rng(1);res.append(('chance (random candidate sentences)',False,curve({k:with_cand(rng.random(len(d['w_len'])),d) for k,d in chunks_d.items()},chunks_d,labs)))
    return res
def natural_points(chunks_d,labs):
    out=[]
    for name,fn in (('cand1 natural point (>= 1 vote)',natural_mask),('cand1 + dilation by 1 sentence',lambda d:post_dilate(natural_mask(d))),('cand1 + gap filling (hide a sentence between two hidden ones)',lambda d:post_gapfill(natural_mask(d)))):
        out.append((name,G.metrics(pooled_counts({k:fn(d) for k,d in chunks_d.items()},chunks_d,labs))))
    return out
def report(chunks,labs,out=None,save_coef=None):
    L=['# E55 32.12 token-selection extensions on the sentence dump (CPU; labels only in the metrics, except (2): LABELS USED IN TRAINING)','','Hide rule for all curves: per chunk the highest-scoring candidate sentences (>= 4 tokens) are hidden until the target share of the window tokens is reached; pooled over the dump chunks. Bar: lower mask fraction than candidate 1 at the same recall (hidden@R0.90 below candidate 1).','']
    for st in sorted(chunks):
        cd=chunks[st];res=methods(cd,labs);nat=natural_points(cd,labs);ks=sorted(cd)
        L+=[f'## state {st} (chunks {ks})','','Natural points (no rank cut):','','| rule | recall | precision | accuracy | F1 | mask fraction | chance precision |','|---|---|---|---|---|---|---|']
        for n,m in nat:L.append(f"| {n} | {m['recall']:.3f} | {m['precision']:.3f} | {m['acc']:.3f} | {m['f1']:.3f} | {m['hidden']:.3f} | {m['chance']:.3f} |")
        L+=['','Curves (summaries; hidden@R = mask fraction needed to reach recall R, interpolated; recall@h = recall at mask fraction h):','','| method | labels used | recall@0.61 | recall@0.45 | recall@0.35 | hidden@R0.90 | hidden@R0.80 | precision@0.35 | mean recall - chance |','|---|---|---|---|---|---|---|---|---|']
        for n,lab,cv in res:
            s=summarize(cv);L.append(f"| {n} | {'LABELS USED IN TRAINING' if lab else 'no'} | {s['recall@0.61']:.3f} | {s['recall@0.45']:.3f} | {s['recall@0.35']:.3f} | {s['hidden@R0.90']:.3f} | {s['hidden@R0.80']:.3f} | {s['precision@0.35']:.3f} | {s['mean_recall_minus_chance']:.3f} |")
        L.append('')
        Path(str(out).replace('.md',f'_{st}_curves.json')).write_text(json.dumps({n:[(f,m) for f,m in cv] for n,_,cv in res},default=float)) if out else None
    if save_coef:                                                    # frozen coefficients: trained on ALL chunks of each state (for other seeds' dumps; labels used in training)
        pk={}
        for st,cd in chunks.items():
            for agg in ('sum','max'):
                for group in ('g3','all'):
                    for lam in (10.0,100.0,1000.0):
                        fe={k:features(d,agg,group) for k,d in cd.items()};X=np.concatenate([fe[k][0] for k in cd]);y=np.concatenate([(np.asarray(labs[k]['w_fact'])[fe[k][1]]>=0).astype(float) for k in cd]);m=fit_lr(X,y,lam)
                        for nm in m:pk[f'{st}/{agg}/{group}/{lam:g}/{nm}']=m[nm]
        np.savez(save_coef,**pk);L.append(f'Frozen coefficients (trained on all chunks of this dump, labels used in training) saved to {save_coef}.')
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def apply_report(chunks,labs,coef,out=None):
    Z=np.load(coef);L=['# E55 32.12 frozen learned head weighting applied to another dump (LABELS USED IN TRAINING of the coefficients, none on this dump)','']
    for st in sorted(chunks):
        cd=chunks[st];L+=[f'## state {st} (chunks {sorted(cd)})','','| agg | layers | L2 | recall@0.61 | recall@0.45 | recall@0.35 | hidden@R0.90 | hidden@R0.80 |','|---|---|---|---|---|---|---|---|']
        ref=curve({k:score_cand1(d) for k,d in cd.items()},cd,labs);s=summarize(ref);L.append(f"| cand1 rank curve | - | - | {s['recall@0.61']:.3f} | {s['recall@0.45']:.3f} | {s['recall@0.35']:.3f} | {s['hidden@R0.90']:.3f} | {s['hidden@R0.80']:.3f} |")
        for agg in ('sum','max'):
            for group in ('g3','all'):
                for lam in (10.0,100.0,1000.0):
                    pre=f'{st}/{agg}/{group}/{lam:g}/'
                    if pre+'w' not in Z.files:continue
                    m={nm:Z[pre+nm] for nm in ('w','b','mu','sd')};s=summarize(curve(lr_scores_frozen(cd,m,agg,group),cd,labs))
                    L.append(f"| {agg} | {group} | {lam:g} | {s['recall@0.61']:.3f} | {s['recall@0.45']:.3f} | {s['recall@0.35']:.3f} | {s['hidden@R0.90']:.3f} | {s['hidden@R0.80']:.3f} |")
        L.append('')
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t

# ---------------- selftest (synthetic planted data) ----------------
def synth(seed=0,nW=80,nQ=14,nL=6,H=4,CH=512):
    rng=np.random.default_rng(seed);wl=rng.integers(5,20,nW);fact=np.zeros(nW,bool);fact[rng.choice(nW,nW//8,replace=False)]=True;fidx=np.nonzero(fact)[0]
    qlen=rng.integers(8,25,nQ);qst=np.cumsum(np.r_[0,qlen[:-1]]);Tm=rng.random((nL,H,CH))*0.05;Tw=rng.integers(0,nW,(nL,H,CH));AL=rng.random((nL,H,nQ,nW))*0.002;tce=rng.random(CH)*2
    for q in range(nQ):
        f_=fidx[q%len(fidx)];Tm[:,:2,qst[q]:qst[q]+qlen[q]]=0.5;Tw[:,:2,qst[q]:qst[q]+qlen[q]]=f_;AL[:,:2,q,f_]+=0.2;tce[qst[q]:qst[q]+qlen[q]//2]+=3
    d={'T_max':Tm.astype(np.float16),'T_wsid':Tw.astype(np.int16),'q_start':qst,'q_len':qlen,'q_tce':tce.astype(np.float16),'q_ce':rng.random(nQ).astype(np.float32),'w_len':wl,'w_start':np.cumsum(np.r_[0,wl[:-1]]),'w_chunk':np.zeros(nW,np.int32),'AL':AL.astype(np.float16)}
    return d,{'w_fact':np.where(fact,np.arange(nW)%16,-1).astype(np.int16)}
def selftest():
    cd={};labs={}
    for k in (1,2,3,4):cd[k],labs[k]=synth(k)
    chunks={'S':cd};np.random.seed(0)
    # candidate 1 on planted data: natural point finds the facts
    nat=dict(natural_points(cd,labs));m=nat['cand1 natural point (>= 1 vote)'];assert m['recall']>0.9 and m['precision']>0.9,m
    cv=curve({k:score_cand1(d) for k,d in cd.items()},cd,labs);s=summarize(cv);assert s['recall@0.35']>0.9 and s['hidden@R0.90']<0.4,s
    h=[m_['hidden'] for _,m_ in cv];r=[m_['recall'] for _,m_ in cv];assert all(b>=a-1e-9 for a,b in zip(r,r[1:])) and h[0]<h[-1]
    ch=curve({k:with_cand(np.random.default_rng(k).random(len(d['w_len'])),d) for k,d in cd.items()},cd,labs);sc=summarize(ch);assert sc['mean_recall_minus_chance']<0.1 and s['mean_recall_minus_chance']>0.15,(s,sc)
    # surprisal: top-CE tokens (planted: first half of each query) still carry the votes
    s1=summarize(curve({k:score_surprisal_top(d,50) for k,d in cd.items()},cd,labs));assert s1['recall@0.35']>0.8,s1
    # learned: leave-one-chunk-out must find the planted heads 0,1
    sl=summarize(curve(lr_scores_loco(cd,labs,'sum','all',10.0),cd,labs));assert sl['recall@0.35']>0.9,sl
    full=methods(cd,labs);assert len(full)>=20 and all(len(c)==len(GRID) for _,_,c in full)
    # frozen apply: coefficients trained on chunks 1-3 evaluated on chunk 4 only
    tr={k:cd[k] for k in (1,2,3)};fe={k:features(d,'sum','all') for k,d in tr.items()};X=np.concatenate([fe[k][0] for k in tr]);y=np.concatenate([(np.asarray(labs[k]['w_fact'])[fe[k][1]]>=0).astype(float) for k in tr]);mdl=fit_lr(X,y,10.0)
    sf=summarize(curve(lr_scores_frozen({4:cd[4]},mdl,'sum','all'),{4:cd[4]},labs));assert sf['recall@0.35']>0.9,sf
    assert post_dilate(np.array([0,1,0,0],bool)).tolist()==[True,True,True,False] and post_gapfill(np.array([1,0,1,0],bool)).tolist()==[True,True,True,False]
    t=report(chunks,labs);assert 'LABELS USED IN TRAINING' in t and '(1)' in t and '(3)' in t
    src=open(HERE/'e55_sentsel.py').read();assert 'np.load' not in src
    print('e55_sentgrid_ext selftest ok: cand1 hidden@R0.90 %.2f, chance mean recall-chance %.2f, LR recall@0.35 %.2f'%(s['hidden@R0.90'],sc['mean_recall_minus_chance'],sl['recall@0.35']))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--dump');ap.add_argument('--labels');ap.add_argument('--out');ap.add_argument('--save-coef');ap.add_argument('--apply-coef');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    chunks,lab=G.load(a.dump,a.labels)
    print(apply_report(chunks,lab,a.apply_coef,a.out) if a.apply_coef else report(chunks,lab,a.out,a.save_coef))
