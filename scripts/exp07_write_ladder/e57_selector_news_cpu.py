"""E57 news sentence-level selector feasibility (new file; CPU only; core-claim / GPM thread, Hans 11/07 '해봐'): can the head-wise attention statistics of the query sentences (prefix 24 layers, job 278 dump e57_attdump, W0) predict (a) whether a write-chunk sentence contains a row with W0 g128 > 2 nats (g = nll_band128 - nll_win of e57_band_W0.npz, eligible rows, row i predicts token i+1; aggregated to the sentence) and (b) whether the sentence is an ORCA sentence (overlaps, by >= 50% of its tokens, a fragment hidden by the ORC2 rule of e57_orca2_masks.npz, rank 7 = ORC2)? Sentences of chunks 8..279 are assigned to the events of the dump (q_event); split by event: train = the 5 tune events of e57_news_split.json, test = the 5 hold-out events. Features (label-free): per head (24 x 32) the attention mass of the sentence on FAR window sentences (window chunk <= k-2), plus per layer (24) the head mean of the mass on the previous chunk (k-1), on earlier sentences of the current chunk (AL_chunk), the head maximum of the far mass and the far-mass entropy over window sentences, plus sentence length and its relative position in the chunk. Model: L2 logistic regression (L-BFGS, standardized features, penalty by leave-one-event-out CV on the training events); a gradient-boosting model is not used. Baselines on the test sentences at the SAME sentence share per chunk: the W0 g sentence sets ABS2 (sentence has a row with g > 2) and TOP5 (sentence contains one of the top 5% rows of the chunk), and random. Reported: recall / precision of each target, selector top-n versus the baselines, Spearman rank correlation of the selector score with the W0 sentence score (max g) per chunk, AUC; bootstrap CIs over test chunks and the per-event values.
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python3 e57_selector_news_cpu.py --out F.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
from scipy.optimize import minimize
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024
import e57_closure_masks as CM
def auc(y,s):
    y=np.asarray(y,bool);s=np.asarray(s,float);n1=y.sum();n0=(~y).sum()
    if n1==0 or n0==0:return float('nan')
    r=np.argsort(np.argsort(s,kind='stable'),kind='stable')+1.0
    # average ranks for ties
    from scipy.stats import rankdata
    r=rankdata(s);return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0))
def fit_logreg(X,y,lam):
    n,d=X.shape
    def f(w):
        z=X@w[:-1]+w[-1];p=1/(1+np.exp(-z));ll=np.mean(np.logaddexp(0,z)-y*z)+0.5*lam*np.sum(w[:-1]**2);g=np.concatenate([X.T@(p-y)/n+lam*w[:-1],[np.mean(p-y)]]);return ll,g
    r=minimize(f,np.zeros(d+1),jac=True,method='L-BFGS-B',options=dict(maxiter=300));return r.x
def predict(w,X):return X@w[:-1]+w[-1]
def spearman(a,b):
    from scipy.stats import rankdata
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return float('nan')
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def sel_top(score,n):
    s=np.zeros(len(score),bool)
    if n>0:s[np.argsort(-score,kind='stable')[:n]]=True
    return s
def selftest():
    r=np.random.default_rng(0);X=r.standard_normal((400,5));y=(X[:,0]+0.3*r.standard_normal(400)>0).astype(float);w=fit_logreg(X,y,1e-3);assert auc(y>0,predict(w,X))>0.9
    assert abs(spearman(np.arange(10.),np.arange(10.)*2)-1)<1e-9 and sel_top(np.array([1.,3,2]),2).tolist()==[False,True,True];print('e57_selector_news_cpu selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e57_attdump_run as AR,e57_news_m3 as NM
    man=json.load(open(EXP/'e57_attdump/e57_attdump_manifest.json'));events=man['events'];sp=json.load(open(EXP/'e57_news_split.json'));ev_tune={events.index(e) for e in sp['events_tune']};ev_hold={events.index(e) for e in sp['events_hold']}
    Z=np.load(EXP/'e57_band/e57_band_W0.npz');b=list(Z['bands']).index(128);O=np.load(EXP/'e57_orca2/e57_orca2_masks.npz');rows=[]
    for k in range(max(8,NM.K0),280):
        d=AR.load_chunk(EXP/'e57_attdump',k);qs=d['q_start'].astype(int);ql=d['q_len'].astype(int);nQ=len(qs);qe=d['q_event'].astype(int)
        lm=np.asarray(Z['lm'][k],bool);sid=np.full(CH,-1)
        for i in range(nQ):sid[qs[i]:qs[i]+ql[i]]=i
        el=CM.rows_eligible(lm,sid)&lm;g=(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)
        tok_sid=np.full(CH+1,-1);tok_sid[:CH]=sid                 # row i predicts token i+1 -> sentence of token i+1
        rs=np.array([sid[i+1] if i+1<CH else -1 for i in range(CH)])
        maxg=np.full(nQ,-np.inf);hasg2=np.zeros(nQ,bool);top5=np.zeros(nQ,bool);n_el=np.zeros(nQ,int)
        idx=np.where(el&(rs>=0))[0];top_rows=set(idx[np.argsort(-g[idx])[:int(round(.05*len(idx)))]].tolist()) if len(idx) else set()
        for i in idx:
            s_=rs[i];n_el[s_]+=1;maxg[s_]=max(maxg[s_],g[i]);hasg2[s_]|=g[i]>2;top5[s_]|=(i in top_rows)
        # ORCA: sentences overlapping an ORC2-hidden v2 fragment
        hid=np.asarray(O[f'ch7_{k}'],bool);cl=np.asarray(O[f'clen{k}'],int);st=np.concatenate([[0],np.cumsum(cl)[:-1]]);orca=np.zeros(nQ,bool)
        for s_ in range(nQ):
            lo,hi=qs[s_],qs[s_]+ql[s_];ov=0
            for j in np.where(hid)[0]:ov+=max(0,min(hi,st[j]+cl[j])-max(lo,st[j]))
            orca[s_]=ov>=0.5*(hi-lo)
        AL=d['AL'].astype(np.float32);AC=d['AL_chunk'].astype(np.float32);wc=d['w_chunk'].astype(int)
        far=wc<=k-2;near=wc==k-1;farm=AL[:,:,:,far].sum(-1);nearm=AL[:,:,:,near].sum(-1);chm=AC.sum(-1)
        fw=AL[:,:,:,far];fwn=fw/np.maximum(fw.sum(-1,keepdims=True),1e-9);ent=-(fwn*np.log(fwn+1e-9)).sum(-1)
        for s_ in range(nQ):
            if n_el[s_]==0 or qe[s_]<0:continue
            feat=np.concatenate([farm[:,:,s_].reshape(-1),nearm[:,:,s_].mean(1),chm[:,:,s_].mean(1),farm[:,:,s_].max(1),ent[:,:,s_].mean(1),[ql[s_],qs[s_]/CH]])
            rows.append(dict(k=k,s=s_,ev=int(qe[s_]),x=feat.astype(np.float32),hasg2=bool(hasg2[s_]),top5=bool(top5[s_]),orca=bool(orca[s_]),maxg=float(maxg[s_]),n_el=int(n_el[s_])))
        if k%40==0:print('chunk',k,'sentences so far',len(rows),flush=True)
    X=np.stack([r['x'] for r in rows]);ev=np.array([r['ev'] for r in rows]);ks=np.array([r['k'] for r in rows]);tr=np.isin(ev,list(ev_tune));te=np.isin(ev,list(ev_hold))
    mu=X[tr].mean(0);sd=X[tr].std(0)+1e-6;Xs=(X-mu)/sd;res={'n_sentences':len(rows),'n_train':int(tr.sum()),'n_test':int(te.sum()),'events_tune':sp['events_tune'],'events_hold':sp['events_hold'],'targets':{}}
    for tname,lab in (('has_g2_sentence',np.array([r['hasg2'] for r in rows])),('ORCA_sentence',np.array([r['orca'] for r in rows]))):
        y=lab.astype(float);best=None;cv={}
        for lam in (1e-3,1e-2,1e-1,1.0):                    # leave-one-event-out on the training events
            aucs=[]
            for e in sorted(ev_tune):
                m_tr=tr&(ev!=e);m_va=tr&(ev==e)
                if y[m_va].sum()==0 or y[m_va].sum()==m_va.sum():continue
                w=fit_logreg(Xs[m_tr],y[m_tr],lam);aucs.append(auc(y[m_va]>0,predict(w,Xs[m_va])))
            cv[str(lam)]=float(np.nanmean(aucs))
            if best is None or cv[str(lam)]>best[1]:best=(lam,cv[str(lam)])
        w=fit_logreg(Xs[tr],y[tr],best[0]);sc=predict(w,Xs);t_=dict(base_rate_train=float(y[tr].mean()),base_rate_test=float(y[te].mean()),cv_auc_by_lambda=cv,penalty_lambda=best[0],auc_test=auc(y[te]>0,sc[te]),auc_train=auc(y[tr]>0,sc[tr]),by_event={})
        # same-share comparison per test chunk
        R={sh:{m:{'rec':[],'pre':[]} for m in ('selector','W0_g','random')} for sh in ('ABS2','TOP5')};sp_=[];rng=np.random.default_rng(0)
        for k in sorted(set(ks[te])):
            ids=np.where(te&(ks==k))[0];yk=y[ids]>0
            if yk.sum()==0:continue
            ha=np.array([rows[i]['hasg2'] for i in ids]);t5=np.array([rows[i]['top5'] for i in ids]);mg=np.array([rows[i]['maxg'] for i in ids]);s_=sc[ids];sp_.append(spearman(s_,mg))
            for sh,base in (('ABS2',ha),('TOP5',t5)):
                n_=int(base.sum());S={'selector':sel_top(s_,n_),'W0_g':base,'random':sel_top(rng.random(len(ids)),n_)}
                for m,Sm in S.items():R[sh][m]['rec'].append((yk&Sm).sum()/yk.sum());R[sh][m]['pre'].append((yk&Sm).sum()/max(Sm.sum(),1))
        def ci(x):
            x=np.asarray(x,float);x=x[~np.isnan(x)];bs=[np.mean(np.random.default_rng(i).choice(x,len(x))) for i in range(1000)];return dict(mean=float(x.mean()),ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],n=len(x))
        for sh in R:t_[f'same_share_as_{sh}_sentences']={m:dict(recall=ci(v['rec']),precision=ci(v['pre'])) for m,v in R[sh].items()}
        t_['spearman_selector_vs_W0_sentence_max_g']=ci(sp_)
        for e in sorted(ev_hold):
            mm=te&(ev==e);t_['by_event'][events[e]]=dict(n=int(mm.sum()),base_rate=float(y[mm].mean()),auc=auc(y[mm]>0,sc[mm]))
        res['targets'][tname]=t_;print(tname,'AUC test',round(t_['auc_test'],3),'lambda',best[0],flush=True)
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out)
if __name__=='__main__':main()
