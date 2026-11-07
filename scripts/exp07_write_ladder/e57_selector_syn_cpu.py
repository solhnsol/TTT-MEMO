"""E57 synthetic row-level selector from the last-prefix-layer hidden states (new file; CPU only; core-claim / GPM thread, Hans 11/07 '고고'): can a logistic regression on the W0 hidden state of a row (job 1180: hid_s*.npz) pick the rows that W0 g128 picks (a: g128>2 = ABS2 rows) or the TRUE deciding rows (b: rows predicting the first content token of the place, as in e57_g_mass_analysis)? Train worlds s62 (16), s71 (4), s72 (32); evaluation worlds s65 (16), s73 (4), s74 (32) (density = facts per chunk setting v3_<k>_1). L2 chosen by leave-one-world-out over the train worlds (mean AUC). Eligible rows only (as CM.rows_eligible). Per chunk, the selector takes the top-n rows by score at the same n as TOP5 (ceil(5% of eligible rows), W0 g ranking) and at the same n as ABS2 (number of rows with g128>2 in the chunk, W0 g ranking = ABS2 itself). Recall / precision of the true deciding rows per chunk (chunks with >=1 true row), chunk-level bootstrap (2000 draws), Spearman(score, g128) per chunk.
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python3 e57_selector_syn_cpu.py --out F.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024
TRAIN=((62,'v3_16_1'),(71,'v3_4_1'),(72,'v3_32_1'));EVAL=((65,'v3_16_1'),(73,'v3_4_1'),(74,'v3_32_1'));DENS={'v3_4_1':4,'v3_16_1':16,'v3_32_1':32}
LAMS=[1e-4,1e-3,1e-2,1e-1,1.0];NB=2000
def auc(y,s):
    y=np.asarray(y,bool);n1=y.sum();n0=(~y).sum()
    if n1==0 or n0==0:return float('nan')
    r=rankdata(s);return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0))
def fit_logreg(X,y,lam,maxiter=200):
    n,d=X.shape;y=y.astype(np.float32)
    def f(w):
        z=X@w[:-1].astype(np.float32)+np.float32(w[-1]);p=1/(1+np.exp(-z.astype(np.float64)));l=np.mean(np.logaddexp(0,z)-y*z)+0.5*lam*np.sum(w[:-1]**2)
        gr=np.concatenate([(X.T@(p-y).astype(np.float32)).astype(np.float64)/n+lam*w[:-1],[np.mean(p-y)]]);return l,gr
    return minimize(f,np.zeros(d+1),jac=True,method='L-BFGS-B',options=dict(maxiter=maxiter)).x
def score(w,X):return (X@w[:-1].astype(np.float32)).astype(np.float64)+w[-1]
def top_mask(s,n):
    m=np.zeros(len(s),bool)
    if n>0:m[np.argsort(-s,kind='stable')[:n]]=True
    return m
def sel_stats(m,t):
    h=(m&t).sum();return h/t.sum(),h/max(1,m.sum())
def spearman(a,b):
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return float('nan')
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def world(seed,cell):
    """per eligible row of the evaluated chunks: chunk id, g128, place-first-token flag, hidden state (float32)"""
    import e55_run as R55,e55_uv as UVm,e57_syn_pass as SP,e57_longcell,e57_dwg_analyze as DW,e57_pqa2_stream as S,e57_closure_masks as CM
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,arms=['NAT']);Sx=R55.streams(w,enc);ids,sha=Sx['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);Z=np.load(EXP/f'e57_syn_s{seed}/syn_s{seed}_nll128.npz');b=list(Z['bands']).index(128);tk=S.tokenizer()
    nch=len(ids)//CH;place=np.zeros(nch*CH+1,bool)
    for r in DW.locate_syn(tk,w,None):place[r['pos'][0]]=True
    D=np.load(EXP/f'e57_hidden_dump/hid_s{seed}.npz');assert (np.asarray(ids)[:nch*CH]==D['ids'][:nch*CH]).all() and str(D['stream_sha'])==str(sha),'stream mismatch'
    H=D['h'];ck=[];g=[];t=[];rows=[];ixs=[]
    for k in range(int(Z['chunks'][0]),int(Z['chunks'][1])):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;ix=np.nonzero(el)[0]
        ck.append(np.full(len(ix),k));ixs.append(ix);g.append((Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)[ix]);t.append(place[k*CH+ix+1]);rows.append(k*CH+ix)
    rows=np.concatenate(rows);u=H[rows].astype(np.uint32)<<16;X=u.view(np.float32)
    return dict(ix=np.concatenate(ixs),ck=np.concatenate(ck),g=np.concatenate(g),t=np.concatenate(t),X=X)
def cat(ws):return {k:np.concatenate([w[k] for w in ws]) for k in ('ck','g','t','X')}
def boot(vals,rng,nb=NB):
    """vals: dict name->per-chunk array (aligned); returns mean + CI of means and requested contrasts handled by caller"""
    n=len(next(iter(vals.values())));idx=rng.integers(0,n,(nb,n));return idx
def ci(x):
    x=np.asarray(x);return [float(np.mean(x)),float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]
def evaluate(models,Ev,rng):
    """Ev: world dict; models: {'a':w,'b':w}; per chunk rows"""
    X,g,t,ck=Ev['X'],Ev['g'],Ev['t'],Ev['ck'];sc={m:score(w,X) for m,w in models.items()};per=[]
    for k in np.unique(ck):
        s=ck==k;gg=g[s];tt=t[s];n=len(gg)
        if tt.sum()==0:continue
        n5=int(np.ceil(0.05*n));na=int((gg>2).sum());r={'n':n,'n5':n5,'nabs':na,'ntrue':int(tt.sum())}
        for nm,cnt in (('5',n5),('abs',na)):
            gm=top_mask(gg,cnt);r[f'g_{nm}_rec'],r[f'g_{nm}_prec']=sel_stats(gm,tt) if cnt>0 else (0.0,0.0)
            for m in sc:
                sm=top_mask(sc[m][s],cnt);r[f'{m}_{nm}_rec'],r[f'{m}_{nm}_prec']=sel_stats(sm,tt) if cnt>0 else (0.0,0.0)
                if cnt>0:r[f'{m}_{nm}_overlap_with_g']=float((sm&gm).sum()/cnt)
        for m in sc:r[f'{m}_spearman_g']=spearman(sc[m][s],gg)
        per.append(r)
    keys=[k for k in per[0] if k not in('n',)];A={k:np.array([p.get(k,np.nan) for p in per],float) for k in keys}
    idx=rng.integers(0,len(per),(NB,len(per)));res={'n_chunks_with_true_rows':len(per),'mean_true_rows_per_chunk':float(A['ntrue'].mean()),'mean_n5':float(A['n5'].mean()),'mean_nabs':float(A['nabs'].mean())}
    for k in keys:
        if k in('n5','nabs','ntrue'):continue
        res[k]=ci(np.nanmean(A[k][idx],1))
    for m in sc:
        for nm in ('5','abs'):
            d=A[f'{m}_{nm}_rec'][idx].mean(1)-A[f'g_{nm}_rec'][idx].mean(1);res[f'{m}_{nm}_rec_minus_g']=ci(d)
            res[f'{m}_{nm}_rec_ratio_to_g']=ci(A[f'{m}_{nm}_rec'][idx].mean(1)/np.maximum(A[f'g_{nm}_rec'][idx].mean(1),1e-9))
    res['auc_rows']={m:auc(t,sc[m]) for m in sc};res['auc_rows_g']=auc(t,g)
    a_lab=g>2;res['auc_a_labels']={m:auc(a_lab,sc[m]) for m in sc};res['n_rows']=int(len(g));res['rate_true']=float(t.mean());res['rate_g_gt2']=float(a_lab.mean())
    return res
def pick_lambda(Ws,key,log):
    out={}
    for lam in LAMS:
        au=[]
        for h in range(len(Ws)):
            tr=cat([w for i,w in enumerate(Ws) if i!=h]);mu=tr['X'].mean(0);sd=tr['X'].std(0)+1e-6;y=(tr['g']>2) if key=='a' else tr['t'];wv=fit_logreg((tr['X']-mu)/sd,y,lam)
            ho=Ws[h];yh=(ho['g']>2) if key=='a' else ho['t'];au.append(auc(yh,score(wv,(ho['X']-mu)/sd)))
        out[lam]=float(np.nanmean(au));log(f'  LOWO {key} lam={lam}: AUC {out[lam]:.4f} {au}')
    return max(out,key=out.get),out
def selftest():
    r=np.random.default_rng(0);n=3000;d=20;X=r.standard_normal((n,d)).astype(np.float32);y=(X[:,0]+.5*X[:,1]+r.standard_normal(n)*.5>1.5);w=fit_logreg(X,y,1e-3)
    assert auc(y,score(w,X))>0.9;g=r.standard_normal(100);t=np.zeros(100,bool);t[:5]=True;g[:5]+=5;m=top_mask(g,5);assert sel_stats(m,t)[0]==1.0;print('e57_selector_syn_cpu selftest ok')
def write_masks(models,Ev,worlds):
    """learned top-5% masks (n = ceil(5% of the eligible rows) per chunk, ranked by the selector score) for the GPU arms M3TLB (model b) / M3TLA (model a): keys qtLB_k / qtLA_k (packbits [128]) added to the world's rules npz -> e57_synq/syn_s<seed>_learned.npz"""
    for (s_,c_),e in zip(worlds,Ev):
        base=dict(np.load(EXP/f'e57_synq/syn_s{s_}_rules.npz'));sc={m:score(w,e['X']) for m,w in models.items()}
        for k in np.unique(e['ck']):
            sel=e['ck']==k;n5=int(np.ceil(0.05*sel.sum()))
            for m,nm in (('b','qtLB'),('a','qtLA')):
                mk=np.zeros(CH,bool);mk[e['ix'][sel][top_mask(sc[m][sel],n5)]]=True;base[f'{nm}_{int(k)}']=np.packbits(mk)
        np.savez(EXP/f'e57_synq/syn_s{s_}_learned.npz',**base);print('masks',s_,flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--masks-only',action='store_true');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    lg=lambda *x:print(*x,flush=True);rng=np.random.default_rng(0)
    Tr=[];[Tr.append(world(s,c)) or lg('train world',s,c,len(Tr[-1]['g']),'rows') for s,c in TRAIN];Ev=[];[Ev.append(world(s,c)) or lg('eval world',s,c,len(Ev[-1]['g']),'rows') for s,c in EVAL]
    res={'definitions':__doc__.split('\n')[0],'train_worlds':[f's{s}_{c}' for s,c in TRAIN],'eval_worlds':[f's{s}_{c}' for s,c in EVAL],'lambdas':LAMS,'lowo':{}}
    full=cat(Tr);mu=full['X'].mean(0);sd=full['X'].std(0)+1e-6;models={}
    for key in 'ab':
        if a.masks_only:
            lam=json.load(open(a.out))['lowo'][key]['chosen_lambda'];y=(full['g']>2) if key=='a' else full['t'];models[key]=fit_logreg((full['X']-mu)/sd,y,lam,maxiter=400);lg('final',key,lam);continue
        lam,tab=pick_lambda(Tr,key,lg);res['lowo'][key]={'chosen_lambda':lam,'auc_by_lambda':{str(k):v for k,v in tab.items()}};y=(full['g']>2) if key=='a' else full['t'];models[key]=fit_logreg((full['X']-mu)/sd,y,lam,maxiter=400);lg('final',key,lam)
    for e in Ev:e['X']=(e['X']-mu)/sd
    if a.masks_only:
        write_masks(models,Ev,EVAL);np.savez(EXP/'e57_synq/syn_selector_weights.npz',mu=mu,sd=sd,wa=models['a'],wb=models['b']);return
    res['models']='a = trained on g128>2 labels; b = trained on true deciding rows (place first token)'
    res['eval']={}
    for (s,c),e in zip(EVAL,Ev):res['eval'][f's{s}_{c}_density{DENS[c]}']=evaluate(models,e,rng);lg('eval',s,c,'done')
    for i,e in enumerate(Ev):e['ck']=e['ck']+1000*i
    res['eval']['pooled']=evaluate(models,cat(Ev),rng)
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));lg('wrote',a.out)
if __name__=='__main__':main()
