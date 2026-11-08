"""E57 row-level student of g' = nll_suffixband128 - nll_win from the W0 LAST-prefix-layer hidden state (new file; CPU only; GPM thread, Hans 11/08 'prefix 마지막이 맞다'): features = job 1200 / 1180 hidden states (frozen), target = the row g' of jobs 1211 (QA, news) / 1210 (synthetic), eligible rows only (as CM.rows_eligible; chunks k >= 8). Real text: train QA tune bundles 8/10/12 + news tune events (5), evaluate QA hold-out bundles 5/9 + news hold-out events (5) (row rows assigned to an event by the stream fragment); synthetic: train s62/s71/s72, evaluate s65/s73/s74 (per density). Models: ridge (closed form on standardized features) and a one-hidden-layer MLP (256 units, MSE, AdamW); the regularization (ridge lambda / MLP weight decay) is chosen by leave-one-unit-out (QA bundle, news event, synthetic world) cross-validation inside the training set by the mean per-chunk Spearman. Reported per evaluation source (chunk-level bootstrap, 2000 draws, 95% CI): Spearman(student, g') per chunk; top rows by the student at the same count as the g'>2 rows (ABS2 count) -> recall = precision of the g'>2 rows; at the 5% count (ceil 5% of the evaluated rows of the chunk) -> precision of the g'>2 rows and overlap with the top 5% by g'; the same quantities for TOP5 / ABS2 by the W0 g128 (reference); also against the g128>2 rows.
  python3 e57_student_cpu.py --out F.json | --selftest"""
import sys,json,argparse,pathlib,time
import numpy as np
from scipy.stats import rankdata
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;K0=8;NB=2000
import e57_closure_masks as CM
RIDGE_L=[1e1,1e2,1e3,1e4,1e5];MLP_WD=[1e-3,1e-1]
def bf16(u):return (u.astype(np.uint32)<<16).view(np.float32)
def spearman(a,b):
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return float('nan')
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def chunk_groups(ck):
    o=np.argsort(ck,kind='stable');s=ck[o];b=np.flatnonzero(np.diff(s))+1;return [o[i:j] for i,j in zip(np.r_[0,b],np.r_[b,len(s)])]
def mean_chunk_spearman(pred,y,ck):
    v=[spearman(pred[ix],y[ix]) for ix in chunk_groups(ck) if len(ix)>=50];return float(np.nanmean(v))
def top_mask(s,n):
    m=np.zeros(len(s),bool)
    if n>0:m[np.argsort(-s,kind='stable')[:n]]=True
    return m
def ci(x):
    x=np.asarray(x,float);return [float(np.nanmean(x)),float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5))]
# ---------------- data
def qa_rows(gid):
    Z=np.load(EXP/'e57_public_qa_n65/e57_dwg_qa_nll.npz');Y=np.load(EXP/'e57_suffixband/qa/e57_dwg_qa_nllsuf.npz');bi=list(Z['bands']).index(128);win=Z[f'g{gid}_win'];H=np.load(EXP/f'e57_hidden_dump_real/hid_qa_g{gid}.npz')['h'];ix_=[];ck=[];g=[];gp=[]
    for k in range(K0,win.shape[0]):
        lm=Z[f'g{gid}_lm'][k];ix=np.nonzero(CM.rows_eligible(lm,np.zeros(CH,np.int32))&lm)[0]
        if len(ix)<50:continue
        ix_.append(k*CH+ix);ck.append(np.full(len(ix),k));g.append((Z[f'g{gid}_band'][bi][k]-win[k])[ix]);gp.append((Y[f'g{gid}_suf'][0][k]-win[k])[ix])
    r=np.concatenate(ix_);return dict(X=bf16(H[r]),y=np.concatenate(gp).astype(np.float64),g=np.concatenate(g).astype(np.float64),ck=np.concatenate(ck),unit=np.full(len(r),gid))
def news_rows():
    import e55_uv as UVm,e57_t1t2v2_run as V,e57_news_m3 as NM
    path=EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl';ids=V.stream_ids(str(path),'A');Z=np.load(EXP/'e57_band/e57_band_W0.npz');Y=np.load(EXP/'e57_suffixband/news/E57_suffixband_W0.npz');b=list(Z['bands']).index(128);ends=UVm.sentence_ends(UVm.tokenizer(str(ROOT/'data/tokenizer/llama3/tokenizer.json')),ids)
    D=np.load(EXP/'e57_hidden_dump_real/hid_news_A.npz');H=D['h'];sp=json.load(open(EXP/'e57_news_split.json'));evs=sorted(set(sp['events_tune'])|set(sp['events_hold']));ev=np.full(len(ids),-1,int)
    for l in open(path):
        x=json.loads(l)
        if x.get('type')=='frag':ev[x['tok0']:x['tok0']+x['ntok']]=evs.index(x['event'])
    out=dict(X=[],y=[],g=[],ck=[],unit=[],part=[])
    for k in range(NM.K0,len(ids)//CH):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];ix=np.nonzero(CM.rows_eligible(lm,sid)&lm)[0];e=ev[k*CH+ix];ok=e>=0;ix=ix[ok];e=e[ok]
        out['X'].append(bf16(H[k*CH+ix]));out['y'].append((Y['nll_suffixband'][0][k]-Z['nll_win'][k])[ix]);out['g'].append((Z['nll_band'][b][k]-Z['nll_win'][k])[ix]);out['ck'].append(np.full(len(ix),k));out['unit'].append(e)
    r={k:np.concatenate(v) for k,v in out.items() if k!='part'};r['y']=r['y'].astype(np.float64);r['g']=r['g'].astype(np.float64);r['part']=np.where(np.isin(r['unit'],[evs.index(x) for x in sp['events_tune']]),0,1);return r
def syn_rows(seed,cell):
    import e55_run as R55,e55_uv as UVm,e57_syn_pass as SP,e57_longcell
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,arms=['NAT']);Sx=R55.streams(w,enc);ids,sha=Sx['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);Z=np.load(EXP/f'e57_syn_s{seed}/syn_s{seed}_nll128.npz');Y=np.load(EXP/f'e57_suffixband/s{seed}/syn_s{seed}_nllsuf.npz');b=list(Z['bands']).index(128);H=np.load(EXP/f'e57_hidden_dump/hid_s{seed}.npz')['h'];ix_=[];ck=[];g=[];gp=[]
    for k in range(int(Z['chunks'][0]),int(Z['chunks'][1])):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];ix=np.nonzero(CM.rows_eligible(lm,sid)&lm)[0];ix_.append(k*CH+ix);ck.append(np.full(len(ix),k));g.append((Z['nll_band'][b][k]-Z['nll_win'][k])[ix]);gp.append((Y['nll_suf'][0][k]-Y['nll_win'][k])[ix])
    r=np.concatenate(ix_);return dict(X=bf16(H[r]),y=np.concatenate(gp).astype(np.float64),g=np.concatenate(g).astype(np.float64),ck=np.concatenate(ck),unit=np.full(len(r),seed))
def cat(rs):return {k:np.concatenate([r[k] for r in rs]) for k in rs[0]}
# ---------------- models
class Ridge:
    def __init__(s,lam):s.lam=lam
    def fit_from_gram(s,G,b,n,ymu):
        d=G.shape[0];s.w=np.linalg.solve(G.astype(np.float64)+s.lam*n*np.eye(d)/1e3,b.astype(np.float64));s.ymu=ymu;return s
    def predict(s,X):return X@s.w.astype(np.float32)+s.ymu
def gram(X,y,ymu):
    Xf=np.ascontiguousarray(X);return (Xf.T@Xf).astype(np.float64),(Xf.T@(y-ymu).astype(np.float32)).astype(np.float64)
def mlp_fit(X,y,wd,epochs=4,hid=256,seed=0):
    import os;os.environ.setdefault('JAX_PLATFORMS','cpu');import jax,jax.numpy as jnp,optax
    k=jax.random.PRNGKey(seed);k1,k2=jax.random.split(k);d=X.shape[1];p={'w1':jax.random.normal(k1,(d,hid))*np.sqrt(2/d),'b1':jnp.zeros(hid),'w2':jax.random.normal(k2,(hid,1))*np.sqrt(1/hid),'b2':jnp.asarray([float(np.mean(y))])}
    def f(p,x):return (jax.nn.relu(x@p['w1']+p['b1'])@p['w2'])[:,0]+p['b2'][0]
    opt=optax.adamw(1e-3,weight_decay=wd);st=opt.init(p)
    @jax.jit
    def step(p,st,x,t):
        l,gr=jax.value_and_grad(lambda p:jnp.mean((f(p,x)-t)**2))(p);u,st=opt.update(gr,st,p);return optax.apply_updates(p,u),st,l
    rng=np.random.default_rng(seed);n=len(y);y32=y.astype(np.float32)
    for e in range(epochs):
        o=rng.permutation(n)
        for i in range(0,n-4095,4096):
            j=o[i:i+4096];p,st,_=step(p,st,jnp.asarray(X[j]),jnp.asarray(y32[j]))
    pj=jax.jit(f);return lambda Z:np.concatenate([np.asarray(pj(p,jnp.asarray(Z[i:i+65536]))) for i in range(0,len(Z),65536)]).astype(np.float64)
def standardize(tr_X,*others):
    mu=tr_X.mean(0);sd=tr_X.std(0)+1e-6;return [(x-mu)/sd for x in (tr_X,)+others]
def cv_choose(tr,units,log):
    """leave-one-unit-out on the training rows; returns best ridge lambda, best mlp wd and the tables"""
    mu=tr['X'].mean(0);sd=tr['X'].std(0)+1e-6;X=(tr['X']-mu)/sd;y=tr['y'];ymu=float(y.mean());G={};B={};N={}
    for u in units:
        m=tr['unit']==u;G[u],B[u]=gram(X[m],y[m],ymu);N[u]=int(m.sum())
    Gt=sum(G.values());Bt=sum(B.values());Nt=sum(N.values());rt={}
    for lam in RIDGE_L:
        sc=[]
        for u in units:
            m=tr['unit']==u;r=Ridge(lam).fit_from_gram(Gt-G[u],Bt-B[u],Nt-N[u],ymu);sc.append(mean_chunk_spearman(r.predict(X[m]),y[m],tr['ck'][m]))
        rt[lam]=float(np.mean(sc));log(f'  CV ridge lam={lam:g}: mean per-chunk Spearman {rt[lam]:.4f}')
    mt={}
    for wd in MLP_WD:
        sc=[]
        for u in units:
            m=tr['unit']==u;pr=mlp_fit(X[~m],y[~m],wd)(X[m]);sc.append(mean_chunk_spearman(pr,y[m],tr['ck'][m]))
        mt[wd]=float(np.mean(sc));log(f'  CV mlp wd={wd:g}: mean per-chunk Spearman {mt[wd]:.4f}')
    return max(rt,key=rt.get),max(mt,key=mt.get),rt,mt,mu,sd,ymu
def eval_rows(pred,y,g,ck,rng):
    """per chunk statistics over the evaluated rows (chunk-level bootstrap)"""
    P=[]
    for ix in chunk_groups(ck):
        if len(ix)<50:continue
        yy=y[ix];gg=g[ix];pp=pred[ix];n=len(ix);n5=int(np.ceil(0.05*n));A=yy>2;nA=int(A.sum());A2=gg>2;nA2=int(A2.sum())
        if nA==0 or nA2==0:continue
        r={'n5':n5,'n_gp2':nA,'n_g2':nA2,'spear':spearman(pp,yy)};ty=top_mask(yy,n5)
        for nm,s in (('student',pp),('g128',gg)):
            m5=top_mask(s,n5);r[f'{nm}_top5_prec_gp2']=(m5&A).sum()/n5;r[f'{nm}_top5_overlap_gp_top5']=(m5&ty).sum()/n5;r[f'{nm}_top5_prec_g2']=(m5&A2).sum()/n5
            ma=top_mask(s,nA);r[f'{nm}_gp2count_recall_gp2']=(ma&A).sum()/nA;ma2=top_mask(s,nA2);r[f'{nm}_g2count_recall_g2']=(ma2&A2).sum()/nA2
        r['rate_gp2']=A.mean();r['rate_g2']=A2.mean();P.append(r)
    keys=list(P[0]);Av={k:np.array([p[k] for p in P],float) for k in keys};idx=rng.integers(0,len(P),(NB,len(P)));res={'n_chunks':len(P)}
    for k in keys:res[k]=ci(np.nanmean(Av[k][idx],1))
    for nm in ('top5_prec_gp2','gp2count_recall_gp2'):
        res[f'student_over_g128_{nm}']=ci(np.nanmean(Av['student_'+nm][idx],1)/np.maximum(np.nanmean(Av['g128_'+nm][idx],1),1e-9))
    return res
def run_source(name,tr,units,ev_sets,rng,log):
    t0=time.monotonic();log(f'== {name}: train rows {len(tr["y"])}, units {units}')
    lam,wd,rt,mt,mu,sd,ymu=cv_choose(tr,units,log);X=(tr['X']-mu)/sd;y=tr['y'];G,Bv=gram(X,y,ymu);rid=Ridge(lam).fit_from_gram(G,Bv,len(y),ymu);mlp=mlp_fit(X,y,wd);out={'chosen_ridge_lambda':lam,'chosen_mlp_wd':wd,'cv_ridge':{str(k):v for k,v in rt.items()},'cv_mlp':{str(k):v for k,v in mt.items()},'train_rows':int(len(y)),'eval':{}}
    for en,ev in ev_sets.items():
        Xe=(ev['X']-mu)/sd;out['eval'][en]={'rows':int(len(ev['y'])),'ridge':eval_rows(rid.predict(Xe),ev['y'],ev['g'],ev['ck'],rng),'mlp':eval_rows(mlp(Xe),ev['y'],ev['g'],ev['ck'],rng)}
        log(f'  {en}: ridge Spearman {out["eval"][en]["ridge"]["spear"][0]:.3f}, mlp {out["eval"][en]["mlp"]["spear"][0]:.3f}')
    log(f'  {name} done in {time.monotonic()-t0:.0f}s');return out
def selftest():
    r=np.random.default_rng(0);X=r.standard_normal((2000,20)).astype(np.float32);y=X[:,0]*2+r.standard_normal(2000)*.3;m=Ridge(1.0).fit_from_gram(*gram(X,y,y.mean()),2000,y.mean());assert spearman(m.predict(X),y)>0.9
    ck=np.repeat(np.arange(20),100);s=eval_rows(m.predict(X),y*3,y*3,ck,r);assert s['spear'][0]>0.9;print('e57_student_cpu selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');ap.add_argument('--only',default='qa_news,synthetic');a=ap.parse_args()
    if a.selftest:selftest();return
    lg=lambda *x:print(*x,flush=True);rng=np.random.default_rng(0);res={'definitions':__doc__.split('\n')[0]}
    if 'qa_news' in a.only:
        Q={g:qa_rows(g) for g in (5,8,9,10,12)};N=news_rows();lg('rows loaded');nt={k:v[N['part']==0] for k,v in N.items()};nh={k:v[N['part']==1] for k,v in N.items()}
        tr=cat([Q[8],Q[10],Q[12],{k:nt[k] for k in ('X','y','g','ck')}|{'unit':100+nt['unit']}]);units=[8,10,12]+sorted(set(100+nt['unit']))
        res['real_text']=run_source('QA tune + news tune events',tr,units,{'qa_hold_5_9':cat([Q[5],Q[9]]),'news_hold_events':{k:nh[k] for k in ('X','y','g','ck','unit')}},rng,lg)
        del Q,N,tr
    if 'synthetic' in a.only:
        W={s:syn_rows(s,c) for s,c in ((62,'v3_16_1'),(71,'v3_4_1'),(72,'v3_32_1'),(65,'v3_16_1'),(73,'v3_4_1'),(74,'v3_32_1'))};tr=cat([W[62],W[71],W[72]])
        res['synthetic']=run_source('synthetic train worlds',tr,[62,71,72],{'s65_density16':W[65],'s73_density4':W[73],'s74_density32':W[74],'pooled_eval_3':cat([{**W[s],'ck':W[s]['ck']+1000*i} for i,s in enumerate((65,73,74))])},rng,lg)
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));lg('wrote',a.out)
if __name__=='__main__':main()
