"""Binding-accumulation formula, CPU check against stored runs (E2E, synthetic v3_16_1, seeds 71..78; new file; thread-server 11/19; EXPLORATORY; request: coordinator / thread 'bind accumulation as an equation'). ALL attempts are reported (variants V1, V2, the two seed splits, the all-seed in-sample fit).
MODEL (fixed before looking at predictions, 2 free parameters):  per exposure t = 1..E:   m_t = (1 - v_t) * M0_hid + v_t * M0_vis(t) + kappa * B_{t-1}  (margin the model has at write time on this sentence's answer);  p_t = 1 / (1 + 15 exp(-m_t))  (probability of the gold among 16 candidates when the 15 wrong candidates sit at the best-wrong level);  B_t = B_{t-1} + eta * (1 - p_t)  (cross-entropy gradient on the answer logit is proportional to 1 - p);  readout out of window after e exposures:  M_out(e) = M0_hid + kappa * B_e.  M0_hid = W0 out-of-window margin of the seed (no copy visible), M0_vis = W0 in-window margin at exposure 128 (in_g0@128, the only stored W0 in-window read: the real previous 7 chunks as context), constant in t, v_t = visibility of an earlier copy at exposure t of the arm: NATS 1 (a copy within the 8192 window; 0 at t = 1); band b: share of the world's facts whose nearest earlier repeat starts less than b tokens before this exposure (V1) / 1 - 1/(1 + number of earlier repeats within b tokens) (V2); NATALLPREV, NATORACLE, ISO-STD: 0 (no earlier copy of the fact visible in the write path).
FIT: eta, kappa on NATS and ISO-STD ONLY (squared error of the seed-mean curve M_out(e) - W0 over e = 16/32/64/128, many Nelder-Mead starts); then NO parameter is fitted for the other arms (NATMASK256, band 64 / 512 / 768 / 1024 / 2048 / 4096, NATALLPREV, NATORACLE). Splits: fit seeds 71-74 -> predict seeds 75-78 and the reverse; in-sample fit on all 8 (descriptive only). Main metric = M_out at exposure 128 (seed-mean over the test seeds) per arm: predicted vs observed, size error and arm order (Kendall tau). Also reported: whether a write-time fact-token loss log exists (diag in saved.json).
  python3 e59_binding_fit.py --out DIR | --selftest"""
import sys,json,math,itertools,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];EXP=ROOT/'experiments/exp07_write_ladder'
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
EXPS=(16,32,64,128);SEEDS=tuple(range(71,79))
ARMS={'NATS':('e57_k0conf16','E55_k0c16_NATS_s{s}.read.json',None),'ISO':('.','E55_q1a71_ISO-STD_s{s}.read.json',0),'NATMASK256':('e57_k0conf16','E55_k0c16_NATMASK_s{s}.read.json',256),
      'band64':('e57_band_dist2/w64','E55_bd64_NATMASK_s{s}.read.json',64),'band512':('e57_band_dist2/w512','E55_bd512_NATMASK_s{s}.read.json',512),'band768':('e57_band_dist2/w768','E55_bd768_NATMASK_s{s}.read.json',768),
      'band1024':('e57_band_dist2/w1024','E55_bd1024_NATMASK_s{s}.read.json',1024),'band2048':('e57_band_dist2/w2048','E55_bd2048_NATMASK_s{s}.read.json',2048),'band4096':('e57_band_dist2/w4096','E55_bd4096_NATMASK_s{s}.read.json',4096),
      'ALLPREV':('e57_allprev','E55_allprev_NATALLPREV_s{s}.read.json',0),'ORACLE':('e57_orc16','E55_orc16_NATORACLE_s{s}.read.json',0)}
FIT_ARMS=('NATS','ISO')
def mar(S,meta):
    x=np.asarray(S,float);g=np.asarray(meta['gi']);f=np.asarray(meta['kind'])=='fact';x=x[f];g=g[f];gold=x[np.arange(len(g)),g];w=x.copy();w[np.arange(len(g)),g]=-np.inf;return float((gold-w.max(1)).mean())
def load():
    D={}
    for arm,(d,pat,band) in ARMS.items():
        for s in SEEDS:
            r=json.load(open(EXP/d/pat.format(s=s)));m=r['meta']
            D[(arm,s)]=dict(out={e:mar(r['read'][str(e)]['out'],m) for e in EXPS},w0_out=mar(r['W0']['out'],m),w0_in=(mar(r['W0']['in_g0@128'],m) if 'in_g0@128' in r['W0'] else None))
    for (arm,s),d in D.items():
        if d['w0_in'] is None:d['w0_in']=D[('NATS',s)]['w0_in']      # the ISO files carry no in-window W0 read: the W0 of the seed is the same model, use the NATS file's value
    return D
def visibility(seed):
    """per band arm: v1[t], v2[t] for t = 0..127 (V1: share of facts with an earlier repeat starting < b tokens before; V2: mean of 1 - 1/(1+n))"""
    import e55_run as R55,e57_syn_pass as SP
    enc,w,tf=R55.build(SP.synth_args(seed,'v3_16_1'),seed,read_apps={16,32,64,128},arms=['NAT'])
    pos={}
    for r,c in enumerate(w['chunks']):
        for s in c['sents']:pos.setdefault(s['fact'],[]).append(r*1024+int(s['start']))
    V={}
    for arm,(d,pat,b) in ARMS.items():
        v1=np.zeros(128);v2=np.zeros(128)
        if b is None:v1[1:]=1.0;v2[1:]=1.0       # NATS: a copy inside the 8192 window at every exposure after the first
        elif b>0:
            for t in range(128):
                cnt=[sum(1 for q in v[:t] if v[t]-q<b) for v in pos.values()];v1[t]=np.mean([c>0 for c in cnt]);v2[t]=np.mean([1-1/(1+c) for c in cnt])
        V[arm]=(v1,v2)
    return V
def simulate(eta,kappa,v,m0h,m0v):      # m0v: float
    B=0.0;out={}
    for t in range(128):
        m=(1-v[t])*m0h+v[t]*m0v+kappa*B;p=1/(1+15*math.exp(-m));B+=eta*(1-p)
        if (t+1) in EXPS:out[t+1]=m0h+kappa*B
    return out
def predict(eta,kappa,D,V,arm,seeds,variant):
    r=[]
    for s in seeds:
        d=D[(arm,s)];o=simulate(eta,kappa,V[s][arm][variant],d['w0_out'],d['w0_in']);r.append([o[e] for e in EXPS])
    return np.mean(r,0)
def obs(D,arm,seeds):return np.mean([[D[(arm,s)]['out'][e] for e in EXPS] for s in seeds],0)
def fit(D,V,seeds,variant):
    from scipy.optimize import minimize
    def loss(th):
        eta,kappa=math.exp(th[0]),math.exp(th[1]);L=0.0
        for arm in FIT_ARMS:
            pr=predict(eta,kappa,D,V,arm,seeds,variant);ob=obs(D,arm,seeds);L+=float(np.sum((pr-ob)**2))
        return L
    best=None
    for a0,b0 in itertools.product((-4,-2,0,2),(-3,-1,1,2)):
        r=minimize(loss,[a0,b0],method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1e-9,maxiter=600))
        if best is None or r.fun<best.fun:best=r
    return math.exp(best.x[0]),math.exp(best.x[1]),best.fun
def kendall(x,y):
    n=len(x);c=d=0
    for i in range(n):
        for j in range(i+1,n):
            s=(x[i]-x[j])*(y[i]-y[j]);c+=s>0;d+=s<0
    return (c-d)/max(c+d,1)
def report(out):
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);D=load();V={s:visibility(s) for s in SEEDS};R={'model':__doc__.split('FIT:')[0].strip()[:1500],'arms':list(ARMS),'seeds':list(SEEDS)}
    R['observed_M_out_128_by_arm_seedmean']={a:float(obs(D,a,SEEDS)[-1]) for a in ARMS};R['W0_out_mean']=float(np.mean([D[('NATS',s)]['w0_out'] for s in SEEDS]));R['W0_in_128_mean']=float(np.mean([D[('NATS',s)]['w0_in'] for s in SEEDS]))
    R['visible_share_by_arm_exposure']={a:{str(e):float(np.mean([V[s][a][0][e-1] for s in SEEDS])) for e in EXPS} for a in ARMS}
    R['observed_curves']={a:{str(e):float(obs(D,a,SEEDS)[i]) for i,e in enumerate(EXPS)} for a in ARMS}
    splits={'fit71-74_test75-78':((71,72,73,74),(75,76,77,78)),'fit75-78_test71-74':((75,76,77,78),(71,72,73,74)),'insample_all8':(SEEDS,SEEDS)}
    R['attempts']={}
    for variant,vn in ((0,'V1_binary_any_visible'),(1,'V2_count_weighted')):
        for sn,(tr,te) in splits.items():
            eta,kappa,loss=fit(D,V,tr,variant);rows={};ob=[];pr=[]
            for a in ARMS:
                p_=predict(eta,kappa,D,V,a,te,variant)[-1];o_=float(obs(D,a,te)[-1]);rows[a]=dict(pred_128=float(p_),obs_128=o_,err=float(p_-o_));
                if a not in FIT_ARMS:ob.append(o_);pr.append(float(p_))
            nf=[a for a in ARMS if a not in FIT_ARMS]
            R['attempts'][f'{vn}|{sn}']=dict(eta=eta,kappa=kappa,fit_loss=loss,arms=rows,heldout_arms_rmse=float(np.sqrt(np.mean([(rows[a]['err'])**2 for a in nf]))),heldout_arms_mean_abs_err=float(np.mean([abs(rows[a]['err']) for a in nf])),kendall_tau_heldout_arms=kendall(pr,ob),n_heldout_arms=len(nf))
    # baseline for orientation: predict every non-fit arm with the mean of the two fit arms (no model)
    for sn,(tr,te) in splits.items():
        base=np.mean([obs(D,a,te)[-1] for a in FIT_ARMS]);nf=[a for a in ARMS if a not in FIT_ARMS];R['attempts'][f'baseline_mean_of_fit_arms|{sn}']=dict(heldout_arms_rmse=float(np.sqrt(np.mean([(base-obs(D,a,te)[-1])**2 for a in nf]))))
    # (4) write-time fact-token loss log
    r=json.load(open(EXP/'e57_k0conf16/E55_k0c16_NATS_s71.saved.json'));dg=r['diag'];R['write_time_fact_loss_log']=dict(exists=True,where='saved.json diag{chunk}: nat_nll_fact_first / nat_nll_fact_later / nat_nll_filler (and upd_*) at chunks '+str(sorted(int(k) for k in dg))[:120],granularity='mean nll of the fact tokens (name, occupation, place together) of the chunk, every 8th chunk; NOT the answer (place) tokens alone',n_logged_chunks=len(dg))
    json.dump(R,open(out/'binding_fit_result.json','w'),indent=1);return R
def selftest():
    o=simulate(0.3,1.0,np.zeros(128),-4.0,2.0);assert o[128]>o[16]>-4.0
    v=np.ones(128);o2=simulate(0.3,1.0,v,-4.0,3.0);assert o2[128]<o[128],'visible copy lowers the write'
    # closed-form check of the late behaviour: increments per doubling become constant (ln 2 / kappa) when p -> 0 for small margins is NOT used; just monotone + finite
    assert all(np.isfinite(list(o.values()))) and kendall([1,2,3],[1,2,3])==1 and kendall([1,2,3],[3,2,1])==-1
    print('e59_binding_fit selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');ap.add_argument('--v3',action='store_true');a,_=ap.parse_known_args()
    if a.selftest:selftest()
    elif a.v3:pass
    else:
        R=report(a.out);print(json.dumps({k:v for k,v in R.items() if k in('observed_M_out_128_by_arm_seedmean','W0_out_mean','W0_in_128_mean')},indent=1))
        for k,v in R['attempts'].items():print(k,{x:(round(y,3) if isinstance(y,float) else y) for x,y in v.items() if x!='arms'})
def v3_measured_p(out):
    """V3 (descriptive, NO fit): measured write-time fact-token loss -> p = exp(-nll) (geometric-mean token probability of the later-repeat fact tokens in the arm's own UPDATE forward, saved.json diag, every 8th chunk >= 8); S = 128 * mean_t (1 - p_t); compare S with the observed gain G = M_out(128) - W0 across arms (seed means): a constant G/S would support 'write amount = eta (1 - p)'. ISO has no diag (no stream forward)."""
    D=load();rows={}
    for arm,(d,pat,b) in ARMS.items():
        if arm=='ISO':continue
        S=[];G=[]
        for s in SEEDS:
            r=json.load(open(EXP/d/pat.format(s=s).replace('.read.json','.saved.json')));dg=r.get('diag',{})
            nl=[v['upd_nll_fact_later'] for k,v in sorted(dg.items(),key=lambda kv:int(kv[0])) if int(k)>=8 and v.get('upd_nll_fact_later') is not None]
            if not nl:continue
            S.append(128*float(np.mean([1-math.exp(-x) for x in nl])));G.append(D[(arm,s)]['out'][128]-D[(arm,s)]['w0_out'])
        if S:rows[arm]=dict(S=float(np.mean(S)),G=float(np.mean(G)),ratio=float(np.mean(G)/np.mean(S)),n_seeds=len(S),n_logged=len(nl))
    xs=np.array([r['S'] for r in rows.values()]);ys=np.array([r['G'] for r in rows.values()])
    return dict(rows=rows,pearson_S_G=float(np.corrcoef(xs,ys)[0,1]) if len(xs)>2 else None,kendall_S_G=kendall(list(xs),list(ys)))
if __name__=='__main__' and not any(a in sys.argv for a in('--selftest',)) and '--v3' in sys.argv:
    R=v3_measured_p(None);print(json.dumps(R,indent=1));json.dump(R,open(EXP/'binding_fit/binding_fit_V3_measured_p.json','w'),indent=1)
