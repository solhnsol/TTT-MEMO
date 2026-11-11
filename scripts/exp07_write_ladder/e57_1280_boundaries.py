"""E57 job 1280: sequential t boundaries (2 looks, n = 10 and 20), method v6 (CPU). Look 1 boundary is ANALYTIC: c1 = t_9^{-1}(1 - alpha_1) with the Lan-DeMets O'Brien-Fleming-type spending alpha_1 = A(0.5) = 2 (1 - Phi(z_(1-alpha/2) / sqrt(0.5))) (alpha = .05: alpha_1 = .005573; alpha = .025: alpha_1 = .001529). Look 2 boundary c2 is found by Monte Carlo ONLY at look 2, on the sets that did not cross at look 1, with the REAL sequential t statistics (the first 10 observations are shared by T10 and T20): c2 solves P(T10 <= c1, T20 > c2) = alpha - alpha_1. Data model: i.i.d. standard normal (estimated variance). Futility is ignored in the calibration (non-binding), alpha is not recycled, no sample-size re-estimation.
Report (run_report): numpy version, bit generator (PCG64 via default_rng), two independent RNG seeds, N sets per run, both boundary estimates and their difference, boundary MC standard error (quantile SE = sqrt(p(1-p)/N) / f(c2) with f the density of T20 at c2 estimated from a normal-theory histogram window), look-wise and cumulative type-I errors on independent draws (look 1 error = alpha_1 exactly by t_9; look 2 incremental; cumulative), and robustness (not a guarantee): t(5) data and a skewed centred lognormal(sigma = 1) data (both tails: lognormal = the upper tail of right-skewed data, lognormal_neg = the opposite tail).
  python3 e57_1280_boundaries.py --selftest | --report OUT.json [--N 10000000]"""
import sys,json,argparse,platform
import numpy as np
from scipy import stats
def spend(alpha,t):return float(2*(1-stats.norm.cdf(stats.norm.ppf(1-alpha/2)/np.sqrt(t))))
def look1(alpha):a1=spend(alpha,0.5);return a1,float(stats.t.ppf(1-a1,9))
def tstats(x):
    m10=x[:,:10].mean(1);s10=x[:,:10].std(1,ddof=1);m20=x.mean(1);s20=x.std(1,ddof=1);return m10/(s10/np.sqrt(10)),m20/(s20/np.sqrt(20))
def gen(rng,n,dist):
    if dist=='normal':return rng.normal(size=(n,20))
    if dist=='t5':return rng.standard_t(5,size=(n,20))
    if dist=='lognormal':
        z=rng.lognormal(0.0,1.0,size=(n,20));return z-np.exp(0.5)             # centred (mean of lognormal(0,1) = e^.5)
    if dist=='lognormal_neg':
        z=rng.lognormal(0.0,1.0,size=(n,20));return -(z-np.exp(0.5))           # mirror image: the opposite tail of the skewed data
    raise ValueError(dist)
def look2_bound(alpha,N,seed,batch=500_000):
    a1,c1=look1(alpha);rng=np.random.default_rng(seed);keep=[];batch=min(batch,N)
    for _ in range(N//batch):
        t10,t20=tstats(rng.normal(size=(batch,20)));keep.append(t20[t10<=c1].astype(np.float32))
    k=np.sort(np.concatenate(keep))[::-1];idx=int((alpha-a1)*N);c2=float(k[idx])
    return c2,len(k)
def boundary_se(alpha,N,c2):
    a1,_=look1(alpha);p=(alpha-a1);f=float(stats.t.pdf(c2,19));return float(np.sqrt(p*(1-p)/N)/f)
def error_rates(alpha,c1,c2,N,seed,dist='normal',batch=500_000):
    rng=np.random.default_rng(seed);r1=0;r2=0;batch=min(batch,N)
    for _ in range(N//batch):
        t10,t20=tstats(gen(rng,batch,dist));h1=t10>c1;h2=(~h1)&(t20>c2);r1+=int(h1.sum());r2+=int(h2.sum())
    return dict(look1=r1/N,look2_incremental=r2/N,cumulative=(r1+r2)/N)
FINAL_C2={0.05:1.7575,0.025:2.1093}         # v8 table = the 4-decimal means of the two 1e7 runs in the regenerated report (1.75748 / 2.10930)
REGISTERED_C2={0.05:1.757,0.025:2.109}      # the v6/v7 table values (mean of two 1e7 runs of the raw-data sampler)
def run_report(N=10_000_000,seeds=(7,12345)):
    out={'numpy':np.__version__,'scipy':__import__('scipy').__version__,'python':platform.python_version(),'bit_generator':'PCG64 (numpy default_rng)','N':N,'rng_seeds':list(seeds),'results':{}}
    for alpha in (0.05,0.025):
        a1,c1=look1(alpha);est=[recompute_look2(alpha,10,20,N=N,seed=s)['c2'] for s in seeds];c2=float(np.mean(est));se=boundary_se(alpha,N,c2)
        chk={d:error_rates(alpha,c1,c2,N,seed=987654 if d=='normal' else 555,dist=d) for d in ('normal','t5','lognormal','lognormal_neg')}
        ind=independent_alpha_check(alpha,c1,c2,N=N);ind_reg=independent_alpha_check(alpha,c1,REGISTERED_C2[alpha],N=N,seed=27182);extra=recompute_look2(alpha,10,20,N=2_000_000,seed=99)['c2']
        out['results'][str(alpha)]=dict(independent_alpha_check_new_c2=ind,independent_alpha_check_registered_c2=dict(c2=REGISTERED_C2[alpha],**ind_reg),c2_at_N_2e6_seed99=extra,alpha_1=a1,c1_analytic=c1,c2_estimates=est,c2_difference=float(abs(est[0]-est[1])),c2_used_mean=c2,c2_boundary_mc_se=se,binomial_se_alpha=float(np.sqrt(alpha*(1-alpha)/N)),error_check_independent_draws=chk)
    out['explanation_c2_1.757_vs_1.7553']='1.7553 was recompute_look2 at N = 2e6 (boundary MC SE about .0017); the registered 1.757 is the mean of two N = 1e7 runs (SE about .0008 each); the difference (.0017) is Monte Carlo noise, not a design difference (same alpha_1 = A(.5), same t_9 look 1, same 1:1 information fraction); see c2_at_N_2e6_seed99'
    out['look1_check']=look1_df_check();return out

# ---------------- v7 additions ----------------
def look1_df_check():
    """look 1 uses df = n1 - 1 = 9 (NOT 3): c1 = t_9^{-1}(1 - alpha_1) = 3.18198; t_3^{-1}(.975) = 3.18245 is a numerical coincidence (differs by .0005)"""
    a1,c1=look1(0.05);assert abs(c1-float(stats.t.ppf(1-a1,9)))<1e-12 and abs(c1-float(stats.t.ppf(0.975,3)))>1e-4;return dict(c1=c1,t3_975=float(stats.t.ppf(0.975,3)),df=9)
def look1_actual(alpha,n,n_planned=20):
    """look-1 boundary for actual n at look 1 out of n_planned final seeds: t_{n-1}^{-1}(1 - A(n/n_planned))"""
    return float(stats.t.ppf(1-spend(alpha,n/n_planned),n-1))
def _suff_t(rng,n1,n2,b):
    """exact simulation of the two t statistics under H0 from sufficient statistics (independent code path: standard normals and chi-squares instead of raw observations)"""
    nb=n2-n1;z1=rng.standard_normal(b);z2=rng.standard_normal(b);ssa=rng.chisquare(n1-1,b);ssb=rng.chisquare(nb-1,b)
    t1=z1/np.sqrt(ssa/(n1-1));tot=np.sqrt(n1)*z1+np.sqrt(nb)*z2;ss=ssa+ssb+z1**2+z2**2-tot**2/n2;t2=(tot/np.sqrt(n2))/np.sqrt(ss/(n2-1));return t1,t2
def recompute_look2(alpha,n1,n2,N=10_000_000,seed=7,batch=2_500_000):
    """look-2 boundary for the REAL information fraction t = n1/n2: alpha_1 = A(n1/n2), c1 = t_{n1-1}^{-1}(1 - alpha_1), c2 = the (alpha - alpha_1) upper quantile of t2 among the sets with t1 <= c1 (sufficient-statistic sampler, N rounded down to a multiple of the batch); also returns the boundary MC SE"""
    z=stats.norm.ppf(1-alpha/2);a1=float(2*(1-stats.norm.cdf(z/np.sqrt(n1/n2))));c1=float(stats.t.ppf(1-a1,n1-1));rng=np.random.default_rng(seed);batch=min(batch,N);keep=[]
    for _ in range(N//batch):
        t1,t2=_suff_t(rng,n1,n2,batch);keep.append(t2[t1<=c1].astype(np.float32))
    M=(N//batch)*batch;k=np.sort(np.concatenate(keep))[::-1];c2=float(k[int((alpha-a1)*M)]);pp=alpha-a1;se=float(np.sqrt(pp*(1-pp)/M)/stats.t.pdf(c2,n2-1))
    return dict(n1=n1,n2=n2,alpha=alpha,alpha_1=a1,c1=c1,c2=c2,c2_mc_se=se,N=M,seed=seed)
def independent_alpha_check(alpha,c1,c2,N=10_000_000,seed=31415,n1=10,n2=20,batch=2_500_000):
    """independent simulation (different RNG seed AND the sufficient-statistic sampler) of the cumulative one-sided type I error of the pair (c1, c2)"""
    rng=np.random.default_rng(seed);r1=r2=0;batch=min(batch,N)
    for _ in range(N//batch):
        t1,t2=_suff_t(rng,n1,n2,batch);h1=t1>c1;h2=(~h1)&(t2>c2);r1+=int(h1.sum());r2+=int(h2.sum())
    M=(N//batch)*batch;c=(r1+r2)/M;return dict(look1=r1/M,look2_incremental=r2/M,cumulative=c,mc_se=float(np.sqrt(c*(1-c)/M)),N=M,seed=seed,target=alpha)
def skew_dgp(kind,param,rng,n,m=20):
    """skewed / contaminated data generators for the sensitivity grid (all centred at mean 0): kind 'lognormal' with skewness gamma = param in {.5, 1, 2} (sigma solved from gamma = (e^{s^2}+2) sqrt(e^{s^2}-1)), kind 'contaminated' (90% N(0,1) + 10% N(0, 3^2)), kind 'mirror_lognormal' = the negative of the lognormal (opposite tail)"""
    def sig_for(g):
        lo,hi=0.01,3.0
        for _ in range(80):
            mid=(lo+hi)/2;gm=(np.exp(mid**2)+2)*np.sqrt(np.exp(mid**2)-1)
            if gm<g:lo=mid
            else:hi=mid
        return hi
    if kind in('lognormal','mirror_lognormal'):
        sg=sig_for(param);z=rng.lognormal(0.0,sg,size=(n,m))-np.exp(sg**2/2);return z if kind=='lognormal' else -z
    if kind=='contaminated':
        z=rng.normal(size=(n,m));mask=rng.random(size=(n,m))<0.10;return np.where(mask,3.0*z,z)
    raise ValueError(kind)
SKEW_GRID=(0.0,0.5,1.0,1.5,2.0)
def skew_grid(alpha,c1,c2,N=1_000_000,seed=2026,batch=250_000):
    """cumulative error of the t-based two-look pair (c1, c2) under mean-zero skewed data: skewness in SKEW_GRID, both tail directions ('upper' = the test's tail is the long tail, 'lower' = the opposite tail), look 1, incremental look 2, cumulative, each with its MC SE; skewness 0 = normal; the contaminated normal is added as an extra cell"""
    out={};cells=[(f'{d}_{g}',('lognormal' if d=='upper' else 'mirror_lognormal') if g>0 else 'normal',g) for g in SKEW_GRID for d in ('upper','lower') if not (g==0 and d=='lower')]+[('contaminated',"contaminated",0)]
    for name,kind,g in cells:
        rng=np.random.default_rng(seed);r1=0;r2=0;b=min(batch,N)
        for _ in range(N//b):
            x=rng.normal(size=(b,20)) if kind=='normal' else skew_dgp(kind,g,rng,b);t10,t20=tstats(x);h1=t10>c1;h2=(~h1)&(t20>c2);r1+=int(h1.sum());r2+=int(h2.sum())
        M=(N//b)*b;c=(r1+r2)/M;out[name]=dict(look1=r1/M,look2_incremental=r2/M,cumulative=c,mc_se=float(np.sqrt(max(c,1e-12)*(1-c)/M)),N=M)
    return out
def signflip_sequential(x,alpha,n1=10,direction='upper'):
    """exact sign-flip sequential test (sensitivity analysis; distribution free under symmetry): data x (length n2 = len(x)); conditional on |x| the 2^n2 sign vectors are enumerated. Look 1 rejects when the exact one-sided p-value of the first n1 observations is <= alpha_1 (alpha_1 = A(n1/n2)); look 2: the smallest threshold c2 on the sum statistic S2 = sum(s_i |x_i|) of all n2 observations such that P(look-1 event OR S2 >= c2) <= alpha, by exact enumeration over the 2^n2 sign vectors; the test rejects at look 2 if the observed S2 >= c2. Returns dict(look1_reject, look2_reject, p1, c2)"""
    x=np.asarray(x,float)*(1 if direction=='upper' else -1);n2=len(x);ax=np.abs(x);z=stats.norm.ppf(1-alpha/2);a1=float(2*(1-stats.norm.cdf(z/np.sqrt(n1/n2))))
    def enum(m):
        idx=np.arange(2**m,dtype=np.uint32);return ((idx[:,None]>>np.arange(m,dtype=np.uint32)[None,:])&1).astype(np.float32)*2-1
    S1=enum(n1);s1=S1@ax[:n1].astype(np.float32);obs1=float(x[:n1].sum());p1=float((s1>=obs1-1e-9).mean())
    if p1<=a1+1e-12:return dict(look1_reject=True,look2_reject=False,p1=p1,c2=None,alpha_1=a1)
    q1=np.sort(s1)[::-1][max(int(np.floor(a1*len(s1)+1e-9))-1,0)];ev1=s1>=q1-1e-9                  # look-1 rejection event as a function of the first n1 signs
    # all 2^n2 sign vectors: S2 = S1-part + remaining part; enumerate remaining signs
    n3=n2-n1;S3=enum(n3);s3=S3@ax[n1:].astype(np.float32);tot=len(s1)*len(s3);target=alpha*tot;cand=np.sort((s1[:,None]+s3[None,:]).ravel())[::-1]
    ok1=ev1.sum()*len(s3);rem=max(target-ok1,0)                                                          # budget left for 'not look-1 event and S2 >= c2'
    nonev=~ev1;vals=np.sort((s1[nonev][:,None]+s3[None,:]).ravel())[::-1];k=int(np.floor(rem+1e-9));c2=float(vals[k]) if k<len(vals) else float(vals[-1])+1.0
    obs2=float(x.sum());return dict(look1_reject=False,look2_reject=bool(obs2>c2),p1=p1,c2=c2,alpha_1=a1)

def selftest():
    a1,c1=look1(0.05);assert abs(a1-0.005573)<1e-5 and 3.17<c1<3.19,(a1,c1);a2,d1=look1(0.025);assert abs(a2-0.001529)<1e-5 and 3.9<d1<4.1
    c2,n=look2_bound(0.05,200_000,1);assert 1.6<c2<1.9,c2
    er=error_rates(0.05,c1,c2,200_000,2);assert abs(er['cumulative']-0.05)<0.005
    chk=look1_df_check();assert chk['df']==9
    r10=recompute_look2(0.05,10,20,N=1_000_000);assert abs(r10['c2']-c2)<0.02 and abs(r10['c1']-c1)<1e-9
    ia=independent_alpha_check(0.05,c1,r10['c2'],N=1_000_000);assert abs(ia['cumulative']-0.05)<4*ia['mc_se']+0.002,ia
    sg=skew_grid(0.05,c1,r10['c2'],N=250_000);assert set(sg)>={'upper_0.0','upper_2.0','lower_2.0','contaminated'} and sg['lower_2.0']['cumulative']>sg['upper_2.0']['cumulative']
    r=recompute_look2(0.05,8,16,N=200_000);assert 1.5<r['c2']<2.0 and r['c1']>3.0
    rng=np.random.default_rng(4);sk=skew_dgp('lognormal',1.0,rng,200_000,20);assert abs(sk.mean())<0.02 and 0.8<float(((sk-sk.mean())**3).mean()/sk.var()**1.5)<1.3
    rej=0;M=400
    for i in range(M):
        x=np.random.default_rng(100+i).normal(size=8);rej+=int(signflip_sequential(x,0.05,n1=4)['look1_reject'] or signflip_sequential(x,0.05,n1=4)['look2_reject'])
    assert rej/M<=0.05+3*np.sqrt(0.05*0.95/M),rej/M                                                    # exact sequential sign-flip is valid (conservative) on symmetric data
    print('e57_1280_boundaries selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--report');ap.add_argument('--N',type=int,default=10_000_000);ap.add_argument('--skew');ap.add_argument('--recompute',nargs=3,type=float,metavar=('N1','N2','ALPHA'));a=ap.parse_args()
    if a.selftest:selftest()
    elif a.recompute:
        n1,n2,al=a.recompute;print(json.dumps(recompute_look2(al,int(n1),int(n2),N=a.N if a.N<10_000_000 else 4_000_000),indent=1))
    elif a.skew:
        out={}
        for key,al in (('05',0.05),('025',0.025)):
            a1,c1=look1(al);c2=FINAL_C2[al];out[key]=dict(c1=c1,c2=c2,grid=skew_grid(al,c1,c2,N=min(a.N,1_000_000)))
        out['look1_df_check']=look1_df_check();out['note']='sensitivity grid: error rates of the t-based sequential boundaries under skewed/contaminated data (mean-zero nulls); lognormal skewness .5/1/2, mirror_lognormal = opposite tail';json.dump(out,open(a.skew,'w'),indent=1);print(json.dumps(out,indent=1))
    elif a.report:
        r=run_report(a.N);json.dump(r,open(a.report,'w'),indent=1);print(json.dumps(r,indent=1))
