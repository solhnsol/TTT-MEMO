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
def run_report(N=10_000_000,seeds=(7,12345)):
    out={'numpy':np.__version__,'scipy':__import__('scipy').__version__,'python':platform.python_version(),'bit_generator':'PCG64 (numpy default_rng)','N':N,'rng_seeds':list(seeds),'results':{}}
    for alpha in (0.05,0.025):
        a1,c1=look1(alpha);est=[look2_bound(alpha,N,s)[0] for s in seeds];c2=float(np.mean(est));se=boundary_se(alpha,N,c2)
        chk={d:error_rates(alpha,c1,c2,N,seed=987654 if d=='normal' else 555,dist=d) for d in ('normal','t5','lognormal','lognormal_neg')}
        out['results'][str(alpha)]=dict(alpha_1=a1,c1_analytic=c1,c2_estimates=est,c2_difference=float(abs(est[0]-est[1])),c2_used_mean=c2,c2_boundary_mc_se=se,binomial_se_alpha=float(np.sqrt(alpha*(1-alpha)/N)),error_check_independent_draws=chk)
    return out
def selftest():
    a1,c1=look1(0.05);assert abs(a1-0.005573)<1e-5 and 3.17<c1<3.19,(a1,c1);a2,d1=look1(0.025);assert abs(a2-0.001529)<1e-5 and 3.9<d1<4.1
    c2,n=look2_bound(0.05,200_000,1);assert 1.6<c2<1.9,c2
    er=error_rates(0.05,c1,c2,200_000,2);assert abs(er['cumulative']-0.05)<0.005
    print('e57_1280_boundaries selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--report');ap.add_argument('--N',type=int,default=10_000_000);a=ap.parse_args()
    if a.selftest:selftest()
    elif a.report:
        r=run_report(a.N);json.dump(r,open(a.report,'w'),indent=1);print(json.dumps(r,indent=1))
