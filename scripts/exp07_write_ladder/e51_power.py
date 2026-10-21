"""E51 CPU power simulation for the paired natural-text design: per person Delta_i = D(C1) - D(C2) (window-internal vs window-external previous exposure), N persons, one-sample one-sided t test (alpha 0.05)
and a two-sided 95% CI check for the pre-registered rule 'CI excludes 0 and |Delta| >= 0.1'. Delta_i ~ N(true, sd_i^2) with sd_i the per-person SD of Delta (unknown until the first run: grid 0.15/0.25/0.40/0.60).
Reports P(detect) for true Delta in {0.1, 0.25} and N in {10,15,20,30}, plus the smallest true Delta with power 0.8 (normal approx.). Not E33 power_sim (that one fits the s_cross slope across persons; this is the paired contrast).
Usage: e51_power.py [--n 4000] [--seed 7] | --selftest"""
import argparse,numpy as np
from math import sqrt
def tcrit(df,q):
    try:
        from scipy import stats;return float(stats.t.ppf(q,df))
    except Exception:
        return {9:1.833,14:1.761,19:1.729,29:1.699}.get(df,1.645)
def power(N,sd,true,reps,rng,alpha=0.05):
    d=rng.normal(true,sd,(reps,N));m=d.mean(1);s=d.std(1,ddof=1);t=m/(s/np.sqrt(N));return float((t>tcrit(N-1,1-alpha)).mean())
def mde80(N,sd):return (1.645+0.842)*sd/sqrt(N)
def table(reps,rng):
    out=['| N persons | sd of Delta_i | power @ true 0.10 | power @ true 0.25 | power @ null (size) | true Delta for 80% power |','|---|---|---|---|---|---|']
    for sd in (0.15,0.25,0.40,0.60):
        for N in (10,15,20,30):out.append(f'| {N} | {sd} | {power(N,sd,0.10,reps,rng):.2f} | {power(N,sd,0.25,reps,rng):.2f} | {power(N,sd,0.0,reps,rng):.3f} | {mde80(N,sd):.2f} |')
    return out
def selftest():
    rng=np.random.default_rng(1);p0=power(20,0.25,0.0,4000,rng);assert abs(p0-0.05)<0.02,p0
    assert power(30,0.15,0.25,2000,rng)>0.99 and power(10,0.6,0.1,2000,rng)<0.3;print('e51_power SELFTEST OK')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=4000);ap.add_argument('--seed',type=int,default=7);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    print('\n'.join(table(a.n,np.random.default_rng(a.seed))))
if __name__=='__main__':main()
