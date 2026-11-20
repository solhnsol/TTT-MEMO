"""1280c same-fact unique share: seed-paired means and 95% t-CI (n=30, seeds 101-130) of b - b_P (nat/token, unique = E_rec - recompute placebo) and M_RECOMP - M_RECOMP_PL (margin, nat/fact) from experiments/exp07_write_ladder/e57_1280_main/record_s*.json. New file; CPU; thread-server 11/20."""
import json,numpy as np
from scipy import stats
rs=[json.load(open(f'experiments/exp07_write_ladder/e57_1280_main/record_s{s}.json')) for s in range(101,131)]
print(list(rs[0])[:30])
A=lambda k:np.array([r[k] for r in rs],float)
def ci(d):
    n=len(d);m=d.mean();h=stats.t.ppf(.975,n-1)*d.std(ddof=1)/np.sqrt(n);return m,m-h,m+h,int((d>0).sum())
b,c,bP,MR,MP=A('b'),A('c'),A('b_P'),A('M_RECOMP'),A('M_RECOMP_PL')
for n,d in (('E_rec = b-c',b-c),('placebo = bP-c',bP-c),('unique = b-bP',b-bP),('M_RECOMP',MR),('M_RECOMP_PL',MP),('M_RECOMP-M_RECOMP_PL',MR-MP),('M_RECOMP-M_NATS',MR-A('M_NATS')),('M_RECOMP_PL-M_NATS',MP-A('M_NATS'))):print(f'{n:26s} {ci(d)[0]:.4f} [{ci(d)[1]:.4f},{ci(d)[2]:.4f}] pos {ci(d)[3]}/30')
