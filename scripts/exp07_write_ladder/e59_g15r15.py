"""G15 - R15 margin-level differences with seed-paired t intervals (CPU; new file; EXPLORATORY descriptive; thread-server 11/22; coordinator request): E2E synthetic v3_16_1 seeds 71..78, job 1308 arms NATB1024G15W8 (G15: loss weight 8 on the top-15% rows by g1024) and NATB1024R15W8 (R15: weight 8 on the same number of random rows), reference NATMASK1024 (M, band 1024), NATS (NAT) and W0; out-of-window margin (gold - best wrong, mean over the 16 facts) at exposure 16 / 32 / 64 / 128 read from the stored read files; per-seed paired differences, mean [95% t-CI], same-direction seeds, also top-1 / rank. python3 e59_g15r15.py"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e59_reagg as RA,e57_allprev_analyze as A
E7=RA.E7;S=list(range(71,79));n=len(S);t=stats.t.ppf(.975,n-1)
SRC={'NAT':A.SRC['NATS'],'M':A.SRC['NATMASK1024'],'G15':'e57_b1024s/E55_b1024s_NATB1024G15W8_s{s}.read.json','R15':'e57_b1024s/E55_b1024s_NATB1024R15W8_s{s}.read.json'}
def rd(arm,s,e):
    r=json.load(open(E7/SRC[arm].format(s=s)));return RA.vec(r['read'][str(e)]['out'],r['meta']),RA.vec(r['W0']['out'],r['meta'])
def ci(d):d=np.asarray(d);m=d.mean();h=t*d.std(ddof=1)/np.sqrt(n);return f'{m:7.3f} [{m-h:7.3f},{m+h:7.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{n}'
for mi,mn in enumerate(RA.MET):
    print(f'[{mn}] out-of-window, seeds 71-78 (rank: negative = better)')
    for e in (16,32,64,128):
        V={a:np.array([rd(a,s,e)[0][mi] for s in S]) for a in SRC};W0=np.array([rd('NAT',s,e)[1][mi] for s in S])
        print(f'  exposure {e:3d}: W0 {W0.mean():7.3f}  means',{a:round(float(V[a].mean()),3) for a in V})
        for x,y in (('G15','R15'),('G15','M'),('R15','M'),('G15','NAT'),('R15','NAT')):print(f'     {x:4s} - {y:4s} {ci(V[x]-V[y])}')
