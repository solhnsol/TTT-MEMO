"""Placebo / share decomposition from stored reads (CPU; new file; EXPLORATORY descriptive; thread-server 11/20). (A) old E2E world v3_16_1 seeds 71-78 (1283/1302o/1307/1308z): margin at exposure E out-of-window for NATS, NATORACLE (same-fact earlier sentences hidden), NATOTHERPREV (other facts hidden only), NATRANDPREV (number-matched random keys), NATALLPREV. (B) new U12 world seeds 231-238 (1328): NAT, SAME, SAME_U1, SAME_U2, SAME_U1U2. Per-seed contrasts with 95% t-CI and same-direction count; ratios with a paired bootstrap over seeds (10000 resamples, 95% percentile). python3 e59_placebo.py [--exp 128]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e59_reagg as RA,e57_allprev_analyze as A,e57_prevctrl_analyze as PC
E7=RA.E7
def mar(r,e):
    return RA.vec(r['read'][str(e)]['out'],r['meta'])[0]
def ci(d):return RA.ci(d)
def boot(num,den,n=10000,seed=0):
    rng=np.random.default_rng(seed);k=len(num);x=[]
    for _ in range(n):
        i=rng.integers(0,k,k);dd=den[i].mean();x.append(num[i].mean()/dd if abs(dd)>1e-9 else np.nan)
    return float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5))
def show(name,d):m,l,h,s=ci(d);print(f'  {name:46s} {m:7.3f} [{l:7.3f},{h:7.3f}] {s}/{len(d)}')
def ratio(name,num,den):print(f'  {name:46s} {num.mean()/den.mean():7.3f} [{boot(num,den)[0]:7.3f},{boot(num,den)[1]:7.3f}] (ratio of means, bootstrap)')
def main():
    e=int(sys.argv[sys.argv.index('--exp')+1]) if '--exp' in sys.argv else 128
    src=dict(PC.SRC);S=range(71,79);G={}
    for a in ('NATS','NATORACLE','NATOTHERPREV','NATRANDPREV','NATALLPREV'):G[a]=np.array([mar(json.load(open(E7/src[a].format(s=s))),e) for s in S])
    print(f'(A) old world seeds 71-78, exposure {e} out-of-window margin; contrasts (mean [95% t] same-dir)')
    N=G['NATS']
    for a in G:
        if a!='NATS':show(f'{a} - NATS',G[a]-N)
    show('NATORACLE - NATOTHERPREV (same-fact unique vs other facts)',G['NATORACLE']-G['NATOTHERPREV'])
    show('NATALLPREV - NATORACLE - NATOTHERPREV + NATS (interaction)',G['NATALLPREV']-G['NATORACLE']-G['NATOTHERPREV']+N)
    ratio('ORACLE share of ALLPREV gain  (ORACLE-NATS)/(ALL-NATS)',G['NATORACLE']-N,G['NATALLPREV']-N);ratio('OTHERPREV share (OTHER-NATS)/(ALL-NATS)',G['NATOTHERPREV']-N,G['NATALLPREV']-N);ratio('RANDPREV share (RAND-NATS)/(ALL-NATS)',G['NATRANDPREV']-N,G['NATALLPREV']-N)
    ratio('ALLPREV-ORACLE over ALLPREV-NATS',G['NATALLPREV']-G['NATORACLE'],G['NATALLPREV']-N)
    S2=range(231,239);U={a:np.array([mar(json.load(open(E7/f'u12/E59_u12_{a}_s{s}.read.json')),e) for s in S2]) for a in ('NAT','SAME','SAME_U1','SAME_U2','SAME_U1U2')};N2=U['NAT']
    print(f'(B) U12 world seeds 231-238, exposure {e}')
    for a in U:
        if a!='NAT':show(f'{a} - NAT',U[a]-N2)
    show('SAME_U1 - SAME',U['SAME_U1']-U['SAME']);show('SAME_U2 - SAME',U['SAME_U2']-U['SAME']);show('SAME_U1U2 - SAME',U['SAME_U1U2']-U['SAME']);show('interaction U1U2-U1-U2+SAME',U['SAME_U1U2']-U['SAME_U1']-U['SAME_U2']+U['SAME'])
    ratio('SAME share of U1U2 gain (SAME-NAT)/(U1U2-NAT)',U['SAME']-N2,U['SAME_U1U2']-N2);ratio('(U1U2-SAME)/(U1U2-NAT)',U['SAME_U1U2']-U['SAME'],U['SAME_U1U2']-N2)
    ratio('(SAME_U1-SAME)/(SAME-NAT)',U['SAME_U1']-U['SAME'],U['SAME']-N2);ratio('(SAME_U2-SAME)/(SAME-NAT)',U['SAME_U2']-U['SAME'],U['SAME']-N2)
if __name__=='__main__':main()
