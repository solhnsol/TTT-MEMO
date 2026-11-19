"""Per-seed contrasts (arm - NAT) of the exposure-E out-of-window margin gain (gold - best wrong candidate, mean over fact items, minus W0 margin); mean, 95% t-CI over seeds, sign agreement. New file; EXPLORATORY descriptive. python3 e59_contrasts.py {e2e|qwen|fqqwen} [--exp 128]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
R=Path(__file__).resolve().parents[2]/'experiments'
def mar(S,m):
    x=np.asarray(S,float);g=np.asarray(m['gi'] if 'gi' in m else m['gold_idx']);f=np.asarray(m['kind'])=='fact';x=x[f];g=g[f];gold=x[np.arange(len(g)),g];w=x.copy();w[np.arange(len(g)),g]=-np.inf;return float((gold-w.max(1)).mean())
def load_e2e(arm,s):
    r=json.load(open(R/f'exp07_write_ladder/e59_4way/E59_a4_{arm}_s{s}.read.json'));m=r['meta'];return {e:mar(r['read'][str(e)]['out'],m) for e in (16,32,64,128)},mar(r['W0']['out'],m)
def load_q(d,arm,s,w0cache={}):
    r=json.load(open(R/f'exp09_qwen_ttt/{d}/e59q_{arm}_s{s}.json'));m=r['meta'];w=json.load(open(R/f'exp09_qwen_ttt/{d}/cband_W0_s{s}.json'))
    w0=w['reads']['out'] if 'out' in w['reads'] else list(w['reads'].values())[0]['out'];return {int(e):mar(v['out'],m) for e,v in r['reads'].items()},mar(w0,m)
def main():
    kind=sys.argv[1];E=int(sys.argv[sys.argv.index('--exp')+1]) if '--exp' in sys.argv else 128
    cfg={'e2e':(range(231,239),['NAT','CHK','ISO','MASK1024'],load_e2e),'qwen':(range(221,229),['NAT','CHK','ISO','MASK1024'],lambda a,s:load_q('q4way',a,s)),'fqqwen':(range(211,219),['NAT','CHK','ISO','BAND1024','BANDALL1024'],lambda a,s:load_q('fq_qwen',a,s))}[kind]
    seeds,arms,ld=cfg;G={}
    for a in arms:
        G[a]=np.array([ (lambda o,w:o[E]-w)(*ld(a,s)) for s in seeds])
    n=len(G['NAT']);t=stats.t.ppf(.975,n-1)
    print(f'{kind} exposure {E}: n={n} seeds; gain over W0 (mean [95% t-CI])')
    for a in arms:
        g=G[a];h=t*g.std(ddof=1)/np.sqrt(n);print(f'  {a:10s} gain {g.mean():6.2f} [{g.mean()-h:6.2f},{g.mean()+h:6.2f}]')
    print('  contrast arm - NAT (mean [95% t-CI], seeds same sign)')
    for a in arms[1:]:
        d=G[a]-G['NAT'];h=t*d.std(ddof=1)/np.sqrt(n);print(f'  {a:10s} {d.mean():6.2f} [{d.mean()-h:6.2f},{d.mean()+h:6.2f}]  sign {int(max((d>0).sum(),(d<0).sum()))}/{n} ({"+" if (d>0).sum()>=(d<0).sum() else "-"})')
    if 'CHK' in G and 'ISO' in G:
        for a,b in (('NAT','MASK1024'),('ISO','CHK')):
            if a in G and b in G:d=G[a]-G[b];h=t*d.std(ddof=1)/np.sqrt(n);print(f'  {a}-{b:9s} {d.mean():6.2f} [{d.mean()-h:6.2f},{d.mean()+h:6.2f}]')
    print('  per-seed NAT gain',np.round(G['NAT'],1).tolist())
main()
