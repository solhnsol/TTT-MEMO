"""Margin gain / top-1 / mean gold rank at exposure E (out-of-window), arm - NAT per seed, mean [95% t-CI], sign agreement; groups: fq (1323 forward), rev21 (1321 reverse), rev25 (1325 reverse, seeds 71-74, NAT/NATF/CHK/CHKF/ISO), qrev (1322 Qwen reverse). New file; EXPLORATORY descriptive. python3 e59_contrasts_all.py GROUP [--exp 128]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
X=Path(__file__).resolve().parents[2]/'experiments'
def met(S,m,W=None):
    x=np.asarray(S,float);g=np.asarray(m['gi'] if 'gi' in m else m['gold_idx']);f=np.asarray(m['kind'])=='fact';x=x[f];g=g[f];gold=x[np.arange(len(g)),g];w=x.copy();w[np.arange(len(g)),g]=-np.inf
    return np.array([(gold-w.max(1)).mean(),float((x>gold[:,None]).sum(1).__eq__(0).mean()),float((1+(x>gold[:,None]).sum(1)).mean())])
def rd_file(p,E):
    r=json.load(open(p));rd=r.get('read') or r.get('reads');return met(rd[str(E)]['out'],r['meta']),met(r['W0']['out'],r['meta'])
def rd_q(arm,s,E):
    r=json.load(open(X/f'exp09_qwen_ttt/q4way/e59q_{arm}_s{s}.json'));w=json.load(open(X/f'exp09_qwen_ttt/q4way/e59q_W0rev_s{s}.json'));w0=w['reads'][str(E)]['out'] if str(E) in w['reads'] else w['reads']['out']
    return met(r['reads_rev'][str(E)]['out'],r['meta_rev']),met(w0,r['meta_rev'])
E7=X/'exp07_write_ladder'
G={'fq':(range(211,219),{'NAT':lambda s,E:rd_file(E7/f'fq/run/E55_fq_NATS_s{s}.read.json',E),'BAND1024':lambda s,E:rd_file(E7/f'fq/run/E55_fq_NATMASK_s{s}.read.json',E),'CHK':lambda s,E:rd_file(E7/f'e59_fq/E59_fq_CHK_s{s}.read.json',E),'ISO':lambda s,E:rd_file(E7/f'e59_fq/E59_fq_ISO_s{s}.read.json',E)}),
   'rev21':(range(231,239),{a:(lambda a:lambda s,E:rd_file(E7/f'e59_4way/E59_a4_{a}_s{s}.rev.read.json',E))(a) for a in ('NAT','CHK','ISO','MASK1024')}),
   'rev25':(range(71,75),dict({a:(lambda a:lambda s,E:rd_file(E7/f'e59_rev/E59_e59_{a}_s{s}.rev.read.json',E))(a) for a in ('NAT','NATF','CHK','CHKF')},ISO=lambda s,E:rd_file(E7/f'e59_rev/iso/E59_e59_ISO_s{s}.rev.read.json',E))),
   'u12':(range(231,239),{a:(lambda a:lambda s,E:rd_file(E7/f'u12/E59_u12_{a}_s{s}.read.json',E))(a) for a in ('NAT','SAME','SAME_U1','SAME_U2','SAME_U1U2')}),
   'qrev':(range(221,229),{a:(lambda a:lambda s,E:rd_q(a,s,E))(a) for a in ('NAT','CHK','ISO','MASK1024')})}
def main():
    k=sys.argv[1];E=int(sys.argv[sys.argv.index('--exp')+1]) if '--exp' in sys.argv else 128;seeds,arms=G[k];seeds=list(seeds);V={a:[arms[a](s,E) for s in seeds] for a in arms};n=len(seeds);t=stats.t.ppf(.975,n-1)
    print(f'{k} exposure {E} out-of-window, n={n} seeds; chance top-1 0.0625, rank 8.5')
    for mi,name in enumerate(('margin gain (arm-W0)','top-1 (arm raw | arm-W0)','mean gold rank (arm raw | arm-W0, lower=better)')):
        print(f'[{name}]  W0 mean {np.mean([v[1][mi] for v in V["NAT"]]):.3f}')
        D={a:np.array([v[0][mi]-v[1][mi] for v in V[a]]) for a in arms};Rw={a:np.mean([v[0][mi] for v in V[a]]) for a in arms}
        for a in arms:h=t*D[a].std(ddof=1)/np.sqrt(n);print(f'  {a:9s} raw {Rw[a]:7.3f} | {D[a].mean():7.3f} [{D[a].mean()-h:7.3f},{D[a].mean()+h:7.3f}]'+('' if a=='NAT' else '   vs NAT: '+(lambda d:f'{d.mean():7.3f} [{d.mean()-t*d.std(ddof=1)/np.sqrt(n):7.3f},{d.mean()+t*d.std(ddof=1)/np.sqrt(n):7.3f}] sign {int(max((d>0).sum(),(d<0).sum()))}/{n}')(D[a]-D['NAT'])))
main()
