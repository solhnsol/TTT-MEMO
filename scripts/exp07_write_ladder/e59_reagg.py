"""CPU re-aggregation of all new-seed results in three references (absolute, vs W0 = before writing, vs NAT) x three metrics (margin = gold - best wrong, top-1, mean gold rank of 16) x exposures 16/32/64/128, out-of-window; mean over seeds, 95% t interval, number of seeds with the sign of the mean (same direction). New file; EXPLORATORY descriptive; thread-server 11/20 (Hans '서버에 바로 맡기자'; no GPU). python3 e59_reagg.py [--out DIR]
Datasets: e2e_syn / e2e_syn_rev (1321, seeds 231-238), qwen_syn / qwen_syn_rev (1322, 221-228), qwen_fq (1324, 211-218), e2e_fq (1323, 211-218; NAT = NATS, BAND1024 = NATMASK ALLQ1024 of 1313), u12 (1328, 231-238)."""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
X=Path(__file__).resolve().parents[2]/'experiments';E7=X/'exp07_write_ladder';Q=X/'exp09_qwen_ttt'
EXPS=tuple(int(x) for x in sys.argv[sys.argv.index('--exps')+1].split(',')) if '--exps' in sys.argv else (16,32,64,128);MET=('margin','top1','rank')
def vec(S,m):
    x=np.asarray(S,float);g=np.asarray(m['gi'] if 'gi' in m else m['gold_idx']);f=np.asarray(m['kind'])=='fact';x=x[f];g=g[f];gold=x[np.arange(len(g)),g];w=x.copy();w[np.arange(len(g)),g]=-np.inf;gt=(x>gold[:,None]).sum(1)
    return np.array([(gold-w.max(1)).mean(),(gt==0).mean(),(1+gt).mean()])
def e2e(path):
    def f(s,arm,e):
        r=json.load(open(path.format(arm=arm,s=s)));rd=r.get('read') or r.get('reads');return vec(rd[str(e)]['out'],r['meta']),vec(r['W0']['out'],r['meta'])
    return f
def qwen(d,rev=False):
    def f(s,arm,e):
        r=json.load(open(Q/f'{d}/e59q_{arm}_s{s}.json'))
        if rev:
            w=json.load(open(Q/f'{d}/e59q_W0rev_s{s}.json'));w0=w['reads'][str(e)]['out'] if str(e) in w['reads'] else w['reads']['out'];return vec(r['reads_rev'][str(e)]['out'],r['meta_rev']),vec(w0,r['meta_rev'])
        w=json.load(open(Q/f'{d}/cband_W0_s{s}.json'));w0=w['reads'][str(e)]['out'] if str(e) in w['reads'] else (w['reads']['out'] if 'out' in w['reads'] else list(w['reads'].values())[0]['out'])
        return vec(r['reads'][str(e)]['out'],r['meta']),vec(w0,r['meta'])
    return f
def fq(s,arm,e):
    p={'NAT':E7/f'fq/run/E55_fq_NATS_s{s}.read.json','BAND1024':E7/f'fq/run/E55_fq_NATMASK_s{s}.read.json','CHK':E7/f'e59_fq/E59_fq_CHK_s{s}.read.json','ISO':E7/f'e59_fq/E59_fq_ISO_s{s}.read.json'}[arm];return e2e(str(p))(s,arm,e)
DS={'e2e_syn':(range(231,239),('NAT','CHK','ISO','MASK1024'),e2e(str(E7/'e59_4way/E59_a4_{arm}_s{s}.read.json'))),
    'e2e_syn_rev':(range(231,239),('NAT','CHK','ISO','MASK1024'),e2e(str(E7/'e59_4way/E59_a4_{arm}_s{s}.rev.read.json'))),
    'qwen_syn':(range(221,229),('NAT','CHK','ISO','MASK1024'),qwen('q4way')),'qwen_syn_rev':(range(221,229),('NAT','CHK','ISO','MASK1024'),qwen('q4way',True)),
    'qwen_fq':(range(211,219),('NAT','CHK','ISO','BANDALL1024'),qwen('fq_qwen')),'e2e_fq':(range(211,219),('NAT','CHK','ISO','BAND1024'),fq),
    'olmo_syn':(range(251,259),('NAT','CHK','ISO','MASK1024'),qwen('../exp11_olmo3/q4way')),'olmo_syn_rev':(range(251,259),('NAT','CHK','ISO','MASK1024'),qwen('../exp11_olmo3/q4way',True)),
    'match':(range(231,239),('NAT','SAME','SAME_U1','RANDTOK','BGSENT'),lambda s,arm,e:e2e(str(E7/('e59_4way/E59_a4_NAT_s{s}.read.json' if arm=='NAT' else 'match/E59_match_{arm}_s{s}.read.json')))(s,arm,e)),
    'show_e2e':(range(231,239),('NAT','CHK','ISO','MASK1024','ALLHID','SHOWSAME','SHOWOTHER'),lambda s,arm,e:e2e(str(E7/('show/E59_show_{arm}_s{s}.read.json' if arm in ('ALLHID','SHOWSAME','SHOWOTHER') else 'e59_4way/E59_a4_{arm}_s{s}.read.json')))(s,arm,e)),
    'show_qwen':(range(221,229),('NAT','CHK','ISO','MASK1024','ALLHID','SHOWSAME','SHOWOTHER'),lambda s,arm,e:qwen('show' if arm in ('ALLHID','SHOWSAME','SHOWOTHER') else 'q4way')(s,arm,e)),
    'fewrep_e2e':(range(231,239),('NAT','CHK','ISO','MASK1024'),e2e(str(E7/'fewrep/E59_fr_{arm}_s{s}.read.json'))),'fewrep_qwen':(range(221,229),('NAT','CHK','ISO','MASK1024'),qwen('fewrep')),
    'ss_qwen_syn':(range(221,229),('NAT','SAME','OTHERONLY','BOTH','RANDF','CHK','ISO','MASK1024'),lambda s,arm,e:qwen('same_spec' if arm in ('SAME','OTHERONLY','BOTH','RANDF') else 'q4way')(s,arm,e)),
    'ss_qwen_fq':(range(211,219),('NAT','SAME','OTHERONLY','BOTH','RANDC','CHK','ISO','BANDALL1024'),lambda s,arm,e:qwen('same_spec_fq' if arm in ('SAME','OTHERONLY','BOTH','RANDC') else 'fq_qwen')(s,arm,e)),
    'ss_e2e_fq':(range(211,219),('NAT','SAME','OTHERONLY','BOTH','RANDC','CHK','ISO','BAND1024'),lambda s,arm,e:fq(s,arm,e) if arm in ('NAT','CHK','ISO','BAND1024') else e2e(str(E7/'same_spec_fq/E59_ss_{arm}_s{s}.read.json'))(s,arm,e)),
    'u12':(range(231,239),('NAT','SAME','SAME_U1','SAME_U2','SAME_U1U2'),e2e(str(E7/'u12/E59_u12_{arm}_s{s}.read.json')))}
def ci(d):
    d=np.asarray(d,float);n=len(d);h=stats.t.ppf(.975,n-1)*d.std(ddof=1)/np.sqrt(n);m=d.mean();return m,m-h,m+h,int(max((d>0).sum(),(d<0).sum()))
def run(name):
    seeds,arms,ld=DS[name];seeds=list(seeds);res={}
    for e in EXPS:
        V={a:[ld(s,a,e) for s in seeds] for a in arms};res[e]={}
        for mi,mn in enumerate(MET):
            raw={a:np.array([v[0][mi] for v in V[a]]) for a in arms};w0=np.array([v[1][mi] for v in V['NAT']]);rows={}
            for a in arms:
                r=dict(abs=ci(raw[a]),vsW0=ci(raw[a]-np.array([v[1][mi] for v in V[a]])))
                if a!='NAT':r['vsNAT']=ci(raw[a]-raw['NAT'])
                rows[a]=r
            res[e][mn]=dict(W0=ci(w0)[:3],arms=rows)
    return res,len(seeds)
def fmt(c):return f'{c[0]:7.3f} [{c[1]:7.3f},{c[2]:7.3f}] {c[3]}/8' if len(c)==4 else f'{c[0]:7.3f} [{c[1]:7.3f},{c[2]:7.3f}]'
def main():
    only=sys.argv[sys.argv.index('--only')+1].split(',') if '--only' in sys.argv else list(DS)
    out=Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else E7/'binding_fit';allr={}
    for n in only:
        res,ns=run(n);allr[n]=res;L=[f'# {n}: n={ns} seeds; out-of-window; mean [95% t-CI] same-direction seeds. chance top-1 0.0625, rank 8.5; rank: lower = better']
        for e in EXPS:
            for mn in MET:
                t=res[e][mn];L.append(f'## exposure {e}  {mn}   W0 {fmt(t["W0"])}')
                for a,r in t['arms'].items():L.append(f'  {a:11s} abs {fmt(r["abs"])} | vs W0 {fmt(r["vsW0"])}'+(f' | vs NAT {fmt(r["vsNAT"])}' if 'vsNAT' in r else ''))
        (out/f'reagg_{n}.txt').write_text('\n'.join(L)+'\n');print(n,'ok',flush=True)
    json.dump(allr,open(out/('reagg_all.json' if len(only)==len(DS) and EXPS==(16,32,64,128) else 'reagg_partial_'+'_'.join(only)+'.json'),'w'),indent=1)
if __name__=='__main__':main()
