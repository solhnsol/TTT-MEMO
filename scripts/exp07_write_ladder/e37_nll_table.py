"""E37 first-chunk NLL table (CPU, no model): reads OUT/e37_firstchunk_<tag>.json (e37_firstchunk_nll.py) and prints, per state, the mean NLL increase over W0 with a 95% bootstrap CI over samples (5000 draws, seed 0), for world A, world B and their per-sample average (cells whose labels are <CELL>_A / <CELL>_B), with the pg11-overlap samples 145-147 excluded as in E37 section 5 (the table also gives the all-sample value), plus the W0 repeat max |diff| (0 = deterministic).
Usage: e37_nll_table.py --json OUT/e37_firstchunk_e37w1.json [--exclude 145,146,147] [--out table.md] | --selftest"""
import sys,json,argparse
from pathlib import Path
import numpy as np
NB=5000
def ci(x,rng):
    x=np.asarray(x,float);n=len(x);b=[x[rng.integers(0,n,n)].mean() for _ in range(NB)];lo,hi=np.percentile(b,[2.5,97.5]);return float(x.mean()),float(lo),float(hi)
def table(res,exclude=(145,146,147)):
    rng=np.random.default_rng(0);S=res['states'];w0=np.array(S['W0']['nll']);n=len(w0);keep=np.array([i not in set(exclude) for i in range(n)]);L=[f"samples {n}, chunk {res.get('chunk')}; main columns exclude samples {sorted(set(exclude))} (kept {int(keep.sum())}); BOS-first-token share {res.get('bos_first_token_frac')}",'']
    wr=res.get('W0_repeat');L.append('W0 repeat max |diff|: '+(f"{wr['max_abs_diff']:.3g}" if wr else 'not read')+'\n')
    L+=['| cell (pass) | world A | world B | A/B average | A/B average, all samples |','|---|---|---|---|---|']
    cells=sorted({l.rsplit('_',1)[0] for l in S if l!='W0' and l.endswith(('_A','_B'))});rows={}
    for c in cells:
        d={w:np.array(S[f'{c}_{w}']['nll'])-w0 for w in 'AB' if f'{c}_{w}' in S}
        f=lambda v,k:'%+.4f [%+.4f, %+.4f]'%ci(v[k],rng)
        a=f(d['A'],keep) if 'A' in d else '-';b=f(d['B'],keep) if 'B' in d else '-'
        avg=((d['A']+d['B'])/2) if len(d)==2 else None
        L.append(f"| {c} | {a} | {b} | {f(avg,keep) if avg is not None else '-'} | {('%+.4f [%+.4f, %+.4f]'%ci(avg,rng)) if avg is not None else '-'} |");rows[c]=float(avg[keep].mean()) if avg is not None else None
    return '\n'.join(L)+'\n',rows
def selftest():
    rng=np.random.default_rng(1);n=200;w0=rng.normal(3,.5,n);res={'chunk':0,'bos_first_token_frac':1.0,'states':{'W0':{'nll':w0.tolist()}},'W0_repeat':{'max_abs_diff':0.0}}
    for c,g in (('native_p4',.04),('sent_p4',.1)):
        for w in 'AB':res['states'][f'{c}_{w}']={'nll':(w0+g+rng.normal(0,.05,n)).tolist()}
    t,r=table(res);assert abs(r['native_p4']-.04)<.02 and abs(r['sent_p4']-.1)<.02,r;print(t);print('e37_nll_table SELFTEST OK')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json');ap.add_argument('--exclude',default='145,146,147');ap.add_argument('--out',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    t,_=table(json.load(open(a.json)),[int(x) for x in a.exclude.split(',') if x]);print(t)
    if a.out:Path(a.out).write_text(t)
