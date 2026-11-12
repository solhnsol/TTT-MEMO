"""tokerr analysis (registered 17:51 KST 10/17). Exposure log r_e = mean over its place-token pieces of log[(1 - p_nat(y)) / (1 - p_alone(y))] (1 - p clipped at 1e-6). Interval means = mean of log r_e over the exposures of the interval
(exponentiated for reporting). Clusters = (seed, world, pair), cluster bootstrap. (a) inside-8K interval: CI upper of the log mean < 0 (r < 1). (b) [(outside 8K + no previous) - (inside 8K)] difference of log means: CI lower > 0.
Same chunk: descriptive. Also descriptive: mean ||x_nat|| / ||x_alone|| and mean cos of the first fast-weight layer input, per interval. Usage: analyze_tokerr.py"""
import json,os,sys
from pathlib import Path
import numpy as np
OUT=Path(os.environ.get('EXP07_OUT',Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder'))
exp={};n_files=0
for sd in (101,102,103):
    for w in 'AB':
        f=OUT/f'tokerr_s{sd}_{w}.json'
        if not f.exists():continue
        n_files+=1
        for r in json.load(open(f))['records']:
            e=exp.setdefault((sd,w,r['idx']),dict(pair=r['pair'],interval=r['interval'],lr=[],xr=[],cos=[]))
            e['lr'].append(np.log(max(1-r['p_nat'],1e-6)/max(1-r['p_alone'],1e-6)));e['xr'].append(r['xn']/max(r['xa'],1e-30));e['cos'].append(r['cos'])
print(f'{n_files} (seed, world) files; {len(exp)} exposures')
cl={}
for (sd,w,idx),e in exp.items():
    c=cl.setdefault((sd,w,e['pair']),{});iv=c.setdefault(e['interval'],[0.0,0]);iv[0]+=float(np.mean(e['lr']));iv[1]+=1
keys=sorted(cl);n=len(keys)
def stat(idx,ivs):
    s=0.0;c=0
    for i in idx:
        for iv in ivs:
            if iv in cl[keys[i]]:s+=cl[keys[i]][iv][0];c+=cl[keys[i]][iv][1]
    return s/c if c else float('nan')
rng=np.random.default_rng(0);B=8000;draws=[rng.integers(0,n,n) for _ in range(B)]
def ci(fn):
    xs=np.array([fn(i) for i in draws]);xs=xs[np.isfinite(xs)];return float(np.percentile(xs,2.5)),float(np.percentile(xs,97.5))
allidx=np.arange(n);res={}
for name,ivs in (('same',['same']),('in8k',['in8k']),('out8k',['out8k']),('none',['none']),('out+none',['out8k','none'])):
    m=stat(allidx,ivs);lo,hi=ci(lambda i:stat(i,ivs));cnt=sum(cl[k][iv][1] for k in keys for iv in ivs if iv in cl[k])
    xr=np.mean([np.mean(e['xr']) for e in exp.values() if e['interval'] in ivs]);cs=np.mean([np.mean(e['cos']) for e in exp.values() if e['interval'] in ivs])
    res[name]=dict(n=cnt,mean_log_r=m,ci=[lo,hi],r=float(np.exp(m)),r_ci=[float(np.exp(lo)),float(np.exp(hi))],xnorm_ratio=float(xr),x_cos=float(cs))
    print(f'{name:9s} n={cnt:4d}: geometric-mean r {np.exp(m):.3f} [{np.exp(lo):.3f},{np.exp(hi):.3f}]; ||x_nat||/||x_alone|| {xr:.3f}; cos {cs:.3f}')
a_ok=res['in8k']['ci'][1]<0
d=lambda i:stat(i,['out8k','none'])-stat(i,['in8k']);dm=d(allidx);dlo,dhi=ci(d)
res['b']=dict(diff=dm,ci=[dlo,dhi]);res['a_pass']=bool(a_ok);res['b_pass']=bool(dlo>0)
ds=lambda i:stat(i,['same'])-stat(i,['in8k']);sm=ds(allidx);slo,shi=ci(ds);res['same_minus_in']=dict(diff=sm,ci=[slo,shi])
print(f'(a) inside-8K geometric-mean r {np.exp(res["in8k"]["mean_log_r"]):.3f}, CI upper {np.exp(res["in8k"]["ci"][1]):.3f} < 1: {a_ok}')
print(f'(b) log-mean difference [(out8k + none) - in8k] {dm:+.3f} CI [{dlo:+.3f},{dhi:+.3f}] > 0: {dlo>0}')
print(f'descriptive: same chunk minus in8k {sm:+.3f} CI [{slo:+.3f},{shi:+.3f}]')
json.dump(res,open(OUT/'tokerr_result.json','w'),indent=1)
