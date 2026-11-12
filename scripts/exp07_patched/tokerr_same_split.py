"""Post-hoc descriptive split of the tokerr 'same chunk' interval (requested after the tokerr result): (i) the exposure also has an earlier same-triple exposure in a previous chunk inside the 8192 window,
(ii) it has none (only the same-chunk previous exposure). Same r definition and (seed, world, pair) cluster bootstrap as analyze_tokerr.py."""
import sys,os,json;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'../exp04_what_ttt_retains'))
import numpy as np
import gen_world as gw,suppress as SP,s3_sets as S3
CH=S3.CH;W=S3.W;OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../experiments/exp07_write_ladder')
exp={}
for sd in (101,102,103):
    w0=gw.build_world(sd)
    for w in 'AB':
        tab=S3.sentence_table(w0,w=='B',gw.TOK,gw,SP);by={s['idx']:s for s in tab}
        recs=json.load(open(f'{OUT}/tokerr_s{sd}_{w}.json'))['records']
        for r in recs:
            if r['interval']!='same':continue
            s=by[r['idx']];k0=(s['place'][0]//CH)*CH
            earlier_in_window=any(u['t']==s['t'] and u['place'][0]<k0 and u['place'][0]>=k0-W for u in tab)
            e=exp.setdefault((sd,w,r['idx']),dict(pair=r['pair'],grp='i' if earlier_in_window else 'ii',lr=[],cos=[]))
            e['lr'].append(np.log(max(1-r['p_nat'],1e-6)/max(1-r['p_alone'],1e-6)));e['cos'].append(r['cos'])
cl={}
for (sd,w,idx),e in exp.items():
    c=cl.setdefault((sd,w,e['pair']),{});iv=c.setdefault(e['grp'],[0.0,0]);iv[0]+=float(np.mean(e['lr']));iv[1]+=1
keys=sorted(cl);n=len(keys);rng=np.random.default_rng(0);draws=[rng.integers(0,n,n) for _ in range(8000)]
def stat(idx,g):
    s=0.0;c=0
    for i in idx:
        if g in cl[keys[i]]:s+=cl[keys[i]][g][0];c+=cl[keys[i]][g][1]
    return s/c if c else float('nan')
res={}
for g,name in (('i','(i) also an earlier exposure in a previous chunk inside 8K'),('ii','(ii) only the same-chunk previous exposure')):
    m=stat(range(n),g);xs=np.array([stat(i,g) for i in draws]);xs=xs[np.isfinite(xs)];lo,hi=np.percentile(xs,[2.5,97.5]);cnt=sum(1 for e in exp.values() if e['grp']==g);cs=np.mean([np.mean(e['cos']) for e in exp.values() if e['grp']==g])
    res[g]=dict(n=cnt,r=float(np.exp(m)),ci=[float(np.exp(lo)),float(np.exp(hi))],cos=float(cs));print(f'{name}: n={cnt}, geometric-mean r {np.exp(m):.3f} [{np.exp(lo):.3f},{np.exp(hi):.3f}], input cos {cs:.3f}')
json.dump(res,open(f'{OUT}/tokerr_same_split.json','w'),indent=1)
