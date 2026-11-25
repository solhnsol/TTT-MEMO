"""1344m (one real step / key-error swap / size restoration / position split) re-cut by the SENTENCE ORDER of the evaluated fact (CPU; new file; thread-server 11/25). Inputs: kvstep/kvstep_s<seed>.json (8 seeds, chunks 16 48 80 112; dmargin_facts per read fact) and kv1344k/sent_order_s<seed>.json (recreated orders, 0 place first, 1 name first). TERMS of 1344m: the position classes own / other / rest are defined with d['second'] = the LATER mentioned entity: own = the later-entity tokens of the 16 READ facts of the chunk (place in name-first sentences, person name in place-first sentences), other = the later-entity tokens of the 16 extra (U2) facts, rest = all other positions; the key / error decomposition and the size / common-key restoration use ALL positions (not class dependent). Values: margin change over W0 (summed log p), mean over seeds [t] same-direction seeds; per seed the mean over the 4 chunks and the facts of the order group. Not recoverable: per-token ||e||, NLL, candidate scores of 1344m. Usage: python e63_kvstep_recut.py --dir kvstep --orders kv1344k"""
import sys,json,glob,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1];OD=a[a.index('--orders')+1];R={};O={}
for f in sorted(glob.glob(D+'/kvstep_s*.json')):
    d=json.load(open(f));sd=d['seed'];R[sd]=d;O[sd]=json.load(open(f'{OD}/sent_order_s{sd}.json'))
SE=sorted(R);n=len(SE)
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):7.3f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
def ps(fn,g):
    out=[]
    for sd in SE:
        v=[]
        for row in R[sd]['rows']:
            x=fn(row);od=np.asarray(O[sd][str(row['k'])]);m=np.ones(16,bool) if g=='all' else (od==g)
            if m.any():v.append(x[m].mean())
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)
A=lambda name:(lambda row:np.asarray(row['combos'][name]['dmargin_facts'],float))
nf={g:sum(int((np.asarray(O[sd][str(row['k'])])==g).sum()) for sd in SE for row in R[sd]['rows']) for g in (0,1)}
print(f'1344m by sentence order of the evaluated fact: {n} seeds, chunks {[r["k"] for r in R[SE[0]]["rows"]]}; facts: place first {nf[0]}, name first {nf[1]}')
def show(title,fns,diffs=()):
    print(f'== {title}')
    for g,gn in ((0,'place first'),(1,'name first'),('all','all facts')):
        vals={nm:ps(fn,g) for nm,fn in fns.items()};line=f'  {gn:11s}'+' | '.join(f'{nm} {tci(v)}' for nm,v in vals.items())
        for dn,(x,y) in diffs:line+=f' | {dn} {tci(vals[x]-vals[y])}'
        print(line)
    dd=[]
    for dn,(x,y) in diffs:
        d1=ps(fns[x],1)-ps(fns[y],1);d0=ps(fns[x],0)-ps(fns[y],0);print(f'  {dn}: (name first) - (place first), seed-paired {tci(d1-d0)}')
show('one real step (autodiff)',{'ACT C0':A('ACT[C0]'),'ACT SAMEVIS':A('ACT[SAMEVIS]'),'ACT NATVIS':A('ACT[NATVIS]')},[('SAMEVIS-C0',('ACT SAMEVIS','ACT C0')),('NATVIS-C0',('ACT NATVIS','ACT C0'))])
show('key / error swap (G of all positions): key effect = G[SV,C0]-G[C0,C0], error effect = G[C0,SV]-G[C0,C0], total = G[SV,SV]-G[C0,C0]',{'G C0,C0':A('G[C0,C0]'),'G SV,C0':A('G[SAMEVIS,C0]'),'G C0,SV':A('G[C0,SAMEVIS]'),'G SV,SV':A('G[SAMEVIS,SAMEVIS]')},[('key effect',('G SV,C0','G C0,C0')),('error effect',('G C0,SV','G C0,C0')),('total',('G SV,SV','G C0,C0'))])
show('size restoration (error size of C0 restored per position and layer, direction SAMEVIS; ALL positions) and common-key restoration',{'G SV,SV':A('G[SAMEVIS,SAMEVIS]'),'size=C0':A('G[SAMEVIS,SAMEVIS|size=C0]'),'common=C0':A('G[SAMEVIS,SAMEVIS|common=C0]')},[('size restoration',('size=C0','G SV,SV')),('common restoration',('common=C0','G SV,SV'))])
for c,cn in (('own','own = later-entity tokens of the 16 read facts'),('other','other = later-entity tokens of the 16 extra facts'),('rest','rest = all other positions')):
    show(f'position class {cn}',{'C0':A(f'G[C0,C0]@{c}'),'SAMEVIS':A(f'G[SAMEVIS,SAMEVIS]@{c}')},[('SAMEVIS-C0',('SAMEVIS','C0'))])
