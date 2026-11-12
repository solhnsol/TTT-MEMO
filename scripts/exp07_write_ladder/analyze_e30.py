"""E30 analysis (CPU) of e30_dose.py eval output (OUT/e30_<tag>_s<seed>.json; one or several files of different cell groups can be merged with --tags a,b).
D1(cell, scale) = per pair [ (M_L3(A) - M_L3(B))/2 - (M_XP(A) - M_XP(B))/2 ] on R1-alt (R1 also reported), M_f = mean over the pair's items of the margin (analyze_ladder.stats / E29 definition), 15 pairs.
Q(L) = D1(F(L)) / D1(F(0)) (ratio of pair means, bootstrap over pairs); X = [D1(F0) - D1(F8192)] / [D1(F0) - D1(A)]; Spearman of D1 over the five F points vs L; gate: F0 vs M relative D1 difference <= 1e-3;
P(L)-F(L) differences; R/F0; N arms vs F8192; canary rise per state (max of gen and tail NLL rise over W0 and over worlds). Pre-registered readings (E30 2.0) are printed as 'reading aids', not applied as a verdict.
Usage: analyze_e30.py --dir OUT --seed 101 --tags e30core[,e30full] [--out report.md] | --selftest"""
import sys,json,argparse
from pathlib import Path
import numpy as np
NB=4000
def load(d,seed,tags):
    res=None
    for t in tags:
        j=json.load(open(Path(d)/f'e30_{t}_s{seed}.json'))
        if res is None:res=j;res['cells']=dict(j['cells'])
        else:
            res['cells'].update(j['cells'])
            assert j['items']==res['items'],'item order differs between files'
    return res
def fam_slices(res):
    o={};i=0
    for k,n in res['fam_sizes'].items():o[k]=(i,i+n);i+=n
    return o
def Mf(res,evA,evB,f,alt=True):
    a,b=fam_slices(res)[f];its=res['items'][a:b];xa=np.asarray(evA['_alt'][f] if alt else evA[f]);xb=np.asarray(evB['_alt'][f] if alt else evB[f]);d={}
    for it,x in zip(its,(xa-xb)/2):d.setdefault(it['pair'],[]).append(x)
    ks=sorted(d,key=lambda s_:int(s_[1:]));return ks,np.asarray([np.mean(d[k]) for k in ks])
def D1(res,cell,s,alt=True):
    A=res['cells'][cell]['A']['scales'][s]['margins'];B=res['cells'][cell]['B']['scales'][s]['margins']
    k3,m3=Mf(res,A,B,'L3',alt);kx,mx=Mf(res,A,B,'XP',alt);return m3-mx   # pair p of L3 <-> pair x_p of XP (both sorted by index)
def canary(res,cell,s):
    r=[]
    for w in 'AB':
        ev=res['cells'][cell][w]['scales'][s]['margins'];r.append(max(ev['_gen_nll']-res['W0']['_gen_nll'],ev['_tail_nll']-res['W0']['_tail_nll']))
    return max(r)
def boot(v,rng):
    v=np.asarray(v);b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(NB)];return float(v.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def boot_ratio(num,den,rng):
    n=len(num);bs=[]
    for _ in range(NB):
        i=rng.integers(0,n,n);d=den[i].mean();bs.append(num[i].mean()/d if abs(d)>1e-9 else np.nan)
    pt=num.mean()/den.mean() if abs(den.mean())>1e-9 else float('nan');lo,hi=np.nanpercentile(bs,[2.5,97.5]);return float(pt),float(lo),float(hi)
def spearman(x,y):
    rx=np.argsort(np.argsort(x));ry=np.argsort(np.argsort(y));return float(np.corrcoef(rx,ry)[0,1]) if np.std(ry)>0 else float('nan')
fmt=lambda t:f'{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]'
def analyze(d,seed,tags):
    res=load(d,seed,tags);rng=np.random.default_rng(30);cells=list(res['cells']);scales=sorted({s for c in cells for s in res['cells'][c]['A']['scales']},key=float);L=[];out={'D1':{},'Q':{},'gates':{}}
    for alt,nm in ((True,'R1-alt'),(False,'R1')):
        L+=[f'\n## D1 per cell and scale, {nm} (pair means, 95% bootstrap CI over {len(D1(res,cells[0],scales[0],alt))} pairs)\n','| cell | '+' | '.join(f'scale {s} D1 [CI]' for s in scales)+' | canary rise (max over scales) |','|---|'+'---|'*(len(scales)+1)]
        for c in cells:
            row=[fmt(boot(D1(res,c,s,alt),rng)) for s in scales];out['D1'][(nm,c)]={s:D1(res,c,s,alt).tolist() for s in scales}
            L.append(f'| {c} | '+' | '.join(row)+f' | {max(canary(res,c,s) for s in scales):+.3f} |')
    pr=scales[-1] if '4' not in scales else '4'
    for s in scales:
        L.append(f'\n## Derived quantities, scale {s} (R1-alt)\n')
        g=lambda c:D1(res,c,s,True)
        if 'F0' in cells:
            F=[(l,f'F{l}') for l in (0,128,512,2048,8192) if f'F{l}' in cells]
            for l,c in F:
                if l>0:L.append(f'- Q({l}) = D1(F{l}) / D1(F0): {fmt(boot_ratio(g(c),g("F0"),rng))}')
            if len(F)>=4:
                sp=spearman([l for l,_ in F],[g(c).mean() for _,c in F]);L.append(f'- Spearman of D1 over L ({len(F)} points): {sp:+.2f}');out['Q'][s]={'spearman':sp}
            if 'F8192' in cells and 'A' in cells:
                num=g('F0')-g('F8192');den=g('F0')-g('A');L.append(f'- X = [D1(F0) - D1(F8192)] / [D1(F0) - D1(A)]: {fmt(boot_ratio(num,den,rng))}')
            if 'M' in cells:
                rel=abs(g('F0').mean()-g('M').mean())/max(abs(g('M').mean()),1e-12);out['gates'][f'F0_vs_M_rel_s{s}']=float(rel);L.append(f'- gate F0 vs M (chunk alone): D1 {g("F0").mean():+.4f} vs {g("M").mean():+.4f}, mean pair difference {abs(g("F0").mean()-g("M").mean()):.4f} (<= 0.01 required, E30 design), relative difference {rel:.2e} (reported only){"  **GATE FAILED**" if abs(g("F0").mean()-g("M").mean())>0.01 else ""}')
            for l in (512,8192):
                c=f'P{l}'
                if c in cells and f'F{l}' in cells:L.append(f'- P{l} - F{l}: {fmt(boot(g(c)-g(f"F{l}"),rng))}')
            if 'R' in cells:L.append(f'- R (first 128 loss tokens excluded) / F0: {fmt(boot_ratio(g("R"),g("F0"),rng))}')
            for c in ('Na','Nb'):
                if c in cells and 'F8192' in cells:L.append(f'- {c} - F8192 (context text kind): {fmt(boot(g(c)-g("F8192"),rng))};  {c} / F0: {fmt(boot_ratio(g(c),g("F0"),rng))}')
        if 'A' in cells and 'M' in cells:L.append(f'- D1(M) - D1(A) (chunk alone minus native): {fmt(boot(g("M")-g("A"),rng))}')
    L.append('\nreading aids (E30 2.0, registered before the result; not applied here): H-context-general: Q(8192) <= 0.8 with CI upper bound < 1 and Spearman <= -0.8, X >= 0.5 main factor, 0.2-0.5 partial; H-facts-only: Q(8192) >= 0.95 with CI containing 1 or X < 0.2; gate F0 = M within 1e-3; controls: P within F spread, R < 0.9 F0 = ramp contribution. One seed (s101), 15 pairs, development seed, not confirmation.')
    return out,'\n'.join(L)+'\n'
def selftest():
    import tempfile
    d=Path(tempfile.mkdtemp());rng=np.random.default_rng(3);P=15
    items=[];fs={'L3':2*P,'XP':2*P}
    for f,n in fs.items():
        for j in range(n):items.append({'level':f,'tid':j,'pair':(f'p{j//2}' if f=='L3' else f'x{j//2}')})
    def ev(gainA,gainB):
        m={'L3':(rng.normal(0,.02,2*P)+gainA).tolist(),'XP':rng.normal(0,.02,2*P).tolist(),'_gen_nll':0.51,'_tail_nll':0.61};m['_alt']={'L3':[x for x in m['L3']],'XP':list(m['XP'])};return m
    D={'A':0.3,'M':1.2,'F0':1.2,'F128':1.1,'F512':0.9,'F2048':0.7,'F8192':0.5,'P512':0.9,'P8192':0.5,'R':1.1,'Na':0.5,'Nb':0.55}
    cells={}
    for c,dv in D.items():
        cells[c]={w:{'gnorm':1.0,'coef':1.0,'rep':c,'scales':{s:{'hash':'h','margins':ev(+dv*float(s)/2,-dv*float(s)/2) if w=='A' else ev(-dv*float(s)/2,dv*float(s)/2)} for s in ('1','4')}} for w in 'AB'}
    # world A: gain on L3 (+), world B: opposite => D1 = gain
    for c in cells:
        for w in 'AB':
            for s in cells[c][w]['scales']:cells[c][w]['scales'][s]['margins']['L3']=[x+(0 if True else 0) for x in cells[c][w]['scales'][s]['margins']['L3']]
    j={'manifest':{},'cells':cells,'items':items,'fam_sizes':fs,'W0':{'_gen_nll':0.5,'_tail_nll':0.6,'L3':[0]*(2*P),'XP':[0]*(2*P),'_alt':{'L3':[0]*(2*P),'XP':[0]*(2*P)}}}
    json.dump(j,open(d/'e30_t_s101.json','w'));out,md=analyze(d,101,['t'])
    r4=np.mean(out['D1'][('R1-alt','F8192')]['4']);r0=np.mean(out['D1'][('R1-alt','F0')]['4']);assert abs(r4/r0-0.5/1.2)<0.1,(r4,r0)
    assert 'gate F0 vs M' in md and 'Spearman' in md and 'X = ' in md
    print(md[-1500:]);print('analyze_e30 SELFTEST OK')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seed',type=int,default=101);ap.add_argument('--tags',default='e30core');ap.add_argument('--out',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    out,md=analyze(a.dir,a.seed,a.tags.split(','));p=Path(a.out or Path(a.dir)/f'e30_report_s{a.seed}.md');p.write_text(f'# E30 report (seed {a.seed}, tags {a.tags})\n'+md);json.dump({k:{str(kk):vv for kk,vv in v.items()} for k,v in out.items()},open(p.with_suffix('.json'),'w'),indent=1,default=str);print(md)
