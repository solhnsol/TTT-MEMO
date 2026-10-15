"""W0 one-step realization test analysis (PROTOCOL_stage01.md, rules fixed 2026-10-16 00:55 KST). CPU."""
import sys,json
from pathlib import Path
import numpy as np
sys.argv=['x','101'];sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_stage01 as S
AL=S.AL;OUT=S.OUT;THR=0.10
meta=json.load(open(OUT/'onestep_meta_s101.json'));pairs=meta['pairs'];pr=meta['pred']
md=['# W0 one-step realization test (seed 101)\n',f"eta = {meta['eta']:.4g}; clip coef A {meta['coef']['A']:.3f}, B {meta['coef']['B']:.3f}; |g| A {meta['gnorm']['A']:.3f}, B {meta['gnorm']['B']:.3f}\n",
    '| direction | s | D (R1) | D (R1-alt) | R0 binding measured | R0 predicted (1st order) | ratio | R1-alt binding measured (M_L3, alt) | ratio alt/pred | familiarity F_name (R1) | max canary rise |','|---|---|---|---|---|---|---|---|---|---|---|']
res={}
for vn in ('g','gb'):
    X=S.load('ONESTEP',vn)
    if not X:continue
    A,B=X;c=S.curve(A,B);res[vn]={}
    for (sc,s1,sa,d),ea,eb in zip(c,A,B):
        if sc==0:continue
        mb=np.asarray([(ea['r0']['L3'][i]-eb['r0']['L3'][i])/2 for i in range(len(pairs))])
        pb=np.asarray([(ea['step']*pr[f'{vn}|A|L3|{p}']-eb['step']*pr[f'{vn}|B|L3|{p}'])/2 for p in pairs])
        da=AL.ci(sa['D']);res[vn][sc]={'D_alt':da,'damaged':d>THR,'ratio':float(mb.mean()/pb.mean()),'ratio_alt':float(sa['M_L3'].mean()/pb.mean())}
        md.append(f"| {vn} | {sc} | {AL.f3(AL.ci(s1['D']))} | {AL.f3(da)} | {mb.mean():+.4f} | {pb.mean():+.4f} | {mb.mean()/pb.mean():.2f} | {AL.f3(AL.ci(sa['M_L3']))} | {sa['M_L3'].mean()/pb.mean():.2f} | {AL.f3(AL.ci(s1['F_name']))} | {d:+.3f}{' (damaged)' if d>THR else ''} |")
def und(vn):return [s for s,v in res.get(vn,{}).items() if not v['damaged']]
def zero(vn,s):lo,hi=res[vn][s]['D_alt'][1:];return lo<=0<=hi
if res.get('g') and res.get('gb') and und('g') and und('gb'):
    sg,sb=max(und('g')),max(und('gb'))
    if all(zero('g',s) for s in und('g')) and res['gb'][sb]['D_alt'][1]>0:v='(ga) familiarity cancels the binding gain'
    elif all(zero('gb',s) for s in und('gb')):v='(na) binding not realized at single-step scale'
    elif res['g'][sg]['D_alt'][1]>0 and res['gb'][sb]['D_alt'][1]>0 and 1 in res['g'] and 0.5<=res['g'][1]['ratio']<=1.5:v='first-order indicator valid for single steps'
    else:v='mixed'
else:v='no verdict (missing or all points damaged)'
md.append(f'\n**verdict: {v}**');(OUT/'onestep_report_s101.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
