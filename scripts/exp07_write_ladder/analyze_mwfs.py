"""M3 L0mw-fs rules (PROTOCOL_stage01.md 2026-10-16). Usage: analyze_mwfs.py select | analyze_mwfs.py SEED TAG"""
import sys,json,importlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));THR=0.10
def mods(sd):
    sys.argv=['x',str(sd)]
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
if sys.argv[1]=='select':
    AL,S=mods(101);sel=None;md=['# L0mw-fs lr selection (s101): canary rise at pass 3 must be <= 0.10\n']
    for t,sc in (('x2',2),('x1',1),('x05',0.5)):
        X=S.load('L0mwfs',t)
        if not X:continue
        c=S.curve(*X);d3=c[3][3];md.append(f'- {t}: pass-3 canary rise {d3:+.3f}')
        if d3<=THR and sel is None:sel=t
    md.append(f"\nSELECTED {sel}" if sel else '\nSELECTED NONE');(S.OUT/'mwfs_select.md').write_text('\n'.join(md)+'\n');print('\n'.join(md));sys.exit(0)
sd,tag=int(sys.argv[1]),sys.argv[2];AL,S=mods(sd)
c=S.curve(*S.load('L0mwfs',tag));L1=S.curve(*S.load('L1','ck'))
ln=S.OUT/f'longnll_L0mwfs_{tag}_s{sd}.json';ln=json.load(open(ln)) if ln.exists() else None
md=[f'# L0mw-fs {tag}, seed {sd}\n','| pass | D (R1) | D (R1-alt) | max canary rise | long-ctx NLL rise (>8K, A/B) |','|---|---|---|---|---|']
for ep,s,sa,d in c:
    lr=''
    if ln and ep>0:lr=' / '.join(f"{x['rise_beyond']:+.3f}" for w in 'AB' for x in ln[w] if x['epoch']==ep)
    md.append(f"| {ep} | {AL.f3(AL.ci(s['D']))} | {AL.f3(AL.ci(sa['D']))} | {d:+.3f} | {lr} |")
k=0
for ep,_,_,d in c[1:]:
    if d>THR:break
    k=ep
crit={}
crit['k*>=3']=k>=3
if k>=1:
    dd=AL.ci(c[k][2]['D']-c[1][2]['D']);crit['D_alt(k*)-D_alt(1) CI lower>0']=dd[1]>0
    drops=[ep for ep in range(2,k+1) if c[ep][2]['D'].mean()<c[ep-1][2]['D'].mean()-(AL.ci(c[ep][2]['D'])[2]-AL.ci(c[ep][2]['D'])[1])/2];crit['no large drop']=not drops
    if k<len(L1):R=S.ratio(c[k][1]['D'],L1[k][1]['D']);crit[f'R*={R[0]:.2f} [{R[1]:.2f},{R[2]:.2f}] >=0.30']=R[0]>=0.30
    if ln:rb=max(x['rise_beyond'] for w in 'AB' for x in ln[w] if x['epoch']==k);crit[f'long-NLL rise at k* {rb:+.3f} <=0.10']=rb<=0.10
    else:crit['long-NLL measured']=False
md.append(f'\nk* = {k}');md+= [f'- {kk}: {"yes" if v else "NO"}' for kk,v in crit.items()]
md.append(f"\n**M3 seed {sd}: {'PASS' if all(crit.values()) else 'FAIL'}**")
(S.OUT/f'mwfs_report_{tag}_s{sd}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
