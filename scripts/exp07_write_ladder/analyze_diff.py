"""Q1 (formerly M1; PROTOCOL_stage01.md 2026-10-16): masked x2 pass 1 minus native one pass (L0mp pass 1), per pair, per seed and pooled."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import importlib
def mods(sd):
    """analysis modules bound to one seed (pair ids come from that seed's L1 meta at import time)."""
    sys.argv=['x',str(sd)]
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
rng=np.random.default_rng(29);SEEDS=(101,102,103);d={}
md=['# Q1. masked (x2, pass 1) minus native (one pass), per pair\n','| seed | readout | masked D | native D | difference [95% CI] |','|---|---|---|---|---|']
for sd in SEEDS:
    AL,S=mods(sd);mw=S.load('L0mw','x2');mp=S.load('L0mp','alt')
    if not (mw and mp):md.append(f'| {sd} | missing | | | |');continue
    cm=S.curve(*mw);cn=S.curve(*mp)
    for key,i in (('alt',2),('R1',1)):
        x=cm[1][i]['D'];y=cn[1][i]['D'];d[(sd,key)]=x-y
        md.append(f"| {sd} | {key} | {x.mean():+.3f} | {y.mean():+.3f} | {AL.f3(AL.ci(x-y))} |")
res={}
for key in ('alt','R1'):
    ss=[sd for sd in SEEDS if (sd,key) in d]
    if len(ss)<3:continue
    bs=[np.mean([d[(sd,key)][rng.integers(0,len(d[(sd,key)]),len(d[(sd,key)]))].mean() for sd in ss]) for _ in range(4000)]
    m=np.mean([d[(sd,key)].mean() for sd in ss]);lo,hi=np.percentile(bs,[2.5,97.5]);pts=[d[(sd,key)].mean() for sd in ss]
    res[key]=(m,lo,hi,pts);md.append(f"\nPooled ({key}): {m:+.3f} [{lo:+.3f}, {hi:+.3f}]; per-seed points {', '.join(f'{p:+.3f}' for p in pts)}")
if 'alt' in res:
    m,lo,hi,pts=res['alt'];ok=lo>0 and all(p>0 for p in pts)
    md.append(f"\n**Q1 verdict (R1-alt, registered): {'PASS' if ok else 'FAIL'}** (pooled lower bound {lo:+.3f} > 0: {lo>0}; all per-seed points > 0: {all(p>0 for p in pts)})")
(S.OUT/'m1_diff_report.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
