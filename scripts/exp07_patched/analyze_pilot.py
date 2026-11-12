"""Masked-write pilot analysis (PROTOCOL_stage01.md, 23:10 KST spec; amended 2026-10-16 00:35 KST). Usage: analyze_pilot.py SEED [125m]"""
import sys,os
from pathlib import Path
import numpy as np
SEED=sys.argv[1];M125=len(sys.argv)>2 and sys.argv[2]=='125m'
sys.argv=['x',SEED,'ck' if M125 else 'main'];sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_stage01 as S
AL=S.AL;THR=0.10
def run_curve(tag):
    X=S.load('L0mw',tag);return S.curve(*X) if X else None
def kstar(c):
    k=0
    for ep,_,_,d in c[1:]:
        if d>THR:break
        k=ep
    return k
def main():
    md=[f"# Masked-write pilot ({'125M' if M125 else '3B'}, seed {SEED})\n"]
    L1=S.curve(*S.load('L1','ck'))
    if M125:nat=S.curve(*S.load('L0','main'))[-1]           # native one pass (state 63)
    else:
        mp=S.load('L0mp','alt');nat=S.curve(*mp)[1] if mp else None   # native pass 1 (= one native pass)
    tags=[f'b{b}x{s}' for b in (1024,256) for s in (1,2,4)] if M125 else [f'x{s}' for s in (1,2)]
    rows={};md.append('| run | pass-1 D (R1) | pass-1 D (R1-alt) | pass-1 canary | k* | D at k* (R1) | R* vs L1 [CI] |\n|---|---|---|---|---|---|---|')
    for t in tags:
        c=run_curve(t)
        if not c:continue
        k=kstar(c);p1=c[1]
        R=S.ratio(c[k][1]['D'],L1[k][1]['D']) if k>=1 else (float('nan'),)*3
        rows[t]=(c,k,R);md.append(f"| {t} | {AL.f3(AL.ci(p1[1]['D']))} | {AL.f3(AL.ci(p1[2]['D']))} | {p1[3]:+.3f} | {k} | {AL.f3(AL.ci(c[k][1]['D'])) if k else '-'} | {R[0]:.2f} [{R[1]:.2f}, {R[2]:.2f}] |")
    if nat:md.append(f"\nNative one pass: D (R1) {AL.f3(AL.ci(nat[1]['D']))}, D (R1-alt) {AL.f3(AL.ci(nat[2]['D']))}")
    groups={}
    for t in rows:groups.setdefault(t.split('x')[0] if M125 else 'b1024',[]).append(t)
    for g,ts in groups.items():
        ok=[t for t in ts if rows[t][0][1][3]<=THR]
        if not ok:md.append(f'\n**{g}: no undamaged lr at pass 1 -> NEGATIVE (damage-limited)**');continue
        sel=max(ok,key=lambda t:float(t.split('x')[-1]));c,k,R=rows[sel]
        a=AL.ci(c[1][2]['D']);nb=AL.ci(nat[2]['D']) if nat else (0,0,0)
        prim=R[0]>=0.30;sec=a[0]>nb[0]+(nb[2]-nb[1])/2;orig=prim and sec
        # amendment 2026-10-16 00:35 KST (before any result): primary = R* only; the one-pass clause is secondary
        md.append(f"\n**{g}: selected lr x{sel.split('x')[-1]}; R* = {R[0]:.2f}; one-pass D_alt {a[0]:+.3f} vs native {nb[0]:+.3f} (secondary clause {'met' if sec else 'not met'})**")
        md.append(f"- amended verdict (primary, R* >= 0.30): {'POSITIVE' if prim else 'NEGATIVE'}\n- original verdict (R* and one-pass clause): {'ORIG_POS' if orig else 'ORIG_NEG'}")
    out=(S.OUT/f"pilot_report_{'125m_' if M125 else ''}s{SEED}.md");out.write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
