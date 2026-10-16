"""One-step replication gate and suppression screen (PROTOCOL_stage01.md 2026-10-16). Usage: analyze_onestep2.py gate|screen"""
import sys,json,importlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));THR=0.10;mode=sys.argv[1]
def mods(sd):
    sys.argv=['x',str(sd)]
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
def pts(S,AL,tag):
    """{s: (mean D_alt, ci, mean D_R1, canary rise, F_name R1, F_name alt)} or None"""
    X=S.load('OS2',tag)
    if not X:return None
    out={}
    for ep,s1,sa,d in S.curve(*X)[1:]:out[ep]=dict(Dalt=float(sa['D'].mean()),ci=AL.ci(sa['D']),DR1=float(s1['D'].mean()),rise=d,Fn=float(s1['F_name'].mean()),Fn_alt=float(sa['F_name'].mean()))
    return out
def row(tag,p):return [f"| {tag} | {s} | {AL.f3(v['ci'])} | {v['DR1']:+.3f} | {v['Fn']:+.3f} / {v['Fn_alt']:+.3f} | {v['rise']:+.3f}{' (damaged)' if v['rise']>THR else ''} |" for s,v in p.items()]
H=['| signal | s | D (R1-alt) [CI] | D (R1) | familiarity F_name R1 / alt | max canary rise |','|---|---|---|---|---|---|']
if mode=='gate':
    md=['# One-step replication gate (g / g_b, s102 and s103)\n'];res={}
    for sd in (101,102,103):
        AL,S=mods(sd);md+=[f'\n## seed {sd}\n']+H;P={t:pts(S,AL,t) for t in ('g_lr','gb_lr','gb_nm')}
        for t,p in P.items():
            if p:md+=row(t,p)
        p=P['gb_lr']
        if sd==101:continue
        if not p or len(p)<4:res[sd]=False;md.append(f'\n- seed {sd}: g_b (lr basis) incomplete -> not replicated');continue
        c1=all(v['rise']<=THR for v in p.values());c2=all(v['ci'][1]>0 for v in p.values());c3=p[8]['Dalt']/p[1]['Dalt']>=4 if p[1]['Dalt']>0 else False
        res[sd]=c1 and c2 and c3;md.append(f'\n- seed {sd}: no damage {c1}; CI lower > 0 at every s {c2}; D(8)/D(1) = {p[8]["Dalt"]/p[1]["Dalt"]:.2f} >= 4 {c3} -> {"replicated" if res[sd] else "NOT replicated"}')
    md.append(f"\n**GATE {'PASS' if all(res.get(s,False) for s in (102,103)) else 'FAIL'}**")
else:
    md=['# Suppression-signal screen (W0 one step)\n'];per={}
    for sd in (101,102,103):
        AL,S=mods(sd);md+=[f'\n## seed {sd}\n']+H;P={}
        for t in ('g_lr','gb_nm','gntpch_nm')+tuple(f'{c}{cond}_{b}' for c in ('c1','c2') for cond in ('iso','ch') for b in ('nm','lr')):
            P[t]=pts(S,AL,t)
            if P[t]:md+=row(t,P[t])
        g=P['g_lr'] or {};gmax=max([v['Dalt'] for v in g.values() if v['rise']<=THR],default=float('nan'));gb=P['gb_nm'] or {}
        for c in ('c1','c2'):
            for cond in ('ch','iso'):
                p=P[f'{c}{cond}_nm'] or {};ok=[(s,v) for s,v in p.items() if v['rise']<=THR]
                if not ok:per[(sd,c,cond)]=dict(dmax=None,pass_=False,kill=True);md.append(f'- {c} {cond}: no damage-free scale');continue
                s_,v=max(ok,key=lambda x:x[1]['Dalt']);ref=gb[s_]['Dalt'] if s_ in gb else (max(x['Dalt'] for x in gb.values()) if gb else float('nan'))
                pr=v['Dalt']>=0.3*ref;fl=v['Dalt']>=2*gmax;per[(sd,c,cond)]=dict(dmax=v['Dalt'],s=s_,ref=ref,gmax=gmax,pass_=pr and fl,kill=v['Dalt']<1.5*gmax)
                md.append(f"- {c} {cond}: max damage-free D_alt {v['Dalt']:+.3f} at s={s_}; g_b (norm-matched) at that s {ref:+.3f} -> ratio {v['Dalt']/ref:.2f} (>= 0.3: {pr}); g max {gmax:+.3f} -> x{v['Dalt']/gmax:.2f} (>= 2: {fl})")
    md.append('\n## Verdict (chunk-only condition decides)')
    anyp=False
    for c in ('c1','c2'):
        n=sum(per.get((sd,c,'ch'),{}).get('pass_',False) for sd in (101,102,103));kill=all(per.get((sd,c,'ch'),{}).get('kill',True) for sd in (101,102,103))
        ni=sum(per.get((sd,c,'iso'),{}).get('pass_',False) for sd in (101,102,103));anyp|=n>=2
        md.append(f"- {c}: chunk-only passes on {n}/3 seeds -> {'PASS' if n>=2 else ('KILL' if kill else 'FAIL')}; isolated-sentence condition passes on {ni}/3 (reported)")
    md.append(f"\n**SCREEN {'PASS' if anyp else 'FAIL'}**")
OUT=mods(101)[1].OUT;(OUT/f'onestep2_{mode}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
