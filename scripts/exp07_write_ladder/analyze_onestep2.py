"""One-step replication gate and suppression screen (PROTOCOL_stage01.md 2026-10-16). Usage: analyze_onestep2.py gate|screen"""
import sys,json,importlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));THR=0.10;mode=sys.argv[1]
def mods(sd):
    sys.argv=['x',str(sd)]
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
def pts(S,AL,tag,pre='OS2'):
    """{s: (mean D_alt, ci, mean D_R1, canary rise, F_name R1, F_name alt)} or None"""
    X=S.load(pre,tag)
    if not X:return None
    out={};raw={w:json.load(open(S.OUT/f'{pre}_{tag}_s{S.SEED}_{w}.json')) for w in 'AB'}
    for i,(ep,s1,sa,d) in enumerate(S.curve(*X)[1:],1):
        pw={w:max(raw[w]['log'][i]['margins'][k]-raw[w]['log'][0]['margins'][k] for k in ('_gen_nll','_tail_nll')) for w in 'AB'}
        sn={w:raw[w]['log'][i]['step']*raw[w]['meta']['vnorm'] for w in 'AB'}
        out[ep]=dict(Dalt=float(sa['D'].mean()),ci=AL.ci(sa['D']),DR1=float(s1['D'].mean()),rise=d,Fn=float(s1['F_name'].mean()),Fn_alt=float(sa['F_name'].mean()),rise_w=pw,step_w=sn)
    return out
def row(tag,p):return [f"| {tag} | {s} | {AL.f3(v['ci'])} | {v['DR1']:+.3f} | {v['Fn']:+.3f} / {v['Fn_alt']:+.3f} | {v['rise']:+.3f}{' (damaged)' if v['rise']>THR else ''} | {v['rise_w']['A']:+.3f} / {v['rise_w']['B']:+.3f} | {v['step_w']['A']:.3f} / {v['step_w']['B']:.3f} |" for s,v in p.items()]
H=['(gb_nm = superseded per-world-normalised g_b, diagnostic only; gbc_nm = common-scale norm basis, the screen reference)\n','| signal | s | D (R1-alt) [CI] | D (R1) | familiarity F_name R1 / alt | max canary rise | canary rise A / B | step norm A / B |','|---|---|---|---|---|---|---|---|']
if mode=='gate':
    md=['# One-step replication gate (g / g_b, s102 and s103)\n'];res={}
    for sd in (101,102,103):
        AL,S=mods(sd);md+=[f'\n## seed {sd}\n']+H;P={t:pts(S,AL,t) for t in ('g_lr','gb_lr','gb_nm','gbc_nm')}
        for t,p in P.items():
            if p:md+=row(t,p)
        p=P['gb_lr']
        if sd==101:continue
        if not p or len(p)<4:res[sd]=False;md.append(f'\n- seed {sd}: g_b (lr basis) incomplete -> not replicated');continue
        c1=all(v['rise']<=THR for v in p.values());c2=all(v['ci'][1]>0 for v in p.values());c3=p[8]['Dalt']/p[1]['Dalt']>=4 if p[1]['Dalt']>0 else False
        res[sd]=c1 and c2 and c3;md.append(f'\n- seed {sd}: no damage {c1}; CI lower > 0 at every s {c2}; D(8)/D(1) = {p[8]["Dalt"]/p[1]["Dalt"]:.2f} >= 4 {c3} -> {"replicated" if res[sd] else "NOT replicated"}')
    md.append(f"\n**GATE {'PASS' if all(res.get(s,False) for s in (102,103)) else 'FAIL'}**")
elif mode=='screen':
    md=['# Suppression-signal screen (W0 one step)\n'];per={}
    for sd in (101,102,103):
        AL,S=mods(sd);md+=[f'\n## seed {sd}\n']+H;P={}
        for t in ('g_lr','gbc_nm','gntpch_nm')+tuple(f'{c}{cond}_{b}' for c in ('c1','c2') for cond in ('iso','ch') for b in ('nm','lr')):
            P[t]=pts(S,AL,t)
            if P[t]:md+=row(t,P[t])
        g=P['g_lr'] or {};gmax=max([v['Dalt'] for v in g.values() if v['rise']<=THR],default=float('nan'));gb=P['gbc_nm'] or {}
        for c in ('c1','c2'):
            for cond in ('ch','iso'):
                p=P[f'{c}{cond}_nm'] or {};ok=[(s,v) for s,v in p.items() if v['rise']<=THR]
                if not ok:per[(sd,c,cond)]=dict(dmax=None,pass_=False,kill=True);md.append(f'- {c} {cond}: no damage-free scale');continue
                s_,v=max(ok,key=lambda x:x[1]['Dalt']);ref=gb[s_]['Dalt'] if s_ in gb else (max(x['Dalt'] for x in gb.values()) if gb else float('nan'))
                pr=v['Dalt']>=0.3*ref;fl=v['Dalt']>=2*gmax;per[(sd,c,cond)]=dict(dmax=v['Dalt'],s=s_,ref=ref,gmax=gmax,pass_=pr and fl,kill=v['Dalt']<1.5*gmax)
                md.append(f"- {c} {cond}: max damage-free D_alt {v['Dalt']:+.3f} at s={s_}; g_b (common-scale norm basis) at that s {ref:+.3f} -> ratio {v['Dalt']/ref:.2f} (>= 0.3: {pr}); g max {gmax:+.3f} -> x{v['Dalt']/gmax:.2f} (>= 2: {fl})")
    md.append('\n## Verdict (chunk-only condition decides)')
    anyp=False
    for c in ('c1','c2'):
        n=sum(per.get((sd,c,'ch'),{}).get('pass_',False) for sd in (101,102,103));kill=all(per.get((sd,c,'ch'),{}).get('kill',True) for sd in (101,102,103))
        ni=sum(per.get((sd,c,'iso'),{}).get('pass_',False) for sd in (101,102,103));anyp|=n>=2
        md.append(f"- {c}: chunk-only passes on {n}/3 seeds -> {'PASS' if n>=2 else ('KILL' if kill else 'FAIL')}; isolated-sentence condition passes on {ni}/3 (reported)")
    md.append(f"\n**SCREEN {'PASS' if anyp else 'FAIL'}**")
elif mode=='lfscreen':   # label-free suppression screen v1 (PROTOCOL 2026-10-16 22:15 KST), chunk condition only, two-part threshold
    md=['# Label-free suppression screen (W0 one step, chunk condition; LF-swap decides A-method)\n'];per={}
    for sd in (101,102,103):
        AL,S=mods(sd);md+=[f'\n## seed {sd}\n']+H;P={'g_lr':pts(S,AL,'g_lr'),'c2ch_nm':pts(S,AL,'c2ch_nm')}
        for t in ('gbc_nm',)+tuple(f'{c}_{b}' for c in ('lfswap','bridge','lfablate') for b in ('nm','lr')):
            P[t]=pts(S,AL,t,'OS2LF')
            if P[t]:md+=row('LF '+t,P[t])
        g=P['g_lr'] or {};gmax=max([v['Dalt'] for v in g.values() if v['rise']<=THR],default=float('nan'));gb=P['gbc_nm'] or {}
        gbmax=max([v['Dalt'] for v in gb.values() if v['rise']<=THR],default=float('nan'))
        c2=P['c2ch_nm'] or {};c2max=max([v['Dalt'] for v in c2.values() if v['rise']<=THR],default=float('nan'))
        for c in ('lfswap','bridge','lfablate'):
            p=P[f'{c}_nm'] or {};ok=[(s,v) for s,v in p.items() if v['rise']<=THR]
            if not ok:per[(sd,c)]=dict(pass_=False);md.append(f'- {c}: no damage-free scale');continue
            s_,v=max(ok,key=lambda x:x[1]['Dalt']);ref=gb[s_]['Dalt'] if s_ in gb else float('nan')
            i1=v['Dalt']>=0.3*ref;i2=v['Dalt']>=0.3*gbmax;fl=v['Dalt']>=2*gmax;per[(sd,c)]=dict(pass_=i1 and i2 and fl)
            md.append(f"- {c}: max damage-free D_alt {v['Dalt']:+.3f} at s={s_}; (i) vs gbc_nm at that s {ref:+.3f}: {i1}; (ii) vs 0.3 x g_b max damage-free {gbmax:+.3f}: {i2}; floor 2 x g max {gmax:+.3f}: {fl}; retention vs c2ch max {c2max:+.3f}: {v['Dalt']/c2max:.2f} -> {'pass' if per[(sd,c)]['pass_'] else 'fail'}")
    md.append('\n## Verdict')
    for c in ('lfswap','bridge','lfablate'):
        n=sum(per.get((sd,c),{}).get('pass_',False) for sd in (101,102,103))
        md.append(f"- {c}: passes on {n}/3 seeds -> {'PASS' if n>=2 else 'FAIL'}{' (decides: A-method if PASS, A-probe if FAIL)' if c=='lfswap' else ' (descriptive)'}")
    n=sum(per.get((sd,'lfswap'),{}).get('pass_',False) for sd in (101,102,103))
    md.append(f"\n**LF-SWAP {'PASS -> A-method' if n>=2 else 'FAIL -> A-probe'}**")
elif mode=='lfv2':   # lfv2: label-free v2 (PROTOCOL 2026-10-17 00:33 KST): judging cells sel, nerent; descriptive bridgeP, selgold
    md=['# Label-free v2 screen (W0 one step, chunk condition; judging cells sel and nerent; candidate -> confirmation on s104/105)\n'];per={}
    for sd in (101,102,103):
        AL,S=mods(sd);md+=[f'\n## seed {sd}\n']+H;P={'g_lr':pts(S,AL,'g_lr'),'c2ch_nm':pts(S,AL,'c2ch_nm')}
        for t in ('gbc_nm',)+tuple(f'{c}_{b}' for c in ('sel','nerent','bridgeP','selgold') for b in ('nm','lr')):
            P[t]=pts(S,AL,t,'OS2V2')
            if P[t]:md+=row('LF '+t,P[t])
        g=P['g_lr'] or {};gmax=max([v['Dalt'] for v in g.values() if v['rise']<=THR],default=float('nan'));gb=P['gbc_nm'] or {}
        gbmax=max([v['Dalt'] for v in gb.values() if v['rise']<=THR],default=float('nan'))
        c2=P['c2ch_nm'] or {};c2max=max([v['Dalt'] for v in c2.values() if v['rise']<=THR],default=float('nan'))
        for c in ('sel','nerent','bridgeP','selgold'):
            p=P[f'{c}_nm'] or {};ok=[(s,v) for s,v in p.items() if v['rise']<=THR]
            if not ok:per[(sd,c)]=dict(pass_=False);md.append(f'- {c}: no damage-free scale');continue
            s_,v=max(ok,key=lambda x:x[1]['Dalt']);ref=gb[s_]['Dalt'] if s_ in gb else float('nan')
            i1=v['Dalt']>=0.3*ref;i2=v['Dalt']>=0.3*gbmax;fl=v['Dalt']>=2*gmax;per[(sd,c)]=dict(pass_=i1 and i2 and fl)
            md.append(f"- {c}: max damage-free D_alt {v['Dalt']:+.3f} at s={s_}; (i) vs gbc_nm at that s {ref:+.3f}: {i1}; (ii) vs 0.3 x g_b max damage-free {gbmax:+.3f}: {i2}; floor 2 x g max {gmax:+.3f}: {fl}; retention vs c2ch max {c2max:+.3f}: {v['Dalt']/c2max:.2f} -> {'pass' if per[(sd,c)]['pass_'] else 'fail'}")
    md.append('\n## Verdict')
    for c in ('sel','nerent','bridgeP','selgold'):
        n=sum(per.get((sd,c),{}).get('pass_',False) for sd in (101,102,103))
        md.append(f"- {c}: passes on {n}/3 seeds -> {('CANDIDATE (needs 2/2 on s104/105)' if n>=2 else 'FAIL') if c in ('sel','nerent') else ('meets' if n>=2 else 'fails')+' (descriptive)'}")
    cand=[c for c in ('sel','nerent') if sum(per.get((sd,c),{}).get('pass_',False) for sd in (101,102,103))>=2]
    md.append(f"\n**V2: {'CANDIDATE(S) '+', '.join(cand) if cand else 'NO CANDIDATE -> A-probe stands'}**")
else:   # keystat stage 1 (PROTOCOL KEYSTAT/PoE v1): s101, all cells; per family the best by max damage-free D_alt / max(thresholds)
    seeds=[int(x) for x in (sys.argv[2].split(',') if len(sys.argv)>2 else ['101'])]
    md=['# KEYSTAT/PoE 1-step screen (W0 one step, chunk condition)\n'];FAM={'P1':['a','a_s','b1','b4','b16','c_hi','c_lo'],'P3':['p05','p10']};ORDER=['a','a_s','b1','b4','b16','c_hi','c_lo','p05','p10']
    for sd in seeds:
        AL,S=mods(sd);md+=[f'\n## seed {sd}\n']+H;P={'g_lr':pts(S,AL,'g_lr')}
        for t in ('gbc_nm',)+tuple(f'{c}_{b}' for c in ['c0']+ORDER for b in ('nm','lr')):
            P[t]=pts(S,AL,t,'OS2KS')
            if P[t]:md+=row('KS '+t,P[t])
        g=P['g_lr'] or {};gmax=max([v['Dalt'] for v in g.values() if v['rise']<=THR],default=float('nan'));gb=P['gbc_nm'] or {}
        gbmax=max([v['Dalt'] for v in gb.values() if v['rise']<=THR],default=float('nan'))
        def best(c):
            p=P.get(f'{c}_nm') or {};ok=[(s_,v) for s_,v in p.items() if v['rise']<=THR]
            return max(ok,key=lambda x:x[1]['Dalt']) if ok else (None,None)
        _,v0=best('c0');c0max=v0['Dalt'] if v0 else float('nan');rows=[]
        for c in ORDER:
            s_,v=best(c)
            if v is None:md.append(f'- {c}: no damage-free scale');continue
            ref=gb[s_]['Dalt'] if s_ in gb else float('nan');thr=max(0.3*ref,0.3*gbmax,2*gmax)
            i1=v['Dalt']>=0.3*ref;i2=v['Dalt']>=0.3*gbmax;fl=v['Dalt']>=2*gmax;att=v['Dalt']>c0max;kill=v['Dalt']>=1.5*gmax
            rows.append((c,v['Dalt']/thr,kill,i1 and i2 and fl and att))
            md.append(f"- {c}: max damage-free D_alt {v['Dalt']:+.3f} at s={s_}; (i) {i1} (ii) {i2} floor {fl} attribution (> c0 {c0max:+.3f}) {att}; score {v['Dalt']/thr:.2f}; above kill line (1.5 g max {1.5*gmax:+.3f}) {kill} -> {'meets all' if i1 and i2 and fl and att else 'fails'}")
        if sd==101:
            md.append('\n## Stage 1 selection (s101)')
            for fam,cells in FAM.items():
                cand=[r for r in rows if r[0] in cells and r[2]]
                if not cand:md.append(f'- {fam}: no cell above the kill line -> FAMILY FAILS');continue
                top=max(r[1] for r in cand);bestc=[r for r in cand if abs(r[1]-top)<1e-12];pick=min(bestc,key=lambda r:ORDER.index(r[0]))
                md.append(f"- {fam}: advances {pick[0]} (score {pick[1]:.2f}; meets all criteria on s101: {pick[3]}); hyperparameters frozen before s102/103")
OUT=mods(101)[1].OUT;(OUT/f'onestep2_{mode}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
