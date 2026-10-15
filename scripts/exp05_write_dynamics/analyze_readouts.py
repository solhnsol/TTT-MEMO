"""Exp05-A run3 analysis (CPU only): readout comparison, null calibration, perturbation sign stability.
Decision rules are fixed in experiments/exp05_write_dynamics/run3_readouts/protocol.md (copy: PROTOCOL_run3.md)."""
import os,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('EXP05_OUT',ROOT/'experiments/exp05_write_dynamics/run3_readouts'));RUN2=ROOT/'experiments/exp05_write_dynamics/run2_detops'
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 101
CK={14:15,33:34,62:63}   # chunk k -> state j (= chunks consumed)
rng=np.random.default_rng(7)
def boot(v,n=4000):
    v=np.asarray(v,float);b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(n)]
    return float(v.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def f3(t):return f'{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]'
def load(tag,w):
    p=OUT/f'readouts_seed{SEED}_{tag}_{w}.json';return json.load(open(p)) if p.exists() else None
def M(A,B,key,j,fam,sub=None):
    a=A[key][str(j)] if sub is None else A[key][str(j)][sub];b=B[key][str(j)] if sub is None else B[key][str(j)][sub]
    a=a[fam];b=b[fam];ps=sorted(a,key=lambda s:int(s[1:]));return np.array([(a[p]-b[p])/2 for p in ps])
def main():
    A,B=load('main','A'),load('main','B')
    if A is None or B is None:print('missing main run');return
    md=[f'# Exp05-A run3: readouts and nulls (seed {SEED})\n'];res={}
    # ---- gates ----
    g={'hash_A':A.get('hash_match_run2'),'hash_B':B.get('hash_match_run2')}
    r2={w:json.load(open(RUN2/f'traj_seed{SEED}_full_{w}.json')) for w in 'AB'}
    d0=max(abs(X['short'][j]['L3'][p]-r2[w]['margins_state'][j][p]) for w,X in (('A',A),('B',B)) for j in X['short'] for p in X['short'][j]['L3'])
    # R2 L3 pair margins vs run2 item scores (l3 items are ordered pair by pair, two per pair)
    d2=0.0
    for w,X in (('A',A),('B',B)):
        for k in (14,33,62):
            for mode in ('empty','carry'):
                items=r2[w]['v3'][str(k)][mode];pm={}
                for i,(gg,fo) in enumerate(items):pm.setdefault(f'p{i//2}',[]).append(gg-fo)
                for p,v in pm.items():d2=max(d2,abs(np.mean(v)-X['exp04'][str(k)][mode]['L3'][p]))
    g['R0_L3_vs_run2_maxdiff']=d0;g['R2_L3_vs_run2_maxdiff']=d2
    g['PASS']=bool(g['hash_A'] and g['hash_B'] and d2<=1e-6)   # R0 vs run2 is informational: run2 fused prefix+suffix in one jit (different numerics, see PROTOCOL_run3.md);res['gates']=g
    md.append(f"## Gates\n- stream identical to run2 (64 hashes): A {g['hash_A']}, B {g['hash_B']}\n- R0 L3 margins (batched) vs run2 (unbatched): max diff {d0:.2e} (informational: compilation-path numerical difference, not a gate); R2 L3 vs run2 V3 probes: max diff {d2:.2e}\n- **gates {'PASS' if g['PASS'] else 'FAIL (no interpretation)'}**\n")
    # ---- readout comparison at checkpoints ----
    md.append('## Readout comparison (pair-level binding M=(m_A-m_B)/2, mean over 15 pairs [95% pair bootstrap CI])\n')
    md.append('| k | readout | L3 | XP (cross-pair null) | PS (pseudo null, 16) | D = L3-XP (paired) |\n|---|---|---|---|---|---|')
    rc={}
    for k,j in CK.items():
        rows={'R0 short':('short',None),'R1 neutral':('neutral',None),'R2 exp04 empty':('exp04','empty'),'R2 exp04 carry':('exp04','carry')}
        for name,(key,sub) in rows.items():
            jj=k if key=='exp04' else j
            if str(jj) not in A[key]:continue
            l3=M(A,B,key,jj,'L3',sub);xp=M(A,B,key,jj,'XP',sub);ps=M(A,B,key,jj,'PS',sub)
            rc[(k,name)]={'L3':boot(l3),'XP':boot(xp),'PS':boot(ps),'D':boot(l3-xp),'L3_sd':float(l3.std(ddof=1)),'XP_sd':float(xp.std(ddof=1)),'PS_sd':float(ps.std(ddof=1)),'l3':l3.tolist()}
            r=rc[(k,name)];md.append(f"| {k} | {name} | {f3(r['L3'])} | {f3(r['XP'])} | {f3(r['PS'])} | {f3(r['D'])} |")
    md.append('\nPair-level sd at k=62 (L3 / XP / PS): '+'; '.join(f"{n}: {rc[(62,n)]['L3_sd']:.3f} / {rc[(62,n)]['XP_sd']:.3f} / {rc[(62,n)]['PS_sd']:.3f}" for n in ('R0 short','R1 neutral','R2 exp04 empty','R2 exp04 carry') if (62,n) in rc))
    if (62,'R1 neutral') in rc and (62,'R2 exp04 empty') in rc:
        l1=np.array(rc[(62,'R1 neutral')]['l3']);l2=np.array(rc[(62,'R2 exp04 empty')]['l3']);l0=np.array(rc[(62,'R0 short')]['l3'])
        md.append(f'\nk=62 pair-level correlation of L3 M: R1 vs R2-empty {np.corrcoef(l1,l2)[0,1]:+.2f}, R1 vs R0 {np.corrcoef(l1,l0)[0,1]:+.2f}; ratio of means R2-empty/R1 {l2.mean()/l1.mean() if l1.mean()!=0 else float("nan"):.2f}, R0/R1 {l0.mean()/l1.mean() if l1.mean()!=0 else float("nan"):.2f}')
    res['readouts']={f'{k}|{n}':v for (k,n),v in rc.items()}
    # ---- primary ----
    prim=rc.get((62,'R1 neutral'))
    if prim:
        res['primary_D62']=prim['D'];res['binding_R1']=bool(prim['D'][1]>0)
        md.append(f"\n## Primary (pre-registered): D_62 under R1 = {f3(prim['D'])} => binding under the native-like readout {'DETECTED' if prim['D'][1]>0 else 'NOT detected'} (pending perturbation stability)\n")
    # ---- trajectories under R1: own-exposure DeltaM, L3 minus XP (same name, same exposure event) ----
    ev=r2['A']['events'];NS=len(A['neutral'])
    if NS==64:
        L=np.stack([M(A,B,'neutral',j,'L3') for j in range(NS)]);X=np.stack([M(A,B,'neutral',j,'XP') for j in range(NS)])
        own={}
        for e in ev:   # own exposure of pair p in chunk k (both sentences of the pair; unique (pair,chunk))
            p=int(e['pair'][1:]);own.setdefault((p,e['chunk']),e['round'])
        dL=[];dX=[];rd=[]
        for (p,k),r in sorted(own.items()):dL.append(L[k+1,p]-L[k,p]);dX.append(X[k+1,p]-X[k,p]);rd.append(r)
        dL,dX,rd=map(np.asarray,(dL,dX,rd))
        res['own_R1']={'L3':boot(dL),'XP':boot(dX),'diff':boot(dL-dX),'n':int(len(dL))}
        md.append(f"## R1 trajectory\n- own-exposure DeltaM (n={len(dL)}): L3 {f3(boot(dL))}, XP {f3(boot(dX))}, L3-XP {f3(boot(dL-dX))}")
        md.append('\n| round | n | mean DeltaM L3 | mean DeltaM XP | L3-XP |\n|---|---|---|---|---|')
        per={}
        for r in sorted(set(rd.tolist())):
            s=rd==r;per[r]=(float(dL[s].mean()),float(dX[s].mean()));md.append(f'| {r} | {int(s.sum())} | {dL[s].mean():+.4f} | {dX[s].mean():+.4f} | {(dL[s]-dX[s]).mean():+.4f} |')
        res['own_R1_by_round']=per
        md.append(f'\nFinal M_63 R1: L3 {f3(boot(L[-1]))}, XP {f3(boot(X[-1]))}')

    # ---- familiarity (only if run with --familiarity): per-world weight-stored familiarity, shadow contrast ----
    if NS==64 and 'FN' in A['neutral']['0']:
        md.append('\n## Familiarity under R1 (per world, averaged over A and B; foil = never-exposed name/place)\n')
        fr={}
        def F(fam):   # [state, pair], mean of worlds A and B
            return np.stack([(np.array([A['neutral'][str(j)][fam][q] for q in sorted(A['neutral'][str(j)][fam],key=lambda s:int(s[1:]))])+np.array([B['neutral'][str(j)][fam][q] for q in sorted(B['neutral'][str(j)][fam],key=lambda s:int(s[1:]))]))/2 for j in range(NS)])
        rnd_chunk={}
        for e in ev:rnd_chunk.setdefault((int(e['pair'][1:]),e['round']),e['chunk'])
        for fam,null in (('FN','FN0'),('FP','FP0')):
            Fm=F(fam);F0=F(null)
            by={}
            for (p,r),k in rnd_chunk.items():by.setdefault(r,[]).append(Fm[k+1,p]-Fm[k,p])
            md.append(f"**{fam}** final (state 63) {f3(boot(Fm[-1]))} vs null {fam}0 {f3(boot(F0[-1]))}\n\n| round | mean DeltaF at own exposure |\n|---|---|")
            for r in sorted(by):md.append(f'| {r} | {np.mean(by[r]):+.3f} |')
            # pre-registered shadow contrasts: non-shadowed rounds 2 and 5 vs adjacent shadowed rounds (saturation-controlled)
            d=lambda r:np.array([Fm[rnd_chunk[(p,r)]+1,p]-Fm[rnd_chunk[(p,r)],p] for p in range(Fm.shape[1])])
            c2=d(2)-d(3);c5=d(5)-(d(4)+d(6))/2;cc=(c2+c5)/2
            fr[fam]={'C2':boot(c2),'C5':boot(c5),'C':boot(cc),'by_round':{r:float(np.mean(v)) for r,v in by.items()}}
            md.append(f"\nshadow contrast {fam}: C2 = DeltaF(r2) - DeltaF(r3) {f3(fr[fam]['C2'])}; C5 = DeltaF(r5) - mean(r4,r6) {f3(fr[fam]['C5'])}; **C = (C2+C5)/2 {f3(fr[fam]['C'])}** (M1 for familiarity supported iff C CI > 0)\n")
        res['familiarity']=fr
    # ---- perturbation replicate ----
    PA,PB=load('perturb','A'),load('perturb','B')
    if PA and PB and prim:
        md.append('\n## Perturbation replicate (fixed random direction, relative size %g of the chunk-0 update)\n'%PA['perturb'])
        md.append('| k | pooled D main | pooled D perturbed | pair sign agreement L3 M | pair corr L3 M |\n|---|---|---|---|---|');pr={}
        for k,j in CK.items():
            if str(j) not in PA['neutral']:continue
            a=M(A,B,'neutral',j,'L3');b=M(PA,PB,'neutral',j,'L3');xa=M(A,B,'neutral',j,'XP');xb=M(PA,PB,'neutral',j,'XP')
            da=boot(a-xa);db=boot(b-xb);agr=float(np.mean(np.sign(a)==np.sign(b)));cr=float(np.corrcoef(a,b)[0,1])
            pr[k]={'D_main':da,'D_pert':db,'sign_agree':agr,'corr':cr};md.append(f'| {k} | {f3(da)} | {f3(db)} | {agr:.2f} | {cr:+.2f} |')
        res['perturb']=pr;p62=pr.get(62)
        if p62:
            stable=bool(np.sign(p62['D_main'][0])==np.sign(p62['D_pert'][0]) and abs(p62['D_pert'][0]-p62['D_main'][0])<=0.5*abs(p62['D_main'][0]))
            res['D62_stable']=stable;res['pair_level_interpretable']=bool(p62['sign_agree']>=0.8)
            md.append(f"\n- pooled D_62 stable (same sign, |change| <= 50%): **{stable}**; pair-level M interpretable (sign agreement >= 0.80): **{res['pair_level_interpretable']}**")
            if 'binding_R1' in res:md.append(f"- **Verdict: binding under R1 = {res['binding_R1'] and stable}**")
    json.dump(res,open(OUT/f'run3_metrics_seed{SEED}.json','w'),indent=1,default=str)
    (OUT/f'run3_report_seed{SEED}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
