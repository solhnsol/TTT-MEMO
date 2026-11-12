"""Exp09 G2 ① judgement (PROTOCOL_stage01.md "Qwen post-G1 runs and G2 ① judgement", 2026-10-16 20:20 KST + 20:52), CPU.
Inputs: per seed a directory with the a/b/c/d/e(/f) logs (s101: a = a2, b = b1 from experiments/exp09_qwen_ttt/g2, the rest
from g2_pod_s101). Pair ids from the seed's meta (the meta of any level has the same items).
Test 1: pairwise b - a D_alt (R1-alt, pass 1), 45 pairs pooled (15 per seed), 97.5% bootstrap CI, pass iff lower > 0;
  per-seed signs; s102+s103-only sensitivity.
Test 2: per seed, d at k* (last pass with canary <= 0.10, all earlier passes too) vs, per pair, the max D_alt over e's passes p
  with p*1.5 <= k* * 2.5 and canary <= 0.10 (the pass maximising the mean is taken, same pass for every pair of that seed);
  pairwise d - e, pooled, 97.5% CI, pass iff lower > 0; k* = 1 on every seed -> not applicable.
Multi-pass rule for d (per seed): k* >= 2; D_alt(k*) CI lower > 0; no pass-to-pass drop of mean D_alt larger than that pass's
  95% CI half-width up to k*; long-NLL beyond-8K rise at k* <= 0.10 (max over A/B).
Usage: analyze_g2.py OUTDIR SEED:DIR[:atag:btag] ..."""
import os,sys,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
F_NAT,F_MFS=1.5,2.5
def load_seed(seed,d,atag='a',btag='b'):
    d=Path(d);meta=next(d.glob(f'meta_*_s{seed}.json'));ref=d/f'meta_L1_g2items_s{seed}.json'
    if not ref.exists():ref.write_text(meta.read_text())
    os.environ['EXP07_OUT']=str(d);saved=sys.argv;sys.argv=['x',str(seed),'g2items']
    for m in ('analyze_ladder','analyze_stage01'):sys.modules.pop(m,None)
    sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
    try:import analyze_ladder as AL,analyze_stage01 as S
    finally:sys.argv=saved
    out={}
    for k,(lev,tag) in {'a':('L0mp',atag),'b':('L0mw',btag),'c':('L0mw','c'),'d':('L0mwfs','d'),'e':('L0mp','e'),'f':('L1','f')}.items():
        X=S.load(lev,tag)
        if X:out[k]=(S.curve(*X),X)
    return AL,out
def boot(v,n=8000,lvl=float(os.environ.get('G2_LVL','97.5')),seed=0):
    v=np.asarray(v);r=np.random.default_rng(seed);b=[v[r.integers(0,len(v),len(v))].mean() for _ in range(n)];a=(100-lvl)/2
    return float(v.mean()),float(np.percentile(b,a)),float(np.percentile(b,100-a))
def f3(t):return f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
def main():
    out=Path(sys.argv[1]);specs=sys.argv[2:];md=['# Qwen G2 ① (s'+','.join(s.split(':')[0] for s in specs)+')\n'];res={'seeds':{}}
    t1={};t2={};mp={}
    for sp in specs:
        p=sp.split(':');seed=int(p[0]);AL,R=load_seed(seed,p[1],*(p[2:4] if len(p)>2 else ()))
        ca=R['a'][0];cb=R['b'][0];da=ca[1][2]['D'];db=cb[1][2]['D'];t1[seed]=db-da
        cd=R['d'][0];canary=[x[3] for x in cd];ks=0
        for i in range(1,len(cd)):
            if canary[i]<=0.10:ks=i
            else:break
        Dd=[AL.ci(x[2]['D']) for x in cd]
        lA=[e.get('longnll') for e in R['d'][1][0]];lB=[e.get('longnll') for e in R['d'][1][1]]
        rise=[max(lA[i]['beyond']-lA[0]['beyond'],lB[i]['beyond']-lB[0]['beyond']) if lA[i] and lB[i] else None for i in range(len(cd))]
        drop_ok=all(not (Dd[i][0]<Dd[i-1][0]-(Dd[i][2]-Dd[i][1])/2) for i in range(2,ks+1))
        mp[seed]={'k*':ks,'D_alt_k*':Dd[ks] if ks else None,'longnll_rise_k*':rise[ks] if ks else None,
                  'pass':bool(ks>=2 and Dd[ks][1]>0 and drop_ok and rise[ks] is not None and rise[ks]<=0.10),'no_large_drop':drop_ok}
        ce=R['e'][0];ok_e=[]
        for i in range(1,len(ce)):
            if ce[i][3]>0.10:break
            if i*F_NAT<=ks*F_MFS:ok_e.append(i)
        if ks>=1 and ok_e:
            best=max(ok_e,key=lambda i:ce[i][2]['D'].mean());t2[seed]=(cd[ks][2]['D']-ce[best][2]['D'],ks,best)
        md.append(f"\n## seed {seed}\n- a (native 1 pass) D_alt {f3(AL.ci(da))}, b (masked 1 pass) {f3(AL.ci(db))}, b-a {f3(AL.ci(db-da))}, sign {'+' if (db-da).mean()>0 else '-'}")
        if 'c' in R:md.append(f"- c (masked lr 0.548, secondary) D_alt {f3(AL.ci(R['c'][0][1][2]['D']))}, canary {R['c'][0][1][3]:+.3f}")
        md.append(f"- d (L0mw-fs): " + ', '.join(f"p{i} {Dd[i][0]:+.3f} (c {canary[i]:+.3f}, lnll {rise[i]:+.3f})" if rise[i] is not None else f"p{i} {Dd[i][0]:+.3f} (c {canary[i]:+.3f})" for i in range(1,len(cd))))
        md.append(f"- d multi-pass rule: k* = {ks}, D_alt(k*) {f3(Dd[ks]) if ks else '-'}, no large drop {drop_ok}, long-NLL rise at k* {rise[ks] if ks else None} -> **{'PASS' if mp[seed]['pass'] else 'FAIL'}**")
        md.append(f"- e (native multi-pass): " + ', '.join(f"p{i} {ce[i][2]['D'].mean():+.3f} (c {ce[i][3]:+.3f})" for i in range(1,len(ce))))
        if seed in t2:md.append(f"- test-2 comparison: d at k*={ks} vs e at pass {t2[seed][2]} (cumulative FLOPs {t2[seed][2]*F_NAT:.1f}F vs k*·F_mfs {ks*F_MFS:.1f}F per chunk, ratio {t2[seed][2]*F_NAT/(ks*F_MFS):.2f}); d-e {f3(AL.ci(t2[seed][0]))}")
        if 'f' in R:md.append(f"- f (L1): "+', '.join(f"ep{x[0]} {x[2]['D'].mean():+.3f}" for x in R['f'][0]))
        res['seeds'][seed]={'b_minus_a':AL.ci(db-da),'multipass':mp[seed],'test2':(AL.ci(t2[seed][0]),t2[seed][1],t2[seed][2]) if seed in t2 else None}
    v=np.concatenate([t1[s] for s in t1]);T1=boot(v);ok1=T1[1]>0
    md.append(f"\n## Test 1 (b - a, D_alt pass 1, {len(v)} pairs pooled, 97.5% CI): {f3(T1)} -> **{'PASS' if ok1 else 'FAIL'}**")
    md.append('- per-seed signs: '+', '.join(f"s{s} {'+' if t1[s].mean()>0 else '-'}{abs(t1[s].mean()):.3f}" for s in t1))
    sens=[s for s in t1 if s!=101]
    if sens:md.append(f"- sensitivity (s{'+s'.join(map(str,sens))} only): {f3(boot(np.concatenate([t1[s] for s in sens])))}")
    if not t2 or all(res['seeds'][s]['multipass']['k*']<=1 for s in res['seeds']):T2=None;ok2=None;md.append('\n## Test 2: not applicable (k* = 1)')
    else:
        v2=np.concatenate([t2[s][0] for s in t2]);T2=boot(v2);ok2=T2[1]>0
        md.append(f"\n## Test 2 (Pareto, d at k* minus FLOP-matched e, {len(v2)} pairs pooled, 97.5% CI): {f3(T2)} -> **{'PASS' if ok2 else 'FAIL'}**")
    g2=bool(ok1 or ok2);md.append(f"\n**G2 ①: {'PASS' if g2 else 'FAIL'}** (test 1 {'PASS' if ok1 else 'FAIL'}, test 2 {'n/a' if ok2 is None else ('PASS' if ok2 else 'FAIL')})")
    res.update(test1=T1,test1_pass=ok1,test2=T2,test2_pass=ok2,G2=g2)
    out.mkdir(parents=True,exist_ok=True);json.dump(res,open(out/'g2_result.json','w'),indent=1,default=str);(out/'g2_report.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
