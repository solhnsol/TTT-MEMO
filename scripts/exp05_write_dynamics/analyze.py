"""Exp05-A analysis (CPU only). Applies the decision rules fixed in protocol.md."""
import os,sys,json,types,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
core=types.ModuleType('core');core.CH=1024;core.clone_state=lambda x:x;core.jax=types.SimpleNamespace(tree_util=None);core.jnp=None;core.eqx=None;sys.modules['core']=core
import stage2,gen_world as gw
OUT=Path(os.environ['EXP05_OUT']) if os.environ.get('EXP05_OUT') else ROOT/'experiments/exp05_write_dynamics';EXP4=ROOT/'experiments/exp04_what_ttt_retains/stage2'
CH=1024;SEED=int(sys.argv[1]) if len(sys.argv)>1 else 101;TAG=sys.argv[2] if len(sys.argv)>2 else 'full'
rng=np.random.default_rng(5)
def boot(v,n=4000,clusters=None):
    v=np.asarray(v,float)
    if len(v)==0:return (float('nan'),)*3
    if clusters is None:
        b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(n)]
    else:
        cl=np.asarray(clusters);u=sorted(set(cl.tolist()));idx={c:np.where(cl==c)[0] for c in u};b=[]
        for _ in range(n):
            pick=rng.choice(len(u),len(u));vals=np.concatenate([v[idx[u[i]]] for i in pick]);b.append(vals.mean())
    return float(v.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def f3(t):return f'{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]'
def load(w,tag=TAG):
    p=OUT/f'traj_seed{SEED}_{tag}_{w}.json'
    return json.load(open(p)) if p.exists() else None
def main():
    A,B=load('A'),load('B');R=load('A','repeat');md=['# Exp05-A report (auto-generated from traj_seed%d_%s_{A,B}.json)\n'%(SEED,TAG)];res={}
    if A is None or B is None:print('missing');return
    rt=types.SimpleNamespace(tok=gw.TOK,bos=128000,eos=128001);w=gw.build_world(SEED,v=2);PA=stage2.Probe(rt,w,False);PB=stage2.Probe(rt,w,True)
    NS=len(A['margins_state']);NC=NS-1
    pairs=sorted({k for k in A['margins_state']['0']});cp=sorted({k for k in A['ctrl_state']['0']})
    def traj(Aw,Bw,key,names):
        return {p:np.array([(Aw[key][str(j)][p]-Bw[key][str(j)][p])/2 for j in range(NS)]) for p in names}
    M=traj(A,B,'margins_state',pairs);MC=traj(A,B,'ctrl_state',cp)
    dM={p:np.diff(M[p]) for p in pairs};dC={p:np.diff(MC[p]) for p in cp}
    theta=float(np.percentile(np.abs(np.concatenate(list(dC.values()))),95))
    res['theta']=theta;res['control_dM_abs_p95']=theta;res['control_dM_sd']=float(np.std(np.concatenate(list(dC.values()))))
    # ---------- validations ----------
    md.append('## Validation\n')
    v2=max(max(r['v2_rel_err'] for r in A['rows']),max(r['v2_rel_err'] for r in B['rows']));res['V2_max_rel_err']=v2
    md.append(f'- **V2** own re-implementation of the native inner step (value_and_grad of the model loss, global-norm clip, SGD, outside e2e) vs native weights, 63 chunks x 2 worlds: max relative error **{v2:.2e}** (criterion <= 1e-6): {"PASS" if v2<=1e-6 else "FAIL"}')
    v1pass=None
    if R is not None:
        hash_eq=(R['hashes']==A['hashes']);mR=traj(R,B,'margins_state',pairs) if False else None
        dmax=max(abs(R['margins_state'][str(j)][p]-A['margins_state'][str(j)][p]) for j in range(NS) for p in pairs)
        dmaxc=max(abs(R['ctrl_state'][str(j)][p]-A['ctrl_state'][str(j)][p]) for j in range(NS) for p in cp)
        nll=float(np.max(np.abs(np.load(OUT/f'traj_seed{SEED}_repeat_A_tok_nll.npz')['tok_nll'].astype(np.float32)-np.load(OUT/f'traj_seed{SEED}_{TAG}_A_tok_nll.npz')['tok_nll'].astype(np.float32))))
        gdiff=max(abs(a['grad_norm_preclip']-b['grad_norm_preclip']) for a,b in zip(R['rows'],A['rows']))
        v1pass=bool(hash_eq and dmax<=(1e-6 if 'deterministic_ops=true' in (A.get('xla_flags') or '') else 1e-3));res['V1']={'hashes_identical':hash_eq,'max_abs_diff_margin':dmax,'max_abs_diff_ctrl_margin':dmaxc,'max_abs_diff_token_nll_fp16':nll,'max_abs_diff_gradnorm':gdiff}
        md.append(f'- **V1** determinism (World A repeated end-to-end): all {len(A["hashes"])} weight hashes identical = **{hash_eq}**; max |diff| of pair-level m over all states {dmax:.2e} (control {dmaxc:.2e}); token-NLL (stored fp16) {nll:.2e}; grad-norm {gdiff:.2e}: {"PASS" if v1pass else "FAIL"}')
    else:md.append('- **V1** not available (repeat run missing)')
    if R is not None:
        dd=np.array([R['margins_state'][str(j)][p]-A['margins_state'][str(j)][p] for j in range(1,NS) for p in pairs]);dc=np.array([R['ctrl_state'][str(j)][p]-A['ctrl_state'][str(j)][p] for j in range(1,NS) for p in cp])
        # run-to-run noise of one world's pair margin, and of M=(mA-mB)/2 if the noise of two worlds is independent
        res['V1_noise']={'sd_pair_margin_between_runs':float(dd.std()),'p95_abs':float(np.percentile(np.abs(dd),95)),'sd_ctrl_margin_between_runs':float(dc.std()),'implied_sd_of_M':float(dd.std()/np.sqrt(2)),'theta':theta}
        md.append(f'  - run-to-run noise of one world\'s pair margin (states 1-63): sd {dd.std():.4f}, 95th percentile |diff| {np.percentile(np.abs(dd),95):.4f} (control items sd {dc.std():.4f}); implied sd of M=(mA-mB)/2 ~ {dd.std()/np.sqrt(2):.4f}; compare theta {theta:.4f}, control per-chunk DeltaM sd {res["control_dM_sd"]:.4f} and mean own-exposure DeltaM (below).')
        md.append('  - **Cause (diagnostic, `diag_determinism_*.json`, no trajectory data):** the forward path is bitwise reproducible (state-0 probes, chunk-0 pre-update NLL and the Exp04 reproduction differ by exactly 0) and, inside ONE process, repeated gradient/update computations are bitwise identical (relative difference 0, with or without deterministic ops). ACROSS processes with default XLA-GPU settings the chunk-0 native update differs by a relative L2 of ~1e-2 (cosine 0.99995; pre-clip gradient norm 0.92045 / 0.92068 in different processes), so the weights diverge from state 1 on. With `XLA_FLAGS=--xla_gpu_deterministic_ops=true` two independent processes produced a bit-identical update (equal hash, relative difference 0), at a ~3e-3 relative offset from the default-setting updates. Run-to-run noise of a pair margin (sd ~0.034) is larger than the mean own-exposure DeltaM, so single-chunk dynamics are not interpretable from this run.')
    # V3 (strict, vs Exp04 which used default non-reproducible settings), V3-pooled, V3b (run2 full vs run2 repeat)
    l3=[it for it in PA.items if it['level']=='L3'];tids=[it['tid'] for it in l3];v3rows=[];strict_ok=True;pooled_ok=True;v3b_ok=True;v3b_rows=[]
    for k in (14,33,62):
        for mode,fn in (('empty','kvempty'),('carry','main')):
            diffs=[];pn=[];po=[]
            for wname,W in (('A',A),('B',B)):
                if str(k) not in W['v3']:strict_ok=pooled_ok=False;continue
                new_=W['v3'][str(k)][mode];old=json.load(open(EXP4/f'seed{SEED}_{fn}_{wname}_ON.json'))['rows']
                om={r['tid']:(r['gold'],r['foil']) for r in old if r['k']==k and r['level']=='L3' and r['cond']=='NONE'}
                for tid,(g,fo) in zip(tids,new_):diffs+=[abs(g-om[tid][0]),abs(fo-om[tid][1])]
                pn.append({tid:g-fo for tid,(g,fo) in zip(tids,new_)});po.append({tid:om[tid][0]-om[tid][1] for tid in tids})
            if len(pn)==2:
                nm=np.mean([pn[0][t]-pn[1][t] for t in tids]);om_=np.mean([po[0][t]-po[1][t] for t in tids])
                v3rows.append((k,mode,max(diffs),nm,om_))
                if max(diffs)>1e-2:strict_ok=False
                if abs(nm-om_)>0.05:pooled_ok=False
            if R is not None and 'v3' in R and str(k) in R['v3'] and str(k) in A['v3']:
                d3=max(abs(g-g2)+abs(fo-fo2) for (g,fo),(g2,fo2) in zip(A['v3'][str(k)][mode],R['v3'][str(k)][mode]));v3b_rows.append((k,mode,d3))
                if d3>1e-6:v3b_ok=False
    if R is None or not v3b_rows:v3b_ok=False
    res['V3']=[{'k':k,'mode':m,'max_abs_item_diff_vs_exp04':d,'pooled_L3_AminusB_new':a_,'pooled_L3_AminusB_exp04':b_} for k,m,d,a_,b_ in v3rows];res['V3b']=[{'k':k,'mode':m,'max_item_diff_full_vs_repeat':d} for k,m,d in v3b_rows]
    res['V3_flags']={'strict':strict_ok,'pooled':pooled_ok,'V3b':v3b_ok}
    md.append(f'- **V3 (strict, original criterion: max |item diff| <= 1e-2 vs Exp04 clean-v2 seed {SEED}; Exp04 used default non-reproducible settings, so exact equality is not expected in this environment)**: {"PASS" if strict_ok else "FAIL (informational)"}')
    md.append(f'- **V3-pooled (decision; pooled L3 m(A)-m(B) within 0.05 nats of the Exp04 value at chunks 14/33/62)**: {"PASS" if pooled_ok else "FAIL"}')
    md.append(f'- **V3b (decision; Exp04-protocol probes at chunks 14/33/62 reproduce between run2 full and run2 repeat, max item diff <= 1e-6)**: {"PASS" if v3b_ok else "FAIL"}\n')
    md.append('| k | readout | max abs item diff vs Exp04 | pooled L3 m(A)-m(B), this run | Exp04 | V3b: max item diff full vs repeat |\n|---|---|---|---|---|---|')
    for (k,m,d,a_,b_) in v3rows:
        vb=[x for kk,mm,x in v3b_rows if kk==k and mm==m];md.append(f'| {k} | {"weights-only (empty KV)" if m=="empty" else "native carry-KV"} | {d:.2e} | {a_:+.4f} | {b_:+.4f} | {("%.2e"%vb[0]) if vb else "n/a"} |')
    v3ok=pooled_ok and v3b_ok
    allpass=(v2<=1e-6) and bool(v1pass) and v3ok;res['validation_pass']=allpass
    md.append(f'\n**All mandatory validations passed: {allpass}**'+('' if allpass else ' — interpretation below must be treated as invalid until the cause is resolved.')+'\n')
    # ---------- event classification ----------
    ev=A['events'];evB=B['events']
    def chunk_of(e):return e['tok_start']//CH
    own={};anyexp={}
    for e in ev:
        k=chunk_of(e);own.setdefault((e['pair'],k),[]).append(e);anyexp.setdefault(k,set()).add(e['pair'])
    cls=np.empty((len(pairs),NC),dtype=object)
    for i,p in enumerate(pairs):
        for k in range(NC):
            cls[i,k]='own' if (p,k) in own else ('other' if (anyexp.get(k) and anyexp[k]-{p}) else 'filler')
    def shadow_status(es):
        s=[e['shadowed'] for e in es];return 'shadowed' if all(s) else ('non' if not any(s) else 'mixed')
    events=[]  # own events
    for i,p in enumerate(pairs):
        ks=sorted(k for (pp,k) in own if pp==p)
        for n,k in enumerate(ks):
            es=own[(p,k)];events.append({'pair':p,'i':i,'k':k,'round':max(e['round'] for e in es),'shadow':shadow_status(es),'dM':float(dM[p][k]),'gain_ref':None,
                                         'next_k':ks[n+1] if n+1<len(ks) else None,'first':any(e['first'] for e in es),'M_after':float(M[p][k+1])})
    md.append('## Trajectory and decision\n')
    md.append(f'Noise floor theta (95th percentile of |DeltaM| over {len(cp)} never-exposed pseudo-pairs x {NC} chunks) = **{theta:.4f}** nats (control DeltaM sd {res["control_dM_sd"]:.4f}).\n')
    # chunk-class DeltaM
    rows=[]
    for c in ('own','other','filler'):
        v=[];cl=[]
        for i,p in enumerate(pairs):
            for k in range(NC):
                if cls[i,k]==c:v.append(dM[p][k]);cl.append(p)
        if v:
            b=boot(v,clusters=cl);res[f'dM_{c}']={'n':len(v),'mean':b[0],'lo':b[1],'hi':b[2],'frac_gt_theta':float(np.mean(np.array(v)>theta)),'frac_lt_minus_theta':float(np.mean(np.array(v)<-theta))}
            rows.append(f'| {c} | {len(v)} | {f3(b)} | {np.mean(np.array(v)>theta):.3f} | {np.mean(np.array(v)<-theta):.3f} |')
    cv=np.concatenate(list(dC.values()));rows.append(f'| control (pseudo-pairs, never exposed) | {len(cv)} | {f3(boot(cv))} | {np.mean(cv>theta):.3f} | {np.mean(cv<-theta):.3f} |')
    md.append('| chunk class | n (pair x chunk) | mean DeltaM [95% CI, pair-cluster bootstrap] | frac DeltaM > theta | frac DeltaM < -theta |\n|---|---|---|---|---|');md+=rows;md.append('')
    # learning criterion
    ev_dm=np.array([e['dM'] for e in events]);ev_pair=[e['pair'] for e in events]
    bl=boot(ev_dm,clusters=ev_pair);fr=float(np.mean(ev_dm>theta));L=(bl[1]>0 and fr>=0.10)
    res['L']={'mean':bl[0],'lo':bl[1],'hi':bl[2],'frac_gt_theta':fr,'L':bool(L)}
    md.append(f'- Own-exposure events: n={len(ev_dm)}, mean DeltaM {f3(bl)}, fraction > theta = {fr:.3f} (noise expectation 0.025) => L (learning detected) = **{L}**')
    # F criterion
    gains=[e for e in events if e['dM']>theta and e['next_k'] is not None]
    fl=[(e['M_after']-float(M[e['pair']][e['next_k']]))/e['dM'] for e in gains]
    flb=boot(fl,clusters=[e['pair'] for e in gains]) if fl else (float('nan'),)*3
    Fv=bool(L and fl and flb[0]>=0.25 and flb[1]>0);res['F']={'n_events':len(fl),'mean_fraction_lost':flb[0],'lo':flb[1],'hi':flb[2],'F':Fv}
    md.append(f'- Forgetting: own events with gain > theta and a later own exposure: n={len(fl)}, mean fraction of the gain lost before the next own exposure {f3(flb) if fl else "n/a"} => F = **{Fv}**')
    # shadow
    sh={s:[e['dM'] for e in events if e['shadow']==s] for s in ('non','shadowed','mixed')}
    shp={s:[e['pair'] for e in events if e['shadow']==s] for s in sh}
    diff=None
    if len(sh['non'])>1 and len(sh['shadowed'])>1:
        d_boot=[]
        for _ in range(4000):
            up=sorted(set(ev_pair));pick=[up[i] for i in rng.integers(0,len(up),len(up))]
            a=[x for pp in pick for x,q in zip(sh['non'],shp['non']) if q==pp];b=[x for pp in pick for x,q in zip(sh['shadowed'],shp['shadowed']) if q==pp]
            if a and b:d_boot.append(np.mean(a)-np.mean(b))
        diff=(float(np.mean(sh['non'])-np.mean(sh['shadowed'])),float(np.percentile(d_boot,2.5)),float(np.percentile(d_boot,97.5)))
    Sv=bool(L and diff is not None and np.mean(sh['shadowed'])<=0.5*np.mean(sh['non']) and diff[1]>0)
    res['S']={'n_non':len(sh['non']),'n_shadowed':len(sh['shadowed']),'n_mixed':len(sh['mixed']),'mean_non':float(np.mean(sh['non'])) if sh['non'] else None,'mean_shadowed':float(np.mean(sh['shadowed'])) if sh['shadowed'] else None,'diff_non_minus_shadowed':diff,'S':Sv}
    md.append(f'- Shadow (descriptive POSITIONAL proxy only: the previous exposure of the same triple ends within 8192 tokens before this sentence starts; it is NOT evidence that attention used that exposure, and round, position, cumulative exposures and fast-weight state are confounded, so no causal reading): non-shadowed own events n={len(sh["non"])}, mean DeltaM {np.mean(sh["non"]) if sh["non"] else float("nan"):+.4f}; shadowed n={len(sh["shadowed"])}, mean {np.mean(sh["shadowed"]) if sh["shadowed"] else float("nan"):+.4f}; mixed n={len(sh["mixed"])}; difference (non - shadowed) {f3(diff) if diff else "n/a"} => S = **{Sv}**')
    verdict='N' if not L else ('F' if Fv else '')+('+S' if Sv and Fv else ('S' if Sv else ''))
    if L and not (Fv or Sv):verdict='L (learning detected; neither F nor S criteria met)'
    res['verdict']=verdict;md.append(f'\n**Provisional verdict (pre-registered rules): {verdict}**'+('' if allpass else ' [validations failed: interpretation suspended]')+'\n')
    # secondary: per round, slope
    md.append('### DeltaM at own exposures by exposure round and shadow status (descriptive)\n\n| round | n | shadow status | mean DeltaM |\n|---|---|---|---|')
    byr={}
    for e in events:byr.setdefault(e['round'],[]).append(e)
    for r_ in sorted(byr):
        es=byr[r_];md.append(f'| {r_} | {len(es)} | {", ".join(sorted(set(e["shadow"]+("(first)" if e["first"] else "") for e in es)))} | {np.mean([e["dM"] for e in es]):+.4f} |')
    # slope per cumulative own exposure
    xs=[];ys=[];cl=[]
    for i,p in enumerate(pairs):
        cum=np.zeros(NS)
        for e in events:
            if e['pair']==p:cum[e['k']+1:]+=1
        x=cum-cum.mean();y=M[p]-M[p].mean();xs+=list(x);ys+=list(y);cl+=[p]*NS
    xs=np.array(xs);ys=np.array(ys);slope=float((xs*ys).sum()/(xs*xs).sum())
    bs=[]
    up=sorted(set(cl))
    for _ in range(2000):
        pick=rng.integers(0,len(up),len(up));X=[];Y=[]
        for i in pick:
            idx=[j for j,c in enumerate(cl) if c==up[i]];X+= [xs[j] for j in idx];Y+= [ys[j] for j in idx]
        X=np.array(X);Y=np.array(Y);bs.append((X*Y).sum()/(X*X).sum())
    res['slope_per_own_exposure']=(slope,float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))
    md.append(f'\nSlope of pair-level M_k per cumulative own-exposure sentence-pair (within-pair demeaned OLS, pair bootstrap): {f3(res["slope_per_own_exposure"])} nats; final mean M_63 = {np.mean([M[p][-1] for p in pairs]):+.4f}; control M_63 = {np.mean([MC[p][-1] for p in cp]):+.4f}\n')
    # ---------- (1) world-wise trajectories and symmetry ----------
    mA_={p:np.array([A['margins_state'][str(j)][p] for j in range(NS)]) for p in pairs};mB_={p:np.array([B['margins_state'][str(j)][p] for j in range(NS)]) for p in pairs}
    cA_={p:np.array([A['ctrl_state'][str(j)][p] for j in range(NS)]) for p in cp};cB_={p:np.array([B['ctrl_state'][str(j)][p] for j in range(NS)]) for p in cp}
    with open(OUT/('mA_mB_trajectory_seed%d.csv'%SEED),'w') as f:
        f.write('state,'+','.join([f'mA_{p}' for p in pairs]+[f'mB_{p}' for p in pairs]+[f'cA_{p}' for p in cp]+[f'cB_{p}' for p in cp])+'\n')
        for j in range(NS):f.write(f'{j},'+','.join(f'{x:.5f}' for x in [mA_[p][j] for p in pairs]+[mB_[p][j] for p in pairs]+[cA_[p][j] for p in cp]+[cB_[p][j] for p in cp])+'\n')
    md.append('## World-wise trajectories (m in A-labelling; A history supports +, B history supports -)\n')
    fin=lambda d:[d[p][-1]-d[p][0] for p in d]
    md.append(f'- Net change over all 63 chunks (mean over pairs): m_A {f3(boot(fin(mA_)))}, m_B {f3(boot(fin(mB_)))}; common-mode C=(m_A+m_B)/2 {f3(boot([(mA_[p][-1]+mB_[p][-1])/2-(mA_[p][0]+mB_[p][0])/2 for p in pairs]))}; binding M {f3(boot([M[p][-1]-M[p][0] for p in pairs]))}. Control pseudo-pairs: c_A {f3(boot(fin(cA_)))}, c_B {f3(boot(fin(cB_)))}.')
    ea=[];eb=[];ep=[]
    for e in events:
        p=e['pair'];k=e['k'];ea.append(mA_[p][k+1]-mA_[p][k]);eb.append(mB_[p][k+1]-mB_[p][k]);ep.append(p)
    ea=np.array(ea);eb=np.array(eb)
    sym=float(np.mean((ea>0)&(eb<0)));corr=float(np.corrcoef(ea,eb)[0,1]) if len(ea)>2 else float('nan')
    bind=(ea-eb)/2;cmon=(ea+eb)/2
    res['symmetry']={'dmA_own':boot(ea,clusters=ep),'dmB_own':boot(eb,clusters=ep),'binding_own':boot(bind,clusters=ep),'common_own':boot(cmon,clusters=ep),'frac_A_up_and_B_down':sym,'corr_dmA_dmB':corr}
    md.append(f'- At own-exposure events (n={len(ea)}): Delta m_A {f3(res["symmetry"]["dmA_own"])}, Delta m_B {f3(res["symmetry"]["dmB_own"])} (symmetric learning would be + and -); binding part (dmA-dmB)/2 {f3(res["symmetry"]["binding_own"])}; common-mode part (dmA+dmB)/2 {f3(res["symmetry"]["common_own"])}; fraction of events with A up and B down {sym:.3f} (independent-noise expectation ~0.25); corr(dmA,dmB) {corr:+.3f} (symmetric binding => negative; common-mode drift => positive).')
    ebase_a=np.array([mA_[p][k+1]-mA_[p][k] for p in pairs for k in range(NC) if cls[pairs.index(p),k]!='own']);ebase_b=np.array([mB_[p][k+1]-mB_[p][k] for p in pairs for k in range(NC) if cls[pairs.index(p),k]!='own'])
    md.append(f'- Same quantities at non-own chunks: Delta m_A {f3(boot(ebase_a))}, Delta m_B {f3(boot(ebase_b))}, corr {np.corrcoef(ebase_a,ebase_b)[0,1]:+.3f}. World-wise series: `mA_mB_trajectory_seed{SEED}.csv`.\n')
    # ---------- (3) round-wise DeltaM and update norm ----------
    md.append('## Exposure round: DeltaM and update statistics (descriptive; cumulative exposure count and the current fast-weight state are confounders)\n')
    ur={k:(0.5*(A['rows'][k]['update_norm']+B['rows'][k]['update_norm']),0.5*(A['rows'][k]['grad_norm_preclip']+B['rows'][k]['grad_norm_preclip']),0.5*(float(A['rows'][k]['clipped'])+float(B['rows'][k]['clipped']))) for k in range(NC)}
    md.append('| round | n own events | shadow status | mean DeltaM | mean update norm of that chunk | mean pre-clip grad norm | clipped |\n|---|---|---|---|---|---|---|')
    for r_ in sorted(byr):
        es=byr[r_];md.append(f'| {r_} | {len(es)} | {", ".join(sorted(set(e["shadow"]+("(first)" if e["first"] else "") for e in es)))} | {np.mean([e["dM"] for e in es]):+.4f} | {np.mean([ur[e["k"]][0] for e in es]):.3f} | {np.mean([ur[e["k"]][1] for e in es]):.3f} | {np.mean([ur[e["k"]][2] for e in es])*100:.0f}% |')
    G1=[e for e in events if e['round'] in (1,2,5)];G2=[e for e in events if e['round'] not in (1,2,5)]
    def grp(g):return [e['dM'] for e in g],[e['pair'] for e in g]
    def dboot(g1,g2):
        up=sorted(set(e['pair'] for e in g1+g2));out=[]
        for _ in range(4000):
            pick=[up[i] for i in rng.integers(0,len(up),len(up))];a=[e['dM'] for pp in pick for e in g1 if e['pair']==pp];b=[e['dM'] for pp in pick for e in g2 if e['pair']==pp]
            if a and b:out.append(np.mean(a)-np.mean(b))
        return float(np.mean([e['dM'] for e in g1])-np.mean([e['dM'] for e in g2])),float(np.percentile(out,2.5)),float(np.percentile(out,97.5))
    if G1 and G2:
        d=dboot(G1,G2);res['rounds_125_vs_rest']={'n125':len(G1),'n_rest':len(G2),'mean125':float(np.mean([e['dM'] for e in G1])),'mean_rest':float(np.mean([e['dM'] for e in G2])),'diff':d,'upd125':float(np.mean([ur[e['k']][0] for e in G1])),'upd_rest':float(np.mean([ur[e['k']][0] for e in G2]))}
        md.append(f'\nRounds {{1,2,5}} (first exposure, or previous exposure >8192 tokens back) vs rounds {{3,4,6-16}} (previous exposure within the window): mean DeltaM {np.mean([e["dM"] for e in G1]):+.4f} (n={len(G1)}) vs {np.mean([e["dM"] for e in G2]):+.4f} (n={len(G2)}); difference {f3(d)}; mean update norm {np.mean([ur[e["k"]][0] for e in G1]):.3f} vs {np.mean([ur[e["k"]][0] for e in G2]):.3f}. Rounds {{1,2,5}} coincide with the non-shadowed events by construction of the layout, and they occur at low cumulative exposure counts and at specific stream positions, so this contrast is confounded with exposure count, stream position and the fast-weight state; it is a description, not a test of shadowing.\n')
    # ---------- (4) hypotheses: pre-registered verdict + non-exclusive evidence ----------
    nonown=[dM[p][k] for i,p in enumerate(pairs) for k in range(NC) if cls[i,k]!='own']
    allev=[e for e in events if e['next_k'] is not None]
    lossall=[(e['M_after']-float(M[e['pair']][e['next_k']])) for e in allev]
    md.append('## Link to the three hypotheses (pre-registered rules unchanged; patterns may coexist)\n')
    md.append(f'- **Pre-registered verdict** (N excludes F and S by rule): **{verdict}**{"" if allpass else " [validations failed]"}.')
    md.append(f'- **N-type evidence** (weak learning right after the pair\'s own exposure): mean DeltaM_own {f3(bl)} vs control sd {res["control_dM_sd"]:.4f} / theta {theta:.4f}; fraction of own events above theta {fr:.3f} (noise ~0.025). Not mutually exclusive with F or S below.')
    md.append(f'- **F-type evidence** (loss after other/filler updates; descriptive, no gain threshold): mean DeltaM at non-own chunks {f3(boot(nonown))}; mean M change from just after an own exposure to just before the next own exposure {f3(boot([-x for x in lossall]))} nats (negative = loss) over n={len(lossall)} intervals; pre-registered F rule (gain > theta, >=25% lost, CI>0): **{Fv}**.')
    md.append(f'- **S-type evidence** (shadowed vs non-shadowed own events; positional proxy only, not evidence of attention use, confounded, no causal claim): non-shadowed {np.mean(sh["non"]) if sh["non"] else float("nan"):+.4f} (n={len(sh["non"])}) vs shadowed {np.mean(sh["shadowed"]) if sh["shadowed"] else float("nan"):+.4f} (n={len(sh["shadowed"])}), difference {f3(diff) if diff else "n/a"}; pre-registered S rule: **{Sv}**.')
    md.append('- Several patterns can hold at once (e.g. weak per-exposure learning, partial loss, and smaller gains for repeated exposures). The three flags above are therefore reported separately; the single-label verdict is the pre-registered convention, not a claim that the others are absent.\n')
    md.append('## Implications for M1 evaluation (no extra evaluation was run)\n')
    md.append('Beyond the final margin, an M1-style method should be judged on **source-free 2-way accuracy** (gold vs matched foil, chance 0.5) and **counterfactual binding flip** (the choice must change when the history is role-swapped; chance 0.0625 for both roles, 0 for a global preference). For reference, Exp04 clean-v2 reached accuracy 0.47-0.53 and bidirectional flip 0.00 with a native readout, while the weights-only L3 margin was +0.24 nats at 16 exposures. These two metrics are follow-up evaluations for any candidate method; they were not recomputed here.\n')
    # ---------- optimizer / clip ----------
    md.append('## Update statistics (both worlds)\n')
    allrows=A['rows']+B['rows'];gn=np.array([r['grad_norm_preclip'] for r in allrows]);cf=np.array([r['clip_coef'] for r in allrows]);un=np.array([r['update_norm'] for r in allrows])
    res['clip']={'frac_clipped':float(np.mean([r['clipped'] for r in allrows])),'gn_mean':float(gn.mean()),'gn_min':float(gn.min()),'gn_max':float(gn.max()),'coef_min':float(cf.min()),'update_norm_mean':float(un.mean())}
    md.append(f'- clipped chunks: {res["clip"]["frac_clipped"]*100:.1f}% ({sum(r["clipped"] for r in allrows)}/{len(allrows)}); pre-clip grad norm mean {gn.mean():.3f} (min {gn.min():.3f}, max {gn.max():.3f}); smallest clip coefficient {cf.min():.3f}; update norm mean {un.mean():.3f}')
    cc={}
    for w_,W in (('A',A),('B',B)):
        for k,r in enumerate(W['rows']):
            kk=[cls[i,k] for i in range(len(pairs))];c='own-any' if 'own' in kk else ('other' if 'other' in kk else 'filler')
            cc.setdefault(c,[]).append((r['grad_norm_preclip'],r['clipped'],r['update_norm']))
    md.append('\n| chunk class (any pair) | n | mean grad norm | clipped | mean update norm |\n|---|---|---|---|---|')
    for c,v in cc.items():md.append(f'| {c} | {len(v)} | {np.mean([x[0] for x in v]):.3f} | {np.mean([x[1] for x in v])*100:.0f}% | {np.mean([x[2] for x in v]):.3f} |')
    ln=np.array([r['layer_update_norm'] for r in A['rows'] if r['layer_update_norm']]);md.append('\nMean per-suffix-layer update norm (World A): '+', '.join(f'L{i}: {x:.3f}' for i,x in enumerate(ln.mean(0))) if len(ln) else '')
    # token-level NLL of exposure sentences by round
    tn=np.load(OUT/f'traj_seed{SEED}_{TAG}_A_tok_nll.npz')['tok_nll'].astype(np.float32);rn={}
    for e in ev:
        idx=[t-1 for t in range(e['tok_start'],e['tok_end']+1) if 0<=t-1<NC*CH]
        vals=[tn[i//CH][i%CH] for i in idx]
        if vals:rn.setdefault((e['round'] if e['round']<=4 or e['round'] in (8,12,16) else None),[]).append(np.mean(vals))
    md.append('\n### Token-level NLL of exposure-sentence tokens (pre-update, World A) by exposure round\n\n| round | n sentences | mean NLL |\n|---|---|---|')
    for r_ in sorted(k for k in rn if k):md.append(f'| {r_} | {len(rn[r_])} | {np.mean(rn[r_]):.3f} |')
    # ---------- C5 carry-KV ----------
    md.append('\n## C5 — native carry-KV margin at checkpoints where the pair has no exposure inside the last 8K\n')
    md.append('| checkpoint k | n qualifying pairs | native carry-KV M (A-B)/2 [95% CI] | short weights-only M (same state, same pairs) |\n|---|---|---|---|');c5={}
    spans_by_pair={}
    for e in ev:spans_by_pair.setdefault(e['pair'],[]).append(e)
    for k in sorted(set(int(x) for x in list(A['carry'].keys())+list(A['v3'].keys()))):
        key=str(k);newA=A['carry'].get(key) or (A['v3'].get(key) or {}).get('carry');newB=B['carry'].get(key) or (B['v3'].get(key) or {}).get('carry')
        if newA is None or newB is None:continue
        limit=(k+1)*CH-(8192-CH);qual=[]
        for p in pairs:
            es=[e for e in spans_by_pair[p] if e['tok_start']//CH<=k]
            if es and max(e['tok_end'] for e in es)<limit:qual.append(p)
        mA={};mB={}
        for tid,it,(g,fo),(g2,fo2) in zip(tids,l3,newA,newB):
            mA.setdefault(it['pair'],[]).append(g-fo);mB.setdefault(it['pair'],[]).append(g2-fo2)
        vals=[(np.mean(mA[p])-np.mean(mB[p]))/2 for p in qual];short=[M[p][k+1] for p in qual]
        if vals:
            b=boot(vals);c5[k]={'n':len(vals),'carry':b,'short':float(np.mean(short))};md.append(f'| {k} | {len(vals)} | {f3(b)} | {np.mean(short):+.4f} |')
    res['C5']={str(k):v for k,v in c5.items()}
    md.append('\nOnly pairs whose last exposure ended before the SWA window of the last probe token (Exp04 definition) are included; pairs with no exposure yet are excluded.\n')
    (OUT/'exp05A_report.md').write_text('\n'.join(md));(OUT/'exp05A_metrics.json').write_text(json.dumps(res,indent=1,default=str));print('\n'.join(md))
    # trajectory CSV for plotting
    with open(OUT/('M_trajectory_seed%d.csv'%SEED),'w') as f:
        f.write('state,'+','.join(pairs)+','+','.join(cp)+'\n')
        for j in range(NS):f.write(f'{j},'+','.join(f'{M[p][j]:.5f}' for p in pairs)+','+','.join(f'{MC[p][j]:.5f}' for p in cp)+'\n')
if __name__=='__main__':main()
