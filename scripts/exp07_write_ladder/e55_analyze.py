"""E55 report. Thin wrapper over e55_metrics (design 10.3 / 4c) for the k >= 16 cells; plus the existence-check report (stage 0', k = 8 < 16, p = 1: candidates = the k exposed places, ranks normalised u = (K - rank) / (K - 1), chance 0.5; familiarity = mean score of the never-exposed places minus mean score of the exposed non-gold places). Records are the JSON files of `e55_run.py read` (format: e55_read.py docstring); several seeds are given as a comma list and pooled over facts.
  python3 e55_analyze.py --nat <nat.read.json[,...]> [--swd <swd.read.json[,...]>] [--a 16] [--out report.md]
  python3 e55_analyze.py --selftest"""
import sys,argparse,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e55_metrics as MT
# preregistered fixed thresholds (design 11b-11h; never derived from data): first test S >= 0.15 in u = 2.25 rank positions (u = (16 - rank)/15); per-arm existence 'present' u 0.10 = 1.5 positions, 'strong' u 0.30 = 4.5 positions, 'absent' = CI upper limit < u 0.05 = 0.75 positions
U_PER_RANK=1/15;THR_S_U=0.15;THR_S_RANK=THR_S_U*15;EX_PRESENT_U=0.10;EX_STRONG_U=0.30;EX_ABSENT_U=0.05;HOLD_RATIO=1.15;NB_V=10000
def _items(rec,a,ctx):
    """per-item arrays for the existence check: u (W), u (W0), familiarity (W, W0), kind mask, valid mask"""
    m=rec['meta'];n=len(m['kind']);ex=np.asarray(m['cand_exposed'],bool);gold=np.asarray(m['gold_idx']);K=int(m['n_rank'][0])
    def full(S):
        S=np.asarray(S,float)
        if len(S)==n:return S
        F=np.full((n,16),np.nan);F[m['item_idx'][str(a)][ctx]]=S;return F
    S=full(rec['read'][str(a)][ctx]);W0=rec['W0'];S0=full(W0[f'{ctx}@{a}'] if f'{ctx}@{a}' in W0 else W0[ctx]);r=np.arange(n)
    def stats(X):
        sg=X[r,gold];rank=1+((X>sg[:,None])&ex).sum(1);u=(K-rank)/(K-1)
        exn=ex.copy();exn[r,gold]=False;fam=np.where((~ex).any(1),(np.where(~ex,X,0).sum(1)/np.maximum((~ex).sum(1),1)),np.nan)-(np.where(exn,X,0).sum(1)/np.maximum(exn.sum(1),1));return u,fam
    u,f=stats(S);u0,f0=stats(S0);valid=~np.isnan(S).any(1)
    if ctx!='out':
        d=m['d'][str(a)][ctx];valid&=np.array([x is not None and 1<=x<=8 for x in d])
    return {'u':u,'u0':u0,'fam':f,'fam0':f0,'fact':(np.asarray(m['kind'])=='fact')&valid,'ctrl':(np.asarray(m['kind'])!='fact')&valid}
def existence(nat_files,swd_files=None,a_list=(4,16),out=None):
    load=lambda fs:[json.load(open(f)) for f in fs.split(',')];nat=load(nat_files);swd=load(swd_files) if swd_files else None
    L=['# E55 existence check (stage 0-prime): K = %d exposed candidates, u = (K - rank)/(K - 1) (chance 0.5; gain = u(W) - u(W0)); %d seeds'%(int(nat[0]['meta']['n_rank'][0]),len(nat)),'']
    def cat(recs,a,ctx,key,kind='fact'):return np.concatenate([_items(r,a,ctx)[key][_items(r,a,ctx)[kind]] for r in recs])
    cells={'NAT':nat,**({'SWD':swd} if swd else {})};ctxs=['out']+sorted(c for c in nat[0]['read'].get(str(max(a_list)),{}) if c!='out')
    L+=['| cell | read | context | gain in u [95% CI] | u(W) | u(W0) | familiarity change (unexposed - exposed non-gold, nats) [95% CI] | control items u (fake gold; expect ~0.5) | per-seed gain in u |','|'+'---|'*9]
    for nm,recs in cells.items():
        for a in a_list:
            for ctx in (['out'] if a!=max(a_list) else ctxs):
                if str(a) not in recs[0]['read'] or ctx not in recs[0]['read'][str(a)]:continue
                g=cat(recs,a,ctx,'u')-cat(recs,a,ctx,'u0');fam=cat(recs,a,ctx,'fam')-cat(recs,a,ctx,'fam0');cu=cat(recs,a,ctx,'u','ctrl')
                ps=[float((_items(r,a,ctx)['u']-_items(r,a,ctx)['u0'])[_items(r,a,ctx)['fact']].mean()) for r in recs]
                L.append(f'| {nm} | appearance {a} | {ctx} | {MT.fmt(MT.ci(g))} | {cat(recs,a,ctx,"u").mean():.3f} | {cat(recs,a,ctx,"u0").mean():.3f} | {MT.fmt(MT.ci(fam))} | {cu.mean():.3f} (n={len(cu)}) | {", ".join("%+.3f"%x for x in ps)} |')
    if swd:
        a=max(a_list);gN=cat(nat,a,'out','u')-cat(nat,a,'out','u0');gS=cat(swd,a,'out','u')-cat(swd,a,'out','u0')
        L+=['',f'shadowing contrast SWD - NAT at appearance {a}, out-of-window, gain in u (paired by fact): {MT.fmt(MT.ci(gS-gN))}']
    L+=['','Existence: gain in u > 0 with the CI excluding 0 at the last appearance (and the control items near 0.5) = the binding exists in this setting; the familiarity column separates "knows these places" from "knows which place goes with the name".']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def is_existence(files):return json.load(open(files.split(',')[0]))['meta'].get('n_rank',[16])[0]<16
def curve(nat,alts,out_apps=None):
    """binding curve (seeds with the full per-chunk read): per appearance G_rank / PM2 / PM3 of every arm (out-of-window), contrasts alt - NAT (paired by fact), and the in-window readout at the appearances that have one. nat: records of NAT, alts: {name: records}; only records whose 'read' holds more than 4 appearances."""
    full=lambda rs:[r for r in rs if len(r['read'])>4];nf=full(nat)
    if not nf:return ''
    apps=sorted(int(x) for x in nf[0]['read']);L=['','## binding curve (seed(s) with the full read: %s), out-of-window read after every appearance'%(','.join(str(r['meta']['seed']) for r in nf)),'','| appearance | arm | G_rank [95% CI] | PM2 | PM3 | contrast vs NAT: G_rank |','|---|---|---|---|---|---|']
    for a in apps:
        if 'out' not in nf[0]['read'][str(a)]:continue
        t=MT.cell_table(nf,a,'out');L.append(f'| {a} | NAT | {MT.fmt(t["G_rank"])} | {MT.fmt(t["PM2"])} | {MT.fmt(t["PM3"])} | |')
        for nm,rs in alts.items():
            rf=[r for r in full(rs) if r['meta']['seed'] in [x['meta']['seed'] for x in nf]]
            if rf:t=MT.cell_table(rf,a,'out');c=MT.contrast(rf,[x for x in nf if x['meta']['seed'] in [y['meta']['seed'] for y in rf]],a,'out');L.append(f'| {a} | {nm} | {MT.fmt(t["G_rank"])} | {MT.fmt(t["PM2"])} | {MT.fmt(t["PM3"])} | {MT.fmt(c["G_rank"])} |')
    L+=['','in-window readout (contexts shorter than 8 chunks while a < 8; decision column = exit drop rank(out) - rank(in)):']
    for nm,rs in (('NAT',nf),)+tuple((k,full(v)) for k,v in alts.items()):
        for a in apps:
            ins=sorted(c for c in rs[0]['read'][str(a)] if c!='out') if rs else []
            if ins:
                w=MT.window_effect(rs,a,ins);L.append(f'  {nm} a={a}: exit drop {MT.fmt(w["exit_rank"])}, G_in-G_out {MT.fmt(w["G_in_minus_G_out"])}, W0 in-window rank {MT.fmt(w["W0_in_rank"])}'+(' [metric mismatch: no call]' if w['metric_mismatch'] else ''))
    return '\n'.join(L)+'\n'
def w0_report(recs,a=16,ctx='out'):
    """W0 (before training) rank of the gold place per question frame: histogram over ranks 1..16, mean, and the ceiling rate (W0 rank 1: no rank gain possible) / floor rate (W0 rank 16: no rank loss possible), pooled over the records; facts only"""
    rk=[];qf=[]
    for r in recs:
        it=MT.rec_items(r,a,ctx);m=MT._sel(r,a,ctx);rk+=list(it['rank_W0'][m]);qf+=list(np.asarray(r['meta']['qframe'])[m])
    rk=np.asarray(rk);qf=np.asarray(qf);L=['','## W0 rank of the gold place per question frame (before any update; chance mean 8.5)','','| question frame | n | mean rank W0 | ceiling (rank 1) | floor (rank 16) | histogram rank 1..16 |','|---|---|---|---|---|---|']
    for q in [None]+sorted(set(qf.tolist())):
        x=rk if q is None else rk[qf==q];h=np.bincount(x.astype(int),minlength=17)[1:17]
        L.append(f"| {'all' if q is None else q} | {len(x)} | {x.mean():.2f} | {np.mean(x==1):.3f} | {np.mean(x==16):.3f} | {' '.join(map(str,h))} |")
    return '\n'.join(L)+'\n'
def train_summary(saved_files):
    """per arm and seed from the saved.json logs: mean pre-clip gradient norm, mean clip coefficient c (1 = not clipped), share of clipped chunks, mean / cumulative update norm, mean CE (the per-chunk values are in the log of the file)"""
    L=['','## training statistics per arm and seed (from the per-chunk logs)','','| arm | seed | chunks | mean pre-clip gnorm | mean clip coefficient c | clipped chunks | mean update norm | cumulative update norm | mean CE |','|---|---|---|---|---|---|---|---|---|']
    for f in saved_files.split(','):
        r=json.load(open(f));lg=r['log'];g=lambda k:np.asarray([e[k] for e in lg],float)
        L.append(f"| {Path(f).name.split('_')[2]} | {r.get('world_seed')} | {len(lg)} | {g('gnorm').mean():.4g} | {g('coef').mean():.3f} | {np.mean(g('coef')<1):.2f} | {g('update_norm').mean():.4g} | {g('update_norm').sum():.4g} | {g('ce').mean():.3f} |")
    return '\n'.join(L)+'\n'
def _fact_arrays(rec,a,ctx='out'):
    """per fact (facts only, W0 subtracted): g (u gain), pm2, pm3, W0 rank"""
    it=MT.rec_items(rec,a,ctx);m=MT._sel(rec,a,ctx);return {'g':it['G_rank'][m]*U_PER_RANK,'pm2':it['PM2'][m],'pm3':it['PM3'][m],'r0':it['rank_W0'][m]}
def _boot_seedmean(per_seed,nb=NB_V,seed=11):
    """pooled CI as in 11h: within every seed resample the facts, average the seed means; per_seed: list of 1-D arrays of fact values -> (mean, lo, hi)"""
    rng=np.random.default_rng(seed);b=np.mean([x[rng.integers(0,len(x),(nb,len(x)))].mean(1) for x in per_seed],axis=0)
    return float(np.mean([x.mean() for x in per_seed])),*np.percentile(b,[2.5,97.5]).tolist()
def _boot_one(x,nb=NB_V,seed=12):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);b=x[rng.integers(0,len(x),(nb,len(x)))].mean(1);return float(x.mean()),*np.percentile(b,[2.5,97.5]).tolist()
def _saved_stats(f):
    r=json.load(open(f));lg=r['log'];g=lambda k:np.asarray([e[k] for e in lg],float);o={'upd':float(g('update_norm').mean()),'gn':float(g('gnorm').mean()),'clip':float(np.mean(g('coef')<1)),'coef':float(g('coef').mean())}
    cw=r.get('canary_w0');cm=None            # e55_run: canary at W0 = r['canary_w0']; later points in r['reads'][chunk]['canary'] (keys _gen_nll, _tail_nll)
    if cw:
        d=[max(e['canary'][k]-cw[k] for k in cw if k in e['canary']) for e in r.get('reads',{}).values() if isinstance(e,dict) and e.get('canary')];cm=float(max(d)) if d else None
    o['canary_max_over_W0']=cm;return o
def _seed_of_saved(f):
    import re;m=re.search(r'_s(\d+)',Path(f).name);return int(m.group(1)) if m else None
def verdict(nat_files,swd_files,saved=None,a=16,unit=None,out=None):
    """first-test verdict table (design 11b-11h, fixed values, nothing derived from the data): per-seed S = g(SWD) - g(NAT) with per-seed CI, pooled S (seed means; CI resampling facts WITHIN each seed), the five conditions, the hold check (update-norm ratio vs 1.15), per-arm existence, W0 ceiling/floor sensitivity. saved: comma list of *.saved.json (arm in the file name _<ARM>_s<seed>); unit: e55_unit_*.json of the same job (identity test)."""
    load=lambda fs:{json.load(open(f))['meta']['seed']:json.load(open(f)) for f in fs.split(',')};N=load(nat_files);S_=load(swd_files);seeds=sorted(set(N)&set(S_))
    L=['# E55 first test: verdict table (fixed rules; seeds %s; read a = %d, out-of-window; u = (16 - rank)/15, ties = average rank)'%(seeds,a),'']
    nat={s_:_fact_arrays(N[s_],a) for s_ in seeds};swd={s_:_fact_arrays(S_[s_],a) for s_ in seeds}
    for s_ in seeds:assert np.array_equal(nat[s_]['r0'],swd[s_]['r0']),('W0 ranks differ between arms',s_)
    d={s_:swd[s_]['g']-nat[s_]['g'] for s_ in seeds}
    L+=['## per-seed S = g(SWD) - g(NAT) (facts paired; CI = bootstrap over facts of that seed, conditional on this one training run)','','| seed | facts | g(NAT) | g(SWD) | S [95% CI] | S > 0 |','|---|---|---|---|---|---|']
    per={}
    for s_ in seeds:
        c=_boot_one(d[s_]);per[s_]=c;L.append(f'| {s_} | {len(d[s_])} | {nat[s_]["g"].mean():+.3f} | {swd[s_]["g"].mean():+.3f} | {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}] | {"yes" if c[0]>0 else "no"} |')
    P=_boot_seedmean([d[s_] for s_ in seeds]);pm2=float(np.mean([swd[s_]['pm2'].mean()-nat[s_]['pm2'].mean() for s_ in seeds]));pm3=float(np.mean([swd[s_]['pm3'].mean()-nat[s_]['pm3'].mean() for s_ in seeds]))
    all_pos=all(per[s_][0]>0 for s_ in seeds)
    sa={'NAT':{},'SWD':{}}
    for f in (saved.split(',') if saved else []):
        nm=Path(f).name;sd_=_seed_of_saved(f)
        for arm in sa:
            if f'_{arm}_' in nm and sd_ in seeds:sa[arm][sd_]=_saved_stats(f)
    ratios={s_:sa['SWD'][s_]['upd']/sa['NAT'][s_]['upd'] for s_ in seeds if s_ in sa['SWD'] and s_ in sa['NAT']}
    pooled_ratio=float(np.mean(list(ratios.values()))) if ratios else float('nan');hold=bool(ratios) and pooled_ratio>HOLD_RATIO
    can_ok=None;clip_ok=None
    if all(s_ in sa[x] for s_ in seeds for x in sa):
        cs=[sa[x][s_]['canary_max_over_W0'] for s_ in seeds for x in sa];can_ok=None if any(c is None for c in cs) else bool(max(cs)<=0.10);clip_ok=bool(max(sa[x][s_]['clip'] for s_ in seeds for x in sa)<0.9)
    uid=None
    if unit:uid=bool(json.load(open(unit)).get('PASS'))
    mark=lambda b:'?' if b is None else ('O' if b else 'X')
    five=(can_ok is not False) and (clip_ok is not False) and (uid is not False) and None not in (can_ok,clip_ok,uid)
    L+=['','## pooled verdict (S pooled = mean of the seed S values; CI resamples facts within each seed: "conditional on the fact sampling of these training runs, seed-level uncertainty NOT included")','','| condition | value | pass |','|---|---|---|',
        f'| 1. S_pooled >= {THR_S_U:.2f} u ({THR_S_RANK:.2f} rank positions) | {P[0]:+.3f} u ({P[0]*15:+.2f} positions) | {mark(P[0]>=THR_S_U)} |',f'| 2. pooled CI excludes 0 | [{P[1]:+.3f}, {P[2]:+.3f}] | {mark(P[1]>0 or P[2]<0)} |',
        f'| 3. PM2 difference >= +5 points and PM3 difference > 0 | PM2 {pm2*100:+.1f} points, PM3 {pm3:+.3f} nats | {mark(pm2>=0.05 and pm3>0)} |',f'| 4. S > 0 in every seed | {", ".join("%d: %+.3f"%(s_,per[s_][0]) for s_ in seeds)} | {mark(all_pos)} |',
        f'| 5. canary <= +0.10 nat, identity test, clipped-chunk share < 0.9 | canary {mark(can_ok)}, clip {mark(clip_ok)}, identity {mark(uid)} ("?" = input not given) | {mark(five if None not in (can_ok,clip_ok,uid) else None)} |']
    c1=P[0]>=THR_S_U and (P[1]>0 or P[2]<0) and pm2>=0.05 and pm3>0 and all_pos and five is True
    if c1:cls='detected (first test, this synthetic world only)'
    elif P[0]<=-0.10:cls='fake window is worse (opposite direction)'
    elif P[0]>=0.10:cls='direction supported, on hold (0.10 <= S < 0.15 or a condition above failed)'
    elif abs(P[0])<0.10 and P[1]>=-THR_S_U and P[2]<=THR_S_U:cls='cannot be separated within the resolution (not an absence)'
    else:cls='unclear'
    if hold:cls+=' -- ON HOLD (update-size ratio > %.2f; only the size-matched NAT control can release it)'%HOLD_RATIO
    L+=['',f'classification: **{cls}**','',f'hold check (update-norm ratio SWD/NAT, chunk mean; hold if the mean over the seeds > {HOLD_RATIO}): '+(', '.join('seed %d: %.3f'%(k,v) for k,v in ratios.items())+f'; mean {pooled_ratio:.3f} -> '+('ON HOLD: only the size-matched NAT control (NATM, seed 55) can release it; an update larger by itself may be part of the shadowing (gradient starvation relieved)' if hold else 'not on hold') if ratios else 'saved.json not given: NOT CHECKED')]
    if sa['NAT'] and sa['SWD']:
        L+=['','| arm | seed | mean update norm | mean pre-clip gnorm | mean coef | clipped chunks | canary max over W0 |','|---|---|---|---|---|---|---|']
        for x in sa:
            for s_ in sorted(sa[x]):v=sa[x][s_];L.append(f'| {x} | {s_} | {v["upd"]:.4g} | {v["gn"]:.4g} | {v["coef"]:.3f} | {v["clip"]:.2f} | {v["canary_max_over_W0"] if v["canary_max_over_W0"] is None else "%+.3f"%v["canary_max_over_W0"]} |')
    L+=['','## per-arm existence (pooled g; present: CI excludes 0 and g >= %.2f and PM2 >= +5 points and PM3 > 0; strong: g >= %.2f; absent: CI upper limit < %.2f; else undetermined)'%(EX_PRESENT_U,EX_STRONG_U,EX_ABSENT_U),'','| arm | g [95% CI] | PM2 | PM3 | call |','|---|---|---|---|---|']
    for nm,D in (('NAT',nat),('SWD',swd)):
        g=_boot_seedmean([D[s_]['g'] for s_ in seeds],seed=13);p2=float(np.mean([D[s_]['pm2'].mean() for s_ in seeds]));p3=float(np.mean([D[s_]['pm3'].mean() for s_ in seeds]))
        call='absent' if g[2]<EX_ABSENT_U else ('strong' if (g[1]>0 and g[0]>=EX_STRONG_U and p2>=0.05 and p3>0) else ('present' if (g[1]>0 and g[0]>=EX_PRESENT_U and p2>=0.05 and p3>0) else 'undetermined'))
        L.append(f'| {nm} | {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}] | {p2*100:+.1f} points | {p3:+.3f} | {call} |')
    r0=np.concatenate([nat[s_]['r0'] for s_ in seeds]);ceil=float(np.mean(r0==1));floor=float(np.mean(r0==16))
    L+=['',f'## W0 ceiling / floor: gold at W0 rank 1: {ceil:.1%} (no gain possible), rank 16: {floor:.1%} (no loss possible)']
    if ceil>=0.25 or floor>=0.25:
        keep={s_:(nat[s_]['r0']!=1)&(nat[s_]['r0']!=16) for s_ in seeds};ok=[s_ for s_ in seeds if keep[s_].sum()>1]
        if ok:
            gN=_boot_seedmean([nat[s_]['g'][keep[s_]] for s_ in ok],seed=14);gS=_boot_seedmean([swd[s_]['g'][keep[s_]] for s_ in ok],seed=15);Sx=_boot_seedmean([d[s_][keep[s_]] for s_ in ok],seed=16)
            L+=[f'sensitivity (ceiling and floor items removed, {sum(int(keep[s_].sum()) for s_ in ok)} facts): g(NAT) {gN[0]:+.3f} [{gN[1]:+.3f}, {gN[2]:+.3f}], g(SWD) {gS[0]:+.3f} [{gS[1]:+.3f}, {gS[2]:+.3f}], S {Sx[0]:+.3f} [{Sx[1]:+.3f}, {Sx[2]:+.3f}]']
    else:L+=['both below 25%: no sensitivity calculation needed']
    L+=['','**시드 수준 유의성은 3시드로 주장할 수 없다.** 위 신뢰구간은 사실 단위(시드 안에서 사실을 다시 뽑음)이며 학습 세 번 사이의 변동(시드 수준)을 담지 못한다.','(thresholds are fixed constants in e55_analyze.py: 0.15 u = 2.25 positions; existence 0.10 / 0.30 / 0.05 u = 1.5 / 4.5 / 0.75 positions; hold 1.15)']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def stage0(nat_files,swd_files=None,a=16,out=None,swb_files=None,saved=None,extra=None):
    """extra: {'MSK': files, 'NATM': files, ...} further arms of the same seeds (contrasts vs NAT; for NATM also SWD - NATM, the size-matched control)"""
    if is_existence(nat_files):return existence(nat_files,swd_files,(4,16),out)
    load=lambda fs:[json.load(open(f)) for f in fs.split(',')]
    t=MT.stage0(nat_files,swd_files,a,None,thr=THR_S_RANK);nat=load(nat_files);alts={'SWD-A':load(swd_files)} if swd_files else {}
    if swb_files:alts['SWD-B']=load(swb_files)
    for k_,v_ in (extra or {}).items():alts[k_]=load(v_)
    sd=lambda rs:[r['meta']['seed'] for r in rs]
    def pair(x,y):
        sh=set(sd(x))&set(sd(y));return [r for r in x if r['meta']['seed'] in sh],[r for r in y if r['meta']['seed'] in sh]
    for nm,rs in alts.items():
        if nm=='SWD-A':continue
        nn,aa=pair(nat,rs)
        if aa:s_=MT.shadowing(nn,aa,a,thr=THR_S_RANK);t+=f"\n{nm} - NAT (seeds {sd(aa)}): G_rank {MT.fmt(s_['S']['G_rank'])}, PM2 {MT.fmt(s_['S']['PM2'])}, PM3 {MT.fmt(s_['S']['PM3'])} -> {s_['shadowing']}"+(' (METRIC MISMATCH)' if s_['metric_mismatch'] else '')+'\n'
    if 'NATM' in alts and 'SWD-A' in alts:
        mm,aa=pair(alts['NATM'],alts['SWD-A'])
        if aa:s_=MT.shadowing(mm,aa,a,thr=THR_S_RANK);t+=f"\nSWD-A - NATM (size-matched control, seeds {sd(aa)}): G_rank {MT.fmt(s_['S']['G_rank'])}, PM2 {MT.fmt(s_['S']['PM2'])}, PM3 {MT.fmt(s_['S']['PM3'])} -> {s_['shadowing']}\n"
    t+=w0_report(nat,a)
    t+=curve(nat,alts)
    if saved:t+=train_summary(saved)
    if out:Path(out).write_text(t)
    return t
def selftest():
    rng=np.random.default_rng(1);K=8;nf=8
    def rec(sig,fam=0.0):
        n=nf+4;gold=rng.integers(0,K,n);ex=np.array([[True]*K+[False]*8]*n);meta={'kind':['fact']*nf+['ctrl']*4,'gold_idx':gold.tolist(),'cand_exposed':ex.tolist(),'n_rank':[K]*n,'d':{'16':{'in_g0':[1]*nf+[None]*4}},'item_idx':{'16':{'in_g0':list(range(n))}}}
        base=rng.normal(size=(n,16));W=base+0.2*rng.normal(size=(n,16));W[np.arange(nf),gold[:nf]]+=sig;W[:,8:]+=fam
        return {'meta':meta,'read':{'4':{'out':W},'16':{'out':W,'in_g0':W}},'W0':{'out':base,'in_g0@16':base}}
    import tempfile,os
    d=tempfile.mkdtemp();fs=[]
    for i,sig in enumerate((3.0,3.0,3.0,3.0)):
        r=rec(sig);r['read']={k:{c:np.asarray(v).tolist() for c,v in cs.items()} for k,cs in r['read'].items()};r['W0']={c:np.asarray(v).tolist() for c,v in r['W0'].items()};fn=f'{d}/s{i}.json';json.dump(r,open(fn,'w'));fs.append(fn)
    t=existence(','.join(fs),','.join(fs));assert 'shadowing contrast' in t and '| NAT | appearance 16 | out |' in t
    # verdict table on synthetic seeds 55-57 (p = 1 layout: gold among 16 candidates, SWD has the stronger signal)
    def vrec(seed,sig,r):
        r=np.random.default_rng(seed);n=16+4;gold=r.integers(0,16,n);meta={'kind':['fact']*16+['ctrl']*4,'gold_idx':gold.tolist(),'seed':seed,'p':1,'k':16,'n':1,'qframe':[0]*n}
        base=r.normal(size=(n,16));W=base+0.3*np.random.default_rng(seed*7+int(sig*10)).normal(size=(n,16));W[np.arange(16),gold[:16]]+=sig;return {'meta':meta,'read':{'16':{'out':W.tolist()}},'W0':{'out':base.tolist()}}
    r=np.random.default_rng(5);vf=[]
    for arm,sig in (('NAT',1.0),('SWD',2.5)):
        for sd in (55,56,57):fn=f'{d}/{arm}_s{sd}.read.json';json.dump(vrec(sd,sig,r),open(fn,'w'))
    nf_=','.join(f'{d}/NAT_s{sd}.read.json' for sd in (55,56,57));sf_=','.join(f'{d}/SWD_s{sd}.read.json' for sd in (55,56,57))
    for arm,mult in (('NAT',1.0),('SWD',1.2)):
        for sd in (55,56,57):json.dump({'world_seed':sd,'log':[{'update_norm':0.5*mult,'gnorm':1.0,'coef':0.9,'ce':3.0}]*4},open(f'{d}/E55_st0_{arm}_s{sd}.saved.json','w'))
    sv_=','.join(f'{d}/E55_st0_{arm}_s{sd}.saved.json' for arm in ('NAT','SWD') for sd in (55,56,57));t2=verdict(nf_,sf_,sv_)
    assert 'classification' in t2 and 'ON HOLD' in t2 and '시드 수준 유의성은 3시드로 주장할 수 없다' in t2 and '| 56 |' in t2 and 'W0 ceiling' in t2,t2
    tie=MT.ranks(np.array([[1.,1.,0.]]));assert tie.tolist()==[[1.5,1.5,3.0]]
    print(t2.split('classification:')[1].splitlines()[0])
    print(t.splitlines()[3]);print('e55_analyze SELFTEST OK (existence report on synthetic records)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--nat');ap.add_argument('--swd',default=None);ap.add_argument('--swb',default=None);ap.add_argument('--extra',default=None,help="further arms 'MSK=f1,f2;NATM=f3'");ap.add_argument('--saved',default=None,help='comma list of *.saved.json (training statistics table)');ap.add_argument('--verdict',action='store_true',help='first-test verdict table (needs --nat --swd; --saved and --unit recommended)');ap.add_argument('--unit',default=None);ap.add_argument('--a',type=int,default=16);ap.add_argument('--out',default=None);a=ap.parse_args()
    if a.selftest:selftest()
    elif a.verdict:print(verdict(a.nat,a.swd,a.saved,a.a,a.unit,a.out))
    else:print(stage0(a.nat,a.swd,a.a,a.out,a.swb,a.saved,dict(x.split('=') for x in a.extra.split(';')) if a.extra else None))
