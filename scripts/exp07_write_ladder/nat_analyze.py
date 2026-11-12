"""E33 R analysis (CPU). Reads {cell}_{tag}_s{stream}_{A,B}.json for cells L0fs (native fs), L0mwfs (chunk alone fs), L1 (+ meta_nat_{tag}_s{stream}.json) from the ladder output dir.
Two readings: (1) CHANGE FROM W0 per world (below), (2) the registered D(A-B) per pair when world B exists (analyze_D). Change from W0: per person i  chg_k(i) = [L3_i - XP_i]_k - [L3_i - XP_i]_0  (state 0 = W0, before any write; XP = cross-pair null of the same person).
Mean over persons, 95% CI by pair-cluster bootstrap (13 pairs, resample pairs with replacement, 5000 draws, seed 0). Both readouts: R1 (pg1342 neutral tail) and R1-alt (SQuAD tail).
Rows: native pass 1..P, masked pass 1..P, L1 epochs, masked - native (person-paired) per pass, R* = masked change / L1 change at the last L1 epoch (point estimate and bootstrap CI of the ratio of means),
W0 margin distribution (L3, XP; optional exclusion --w0-cut: persons of pairs with any |L3 W0| above the cut are dropped from a second table), canary (gen / tail NLL change from W0), other families (FN, FP, L3b, L3e) mean change.
Verdict thresholds are NOT applied here beyond printing the E33 2.10 reading aids: masked-native at pass 3 >= 0.25, canary change <= 0.10.
Usage: nat_analyze.py --dir OUT --stream 9105 [--tag nat] [--w0-cut X] [--out report.md] | --selftest"""
import sys,json,argparse
from pathlib import Path
import numpy as np
NB=5000
def load(d,cell,tag,stream,w='A'):
    f=Path(d)/f'{cell}_{tag}_s{stream}_{w}.json';return json.load(open(f)) if f.exists() else None
def states(log):return [e for e in log if 'margins' in e and '_file' not in e['margins']]
def fam_slices(meta):
    o={};i=0
    for k,n in meta['fam_sizes'].items():o[k]=(i,i+n);i+=n
    return o
def fam_items(meta,k):
    a,b=fam_slices(meta)[k];return meta['items'][a:b]
def person_vec(meta,m,fam,alt):
    """margin list of a family -> dict tid -> margin"""
    its=fam_items(meta,fam);src=m['_alt'][fam] if alt else m[fam];assert len(src)==len(its),(fam,len(src),len(its))
    return {it['tid']:v for it,v in zip(its,src)}
def bind(meta,m,alt):
    l3=person_vec(meta,m,'L3',alt);xp=person_vec(meta,m,'XP',alt);return {t:l3[t]-xp[t] for t in l3}
def pair_of(meta):return {it['tid']:it['pair'] for it in fam_items(meta,'L3')}
def boot(vals,pairs,rng,fn=np.mean):
    """vals: dict tid->value; pairs: dict tid->pair. returns (point, lo, hi) cluster bootstrap over pairs."""
    P=sorted(set(pairs[t] for t in vals));grp={p:[vals[t] for t in vals if pairs[t]==p] for p in P}
    pt=float(np.mean([x for v in grp.values() for x in v]));out=[]
    for _ in range(NB):
        s=rng.integers(0,len(P),len(P));out.append(np.mean([x for i in s for x in grp[P[i]]]))
    lo,hi=np.percentile(out,[2.5,97.5]);return pt,float(lo),float(hi)
def change(meta,log,k_idx,alt):
    S=states(log);b0=bind(meta,S[0]['margins'],alt);bk=bind(meta,S[k_idx]['margins'],alt);return {t:bk[t]-b0[t] for t in b0}
def fmt(x):return f'{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]'
def analyze(d,stream,tag='nat',w0cut=None,w='A'):
    meta=json.load(open(Path(d)/f'meta_nat_{tag}_s{stream}.json'));pairs=pair_of(meta)
    cells={c:load(d,c,tag,stream,w) for c in ('L0fs','L0mwfs','L1')}
    rng=np.random.default_rng(0);rep={'tables':{},'w0':{},'canary':{},'other':{}};L=[f'\n## World {w}: change from W0\n']
    for alt,nm in ((True,'R1-alt (SQuAD tail)'),(False,'R1 (pg1342 tail)')):
        rows=[];tab={}
        for c,lab in (('L0fs','native-fs'),('L0mwfs','masked-fs (chunk alone)')):
            if not cells[c]:continue
            S=states(cells[c]['log'])
            for k in range(1,len(S)):
                ch=change(meta,cells[c]['log'],k,alt);tab[(c,k)]=ch;rows.append((f'{lab} pass {k}',boot(ch,pairs,rng)))
        if cells['L1']:
            S=states(cells['L1']['log'])
            for k in range(1,len(S)):
                ch=change(meta,cells['L1']['log'],k,alt);tab[('L1',S[k].get('epoch',k))]=ch;rows.append((f"L1 doc epoch {S[k].get('epoch',k)}",boot(ch,pairs,rng)))
        for k in sorted({k for (c,k) in tab if c=='L0fs'}&{k for (c,k) in tab if c=='L0mwfs'}):
            df={t:tab[('L0mwfs',k)][t]-tab[('L0fs',k)][t] for t in tab[('L0fs',k)]};rows.append((f'masked - native pass {k}',boot(df,pairs,rng)))
        l1k=[k for (c,k) in tab if c=='L1']
        if l1k and ('L0mwfs',max(k for (c,k) in tab if c=='L0mwfs')) in tab:
            kp=max(k for (c,k) in tab if c=='L0mwfs');kl=max(l1k);P=sorted(set(pairs.values()));a=tab[('L0mwfs',kp)];b=tab[('L1',kl)]
            ratio=lambda ts:np.mean([a[t] for t in ts])/np.mean([b[t] for t in ts]) if abs(np.mean([b[t] for t in ts]))>1e-9 else float('nan')
            allt=list(a);pt=ratio(allt);bs=[]
            for _ in range(NB):
                s=rng.integers(0,len(P),len(P));ts=[t for i in s for t in allt if pairs[t]==P[i]];bs.append(ratio(ts))
            lo,hi=np.nanpercentile(bs,[2.5,97.5]);rows.append((f'R* = masked pass {kp} / L1 epoch {kl} (ratio of means)',(float(pt),float(lo),float(hi))))
        rep['tables'][nm]=[(r,v) for r,v in rows];L+=[f'\n### world {w}, {nm}: change from W0 of [L3 - XP], mean over persons (95% pair-cluster bootstrap CI)\n','| cell | change |','|---|---|']+[f'| {r} | {fmt(v)} |' for r,v in rows]
    # W0 distribution
    for alt,nm in ((True,'R1-alt'),(False,'R1')):
        c=cells['L0fs'] or cells['L0mwfs'] or cells['L1'];m0=states(c['log'])[0]['margins']
        l3=person_vec(meta,m0,'L3',alt);xp=person_vec(meta,m0,'XP',alt);v=np.array(list(l3.values()));x=np.array(list(xp.values()))
        q=lambda a:[float(np.percentile(a,p)) for p in (5,25,50,75,95)]
        rep['w0'][nm]={'L3_mean':float(v.mean()),'L3_sd':float(v.std()),'L3_quantiles_5_25_50_75_95':q(v),'L3_frac_pos':float((v>0).mean()),'XP_mean':float(x.mean()),'XP_sd':float(x.std()),'abs_L3_p90':float(np.percentile(np.abs(v),90))}
        L.append(f"\nW0 {nm}: L3 mean {v.mean():+.3f} sd {v.std():.3f} (5/25/50/75/95%: {', '.join(f'{y:+.2f}' for y in q(v))}), fraction > 0 {np.mean(v>0):.2f}; XP mean {x.mean():+.3f} sd {x.std():.3f}; |L3| 90th percentile {np.percentile(np.abs(v),90):.3f}")
    # canary and other families
    for c,lab in (('L0fs','native-fs'),('L0mwfs','masked-fs'),('L1','L1')):
        if not cells[c]:continue
        S=states(cells[c]['log']);g0=S[0]['margins']['_gen_nll'];t0=S[0]['margins']['_tail_nll']
        rep['canary'][lab]=[(S[k].get('epoch',k),S[k]['margins']['_gen_nll']-g0,S[k]['margins']['_tail_nll']-t0) for k in range(1,len(S))]
        rep['other'][lab]={f:[float(np.mean(S[k]['margins'][f])-np.mean(S[0]['margins'][f])) for k in range(1,len(S))] for f in ('FN','FP','L3b','L3e') if f in S[0]['margins'] and len(S[0]['margins'][f])}
    L+=['\n### canary: change of NLL from W0 (gen pg11 / neutral tail pg1342), criterion <= 0.10\n','| cell | state | gen NLL change | tail NLL change |','|---|---|---|---|']
    for lab,v in rep['canary'].items():L+=[f'| {lab} | {e} | {g:+.3f} | {t:+.3f} |' for e,g,t in v]
    L+=['\n### other families: mean margin change from W0 (R1, per state)\n','| cell | family | change per state |','|---|---|---|']
    for lab,v in rep['other'].items():L+=[f"| {lab} | {f} | {', '.join(f'{x:+.3f}' for x in xs)} |" for f,xs in v.items()]
    if w0cut is not None:
        c=cells['L0fs'] or cells['L0mwfs'];m0=states(c['log'])[0]['margins'];l3=person_vec(meta,m0,'L3',True);bad={pairs[t] for t,v in l3.items() if abs(v)>w0cut};keep=[t for t in pairs if pairs[t] not in bad]
        L.append(f'\nW0 exclusion (|L3 R1-alt| > {w0cut} in any member): excluded pairs {sorted(bad)}, persons kept {len(keep)}/{len(pairs)}')
        for cc,lab in (('L0fs','native-fs'),('L0mwfs','masked-fs')):
            if not cells[cc]:continue
            S=states(cells[cc]['log'])
            for k in range(1,len(S)):
                ch=change(meta,cells[cc]['log'],k,True);L.append(f'- {lab} pass {k} (kept persons, R1-alt): {fmt(boot({t:ch[t] for t in keep},pairs,rng))}')
    L.append(f"\nreading aids (E33 2.10): masked - native at the last pass >= 0.25 -> a synthetic-size gain (>= 30%) is visible; CI within +-0.2 around 0 -> not visible in this stream; 26 persons / 13 clusters per world; the change-from-W0 tables are per world, the D table is the registered A-B quantity.")
    rep['pass_info']={c:cells[c].get('pass_info') for c in ('L0fs','L0mwfs') if cells[c]}
    return rep,'\n'.join(L)+'\n'

def Dperson(meta,eA,eB,alt):
    """E33 2.11 D per person: [mA(L3_i) - mB(L3_i)] - [mA(XP_i) - mB(XP_i)] (un-halved; analyze_ladder.py's D is half of this)."""
    l3a,l3b=person_vec(meta,eA,'L3',alt),person_vec(meta,eB,'L3',alt);xpa,xpb=person_vec(meta,eA,'XP',alt),person_vec(meta,eB,'XP',alt)
    return {t:(l3a[t]-l3b[t])-(xpa[t]-xpb[t]) for t in l3a}
def accflip(meta,eA,eB,alt):
    a=np.asarray((eA['_alt']['L3'] if alt else eA['L3']));b=np.asarray((eB['_alt']['L3'] if alt else eB['L3']));return float(((a>0).mean()+(b<0).mean())/2),float(((a>0)&(b<0)).mean())
def ratio_boot(a,b,pairs,rng):
    """ratio of person means a/b with pair-cluster bootstrap"""
    P=sorted(set(pairs.values()));ts=list(a);pt=np.mean([a[t] for t in ts])/np.mean([b[t] for t in ts]);bs=[]
    for _ in range(NB):
        s=rng.integers(0,len(P),len(P));tt=[t for i in s for t in ts if pairs[t]==P[i]];den=np.mean([b[t] for t in tt]);bs.append(np.mean([a[t] for t in tt])/den if abs(den)>1e-9 else np.nan)
    lo,hi=np.nanpercentile(bs,[2.5,97.5]);return float(pt),float(lo),float(hi)
def exposure_groups(cells):
    """persons by their pass-1 fact exposures (real tokenization, world A): single = one fact unit, window = a repeat inside 8K (s_cross + s_same >= 1), out = repeats only outside the window."""
    pi=None
    for c in ('L0fs','L0mwfs'):
        j=cells[c][0]
        if j and j.get('pass_info'):pi=j['pass_info'].get('False') or j['pass_info'].get('true');break
    if not pi:return {}
    st=pi[0]['strata'];g={'single':[],'window':[],'out':[]}
    for t,v in st.items():
        t=int(t);(g['single'] if v['n_fact']<=1 else g['window'] if v['s_cross']+v['s_same']>=1 else g['out']).append(t)
    return g
def analyze_D(d,stream,tag='nat'):
    meta=json.load(open(Path(d)/f'meta_nat_{tag}_s{stream}.json'));pairs=pair_of(meta);rng=np.random.default_rng(1);L=['\n## D(A-B) (E33 2.11 definition, per person, un-halved; 13 pair clusters; state 0 = W0 gives D = 0)\n'];rep={}
    cells={c:(load(d,c,tag,stream,'A'),load(d,c,tag,stream,'B')) for c in ('L0fs','L0mwfs','L1')}
    if not all(cells[c][0] and cells[c][1] for c in cells if cells[c][0] or cells[c][1]):return rep,'\nD(A-B): world B missing for some cell, skipped\n'
    grp=exposure_groups(cells)
    for alt,nm in ((True,'R1-alt (SQuAD tail)'),(False,'R1 (pg1342 tail)')):
        tab={};rows=[]
        for c,lab in (('L0fs','native-fs pass'),('L0mwfs','masked-fs pass'),('L1','L1 doc epoch')):
            if not cells[c][0]:continue
            SA=states(cells[c][0]['log']);SB=states(cells[c][1]['log']);assert len(SA)==len(SB)
            for k in range(1,len(SA)):
                e=SA[k].get('epoch',k);D=Dperson(meta,SA[k]['margins'],SB[k]['margins'],alt);tab[(c,e)]=D;ac,fl=accflip(meta,SA[k]['margins'],SB[k]['margins'],alt)
                rows.append((f'{lab} {e}',boot(D,pairs,rng),f'{ac:.2f} / {fl:.2f}'))
        for e in sorted({e for c,e in tab if c=='L0fs'}&{e for c,e in tab if c=='L0mwfs'}):rows.append((f'masked - native pass {e}',boot({t:tab[('L0mwfs',e)][t]-tab[('L0fs',e)][t] for t in tab[('L0fs',e)]},pairs,rng),''))
        kl=[e for c,e in tab if c=='L1'];km=[e for c,e in tab if c=='L0mwfs']
        if kl and km:rows.append((f'R* = masked pass {max(km)} D / L1 epoch {max(kl)} D (ratio of means)',ratio_boot(tab[('L0mwfs',max(km))],tab[('L1',max(kl))],pairs,rng),''))
        if kl:
            e=max(kl);rows.append((f'L1 epoch {e} D lower bound > 0 (readable)?',boot(tab[('L1',e)],pairs,rng),'yes' if boot(tab[('L1',e)],pairs,rng)[1]>0 else 'NO: stream not readable'))
        if grp and km and ('L0fs',max(km)) in tab:
            e=max(km)
            for g,ts in grp.items():
                if ts:
                    df={t:tab[('L0mwfs',e)][t]-tab[('L0fs',e)][t] for t in ts};rows.append((f'S4 masked - native pass {e}, persons {g} (n={len(ts)}, table only)',(float(np.mean(list(df.values()))),float('nan'),float('nan')),''))
        rep[nm]=[(r,v,x) for r,v,x in rows]
        L+=[f'\n### {nm}: D, mean over persons (95% CI by bootstrap over pairs)\n','| cell | D | acc / flip |','|---|---|---|']+[f'| {r} | {fmt(v)} | {x} |' for r,v,x in rows]
    L.append('\nE33 2.11 reading: S1 masked - native D(3) point >= +0.25 -> signal; CI inside +-0.2 -> not visible; S2 R* >= 0.30 at pass 3 (needs L1 e8 D lower bound > 0); S3 canary <= 0.10 (see canary table) and first-chunk NLL (e37 step); S4 tables without interpretation. D = C_A - C_B (change-from-W0 tables per world). analyze_ladder.py D is half of this D.')
    return rep,'\n'.join(L)+'\n'

def export_person_margins(d,stream,tag='nat'):
    """one json with every item margin (gold - foil) per cell / world / state, with the person id: {cell: {world: [{state, family, tid, pair, R1, R1_alt}]}} (state 0 = W0)."""
    meta=json.load(open(Path(d)/f'meta_nat_{tag}_s{stream}.json'));out={}
    for c in ('L0fs','L0mwfs','L1'):
        for w in 'AB':
            j=load(d,c,tag,stream,w)
            if not j:continue
            rows=[]
            for k,e in enumerate(states(j['log'])):
                m=e['margins']
                for f,(a,b) in fam_slices(meta).items():
                    for it,x,y in zip(meta['items'][a:b],m[f],m['_alt'][f]):rows.append({'state':e.get('epoch',k),'family':f,'tid':it['tid'],'pair':it['pair'],'R1':x,'R1_alt':y})
            out.setdefault(c,{})[w]=rows
    return out
def selftest():
    import tempfile
    d=Path(tempfile.mkdtemp());rng=np.random.default_rng(1);P=13
    items=[];fam_sizes={'L3':2*P,'XP':2*P,'PS':2*P,'FN':2*P,'FP':2*P}
    for f,n in fam_sizes.items():
        for j in range(n):items.append({'level':{'FN':'L1p','FP':'L1l'}.get(f,f),'tid':j,'pair':f'p{j//2}'})
    json.dump({'items':items,'fam_sizes':fam_sizes},open(d/'meta_nat_nat_s9105.json','w'))
    def margins(gain,w0,noise=0.05):
        m={}
        for f,n in fam_sizes.items():m[f]=(rng.normal(0,0.05,n)+(w0+gain if f=='L3' else 0)).tolist()
        m['_alt']={f:(np.array(v)+0.01).tolist() for f,v in m.items() if f!='_alt'};m['_gen_nll']=0.5;m['_tail_nll']=0.6;return m
    w0=rng.normal(0.2,0.1,2*P)
    def mk(gains):
        log=[]
        for k,g in enumerate([0]+gains):
            m=margins(0,0);m['L3']=(w0+g+rng.normal(0,0.05,2*P)).tolist();m['_alt']['L3']=[x+0.01 for x in m['L3']];log.append({'epoch':k,'margins':m})
        return {'log':log}
    json.dump(mk([0.05,0.1,0.15]),open(d/'L0fs_nat_s9105_A.json','w'));json.dump(mk([0.1,0.3,0.6]),open(d/'L0mwfs_nat_s9105_A.json','w'));json.dump(mk([0.7,1.2]),open(d/'L1_nat_s9105_A.json','w'))
    # world B logs: L3 margin = w0 - gain (swapped binding), XP unchanged
    for c,g in (('L0fs',[0.05,0.1,0.15]),('L0mwfs',[0.1,0.3,0.6]),('L1',[0.7,1.2])):
        jB=json.load(open(d/f'{c}_nat_s9105_A.json'))
        for k,e in enumerate(jB['log']):
            gg=([0]+g)[k];e['margins']['L3']=[w0[i]-gg+0.0 for i in range(2*P)];e['margins']['_alt']['L3']=[x+0.01 for x in e['margins']['L3']]
        json.dump(jB,open(d/f'{c}_nat_s9105_B.json','w'))
    jA=json.load(open(d/'L0fs_nat_s9105_A.json'));jA['pass_info']={'False':[{'strata':{str(t):{'n_fact':1 if t<8 else 2,'s_cross':1 if t>=14 else 0,'s_same':0,'n_out':0 if t>=14 else (1 if t>=8 else 0)} for t in range(2*P)}}]};json.dump(jA,open(d/'L0fs_nat_s9105_A.json','w'))
    rd,mdD=analyze_D(d,9105);assert sum('S4' in r for r,v,x in rd['R1-alt (SQuAD tail)'])==3,[r for r,v,x in rd['R1-alt (SQuAD tail)']];tD=dict((r,v) for r,v,x in rd['R1-alt (SQuAD tail)'])
    assert abs(tD['native-fs pass 3'][0]-0.30)<0.08 and abs(tD['masked-fs pass 3'][0]-1.2)<0.08,tD
    rep,md=analyze(d,9105,w0cut=0.5);t=dict(rep['tables']['R1-alt (SQuAD tail)'])
    assert abs(t['native-fs pass 3'][0]-0.15)<0.06 and abs(t['masked-fs (chunk alone) pass 3'][0]-0.6)<0.06,t
    assert abs(t['masked - native pass 3'][0]-0.45)<0.08 and t['masked - native pass 3'][1]>0.3
    r=[v for k,v in t.items() if k.startswith('R*')][0];assert abs(r[0]-0.5)<0.1,r
    ex=export_person_margins(d,9105);assert set(ex)=={'L0fs','L0mwfs','L1'} and len(ex['L0fs']['A'])==4*sum(fam_sizes.values()),len(ex['L0fs']['A'])
    print(md);print('nat_analyze SELFTEST OK')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--stream',type=int,default=9105);ap.add_argument('--tag',default='nat');ap.add_argument('--w0-cut',type=float,default=None);ap.add_argument('--out',default=None);ap.add_argument('--selftest',action='store_true')
    a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    rep,md=analyze(a.dir,a.stream,a.tag,a.w0_cut)
    if load(a.dir,'L0fs',a.tag,a.stream,'B') or load(a.dir,'L0mwfs',a.tag,a.stream,'B'):
        r2,m2=analyze(a.dir,a.stream,a.tag,a.w0_cut,'B');rep['world_B']=r2;md+=m2
        rd,md_d=analyze_D(a.dir,a.stream,a.tag);rep['D']=rd;md=md_d+md
    out=Path(a.out or Path(a.dir)/f'nat_r_report_{a.stream}.md');out.write_text('# E33 R report (stream %d)\n'%a.stream+md);json.dump(rep,open(out.with_suffix('.json'),'w'),indent=1,default=str);json.dump(export_person_margins(a.dir,a.stream,a.tag),open(Path(a.dir)/f'nat_person_margins_{a.stream}.json','w'));print(md)
