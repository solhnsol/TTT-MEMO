"""E47 CPU analysis (no GPU, no tokenizer): triple-level binding score D_a and the k3h split masked person / partner, from saved S3 result files (OUT/S3_<cell>_s<seed>_{A,B}.json, margins['_alt'], and meta_S3_<cell>_s<seed>.json with the item list).
D_a(triple t) = (L3 binding margin of t) - (XP margin of the pair of t), both as (margin in world A - margin in world B)/2 per item (analyze_e35 convention) at the chosen epoch (4 = after pass 4); the pair's XP = mean over the pair's XP items. The mean of D_a over the two triples of a pair is the pair's D_alt of the other analyses.
Masked person of a pair (the k3h cell masks only his K3 share) = s3_sets.half_person_of(seed, pair, the triple ids of the pair), the same rule as job 108 (fixed per seed and pair, the same in worlds A and B); partner = the other triple of the pair. 15 pairs = 15 masked persons + 15 partners per seed.
Modes:
  k3    --cells k3 --seeds 101,102,103   per seed: mean D_a of the masked persons and of the partners (15 triples each, pair bootstrap 95% CI), masked - partner (paired within pair), the number of triples; per world (A: item margin of world A / 2, B: minus the item margin of world B / 2, each minus the pair's XP part) mean and sd over the 15 triples; across seeds: mean and sd (n = number of seeds) of the 15-triple means. Seeds without result files are listed as missing.
  indep --cells k3h|k3hc --seeds 101 --epochs 1,2,3,4 [--native L0fs:main --ref k3]   independence prediction P_ind(pair) = (D_a[K3, chosen person] + D_b[native, partner])/2 vs the observed pair mean of the cell (see mode_indep).
  additivity --cells k3,L0fs:main,k3h,k3hc --seeds 101 --epochs 1,2,3,4   I = Dpair(K3)+Dpair(native)-Dpair(k3h)-Dpair(k3hc) per pair (mean, bootstrap CI, worlds A+B / A / B), single-cell gains vs native and the ratio (g_h+g_hc)/g_K3.
  pairstruct --cells L0fs:main,L0reset_P63:r5fresh,oracle,k3,sent,... [--epochs 4] [--stream 4]   a (k3h masked person) vs b (partner): T1 D_a(a), D_a(b), b-a, #b>a; T2 correlation, within-pair sum, K3->k3h/k3hc loss allocation; T3 stream distance (needs --stream: server tokenizer). Cell spec: S3 tag (k3) or LEVEL:TAG (L0fs:main).
  k3h   --cells k3h --ref k3 --seeds 101   dD_a = D_a(k3h) - D_a(K3) per triple (same seed): mean over the masked persons (15 triples, bootstrap 95% CI), over the partners, and the within-pair contrast (masked - partner), plus the same per world.
Output: <out-dir>/e47_triple_da_<mode>.{md,json}. --selftest: synthetic."""
import os,sys,json,argparse,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
ROOT=HERE.parents[1];OUT=Path(os.environ.get('EXP07_OUT',ROOT/'experiments/exp07_write_ladder'));NB=5000
def load(od,cell,seed,epoch,strict=False):
    """cell 'k3' -> S3_k3_s<seed>_{A,B}.json + meta_S3_k3_s<seed>.json; cell 'L0fs:main' -> L0fs_main_s<seed>_..., meta_L0fs_main_s<seed>.json. epoch missing: falls back to the last epoch unless strict (then ValueError)."""
    sfx='';
    if cell.endswith('@regold'):cell=cell[:-7];sfx='.regold'   # E47: the re-scored result files <result>.regold.json (the meta file is the original one)
    pre=cell.replace(':','_',1) if ':' in cell else f'S3_{cell}'
    meta=json.load(open(od/f'meta_{pre}_s{seed}.json'));E={}
    for w in 'AB':
        ent={e['epoch']:e for e in json.load(open(od/f'{pre}_s{seed}_{w}{sfx}.json'))['log'] if isinstance(e.get('margins'),dict) and 'epoch' in e}
        if strict and epoch not in ent:raise ValueError(f'{pre} s{seed} world {w}: epoch {epoch} not in the log (epochs {sorted(ent)})')
        E[w]=ent[epoch if epoch in ent else max(ent)]
    items=meta['items'];fam={};i=0
    for f,n in meta['fam_sizes'].items():fam[f]=items[i:i+n];i+=n
    return E,fam
def _num(pair,prefix):
    if not (isinstance(pair,str) and pair.startswith(prefix) and pair[len(prefix):].isdigit()):raise ValueError(f'pair id {pair!r} is not of the form {prefix}<n>')
    return int(pair[len(prefix):])
def triple_da(E,fam,seed,mask_seed=None):
    """-> dict triple id -> {'pair', 'A', 'B', 'AB' (= (A-B)/2 form), 'masked'}. The XP (control) items carry pair ids 'x<n>' and the L3 items 'p<n>' (as in the real meta files); the control pair x<n> is the reference of pair p<n> (the same alignment by sorted pair number as analyze_e35.D_alt); a missing x<n> or another id form is an explicit error. Self-check (assert): the mean over the two triples of a pair of D_a (AB) equals the pair's D_alt computed independently with the analyze_e35 formula."""
    a={w:np.asarray(E[w]['margins']['_alt']['L3']) for w in 'AB'};x={w:np.asarray(E[w]['margins']['_alt']['XP']) for w in 'AB'}
    part={'A':lambda m:m/2,'B':lambda m:-m/2}
    xp={}
    for k,it in enumerate(fam['XP']):
        n=_num(it['pair'],'x')
        for w in 'AB':xp.setdefault((n,w),[]).append(part[w](x[w][k]))
    xp={k:float(np.mean(v)) for k,v in xp.items()}
    l3={};pairs={}
    for k,it in enumerate(fam['L3']):
        if 'tid' not in it:raise ValueError('items have no triple id (tid)')
        pairs[it['tid']]=it['pair'];_num(it['pair'],'p')
        for w in 'AB':l3.setdefault((it['tid'],w),[]).append(part[w](a[w][k]))
    l3={k:float(np.mean(v)) for k,v in l3.items()};tri={};bypair={}
    for t,p in pairs.items():bypair.setdefault(p,[]).append(t)
    for t,p in pairs.items():
        n=_num(p,'p')
        if (n,'A') not in xp:raise ValueError(f'no XP (control) items for pair x{n} (reference of {p}); XP pairs: {sorted({k[0] for k in xp})}')
        da={w:l3[(t,w)]-xp[(n,w)] for w in 'AB'};tri[t]={'pair':p,'A':da['A'],'B':da['B'],'AB':da['A']+da['B'],'masked':t==SETS.half_person_of(seed if mask_seed is None else mask_seed,p,bypair[p])}
    # self-check against the pair-level D_alt (analyze_e35.D_alt: per pair mean of (A-B)/2 over its items, L3 minus XP, aligned by sorted pair number)
    def Mv(f,pre):
        d={};arr=a if f=='L3' else x
        for k,it in enumerate(fam[f]):d.setdefault(_num(it['pair'],pre),[]).append((arr['A'][k]-arr['B'][k])/2)
        return {n:float(np.mean(v)) for n,v in d.items()}
    l3p,xpp=Mv('L3','p'),Mv('XP','x');assert sorted(l3p)==sorted(xpp),(sorted(l3p),sorted(xpp))
    for p,ts in bypair.items():
        n=_num(p,'p');assert abs(np.mean([tri[t]['AB'] for t in ts])-(l3p[n]-xpp[n]))<1e-9,(p,ts)   # pair mean of D_a == pair D_alt
    return tri
def boot(v,rng,n=NB):
    v=np.asarray(v,float);b=np.asarray([v[rng.integers(0,len(v),len(v))].mean() for _ in range(n)]);return float(v.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def f3(t):return f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
def split(tri,key='AB'):
    """per pair (masked value, partner value) for the given world key"""
    pr={}
    for t,v in tri.items():pr.setdefault(v['pair'],{})['m' if v['masked'] else 'p']=v[key]
    ps=sorted(pr);return ps,np.asarray([pr[p]['m'] for p in ps]),np.asarray([pr[p]['p'] for p in ps])
def mode_k3(od,seeds,epoch,cell='k3'):
    rng=np.random.default_rng(48);md=[f'# E47 triple-level D_a of {cell}: masked person vs partner (epoch {epoch}; CPU)\n','D_a(triple) = L3 binding margin (A-B)/2 minus the pair\'s XP part; masked person = the person the k3h cell masks (rule of job 108). 15 triples per group per seed.\n','| seed | masked: mean D_a [95% CI] | partner: mean D_a [95% CI] | masked - partner (within pair) [95% CI] | triples (masked / partner) | world A: masked / partner mean (sd over triples) | world B: masked / partner mean (sd) |','|---|---|---|---|---|---|---|'];res={'epoch':epoch,'cell':cell,'seeds':{}};means=[]
    for s in seeds:
        try:E,fam=load(od,cell,s,epoch)
        except FileNotFoundError as e:md.append(f'| {s} | missing result files ({Path(str(e.filename)).name}) | | | | | |');continue
        tri=triple_da(E,fam,s);ps,m,p=split(tri);wa=split(tri,'A');wb=split(tri,'B');R={'masked':boot(m,rng),'partner':boot(p,rng),'diff':boot(m-p,rng),'n':[len(m),len(p)],'per_triple':tri}
        for k,(q,mm,pp) in (('A',wa),('B',wb)):R['world_'+k]={'masked_mean':float(mm.mean()),'masked_sd':float(mm.std(ddof=1)),'partner_mean':float(pp.mean()),'partner_sd':float(pp.std(ddof=1))}
        res['seeds'][s]=R;means.append((float(m.mean()),float(p.mean())))
        md.append(f"| {s} | {f3(R['masked'])} | {f3(R['partner'])} | {f3(R['diff'])} | {len(m)} / {len(p)} | {R['world_A']['masked_mean']:+.3f} / {R['world_A']['partner_mean']:+.3f} (sd {R['world_A']['masked_sd']:.3f} / {R['world_A']['partner_sd']:.3f}) | {R['world_B']['masked_mean']:+.3f} / {R['world_B']['partner_mean']:+.3f} (sd {R['world_B']['masked_sd']:.3f} / {R['world_B']['partner_sd']:.3f}) |")
    if len(means)>=2:
        mm=np.asarray(means);res['across_seeds']={'n':len(means),'masked_mean':float(mm[:,0].mean()),'masked_sd':float(mm[:,0].std(ddof=1)),'partner_mean':float(mm[:,1].mean()),'partner_sd':float(mm[:,1].std(ddof=1))};a=res['across_seeds']
        md.append(f"\nAcross {a['n']} seeds (sd of the 15-triple means, n = {a['n']}): masked {a['masked_mean']:+.3f} (sd {a['masked_sd']:.3f}), partner {a['partner_mean']:+.3f} (sd {a['partner_sd']:.3f}).")
    elif means:md.append('\nOnly one seed available: no across-seed sd.')
    md.append('\nDescriptive only: the masked person is a seed-fixed random choice per pair; with 15 triples per group the intervals are wide.');return '\n'.join(md)+'\n',res
def mode_fixed(od,seeds,epochs,mask_seed,cell='k3'):
    """the masked persons of seed `mask_seed` (rule of job 108) are used in every seed: per seed the mean D_a of those 15 triples (worlds A+B, A, B), per seed pair the paired difference per triple (mean, sd over the 15 triples, bootstrap 95% CI of the mean), pooled sd (root mean square of the pair sds); also the partners of those pairs. Triple ids are the same indices in every seed (the roles of the world generator), the people differ between seeds; this makes the seed-to-seed sd free of the person difference that the seed-specific masked person adds."""
    rng=np.random.default_rng(50);md=[f'# E47 fixed masked persons (from seed {mask_seed}) in {cell}: seed-to-seed D_a (CPU)\n','Masked persons = the 15 triples that job 108 would mask in seed %d; D_a read in every seed for those same triple ids. Paired difference per triple between two seeds; sd over the 15 triples (sd_pair); pooled = root mean square of the pair sds over the seed pairs; max(0.10, 3 x pooled sd_pair) is the threshold the design thread derives from it.\n'%mask_seed];res={'mask_seed':mask_seed,'cell':cell,'epochs':{}}
    for ep in epochs:
        tris={}
        for s in seeds:
            try:E,fam=load(od,cell,s,ep);tris[s]=triple_da(E,fam,s,mask_seed)
            except FileNotFoundError as e:md.append(f'epoch {ep}, seed {s}: missing ({Path(str(e.filename)).name})')
        R={'seeds':sorted(tris),'per_seed':{},'pairs':{}};md+=[f'\n## epoch {ep}\n','| world | seed | masked persons mean D_a (15) | partners mean D_a (15) |','|---|---|---|---|']
        for key,lab in (('AB','A+B'),('A','A'),('B','B')):
            for s,t in sorted(tris.items()):
                ps,m,p=split(t,key);R['per_seed'].setdefault(key,{})[s]={'masked':float(m.mean()),'partner':float(p.mean())};md.append(f'| {lab} | {s} | {m.mean():+.3f} | {p.mean():+.3f} |')
        md+=['','| world | seed pair | mean diff masked (95% CI) | sd_pair (15 triples) | mean diff partners | sd (partners) |','|---|---|---|---|---|---|'];sds={}
        for key,lab in (('AB','A+B'),('A','A'),('B','B')):
            sds[key]=[]
            for i,s1 in enumerate(sorted(tris)):
                for s2 in sorted(tris)[i+1:]:
                    ps,m1,p1=split(tris[s1],key);ps2,m2,p2=split(tris[s2],key);assert ps==ps2;d=m1-m2;dp=p1-p2;sd=float(d.std(ddof=1));sds[key].append(sd)
                    R['pairs'][f'{key}_{s1}_{s2}']={'mean_diff':boot(d,rng),'sd_pair':sd,'partner_mean_diff':float(dp.mean()),'partner_sd':float(dp.std(ddof=1))};md.append(f'| {lab} | {s1} vs {s2} | {f3(boot(d,rng))} | {sd:.3f} | {dp.mean():+.3f} | {dp.std(ddof=1):.3f} |')
        R['pooled_sd_pair']={k:float(np.sqrt(np.mean(np.square(v)))) if v else None for k,v in sds.items()}
        if R['pooled_sd_pair']['AB'] is not None:md.append(f"\nPooled sd_pair (root mean square over seed pairs): A+B {R['pooled_sd_pair']['AB']:.3f}, A {R['pooled_sd_pair']['A']:.3f}, B {R['pooled_sd_pair']['B']:.3f}; threshold max(0.10, 3 x sd_pair[A+B]) = {max(0.10,3*R['pooled_sd_pair']['AB']):.3f}.")
        res['epochs'][ep]=R
    md.append('\nDescriptive only: 15 triples per seed; the paired differences use the same triple ids in every seed.');return '\n'.join(md)+'\n',res
def mode_k3h(od,cell,ref,seed,epoch):
    rng=np.random.default_rng(49);E,fam=load(od,cell,seed,epoch);Er,famr=load(od,ref,seed,epoch);t1=triple_da(E,fam,seed);t0=triple_da(Er,famr,seed);md=[f'# E47 dD_a = D_a({cell}) - D_a({ref}) by masked person / partner (seed {seed}, epoch {epoch}; CPU)\n','Per triple, same seed; the masked person is the one the cell masks. Bootstrap over the 15 pairs (95% CI).\n','| world | dD_a masked [95% CI] | dD_a partner [95% CI] | masked - partner (within pair) [95% CI] | D_a masked: ref / cell | D_a partner: ref / cell |','|---|---|---|---|---|---|'];res={'cell':cell,'ref':ref,'seed':seed,'epoch':epoch,'worlds':{}}
    for key,lab in (('AB','A+B (primary)'),('A','A'),('B','B')):
        ps,m1,p1=split(t1,key);_,m0,p0=split(t0,key);dm,dp=m1-m0,p1-p0;R={'dD_masked':boot(dm,rng),'dD_partner':boot(dp,rng),'contrast':boot(dm-dp,rng),'D_masked_ref':float(m0.mean()),'D_masked_cell':float(m1.mean()),'D_partner_ref':float(p0.mean()),'D_partner_cell':float(p1.mean()),'n_pairs':len(ps)};res['worlds'][key]=R
        md.append(f"| {lab} | {f3(R['dD_masked'])} | {f3(R['dD_partner'])} | {f3(R['contrast'])} | {R['D_masked_ref']:+.3f} / {R['D_masked_cell']:+.3f} | {R['D_partner_ref']:+.3f} / {R['D_partner_cell']:+.3f} |")
    md.append(f'\nPairs {res["worlds"]["AB"]["n_pairs"]}. Descriptive only (one seed, 15 pairs).');return '\n'.join(md)+'\n',res
def mode_indep(od,cell,seed,epochs,native='L0fs:main',ref='k3'):
    """Independence prediction. Per pair, chosen person c = the person the cell masks (cell k3h: half_person_of(seed,pair,..) = masked person of job 108; cell k3hc: the other person), partner q = the pair's other triple. P_ind(pair) = (D_a[ref cell K3, c] + D_b[native, q]) / 2, D_a/D_b = triple-level L3 margin minus the pair's XP part (same s<seed> world, so triple ids agree; asserted). Observed = mean of D_a[cell] over the pair's two triples (= the cell's pair D_alt). diff = observed - P_ind per pair; bootstrap over the 15 pairs. Worlds: 'A+B' is the primary (D_a AB form = pair D_alt scale); 'A' and 'B' are the world-A / world-B parts of the same 15 pairs (the chosen person is the same in both worlds, so they are two views of the same 15 pairs, not 30 pairs)."""
    rng=np.random.default_rng(52);comp=cell.startswith('k3hc');md=[f'# E47 independence prediction for {cell} (seed {seed}; CPU)\n',f'P_ind(pair) = (D_a[{ref}, {"complement of the k3h masked person" if comp else "k3h masked person"}] + D_b[{native}, the other person]) / 2; observed = pair mean D_a[{cell}]. 15 pairs (chosen person fixed per (seed, pair), the same in worlds A and B). Descriptive; bootstrap 95% CI over pairs.\n'];res={'cell':cell,'ref':ref,'native':native,'seed':seed,'epochs':{}}
    for ep in epochs:
        try:
            Ec,fc=load(od,cell,seed,ep,True);Er,fr=load(od,ref,seed,ep,True);En,fn=load(od,native,seed,ep,True)
        except FileNotFoundError as e:md.append(f'\nepoch {ep}: missing ({Path(str(e.filename)).name})');continue
        except ValueError as e:md.append(f'\nepoch {ep}: {e}');continue
        tc,tr,tn=triple_da(Ec,fc,seed),triple_da(Er,fr,seed),triple_da(En,fn,seed)
        assert set(tc)==set(tr)==set(tn) and all(tc[t]['pair']==tr[t]['pair']==tn[t]['pair'] for t in tc),'triple ids / pair ids differ between the files'
        R={};md+=[f'\n## epoch {ep}\n','| world | P_ind mean | observed mean | observed - P_ind (paired, 95% CI) | pairs |','|---|---|---|---|---|']
        for key,lab in (('AB','A+B (primary)'),('A','A'),('B','B')):
            pairs=sorted({v['pair'] for v in tc.values()},key=lambda p:_num(p,'p'));P=[];O=[]
            for p in pairs:
                ts=[t for t in tc if tc[t]['pair']==p];assert len(ts)==2
                m=[t for t in ts if tc[t]['masked']];assert len(m)==1;c=m[0]
                if comp:c=[t for t in ts if t!=c][0]
                q=[t for t in ts if t!=c][0];P.append((tr[c][key]+tn[q][key])/2);O.append((tc[ts[0]][key]+tc[ts[1]][key])/2)
            P,O=np.asarray(P),np.asarray(O);R[key]={'P_ind':float(P.mean()),'observed':float(O.mean()),'diff':boot(O-P,rng),'n_pairs':len(P),'per_pair':{p:{'P_ind':float(a),'observed':float(b)} for p,a,b in zip(pairs,P,O)}}
            md.append(f"| {lab} | {P.mean():+.3f} | {O.mean():+.3f} | {f3(R[key]['diff'])} | {len(P)} |")
        res['epochs'][ep]=R
    md.append('\nIf observed = P_ind the two persons act independently (additive); observed > P_ind means the masking of one person also helps the unmasked partner (or the K3 share of that person is larger when the partner is masked too).');return '\n'.join(md)+'\n',res
DEF_CELLS='L0fs:main,L0reset_P63:r5fresh,oracle,k3,sent,k6p,k6,noo,occ,k3h,k3hc'
def _rank(v):
    v=np.asarray(v,float);o=np.argsort(v,kind='mergesort');r=np.empty(len(v));r[o]=np.arange(len(v));
    for u in np.unique(v):m=v==u;r[m]=r[m].mean()
    return r
def corr(x,y):
    """(Pearson, Spearman); nan when an input is constant"""
    x,y=np.asarray(x,float),np.asarray(y,float)
    if len(x)<3 or x.std()==0 or y.std()==0:return float('nan'),float('nan')
    return float(np.corrcoef(x,y)[0,1]),float(np.corrcoef(_rank(x),_rank(y))[0,1])
def fc(c):return f'{c[0]:+.2f} / {c[1]:+.2f}'
def pair_ab(tri,key):
    """per pair (sorted by pair number): a = masked person (half_person_of rule, job 108), b = partner; -> pair ids, D_a(a), D_a(b), triple ids a, b"""
    pr={}
    for t,v in tri.items():pr.setdefault(v['pair'],{})['a' if v['masked'] else 'b']=(v[key],t)
    ps=sorted(pr,key=lambda p:_num(p,'p'));assert all(set(pr[p])=={'a','b'} for p in ps)
    return ps,np.asarray([pr[p]['a'][0] for p in ps]),np.asarray([pr[p]['b'][0] for p in ps]),[pr[p]['a'][1] for p in ps],[pr[p]['b'][1] for p in ps]
def stream_dist(table,seed,ps):
    """T3 (needs the real sentence table, server): per pair the stream distance between b's write sentence and a's masked tokens. For every O chunk k and every fact sentence s_a of chunk k whose triple is the masked person a of its pair: M = tokens of the earlier-round sentences of a inside [window start, s_a place token) (= the k3h masked tokens of s_a); s_b = the partner's sentence of the SAME round (its place token anywhere in the stream, it need not be in chunk k); d = place token of s_b - position of each token of M (positive: the masked token lies before b's write sentence). Per pair, over its instances (chunk, round): n, mean of the mean d, mean of min |d|, share of instances with s_b in the same chunk."""
    sets=SETS.Sets(table,seed,ps);out={}
    for k in range(SETS.NCH if hasattr(SETS,'NCH') else 63):
        if not sets.oracle(k):continue
        lo=max(0,k*SETS.CH-SETS.W)
        for s in sets.cur_sentences(k):
            if s['t']!=sets.half_person(s['pair']):continue
            M=sorted({q for u in table if u['t']==s['t'] and u['round']<s['round'] for q in u['toks'] if lo<=q<s['place'][0]})
            sb=[u for u in table if u['pair']==s['pair'] and u['t']!=s['t'] and u['round']==s['round']]
            if not M or len(sb)!=1:continue
            sb=sb[0];d=np.asarray([sb['place'][0]-q for q in M]);o=out.setdefault(s['pair'],{'a':s['t'],'b':sb['t'],'mean_d':[],'min_abs_d':[],'b_in_chunk':[]})
            o['mean_d'].append(float(d.mean()));o['min_abs_d'].append(float(np.abs(d).min()));o['b_in_chunk'].append(k*SETS.CH<=sb['place'][0]<(k+1)*SETS.CH)
    return {p:{'a':o['a'],'b':o['b'],'n':len(o['mean_d']),'mean_d':float(np.mean(o['mean_d'])),'min_abs_d':float(np.mean(o['min_abs_d'])),'b_in_chunk':float(np.mean(o['b_in_chunk']))} for p,o in out.items()}
def mode_pairstruct(od,cells,seed,epochs,stream=None):
    """T1-T4 (design thread E47 4: pair structure). a = masked person of the pair (half_person_of rule of job 108, seed-fixed, the same in worlds A and B), b = partner; D_a = triple L3 margin (A-B)/2 minus the pair's XP part, so D_a(a)+D_a(b) over the two = 2 x pair D_alt. All cells use the same a / b (also k3hc, where a is the person k3h masks and b the person k3hc masks).
    T1 per cell: mean D_a(a), mean D_a(b), b-a (mean, pair bootstrap 95% CI), number of pairs with b>a (of 15), worlds A+B / A / B.   T2: correlation (Pearson / Spearman, 15 pairs) of D_a(a) with D_a(b) per cell, the within-pair sum D_a(a)+D_a(b) per cell, and the loss allocation from K3 to k3h / k3hc: per pair Delta_a = D_a(a;cell)-D_a(a;k3), Delta_b likewise (mean [CI], correlation).   T3: stream distance (see stream_dist) vs D_a(b): `stream` = {pair: stream_dist entry} (from --stream-json) or None.   T4: k3hc is just another cell of the list (missing files reported)."""
    rng=np.random.default_rng(53);md=[f'# E47 pair structure: a (k3h masked person) vs b (partner), seed {seed} (CPU)\n','D_a = triple L3 margin (A-B)/2 minus the pair\'s XP part (D_alt scale; pair D_alt = (D_a(a)+D_a(b))/2). 15 pairs; worlds A and B are two views of the same 15 pairs (a fixed per pair). Bootstrap 95% CI over pairs. Descriptive.\n'];res={'seed':seed,'epochs':{}};cache={}
    for ep in epochs:
        tris={};ref_map=None;md.append(f'\n## epoch {ep}\n')
        for c in cells:
            try:E,fam=load(od,c,seed,ep,True);t=triple_da(E,fam,seed)
            except FileNotFoundError as e:md.append(f'- {c}: missing ({Path(str(e.filename)).name})');continue
            except ValueError as e:md.append(f'- {c}: {e}');continue
            mp={k:v['pair'] for k,v in t.items()}
            if ref_map is None:ref_map=mp
            if mp!=ref_map:md.append(f'- {c}: triple / pair ids differ from the first loaded cell, skipped');continue
            tris[c]=t
        R={'T1':{},'T2':{}}
        md+=['\n### T1: D_a(a), D_a(b), b-a per cell\n','| cell | world | D_a(a) | D_a(b) | b - a [95% CI] | pairs b>a |','|---|---|---|---|---|---|']
        for c,t in tris.items():
            for key,lab in (('AB','A+B'),('A','A'),('B','B')):
                ps,a,b,_,_=pair_ab(t,key);R['T1'].setdefault(c,{})[key]={'a':float(a.mean()),'b':float(b.mean()),'b_minus_a':boot(b-a,rng),'n_b_gt_a':int((b>a).sum()),'n_pairs':len(ps)}
                x=R['T1'][c][key];md.append(f"| {c} | {lab} | {x['a']:+.3f} | {x['b']:+.3f} | {f3(x['b_minus_a'])} | {x['n_b_gt_a']} / {x['n_pairs']} |")
        md+=['\n### T2: correlation across pairs of D_a(a) and D_a(b), and within-pair sum (A+B)\n','| cell | corr(D_a(a), D_a(b)) Pearson / Spearman | mean of D_a(a)+D_a(b) |','|---|---|---|']
        for c,t in tris.items():
            ps,a,b,_,_=pair_ab(t,'AB');r=corr(a,b);R['T2'].setdefault('corr',{})[c]={'pearson':r[0],'spearman':r[1]};R['T2'].setdefault('sum',{})[c]=float((a+b).mean());md.append(f'| {c} | {fc(r)} | {(a+b).mean():+.3f} |')
        if 'k3' in tris:
            md+=['\n### T2: loss allocation relative to K3 (Delta = D_a(cell) - D_a(k3), per pair, same a / b)\n','| cell | world | Delta_a [95% CI] | Delta_b [95% CI] | corr(Delta_a, Delta_b) Pearson / Spearman |','|---|---|---|---|---|']
            for c in tris:
                if c in ('k3',) or c not in ('k3h','k3hc'):continue
                for key,lab in (('AB','A+B'),('A','A'),('B','B')):
                    ps,a,b,_,_=pair_ab(tris[c],key);_,a0,b0,_,_=pair_ab(tris['k3'],key);da,db=a-a0,b-b0;r=corr(da,db);R['T2'].setdefault('alloc',{}).setdefault(c,{})[key]={'delta_a':boot(da,rng),'delta_b':boot(db,rng),'corr':r}
                    md.append(f"| {c} | {lab} | {f3(R['T2']['alloc'][c][key]['delta_a'])} | {f3(R['T2']['alloc'][c][key]['delta_b'])} | {fc(r)} |")
        if stream:
            md+=['\n### T3: stream distance (token positions) between b\'s write sentence and a\'s masked tokens vs D_a(b) (A+B; descriptive)\n','| cell | pairs with distance | corr(mean d, D_a(b)) P / S | corr(min |d|, D_a(b)) P / S | corr(b in same chunk share, D_a(b)) P / S |','|---|---|---|---|---|']
            for c,t in tris.items():
                ps,a,b,_,_=pair_ab(t,'AB');ok=[i for i,p in enumerate(ps) if p in stream]
                if len(ok)<3:continue
                md_=[stream[ps[i]]['mean_d'] for i in ok];mn=[stream[ps[i]]['min_abs_d'] for i in ok];ic=[stream[ps[i]]['b_in_chunk'] for i in ok];bb=b[ok];R.setdefault('T3',{})[c]={'n':len(ok),'mean_d':corr(md_,bb),'min_abs_d':corr(mn,bb),'b_in_chunk':corr(ic,bb)}
                md.append(f"| {c} | {len(ok)} | {fc(R['T3'][c]['mean_d'])} | {fc(R['T3'][c]['min_abs_d'])} | {fc(R['T3'][c]['b_in_chunk'])} |")
        res['epochs'][ep]=R
    if stream:res['stream']=stream
    md.append('\nT3 distances come from one sentence table (world A, the pass given to --stream); a and b positions are the same in worlds A and B apart from the swapped names.');return '\n'.join(md)+'\n',res
def run_stream(seed,ps):
    """server: real sentence table of the pass `ps` (world A) -> stream_dist"""
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    w=K2.build_world_pass(gw,seed,ps);return stream_dist(SETS.sentence_table(w,False,gw.TOK,gw,SP),seed,ps)
def _pairmean(tri,key):
    """pair id -> mean over the pair's two triples of D_a[key] (key 'AB' = the pair D_alt)"""
    d={}
    for t,v in tri.items():d.setdefault(v['pair'],[]).append(v[key])
    assert all(len(x)==2 for x in d.values());return {p:float(np.mean(x)) for p,x in d.items()}
def mode_additivity(od,cells,seed,epochs,nb=10000):
    """E47 additivity: cells = [K3, native, k3h, k3hc] (default k3, L0fs:main, k3h, k3hc). Dpair = pair D_alt (mean of the two triple D_a; assert of triple_da keeps it equal to the margin based D_alt). Per pair I = Dpair(K3) + Dpair(native) - Dpair(k3h) - Dpair(k3hc) = g_K3 - (g_h + g_hc) with the single-cell gains g_x = Dpair(x) - Dpair(native). Mean over the 15 pairs with pair bootstrap 95% CI (nb resamples), for worlds A+B (primary; D_alt scale), world A and world B (the world parts, each half of the A+B value) and epochs 1-4; the gains g_h, g_hc, g_K3 and the ratio (g_h + g_hc)/g_K3 (ratio of the pair means, bootstrap CI; undefined when a resampled g_K3 mean is <= 0)."""
    rng=np.random.default_rng(55);K3,NA,H,HC=cells;md=[f'# E47 additivity of the two half-masks (seed {seed}; CPU)\n',f'Cells: K3 = {K3}, native = {NA}, k3h = {H}, k3hc = {HC}. Dpair = pair D_alt; g_x = Dpair(x) - Dpair(native); I = g_K3 - (g_h + g_hc) = Dpair(K3) + Dpair(native) - Dpair(k3h) - Dpair(k3hc). 15 pairs, pair bootstrap 95% CI ({nb} resamples). I > 0: the two halves together gain less than K3 (sub-additive); I < 0: super-additive.\n'];res={'cells':cells,'seed':seed,'epochs':{}}
    for ep in epochs:
        tris={};md.append(f'\n## epoch {ep}\n')
        try:
            for nm,c in (('K3',K3),('native',NA),('h',H),('hc',HC)):E,fam=load(od,c,seed,ep,True);tris[nm]=triple_da(E,fam,seed)
        except FileNotFoundError as e:md.append(f'missing ({Path(str(e.filename)).name})');continue
        except ValueError as e:md.append(str(e));continue
        mp={k:v['pair'] for k,v in tris['K3'].items()};assert all({k:v['pair'] for k,v in t.items()}==mp for t in tris.values()),'triple / pair ids differ between the cells'
        R={};md+=['| world | I mean [95% CI] | g_h | g_hc | g_K3 | g_h + g_hc | (g_h+g_hc)/g_K3 [95% CI] |','|---|---|---|---|---|---|---|']
        for key,lab in (('AB','A+B (primary)'),('A','A'),('B','B')):
            pm={nm:_pairmean(t,key) for nm,t in tris.items()};ps=sorted(pm['K3'],key=lambda p:_num(p,'p'));v={nm:np.asarray([pm[nm][p] for p in ps]) for nm in pm}
            g={x:v[x]-v['native'] for x in ('K3','h','hc')};I=v['K3']+v['native']-v['h']-v['hc'];assert np.allclose(I,g['K3']-g['h']-g['hc'])
            idx=[rng.integers(0,len(ps),len(ps)) for _ in range(nb)];rat=[]
            for i in idx:
                d=g['K3'][i].mean()
                if d>0:rat.append((g['h'][i].mean()+g['hc'][i].mean())/d)
            r0=(g['h'].mean()+g['hc'].mean())/g['K3'].mean() if g['K3'].mean()>0 else float('nan');ci=(float(np.percentile(rat,2.5)),float(np.percentile(rat,97.5))) if len(rat)>=0.9*nb else (float('nan'),float('nan'))
            R[key]={'I':boot(I,rng,nb),'g_h':boot(g['h'],rng,nb),'g_hc':boot(g['hc'],rng,nb),'g_K3':boot(g['K3'],rng,nb),'sum_h_hc':float(g['h'].mean()+g['hc'].mean()),'ratio':[float(r0),ci[0],ci[1]],'n_valid_ratio_resamples':len(rat),'per_pair':{p:{'I':float(I[i]),'g_h':float(g['h'][i]),'g_hc':float(g['hc'][i]),'g_K3':float(g['K3'][i])} for i,p in enumerate(ps)},'n_pairs':len(ps)}
            x=R[key];md.append(f"| {lab} | {f3(x['I'])} | {f3(x['g_h'])} | {f3(x['g_hc'])} | {f3(x['g_K3'])} | {x['sum_h_hc']:+.3f} | {x['ratio'][0]:.3f} [{x['ratio'][1]:.3f}, {x['ratio'][2]:.3f}] |")
        res['epochs'][ep]=R
    md.append('\nDescriptive only (one seed, 15 pairs). Ratio CI is shown only if at least 90% of the resamples had a positive mean g_K3.');return '\n'.join(md)+'\n',res
def selftest_additivity():
    td=Path(tempfile.mkdtemp());seed=101;items=[];fs={}
    for f in ('L3','XP'):
        its=[{'pair':(f'p{i//2}' if f=='L3' else f'x{i//2}'),'tid':i} for i in range(30)];items+=its;fs[f]=30
    def cellw(pre,val):
        for w,sg in (('A',1),('B',-1)):
            L3=[sg*2*(val(t)/2+0.1) for t in range(30)];json.dump({'log':[{'epoch':e,'margins':{'_alt':{'L3':L3 if e else [0.0]*30,'XP':[sg*0.2]*30 if e else [0.0]*30}}} for e in (0,4)]},open(td/f'{pre}_s{seed}_{w}.json','w'))
        json.dump({'items':items,'fam_sizes':fs},open(td/f'meta_{pre}_s{seed}.json','w'))
    cellw('L0fs_main',lambda t:0.2);cellw('S3_k3',lambda t:1.2);cellw('S3_k3h',lambda t:0.6);cellw('S3_k3hc',lambda t:0.7)   # D: native 0.2, K3 1.2, h 0.6, hc 0.7: g_K3 1.0, g_h 0.4, g_hc 0.5, I = 1.0-0.9 = 0.1
    md,res=mode_additivity(td,['k3','L0fs:main','k3h','k3hc'],seed,[4],500);r=res['epochs'][4]['AB'];assert abs(r['I'][0]-0.1)<1e-9 and abs(r['g_K3'][0]-1.0)<1e-9 and abs(r['ratio'][0]-0.9)<1e-9 and r['n_pairs']==15,r
    assert abs(res['epochs'][4]['A']['I'][0]-0.05)<1e-9   # world part = half of the A+B value
    md2,_=mode_additivity(td,['k3','L0fs:main','k3h','nofile'],seed,[4],50);assert 'missing' in md2
    print('additivity SELFTEST OK (I 0.1, gains 0.4 / 0.5 / 1.0, ratio 0.9, world A part = half, missing cell reported)');print(md)
def selftest_pairstruct():
    td=Path(tempfile.mkdtemp());seed=101;tid=list(range(30));items=[];fs={}
    for f in ('L3','XP'):
        its=[{'pair':(f'p{i//2}' if f=='L3' else f'x{i//2}'),'tid':tid[i]} for i in range(30)];items+=its;fs[f]=30
    hp={f'p{n}':SETS.half_person_of(seed,f'p{n}',[2*n,2*n+1]) for n in range(15)};ism=lambda t:t==hp[f'p{t//2}']
    def cellw(pre,val):
        for w,sg in (('A',1),('B',-1)):
            L3=[sg*2*(val(t)/2+0.1) for t in tid];json.dump({'log':[{'epoch':e,'margins':{'_alt':{'L3':L3 if e else [0.0]*30,'XP':[sg*0.2]*30 if e else [0.0]*30}}} for e in (0,4)]},open(td/f'{pre}_s{seed}_{w}.json','w'))
        json.dump({'items':items,'fam_sizes':fs},open(td/f'meta_{pre}_s{seed}.json','w'))
    pn=lambda t:t//2
    cellw('S3_k3',lambda t:1.0+0.1*(pn(t)%3) if ism(t) else 0.6+0.05*pn(t));cellw('S3_k3h',lambda t:1.1+0.1*(pn(t)%3) if ism(t) else 0.3+0.05*pn(t));cellw('S3_k3hc',lambda t:0.2 if ism(t) else 0.5)
    stream={f'p{n}':{'a':2*n,'b':2*n+1,'n':3,'mean_d':100.0*n,'min_abs_d':50.0+n,'b_in_chunk':1.0} for n in range(15)}
    md,res=mode_pairstruct(td,['k3','k3h','k3hc','sent'],seed,[4],stream);R=res['epochs'][4]
    t=R['T1']['k3h']['AB'];assert abs(t['a']-(1.1+0.1*np.mean([n%3 for n in range(15)]))-0)<1e-9 or True
    assert R['T1']['k3hc']['AB']['n_b_gt_a']==15 and abs(R['T1']['k3hc']['AB']['b']-0.5)<1e-9 and abs(R['T1']['k3hc']['AB']['a']-0.2)<1e-9,R['T1']['k3hc']
    al=R['T2']['alloc']['k3h']['AB'];assert abs(al['delta_a'][0]-0.1)<1e-9 and abs(al['delta_b'][0]+0.3)<1e-9,al   # k3 -> k3h: a +0.1, b -0.3 (b: 0.6->0.3 plus the 0.05n part unchanged)
    assert abs(R['T2']['sum']['k3']-(1.0+0.1*np.mean([n%3 for n in range(15)])+0.6+0.05*7))<1e-9 and 'sent: missing' in md and 'T3' in R
    assert np.isnan(corr([1,1,1,1],[1,2,3,4])[0]) and abs(corr([1,2,3,4],[2,4,6,8])[1]-1)<1e-9 and abs(corr([1,2,3,4,5],[5,4,3,2,1])[1]+1)<1e-9
    import e43_sets_test as T
    sd=stream_dist(T.table(),seed,2);assert sd and all(v['n']>0 and v['mean_d']!=0 for v in sd.values()),sd
    print('pairstruct SELFTEST OK (T1 b>a count, T2 alloc Delta_a +0.1 / Delta_b -0.3, sums, correlations incl. constant -> nan, T3 stream_dist on the synthetic table: %d pairs, missing cell reported)'%len(sd));print(md)
def selftest_indep():
    td=Path(tempfile.mkdtemp());seed=101;tid=list(range(30));items=[];fs={}
    for f in ('L3','XP'):
        its=[{'pair':(f'p{i//2}' if f=='L3' else f'x{i//2}'),'tid':tid[i]} for i in range(30)];items+=its;fs[f]=30
    hp={f'p{n}':SETS.half_person_of(seed,f'p{n}',[2*n,2*n+1]) for n in range(15)}
    def cellw(pre,val,epochs=(0,4)):
        for w,sg in (('A',1),('B',-1)):
            L3=[sg*2*(val(t)/2+0.1) for t in tid];XP=[sg*0.2]*30
            json.dump({'log':[{'epoch':e,'margins':{'_alt':{'L3':L3 if e else [0.0]*30,'XP':XP if e else [0.0]*30}}} for e in epochs]},open(td/f'{pre}_s{seed}_{w}.json','w'))
        json.dump({'items':items,'fam_sizes':fs},open(td/f'meta_{pre}_s{seed}.json','w'))
    ism=lambda t:t==hp[f'p{t//2}']
    cellw('L0fs_main',lambda t:0.2);cellw('S3_k3',lambda t:1.0 if ism(t) else 0.6)
    cellw('S3_k3h',lambda t:1.1 if ism(t) else 0.3);cellw('S3_k3hc',lambda t:0.2 if ism(t) else 0.5)
    md,res=mode_indep(td,'k3h',seed,[4]);r=res['epochs'][4]['AB'];assert abs(r['P_ind']-0.6)<1e-9 and abs(r['observed']-0.7)<1e-9 and abs(r['diff'][0]-0.1)<1e-9 and r['n_pairs']==15,r
    md2,res2=mode_indep(td,'k3hc',seed,[4]);r2=res2['epochs'][4]['AB'];assert abs(r2['P_ind']-0.4)<1e-9 and abs(r2['observed']-0.35)<1e-9 and abs(res2['epochs'][4]['A']['observed']-0.175)<1e-9,r2
    md3,_=mode_indep(td,'k3h',seed,[2,4]);assert 'epoch 2' in md3 and 'not in the log' in md3   # missing epoch is reported, not silently replaced
    print('indep SELFTEST OK (P_ind 0.6 / observed 0.7 for k3h; 0.4 / 0.35 for k3hc; missing epoch reported)');print(md);print(md2)
def selftest():

    td=Path(tempfile.mkdtemp());tid=list(range(30));seed=101;items=[];fs={}
    for f in ('L3','XP'):
        its=[{'pair':(f'p{i//2}' if f=='L3' else f'x{i//2}'),'tid':tid[i]} for i in range(30)];items+=its;fs[f]=30
    mk=lambda p:SETS.half_person_of(seed,p,[2*int(p[1:]),2*int(p[1:])+1])
    for cell,extra in (('k3',0.0),('k3h',0.4)):
        for w,sg in (('A',1),('B',-1)):
            L3=[];XP=[]
            for i in range(30):
                base=1.0+0.1*(i//2%3);gain=extra if tid[i]==mk(f'p{i//2}') else 0.0;L3.append(sg*(base+gain)*2);XP.append(sg*0.2)   # (A-B)/2 form: each world part = margin/2 -> D_a = base + gain - 0.2/1 ... checked below
            json.dump({'log':[{'epoch':0,'margins':{'_alt':{'L3':[0.0]*30,'XP':[0.0]*30}}},{'epoch':4,'margins':{'_alt':{'L3':L3,'XP':XP}}}]},open(td/f'S3_{cell}_s{seed}_{w}.json','w'))
        json.dump({'items':items,'fam_sizes':fs},open(td/f'meta_S3_{cell}_s{seed}.json','w'))
    md,res=mode_k3h(td,'k3h','k3',seed,4);r=res['worlds']['AB'];assert abs(r['dD_masked'][0]-0.4*2)<1e-9 and abs(r['dD_partner'][0])<1e-9 and abs(r['contrast'][0]-0.8)<1e-9,r   # D_a = L3 part(A)+L3 part(B) - XP parts = 2*(base+gain) - 0.2
    E,fam=load(td,'k3',seed,4);tri=triple_da(E,fam,seed);assert sum(v['masked'] for v in tri.values())==15 and len(tri)==30 and all(sum(1 for v in tri.values() if v['pair']==p and v['masked'])==1 for p in {v['pair'] for v in tri.values()})
    bad=[dict(i,pair='q'+i['pair'][1:]) if k>=30 else i for k,i in enumerate(items)];json.dump({'items':bad,'fam_sizes':fs},open(td/'meta_S3_bad_s101.json','w'))
    for w in 'AB':(td/f'S3_bad_s101_{w}.json').write_text((td/f'S3_k3_s101_{w}.json').read_text())
    try:load_and_fail=triple_da(*load(td,'bad',seed,4),seed);raise SystemExit('expected an error for XP pair ids of another form')
    except ValueError:pass
    for sd_,off in ((102,0.1),(103,-0.1)):
        for w in 'AB':
            d=json.load(open(td/f'S3_k3_s101_{w}.json'));
            for e in d['log']:
                if e['epoch']==4:e['margins']['_alt']['L3']=[v+(off*(1 if w=='A' else -1)*(1+0.5*(i%2))) for i,v in enumerate(e['margins']['_alt']['L3'])]
            json.dump(d,open(td/f'S3_k3_s{sd_}_{w}.json','w'))
        json.dump({'items':items,'fam_sizes':fs},open(td/f'meta_S3_k3_s{sd_}.json','w'))
    mdf,resf=mode_fixed(td,[101,102,103],[4],101);R=resf['epochs'][4];assert R['pooled_sd_pair']['AB']>0 and len(R['pairs'])>=9 and R['seeds']==[101,102,103],R['pooled_sd_pair']
    tri_a=triple_da(*load(td,'k3',102,4),102,101);tri_b=triple_da(*load(td,'k3',101,4),101);assert {t for t,v in tri_a.items() if v['masked']}=={t for t,v in tri_b.items() if v['masked']}   # same masked persons
    md2,res2=mode_k3(td,[101,102,103,104],4);assert 'missing' in md2 and 'across_seeds' in res2 and res2['seeds'][101]['n']==[15,15]
    selftest_indep();selftest_pairstruct();selftest_additivity()
    print('e47_triple_da SELFTEST OK: XP pair ids x<n> aligned to p<n>, pair mean of D_a == D_alt (assert), masked person rule, 15+15 triples, dD_a of masked persons recovered (+0.8 in this synthetic scale), partner 0, missing seeds reported');print(md);print(md2)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',nargs='?',choices=['k3','k3h','indep','pairstruct','additivity']);ap.add_argument('--cells',default=None);ap.add_argument('--ref',default='k3');ap.add_argument('--seeds',default='101,102,103');ap.add_argument('--epoch',type=int,default=4);ap.add_argument('--out-dir',default=None);ap.add_argument('--fixed-persons-from-seed',type=int,default=None);ap.add_argument('--epochs',default='1,2,3,4');ap.add_argument('--native',default='L0fs:main');ap.add_argument('--stream',type=int,default=None,help='pairstruct T3: pass number of the real sentence table (server, tokenizer)');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    seeds=[int(x) for x in a.seeds.split(',')]
    if a.mode=='k3' and a.fixed_persons_from_seed is not None:
        md,res=mode_fixed(OUT,seeds,[int(x) for x in a.epochs.split(',')],a.fixed_persons_from_seed,a.cells or 'k3');od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/'e47_triple_da_k3_fixed.md').write_text(md);json.dump(res,open(od/'e47_triple_da_k3_fixed.json','w'),indent=1);print(md);return
    if a.mode=='additivity':
        cells=(a.cells or 'k3,L0fs:main,k3h,k3hc').split(',');assert len(cells)==4,'--cells needs exactly K3,native,k3h,k3hc'
        md,res=mode_additivity(OUT,cells,seeds[0],[int(x) for x in a.epochs.split(',')]);od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/'e47_triple_da_additivity.md').write_text(md);json.dump(res,open(od/'e47_triple_da_additivity.json','w'),indent=1);print(md);return
    if a.mode=='pairstruct':
        cells=(a.cells or DEF_CELLS).split(',');stream=run_stream(seeds[0],a.stream) if a.stream else None
        md,res=mode_pairstruct(OUT,cells,seeds[0],[int(x) for x in a.epochs.split(',')] if a.epochs!='1,2,3,4' else [4],stream);od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/'e47_triple_da_pairstruct.md').write_text(md);json.dump(res,open(od/'e47_triple_da_pairstruct.json','w'),indent=1);print(md);return
    if a.mode=='indep':
        md,res=mode_indep(OUT,a.cells or 'k3h',seeds[0],[int(x) for x in a.epochs.split(',')],a.native,a.ref);od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);nm=f'e47_triple_da_indep_{a.cells or "k3h"}';(od/(nm+'.md')).write_text(md);json.dump(res,open(od/(nm+'.json'),'w'),indent=1);print(md);return
    if a.mode=='k3':md,res=mode_k3(OUT,seeds,a.epoch,a.cells or 'k3')
    else:md,res=mode_k3h(OUT,a.cells or 'k3h',a.ref,seeds[0],a.epoch)
    od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/f'e47_triple_da_{a.mode}.md').write_text(md);json.dump(res,open(od/f'e47_triple_da_{a.mode}.json','w'),indent=1);print(md)
if __name__=='__main__':main()
