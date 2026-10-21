"""E47 CPU analysis (no GPU, no tokenizer): triple-level binding score D_a and the k3h split masked person / partner, from saved S3 result files (OUT/S3_<cell>_s<seed>_{A,B}.json, margins['_alt'], and meta_S3_<cell>_s<seed>.json with the item list).
D_a(triple t) = (L3 binding margin of t) - (XP margin of the pair of t), both as (margin in world A - margin in world B)/2 per item (analyze_e35 convention) at the chosen epoch (4 = after pass 4); the pair's XP = mean over the pair's XP items. The mean of D_a over the two triples of a pair is the pair's D_alt of the other analyses.
Masked person of a pair (the k3h cell masks only his K3 share) = s3_sets.half_person_of(seed, pair, the triple ids of the pair), the same rule as job 108 (fixed per seed and pair, the same in worlds A and B); partner = the other triple of the pair. 15 pairs = 15 masked persons + 15 partners per seed.
Modes:
  k3    --cells k3 --seeds 101,102,103   per seed: mean D_a of the masked persons and of the partners (15 triples each, pair bootstrap 95% CI), masked - partner (paired within pair), the number of triples; per world (A: item margin of world A / 2, B: minus the item margin of world B / 2, each minus the pair's XP part) mean and sd over the 15 triples; across seeds: mean and sd (n = number of seeds) of the 15-triple means. Seeds without result files are listed as missing.
  k3h   --cells k3h --ref k3 --seeds 101   dD_a = D_a(k3h) - D_a(K3) per triple (same seed): mean over the masked persons (15 triples, bootstrap 95% CI), over the partners, and the within-pair contrast (masked - partner), plus the same per world.
Output: <out-dir>/e47_triple_da_<mode>.{md,json}. --selftest: synthetic."""
import os,sys,json,argparse,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
ROOT=HERE.parents[1];OUT=Path(os.environ.get('EXP07_OUT',ROOT/'experiments/exp07_write_ladder'));NB=5000
def load(od,cell,seed,epoch):
    meta=json.load(open(od/f'meta_S3_{cell}_s{seed}.json'));E={}
    for w in 'AB':
        ent={e['epoch']:e for e in json.load(open(od/f'S3_{cell}_s{seed}_{w}.json'))['log'] if isinstance(e.get('margins'),dict) and 'epoch' in e};E[w]=ent[epoch if epoch in ent else max(ent)]
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
    print('e47_triple_da SELFTEST OK: XP pair ids x<n> aligned to p<n>, pair mean of D_a == D_alt (assert), masked person rule, 15+15 triples, dD_a of masked persons recovered (+0.8 in this synthetic scale), partner 0, missing seeds reported');print(md);print(md2)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',nargs='?',choices=['k3','k3h']);ap.add_argument('--cells',default=None);ap.add_argument('--ref',default='k3');ap.add_argument('--seeds',default='101,102,103');ap.add_argument('--epoch',type=int,default=4);ap.add_argument('--out-dir',default=None);ap.add_argument('--fixed-persons-from-seed',type=int,default=None);ap.add_argument('--epochs',default='1,2,3,4');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    seeds=[int(x) for x in a.seeds.split(',')]
    if a.mode=='k3' and a.fixed_persons_from_seed is not None:
        md,res=mode_fixed(OUT,seeds,[int(x) for x in a.epochs.split(',')],a.fixed_persons_from_seed,a.cells or 'k3');od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/'e47_triple_da_k3_fixed.md').write_text(md);json.dump(res,open(od/'e47_triple_da_k3_fixed.json','w'),indent=1);print(md);return
    if a.mode=='k3':md,res=mode_k3(OUT,seeds,a.epoch,a.cells or 'k3')
    else:md,res=mode_k3h(OUT,a.cells or 'k3h',a.ref,seeds[0],a.epoch)
    od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/f'e47_triple_da_{a.mode}.md').write_text(md);json.dump(res,open(od/f'e47_triple_da_{a.mode}.json','w'),indent=1);print(md)
if __name__=='__main__':main()
