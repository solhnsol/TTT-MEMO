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
def triple_da(E,fam,seed):
    """-> dict with per-triple D_a for 'AB' (= (A-B)/2 form), 'A' and 'B' world parts, masked person flag and pair."""
    a={w:np.asarray(E[w]['margins']['_alt']['L3']) for w in 'AB'};x={w:np.asarray(E[w]['margins']['_alt']['XP']) for w in 'AB'}
    part={'A':lambda m:m/2,'B':lambda m:-m/2}
    xp={}
    for k,it in enumerate(fam['XP']):
        for w in 'AB':xp.setdefault((it['pair'],w),[]).append(part[w](x[w][k]))
    xp={k:float(np.mean(v)) for k,v in xp.items()}
    l3={};pairs={}
    for k,it in enumerate(fam['L3']):
        if 'tid' not in it:raise ValueError('items have no triple id (tid)')
        pairs[it['tid']]=it['pair']
        for w in 'AB':l3.setdefault((it['tid'],w),[]).append(part[w](a[w][k]))
    l3={k:float(np.mean(v)) for k,v in l3.items()};tri={}
    bypair={}
    for t,p in pairs.items():bypair.setdefault(p,[]).append(t)
    for t,p in pairs.items():
        da={w:l3[(t,w)]-xp[(p,w)] for w in 'AB'};tri[t]={'pair':p,'A':da['A'],'B':da['B'],'AB':da['A']+da['B'],'masked':t==SETS.half_person_of(seed,p,bypair[p])}
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
def mode_k3h(od,cell,ref,seed,epoch):
    rng=np.random.default_rng(49);E,fam=load(od,cell,seed,epoch);Er,famr=load(od,ref,seed,epoch);t1=triple_da(E,fam,seed);t0=triple_da(Er,famr,seed);md=[f'# E47 dD_a = D_a({cell}) - D_a({ref}) by masked person / partner (seed {seed}, epoch {epoch}; CPU)\n','Per triple, same seed; the masked person is the one the cell masks. Bootstrap over the 15 pairs (95% CI).\n','| world | dD_a masked [95% CI] | dD_a partner [95% CI] | masked - partner (within pair) [95% CI] | D_a masked: ref / cell | D_a partner: ref / cell |','|---|---|---|---|---|---|'];res={'cell':cell,'ref':ref,'seed':seed,'epoch':epoch,'worlds':{}}
    for key,lab in (('AB','A+B (primary)'),('A','A'),('B','B')):
        ps,m1,p1=split(t1,key);_,m0,p0=split(t0,key);dm,dp=m1-m0,p1-p0;R={'dD_masked':boot(dm,rng),'dD_partner':boot(dp,rng),'contrast':boot(dm-dp,rng),'D_masked_ref':float(m0.mean()),'D_masked_cell':float(m1.mean()),'D_partner_ref':float(p0.mean()),'D_partner_cell':float(p1.mean()),'n_pairs':len(ps)};res['worlds'][key]=R
        md.append(f"| {lab} | {f3(R['dD_masked'])} | {f3(R['dD_partner'])} | {f3(R['contrast'])} | {R['D_masked_ref']:+.3f} / {R['D_masked_cell']:+.3f} | {R['D_partner_ref']:+.3f} / {R['D_partner_cell']:+.3f} |")
    md.append(f'\nPairs {res["worlds"]["AB"]["n_pairs"]}. Descriptive only (one seed, 15 pairs).');return '\n'.join(md)+'\n',res
def selftest():
    td=Path(tempfile.mkdtemp());tid=list(range(30));seed=101;items=[];fs={}
    for f in ('L3','XP'):
        its=[{'pair':f'p{i//2}','tid':tid[i]} for i in range(30)];items+=its;fs[f]=30
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
    md2,res2=mode_k3(td,[101,102,103],4);assert 'missing' in md2 and res2['seeds'][101]['n']==[15,15]
    print('e47_triple_da SELFTEST OK: masked person rule, 15+15 triples, dD_a of masked persons recovered (+0.8 in this synthetic scale), partner 0, missing seeds reported');print(md);print(md2)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',nargs='?',choices=['k3','k3h']);ap.add_argument('--cells',default=None);ap.add_argument('--ref',default='k3');ap.add_argument('--seeds',default='101,102,103');ap.add_argument('--epoch',type=int,default=4);ap.add_argument('--out-dir',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    seeds=[int(x) for x in a.seeds.split(',')]
    if a.mode=='k3':md,res=mode_k3(OUT,seeds,a.epoch,a.cells or 'k3')
    else:md,res=mode_k3h(OUT,a.cells or 'k3h',a.ref,seeds[0],a.epoch)
    od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/f'e47_triple_da_{a.mode}.md').write_text(md);json.dump(res,open(od/f'e47_triple_da_{a.mode}.json','w'),indent=1);print(md)
if __name__=='__main__':main()
