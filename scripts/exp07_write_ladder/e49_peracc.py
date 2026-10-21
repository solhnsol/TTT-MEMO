"""E49 (CPU only, no model): per-pass 2-candidate accuracy from the margins stored in the result jsons (existing results, no re-scoring needed).
(i) item accuracy: an item (L3, 30 items) is right in world A when margin > 0 and in world B when margin < 0 (the binding is flipped in B); accuracy = [(m_A > 0) + (m_B < 0)]/2 per item (nat_analyze formula), a pair's value = mean of its 2 items.
(ii) pair accuracy: share of the 15 pairs with D > 0, D(pair) = mean over its L3 items of (m_A - m_B)/2 (the D_alt L3 value; the sign equals that of the summed D); also with the XP-control subtracted (D L3 - XP, aligned by pair number as in analyze_e35.D_alt).
Reference column: the same item accuracy on the XP (null) items, which carry no binding (expected about 0.5 only if the readout has no bias; it shows the readout's bias level).
CIs: bootstrap over the 15 pairs (4000 draws). Readout 'alt' (SQuAD tail, headline) and 'R1' (neutral tail). Cells: S3 tags ('k3all', 'k3h', 'sent', 'oracle_p4', ...), 'LEVEL:TAG' ('L0fs:main', 'L0fs:fskeep'), optional '@regold'. Every logged pass (epoch) of every cell is listed; missing files are reported, not skipped silently.
Usage: e49_peracc.py --cells k3all,k3h,k3hc,sent,kx_s,kx_o,k6p,oracle_p4,oracle_p8,L0fs:main,L0fs:fskeep --seed 101 --out-dir /tmp/e49_peracc"""
import os,sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
OUT=Path(os.environ.get('EXP07_OUT',Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder'))
def _num(p):return int(str(p).lstrip('px'))
def load_all(od,cell,seed):
    """-> (meta fam, {epoch: {'A': entry, 'B': entry}}) for every epoch logged in BOTH worlds with margins."""
    sfx='';c=cell
    if c.endswith('@regold'):c=c[:-7];sfx='.regold'
    pre=c.replace(':','_',1) if ':' in c else f'S3_{c}';meta=json.load(open(od/f'meta_{pre}_s{seed}.json'));L={}
    for w in 'AB':L[w]={e['epoch']:e for e in json.load(open(od/f'{pre}_s{seed}_{w}{sfx}.json'))['log'] if isinstance(e.get('margins'),dict) and 'epoch' in e and (('L3' in e['margins']) or ('_alt' in e['margins']))}
    items=meta['items'];fam={};i=0
    for f,n in meta['fam_sizes'].items():fam[f]=items[i:i+n];i+=n
    return fam,{ep:{w:L[w][ep] for w in 'AB'} for ep in sorted(set(L['A'])&set(L['B']))}
def boot(v,rng,n=4000):
    v=np.asarray(v,float);b=v[rng.integers(0,len(v),(n,len(v)))].mean(1);return float(v.mean()),*np.percentile(b,[2.5,97.5]).tolist()
def acc_rows(mA,mB,fam_items):
    """item accuracy per pair + D per pair for one family; mA, mB arrays aligned to fam_items (dicts with 'pair')."""
    mA=np.asarray(mA,float);mB=np.asarray(mB,float);acc=((mA>0)*1.0+(mB<0)*1.0)/2;d=(mA-mB)/2;P={}
    for it,a,x in zip(fam_items,acc,d):P.setdefault(_num(it['pair']),[]).append((a,x))
    ks=sorted(P);return ks,np.asarray([np.mean([a for a,_ in P[k]]) for k in ks]),np.asarray([np.mean([x for _,x in P[k]]) for k in ks]),acc
def one(E,fam,ro,rng):
    g=(lambda w,f:E[w]['margins']['_alt'][f]) if ro=='alt' else (lambda w,f:E[w]['margins'][f])
    ks,pa,pd_,ia=acc_rows(g('A','L3'),g('B','L3'),fam['L3']);kx,xa,xd,_=acc_rows(g('A','XP'),g('B','XP'),fam['XP']);assert len(ks)==len(kx)
    return {'n_pairs':len(ks),'item_acc':boot(pa,rng),'pair_acc':boot((pd_>0)*1.0,rng),'pair_acc_minus_xp':boot(((pd_-xd)>0)*1.0,rng),'D_L3':boot(pd_,rng),'D_L3_minus_XP':boot(pd_-xd,rng),'xp_item_acc':boot(xa,rng)}
def analyse(od,cells,seed):
    rng=np.random.default_rng(49);md=['# E49 per-pass 2-candidate accuracy from the stored margins (CPU)\n','item acc = [(margin_A > 0) + (margin_B < 0)]/2 over the 30 L3 items (pair mean, 15 pairs); pair acc = share of pairs with D > 0 (D = mean (A-B)/2 of the pair\'s L3 items), "- XP" = with the XP control subtracted; XP item acc = the same accuracy on the null (XP) items. mean [95% CI over pairs]; chance 0.5.\n'];res={}
    for ro in ('alt','R1'):
        md.append(f'\n## readout {ro}\n');md.append('| cell | pass | item acc | pair acc | pair acc (D - XP) | mean D L3 | XP item acc |');md.append('|---|---|---|---|---|---|---|')
        for c in cells:
            try:fam,E=load_all(od,c,seed)
            except (FileNotFoundError,KeyError) as e:md.append(f'| {c} | missing: {Path(str(getattr(e,"filename",e))).name} | | | | | |');continue
            for ep,en in E.items():
                try:r=one(en,fam,ro,rng)
                except KeyError as e:md.append(f'| {c} | {ep} | no {e.args[0]} margins | | | | |');continue
                res.setdefault(ro,{}).setdefault(c,{})[ep]=r;f=lambda t:'%.3f [%.3f, %.3f]'%t
                md.append(f"| {c} | {ep} | {f(r['item_acc'])} | {f(r['pair_acc'])} | {f(r['pair_acc_minus_xp'])} | {r['D_L3'][0]:+.3f} | {f(r['xp_item_acc'])} |")
    return '\n'.join(md)+'\n',res
def selftest():
    import tempfile;td=Path(tempfile.mkdtemp());seed=101;items=[]
    for f in ('L3','XP'):items+=[{'pair':(f'p{i//2}' if f=='L3' else f'x{i//2}'),'tid':i} for i in range(30)]
    json.dump({'items':items,'fam_sizes':{'L3':30,'XP':30}},open(td/f'meta_S3_t_s{seed}.json','w'))
    def entry(ep,sA,sB,xA=0.0):   # L3 margins: world A +sA for the first pair-item and +sA/2... deterministic pattern
        L3A=[sA*(1 if i%2==0 else -0.5) for i in range(30)];L3B=[-sB*(1 if i%2==0 else -0.5) for i in range(30)]
        return {A:{'epoch':ep,'margins':{'L3':L3,'XP':[xA]*30,'_alt':{'L3':L3,'XP':[xA]*30}}} for A,L3 in (('A',L3A),('B',L3B))}
    logs={'A':[],'B':[]}
    for ep,(sA,sB) in enumerate([(0.0,0.0),(1.0,1.0),(-1.0,-1.0)]):
        e=entry(ep,sA,sB);[logs[w].append(e[w]) for w in 'AB']
    for w in 'AB':json.dump({'log':logs[w]},open(td/f'S3_t_s{seed}_{w}.json','w'))
    fam,E=load_all(td,'t',seed);assert sorted(E)==[0,1,2]
    rng=np.random.default_rng(0);r1=one(E[1],fam,'alt',rng)
    # pass 1: even items: m_A=+1>0 right, m_B=-1<0 right -> 1.0 ; odd items: m_A=-0.5 wrong, m_B=+0.5 wrong -> 0 ; pair mean 0.5, D per pair = ((1-(-1))/2 + (-0.5-0.5)/2)/2 = (1-0.5)/2 = 0.25 > 0
    assert abs(r1['item_acc'][0]-0.5)<1e-12 and abs(r1['pair_acc'][0]-1.0)<1e-12 and abs(r1['D_L3'][0]-0.25)<1e-12 and r1['n_pairs']==15,r1
    r2=one(E[2],fam,'alt',rng);assert abs(r2['item_acc'][0]-0.5)<1e-12 and abs(r2['pair_acc'][0]-0.0)<1e-12 and abs(r2['D_L3'][0]+0.25)<1e-12   # signs flipped: pair accuracy 0
    r0=one(E[0],fam,'alt',rng);assert abs(r0['item_acc'][0]-0.0)<1e-12 and r0['pair_acc'][0]==0.0   # margin exactly 0 is not a correct pick (strict inequality)
    md,res=analyse(td,['t','nofile'],seed);assert 'nofile' in md and res['alt']['t'][1]['pair_acc'][0]==1.0;print(md);print('e49_peracc selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cells',default='k3all,k3h,k3hc,sent,kx_s,kx_o,k6p,oracle_p4,oracle_p8,L0fs:main,L0fs:fskeep');ap.add_argument('--seed',type=int,default=101);ap.add_argument('--out-dir',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    md,res=analyse(OUT,a.cells.split(','),a.seed);od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/'e49_peracc.md').write_text(md);json.dump(res,open(od/'e49_peracc.json','w'),indent=1);print(md)
