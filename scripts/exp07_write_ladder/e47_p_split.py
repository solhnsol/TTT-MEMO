"""E47 CPU analysis: does the D gain of the cell kp (K3 + P) sit on the pairs whose P tokens were masked? Per pair (15 pairs, worlds A and B averaged like analyze_e35.D_alt) the binding score D_alt = M(L3) - M(XP) at the last epoch, read from the saved S3 result files of a cell (S3_<cell>_s<seed>_{A,B}.json, margins['_alt']), optionally minus a reference cell (default k3, same seed): dD = D(cell) - D(ref).
Pairs are grouped by c(pair) = number of P tokens masked on behalf of the pair's facts, summed over passes 1..P, mean over worlds A and B (Sets.p_by_fact per chunk; a triple is counted when its sentence in the chunk has an earlier same-triple exposure in the window and the other triple of its pair is not itself a fact triple of the chunk). Groups: (a) c == 0 vs c > 0 (only if both are non-empty), (b) lower vs upper half by c (median split), (c) Spearman(c, D) and Spearman(c, dD). Mean, n, pair bootstrap 95% CI of the group means and of the group difference (5000, seed fixed). With 15 pairs the power is low; every number is descriptive.
The P-masked counts need the real tokenizer (server): build them once with  e47_p_split.py counts --seed 101 --passes 4 --out counts.json  (gen_world, suppress and the tokenizer, like e47_counts.py), then analyse with  e47_p_split.py analyse --cell kp --ref k3 --seed 101 --counts counts.json [--epoch 4] [--out-dir D]. D is also split per triple where the item list has the triple id ('tid'): the L3 binding margin (A - B)/2 per triple (not minus XP), same grouping by the per-triple count; reported as a second table.   --selftest: synthetic."""
import os,sys,json,argparse,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
ROOT=HERE.parents[1];OUT=Path(os.environ.get('EXP07_OUT',ROOT/'experiments/exp07_write_ladder'));NB=5000
def build_counts(a):
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    res={'seed':a.seed,'passes':a.passes,'per_triple':{},'pair_of':{}}
    for wd in 'AB':
        for ps in range(1,a.passes+1):
            w=K2.build_world_pass(gw,a.seed,ps);table=SETS.sentence_table(w,wd=='B',gw.TOK,gw,SP);sets=SETS.Sets(table,a.seed,ps)
            for u in table:res['pair_of'][u['t']]=u['pair']
            for k in range(63):
                for t,v in sets.p_by_fact(k).items():res['per_triple'].setdefault(t,{}).setdefault(wd,[0]*a.passes)[ps-1]+=v
    return res
def pair_triple_counts(counts):
    """-> (triple -> mean over worlds of the sum over passes, pair -> sum over its triples)"""
    tc={t:float(np.mean([sum(v) for v in d.values()])) for t,d in counts['per_triple'].items()}
    for t in counts['pair_of']:tc.setdefault(t,0.0)
    pc={}
    for t,c in tc.items():pc[counts['pair_of'][t]]=pc.get(counts['pair_of'][t],0.0)+c
    return tc,pc
def load_cell(od,cell,seed,epoch):
    fa,fb=od/f'S3_{cell}_s{seed}_A.json',od/f'S3_{cell}_s{seed}_B.json';meta=json.load(open(od/f'meta_S3_{cell}_s{seed}.json'))
    def ent(f):
        E={e['epoch']:e for e in json.load(open(f))['log'] if isinstance(e.get('margins'),dict) and 'epoch' in e};return E[epoch if epoch in E else max(E)]
    return ent(fa),ent(fb),meta
def fam_slices(meta):
    items=meta['items'];out={};i=0
    for f,n in meta['fam_sizes'].items():out[f]=items[i:i+n];i+=n
    return out
def d_pair(eA,eB,meta):
    fam=fam_slices(meta)
    def M(f):
        a=np.asarray(eA['margins']['_alt'][f]);b=np.asarray(eB['margins']['_alt'][f]);d={}
        for it,x in zip(fam[f],(a-b)/2):d.setdefault(it['pair'],[]).append(x)
        return {k:float(np.mean(v)) for k,v in d.items()}
    l3,xp=M('L3'),M('XP');return {p:l3[p]-xp[p] for p in l3}
def l3_triple(eA,eB,meta):
    fam=fam_slices(meta);a=np.asarray(eA['margins']['_alt']['L3']);b=np.asarray(eB['margins']['_alt']['L3']);d={}
    for it,x in zip(fam['L3'],(a-b)/2):
        if 'tid' not in it:return None
        d.setdefault(it['tid'],[]).append(float(x))
    return {k:float(np.mean(v)) for k,v in d.items()}
def spearman(x,y):
    from itertools import groupby
    def rk(v):
        o=np.argsort(v,kind='stable');r=np.empty(len(v));i=0;v=np.asarray(v)[o]
        while i<len(v):
            j=i
            while j+1<len(v) and v[j+1]==v[i]:j+=1
            r[o[i:j+1]]=(i+j)/2;i=j+1
        return r
    x,y=rk(x),rk(y)
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else float('nan')
def boot(v,rng,n=NB):
    v=np.asarray(v,float);b=np.asarray([v[rng.integers(0,len(v),len(v))].mean() for _ in range(n)]);return float(v.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def boot_diff(a,b,rng,n=NB):
    a=np.asarray(a,float);b=np.asarray(b,float);d=np.asarray([a[rng.integers(0,len(a),len(a))].mean()-b[rng.integers(0,len(b),len(b))].mean() for _ in range(n)]);return float(a.mean()-b.mean()),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))
def f3(t):return f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
def groups(keys,c,vals,rng,label):
    """vals: dict key -> value; c: dict key -> count. Returns md lines for the grouping (a) zero vs positive and (b) median split, and the Spearman."""
    ks=sorted(keys);cv=np.asarray([c[k] for k in ks]);v=np.asarray([vals[k] for k in ks]);L=[]
    z=cv==0
    if z.any() and (~z).any():L.append(f'| {label}: c = 0 (n={int(z.sum())}) / c > 0 (n={int((~z).sum())}) | {f3(boot(v[z],rng))} | {f3(boot(v[~z],rng))} | {f3(boot_diff(v[~z],v[z],rng))} |')
    else:L.append(f'| {label}: c = 0 vs c > 0 | not possible: all {len(ks)} keys have c {"= 0" if z.all() else "> 0"} | | |')
    med=float(np.median(cv));lo=cv<=med if (cv>med).any() else cv<med;hi=~lo
    if lo.any() and hi.any():L.append(f'| {label}: lower half c <= {med:.0f} (n={int(lo.sum())}) / upper (n={int(hi.sum())}) | {f3(boot(v[lo],rng))} | {f3(boot(v[hi],rng))} | {f3(boot_diff(v[hi],v[lo],rng))} |')
    L.append(f'| {label}: Spearman(c, value) | {spearman(cv,v):+.2f} | | |');return L
def analyse(od,cell,ref,seed,counts,epoch=4):
    rng=np.random.default_rng(47);tc,pc=pair_triple_counts(counts);eA,eB,meta=load_cell(od,cell,seed,epoch);D=d_pair(eA,eB,meta);keys=[p for p in D if p in pc];md=[f'# E47 {cell} by P-masked pairs (seed {seed}, epoch {epoch}; CPU)\n',f'c(pair) = P tokens masked on behalf of the pair summed over passes (mean of worlds A/B): min {min(pc[p] for p in keys):.0f}, median {np.median([pc[p] for p in keys]):.0f}, max {max(pc[p] for p in keys):.0f}; pairs {len(keys)}.\n','| grouping | low group mean D_alt [95% CI] | high group mean [95% CI] | high - low [95% CI] |','|---|---|---|---|']
    res={'cell':cell,'ref':ref,'seed':seed,'epoch':epoch,'pairs':{p:{'c':pc[p],'D':D[p]} for p in keys},'D_mean':boot([D[p] for p in keys],rng)}
    md+=groups(keys,pc,D,rng,f'D({cell})')
    if ref:
        rA,rB,rm=load_cell(od,ref,seed,epoch);Dr=d_pair(rA,rB,rm);dD={p:D[p]-Dr[p] for p in keys};res['D_ref_mean']=boot([Dr[p] for p in keys],rng);res['dD_mean']=boot(list(dD.values()),rng)
        for p in keys:res['pairs'][p]['dD']=dD[p]
        md+=groups(keys,pc,dD,rng,f'dD = D({cell}) - D({ref})');md.append(f'\nmean D({cell}) {f3(res["D_mean"])}, mean D({ref}) {f3(res["D_ref_mean"])}, mean dD {f3(res["dD_mean"])}')
    t3=l3_triple(eA,eB,meta)
    if t3 is not None:
        tk=[t for t in t3 if t in tc];md+=['','| triple level (L3 binding margin (A-B)/2, not minus XP) | low | high | high - low |','|---|---|---|---|']+groups(tk,tc,t3,rng,f'M_L3({cell})')
        if ref:
            r3=l3_triple(rA,rB,rm);md+=groups(tk,tc,{t:t3[t]-r3[t] for t in tk},rng,f'M_L3({cell}) - M_L3({ref})')
    else:md.append('\n(triple level skipped: the items carry no triple id)')
    md.append('\nDescriptive only: 15 pairs; the grouping is by an observed count, not randomised.');return '\n'.join(md)+'\n',res
def selftest():
    rng=np.random.default_rng(1);pairs=[f'p{i}' for i in range(15)];tid=[f't{i}' for i in range(30)];td=Path(tempfile.mkdtemp())
    items=[];fs={}
    for f in ('L3','XP'):
        its=[]
        for i in range(30):its.append({'pair':f'p{i//2}','tid':tid[i]})
        items+=its;fs[f]=len(its)
    for cell,boost in (('kp',1.0),('k3',0.0)):
        for w,sgn in (('A',1),('B',-1)):
            al={'L3':[],'XP':[]}
            for i in range(30):
                gain=boost*(0.5 if i//2 >= 8 else 0.0);al['L3'].append(sgn*(1.0+gain)+0.0);al['XP'].append(0.0)
            json.dump({'log':[{'epoch':0,'margins':{'_alt':{k:[0.0]*30 for k in al}}},{'epoch':4,'margins':{'_alt':al}}]},open(td/f'S3_{cell}_s101_{w}.json','w'))
        json.dump({'items':items,'fam_sizes':fs},open(td/f'meta_S3_{cell}_s101.json','w'))
    counts={'seed':101,'passes':4,'pair_of':{tid[i]:f'p{i//2}' for i in range(30)},'per_triple':{tid[i]:{'A':[0,0,5,5],'B':[0,0,5,5]} for i in range(30) if i//2>=8}}
    md,res=analyse(td,'kp','k3',101,counts);assert abs(res['dD_mean'][0]-0.5*7/15)<1e-9,res['dD_mean']
    hi=[v['dD'] for v in res['pairs'].values() if v['c']>0];lo=[v['dD'] for v in res['pairs'].values() if v['c']==0];assert set(np.round(hi,6))=={0.5} and set(np.round(lo,6))=={0.0} and len(hi)==7 and len(lo)==8
    assert 'c = 0 (n=8) / c > 0 (n=7)' in md and 'triple level' in md and abs(spearman([0,0,1,2],[1,1,5,9])-1)<1e-9
    print('e47_p_split SELFTEST OK: synthetic pairs, the dD of the P-masked pairs (+0.5) vs the others (0) is recovered; group sizes, Spearman and the triple-level table run');print(md)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',nargs='?',choices=['counts','analyse']);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--passes',type=int,default=4);ap.add_argument('--out',default='/tmp/e47_p_counts.json');ap.add_argument('--cell',default='kp');ap.add_argument('--ref',default='k3');ap.add_argument('--counts',default='/tmp/e47_p_counts.json');ap.add_argument('--epoch',type=int,default=4);ap.add_argument('--out-dir',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    if a.mode=='counts':res=build_counts(a);json.dump(res,open(a.out,'w'));print('wrote',a.out,'triples',len(res['per_triple']));return
    md,res=analyse(OUT,a.cell,a.ref if a.ref!='none' else None,a.seed,json.load(open(a.counts)),a.epoch);od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/f'e47_p_split_{a.cell}_s{a.seed}.md').write_text(md);json.dump(res,open(od/f'e47_p_split_{a.cell}_s{a.seed}.json','w'),indent=1);print(md)
if __name__=='__main__':main()
