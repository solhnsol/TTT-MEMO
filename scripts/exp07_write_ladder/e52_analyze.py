"""E52 Level-2 analysis (CPU). Inputs: the two native-fs runs on the two layouts of the Level-2 stream, saved with separate tags (default e52a = L2a, e52b = L2b; files {cell}_{tag}_s{stream}_{A,B}.json and meta_nat_{tag}_s{stream}.json as nat_ladder.py writes them), and the world files (conditions per pair).
Per person D_i (nat_analyze.Dperson: [mA(L3_i)-mB(L3_i)] - [mA(XP_i)-mB(XP_i)], un-halved, state k = pass k) in each layout. Within-person contrast, as pre-specified:  Delta_i = D_i(C1) - D_i(C2)  (the layout in which the person is C1 minus the layout in which the same person is C2; flip pairs only).
Shadowing signature = D(C1) < D(C2), i.e. S_i = -Delta_i = D_i(C2) - D_i(C1) > 0. Unit of analysis = the PAIR (mean of its two persons' Delta), 95% CI by bootstrap over the 24 flip pairs (5000 draws, seed 0), plus the person-level one-sided t as a secondary line.
Pre-result rule (E52 section 4.3, as written by design): the signature is signed when the CI lower bound of S (= -Delta) is > 0.10; the report prints the number and the verdict, nothing else is inferred.
Also printed: C0 persons (1 exposure in both layouts, no shadowing source): D in each layout, the layout difference (replicate noise of the same persons), n; the gold answer token count (place class); no name-only units exist in this stream (persons are fact units only, so there is nothing to count for name-only); both readouts (R1-alt primary, R1) and every pass.
Usage: e52_analyze.py --dir OUT --nat-dir NATDIR [--stream 9201] [--tags e52a,e52b] [--cell L0fs] | --selftest"""
import sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_analyze as NA
NB=5000;THR=0.10
def boot_pairs(pair_vals,rng):
    P=list(pair_vals);v=np.array([pair_vals[p] for p in P]);pt=float(v.mean());bs=[float(v[rng.integers(0,len(v),len(v))].mean()) for _ in range(NB)];lo,hi=np.percentile(bs,[2.5,97.5]);return pt,float(lo),float(hi)
def one_sided_t(x):
    x=np.asarray(x,float);n=len(x)
    if n<2 or x.std(ddof=1)==0:return float('nan'),float('nan')
    t=x.mean()/(x.std(ddof=1)/np.sqrt(n))
    try:
        from scipy import stats;return float(t),float(1-stats.t.cdf(t,n-1))
    except Exception:return float(t),float('nan')
def load_layout(d,stream,tag,cell):
    A=NA.load(d,cell,tag,stream,'A');B=NA.load(d,cell,tag,stream,'B');assert A and B,f'missing {cell}_{tag}_s{stream}_[AB].json in {d}';meta=json.load(open(Path(d)/f'meta_nat_{tag}_s{stream}.json'));return meta,NA.states(A['log']),NA.states(B['log'])
def person_D(meta,SA,SB,k,alt):return NA.Dperson(meta,SA[k]['margins'],SB[k]['margins'],alt)
def analyze(d,nat_dir,stream=9201,tags=('e52a','e52b'),cell='L0fs'):
    import nat_filler as NF,nat_world as NW
    wa=NW.load_world(NF.world_path(nat_dir,stream,'L2a'));ca=wa['insert']['cond'];gold_tok=wa['insert'].get('name_class_tokens',{}).get('place')
    (ma,Aa,Ba),(mb,Ab,Bb)=load_layout(d,stream,tags[0],cell),load_layout(d,stream,tags[1],cell);assert len(Aa)==len(Ab)
    rng=np.random.default_rng(0);L=[f'# E52 Level-2 analysis (stream {stream}; cell {cell}; layouts {tags[0]}=L2a, {tags[1]}=L2b)','',f'gold answer (place) token count: {gold_tok}. Name-only units: none in this stream (persons are fact units only). Unit of analysis: pair (24 flip pairs). Rule: S = D(C2) - D(C1); signed when CI lower bound of S > {THR}.','']
    rep={}
    for alt,nm in ((True,'R1-alt (SQuAD tail, primary)'),(False,'R1 (pg1342 tail)')):
        L+=[f'## {nm}','','| state (pass) | Delta = D(C1)-D(C2), pair mean [95% CI over 24 pairs] | S = -Delta lower bound > 0.10 | person-level one-sided t (n persons) | C0: D in L2a / L2b (n) | C0 layout diff mean (sd) |','|---|---|---|---|---|---|']
        for k in range(1,len(Aa)):
            Da=person_D(ma,Aa,Ba,k,alt);Db=person_D(mb,Ab,Bb,k,alt);delta={};c0a=[];c0b=[];pairs={}
            for t in Da:
                p=t//2;c=ca[str(p)];pairs[t]=p
                if c=='C0':c0a.append(Da[t]);c0b.append(Db[t])
                else:delta[t]=(Da[t]-Db[t]) if c=='C1' else (Db[t]-Da[t])      # C1 layout minus C2 layout
            pv={};
            for t,v in delta.items():pv.setdefault(pairs[t],[]).append(v)
            pm={p:float(np.mean(v)) for p,v in pv.items()};pt,lo,hi=boot_pairs(pm,rng);S=(-hi,-lo);signed=S[0]>THR;t,pval=one_sided_t([-v for v in delta.values()])
            dd=np.array(c0a)-np.array(c0b)
            L.append(f'| {k} | {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] | S lower {S[0]:+.3f}: '+('SIGNED' if signed else 'not signed')+f' | t={t:.2f}, p={pval:.3f} (n={len(delta)}) | {np.mean(c0a):+.3f} / {np.mean(c0b):+.3f} (n={len(c0a)}) | {dd.mean():+.3f} ({dd.std(ddof=1) if len(dd)>1 else float("nan"):.3f}) |')
            rep[(nm,k)]=dict(delta=(pt,lo,hi),signed=bool(signed),n_pairs=len(pm),c0_n=len(c0a),c0_mean=(float(np.mean(c0a)),float(np.mean(c0b))),c0_diff_sd=float(dd.std(ddof=1)) if len(dd)>1 else None)
        L.append('')
    L.append('C0 persons (1 exposure in both layouts) are a descriptive baseline and a replicate-noise estimate (their exposures are identical; the layouts differ only in the other persons and the filler paragraphs), not a within-person C1/C2 contrast.')
    return rep,'\n'.join(L)+'\n'
def selftest():
    import tempfile,nat_world as NW
    d=Path(tempfile.mkdtemp());rng=np.random.default_rng(3);npairs=30;ca={str(p):('C0' if p>=24 else ('C1' if p%2 else 'C2')) for p in range(npairs)};nt=2*npairs
    fam={'L3':nt,'XP':nt};items=[{'level':'L3','tid':i,'pair':f'p{i//2}'} for i in range(nt)]+[{'level':'XP','tid':i,'pair':f'p{i//2}'} for i in range(nt)]
    def mk(Dtrue):
        # margins such that D_i = (L3a-L3b)-(XPa-XPb) = Dtrue_i; XP = 0; world A L3 = +D/2, world B L3 = -D/2
        out={}
        for w,s in (('A',0.5),('B',-0.5)):
            log=[]
            for k in range(3):
                g=Dtrue*(k/2.0) if k else np.zeros(nt);m={'L3':(s*g).tolist(),'XP':[0.0]*nt,'_alt':{'L3':(s*g).tolist(),'XP':[0.0]*nt}};log.append({'epoch':k,'margins':m})
            out[w]={'log':log}
        return out
    wd=Path(d)/'worlds';wd.mkdir()
    for lay in ('a','b'):
        Dt=np.zeros(nt)
        for i in range(nt):
            p=i//2;c=ca[str(p)];c=c if lay=='a' else {'C1':'C2','C2':'C1','C0':'C0'}[c];Dt[i]={'C1':0.10,'C2':0.40,'C0':0.30}[c]+rng.normal(0,0.02)
        o=mk(Dt);json.dump({'items':items,'fam_sizes':fam},open(d/f'meta_nat_e52{lay}_s9201.json','w'))
        for w in 'AB':json.dump(o[w],open(d/f'L0fs_e52{lay}_s9201_{w}.json','w'))
    for lay,cd in (('L2a',ca),('L2b',{p:{'C1':'C2','C2':'C1','C0':'C0'}[c] for p,c in ca.items()})):json.dump({'insert':{'cond':cd,'name_class_tokens':{'person':2,'place':3}}},open(wd/f'nat_9201_{lay}.json','w'))
    rep,md=analyze(d,wd);print(md)
    r=rep[('R1-alt (SQuAD tail, primary)',2)];assert abs(r['delta'][0]+0.30)<0.03 and r['signed'] and r['n_pairs']==24 and r['c0_n']==12,r
    print('e52_analyze SELFTEST OK')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--nat-dir');ap.add_argument('--stream',type=int,default=9201);ap.add_argument('--tags',default='e52a,e52b');ap.add_argument('--cell',default='L0fs');ap.add_argument('--out',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    rep,md=analyze(a.dir,a.nat_dir,a.stream,tuple(a.tags.split(',')),a.cell);print(md);out=Path(a.out or Path(a.dir)/f'e52_report_{a.stream}.md');out.write_text(md);json.dump({f'{k[0]}|{k[1]}':v for k,v in rep.items()},open(out.with_suffix('.json'),'w'),indent=1)
