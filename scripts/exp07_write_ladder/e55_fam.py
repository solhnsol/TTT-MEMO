"""E55 design 17: familiarity-vs-binding readout on SAVED STATES (GPU read, frozen weights; analysis CPU). Per fact (kind 'fact' items of the world, k >= 16): candidates = [gold] + [the 15 other exposed places of the same world] + [16 never-exposed fresh places with the same token length]. For every candidate the summed span log-probability (full span) and the log-probability of the span's FIRST token are read at W (saved state) and at W0 (before any update) in the neutral out-of-window context. Deltas D = S(W) - S(W0). Per fact:
  gold = D[gold]; exposed = mean D over the 15 exposed non-gold; fresh = mean D over the 16 fresh;
  familiarity = exposed - fresh (every shown place goes up relative to never-shown places: place-name / format learning)
  binding = gold - exposed (this fact's own place goes up more than the other shown places)
  margin_Dalt = D[gold] - D[first exposed non-gold candidate of the item] (the old pair-style margin: gold vs one alternative) and margin_mean = gold - exposed (= binding)
  rank16 = rank of the gold among the 16 exposed candidates at W and W0 (the E55 metric) and the gain.
CI = 95% bootstrap over facts (10000), pooled over seeds as in the verdict (facts resampled within each seed, seed means averaged); NAT vs SWD contrasts paired by fact.
  python3 e55_fam.py --analyze --files FILE1,FILE2,... [--out report.md]      (files of `e55_run.py fam`)    |    python3 e55_fam.py --selftest"""
import sys,json,argparse,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[0]/'exp04_what_ttt_retains'))
NE=15;NF=16;NCF=1+NE+NF
def fam_items(world,fresh):
    """one item per fact: cands = ' '+place of [gold, 15 exposed non-gold (item order of gen_facts.build_items), 16 fresh]; meta for the analysis"""
    import gen_facts as G
    assert world['k']>=G.NCAND and len(fresh)==NF;items=[];ex=[it for it in G.build_items(world) if it['kind']=='fact']
    for it in ex:
        gi=it['gold_idx'];c=it['cands'][:G.NCAND];cl=[c[gi]]+[x for j,x in enumerate(c) if j!=gi];assert len(cl)==1+NE
        order16=list(range(G.NCAND));items.append({'fact':it['fact'],'prefix':it['prefix'],'cands':cl+[' '+x for x in fresh],'gold_idx16':gi,'cands16':c,'qframe':it['qframe']})
    return items
def read_fam(rd,rows_full,rows_first,n_items,out_tail,state_files,m0,log=print):
    """-> {'W0': {'full','first'}, arm: {a: {'full','first'}}}; rows_*: item-major rows of NCF candidates; weights are frozen (a state is loaded, scored on the shared Packed and dropped)"""
    assert len(out_tail)==8192;out={};t0=time.monotonic();Pf=rd.packed(rows_full,out_tail);P1=rd.packed(rows_first,out_tail)
    out['W0']={'full':rd.score(Pf,m0,n_items,NCF),'first':rd.score(P1,m0,n_items,NCF)};log(f'  W0 read: {time.monotonic()-t0:.0f}s')
    for ar in sorted(state_files):
        out[ar]={}
        for a,(fn,h) in sorted(state_files[ar].items()):
            t0=time.monotonic();m=rd.load_state(fn,h);out[ar][a]={'full':rd.score(Pf,m,n_items,NCF),'first':rd.score(P1,m,n_items,NCF)};del m;log(f'  {ar} a={a} read: {time.monotonic()-t0:.0f}s')
    return out
def first_rows(raw):
    """span (a, b) -> (a, a+1): the first token of the candidate"""
    return [(ids,(a,a+1)) for ids,(a,b) in raw]
def per_fact(Sw,Sw0,which='full'):
    """Sw, Sw0: dicts with 'full'/'first' (n,32) -> per-fact arrays"""
    D=np.asarray(Sw[which],float)-np.asarray(Sw0[which],float);gold=D[:,0];ex=D[:,1:1+NE].mean(1);fr=D[:,1+NE:].mean(1)
    return {'gold':gold,'exposed':ex,'fresh':fr,'familiarity':ex-fr,'binding':gold-ex,'margin_Dalt':gold-D[:,1]}
def rank16(S,gold_idx16):
    """rank (1 = best, ties averaged) of the gold among the 16 exposed candidates; S (n,32) in fam order [gold, 15 exposed, fresh] -> the first 16 columns, gold = column 0"""
    X=np.asarray(S,float)[:,:1+NE];g=X[:,0:1];return 1+(X>g).sum(1)+0.5*((X==g).sum(1)-1)
def boot(per_seed,nb=10000,seed=21):
    rng=np.random.default_rng(seed);b=np.mean([x[rng.integers(0,len(x),(nb,len(x)))].mean(1) for x in per_seed],axis=0);return float(np.mean([x.mean() for x in per_seed])),*np.percentile(b,[2.5,97.5]).tolist()
def analyze(files,out=None):
    R={}
    for f in files:
        r=json.load(open(f));R.setdefault(r['arm'],{})[r['seed']]=r
    arms=sorted(R);seeds=sorted(set.intersection(*[set(R[a]) for a in arms]));L=[f'# E55 familiarity vs binding (design 17): arms {arms}, seeds {seeds}; D = change W - W0 in log-probability (nats), CI = 95% bootstrap over facts (within seed, seed means averaged)','']
    P={};rk={}
    for a in arms:
        for sd in seeds:
            r=R[a][sd];S=r['scores'];S0=r['W0']
            for wh in ('full','first'):P[(a,sd,wh)]=per_fact(S,S0,wh)
            rk[(a,sd)]=(rank16(S0['full'],None)-rank16(S['full'],None))/15
    for wh,title in (('full','full span (summed log-probability)'),('first','first token of the span')):
        L+=[f'## {title}','','| arm | gold D | exposed non-gold D | fresh D | familiarity (exposed - fresh) | binding (gold - exposed) | margin vs one alternative (D_alt style) |','|---|---|---|---|---|---|---|']
        for a in arms:
            row=[a]
            for k in ('gold','exposed','fresh','familiarity','binding','margin_Dalt'):
                c=boot([P[(a,sd,wh)][k] for sd in seeds]);row.append('%+.3f [%+.3f, %+.3f]'%c)
            L.append('| '+' | '.join(row)+' |')
        if len(arms)>=2 and 'NAT' in arms:
            for a in arms:
                if a=='NAT':continue
                L.append(f'| {a} - NAT | '+' | '.join('%+.3f [%+.3f, %+.3f]'%boot([P[(a,sd,wh)][k]-P[('NAT',sd,wh)][k] for sd in seeds],seed=22) for k in ('gold','exposed','fresh','familiarity','binding','margin_Dalt'))+' |')
        L.append('')
    L+=['## per seed (full span): familiarity / binding','','| arm | seed | familiarity | binding |','|---|---|---|---|']
    for a in arms:
        for sd in seeds:L.append(f"| {a} | {sd} | {P[(a,sd,'full')]['familiarity'].mean():+.3f} | {P[(a,sd,'full')]['binding'].mean():+.3f} |")
    L+=['','## rank gain among the 16 exposed candidates (E55 metric, u units = rank gain / 15)','','| arm | rank gain in u [95% CI] |','|---|---|']
    for a in arms:L.append(f"| {a} | {'%+.3f [%+.3f, %+.3f]'%boot([rk[(a,sd)] for sd in seeds],seed=23)} |")
    L+=['','Reading: familiarity > 0 with binding ~ 0 = the rise of the old measurements is place-name / format familiarity; binding > 0 = the rank metric of E55 was insensitive. Facts within a seed are not independent training runs: the CIs do not include seed-level variation.']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def selftest():
    rng=np.random.default_rng(3);n=16
    def sc(gold_b,exp_b,fresh_b):
        S=rng.normal(size=(n,NCF))*0.05;S[:,0]+=gold_b;S[:,1:1+NE]+=exp_b;S[:,1+NE:]+=fresh_b;return S
    S0={'full':np.zeros((n,NCF)),'first':np.zeros((n,NCF))};import tempfile;d=tempfile.mkdtemp();fs=[]
    for arm,(g,e,f) in (('NAT',(1.0,0.6,0.1)),('SWD',(1.5,0.6,0.1))):
        for sd in (61,62):
            S={'full':sc(g,e,f),'first':sc(g/2,e/2,f/2)};fn=f'{d}/{arm}_{sd}.json';json.dump({'arm':arm,'seed':sd,'scores':{k:v.tolist() for k,v in S.items()},'W0':{k:v.tolist() for k,v in S0.items()}},open(fn,'w'));fs.append(fn)
    pf=per_fact({'full':sc(1.0,0.6,0.1)},S0,'full');assert abs(pf['familiarity'].mean()-0.5)<0.05 and abs(pf['binding'].mean()-0.4)<0.05 and abs(pf['margin_Dalt'].mean()-0.4)<0.1
    t=analyze(fs);assert '| SWD - NAT |' in t and 'binding' in t
    X=np.zeros((2,NCF));X[0,0]=1;X[1,1]=1;assert rank16(X,None).tolist()==[1.0,9.0]
    print(t.splitlines()[4]);print('e55_fam SELFTEST OK (synthetic scores)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--analyze',action='store_true');ap.add_argument('--files');ap.add_argument('--out',default=None);a=ap.parse_args()
    if a.selftest:selftest()
    elif a.analyze:print(analyze(a.files.split(','),a.out))
