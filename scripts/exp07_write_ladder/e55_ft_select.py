"""E55 design 22.5: learning-rate choice for the stage-1 test job from the dev sweep (job 186, seed 55 only), applied SEPARATELY for each arm:
 1. drop a learning rate that produced NaN / inf (any logged CE, grad norm, update norm, epoch mean) or whose fact-token CE at the last epoch (4) is HIGHER than at epoch 2;
 2. of the rest pick the one with the highest mean out-of-window g (u gain, W0 subtracted, mean over facts) at epoch 4;
 3. if the top two differ by less than 0.05 in g, take the one with the lower epoch-4 fact-token CE;
 4. if the arm's maximal g (over the kept learning rates) is <= 0.05, take the stable (kept) learning rate with the lowest epoch-4 fact-token CE (mean CE if that is missing) and flag the arm 'no dev signal';
 5. if the pick is at an end of the grid, keep it and flag 'grid edge, upper bound may be underestimated';
 6. 187 runs once with these rates; they are never changed after seeing results.
 A learning rate whose fact-token CE is missing at epoch 2 or 4 is not dropped by rule 1 and gets the flag 'fact-token CE missing' (the rule cannot be applied); an arm with no kept learning rate has no choice (flag 'all learning rates dropped').
  python3 e55_ft_select.py --dir OUT [--tag ftdev --seed 55 --lrs 1e-5,3e-5,1e-4 --ep-lo 2 --ep-hi 4] [--out table.md]      prints the table and the arguments of job 187"""
import sys,json,argparse,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
ARMS=('FT-NAT','FT-MASK','FT-ISO');EPS_G=0.05
def finite_all(x):
    if isinstance(x,dict):return all(finite_all(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(finite_all(v) for v in x)
    if isinstance(x,float):return math.isfinite(x)
    return True
def choose(cands,grid,eps=EPS_G):
    """cands: {lr: {'finite': bool, 'g4': float or None, 'fce2': float or None, 'fce4': float or None, 'ce4': float or None}} -> (lr or None, [flags], {lr: reason dropped})"""
    flags=[];dropped={};keep={}
    for lr,c in cands.items():
        if not c['finite']:dropped[lr]='NaN/inf';continue
        if c['fce2'] is not None and c['fce4'] is not None and c['fce4']>c['fce2']:dropped[lr]='fact-token CE at epoch 4 higher than at epoch 2';continue
        if c['fce2'] is None or c['fce4'] is None:flags.append(f'fact-token CE missing for lr {lr:g}')
        keep[lr]=c
    if not keep:return None,flags+['all learning rates dropped'],dropped
    ok=lambda c:c['g4'] is not None and math.isfinite(c['g4'])
    gmax=max((c['g4'] for c in keep.values() if ok(c)),default=None)
    key=lambda lr:(keep[lr]['fce4'] if keep[lr]['fce4'] is not None else (keep[lr]['ce4'] if keep[lr]['ce4'] is not None else float('inf')))
    if gmax is None or gmax<=eps:
        pick=min(keep,key=key);flags.append('no dev signal')
    else:
        order=sorted([lr for lr in keep if ok(keep[lr])],key=lambda lr:-keep[lr]['g4']);pick=order[0]
        if len(order)>1 and keep[order[0]]['g4']-keep[order[1]]['g4']<eps:pick=min(order[:2],key=key)
    if pick in (min(grid),max(grid)):flags.append('grid edge, upper bound may be underestimated')
    return pick,flags,dropped
def gather(d,tag,seed,lrs,ep_lo,ep_hi,load=None):
    import e55_ic_report as RP
    load=load or (lambda f:json.load(open(f)));res={}
    for ar in ARMS:
        res[ar]={}
        for lr in lrs:
            fs=Path(d)/f'E55_{tag}_{ar}_lr{lr:g}_s{seed}.saved.json'
            if not fs.exists():continue
            sv=load(fs);ep={e['epoch']:e for e in sv['epochs']};fin=finite_all(sv.get('log',[]))and finite_all(sv['epochs'])
            g4=None;fo=Path(d)/f'E55_{tag}_{ar}_lr{lr:g}_s{seed}_ep{ep_hi}.read.json'
            if fo.exists():
                try:g4=float(np.mean(RP.fact_g(load(fo),64,'out')))
                except Exception:g4=None
            res[ar][lr]={'finite':bool(fin),'g4':g4,'fce2':(ep.get(ep_lo) or {}).get('fact_token_ce'),'fce4':(ep.get(ep_hi) or {}).get('fact_token_ce'),'ce4':(ep.get(ep_hi) or {}).get('mean_ce')}
    return res
def table(res,grid):
    L=['# E55 stage 1: learning-rate choice from the dev sweep (rule E55 22.5)','','| arm | lr | finite | g out (epoch 4) | fact-token CE ep2 | fact-token CE ep4 | mean CE ep4 | dropped / chosen |','|---|---|---|---|---|---|---|---|'];args={};allflags={}
    f=lambda x:'n/a' if x is None else f'{x:.4f}'
    for ar,c in res.items():
        pick,flags,dropped=choose(c,grid)
        for lr,v in sorted(c.items()):
            L.append(f"| {ar} | {lr:g} | {v['finite']} | {f(v['g4'])} | {f(v['fce2'])} | {f(v['fce4'])} | {f(v['ce4'])} | {dropped.get(lr,'CHOSEN' if lr==pick else '')} |")
        args[ar]=pick;allflags[ar]=flags
    L+=['']+[f"- {ar}: {'lr '+format(args[ar],'g') if args[ar] is not None else 'NO CHOICE'}; flags: {allflags[ar] or 'none'}" for ar in res]
    if all(v is not None for v in args.values()):L+=['',f"Arguments of job 187: LR_NAT={args['FT-NAT']:g} LR_MASK={args['FT-MASK']:g} LR_ISO={args['FT-ISO']:g}   (e.g. LR_NAT={args['FT-NAT']:g} LR_MASK={args['FT-MASK']:g} LR_ISO={args['FT-ISO']:g} PY=... bash jobs_proposed/187_e55_ft_test_seed61.sh)"]
    return '\n'.join(L)+'\n',args,allflags
def selftest():
    grid=[1e-5,3e-5,1e-4];c=lambda f=True,g=0.3,a=2.0,b=1.5,ce=2.0:{'finite':f,'g4':g,'fce2':a,'fce4':b,'ce4':ce}
    p,fl,dr=choose({1e-5:c(g=0.1),3e-5:c(g=0.4),1e-4:c(g=0.2)},grid);assert p==3e-5 and fl==[] and not dr
    p,fl,dr=choose({1e-5:c(g=0.1),3e-5:c(g=0.4),1e-4:c(f=False,g=0.9)},grid);assert p==3e-5 and 1e-4 in dr
    p,fl,dr=choose({1e-5:c(g=0.1),3e-5:c(g=0.4),1e-4:c(g=0.9,a=1.0,b=1.2)},grid);assert p==3e-5 and '4 higher' in dr[1e-4]
    p,fl,dr=choose({1e-5:c(g=0.1),3e-5:c(g=0.40,b=1.9),1e-4:c(g=0.43,b=1.2)},grid);assert p==1e-4 and 'grid edge' in fl[0]           # rule 3 (|diff| < 0.05 -> lower CE) and rule 5
    p,fl,dr=choose({1e-5:c(g=0.1),3e-5:c(g=0.40,b=1.2),1e-4:c(g=0.43,b=1.9)},grid);assert p==3e-5 and fl==[]
    p,fl,dr=choose({1e-5:c(g=0.01,b=1.8),3e-5:c(g=0.02,b=1.1),1e-4:c(g=0.03,b=1.4)},grid);assert p==3e-5 and 'no dev signal' in fl
    p,fl,dr=choose({1e-5:c(f=False),3e-5:c(a=1.0,b=2.0)},grid);assert p is None and 'all learning rates dropped' in fl
    p,fl,dr=choose({1e-5:c(g=0.5,a=None,b=None),3e-5:c(g=0.1)},grid);assert p==1e-5 and any('missing' in x for x in fl) and any('grid edge' in x for x in fl)
    # file round trip with synthetic records
    import tempfile,e55_ic_report as RP
    d=Path(tempfile.mkdtemp());rng=np.random.default_rng(2);K=8;nf=8
    def rec(sig):
        n=nf+4;gold=rng.integers(0,K,n);ex=np.array([[True]*K+[False]*8]*n);meta={'kind':['fact']*nf+['ctrl']*4,'gold_idx':gold.tolist(),'cand_exposed':ex.tolist(),'n_rank':[K]*n,'d':{'64':{'in_g0':[1]*nf+[None]*4}},'item_idx':{'64':{'in_g0':list(range(n))}}}
        base=rng.normal(size=(n,16));W=base+0.2*rng.normal(size=(n,16));W[np.arange(nf),gold[:nf]]+=sig
        return {'meta':meta,'read':{'64':{'out':W.tolist(),'in_g0':W.tolist()}},'W0':{'out':base.tolist(),'in_g0@64':base.tolist()}}
    for ar in ARMS:
        for i,lr in enumerate(grid):
            (d/f'E55_ftdev_{ar}_lr{lr:g}_s55.saved.json').write_text(json.dumps({'log':[{'ce':2.0,'gnorm':1.0,'update_norm':0.1}],'epochs':[{'epoch':2,'mean_ce':2.0,'fact_token_ce':2.5},{'epoch':4,'mean_ce':1.8,'fact_token_ce':2.0-0.1*i}]}));(d/f'E55_ftdev_{ar}_lr{lr:g}_s55_ep4.read.json').write_text(json.dumps(rec(0.5+i)))
    res=gather(d,'ftdev',55,grid,2,4);t,args,fl=table(res,grid);assert args['FT-NAT']==1e-4 and 'LR_NAT=0.0001' in t and 'grid edge' in str(fl['FT-NAT']),t
    print('e55_ft_select SELFTEST OK')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--dir');ap.add_argument('--tag',default='ftdev');ap.add_argument('--seed',type=int,default=55);ap.add_argument('--lrs',default='1e-5,3e-5,1e-4');ap.add_argument('--ep-lo',type=int,default=2);ap.add_argument('--ep-hi',type=int,default=4);ap.add_argument('--out');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    grid=[float(x) for x in a.lrs.split(',')];t,args,fl=table(gather(a.dir,a.tag,a.seed,grid,a.ep_lo,a.ep_hi),grid);print(t)
    if a.out:Path(a.out).write_text(t)
