"""E47 CPU analysis (no GPU, no tokenizer): the four log-probability terms behind the L3 binding margin of a pair, per cell and world, and how they move from one cell to another (e.g. K3 -> k3h). Reads result files that carry the gold / foil span log-probs themselves (margins['_alt_gold'], margins['_alt_foil'] = per item the SUMMED token log-prob of the gold / foil answer span under the SQuAD-tail readout; margin = gold - foil), which ladder.evaluate writes since the E47 change (older result files hold only the margin: reported as 'no gold/foil stored', the cell must be run again).
Definitions (gen_world.probes, readouts.py): L3 item of person a of pair (a, b): prompt PR_HOME with the canonical (world-A) name of a, gold = a's world-A place P_a, foil = the other place P_b of the pair; item of b: gold P_b, foil P_a. The items are the same in worlds A and B; in world B the two names of every pair are swapped in the stream (bindings flip), so the same prompt now has the wrong binding. Per world, per pair, four terms:  s(P_a|a) = gold of a's item, s(P_b|a) = foil of a's item, s(P_b|b) = gold of b's item, s(P_a|b) = foil of b's item (log-probs summed over the place-name tokens). XP items (family XP, pair 'x<n>', prompt with the name of person j of pair n, gold = the place of person j of the NEXT pair, foil = the other place of that pair) are the null that D_alt subtracts: xg = mean gold, xf = mean foil of the two XP items of the pair.
D_alt convention: per item (A-B)/2. So per term the world effect is t = (term_A - term_B)/2 and  D_a(a) = [t(P_a|a) - t(P_b|a)] - [t(xg) - t(xf)],  D_a(b) = [t(P_b|b) - t(P_a|b)] - [t(xg) - t(xf)];  the pair D_alt = (D_a(a) + D_a(b))/2 (assert against the margin based triple_da). a = masked person of k3h (half_person_of rule), b = partner.
Usage: e47_terms.py --cells L0fs:main,k3,k3h,k3hc --seeds 101 --epochs 4 [--ref k3] [--out-dir D]   |   --selftest.   Output <out-dir>/e47_terms.{md,json}: per cell the 15-pair means of t(P_a|a), t(P_b|a), t(P_b|b), t(P_a|b), xg, xf (+ raw world A and B terms), then the change of every term from --ref cell to each other cell (mean, pair bootstrap 95% CI)."""
import os,sys,json,argparse,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
import e47_triple_da as TD
ROOT=HERE.parents[1];OUT=Path(os.environ.get('EXP07_OUT',ROOT/'experiments/exp07_write_ladder'))
TERMS=('s(Pa|a)','s(Pb|a)','s(Pb|b)','s(Pa|b)','xg','xf')
def terms(E,fam,seed):
    """-> {pair: {term: {'A': v, 'B': v}}} (raw summed log-probs per world), a / b by the half_person_of rule; raises KeyError('gold/foil') when the file has no gold / foil fields"""
    for w in 'AB':
        m=E[w]['margins']
        if '_alt_gold' not in m or '_alt_foil' not in m:raise KeyError('no gold/foil stored (older result file: only margins)')
    g={w:np.asarray(E[w]['margins']['_alt_gold']['L3']) for w in 'AB'};f={w:np.asarray(E[w]['margins']['_alt_foil']['L3']) for w in 'AB'}
    xg={w:np.asarray(E[w]['margins']['_alt_gold']['XP']) for w in 'AB'};xf={w:np.asarray(E[w]['margins']['_alt_foil']['XP']) for w in 'AB'}
    by={}
    for k,it in enumerate(fam['L3']):by.setdefault(it['pair'],[]).append((it['tid'],k))
    xk={}
    for k,it in enumerate(fam['XP']):xk.setdefault(TD._num(it['pair'],'x'),[]).append(k)
    out={}
    for p,lst in by.items():
        assert len(lst)==2,(p,lst);ids=[t for t,_ in lst];a=SETS.half_person_of(seed,p,ids);ka=[k for t,k in lst if t==a][0];kb=[k for t,k in lst if t!=a][0];n=TD._num(p,'p')
        if n not in xk:raise ValueError(f'no XP items for pair x{n}')
        out[p]={'s(Pa|a)':{w:float(g[w][ka]) for w in 'AB'},'s(Pb|a)':{w:float(f[w][ka]) for w in 'AB'},'s(Pb|b)':{w:float(g[w][kb]) for w in 'AB'},'s(Pa|b)':{w:float(f[w][kb]) for w in 'AB'},
                'xg':{w:float(np.mean(xg[w][xk[n]])) for w in 'AB'},'xf':{w:float(np.mean(xf[w][xk[n]])) for w in 'AB'}}
    return out
def effects(T):
    """-> pair -> term -> (A-B)/2 plus the derived D_a(a), D_a(b)"""
    R={}
    for p,d in T.items():
        t={k:(v['A']-v['B'])/2 for k,v in d.items()};xp=t['xg']-t['xf'];t['D_a(a)']=t['s(Pa|a)']-t['s(Pb|a)']-xp;t['D_a(b)']=t['s(Pb|b)']-t['s(Pa|b)']-xp;R[p]=t
    return R
def beta_table(T):
    """per pair: place preference beta_w = ([s(Pa|a)+s(Pa|b)] - [s(Pb|a)+s(Pb|b)])/2 in world w (how much more the readout favours place P_a over P_b over the two prompts, whatever the binding), and the identities (assert): D_a + D_b = {[t(Pa|a)-t(Pa|b)] + [t(Pb|b)-t(Pb|a)]} - 2 (t(xg)-t(xf)) (beta cancels in the pair sum), D_a - D_b = beta_A - beta_B (the beta part of the individual D_a that a world average does not remove); t = (A-B)/2."""
    R={};E=effects(T)
    for p,d in T.items():
        b={w:((d['s(Pa|a)'][w]+d['s(Pa|b)'][w])-(d['s(Pb|a)'][w]+d['s(Pb|b)'][w]))/2 for w in 'AB'};t={k:(v['A']-v['B'])/2 for k,v in d.items()};xp=t['xg']-t['xf']
        ident=(t['s(Pa|a)']-t['s(Pa|b)'])+(t['s(Pb|b)']-t['s(Pb|a)'])-2*xp
        assert abs(E[p]['D_a(a)']+E[p]['D_a(b)']-ident)<1e-9,(p,E[p],ident)
        assert abs(E[p]['D_a(a)']-E[p]['D_a(b)']-(b['A']-b['B']))<1e-9,(p,E[p],b)
        R[p]={'beta_A':b['A'],'beta_B':b['B'],'D_sum':ident,'D_diff':b['A']-b['B']}
    return R
def abs_table(T):
    """E49 (iii) and 2-candidate accuracy per pair from the raw terms. For an item the world-A bound place is its gold and the world-B bound place is its foil (the names are swapped in B), so  bound = [lp(gold|A) + lp(foil|B)]/2, unbound = [lp(foil|A) + lp(gold|B)]/2 (the log-prob of the place that is / is not bound to the person in that world, averaged over the two worlds; bound - unbound = the item's (A-B)/2 margin);  accuracy = [(margin_A > 0) + (margin_B < 0)]/2 = share of the two worlds in which the readout prefers the bound place of the two candidates (the formula of nat_analyze); both items of a pair are averaged. xbase = mean over the pair's XP items and both worlds of the two place log-probs (a binding-free reference: a shift of it is a general shift of place log-probs, not binding)."""
    R={}
    for p,d in T.items():
        its={'a':('s(Pa|a)','s(Pb|a)'),'b':('s(Pb|b)','s(Pa|b)')};bo=[];un=[];ac=[]
        for g,f in its.values():
            bo.append((d[g]['A']+d[f]['B'])/2);un.append((d[f]['A']+d[g]['B'])/2);ac.append(((d[g]['A']-d[f]['A']>0)+(d[g]['B']-d[f]['B']<0))/2)
        R[p]={'bound':float(np.mean(bo)),'unbound':float(np.mean(un)),'acc':float(np.mean(ac)),'xbase':float((d['xg']['A']+d['xf']['A']+d['xg']['B']+d['xf']['B'])/4)}
    return R
def analyse(od,cells,seed,epochs,ref,compare=()):
    rng=np.random.default_rng(54);md=['# E47 log-prob terms behind the binding margin (CPU)\n','t(term) = (term in world A - term in world B)/2 of the SUMMED log-prob of the answer span; a = k3h masked person, b = partner; D_a(a) = [t(Pa|a)-t(Pb|a)] - [t(xg)-t(xf)], D_a(b) = [t(Pb|b)-t(Pa|b)] - [t(xg)-t(xf)] (pair D_alt = their mean). 15 pairs, bootstrap 95% CI over pairs.\n'];res={'seed':seed,'epochs':{}};cols=list(TERMS)+['D_a(a)','D_a(b)']
    for ep in epochs:
        eff={};raw={};md.append(f'\n## epoch {ep}\n')
        for c in cells:
            try:
                E,fam=TD.load(od,c,seed,ep,True);T=terms(E,fam,seed);eff[c]=effects(T);raw[c]=T
                tri=TD.triple_da(E,fam,seed)   # margin based D_a: assert that the term based D_a(a), D_a(b) reproduce it
                for p,t in eff[c].items():
                    for key,role in (('D_a(a)',True),('D_a(b)',False)):
                        tt=[v for v in tri.values() if v['pair']==p and v['masked']==role][0]['AB'];assert abs(t[key]-tt)<1e-6,(c,p,key,t[key],tt)
            except FileNotFoundError as e:md.append(f'- {c}: missing ({Path(str(e.filename)).name})')
            except KeyError as e:md.append(f'- {c}: {e.args[0]}')
            except ValueError as e:md.append(f'- {c}: {e}')
        R={'cells':{},'change':{},'beta':{},'abs':{}}
        md+=['| cell | '+' | '.join(cols)+' |','|---|'+'---|'*len(cols)]
        for c,e in eff.items():
            ps=sorted(e,key=lambda p:TD._num(p,'p'));R['cells'][c]={k:TD.boot([e[p][k] for p in ps],rng) for k in cols};md.append(f'| {c} | '+' | '.join(f"{R['cells'][c][k][0]:+.3f}" for k in cols)+' |')
        md+=['\nRaw terms per world (mean of the 15 pairs; A / B):\n','| cell | '+' | '.join(TERMS)+' |','|---|'+'---|'*len(TERMS)]
        for c,T in raw.items():
            R['cells'][c]['raw']={k:{w:float(np.mean([T[p][k][w] for p in T])) for w in 'AB'} for k in TERMS};md.append(f'| {c} | '+' | '.join(f"{R['cells'][c]['raw'][k]['A']:+.2f} / {R['cells'][c]['raw'][k]['B']:+.2f}" for k in TERMS)+' |')
        md+=['\nPlace preference beta_w = ([s(Pa|a)+s(Pa|b)] - [s(Pb|a)+s(Pb|b)])/2 per world (the same for both persons of a pair); identities asserted: D_a + D_b = [t(Pa|a)-t(Pa|b)] + [t(Pb|b)-t(Pb|a)] - 2(t(xg)-t(xf)) (beta cancels), D_a - D_b = beta_A - beta_B. Signs of beta are arbitrary (P_a = the world-A place of the k3h masked person), so |beta| is the size; "before" = per world, "after" = after the (A-B)/2 average.\n','| cell | mean beta_A | mean beta_B | mean abs beta_A | mean abs beta_B (before) | mean (D_a - D_b) | mean abs (D_a - D_b) (after) | mean D_a + D_b |','|---|---|---|---|---|---|---|---|']
        for c,T in raw.items():
            bt=beta_table(T);ps=sorted(bt,key=lambda p:TD._num(p,'p'));v=lambda k:np.asarray([bt[p][k] for p in ps]);R['beta'][c]={'beta_A':float(v('beta_A').mean()),'beta_B':float(v('beta_B').mean()),'abs_beta_A':float(np.abs(v('beta_A')).mean()),'abs_beta_B':float(np.abs(v('beta_B')).mean()),'D_diff':float(v('D_diff').mean()),'abs_D_diff':float(np.abs(v('D_diff')).mean()),'D_sum':float(v('D_sum').mean()),'per_pair':bt}
            x=R['beta'][c];md.append(f"| {c} | {x['beta_A']:+.3f} | {x['beta_B']:+.3f} | {x['abs_beta_A']:.3f} | {x['abs_beta_B']:.3f} | {x['D_diff']:+.3f} | {x['abs_D_diff']:.3f} | {x['D_sum']:+.3f} |")
        ab={c:abs_table(T) for c,T in raw.items()};md+=['\n### E49: absolute log-prob of the bound / unbound place and 2-candidate accuracy (L3 items, alt readout; 15 pairs, bootstrap 95% CI over pairs)\n','bound = log-prob of the place bound to the person in that world (world A: gold, world B: foil), averaged over A and B; unbound = the other candidate; acc = share of (item, world) cases in which the bound place of the two candidates is preferred (chance 0.5); xbase = mean place log-prob of the binding-free XP items (a shift of it is not binding).\n','| cell | mean bound lp [CI] | mean unbound lp [CI] | 2-cand accuracy [CI] | xbase |','|---|---|---|---|---|']
        for c,t in ab.items():
            ps=sorted(t,key=lambda p:TD._num(p,'p'));v=lambda k:[t[p][k] for p in ps];R['abs'][c]={k:TD.boot(v(k),rng) for k in ('bound','unbound','acc','xbase')};x=R['abs'][c];md.append(f"| {c} | {TD.f3(x['bound'])} | {TD.f3(x['unbound'])} | {x['acc'][0]:.3f} [{x['acc'][1]:.3f},{x['acc'][2]:.3f}] | {x['xbase'][0]:+.3f} |")
        if ref in ab:
            md+=[f'\nChange from `{ref}` (cell - {ref}): delta bound, delta unbound, delta XP base (general place shift), delta bound net of the XP shift, delta accuracy; mean [95% CI] over pairs. Pre-registered reading (E49): a change counts only if the CI excludes 0 and |delta| >= 0.2 nats (accuracy: >= 0.10).\n','| cell | delta bound | delta unbound | delta xbase | delta bound - delta xbase | delta accuracy |','|---|---|---|---|---|---|'];R['abs_change']=R.get('abs_change',{})
            for c,t in ab.items():
                if c==ref:continue
                ps=sorted(t,key=lambda p:TD._num(p,'p'));d=lambda k:[t[p][k]-ab[ref][p][k] for p in ps];ch={'bound':TD.boot(d('bound'),rng),'unbound':TD.boot(d('unbound'),rng),'xbase':TD.boot(d('xbase'),rng),'bound_net':TD.boot([t[p]['bound']-ab[ref][p]['bound']-(t[p]['xbase']-ab[ref][p]['xbase']) for p in ps],rng),'acc':TD.boot(d('acc'),rng)};R['abs_change'][c]=ch
                md.append(f"| {c} | {TD.f3(ch['bound'])} | {TD.f3(ch['unbound'])} | {TD.f3(ch['xbase'])} | {TD.f3(ch['bound_net'])} | {TD.f3(ch['acc'])} |")
        for pair in compare:
            c1,c2=pair.split('=')
            try:
                E1,f1=TD.load(od,c1,seed,ep,True);E2,f2=TD.load(od,c2,seed,ep,True);dm=max(float(np.max(np.abs(np.asarray(E1[w]['margins']['_alt'][f])-np.asarray(E2[w]['margins']['_alt'][f])))) for w in 'AB' for f in ('L3','XP'));R.setdefault('compare',{})[pair]=dm;md.append(f'\nCompare {c1} vs {c2} (epoch {ep}): max |alt margin difference| over L3 and XP items, worlds A and B = {dm:.3g}')
            except (FileNotFoundError,KeyError,ValueError) as e:md.append(f'\nCompare {pair}: not possible ({e})')
        if ref in eff:
            md+=[f'\nChange of every term from `{ref}` to the cell (cell - {ref}; mean [95% CI] over pairs):\n','| cell | '+' | '.join(cols)+' |','|---|'+'---|'*len(cols)]
            for c,e in eff.items():
                if c==ref:continue
                ps=sorted(e,key=lambda p:TD._num(p,'p'));R['change'][c]={k:TD.boot([e[p][k]-eff[ref][p][k] for p in ps],rng) for k in cols};md.append(f'| {c} | '+' | '.join(TD.f3(R['change'][c][k]) for k in cols)+' |')
        res['epochs'][ep]=R
    md.append('\nDescriptive only (one seed, 15 pairs). The foil of a\'s item is the place bound to b in world A and to a in world B, so t(Pb|a) is the pair\'s "wrong place" term of a; terms are summed span log-probs, not NLLs per token.');return '\n'.join(md)+'\n',res
def selftest():
    td=Path(tempfile.mkdtemp());seed=101;items=[];fs={}
    for f in ('L3','XP'):
        its=[{'pair':(f'p{i//2}' if f=='L3' else f'x{i//2}'),'tid':i} for i in range(30)];items+=its;fs[f]=30
    hp={n:SETS.half_person_of(seed,f'p{n}',[2*n,2*n+1]) for n in range(15)}
    def cellw(pre,sa,sb,gold=True):
        # per-term world values: s(Pa|a)=sa (A), sa-1 (B) ...; a-item gold/foil, b-item gold/foil
        for w in 'AB':
            d=0.0 if w=='A' else 1.0;G=[0.0]*30;F=[0.0]*30;XG=[0.0]*30;XF=[0.0]*30
            for n in range(15):
                a=hp[n];b=2*n+1-(a-2*n)
                G[a]=-2.0-sa*(1-d)+(-d);F[a]=-3.0+d;G[b]=-2.5-sb*(1-d)+(-d)*0.5;F[b]=-3.5+d*0.5;XG[2*n]=XG[2*n+1]=-4.0;XF[2*n]=XF[2*n+1]=-4.5
            L3=[g-f for g,f in zip(G,F)];XP=[g-f for g,f in zip(XG,XF)]
            m={'_alt':{'L3':L3,'XP':XP}}
            if gold:m.update({'_alt_gold':{'L3':G,'XP':XG},'_alt_foil':{'L3':F,'XP':XF}})
            json.dump({'log':[{'epoch':0,'margins':{'_alt':{'L3':[0.0]*30,'XP':[0.0]*30}}},{'epoch':4,'margins':m}]},open(td/f'{pre}_s{seed}_{w}.json','w'))
        json.dump({'items':items,'fam_sizes':fs},open(td/f'meta_{pre}_s{seed}.json','w'))
    cellw('S3_k3',0.0,0.0);cellw('S3_k3h',-0.5,0.3);cellw('S3_old',0.0,0.0,False)
    md,res=analyse(td,['k3','k3h','old','nofile'],seed,[4],'k3');R=res['epochs'][4]
    # k3: t(Pa|a) = ((-2-0)-(-3))/2... check by hand: gold_a A=-2, B=-3 -> t=+0.5 ; foil_a A=-3, B=-2 -> t=-0.5 ; D_a(a)=0.5+0.5-(0.25-(-0.25)=0 : xg 0, xf 0)=1.0
    c=R['cells']['k3'];assert abs(c['s(Pa|a)'][0]-0.5)<1e-9 and abs(c['s(Pb|a)'][0]+0.5)<1e-9 and abs(c['D_a(a)'][0]-1.0)<1e-9 and abs(c['xg'][0])<1e-9,c
    ch=R['change']['k3h'];assert abs(ch['s(Pa|a)'][0]-0.25)<1e-9 and abs(ch['s(Pb|a)'][0])<1e-9 and abs(ch['D_a(a)'][0]-0.25)<1e-9 and abs(ch['s(Pb|b)'][0]+0.15)<1e-9 and abs(ch['D_a(b)'][0]+0.15)<1e-9,ch   # sa=-0.5 raises t(Pa|a) by 0.25, sb=+0.3 lowers t(Pb|b) by 0.15
    A_=R['abs']['k3'];assert abs(A_['bound'][0]+2.375)<1e-9 and abs(A_['unbound'][0]+3.125)<1e-9 and abs(A_['acc'][0]-0.75)<1e-9 and abs(A_['xbase'][0]+4.25)<1e-9,A_   # hand-computed from the synthetic k3 cell (a item: bound -2 / unbound -3 / acc 1; b item: -2.75 / -3.25 / 0.5)
    assert 'old: no gold/foil stored' in md and 'nofile: missing' in md
    import shutil
    for w in 'AB':shutil.copy(td/f'S3_k3_s{seed}_{w}.json',td/f'S3_k3_s{seed}_{w}.regold.json')
    md4,res4=analyse(td,['k3@regold'],seed,[4],'k3@regold',['k3@regold=k3']);assert res4['epochs'][4]['compare']['k3@regold=k3']==0.0 and abs(res4['epochs'][4]['cells']['k3@regold']['D_a(a)'][0]-1.0)<1e-9   # the .regold file naming works and the compare check reports 0
    print('e47_terms SELFTEST OK (terms, effects, D_a reproduces the margin based triple_da, change K3 -> k3h, old file and missing file reported)');print(md)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cells',default='L0fs:main,k3,k3h,k3hc');ap.add_argument('--seeds',default='101');ap.add_argument('--epochs',default='4');ap.add_argument('--ref',default='k3');ap.add_argument('--compare',default='',help='comma list of cellA=cellB: max |alt margin difference| between two result files (e.g. k3@regold=k3 to check the re-scoring, L0fs:fskeep@regold=L0fs:main for the native)');ap.add_argument('--out-dir',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    md,res=analyse(OUT,a.cells.split(','),int(a.seeds.split(',')[0]),[int(x) for x in a.epochs.split(',')],a.ref,[x for x in a.compare.split(',') if x]);od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/'e47_terms.md').write_text(md);json.dump(res,open(od/'e47_terms.json','w'),indent=1);print(md)
