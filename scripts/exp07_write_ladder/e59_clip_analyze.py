"""Analysis of job 1341 (clip-budget causal check; CPU; new file; EXPLORATORY): U12 world, seeds 231-238, exposure 128 out-of-window margin / top-1 / rank of SAME and SAME_U1U2 (and SAME_U1 / SAME_U2 if present) with the inner rule changed to a constant step (lr c, clip high; experiments/exp07_write_ladder/clip_off, tag cl) versus the ORIGINAL rule (lr 1, clip 1; job 1328, experiments/exp07_write_ladder/u12, tag u12). Main metric: G = SAME_U1U2 - SAME (the gain lost when the other facts are visible) in each rule; G_orig - G_const (paired by seed, mean [95% t-CI], same-direction seeds) and the ratio of the means with a seed-paired bootstrap. Also the update-norm statistics of both rules (steps[].applied_norm) and the share of clipped steps (log[].coef < 1). python3 e59_clip_analyze.py [--exp 128]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e59_reagg as RA
E7=RA.E7;S=list(range(231,239));n=len(S);t=stats.t.ppf(.975,n-1);EXP=int(sys.argv[sys.argv.index('--exp')+1]) if '--exp' in sys.argv else 128
def rd(rule,arm,s):
    d,tag=(('clip_off','cl') if rule=='const' else ('u12','u12'));r=json.load(open(E7/f'{d}/E59_{tag}_{arm}_s{s}.read.json'));return RA.vec(r['read'][str(EXP)]['out'],r['meta']),RA.vec(r['W0']['out'],r['meta'])
def ci(d):d=np.asarray(d);m=d.mean();h=t*d.std(ddof=1)/np.sqrt(n);return f'{m:7.3f} [{m-h:7.3f},{m+h:7.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{n}'
arms=[a for a in ('SAME','SAME_U1','SAME_U2','SAME_U1U2') if (E7/f'clip_off/E59_cl_{a}_s231.read.json').exists()]
for mi,mn in enumerate(RA.MET):
    V={r_:{a:np.array([rd(r_,a,s)[0][mi] for s in S]) for a in arms} for r_ in ('orig','const')};W0=np.array([rd('orig','SAME',s)[1][mi] for s in S])
    print(f'[{mn}] exposure {EXP}; W0 {W0.mean():.3f}; means original',{a:round(float(V['orig'][a].mean()),3) for a in arms},'constant step',{a:round(float(V['const'][a].mean()),3) for a in arms})
    for r_ in ('orig','const'):
        print(f'  {r_:6s} gain over W0: '+'  '.join(f'{a} {ci(V[r_][a]-W0)}' for a in arms))
    g_o=V['orig']['SAME_U1U2']-V['orig']['SAME'];g_c=V['const']['SAME_U1U2']-V['const']['SAME'];print(f'  G = SAME_U1U2 - SAME: original {ci(g_o)} | constant step {ci(g_c)} | G_orig - G_const {ci(g_o-g_c)}')
    rng=np.random.default_rng(0);bs=[g_c[i].mean()/g_o[i].mean() for i in (rng.integers(0,n,n) for _ in range(10000)) if abs(g_o[i].mean())>1e-6];print(f'  ratio of means G_const / G_orig {g_c.mean()/g_o.mean():.3f} [bootstrap 95% {np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]  (1 = gap unchanged, 0 = gap closed)')
for rule,d,tag in (('original','u12','u12'),('constant step','clip_off','cl')):
    for arm in arms:
        un=[];cb=[]
        for s in S:
            r=json.load(open(E7/f'{d}/E59_{tag}_{arm}_s{s}.json'));un.append(np.mean([x['applied_norm'] for x in r['steps']]));cb.append(np.mean([x['coef']<0.9999 for x in r['log']]))
        print(f'{rule:14s} {arm:10s} mean update norm {np.mean(un):.3f} (seed range {min(un):.2f}-{max(un):.2f}); share of clipped steps {np.mean(cb):.3f}')
