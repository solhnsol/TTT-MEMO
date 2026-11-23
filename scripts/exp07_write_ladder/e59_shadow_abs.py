"""ABSOLUTE binding component of the pre-clip gradient (CPU; new file; thread-server 11/23; coordinator request for Hans 'the absolute size of the binding component'; EXPLORATORY descriptive; works on the files of e59_shadow.py: shadow_s<seed>.json (exact global norms gnorm_ans / gnorm_frame / gnorm_all and the NLLs of the answer / first-entity / frame / all tokens) and shadow_s<seed>_sketch.npz (262,144-coordinate sketches of g_ans, g_frame, g_all; keys world|cond|chunk); frozen W0, no update). Binding direction = the answer-gradient of the reference condition C0 (all other facts hidden, the same fact hidden), d0 = g_ans(C0)/|g_ans(C0)| (cosines from the sketch, magnitudes from the exact norms). For every condition X in a world, per chunk, then chunk mean per seed, then mean [95% t] over seeds with the number of seeds in the sign direction, as X - C0 differences and as ratios of the seed means X / C0:
  (1) coupling component of the ANSWER gradient: B_ans = |g_ans(X)| cos(g_ans(X), d0) (absolute, along the C0 direction), its residual R_ans = |g_ans(X)| sin(...) (everything else in the answer gradient), and the gradient norm per nat of the answer loss: |g_ans| / NLL_ans, B_ans / NLL_ans (the copy effect of a visible same fact lowers the loss and the gradient together; per-nat values separate that);
  (2) the UPDATE objective g_all (mean NLL of all tokens of the chunk, the gradient that the inner rule clips): its absolute components in the orthonormal frame (d0, frame direction of C0 orthogonalised against d0, remainder): bind = |g_all| cos(g_all, d0) (the binding share of the update), frame = component along the orthogonalised C0 frame gradient, rest = |g_all| sqrt(1 - cos^2 - frame^2) (everything else), and the same quantities after the inner rule of the engine (lr 1, clip 1: step = g * min(1, 1/|g_all|); the factor is applied to all components): bind_applied, rest_applied; plus the share of the applied step that lies along d0 = cos(g_all, d0).
  (3) RATIOS (Hans 11/23: 'the share of the binding direction relative to the total loss / the total update, independent of the loss size'): bind / NLL_all (binding component of the update gradient per nat of the total chunk loss), B_ans / NLL_all, bind_applied / NLL_all (after the engine's clip), and the share of the update gradient along the binding direction, bind / |g_all| = cos(g_all, d0) (= cos_all_d0).
  python3 e59_shadow_abs.py --dir D [--seeds 231-238] [--lr 1 --clip 1]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
a=sys.argv;D=Path(a[a.index('--dir')+1]);lo,hi=(231,238)
if '--seeds' in a:lo,hi=[int(x) for x in a[a.index('--seeds')+1].split('-')]
LR=float(a[a.index('--lr')+1]) if '--lr' in a else 1.0;CLIP=float(a[a.index('--clip')+1]) if '--clip' in a else 1.0
SEEDS=list(range(lo,hi+1));n=len(SEEDS);T=stats.t.ppf(.975,n-1)
def ci(d):d=np.asarray(d,float);m=d.mean();h=T*d.std(ddof=1)/np.sqrt(len(d));return f'{m:8.3f} [{m-h:8.3f},{m+h:8.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
def cosv(x,y):return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-30))
def comps(sk,row,row0,sk0):
    """-> dict of absolute components of condition X (sketch sk [3,NS], exact row) in the frame of C0 (sk0, row0)"""
    d0=sk0[0]/np.linalg.norm(sk0[0]);f0=sk0[1]-(sk0[1]@d0)*d0;f0=f0/np.linalg.norm(f0)
    Na=row['gnorm_ans'];c=cosv(sk[0],d0);Nall=row['gnorm_all'];ca=cosv(sk[2],d0);gf=float(sk[2]@f0)/np.linalg.norm(sk[2]);rest=max(1-ca**2-gf**2,0.0)**.5
    s=min(LR,CLIP/Nall) if Nall>0 else 0.0;s=min(LR,CLIP/Nall)  # engine: step = lr * g, clipped to norm <= CLIP
    return dict(B_ans=Na*c,R_ans=Na*(max(1-c*c,0.0))**.5,N_ans=Na,N_ans_per_nll=Na/row['nll_ans'],B_ans_per_nll=Na*c/row['nll_ans'],
                bind=Nall*ca,frame=Nall*gf,rest=Nall*rest,N_all=Nall,bind_applied=Nall*ca*s,rest_applied=Nall*rest*s,frame_applied=Nall*gf*s,step_norm=Nall*s,cos_all_d0=ca,bind_per_nll_all=Nall*ca/row['nll_all'],B_ans_per_nll_all=Na*c/row['nll_all'],bind_applied_per_nll_all=Nall*ca*s/row['nll_all'])
per={}
for s in SEEDS:
    r=json.load(open(D/f'shadow_s{s}.json'));z=np.load(D/f'shadow_s{s}_sketch.npz');SK={}
    for i,k in enumerate(z['keys']):w_,c_,k_=str(k).split('|');SK[(w_,c_,int(k_))]=z['sk'][i]
    rows={(x['world'],x['cond'],x['k']):x for x in r['rows']};worlds=sorted({k[0] for k in rows});ks=sorted({k[2] for k in rows})
    for w_ in worlds:
        for c_ in sorted({k[1] for k in rows if k[0]==w_}):
            M={}
            for k in ks:
                cm=comps(SK[(w_,c_,k)],rows[(w_,c_,k)],rows[(w_,'C0',k)],SK[(w_,'C0',k)])
                for key,v in cm.items():M.setdefault(key,[]).append(v)
            M={key:float(np.mean(v)) for key,v in M.items()};M['nll_ans']=float(np.mean([rows[(w_,c_,k)]['nll_ans'] for k in ks]));M['nll_all']=float(np.mean([rows[(w_,c_,k)]['nll_all'] for k in ks]));per.setdefault((w_,c_),[]).append(M)
KEYS=('nll_ans','N_ans','B_ans','R_ans','N_ans_per_nll','B_ans_per_nll','N_all','bind','frame','rest','step_norm','bind_applied','frame_applied','rest_applied','cos_all_d0','bind_per_nll_all','B_ans_per_nll_all','bind_applied_per_nll_all')
print(f'ABSOLUTE binding component, seeds {lo}-{hi}, engine lr {LR} clip {CLIP}; per seed the chunk mean, then mean [95% t] same-direction seeds; direction d0 = g_ans of C0 (C0 row = the reference itself)')
for (w_,c_),L in sorted(per.items()):
    print(f'== world {w_}  condition {c_}:  levels')
    for key in KEYS:print(f'   {key:16s} {ci([m[key] for m in L])}')
for w_ in sorted({k[0] for k in per}):
    for c_ in sorted({k[1] for k in per if k[0]==w_}):
        if c_=='C0':continue
        print(f'-- world {w_}  {c_} - C0 (absolute) and ratio of seed means {c_} / C0')
        for key in KEYS:
            x=np.array([m[key] for m in per[(w_,c_)]]);y=np.array([m[key] for m in per[(w_,'C0')]]);print(f'   {key:16s} diff {ci(x-y)} | ratio {x.mean()/y.mean():7.3f}')
