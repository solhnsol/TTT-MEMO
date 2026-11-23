"""Analysis of the shadowing / learning-path check (CPU; new file; EXPLORATORY): per seed and world, for every condition X against the reference C0 (all other facts hidden) and every measured chunk: (i) NLL differences of the answer / first-entity / frame / all tokens (X - C0; shadowing = lower), (ii) the gradient sketches of the three objectives: cosine between g(X) and g(C0), norm ratio, projection coefficient a = <gX,g0> / <g0,g0>, the residual r = gX - a g0 of the answer gradient and its cosine with the FRAME gradient of X and of C0 (do the extra components align with frame-token learning?), (iii) the cross-chunk baseline cosine of g_ans(C0) between different chunks; chunk means per seed, then mean [95% t-CI] over seeds and paired contrasts between conditions. Main metric: cos(g_ans(X), g_ans(C0)); co-primary: NLL_ans(X) - NLL_ans(C0). python3 e59_shadow_analyze.py --dir D [--seeds 231-238]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
a=sys.argv;D=Path(a[a.index('--dir')+1]);lo,hi=(231,238)
if '--seeds' in a:lo,hi=[int(x) for x in a[a.index('--seeds')+1].split('-')]
SEEDS=list(range(lo,hi+1));n=len(SEEDS);t=stats.t.ppf(.975,n-1)
def cos(x,y):return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-30))
def ci(d):d=np.asarray(d,float);m=d.mean();h=t*d.std(ddof=1)/np.sqrt(n);return f'{m:8.4f} [{m-h:8.4f},{m+h:8.4f}] {int(max((d>0).sum(),(d<0).sum()))}/{n}'
per={}
for s in SEEDS:
    r=json.load(open(D/f'shadow_s{s}.json'));z=np.load(D/f'shadow_s{s}_sketch.npz');SK={tuple(k.split('|')):z['sk'][i] for i,k in enumerate(z['keys'])};SK={(w_,c_,int(k_)):v for (w_,c_,k_),v in SK.items()}
    rows={(x['world'],x['cond'],x['k']):x for x in r['rows']};worlds=sorted({k[0] for k in rows});ks=sorted({k[2] for k in rows})
    for w_ in worlds:
        conds=sorted({k[1] for k in rows if k[0]==w_});assert 'C0' in conds
        for c_ in conds:
            if c_=='C0':continue
            M={}
            for nm in ('ans','first','frame','all'):M['dnll_'+nm]=np.mean([rows[(w_,c_,k)]['nll_'+nm]-rows[(w_,'C0',k)]['nll_'+nm] for k in ks])
            for oi,obj in enumerate(('ans','frame','all')):
                M['cos_'+obj]=np.mean([cos(SK[(w_,c_,k)][oi],SK[(w_,'C0',k)][oi]) for k in ks]);M['normratio_'+obj]=np.mean([rows[(w_,c_,k)]['gnorm_'+obj]/rows[(w_,'C0',k)]['gnorm_'+obj] for k in ks])
                M['proj_'+obj]=np.mean([float(SK[(w_,c_,k)][oi]@SK[(w_,'C0',k)][oi]/(SK[(w_,'C0',k)][oi]@SK[(w_,'C0',k)][oi])) for k in ks])
            al=[];al0=[]
            for k in ks:
                gx=SK[(w_,c_,k)][0];g0=SK[(w_,'C0',k)][0];aa=float(gx@g0/(g0@g0));rr=gx-aa*g0;al.append(cos(rr,SK[(w_,c_,k)][1]));al0.append(cos(rr,SK[(w_,'C0',k)][1]))
            M['resid_norm_ratio']=np.mean([np.linalg.norm(SK[(w_,c_,k)][0]-float(SK[(w_,c_,k)][0]@SK[(w_,'C0',k)][0]/(SK[(w_,'C0',k)][0]@SK[(w_,'C0',k)][0]))*SK[(w_,'C0',k)][0])/np.linalg.norm(SK[(w_,'C0',k)][0]) for k in ks]);M['resid_cos_frame_X']=np.mean(al);M['resid_cos_frame_C0']=np.mean(al0)
            per.setdefault((w_,c_),[]).append(M)
        base=[cos(SK[(w_,'C0',k)][0],SK[(w_,'C0',k2)][0]) for i,k in enumerate(ks) for k2 in ks[i+1:]];per.setdefault((w_,'baseline'),[]).append({'cross_chunk_cos_ans_C0':float(np.mean(base))})
print(f'shadowing / learning-path check, seeds {lo}-{hi}; chunk means per seed, then mean [95% t] same-direction seeds over seeds (X vs C0 = all other facts hidden)')
for (w_,c_),L in sorted(per.items()):
    print(f'== world {w_}  condition {c_}')
    for key in L[0]:print(f'  {key:28s} {ci([m[key] for m in L])}')
for w_ in ('intact','scr'):
    pairs=[('U2vis','ALLVIS'),('U1vis','ALLVIS'),('U1vis','U2vis')] if w_=='intact' else []
    for x,y in pairs:
        if (w_,x) in per and (w_,y) in per:print(f'-- {w_}: {x} - {y}: cos_ans {ci([a_["cos_ans"]-b_["cos_ans"] for a_,b_ in zip(per[(w_,x)],per[(w_,y)])])}  dNLL_ans {ci([a_["dnll_ans"]-b_["dnll_ans"] for a_,b_ in zip(per[(w_,x)],per[(w_,y)])])}')
if ('intact','U2vis') in per and ('scr','U2vis') in per:print('-- U2vis scrambled - intact: cos_ans',ci([a_['cos_ans']-b_['cos_ans'] for a_,b_ in zip(per[('scr','U2vis')],per[('intact','U2vis')])]),' dNLL_ans',ci([a_['dnll_ans']-b_['dnll_ans'] for a_,b_ in zip(per[('scr','U2vis')],per[('intact','U2vis')])]),' resid_norm_ratio',ci([a_['resid_norm_ratio']-b_['resid_norm_ratio'] for a_,b_ in zip(per[('scr','U2vis')],per[('intact','U2vis')])]))
