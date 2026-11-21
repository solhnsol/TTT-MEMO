"""Analysis of job 1337 (partner-cue check; CPU; new file; EXPLORATORY): per seed and arm (ALLHID / PARTNER / SHOWOTHER) the out-of-window margin (gold - best wrong, mean over the 16 fact items) at exposure E (128) in four reads: no context (standard read), context = the sentence of the partner B(A), of the non-partner controls C1(A), C2(A). Cue effect of an arm = M(B) - mean(M(C1), M(C2)); main contrast = cue effect(PARTNER) - cue effect(ALLHID) (difference of differences), paired over seeds, mean [95% t-CI], same-direction seeds; also SHOWOTHER - ALLHID, PARTNER - SHOWOTHER, the no-context margin gain of each arm over W0 and over ALLHID, top-1 and mean-rank versions. python3 e59_cue_analyze.py --dir D [--seeds 231-238] [--exp 128] [--tag pc]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e59_reagg as RA
a=sys.argv;D=Path(a[a.index('--dir')+1]);E=int(a[a.index('--exp')+1]) if '--exp' in a else 128;tag=a[a.index('--tag')+1] if '--tag' in a else 'pc'
lo,hi=(231,238)
if '--seeds' in a:lo,hi=[int(x) for x in a[a.index('--seeds')+1].split('-')]
ARMS=('ALLHID','PARTNER','SHOWOTHER');SEEDS=list(range(lo,hi+1));n=len(SEEDS);t=stats.t.ppf(.975,n-1)
def rd(arm,s,suffix):
    r=json.load(open(D/f'E59_{tag}_{arm}_s{s}.{suffix}read.json'));return RA.vec(r['read'][str(E)]['out'],r['meta']),RA.vec(r['W0']['out'],r['meta'])
V={}
for arm in ARMS:
    for s in SEEDS:
        for ctx,suf in (('none',''),('B','cue_B.'),('C1','cue_C1.'),('C2','cue_C2.')):V[(arm,s,ctx)]=rd(arm,s,suf)
def arr(arm,ctx,mi,w0=False):return np.array([V[(arm,s,ctx)][1 if w0 else 0][mi] for s in SEEDS])
def ci(d):d=np.asarray(d);m=d.mean();h=t*d.std(ddof=1)/np.sqrt(n);return f'{m:7.3f} [{m-h:7.3f},{m+h:7.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{n}'
print(f'partner-cue check: seeds {lo}-{hi}, exposure {E} out-of-window; mean [95% t] same-direction seeds; chance top-1 0.0625, rank 8.5 (rank: lower = better, so a cue effect is NEGATIVE for rank)')
for mi,mn in enumerate(RA.MET):
    print(f'[{mn}]')
    cue={arm:arr(arm,'B',mi)-(arr(arm,'C1',mi)+arr(arm,'C2',mi))/2 for arm in ARMS}
    for arm in ARMS:print(f'  {arm:9s} no-context {arr(arm,"none",mi).mean():7.3f} (vs W0 {ci(arr(arm,"none",mi)-arr(arm,"none",mi,True))}) | B {arr(arm,"B",mi).mean():7.3f}  C1 {arr(arm,"C1",mi).mean():7.3f}  C2 {arr(arm,"C2",mi).mean():7.3f} | cue effect B-C {ci(cue[arm])}')
    print('  difference of differences (cue effect):')
    for x,y in (('PARTNER','ALLHID'),('SHOWOTHER','ALLHID'),('PARTNER','SHOWOTHER')):print(f'    {x:9s} - {y:9s} {ci(cue[x]-cue[y])}')
    print('  no-context binding (read without context) vs ALLHID:')
    for x in ('PARTNER','SHOWOTHER'):print(f'    {x:9s} - ALLHID    {ci(arr(x,"none",mi)-arr("ALLHID","none",mi))}')
    print('  context effect on its own (B - no context):')
    for arm in ARMS:print(f'    {arm:9s} {ci(arr(arm,"B",mi)-arr(arm,"none",mi))}')
