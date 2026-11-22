"""Analysis of job 1339 (scrambled-U2 check; CPU; new file; EXPLORATORY): per seed (231-238) margin / top-1 / rank at exposure 128 (out-of-window) of NAT, SAME, SAME_U1, SAME_U2, SAME_U1U2 in the scrambled-U2 world (experiments/exp07_write_ladder/u12_scr, tag us) and in the intact U12 world of job 1328 (u12, tag u12). Main: D = SAME_U1U2 - SAME_U1 (what hiding U2 adds on top of U1 hidden) in each world and D_scr - D_intact (paired by seed, mean [95% t-CI], same-direction seeds); secondary: SAME_U2 - SAME, SAME_U1 - SAME, SAME - NAT, interaction (U1U2 - U1 - U2 + SAME), each also scr - intact. python3 e59_scr_analyze.py [--exp 128]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e59_reagg as RA
E=int(sys.argv[sys.argv.index('--exp')+1]) if '--exp' in sys.argv else 128;E7=RA.E7;S=list(range(231,239));n=len(S);t=stats.t.ppf(.975,n-1)
def rd(world,arm,s):
    d,tag=(('u12_scr','us') if world=='scr' else ('u12','u12'));r=json.load(open(E7/f'{d}/E59_{tag}_{arm}_s{s}.read.json'));return RA.vec(r['read'][str(E)]['out'],r['meta'])
ARMS=('NAT','SAME','SAME_U1','SAME_U2','SAME_U1U2')
def ci(d):d=np.asarray(d);m=d.mean();h=t*d.std(ddof=1)/np.sqrt(n);return f'{m:7.3f} [{m-h:7.3f},{m+h:7.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{n}'
for mi,mn in enumerate(RA.MET):
    V={w:{a:np.array([rd(w,a,s)[mi] for s in S]) for a in ARMS} for w in ('scr','intact')};print(f'[{mn}] exposure {E}; means scrambled',{a:round(float(V["scr"][a].mean()),3) for a in ARMS},'intact',{a:round(float(V["intact"][a].mean()),3) for a in ARMS})
    f={'D = U1U2 - U1 (U2 hiding on top of U1)':lambda v:v['SAME_U1U2']-v['SAME_U1'],'SAME_U2 - SAME':lambda v:v['SAME_U2']-v['SAME'],'SAME_U1 - SAME':lambda v:v['SAME_U1']-v['SAME'],'SAME - NAT':lambda v:v['SAME']-v['NAT'],'U1U2 - SAME (all others hidden)':lambda v:v['SAME_U1U2']-v['SAME'],'interaction U1U2-U1-U2+SAME':lambda v:v['SAME_U1U2']-v['SAME_U1']-v['SAME_U2']+v['SAME']}
    for name,fn in f.items():
        a=fn(V['scr']);b=fn(V['intact']);print(f'  {name:42s} scrambled {ci(a)} | intact {ci(b)} | scrambled - intact {ci(a-b)}')
