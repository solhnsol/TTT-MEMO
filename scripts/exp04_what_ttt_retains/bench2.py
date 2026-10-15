import sys,time;sys.path.insert(0,'/home/solhnsol/workspace/TTT-MEMO/scripts/exp04_what_ttt_retains')
from core import *
rt=Runtime();print('loaded',flush=True)
rng=np.random.default_rng(0)
for n in (28672,28672,64*1024):
    ids=rng.integers(1000,100000,size=n).tolist()
    for on in (True,False):
        t=time.monotonic();lp,m=rt.stream(ids,on)
        print(n,'ON' if on else 'OFF','sec',round(time.monotonic()-t,1),flush=True)
