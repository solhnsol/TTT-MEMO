import sys,time;sys.path.insert(0,'/home/solhnsol/workspace/TTT-MEMO/scripts/exp04_what_ttt_retains')
from core import *
rt=Runtime();print('loaded',flush=True)
rng=np.random.default_rng(0)
for n in (16384,32768,65536):
    ids=rng.integers(1000,100000,size=n).tolist()
    for on in (False,True):
        t=time.monotonic()
        try:
            lp,m=rt.stream(ids,on)
            print(n,'ON' if on else 'OFF','sec',round(time.monotonic()-t,1),'mean nll',float(-lp.mean()),jax.devices()[0].memory_stats()['peak_bytes_in_use']/2**30,flush=True)
        except Exception as e:print(n,on,'FAIL',str(e)[:200],flush=True);break
