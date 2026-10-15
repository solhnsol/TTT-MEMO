import sys,time,json;sys.path.insert(0,'/home/solhnsol/workspace/TTT-MEMO/scripts/exp04_what_ttt_retains')
from core import *
import stage2,gen_world as gw
OUT=Path('/home/solhnsol/workspace/TTT-MEMO/experiments/exp04_what_ttt_retains/smoke_probe')
rt=Runtime();w=gw.build_world(1000)
for swap in (False,True):
    t=time.monotonic();stage2.run_cal(rt,w,swap,True,'smoke',OUT);print('CAL ON swap',swap,round(time.monotonic()-t),flush=True)
