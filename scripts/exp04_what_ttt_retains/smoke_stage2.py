import sys,time,json;sys.path.insert(0,'/home/solhnsol/workspace/TTT-MEMO/scripts/exp04_what_ttt_retains')
from core import *
import stage2,gen_world as gw,validate
ROOT=Path('/home/solhnsol/workspace/TTT-MEMO');OUT=ROOT/'experiments/exp04_what_ttt_retains'
rt=Runtime()
t=time.monotonic();print('VALIDATE',json.dumps(validate.run_validate(rt,OUT/'data',OUT/'stage0'),indent=1),round(time.monotonic()-t),flush=True)
w=gw.build_world(1000)
for on in (True,False):
    t=time.monotonic();stage2.run_cal(rt,w,False,on,'smoke',OUT/'smoke');print('CAL',on,round(time.monotonic()-t),flush=True)
t=time.monotonic();stage2.run_world(rt,w,False,True,'smoke',OUT/'smoke',snapshot=False);print('WORLD A ON',round(time.monotonic()-t),flush=True)
