"""Overnight driver: sequential, resumable, deadline-aware. Every task writes its own artifact; completed artifacts are skipped
on restart. Failures are logged and skipped (the watchdog restarts the process, which resumes)."""
import os,sys,time,traceback,json,gc,subprocess,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import *
import stage1,stage2,stage2r,validate,gen_world as gw
EXP=ROOT/'experiments/exp04_what_ttt_retains'
DEADLINE=datetime.datetime(2026,10,13,5,45,tzinfo=datetime.timezone.utc).timestamp()  # 14:45 KST
def log(*a):
    s=datetime.datetime.now(datetime.timezone.utc).strftime('%m-%d %H:%M:%S')+' '+' '.join(map(str,a));print(s,flush=True)
    with open(EXP/'driver_events.log','a') as f:f.write(s+'\n')
def summarize():
    try:subprocess.run([sys.executable,'-I',str(Path(__file__).resolve().parent/'summarize.py')],capture_output=True,timeout=900)
    except Exception as e:log('summarize failed',e)
def safe(name,fn):
    if time.time()>DEADLINE:return False
    try:
        r=fn();return r
    except Exception as e:
        log('TASK FAILED',name,repr(e)[:300]);(EXP/'errors.log').open('a').write(f'--- {name}\n{traceback.format_exc()}\n');gc.collect()
        try:jax.clear_caches()
        except Exception:pass
        return False
def wv(seed):return 1 if seed<100 else 2
def stage2_seed(rt,seed):
    w=gw.build_world(seed,v=wv(seed));tag=f'seed{seed}';D=EXP/'stage2'
    ons=(True,False) if seed<=2 else (True,)   # OFF is an auxiliary diagnostic: first two seeds only
    for swap in (False,True):
        for on in ons:
            safe(f'cal {tag} {swap} {on}',lambda:stage2.run_cal(rt,w,swap,on,tag,D))
    for swap in (False,True):
        for on in ons:
            safe(f'world {tag} {swap} {on}',lambda:stage2.run_world(rt,w,swap,on,tag,D,snapshot=(seed==1 and on and not swap)))
    log('stage2 seed done',seed)
def dyn_seed(rt,seed):
    for cond in ('NOUPD','UNREL','RELSAME','RELCONF'):
        safe(f'dyn seed{seed} {cond}',lambda:stage2r.run_dyn(rt,seed,cond,EXP/'stage2'))
    log('dyn seed done',seed)
def kvempty_seed(rt,seed):
    """weights-only readout: adapted fast weights, ZERO cached KV at probe time (ON worlds A,B + OFF baseline world A)."""
    w=gw.build_world(seed,v=wv(seed));tag=f'seed{seed}';D=EXP/'stage2'
    for swap,on in ((False,True),(True,True),(False,False)):
        safe(f'kvempty {tag} {swap} {on}',lambda:stage2.run_world(rt,w,swap,on,tag,D,kv_mode='empty'))
    log('kvempty seed done',seed)
def stage2c_seed(rt,seed):
    """clean (v2) seed: carry-KV main runs (ON A,B; OFF A,B for the first two clean seeds) + weights-only (empty KV) readouts."""
    w=gw.build_world(seed,v=2);tag=f'seed{seed}';D=EXP/'stage2'
    for swap in (False,True):
        safe(f'main {tag} {swap} ON',lambda:stage2.run_world(rt,w,swap,True,tag,D))
    for swap,on in ((False,True),(True,True),(False,False)):
        safe(f'kvempty {tag} {swap} {on}',lambda:stage2.run_world(rt,w,swap,on,tag,D,kv_mode='empty'))
    if seed in (101,102):
        for swap in (False,True):
            safe(f'main {tag} {swap} OFF',lambda:stage2.run_world(rt,w,swap,False,tag,D))
    log('clean seed done',seed)
def gate_open():
    g=EXP/'gate.json'
    try:return bool(json.loads(g.read_text())['run_stage3'])
    except Exception:return False
def stage3_seed(rt,seed):
    """retention pass: relevant ON/OFF, plus two ON models adapted on DIFFERENT unrelated worlds (noise floor)."""
    w=gw.build_world(seed,v=wv(seed));tag=f'seed{seed}';D=EXP/'stage2'
    safe(f's3 {tag} A ON',lambda:stage2.run_world(rt,w,False,True,tag,D,retention=True))
    safe(f's3 {tag} A OFF',lambda:stage2.run_world(rt,w,False,False,tag,D,retention=True))
    for nm,off in (('U1',1000),('U2',2000)):
        u=gw.build_world(seed+off,v=wv(seed))
        safe(f's3 {tag} {nm}',lambda:stage2.run_world(rt,u,False,True,tag,D,retention=True,probe_world=w,name=nm))
    log('stage3 seed done',seed)
def main():
    rt=Runtime();log('runtime loaded; fast_hash',rt.fast_hash[:12])
    safe('validate',lambda:validate.run_validate(rt,EXP/'data',EXP/'stage0'));log('stage0 done')
    s3done=set();seeds_done=[]
    plan=[('s2',[1,2,3]),('s1h',range(0,20)),('s2c',[101]),('s1',range(0,40)),('s2c',[102]),('s1h',range(20,40)),('s2c',list(range(103,140)))]
    if os.environ.get('EXP04_SKIP_STAGE2'):
        log('EXP04_SKIP_STAGE2 set: probe-path check failed, running Stage 1 only');plan=[p for p in plan if p[0]=='s1']
    for kind,arg in plan:
        if time.time()>DEADLINE:break
        if kind=='s1':
            safe(f's1 {arg}',lambda:stage1.run_stage1(rt,EXP/'data',EXP/'stage1',list(arg),DEADLINE))
        elif kind=='s1h':
            safe(f's1h {arg}',lambda:stage1.run_stage1h(rt,EXP/'data',EXP/'stage1h',list(arg),DEADLINE))
        elif kind=='s2c':
            for sd in arg:
                if time.time()>DEADLINE:break
                stage2c_seed(rt,sd);summarize()
        else:
            for sd in arg:
                if time.time()>DEADLINE:break
                stage2_seed(rt,sd);seeds_done.append(sd);summarize()
                if sd<=3:dyn_seed(rt,sd)
                if gate_open():
                    for s_ in seeds_done:
                        if s_ not in s3done and time.time()<DEADLINE:stage3_seed(rt,s_);s3done.add(s_)
        summarize();log('round done',kind,list(arg)[:3],'gate',gate_open())
    summarize();(EXP/'DRIVER_DONE').write_text(str(time.time()));log('DRIVER_DONE')
if __name__=='__main__':main()
