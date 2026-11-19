"""Repeat-distance classes of the fact sentences in a world (CPU; new file; council request 11/19, definition 'previous repeats remaining inside the window'): for the LAST exposure of every fact (the one the exposure-128 read follows) the distance in tokens (sentence start to sentence start, absolute stream positions) to every EARLIER occurrence of the same fact sentence is computed and the number of earlier repeats is counted per class: <=1024 | 1024-4096 | 4096-8192 | >8192 (outside the 8192 write window of every arm). Also per fact the same counts for ALL exposures averaged. Output: per-world means (JSON-able). Use: dist_table(world) for any gen_facts / fq world.
  python3 e59_dist.py --seed 71 [--cell v3_16_1] | --selftest"""
import sys,json,numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CLASSES=((0,1024,'<=1024'),(1024,4096,'1024-4096'),(4096,8192,'4096-8192'),(8192,10**9,'>8192'))
def positions(world):
    pos={}
    for r,c in enumerate(world['chunks']):
        for s in c['sents']:pos.setdefault(s['fact'],[]).append(r*1024+int(s['start']))
    return pos
def dist_table(world):
    pos=positions(world);last={c[2]:[] for c in CLASSES};allc={c[2]:[] for c in CLASSES};nrep=[]
    for f,v in pos.items():
        v=sorted(v)
        for i,x in enumerate(v):
            d=np.asarray([x-y for y in v[:i]])
            cnt={c[2]:int(((d>c[0])&(d<=c[1])).sum()) if c[0]>0 else int((d<=c[1]).sum()) for c in CLASSES}
            for k,vv in cnt.items():allc[k].append(vv)
            if i==len(v)-1:
                for k,vv in cnt.items():last[k].append(vv)
        nrep.append(len(v))
    return dict(n_facts=len(pos),exposures_per_fact=float(np.mean(nrep)),last_exposure_mean_earlier_repeats={k:float(np.mean(v)) for k,v in last.items()},all_exposures_mean_earlier_repeats={k:float(np.mean(v)) for k,v in allc.items()})
def selftest():
    w=dict(chunks=[dict(sents=[dict(fact=0,start=100)]) for _ in range(12)]);t=dist_table(w);L=t['last_exposure_mean_earlier_repeats'];assert L['<=1024']==1 and L['1024-4096']==3 and L['4096-8192']==4 and L['>8192']==3,L;print('e59_dist selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        import e55_run as R55,e57_syn_pass as SP
        a=sys.argv;seed=int(a[a.index('--seed')+1]);cell=a[a.index('--cell')+1] if '--cell' in a else 'v3_16_1'
        enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps={16,32,64,128},arms=['NAT']);print(json.dumps(dist_table(w),indent=1))
