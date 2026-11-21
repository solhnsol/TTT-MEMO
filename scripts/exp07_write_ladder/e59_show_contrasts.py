"""Paired contrasts of jobs 1333 / 1334 (show only the same fact; E2E 231-238, Qwen 221-228; exposure 128 out-of-window; new file; CPU; EXPLORATORY): per-seed differences of margin / top-1 / rank between the arms, mean [95% t-CI], same-direction seeds. python3 e59_show_contrasts.py"""
import sys,numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
import e59_reagg as RA
for ds in ('show_e2e','show_qwen'):
    seeds,arms,ld=RA.DS[ds];seeds=list(seeds);V={a:np.array([ld(s,a,128)[0] for s in seeds]) for a in arms};print('==',ds)
    for mi,mn in enumerate(RA.MET):
        print(f'[{mn}] (rank: negative = better)')
        for a,b in (('SHOWSAME','NAT'),('SHOWSAME','ALLHID'),('SHOWOTHER','NAT'),('SHOWOTHER','ALLHID'),('SHOWSAME','SHOWOTHER'),('ALLHID','CHK'),('ALLHID','NAT')):
            d=V[a][:,mi]-V[b][:,mi];m,l,h,k=RA.ci(d);print(f'  {a:9s} - {b:9s} {m:7.3f} [{l:7.3f},{h:7.3f}] {k}/{len(seeds)}')
    print('  raw means',{a:[round(float(V[a][:,i].mean()),3) for i in range(3)] for a in arms})
