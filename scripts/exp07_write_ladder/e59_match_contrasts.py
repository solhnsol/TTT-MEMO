"""Paired contrasts of job 1329 (number-matched hiding controls; seeds 231-238; exposure 128 out-of-window; new file; CPU; EXPLORATORY): arm - SAME and arm - arm per seed, three metrics, mean [95% t-CI], same-direction seeds. python3 e59_match_contrasts.py"""
import sys,numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
import e59_reagg as RA
seeds,arms,ld=RA.DS['match'];seeds=list(seeds);V={a:np.array([ld(s,a,128)[0] for s in seeds]) for a in arms}
for mi,mn in enumerate(RA.MET):
    print(f'[{mn}] (rank: negative = better)')
    for a,b in (('SAME_U1','SAME'),('RANDTOK','SAME'),('BGSENT','SAME'),('SAME_U1','RANDTOK'),('SAME_U1','BGSENT'),('BGSENT','RANDTOK')):
        d=V[a][:,mi]-V[b][:,mi];m,l,h,k=RA.ci(d);print(f'  {a:8s} - {b:8s} {m:7.3f} [{l:7.3f},{h:7.3f}] {k}/8')
