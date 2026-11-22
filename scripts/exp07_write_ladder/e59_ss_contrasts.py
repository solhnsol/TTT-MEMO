"""Paired contrasts of the same-fact specificity jobs (1330 Qwen synthetic, 1331 Qwen FictionalQA, 1332 E2E FictionalQA; exposure 128 out-of-window; new file; CPU; EXPLORATORY): per-seed margin / top-1 / rank of SAME, OTHERONLY, BOTH, RAND (RANDF in 1330, RANDC in 1331 / 1332) against NAT and against each other, additivity (BOTH - SAME - OTHERONLY + NAT), mean [95% t-CI], same-direction seeds. python3 e59_ss_contrasts.py {ss_qwen_syn|ss_qwen_fq|ss_e2e_fq}"""
import sys,numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
import e59_reagg as RA
ds=sys.argv[1];seeds,arms,ld=RA.DS[ds];seeds=list(seeds);rnd='RANDF' if 'RANDF' in arms else 'RANDC';V={a:np.array([ld(s,a,128)[0] for s in seeds]) for a in arms};W0=np.array([ld(s,'NAT',128)[1] for s in seeds])
print('==',ds,'n =',len(seeds),'(rank: negative = better)')
for mi,mn in enumerate(RA.MET):
    print(f'[{mn}] W0 {W0[:,mi].mean():.3f}; raw means',{a:round(float(V[a][:,mi].mean()),3) for a in arms})
    def show(name,d):m,l,h,k=RA.ci(d);print(f'  {name:42s} {m:7.3f} [{l:7.3f},{h:7.3f}] {k}/{len(seeds)}')
    g=lambda a:V[a][:,mi]
    for a in ('SAME','OTHERONLY','BOTH',rnd):show(f'{a} - NAT',g(a)-g('NAT'))
    show('BOTH - SAME',g('BOTH')-g('SAME'));show('BOTH - OTHERONLY',g('BOTH')-g('OTHERONLY'));show(f'{rnd} - SAME',g(rnd)-g('SAME'));show(f'BOTH - {rnd}',g('BOTH')-g(rnd));show('SAME - OTHERONLY',g('SAME')-g('OTHERONLY'))
    show('additivity: BOTH - SAME - OTHERONLY + NAT',g('BOTH')-g('SAME')-g('OTHERONLY')+g('NAT'))
