"""Summary of the key diagnostic (job 1344t3 'kvkeys'; e59_keydiag.py; CPU; new file; thread-server 11/26; descriptive: values only). Per chunk (16, 64, 128), weights and mask: per layer (1-8) the mean over facts and seeds of cos_own - cos_oth (the write key at the place-prediction position vs the own read key minus the mean over the 15 other facts) and of cos_to_C0 (the write key vs the write key under the C0 mask); per-seed values of the layer-8 mean in brackets (min..max over seeds).
  python e59_keydiag_analyze.py --dir D"""
import sys,json,glob,numpy as np
a=sys.argv;D=a[a.index('--dir')+1];F=sorted(glob.glob(D+'/kvkeys*_s*.json'));R={}
for f in F:
    r=json.load(open(f));R.setdefault(r['seed'],[]).append(r)
seeds=sorted(R);print('seeds',seeds,'; files',len(F));print('LIMIT:',R[seeds[0]][0]['limitations'])
allw=[];[allw.append(w) for s in seeds for r in R[s] for w in r['weights'] if w not in allw]
for e in ('16','64','128'):
    print(f'== chunk {e}')
    for key,nm in (('cos_own_minus_oth','cos_own - cos_oth (mean over the 15 other facts)'),('cos_to_C0_write_key','cos(write key under the mask, write key under C0)')):
        print(f'  [{nm}]  layers 1..8 (mean over facts and seeds) | layer-8 per seed min..max')
        for wn in allw:
            for mk in ('C0','SAMEVIS','M2As-q'):
                vals=[]
                for s in seeds:
                    for r in R[s]:
                        if wn in r['chunks'][e]:vals.append(np.mean(r['chunks'][e][wn][mk][key],axis=0))
                if not vals:continue
                v=np.array(vals);print(f'    {wn:7s} {mk:7s}: '+' '.join(f'{x:6.3f}' for x in v.mean(0))+f' | {v[:,7].min():.3f}..{v[:,7].max():.3f}')
