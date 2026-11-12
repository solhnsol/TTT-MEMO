"""S3 equivalence check (professor 17:03): empty-mask S3 path (cell native3, flag on, pass 1) vs the original native fs path (L0fs fskeep, job 37) on s101-103.
(1) pass-1 DeltaW relative difference ||W_s3 - W_fs|| / ||W_fs - W0|| over the inner parameters (tolerance 1e-3), per seed and world;
(2) fp32 R1-alt D_alt, mean over the 45 pairs (3 seeds x 15), difference S3-empty minus L0fs main pass 1 (tolerance 0.005; an equivalence check, not a verdict).
Reads kept states (files) and result jsons; CPU."""
import sys,json,importlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
OUT=Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder'
def leaves(fn):
    z=np.load(fn);return [z[f'arr_{i}'].astype(np.float64) for i in range(len(z.files))]
def state_files(tag_lev,tag,sd,w):
    d=json.load(open(OUT/f'{tag_lev}_{tag}_s{sd}_{w}.saved.json'));return [e['margins']['_file'] for e in d['log'] if 'margins' in e and '_file' in e['margins']]
res={}
for sd in (101,102,103):
    for w in 'AB':
        try:
            a=state_files('S3','native3_p1',sd,w);b=state_files('L0fs','fskeep',sd,w)
        except Exception as e:print(sd,w,'missing',e);continue
        W0=leaves(a[0]);s3=leaves(a[1]);fs=leaves(b[1])
        num=np.sqrt(sum(float(np.sum((x-y)**2)) for x,y in zip(s3,fs)));den=np.sqrt(sum(float(np.sum((x-y)**2)) for x,y in zip(fs,W0)))
        res[f'{sd}{w}']=num/den;print(f'seed {sd} world {w}: pass-1 DeltaW relative difference {num/den:.2e} (tolerance 1e-3)')
sys.argv=['x','101','main'];import analyze_ladder as AL,analyze_stage01 as S
ds=[];dl=[]
for sd in (101,102,103):
    sys.argv=['x',str(sd),'main']
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    AL=importlib.import_module('analyze_ladder');S=importlib.import_module('analyze_stage01')
    def cur(lev,tag):
        A,B=S.load(lev,tag);alt=lambda e:{'epoch':e.get('epoch'),'margins':e['margins']['_alt']};b2=(alt(A[0]),alt(B[0]));return AL.stats(alt(A[1]),alt(B[1]),b2)['D']
    try:ds.append(cur('S3','native3_p1'));dl.append(cur('L0fs','main'))
    except Exception as e:print(sd,'D_alt missing',e)
if ds:
    a=np.concatenate(ds);b=np.concatenate(dl);print(f'D_alt pass 1, {len(a)} pairs: S3-empty {a.mean():+.4f}, L0fs {b.mean():+.4f}, difference {a.mean()-b.mean():+.4f} (tolerance 0.005)')
    res['D_alt_diff']=float(a.mean()-b.mean())
json.dump(res,open(OUT/'s3_equiv.json','w'),indent=1)
