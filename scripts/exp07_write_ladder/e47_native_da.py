"""E47 CPU (no GPU, no tokenizer): the pair-level D_alt of saved ladder result files per epoch (pass), for the native (no mask) runs of seed 101 and any other ladder / S3 cell, read from OUT/<prefix>_<tag>_s<seed>_{A,B}.json (margins['_alt'], A/B averaged as in analyze_e35.D_alt; 15 pairs; pair bootstrap 95% CI) and the item list of the matching meta file.
  e47_native_da.py --seed 101 --cells L0fs:main,L0reset_P63:r5fresh,S3:k3,S3:oracle [--out-dir D]
A cell 'LEVEL:TAG' reads OUT/LEVEL_TAG_s<seed>_{A,B}.json and meta_LEVEL_TAG_s<seed>.json (S3 cells: S3_<tag>_..., meta_S3_<tag>_...); a missing file is reported, not an error. Per epoch it prints D_alt (mean [CI]) and the file names with the sha256 (first 12 hex) of the result files, so that the value is tied to the file. Rules: D_alt = M(L3) - M(XP) per pair, M = mean over the pair's items of (margin in world A - margin in world B)/2; L3 items 'p<n>' align with XP items 'x<n>' by number.   --selftest: synthetic."""
import os,sys,json,argparse,hashlib,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
ROOT=HERE.parents[1];OUT=Path(os.environ.get('EXP07_OUT',ROOT/'experiments/exp07_write_ladder'));NB=5000
def sha12(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
def pairs_D(eA,eB,meta):
    items=meta['items'];fam={};i=0
    for f,n in meta['fam_sizes'].items():fam[f]=items[i:i+n];i+=n
    def M(f,pre):
        a=np.asarray(eA['margins']['_alt'][f]);b=np.asarray(eB['margins']['_alt'][f]);d={}
        for it,x in zip(fam[f],(a-b)/2):
            pr=it['pair']
            if not(pr.startswith(pre) and pr[len(pre):].isdigit()):raise ValueError(f'pair id {pr!r} is not {pre}<n>')
            d.setdefault(int(pr[len(pre):]),[]).append(float(x))
        return {k:float(np.mean(v)) for k,v in d.items()}
    l3,xp=M('L3','p'),M('XP','x');assert sorted(l3)==sorted(xp);return np.asarray([l3[k]-xp[k] for k in sorted(l3)])
def boot(v,rng):
    v=np.asarray(v,float);b=np.asarray([v[rng.integers(0,len(v),len(v))].mean() for _ in range(NB)]);return float(v.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def read(od,spec,seed):
    lev,tag=spec.split(':',1);pre=f'{lev}_{tag}' if lev!='S3' else f'S3_{tag}';fa,fb,fm=od/f'{pre}_s{seed}_A.json',od/f'{pre}_s{seed}_B.json',od/f'meta_{pre}_s{seed}.json'
    miss=[f.name for f in (fa,fb,fm) if not f.exists()]
    if miss:return None,f'missing: {miss}'
    la,lb=json.load(open(fa))['log'],json.load(open(fb))['log'];E={w:{e['epoch']:e for e in L if isinstance(e.get('margins'),dict) and 'epoch' in e and '_alt' in e['margins']} for w,L in (('A',la),('B',lb))}
    meta=json.load(open(fm));rng=np.random.default_rng(51);res={}
    for ep in sorted(set(E['A'])&set(E['B'])):
        if ep==0:continue
        res[ep]=boot(pairs_D(E['A'][ep],E['B'][ep],meta),rng)
    return res,f'files {fa.name} ({sha12(fa)}), {fb.name} ({sha12(fb)}), {fm.name} ({sha12(fm)})'
def main_(od,seed,cells):
    md=[f'# D_alt per epoch, seed {seed} (CPU; pair-level, worlds A/B averaged; pair bootstrap 95% CI)\n','| cell | epoch 1 | epoch 2 | epoch 3 | epoch 4 | files |','|---|---|---|---|---|---|'];out={}
    for c in cells:
        r,info=read(od,c,seed);out[c]={'D_alt':r,'info':info}
        if r is None:md.append(f'| {c} | {info} | | | | |');continue
        md.append(f"| {c} | "+' | '.join((f'{r[e][0]:+.3f} [{r[e][1]:+.3f},{r[e][2]:+.3f}]' if e in r else '') for e in (1,2,3,4))+f' | {info} |')
    return '\n'.join(md)+'\n',out
def selftest():
    td=Path(tempfile.mkdtemp());items=[];fs={}
    for f in ('L3','XP'):items+=[{'pair':(f'p{i//2}' if f=='L3' else f'x{i//2}'),'tid':i} for i in range(30)];fs[f]=30
    json.dump({'items':items,'fam_sizes':fs},open(td/'meta_L0fs_main_s101.json','w'))
    for w,sg in (('A',1),('B',-1)):
        log=[{'epoch':0,'margins':{'_alt':{'L3':[0.0]*30,'XP':[0.0]*30}}}]+[{'epoch':e,'margins':{'_alt':{'L3':[sg*0.1*e*2]*30,'XP':[0.0]*30}}} for e in (1,2,3,4)]
        json.dump({'log':log},open(td/f'L0fs_main_s101_{w}.json','w'))
    md,out=main_(td,101,['L0fs:main','S3:k3']);r=out['L0fs:main']['D_alt'];assert abs(r[4][0]-0.8)<1e-9 and abs(r[1][0]-0.2)<1e-9 and 'missing' in md;print('e47_native_da SELFTEST OK');print(md)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--cells',default='L0fs:main,L0reset_P63:r5fresh,S3:native3,S3:k3,S3:oracle');ap.add_argument('--out-dir',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    md,out=main_(OUT,a.seed,a.cells.split(','));od=Path(a.out_dir) if a.out_dir else OUT;od.mkdir(parents=True,exist_ok=True);(od/f'e47_native_da_s{a.seed}.md').write_text(md);json.dump(out,open(od/f'e47_native_da_s{a.seed}.json','w'),indent=1);print(md)
