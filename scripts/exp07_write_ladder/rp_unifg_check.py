"""UNIF-G size assertion (CPU, no GPU): per pass (1-4), world (A, B) and chunk k, the TOTAL number of masked tokens of the UNIF-G cell (O + uniform fill) must equal that of the BG cell (n_masked in OUT/S3_<tag>_s<seed>_<world>.saved.json, log[pass]['chunks']).
  rp_unifg_check.py --bg w1_bg --unifg w1_unifg [--seed 101] [--passes 1,2,3,4] | --selftest      exit 0 equal everywhere, 1 any mismatch (the mismatching chunks are listed), 2 file missing."""
import os,sys,json,argparse,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
OUT=Path(os.environ.get('EXP07_OUT',HERE.parents[1]/'experiments/exp07_write_ladder'))
def check(od,bg,uf,seed,passes):
    bad=[];n=0
    for w in 'AB':
        fb,fu=od/f'S3_{bg}_s{seed}_{w}.saved.json',od/f'S3_{uf}_s{seed}_{w}.saved.json'
        if not(fb.exists() and fu.exists()):return None,[f'missing {f.name}' for f in (fb,fu) if not f.exists()]
        lb,lu=json.load(open(fb))['log'],json.load(open(fu))['log']
        for ps in passes:
            cb={c['k']:int(c['n_masked']) for c in lb[ps]['chunks']};cu={c['k']:int(c['n_masked']) for c in lu[ps]['chunks']}
            if set(cb)!=set(cu):bad.append((w,ps,'chunk sets differ'));continue
            for k in sorted(cb):
                n+=1
                if cb[k]!=cu[k]:bad.append((w,ps,k,cb[k],cu[k]))
    return n,bad
def selftest():
    td=Path(tempfile.mkdtemp())
    for tag,off in (('b',0),('u',0),('x',1)):
        for w in 'AB':json.dump({'log':[{'chunks':[]}]+[{'chunks':[{'k':k,'n_masked':100+k+(off if k==2 else 0)} for k in range(4)]} for _ in range(4)]},open(td/f'S3_{tag}_s101_{w}.saved.json','w'))
    n,bad=check(td,'b','u',101,[1,2,3,4]);assert n==32 and not bad
    n,bad=check(td,'b','x',101,[1,2,3,4]);assert len(bad)==8 and bad[0][2]==2
    n,bad=check(td,'b','nope',101,[1]);assert n is None;print('rp_unifg_check SELFTEST OK')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--bg');ap.add_argument('--unifg');ap.add_argument('--seed',type=int,default=101);ap.add_argument('--passes',default='1,2,3,4');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    n,bad=check(OUT,a.bg,a.unifg,a.seed,[int(x) for x in a.passes.split(',')])
    if n is None:print('MISSING',bad);sys.exit(2)
    print(f'{n} (world, pass, chunk) totals compared; mismatches: {len(bad)}');[print(' ',b) for b in bad[:20]];sys.exit(1 if bad else 0)
