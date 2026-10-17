"""S3 stage 2 analysis (spec v1.1 + the 10/17 amendments: S1 = A(oracle) - A(randdist), S2 = pooled R_rec against prefix-all-pos, (1) against prefix-all-pos with the masked-fs version alongside, common damage truncation K_c).
D = fp32 R1-alt D_alt per pair (15 pairs per seed, worlds A/B paired as in analyze_ladder.stats). A(cell) = mean over passes 1..K_c of D. Cells: native-fs (L0fs main, passes 1-4 of the 8-pass run), masked-fs (L0mwfs x1), S3 cells S3_{cell}_p4.
Damage: canary (max of gen / tail NLL rise, both worlds) > 0.10 at a pass, or Q2 (beyond-window long-context rise at pass 4, S3 cells: longnll_S3_{cell}_p4) > 0.10. Per seed and test K_c = min(first damaged pass over all cells entering the test - 1, 4); seeds with K_c < 2 are dropped.
Bootstrap: pairs resampled within each seed, mean over seeds of the per-seed means (8000 draws). Usage: analyze_s3_stage2.py [--dry]   (--dry replaces the missing S3 cells by base cells to test the plumbing; it prints a banner and writes nothing)."""
import sys,os,json,importlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
OUT=Path(os.environ.get('EXP07_OUT',Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder'))
DRY='--dry' in sys.argv
def mods(sd):
    sys.argv=['x',str(sd),'main']
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
def curve(AL,S,lev,tag,maxp=4):
    A,B=S.load(lev,tag);alt=lambda e:{'epoch':e.get('epoch'),'margins':e['margins']['_alt']};b2=(alt(A[0]),alt(B[0]));D={};can={}
    for a,b in zip(A,B):
        ep=a.get('epoch')
        if ep is None or ep==0 or ep>maxp:continue
        ks=[k for k in ('_gen_nll','_tail_nll') if k in a['margins']]
        can[ep]=max([a['margins'][k]-A[0]['margins'][k] for k in ks]+[b['margins'][k]-B[0]['margins'][k] for k in ks]);D[ep]=AL.stats(alt(a),alt(b),b2)['D']
    return D,can
CELLS={'native':('L0fs','main'),'masked':('L0mwfs','x1')}
S3CELLS=['oracle','randdist','prefixall','recency','oraclesub','othentsub','attnsel','randattn','recencyattn','attnk16','attnstoch']
def load_all():
    data={};q2={}
    for sd in (101,102,103):
        AL,S=mods(sd);data[sd]={}
        for nm,(lev,tag) in CELLS.items():data[sd][nm]=curve(AL,S,lev,tag)
        for c in S3CELLS:
            f=OUT/f'S3_{c}_p4_s{sd}_A.json'
            if DRY:data[sd][c]=data[sd]['masked' if c in('oracle','prefixall','attnsel') else 'native'];continue   # dry run never opens a real S3 cell
            if f.exists():
                try:data[sd][c]=curve(AL,S,'S3',f'{c}_p4')
                except Exception as e:print('could not load',c,sd,e)
            lf=OUT/f'longnll_S3_{c}_p4_s{sd}.json'
            if lf.exists():
                d=json.load(open(lf));q2[(sd,c)]=np.mean([[e for e in d[w] if e['epoch']==4][0]['rise_beyond'] for w in 'AB'])
        for nm,f in (('native',f'longnll_L0fs_fskeep_s{sd}.json'),('masked',f'longnll_L0mwfs_x1_s{sd}.json')):
            d=json.load(open(OUT/f));q2[(sd,nm)]=np.mean([[e for e in d[w] if e['epoch']==4][0]['rise_beyond'] for w in 'AB'])
    return data,q2
def first_damaged(data,q2,sd,cell):
    D,can=data[sd][cell]
    for p in range(1,5):
        if can.get(p,0)>0.10:return p
    if q2.get((sd,cell),0)>0.10:return 4
    return 99
def Kc(data,q2,sd,cells):
    return min(min(first_damaged(data,q2,sd,c) for c in cells)-1,4)
def area(data,sd,cell,K):return np.mean([data[sd][cell][0][p] for p in range(1,K+1)],axis=0)
def boot(vals,B=8000,seed=0):
    """vals: {sd: per-pair array}; pairs resampled within seed, mean over seeds of per-seed means -> (point, lo, hi)"""
    r=np.random.default_rng(seed);sds=sorted(vals);pt=float(np.mean([vals[s].mean() for s in sds]));xs=[]
    for _ in range(B):xs.append(np.mean([vals[s][r.integers(0,len(vals[s]),len(vals[s]))].mean() for s in sds]))
    return pt,float(np.percentile(xs,2.5)),float(np.percentile(xs,97.5))
def ratio_boot(num,den,B=8000,seed=1):
    r=np.random.default_rng(seed);sds=sorted(num);f=lambda n,d:np.mean([n[s].mean() for s in sds])/np.mean([d[s].mean() for s in sds]);pt=float(f(num,den));xs=[];dn=[]
    for _ in range(B):
        idx={s:r.integers(0,len(num[s]),len(num[s])) for s in sds};xs.append(f({s:num[s][idx[s]] for s in sds},{s:den[s][idx[s]] for s in sds}));dn.append(np.mean([den[s][idx[s]].mean() for s in sds]))
    return pt,float(np.percentile(xs,2.5)),float(np.percentile(xs,97.5)),float(np.percentile(dn,2.5)),float(np.percentile(dn,97.5))
def per_seed(data,q2,cells,fn):
    """fn(area_dict, sd) -> per-pair array for the seeds with K_c >= 2; returns ({sd: array}, {sd: K_c})"""
    out={};ks={}
    for sd in (101,102,103):
        if any(c not in data[sd] for c in cells):continue
        K=Kc(data,q2,sd,cells);ks[sd]=K
        if K>=2:out[sd]=fn({c:area(data,sd,c,K) for c in cells})
    return out,ks
def main():
    data,q2=load_all()
    if DRY:print('*** DRY RUN: missing S3 cells replaced by base cells; the numbers below are plumbing checks only, not results ***')
    res={}
    def show(name,vals,ks,extra=''):
        if not vals:print(f'{name}: no usable seed (K_c {ks})');return None
        pt,lo,hi=boot(vals);ps={sd:round(float(v.mean()),3) for sd,v in vals.items()};print(f'{name}: pooled {pt:+.3f} [{lo:+.3f},{hi:+.3f}]; per seed {ps}; K_c {ks} {extra}');return pt,lo,hi,ps
    # base gains
    for ref in ('prefixall','masked'):
        if not all(ref in data[s] for s in data):continue
        cells=['native','oracle',ref]
        v,ks=per_seed(data,q2,cells,lambda a:(a['oracle']-a['native'])-0.8*(a[ref]-a['native']))
        res[f'(1) vs {ref}']=show(f'(1) A(oracle)-A(native)-0.8[A({ref})-A(native)]',v,ks)
    if all('randdist' in data[s] for s in data):
        v,ks=per_seed(data,q2,['oracle','randdist'],lambda a:a['oracle']-a['randdist']);r=show('S1  A(oracle)-A(randdist)',v,ks)
        if r:res['S1']=r;print('    all seed point estimates positive:',all(x>0 for x in r[3].values()))
    for ref in ('prefixall','masked'):
        if not all(ref in data[s] for s in data):continue
        cells=['native','oracle',ref];v,ks=per_seed(data,q2,cells,lambda a:a['oracle']-a['native']);w,_=per_seed(data,q2,cells,lambda a:a[ref]-a['native'])
        if v and w:
            sds=sorted(set(v)&set(w));rt=ratio_boot({s:v[s] for s in sds},{s:w[s] for s in sds});res[f'S2 vs {ref}']=rt
            print(f'S2 (vs {ref}) R_rec = [A(oracle)-A(native)]/[A({ref})-A(native)] pooled {rt[0]:.3f} [{rt[1]:.3f},{rt[2]:.3f}]; denominator CI [{rt[3]:+.3f},{rt[4]:+.3f}] ({"defined" if rt[3]>0 else "NOT defined: no verdict"}); K_c {ks}')
    # descriptive
    for a_,b_,name in (('masked','prefixall','masked-fs - prefix-all-pos'),('oraclesub','othentsub','oracle-sub - oth-ent-sub'),('oracle','recency','oracle - recency'),('attnsel','randattn','attn-sel - rand-dist@attn'),('attnsel','recencyattn','attn-sel - recency@attn'),('attnsel','prefixall','attn-sel - prefix-all-pos')):
        if all(a_ in data[s] and b_ in data[s] for s in data):
            v,ks=per_seed(data,q2,['native',a_,b_],lambda a:a[a_]-a[b_]);show('desc '+name,v,ks)
    if not DRY:json.dump({k:v for k,v in res.items() if v},open(OUT/'s3_stage2_result.json','w'),indent=1,default=str)
if __name__=='__main__':main()
