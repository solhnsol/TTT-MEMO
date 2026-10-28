"""E57 mass / threshold analysis of the shadowing-key sets (CPU only; reads e57_shadowkey_masks.npz, the dense_ALL / dense_SH arrays and the stored top-128 head counts of e57_attdump3 on the 17 chunks k % 16 == 0; numbers only, no criterion).
Mass = the dump's post-softmax probability of a key summed over the heads of the set (ALL = 768 heads, SH = the sharp heads of the chunk) and the layers; dense_* is NOT value corrected. A value-corrected dense array was not dumped (ALLV / SHV exist only as the top-64 keys, mass x ||v||), so the value-corrected last-key mass is given only for the fixed top-64 set (V64).
Parts (selected rows = the rows of the mask file: top 10 % eligible rows by window gain, band 128 far keys):
 1 last (N-th) key of the p1 / p3 / p10 set: its mass, ALL and SH; mean, quartiles; also the share of the row's far mass held by the set; for p1 (N <= 82 <= 128, inside the stored top-128) the number of heads with probability >= each of .005/.01/.02/.05/.1/.2 at the last key (stored counts; p3 / p10 sets lie beyond the stored top-128 so these counts do not exist for them)
 2 threshold sets {far keys with summed mass >= tau}, tau in .005 .01 .02 .05 .1 .2 (the SUMMED mass, not a per-head probability): keys per row and share of the far context (mean, quartiles, share of rows with none)
 3 overlap of the p-sets with the threshold sets (Jaccard and containment of the smaller in the larger, mean over rows), and of ALL with SH at each p
  .venv/bin/python scripts/exp07_write_ladder/e57_shadowkey_mass_analysis.py --exp EXP_DIR --out OUT.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_shadowkey_masks as SKM
WIN=SKM.WIN;BAND=128;TAUS=(.005,.01,.02,.05,.1,.2);PS=(('p1',.01),('p3',.03),('p10',.10))
def q(x):
    x=np.asarray(x,float);return None if len(x)==0 else dict(mean=float(x.mean()),q25=float(np.percentile(x,25)),median=float(np.median(x)),q75=float(np.percentile(x,75)),n=int(len(x)))
def thr_set(row,i):
    lo,hi=SKM.far_range(i,BAND);f=np.arange(lo,hi+1);return f[row[lo:hi+1].astype(np.float32)>=TAU_CUR[0]]
TAU_CUR=[0.0]
def set_stats(keys_a,keys_b):
    a=set(keys_a.tolist());b=set(keys_b.tolist());i=len(a&b);u=len(a|b);return (i/u if u else np.nan),(i/min(len(a),len(b)) if min(len(a),len(b)) else np.nan)
def analyse(chunk_data,band=BAND):
    """chunk_data: list of dict(k, rows, D{'ALL','SH'} dense [1024,8192], hk{arm: int16 [R,L]}, cnt{'ALL','SH'}: (idx [1024,128], cnt [1024,128,C]) or None) -> dict"""
    R={'last_mass':{},'far_share_held':{},'last_heads_ge':{},'thr_keys':{},'thr_share':{},'thr_none':{},'overlap_p_thr':{},'overlap_all_sh':{}}
    acc={}
    def add(path,v):acc.setdefault(path,[]).append(v)
    for c in chunk_data:
        rows=c['rows']
        for src in ('ALL','SH'):
            D=c['D'][src].astype(np.float32)
            for pn,f in PS:
                hk=c['hk'][f'{src}{pn}']
                for r,i in enumerate(rows):
                    i=int(i);h=hk[r][hk[r]>=0].astype(int);lo,hi=SKM.far_range(i,band);nfar=hi-lo+1
                    add(f'last_mass|{src}|{pn}',float(D[i,h[-1]]));add(f'far_share_held|{src}|{pn}',float(D[i,h].sum()/D[i,lo:hi+1].sum()))
                    if pn=='p1' and c['cnt'].get(src) is not None:
                        I,C=c['cnt'][src];pos=np.nonzero(I[i]==h[-1])[0]
                        if len(pos):
                            C_=C[i,pos[0]];e=C_[-6:]
                            for j,t in enumerate(TAUS):add(f'last_heads_ge|{src}|{t}',int(e[j]))
            for r,i in enumerate(rows):
                i=int(i);lo,hi=SKM.far_range(i,band);nfar=hi-lo+1
                for t in TAUS:
                    TAU_CUR[0]=t;ts=thr_set(D[i],i);add(f'thr_keys|{src}|{t}',len(ts));add(f'thr_share|{src}|{t}',len(ts)/nfar)
                    for pn,f in PS:
                        hk=c['hk'][f'{src}{pn}'];h=hk[r][hk[r]>=0].astype(int)
                        if len(ts)==0:continue
                        j,cont=set_stats(h,ts);add(f'jac|{src}|{pn}|{t}',j);add(f'cont|{src}|{pn}|{t}',cont)
        for pn,f in PS:
            for r,i in enumerate(rows):
                ha=c['hk'][f'ALL{pn}'][r];ha=ha[ha>=0];hs=c['hk'][f'SH{pn}'][r];hs=hs[hs>=0];j,cont=set_stats(ha,hs);add(f'allsh_jac|{pn}',j)
    out={}
    for path,v in acc.items():
        if path.startswith('thr_none'):continue
        out[path]=q(v)
    for src in ('ALL','SH'):
        for t in TAUS:
            v=np.asarray(acc[f'thr_keys|{src}|{t}']);out[f'thr_none|{src}|{t}']=float((v==0).mean())
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--exp');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    E=pathlib.Path(a.exp);M=np.load(E/'e57_shadowkey_masks.npz');man=json.loads((E/'e57_attdump3'/'e57_attdump3_manifest.json').read_text())['chunk_to_shard'];data=[]
    for k in [int(x) for x in M['chunks']]:
        Zs=np.load(E/'e57_attdump3'/f'e57_attdump3_s{man[str(k)]:02d}.npz');cnt={}
        for src in ('ALL','SH'):cnt[src]=(Zs[f'W0/c{k}/{src}{BAND}_idx'].astype(np.int64),Zs[f'W0/c{k}/{src}{BAND}_cnt'])
        data.append(dict(k=k,rows=M[f'rows_{k}'],D={s:Zs[f'W0/c{k}/dense_{s}'] for s in('ALL','SH')},hk={n:M[f'hk_{n}_{k}'] for n in [str(x) for x in M['names']]},cnt=cnt));print('chunk',k,flush=True)
    r=analyse(data);json.dump(r,open(a.out,'w'))
    for key in sorted(r):print(key,r[key] if not isinstance(r[key],dict) else {kk:round(v,4) for kk,v in r[key].items()})
def selftest():
    rng=np.random.default_rng(0);D=rng.random((1024,WIN)).astype(np.float16)*.3;rows=np.array([500,900]);hk={}
    for src in('ALL','SH'):
        for pn,f in PS:
            L=[]
            for i in rows:
                n,N=SKM.sizes(int(i),BAND,f);L.append(SKM.select(src,int(i),N,BAND,score_row=D[int(i)]))
            m=max(len(x) for x in L);a=np.full((2,m),-1,np.int16)
            for r,x in enumerate(L):a[r,:len(x)]=x
            hk[f'{src}{pn}']=a
    I=np.argsort(-D.astype(np.float32),1)[:,:128];C=np.zeros((1024,128,10),np.uint8)
    o=analyse([dict(k=16,rows=rows,D={'ALL':D,'SH':D},hk=hk,cnt={'ALL':(I,C),'SH':(I,C)})])
    n,N=SKM.sizes(500,BAND,.01);assert o['last_mass|ALL|p1']['n']==2 and o['allsh_jac|p1']['mean']==1.0
    ts=lambda t:D[500,4:500+8192-BAND+1].astype(np.float32)>=t
    assert abs(o['thr_keys|ALL|0.005']['mean']-np.mean([ (D[i,SKM.far_range(int(i),BAND)[0]:SKM.far_range(int(i),BAND)[1]+1].astype(np.float32)>=.005).sum() for i in rows]))<1e-9
    assert o['cont|ALL|p10|0.02']['mean']==1.0
    print('e57_shadowkey_mass_analysis selftest ok')
if __name__=='__main__':main()
