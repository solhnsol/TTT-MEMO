"""E57 shadowing-key closure report (CPU only; reads E57_shadowkey_W0.npz of e57_shadowkey_run.py and the mask file; numbers only, no criterion). Per arm: closure = sum over the selected rows of (nll_mask - nll_win) / sum of (nll_chunk - nll_win) (pooled per chunk, then ratio of sums over chunks with a bootstrap over chunks, 5000 draws, seed 0); also the pooled mean of nll_mask - nll_win and of nll_chunk - nll_win (nats per token) over the selected rows and the number of rows. Row i of a chunk predicts token i+1, nll arrays are indexed by row.
  python3 e57_shadowkey_report.py --res E57_shadowkey_W0.npz --masks M.npz --out R.json | --selftest"""
import sys,json,argparse
import numpy as np
def boot(num,den,rng,nb=5000):
    num=np.asarray(num,float);den=np.asarray(den,float);ix=rng.integers(0,len(num),(nb,len(num)));b=num[ix].sum(1)/den[ix].sum(1);return [float(num.sum()/den.sum()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]
def report(R,M):
    rng=np.random.default_rng(0);names=[str(x) for x in M['names']];ch=[int(x) for x in M['chunks']];out={}
    for ai,nm in enumerate(names):
        num=[];den=[];dm=[];dc=[];n=0
        for k in ch:
            rows=np.asarray(M[f'rows_{k}'],int);w=R['nll_win'][k][rows];num.append(float((R['nll_mask'][ai][k][rows]-w).sum()));den.append(float((R['nll_chunk'][k][rows]-w).sum()));n+=len(rows)
        out[nm]=dict(rows=n,chunks=len(ch),closure=boot(num,den,rng),mask_minus_win=float(sum(num)/n),chunk_minus_win=float(sum(den)/n))
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--res');ap.add_argument('--masks');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    o=report(np.load(a.res),np.load(a.masks));open(a.out,'w').write(json.dumps(o))
    for k,v in o.items():print(f'{k:10s} rows {v["rows"]} closure {v["closure"][0]:.3f} [{v["closure"][1]:.3f},{v["closure"][2]:.3f}]  mask-win {v["mask_minus_win"]:.3f}  chunk-win {v["chunk_minus_win"]:.3f}')
def selftest():
    rng=np.random.default_rng(0);nll_w=rng.random((280,1024));nll_c=nll_w+1.0;nm=np.stack([nll_w+.25,nll_w+.5]);R=dict(nll_win=nll_w,nll_chunk=nll_c,nll_mask=nm)
    M=dict(names=np.asarray(['A','B']),chunks=np.asarray([16,32,48]),**{f'rows_{k}':np.arange(100) for k in (16,32,48)})
    o=report(R,M);assert abs(o['A']['closure'][0]-.25)<1e-9 and abs(o['B']['closure'][0]-.5)<1e-9 and o['A']['rows']==300 and o['A']['closure'][1]<=.25<=o['A']['closure'][2]
    print('e57_shadowkey_report selftest ok')
if __name__=='__main__':main()
