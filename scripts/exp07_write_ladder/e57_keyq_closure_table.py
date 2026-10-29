"""E57 key-mask closure table (CPU only; new file; numbers only) from E57_keyq_closure_W0.npz (job 346: masked per-token NLL of the arms of the key-mask files on the GAINQ512 p=.26 rows, W0) and e57_band_W0.npz (band-512 and window NLL). closure = sum over the selected rows of (masked NLL - window NLL) / sum over the same rows of (band-512 NLL - window NLL), pooled over the 17 chunks (k % 16 == 0, 16..272) with the rows of qb_<arm>_<k> (the GAINQ512 rows); 95 % interval: bootstrap over the chunks. Also the mean (masked - window) and (band - window) per row.
  .venv/bin/python scripts/exp07_write_ladder/e57_keyq_closure_table.py --closure C.npz --band B.npz --masks M1.npz,M2.npz --out O.json"""
import sys,json,argparse
import numpy as np
sys.path.insert(0,__import__('pathlib').Path(__file__).resolve().parent.__str__())
import e57_closure_masks as CM
CH=1024
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--closure');ap.add_argument('--band');ap.add_argument('--masks');ap.add_argument('--out');ap.add_argument('--bandk',type=int,default=512);a=ap.parse_args()
    C=np.load(a.closure);B=np.load(a.band);M={}
    for f in a.masks.split(','):M.update({k:v for k,v in np.load(f,allow_pickle=True).items() if k.startswith('qb_')})
    arms=[str(x) for x in C['arms']];ch=[int(x) for x in C['chunks']];bi=list(B['bands']).index(a.bandk);rng=np.random.default_rng(0);out={}
    for ai,arm in enumerate(arms):
        num=[];den=[];nr=[]
        for ci,k in enumerate(ch):
            qb=CM.up(M[f'qb_{arm}_{k}'],CH)&C['lm'][ci];w=C['nll_win'][ci];m=C['nll_mask'][ai][ci];b=B['nll_band'][bi][k]
            num.append(float((m-w)[qb].sum()));den.append(float((b-w)[qb].sum()));nr.append(int(qb.sum()))
        num=np.array(num);den=np.array(den);nr=np.array(nr);cl=num.sum()/den.sum();bs=[]
        for _ in range(1000):
            i=rng.integers(0,len(ch),len(ch));bs.append(num[i].sum()/den[i].sum())
        out[arm]=dict(closure=float(cl),ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],rows=int(nr.sum()),mean_masked_minus_window=float(num.sum()/nr.sum()),mean_band_minus_window=float(den.sum()/nr.sum()))
    json.dump(dict(definition='closure = sum(masked-window)/sum(band%d-window) over the GAINQ512 p=.26 rows of the 17 chunks k%%16==0 (16..272)'%a.bandk,chunks=ch,arms=out),open(a.out,'w'),indent=1)
    for arm,v in out.items():print(f"{arm:26s} closure {v['closure']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] rows {v['rows']} masked-win {v['mean_masked_minus_window']:+.3f} band-win {v['mean_band_minus_window']:+.3f}")
if __name__=='__main__':main()
