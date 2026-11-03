"""E57 synthetic s64: key-level masks for the arms KL16 / KL64 of e57_syn_mrun.py (CPU only; new file; method thread, Hans '두 팔 실행'): copy of syn_s64_m30_masks.npz (qb_TOP30_<k>, deciding rows) plus per chunk k = 8..127 and m = 16 / 64 the array qk_ALLV<m>_<k> [n_deciding, m] int16 = for every deciding row (in the order of np.where(dec)[0]) its top-m far keys by ALLV (W0 key-mass dump, band 256; window-key indices, -1 padded).
  python3 e57_syn_klmasks.py --out F.npz | --selftest"""
import sys,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        dec=np.zeros(CH,bool);dec[[3,7]]=True;assert CM.pk(dec) is not None;print('e57_syn_klmasks selftest ok');return
    M=dict(np.load(EXP/'e57_synq/syn_s64_m30_masks.npz',allow_pickle=True));out=dict(M)
    for j in range(8):
        D=np.load(EXP/f'e57_syn_s64/syn_s64_dump_j{j:02d}.npz')
        for k in [int(x) for x in D['chunks']]:
            dec=CM.up(M[f'qb_TOP30_{k}'],CH);rows=np.where(dec)[0];idx=D[f'c{k}/ALLV_b256_idx'][rows]
            for m in (16,64):out[f'qk_ALLV{m}_{k}']=idx[:,:m].astype(np.int16)
    np.savez_compressed(a.out,**out);print('wrote',a.out,len(out),'arrays')
if __name__=='__main__':main()
