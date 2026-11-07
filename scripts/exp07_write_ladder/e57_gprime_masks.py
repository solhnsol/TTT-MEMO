"""E57 W0 suffix-band g' rule masks (new file; CPU; GPM thread, Hans 11/07 '온라인 쪽 실험부터'): key qtS2_<k> = eligible rows with g' = nll_suffixband128 - nll_win > 2 nats (W0, job 1210) of a synthetic world, added to the online mask file (which holds chunks and el_<k>) -> <out>. python3 e57_gprime_masks.py --seed S --online F --out F2 | --selftest"""
import sys,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));EXP=ROOT/'experiments/exp07_write_ladder'
import e57_closure_masks as CM
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--online');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:print('e57_gprime_masks selftest ok');return
    d=dict(np.load(a.online));Y=np.load(EXP/f'e57_suffixband/s{a.seed}/syn_s{a.seed}_nllsuf.npz');n=[]
    for k in range(int(d['chunks'][0]),int(d['chunks'][1])):
        el=CM.up(d[f'el_{k}'],CM.CH if hasattr(CM,'CH') else 1024);lm=Y['lm'][k];g=(Y['nll_suf'][0][k]-Y['nll_win'][k]).astype(np.float64);m=el&lm&(g>2);d[f'qtS2_{k}']=np.packbits(m);n.append(m.sum())
    np.savez(a.out,**d);print('qtS2 rows per chunk: mean %.1f'%np.mean(n))
if __name__=='__main__':main()
