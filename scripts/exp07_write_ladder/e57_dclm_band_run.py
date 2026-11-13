"""E57 DCLM preservation run with a configurable band length (new file, GPU; wrapper of the unchanged e57_dclm_ttt_run.py for the band-distance experiment). --band W replaces the registered ALLQ256 of the arm NATMASK by ALLQ<W> (band mask of W tokens on every row of the update path, mask from chunk K0 given by --k0) by patching the arm specification and the mask table of the unchanged runner; every other option is the option of e57_dclm_ttt_run.py (same streams, same engine, same hyper-parameters); the record of a stream carries mask_arm = ALLQ<W>. W0 and NATS do not depend on the band and are taken from job 1283d by the caller.
  python3 e57_dclm_band_run.py --band W --out O --k0 0 --n-streams 16 --docs-per-stream 8 --streams 0,1,...,15 --arms NATMASK | --selftest"""
import sys,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'e57'))
import e57_dclm_ttt_run as D,e57_qband_run as QR
def patch(W):
    spec=D.arm_spec;D.arm_spec=lambda arm:(f'ALLQ{W}',1.0) if arm=='NATMASK' else spec(arm)
    orig=QR.SelMaskQ
    def sel(ma,masks,lg=print):
        masks=dict(masks);masks[f'band_ALLQ{W}']=np.asarray(W);masks[f'all_ALLQ{W}']=np.asarray(1);return orig(ma,masks,lg)
    QR.SelMaskQ=sel
def selftest():
    patch(64);assert D.arm_spec('NATMASK')==('ALLQ64',1.0) and D.arm_spec('NATS')==('NATQ',1.0) and D.arm_spec('W0')==('NATQ',0.0)
    m=QR.SelMaskQ('ALLQ64',{'band_ALLQ256':np.asarray(256),'all_ALLQ256':np.asarray(1)},lambda *x:None);assert m.band==64 and m.allq is True
    print('e57_dclm_band_run selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--band',type=int);ap.add_argument('--stage',default='run');ap.add_argument('--out');ap.add_argument('--arms',default='NATMASK');ap.add_argument('--streams',default='0,1,2,3');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--rerun',action='store_true');ap.add_argument('--k0',type=int,default=8);ap.add_argument('--n-streams',type=int,default=D.NSTREAM);ap.add_argument('--docs-per-stream',type=int,default=D.DOCS);ap.add_argument('--run-type',default=None);ap.add_argument('--ref-dir');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        patch(a.band);D.run(a)
