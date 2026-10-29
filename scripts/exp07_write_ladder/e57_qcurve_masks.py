"""E57 merged mask file of the arms GAINQ512 (p = .26) and RANDQ512 (p = .10) for job 401 (CPU only; new file): copies the keys of the arms GAINQ512 of e57_gainq_p0.26/e57_qband_masks.npz and RANDQ512 of e57_randq_p0.10/e57_qband_masks.npz (names, band_<ARM>, all_<ARM>, qb_<ARM>_<k>, chunks) into ONE npz that e57_qcurve_run.py reads; p is stored as p_<ARM>. Content hash = e57_qband_masks.content_det.
  python3 e57_qcurve_masks.py --gain GAINQ_FILE --rand RAND_FILE --out OUT.npz [--expect-det H] | --verify FILE [--expect-det H] | --selftest"""
import sys,argparse,tempfile,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_qband_masks as QM
def merge(gain,rand,out):
    G=np.load(gain);R=np.load(rand);o={};names=[]
    for Z,arm in ((G,'GAINQ512'),(R,'RANDQ512')):
        assert arm in [str(x) for x in Z['names']],(arm,[str(x) for x in Z['names']]);names.append(arm)
        for k in Z.files:
            if k in (f'band_{arm}',f'all_{arm}') or k.startswith(f'qb_{arm}_'):o[k]=Z[k]
        o[f'p_{arm}']=Z['p'];assert np.array_equal(Z['chunks'],[8,280])
    o['names']=np.asarray(names);o['chunks']=np.asarray([8,280]);np.savez_compressed(out,**o);return QM.content_det(out)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gain');ap.add_argument('--rand');ap.add_argument('--out');ap.add_argument('--verify');ap.add_argument('--expect-det');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    det,al=QM.content_det(a.verify) if a.verify else merge(a.gain,a.rand,a.out);print(f'content sha256 det={det} all={al}')
    if a.expect_det and a.expect_det!=det:print('MISMATCH');sys.exit(9)
def selftest():
    import e57_closure_masks as CM
    rng=np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as t:
        t=pathlib.Path(t)
        for fn,arm,p in (('g.npz','GAINQ512',.26),('r.npz','RANDQ512',.10)):
            d={f'band_{arm}':np.asarray(512),f'all_{arm}':np.asarray(0),'names':np.asarray([arm]),'p':np.asarray(p),'chunks':np.asarray([8,280])}
            for k in range(8,280):d[f'qb_{arm}_{k}']=CM.pk(rng.random(1024)<.3)
            np.savez_compressed(t/fn,**d)
        det,_=merge(t/'g.npz',t/'r.npz',t/'m.npz');z=np.load(t/'m.npz');assert [str(x) for x in z['names']]==['GAINQ512','RANDQ512'] and sum(1 for k in z.files if k.startswith('qb_GAINQ512_'))==272 and sum(1 for k in z.files if k.startswith('qb_RANDQ512_'))==272 and float(z['p_RANDQ512'])==.10
    print('e57_qcurve_masks selftest ok')
if __name__=='__main__':main()
