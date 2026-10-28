"""E57 key-level query-mask write run (job 305 ff.; GPU, one news cycle, 280 chunks; post-hoc, exploratory). The run of e57_qband_run.py (SELC1 engine, Adam direction, 0.5 x the NAT update norm, reads at the cycle end) with a different mask object: for the arms of a key-mask file (e57_keyq_masks.py: key_<arm> = 1) the selected query rows (qb_<arm>_<k>, the GAINQ rows) hide ONLY the listed window keys (kq_<arm>_<k> int16 [nsel, 128], -1 padded, rows in ascending order) in prefix and suffix layers (per-query keep [CH, WIN+CH]); every other row, every other key and chunks < 8 are natural. Arms without key_<arm> behave as in e57_qband_run.py.
  python3 e57_keyq_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 10 --arms KQ512_ALL_topN_min,... --qband-masks M.npz --ntg-from E57_main_NAT.arms.json [--smoke N] --stage arms|reads   |   --selftest"""
import sys,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_qband_run as QR
CH=1024;WIN=8192;K0=8
class SelMaskKQ(QR.SelMaskQ):
    def __init__(self,arm,masks,log=print):
        self.kq=f'key_{arm}' in masks and int(masks[f'key_{arm}'])==1
        if self.kq:self.arm=arm;self.m=masks;self.log=log;self.none=False;self.band=int(masks[f'band_{arm}']);self.allq=False
        else:super().__init__(arm,masks,log)
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        if not self.kq:return super().keep(k,m,Kc,Vc,ids_k,keep_all)
        base=np.asarray(keep_all,bool)
        if k<K0:return base,{'uv_arm':self.arm,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        qb=CM.up(self.m[f'qb_{self.arm}_{k}'],CH);rows=np.nonzero(qb)[0];kq=np.asarray(self.m[f'kq_{self.arm}_{k}']).astype(np.int64);assert kq.shape[0]==len(rows),(kq.shape,len(rows))
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();nh=0
        for r,row in enumerate(rows):
            ii=kq[r];ii=ii[(ii>=0)&(ii<WIN)];kp[row,ii]=False;nh+=len(ii)
        if k%16==0:self.log(f'  {self.arm} chunk {k}: {len(rows)} query rows, mean hidden keys {nh/max(len(rows),1):.1f}')
        return kp,{'uv_arm':self.arm,'q_banded':int(len(rows)),'q_band':self.band,'uv_hidden_tokens':nh/max(len(rows),1),'uv_hidden_chunk_tokens':0}
def selftest():
    nat=np.ones(WIN+CH,bool);nat[:50]=False;rng=np.random.default_rng(0);qb=rng.random(CH)<.25;rows=np.nonzero(qb)[0]
    kq=np.full((len(rows),128),-1,np.int16)
    for r in range(len(rows)):n=rng.integers(0,60);kq[r,:n]=rng.choice(np.arange(4,7000),n,replace=False)
    M={'key_K':np.asarray(1),'band_K':np.asarray(512),'qb_K_9':CM.pk(qb),'kq_K_9':kq}
    s=SelMaskKQ('K',M,log=lambda *x:None);kp,info=s.keep(9,None,None,None,None,nat)
    exp=np.broadcast_to(nat,(CH,WIN+CH)).copy()
    for r,row in enumerate(rows):
        ii=kq[r][kq[r]>=0].astype(int);exp[row,ii]=False
    assert np.array_equal(kp,exp) and info['q_banded']==len(rows)
    assert np.array_equal(kp[~qb],np.broadcast_to(nat,(int((~qb).sum()),WIN+CH)))
    k0,_=s.keep(3,None,None,None,None,nat);assert k0.ndim==1
    M2={'band_X128':np.asarray(128),'all_X128':np.asarray(0),'qb_X128_9':CM.pk(qb)};b,_=SelMaskKQ('X128',M2,log=lambda *x:None).keep(9,None,None,None,None,nat);assert np.array_equal(b[qb],CM.band_keep(nat,128)[qb])
    print('e57_keyq_run selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        QR.SelMaskQ=SelMaskKQ;QR.main()
