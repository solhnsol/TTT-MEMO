"""E57 T1/T2 v2 band-512 arms on stream B20 (job 590; GPU; new file; post-hoc, exploratory): GAINQ512 p=.26 and RANDQ512 p=.10 on the frozen v2 setup of jobs 570 / 575 / 585 (norm targets of the NAT arm of 570 on B20).
Stage score = e57_t1t2v2_gainq_run.score_stage with the band set to 512 (forward-only W0 pass: natural nll_win and band-512 nll per row, both layers); stage mask = top p=.26 of the eligible rows per chunk 8..285 by nll_band512 - nll_win (GAINQ512P26) and a random p=.10 of the eligible rows (RANDQ512P10; default_rng(SEED)); stage write = e57_t1t2v2_lowp_run.write_stage (SELC1 + SelMaskQ, one cycle per arm, end canary in the arms.json trajectory).
  python3 e57_t1t2v2_b512_run.py --stage score|mask|write --sentences S.jsonl --out O [--nat-ref NATB20.arms.json] | --selftest"""
import sys,json,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_t1t2v2_gainq_run as G
import e57_t1t2v2_lowp_run as LP
BAND=512;K0=8;CH=1024;SEED=586
ARMS=('GAINQ512P26','RANDQ512P10');P_GAIN=0.26;P_RAND=0.10
def build(Z,sid,seed=SEED):
    nch=Z['nll_win'].shape[0];rng=np.random.default_rng(seed);out={'names':np.asarray(ARMS),'p':np.asarray([P_GAIN,P_RAND]),'chunks':np.asarray([K0,nch])}
    for a in ARMS:out[f'band_{a}']=np.asarray(BAND);out[f'all_{a}']=np.asarray(0)
    for k in range(K0,nch):
        el=CM.rows_eligible(Z['lm'][k],sid[k])
        out[f'qb_GAINQ512P26_{k}']=CM.pk(CM.top_rows(Z['nll_band'][k]-Z['nll_win'][k],el,int(round(P_GAIN*el.sum()))))
        out[f'qb_RANDQ512P10_{k}']=CM.pk(CM.rand_rows(el,int(round(P_RAND*el.sum())),rng))
    return out
def mask_stage(a):
    out=pathlib.Path(a['out']);Z=np.load(out/'E57_v2B20_score_W0.npz');nch=Z['nll_win'].shape[0];assert nch==286 and int(Z['band'])==BAND,(nch,int(Z['band']))
    M=build(Z,G.frag_sids(a['sentences'],nch));np.savez_compressed(out/'e57_v2B20_lowp_masks.npz',**M)      # file name read by the lowp write stage
    for arm in ARMS:
        rows=[int(CM.up(M[f'qb_{arm}_{k}'],CH).sum()) for k in range(K0,nch)];print(f'{arm}: mean banded rows per chunk {np.mean(rows):.1f} (min {min(rows)} max {max(rows)})',flush=True)
def selftest():
    rng=np.random.default_rng(0);nch=10;sid=np.zeros((nch,CH),np.int32)
    Z={'nll_win':rng.random((nch,CH)),'lm':np.ones((nch,CH),bool)};Z['nll_band']=Z['nll_win']+rng.random((nch,CH))
    M=build(Z,sid);M2=build(Z,sid);el=CM.rows_eligible(Z['lm'][9],sid[9])
    assert CM.up(M['qb_GAINQ512P26_9'],CH).sum()==int(round(.26*el.sum())) and CM.up(M['qb_RANDQ512P10_9'],CH).sum()==int(round(.10*el.sum())) and all(np.array_equal(M[k],M2[k]) for k in M if k!='names')
    import e57_qband_run as QR;s=QR.SelMaskQ('GAINQ512P26',M,log=lambda *x:None);nat=np.ones(8192+CH,bool);kp,info=s.keep(9,None,None,None,None,nat);assert info['q_band']==512
    print('e57_t1t2v2_b512_run selftest ok')
def main():
    stage=G._pop('--stage');a=dict(sentences=G._pop('--sentences'),out=G._pop('--out'),nat_ref=G._pop('--nat-ref'),arms=list(ARMS))
    if stage=='score':G.BAND=BAND;G.score_stage(a)
    elif stage=='mask':mask_stage(a)
    elif stage=='write':LP.write_stage(a)
    else:raise SystemExit('--stage score|mask|write')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:main()
