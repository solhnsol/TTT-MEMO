"""E57 T1/T2 v2 low mask budget (job 585; GPU; new file; post-hoc, exploratory): GAINQ128 p=.10 vs RANDQ128 p=.10 on the SAME frozen v2 setup as jobs 570 / 575 (e57_t1t2v2_run.py read stage; norm targets of the NAT arm of 570 on B20).
Stream A: the existing chunk-279 states of GAINQ128 p=.10 (e57_gainq_p0.10) and RANDQ128 p=.10 (e57_randq_p0.10), SELC1 engine, Adam direction, 0.5 x NAT norm of E57_main_NAT, are only read.
Stream B20: stage mask = from the ALREADY SAVED W0 score file of job 575 (E57_v2B20_score_W0.npz, no new scoring pass) the top p of the eligible rows per chunk 8..285 by nll_band128 - nll_win (GAINQ128P10) and a random p of the eligible rows (RANDQ128P10; one default_rng(SEED) stream over the chunks) -> mask file; stage write = SELC1 + e57_qband_run.SelMaskQ, one cycle per arm, end canary (gen / tail rise) logged per arm. Read = e57_t1t2v2_run.py --stage read.
  python3 e57_t1t2v2_lowp_run.py --stage mask|write --sentences S.jsonl --score W0.npz --out O --nat-ref NATB20.arms.json [--p 0.10] [--arms GAINQ128P10,RANDQ128P10]  |  --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_t1t2v2_run as V
import e57_t1t2v2_gainq_run as G
CH=1024;WIN_=8*CH;K0=8;BAND=128;SEED=585
def build_lowp(Z,sid,p,seed=SEED):
    nch=Z['nll_win'].shape[0];rng=np.random.default_rng(seed);out={'names':np.asarray(['GAINQ128P10','RANDQ128P10']),'p':np.asarray(p),'chunks':np.asarray([K0,nch])}
    for a in ('GAINQ128P10','RANDQ128P10'):out[f'band_{a}']=np.asarray(BAND);out[f'all_{a}']=np.asarray(0)
    for k in range(K0,nch):
        el=CM.rows_eligible(Z['lm'][k],sid[k]);n=int(round(p*el.sum()))
        out[f'qb_GAINQ128P10_{k}']=CM.pk(CM.top_rows(Z['nll_band'][k]-Z['nll_win'][k],el,n));out[f'qb_RANDQ128P10_{k}']=CM.pk(CM.rand_rows(el,n,rng))
    return out
def mask_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);Z=np.load(a['score']);nch=Z['nll_win'].shape[0];assert nch==286 and int(Z['band'])==BAND
    M=build_lowp(Z,G.frag_sids(a['sentences'],nch),float(a.get('p') or 0.10));np.savez_compressed(out/'e57_v2B20_lowp_masks.npz',**M)
    for arm in ('GAINQ128P10','RANDQ128P10'):
        rows=[int(CM.up(M[f'qb_{arm}_{k}'],CH).sum()) for k in range(K0,nch)];print(f'{arm}: mean banded rows per chunk {np.mean(rows):.1f} (min {min(rows)} max {max(rows)})',flush=True)
    g=np.mean([ (CM.up(M[f'qb_GAINQ128P10_{k}'],CH)&CM.up(M[f'qb_RANDQ128P10_{k}'],CH)).sum()/max(1,CM.up(M[f'qb_GAINQ128P10_{k}'],CH).sum()) for k in range(K0,nch)]);print('overlap GAINQ/RANDQ rows %.3f'%g,flush=True)
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_run as R,e57_qband_run as QR
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    ids=V.stream_ids(a['sentences'],'B20');nch=len(ids)//CH;masks={k:v for k,v in np.load(out/'e57_v2B20_lowp_masks.npz').items()}
    ntg=list(json.load(open(a['nat_ref']))['norm_targets'][:nch]);assert len(ntg)==nch;rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);del L0;gc.collect();LD.jax.clear_caches()
    for arm in a['arms']:
        fnA=out/f'E57_v2B20_{arm}.arms.json'
        if fnA.exists() and not a.get('rerun'):print('exists, reused:',fnA);continue
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_v2B20_{arm}',canary=None);L.rec_cache=rc;traj=[]
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream('SELC1',1,ids,nch=nch,tag=f'v2B20{arm}',log=lg,read_hook=hook,hook_chunks={nch-1},read_chunks={nch-1},chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=QR.SelMaskQ(arm,masks,lg));log=res['log']
        fnA.write_text(json.dumps(dict(arm=arm,engine='SELC1',stream='B20',n_chunks=nch,ids_sha=V.STREAM_SHA['B20'],canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','q_banded','q_band') if k_ in e} for e in log]),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def selftest():
    rng=np.random.default_rng(0);nch=10;sid=np.zeros((nch,CH),np.int32)
    Z={'nll_win':rng.random((nch,CH)),'lm':np.ones((nch,CH),bool)};Z['nll_band']=Z['nll_win']+rng.random((nch,CH))
    M=build_lowp(Z,sid,0.10);M2=build_lowp(Z,sid,0.10);el=CM.rows_eligible(Z['lm'][9],sid[9]);n=int(round(.10*el.sum()))
    for a in ('GAINQ128P10','RANDQ128P10'):assert CM.up(M[f'qb_{a}_9'],CH).sum()==n and not CM.up(M[f'qb_{a}_9'],CH)[~el].any()
    assert all(np.array_equal(M[k],M2[k]) for k in M if k!='names') and not np.array_equal(M['qb_GAINQ128P10_9'],M['qb_RANDQ128P10_9'])
    import e57_qband_run as QR;s=QR.SelMaskQ('RANDQ128P10',M,log=lambda *x:None);nat=np.ones(WIN_+CH,bool);kp,info=s.keep(9,None,None,None,None,nat);assert info['q_banded']==n and info['q_band']==128
    print('e57_t1t2v2_lowp_run selftest ok')
def main():
    stage=G._pop('--stage');a=dict(sentences=G._pop('--sentences'),out=G._pop('--out'),nat_ref=G._pop('--nat-ref'),p=G._pop('--p'),score=G._pop('--score'),arms=(G._pop('--arms') or 'GAINQ128P10,RANDQ128P10').split(','))
    if stage=='mask':mask_stage(a)
    elif stage=='write':write_stage(a)
    else:raise SystemExit('--stage mask|write')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:main()
