"""E57 plain-SGD (NAT direction) write runs at larger step sizes (job 540; GPU, one news cycle, no mask, no Adam; post-hoc, exploratory; new file, same stage structure as e57_adamdecomp_run.decomp_arms). Arms NATX2 / NATX4 = e55_core arm NATM ("size-matched NAT": the NAT update of chunk k, clipped gradient direction, rescaled to the update norm norm_targets[k]) with norm_targets = 2 x / 4 x the per-chunk update norms of the saved NAT run (E57_main_NAT.arms.json), i.e. the same norm convention as the 0.5 x NAT norm of the Adam / momentum arms. The direction is the NAT gradient direction at the CURRENT weights (not the weights of the saved NAT run). States are saved at --read-at (99, 179, 279), canary there, OUT reads of the 42 questions by the unchanged reads stage of e57_adamdecomp_run (E57_qcurve_scores.json).
  python3 e57_natsgd_run.py <the arguments of e57_adamdecomp_run.py with --arms NATX2,NATX4>   |   --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_adamdecomp_run as AD
FACT={'NATX2':2.0,'NATX4':4.0};CH=1024
def selftest():
    AD.selftest();assert FACT=={'NATX2':2.0,'NATX4':4.0};print('e57_natsgd_run selftest ok')
def natsgd_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    assert N==1,'one cycle'
    ntg0=list(json.load(open(AD.AD['ntg']))['norm_targets'][:nch]);assert len(ntg0)==nch
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_natsgd_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V of the stream recorded in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    for arm in arms:
        fnA=out/f'E57_main_{arm}.arms.json'
        if fnA.exists() and not a.rerun:print('exists, reused:',fnA);continue
        ntg=[FACT[arm]*x for x in ntg0]
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_main_{arm}',canary=None);L.rec_cache=rc;traj=[]
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),cycle=k//n+1,canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream('NATM',1,ids_all,nch=nch,tag=f'sg{arm}',log=lg,read_hook=hook,hook_chunks=set(c for c in cch if c<nch),read_chunks=set(c for c in rch if c<nch),norm_targets=ntg,chunks_meta=dummy_meta);log=res['log']
        assert set(res['reads'])>=set(c for c in rch if c<nch),('weights at some positions were not saved',sorted(rch),sorted(res['reads']))
        fnA.write_text(json.dumps(dict(arm=arm,engine='NATM',factor=FACT[arm],bank=bsrc,cycles=N,n_chunks=nch,canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','gnorm','scale') if k_ in e} for e in log],read_positions=sorted(int(c) for c in rch)),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def main():
    if '--selftest' in sys.argv:selftest();return
    import e55_swap as E
    for k in FACT:
        AD.NEW.setdefault(k,('NAT',0.5,False))
        if k not in E.ARMS:E.ARMS.append(k)      # names only for the CLI check of e57_main_run (these arms use e55_core arm NATM, no Adam)
    AD.decomp_arms=natsgd_arms;AD.main()
if __name__=='__main__':main()
