"""E57 POD RUN (one news cycle, 280 chunks): masks of candidate 1 (C1) and of the learned head weighting (LRN) that act on the attention of the window AND of the current chunk, with their same-amount random controls (RC1, RLRN). Uses the engine and the step sizes of the arms of job 261 (Adam direction, size = 0.5 x the NAT update norm of the same chunk, the 'SELC1' engine) so the arms are comparable with the cycle-1 reads of NAT / NATADAMH / ORC / RAND of job 261 / 262.
Stages (the same process interface as e57_main_run.py; this file only replaces its arms stage and adds the arms C1 LRN RC1 RLRN (+ ORCA RORCA: the oracle answer mask of 261 extended to the chunk and its random control, mask stage --oracle-all)):
  python3 e57_mask_stage.py --bundle B --out O                                   (GPU forward only; writes O/e57_pod_masks.npz)
  python3 e57_pod_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms C1,LRN,RC1,RLRN --pod-masks O/e57_pod_masks.npz --ntg-from E57_main_NAT.arms.json [--guard 4] --stage arms
  python3 e57_pod_run.py ... --stage reads  [then  e57_main_report.py --dir O]
The NAT step norms (norm_targets, 1680 values) are the ones recorded by job 261 (E57_main_NAT.arms.json); the first 280 are those of a one-cycle NAT run (the update at chunk k depends on the weights before k only; the cycle-1 canary of job 262 equals that of job 261 to 4e-4)."""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
POD={'masks':None,'ntg':None,'guard':4,'ends_v2':False}
def _pop(flag,has_value=True):
    if flag in sys.argv:
        i=sys.argv.index(flag)
        if not has_value:del sys.argv[i];return True
        v=sys.argv[i+1];del sys.argv[i:i+2];return v
RANK={'C1':1,'LRN':2,'RC1':3,'RLRN':4,'ORCA':5,'RORCA':6,'ORC2':7,'RAND2':8}
def pod_stage_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import e55_uv as UVm,e57_mask_stage as MS
    CH=1024;assert N==1,'the pod run is one cycle'
    if POD['ends_v2']:
        import e57_orc2 as O2;ends=O2.sentence_ends_v2(UVm.tokenizer(tfile),ids_all)             # abbreviation-aware splitting (ORC v2 arms ORC2 / RAND2)
    else:ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids_all)
    masks={k:v for k,v in np.load(POD['masks']).items()}
    ntg=json.load(open(POD['ntg']))['norm_targets'];assert ntg is not None and len(ntg)>=nch,(None if ntg is None else len(ntg),nch);ntg=list(ntg[:nch])
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_pod_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V of the stream recorded ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    def mk(arm,log):return MS.SelMaskAll('SELC1',RANK[arm],masks,ends,UVm,log=log,guard=POD['guard'])
    if a.smoke>0:
        for arm in arms:
            Ls=C.Ladder55(rt,save_dir=out/'states'/f'E57_pod_smoke_{arm}',canary=None);Ls.rec_cache=rc
            rs=Ls.run_stream('SELC1',1,ids_all,nch=a.smoke,tag=f'smoke{arm}',log=lambda *x:None,chunks_meta=dummy_meta,norm_targets=ntg,uv=mk(arm,lambda *x:None))
            lg(f'  smoke {arm}: {a.smoke} chunks ok, mean update norm {np.mean([e["update_norm"] for e in rs["log"]]):.3g}, mean ce {np.mean([e["ce"] for e in rs["log"]]):.3f}, mean hidden window tokens {np.mean([e.get("uv_hidden_tokens",0) for e in rs["log"]]):.0f}, chunk tokens {np.mean([e.get("uv_hidden_chunk_tokens",0) for e in rs["log"]]):.0f}');del Ls,rs;gc.collect()
    for arm in arms:
        fnA=out/f'E57_main_{arm}.arms.json'
        if fnA.exists() and not a.rerun:print('exists, reused:',fnA);continue
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_main_{arm}',canary=None);L.rec_cache=rc;traj=[]
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),cycle=k//n+1,canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream('SELC1',1,ids_all,nch=nch,tag=f'pod{arm}',log=lg,read_hook=hook,hook_chunks=set(c for c in cch if c<nch),read_chunks=set(c for c in rch if c<nch),chunks_meta=dummy_meta,norm_targets=ntg,uv=mk(arm,lg));log=res['log']
        fnA.write_text(json.dumps(dict(arm=arm,engine='SELC1',mask_rank=RANK[arm],guard=POD['guard'],bank=bsrc,cycles=N,n_chunks=nch,canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','uv_hidden_tokens','uv_hidden_chunk_tokens') if k_ in e} for e in log],norm_targets=None,rs_targets=None,read_positions_in_cycle=kks),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def main():
    POD['ends_v2']=_pop('--ends-v2',False) is True;POD['masks']=_pop('--pod-masks');POD['ntg']=_pop('--ntg-from');g=_pop('--guard');POD['guard']=int(g) if g is not None else POD['guard']
    import e57_main_run as M
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(RANK);M.stage_arms=pod_stage_arms
    if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='arms':assert POD['masks'] and POD['ntg'],'--pod-masks and --ntg-from are required for the arms stage'
    M.main()
if __name__=='__main__':main()
