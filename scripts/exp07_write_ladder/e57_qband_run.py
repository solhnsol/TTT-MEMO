"""E57 query-band write run (job 291 / 292; GPU, one news cycle, 280 chunks; post-hoc, exploratory). Engine and step sizes of the pod run (e57_pod_run.py: SELC1 engine, Adam direction, size 0.5 x the NAT update norm of the same chunk, norm_targets from E57_main_NAT.arms.json), but the mask object is SelMaskQ: for chunks k >= 8 the rows of the mask file (e57_qband_masks.py) see only the previous band-k tokens (e57_closure_masks.band_keep, window AND chunk keys, prefix AND suffix layers through the per-query keep), all other rows see the natural window; arm NATQ = no mask (NATADAMH of the SELC1 engine). Stage 'reads' is the unchanged stage of e57_main_run.py (OUT / IN reads and canary at the cycle end).
  python3 e57_qband_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 10 --arms ALLQ128,... --qband-masks M.npz --ntg-from E57_main_NAT.arms.json [--smoke N] --stage arms|reads   |   --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
CH=1024;WIN=8192;K0=8
QB={'masks':None,'ntg':None}
def _pop(flag):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
class SelMaskQ:
    """UV-compatible mask object (.keep) for the SELC1 engine: returns (per-query keep [CH, WIN+CH], info); chunks < K0 and arm NATQ: the natural 1-D keep"""
    def __init__(self,arm,masks,log=print):
        self.arm=arm;self.m=masks;self.log=log;self.none=(arm=='NATQ')
        if not self.none:self.band=int(masks[f'band_{arm}']);self.allq=bool(int(masks[f'all_{arm}']))
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if self.none or k<K0:return base,{'uv_arm':self.arm,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        qb=np.ones(CH,bool) if self.allq else CM.up(self.m[f'qb_{self.arm}_{k}'],CH)
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();bk=CM.band_keep(base,self.band);kp[qb]=bk[qb]
        if k%16==0:self.log(f'  {self.arm} chunk {k}: banded rows {int(qb.sum())} of {CH} (band {self.band})')
        return kp,{'uv_arm':self.arm,'q_banded':int(qb.sum()),'q_band':self.band,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def qband_stage_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    assert N==1,'one cycle'
    masks={k:v for k,v in np.load(QB['masks']).items()}
    ntg=json.load(open(QB['ntg']))['norm_targets'];assert ntg is not None and len(ntg)>=nch;ntg=list(ntg[:nch])
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_qband_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V of the stream recorded ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    mk=lambda arm,log:SelMaskQ(arm,masks,log)
    if a.smoke>0:
        for arm in arms:
            Ls=C.Ladder55(rt,save_dir=out/'states'/f'E57_qband_smoke_{arm}',canary=None);Ls.rec_cache=rc
            rs=Ls.run_stream('SELC1',1,ids_all,nch=a.smoke,tag=f'smoke{arm}',log=lambda *x:None,chunks_meta=dummy_meta,norm_targets=ntg,uv=mk(arm,lambda *x:None))
            lg(f'  smoke {arm}: {a.smoke} chunks ok, mean update norm {np.mean([e["update_norm"] for e in rs["log"]]):.3g}, mean ce {np.mean([e["ce"] for e in rs["log"]]):.3f}');del Ls,rs;gc.collect()
    for arm in arms:
        fnA=out/f'E57_main_{arm}.arms.json'
        if fnA.exists() and not a.rerun:print('exists, reused:',fnA);continue
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_main_{arm}',canary=None);L.rec_cache=rc;traj=[]
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),cycle=k//n+1,canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream('SELC1',1,ids_all,nch=nch,tag=f'qb{arm}',log=lg,read_hook=hook,hook_chunks=set(c for c in cch if c<nch),read_chunks=set(c for c in rch if c<nch),chunks_meta=dummy_meta,norm_targets=ntg,uv=mk(arm,lg));log=res['log']
        fnA.write_text(json.dumps(dict(arm=arm,engine='SELC1',qband=True,bank=bsrc,cycles=N,n_chunks=nch,canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','q_banded','q_band') if k_ in e} for e in log],norm_targets=None,rs_targets=None,read_positions_in_cycle=kks),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def main():
    QB['masks']=_pop('--qband-masks');QB['ntg']=_pop('--ntg-from')
    import e57_main_run as M
    names=[]
    if QB['masks']:names=[str(x) for x in np.load(QB['masks'])['names']]
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(names)+('NATQ',);M.stage_arms=qband_stage_arms
    if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='arms':assert QB['masks'] and QB['ntg'],'--qband-masks and --ntg-from are required for the arms stage'
    M.main()
def selftest():
    rng=np.random.default_rng(0);nat=np.ones(WIN+CH,bool);nat[:50]=False
    qb=rng.random(CH)<.3;M={'band_X128':np.asarray(128),'all_X128':np.asarray(0),'qb_X128_9':CM.pk(qb),'band_A512':np.asarray(512),'all_A512':np.asarray(1)}
    s=SelMaskQ('X128',M,log=lambda *x:None);kp,info=s.keep(9,None,None,None,None,nat);assert kp.shape==(CH,WIN+CH) and info['q_banded']==int(qb.sum())
    bk=CM.band_keep(nat,128);assert np.array_equal(kp[qb],bk[qb]) and kp[~qb].all(axis=None)==False and np.array_equal(kp[~qb],np.broadcast_to(nat,(int((~qb).sum()),WIN+CH)))
    k0,_=s.keep(3,None,None,None,None,nat);assert k0.ndim==1 and np.array_equal(k0,nat)
    a,_=SelMaskQ('A512',M,log=lambda *x:None).keep(9,None,None,None,None,nat);assert np.array_equal(a,CM.band_keep(nat,512))
    n,_=SelMaskQ('NATQ',M).keep(9,None,None,None,None,nat);assert n.ndim==1
    print('e57_qband_run selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:main()
