"""E57 query-band arms written again with states at several chunks (jobs 401ff; GPU, one news cycle; post-hoc, exploratory; new file). The arms GAINQ512 (p = .26 rows of the mask file), RANDQ512 (p = .10 rows) and NATADAMH (no mask, engine NATADAMH = Adam direction x 0.5 x the NAT update norm of the chunk) are written for one cycle (280 chunks) exactly like e57_qband_run (engine SELC1 for the masked arms, SelMaskQ per-query band keep for chunks >= 8, norm_targets of E57_main_NAT.arms.json) but with --read-at positions (default 99,179,279): the weights are saved at those chunks, the canary is read there (--canary-npos), and the 42 questions of bank v1 are read OUT only (8192-token neutral tail, like e57_curve_run) from every saved state and from W0. Output: <out>/E57_main_<ARM>.arms.json (states: {chunk: {file, hash}}, trajectory, update_log), <out>/E57_qcurve_scores.json (rows of e57_reads.log_rows, cycle 1, read_chunk = chunk, condition OUT).
  python3 e57_qcurve_run.py --bundle B --out O --cycles 1 --read-at 99,179,279 --canary-npos 1 --arms NATADAMH,GAINQ512,RANDQ512 --qband-masks M.npz --ntg-from E57_main_NAT.arms.json --stage arms|reads   |   --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_qband_run as QR
CH=1024
QC={'masks':None,'ntg':None}
def _pop(flag):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
def engine_of(arm):return 'NATADAMH' if arm=='NATADAMH' else 'SELC1'
def qcurve_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    assert N==1,'one cycle'
    masks={k:v for k,v in np.load(QC['masks']).items()};ntg=list(json.load(open(QC['ntg']))['norm_targets'][:nch]);assert len(ntg)==nch
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_qcurve_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V of the stream recorded ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    uv_of=lambda arm,log:None if arm=='NATADAMH' else QR.SelMaskQ(arm,masks,log)
    if a.smoke>0:
        for arm in arms:
            Ls=C.Ladder55(rt,save_dir=out/'states'/f'E57_qcurve_smoke_{arm}',canary=None);Ls.rec_cache=rc
            rs=Ls.run_stream(engine_of(arm),1,ids_all,nch=a.smoke,tag=f'smoke{arm}',log=lambda *x:None,chunks_meta=dummy_meta,norm_targets=ntg,uv=uv_of(arm,lambda *x:None))
            lg(f'  smoke {arm}: {a.smoke} chunks ok, mean update norm {np.mean([e["update_norm"] for e in rs["log"]]):.3g}, mean ce {np.mean([e["ce"] for e in rs["log"]]):.3f}');del Ls,rs;gc.collect()
    for arm in arms:
        fnA=out/f'E57_main_{arm}.arms.json'
        if fnA.exists() and not a.rerun:print('exists, reused:',fnA);continue
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_main_{arm}',canary=None);L.rec_cache=rc;traj=[]
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),cycle=k//n+1,canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream(engine_of(arm),1,ids_all,nch=nch,tag=f'qc{arm}',log=lg,read_hook=hook,hook_chunks=set(c for c in cch if c<nch),read_chunks=set(c for c in rch if c<nch),chunks_meta=dummy_meta,norm_targets=ntg,uv=uv_of(arm,lg));log=res['log']
        assert set(res['reads'])>=set(c for c in rch if c<nch),('weights at some positions were not saved',sorted(rch),sorted(res['reads']))
        fnA.write_text(json.dumps(dict(arm=arm,engine=engine_of(arm),qband=arm!='NATADAMH',bank=bsrc,cycles=N,n_chunks=nch,canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','q_banded','q_band') if k_ in e} for e in log],norm_targets=None,rs_targets=None,read_positions=sorted(int(c) for c in rch)),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def qcurve_reads(a,out,items,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    import e55_gpu as RG,e57_reads as RD,e57_main_run as M
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)})
    out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);P=rd.packed(raw,out_tail);csa={it['id']:it['q'].get('c_SA') for it in items};lg(f'OUT tail {len(out_tail)} tokens; {len(raw)} candidate rows; scorer built')
    def score(m):
        m32=rd.to_fp32(m)
        with LD.jax.default_matmul_precision('highest'):return RD.split_scores(np.asarray(P.r1(m32)),items)
    live=M.inner_dtypes(LD,rd,rt.base);rows=[];tim={}
    def read(arm,kk,m):
        t0=time.monotonic();rows.extend(RD.log_rows(items,score(m),ntok,cycle=0 if arm=='W0' else 1,read_chunk=kk,pos_tokens=(kk+1)*CH,win=None,arm=arm,condition='OUT',iv=iv,hits=hits,csa=csa,shadow=None));tim[f'{arm}@{kk}']=time.monotonic()-t0
    read('W0',0,rt.base);lg(f'  W0 read: {tim["W0@0"]:.0f}s');pos=None
    for arm in arms:
        A=json.load(open(out/f'E57_main_{arm}.arms.json'));ks=sorted(int(k) for k in A['states']);pos=ks if pos is None else pos;assert ks==pos,(arm,ks,pos)
        for k_str,st in sorted(A['states'].items(),key=lambda x:int(x[0])):
            m=rd.load_state(st['file'],expect_hash=st['hash']);dt=M.inner_dtypes(LD,rd,m);assert dt==live,('precision',arm,k_str,dt,live)
            read(arm,int(k_str),m);del m;lg(f'  {arm} read at chunk {k_str}: {tim[f"{arm}@{k_str}"]:.0f}s')
        (out/'E57_qcurve_scores_partial.json').write_text(json.dumps(dict(done_arms=[arm],rows=rows),default=float))
    assert all(len([r for r in rows if r['arm']==arm and r['question']==it['id']])==len(pos) for arm in arms for it in items),'a state x question row is missing'
    (out/'E57_qcurve_scores.json').write_text(json.dumps(dict(meta=dict(bank=bsrc,context='OUT (8192-token neutral tail of job 261)',positions=pos,arms=list(arms),timings=tim),rows=rows),default=float));lg('wrote E57_qcurve_scores.json')
def selftest():
    assert engine_of('NATADAMH')=='NATADAMH' and engine_of('GAINQ512')=='SELC1' and engine_of('RANDQ512')=='SELC1'
    QR.selftest();print('e57_qcurve_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    QC['masks']=_pop('--qband-masks');QC['ntg']=_pop('--ntg-from')
    import e57_main_run as M
    names=[str(x) for x in np.load(QC['masks'])['names']] if QC['masks'] else []
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(names);M.stage_arms=qcurve_arms;M.stage_reads=qcurve_reads
    if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='arms':assert QC['masks'] and QC['ntg']
    M.main()
if __name__=='__main__':main()
