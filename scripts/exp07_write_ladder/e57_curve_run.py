"""E57 fine binding curve over the exposure of ONE cycle (GPU; post-hoc, exploratory; Hans 09:59Z approved in advance): the arms NAT (plain SGD), NATADAMH (Adam direction x0.5 NAT norm, no mask), ORC2 (ORC v2 mask, window + current chunk, per-query keep, guard 0, as job 282) and RAND2 (same hidden token counts, random fragments) are written for one cycle (280 chunks) with the weights saved at 14 positions (chunks 19, 39, ..., 279; the mean exposure n_k of a question grows from about 2 to about 34 articles), the canary is read at the same positions, and the 42 questions of bank v1 are read OUT only (the 8192-token SQuAD tail of job 261, bank-v1 forbid list; one scorer, rows as job 261: cycle 1, read_chunk = chunk, n_k = exposure at that position, accuracy / ranks over all candidates, hard-only, pool; summed log-prob, E49 convention). NAT / NATADAMH use the same engines as job 261; the NAT update norms of the 261 run (first 280) are the step-size targets of NATADAMH / ORC2 / RAND2. IN reads are not made (one scorer context per position would add about 8 minutes each).
  python3 e57_curve_run.py --bundle B --out O --cycles 1 --read-at 19,39,...,279 --canary-npos 1 --arms NAT,NATADAMH,ORC2,RAND2 --pod-masks O/e57_orca2_masks.npz --ntg-from E57_main_NAT.arms.json --stage arms | reads     |  --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
POD={'masks':None,'ntg':None}
RANK={'ORC2':7,'RAND2':8}
ENGINE={'NAT':'NAT','NATADAMH':'NATADAMH','ORC2':'SELC1','RAND2':'SELC1'}
POSITIONS=[19+20*j for j in range(14)]
def _pop(flag):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
def curve_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import e55_uv as UVm,e57_mask_stage as MS,e57_orc2 as O2,e57_main_run as M
    CH=1024;assert N==1 and nch==n and list(kks)==POSITIONS,(N,nch,kks)
    ends=O2.sentence_ends_v2(UVm.tokenizer(tfile),ids_all);masks={k:v for k,v in np.load(POD['masks']).items()}
    ntg=list(json.load(open(POD['ntg']))['norm_targets'][:nch]);assert len(ntg)==nch;dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_curve_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V recorded ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    def uv_of(arm,log):return MS.SelMaskAll('SELC1',RANK[arm],masks,ends,UVm,log=log,guard=0) if arm in RANK else None
    if a.smoke>0:
        for arm in arms:
            Ls=C.Ladder55(rt,save_dir=out/'states'/f'E57_curve_smoke_{arm}',canary=None);Ls.rec_cache=rc
            rs=Ls.run_stream(ENGINE[arm],1,ids_all,nch=a.smoke,tag=f'smoke{arm}',log=lambda *x:None,chunks_meta=dummy_meta,norm_targets=ntg,uv=uv_of(arm,lambda *x:None))
            lg(f'  smoke {arm}: {a.smoke} chunks ok, mean update norm {np.mean([e["update_norm"] for e in rs["log"]]):.3g}, mean ce {np.mean([e["ce"] for e in rs["log"]]):.3f}');del Ls,rs;gc.collect()
    for arm in arms:
        fnA=out/f'E57_main_{arm}.arms.json'
        if fnA.exists() and not a.rerun:print('exists, reused:',fnA);continue
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_main_{arm}',canary=None);L.rec_cache=rc;traj=[]
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),cycle=1,canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream(ENGINE[arm],1,ids_all,nch=nch,tag=f'curve{arm}',log=lg,read_hook=hook,hook_chunks=set(c for c in cch if c<nch),read_chunks=set(c for c in rch if c<nch),chunks_meta=dummy_meta,norm_targets=ntg if arm!='NAT' else None,uv=uv_of(arm,lg));log=res['log']
        assert set(res['reads'])>=set(POSITIONS),('weights at some positions were not saved',sorted(set(POSITIONS)-set(res['reads'])))
        fnA.write_text(json.dumps(dict(arm=arm,engine=ENGINE[arm],bank=bsrc,cycles=N,n_chunks=nch,canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','uv_hidden_tokens','uv_hidden_chunk_tokens') if k_ in e} for e in log],read_positions=POSITIONS),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def curve_reads(a,out,items,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
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
        t0=time.monotonic();rows.extend(RD.log_rows(items,score(m),ntok,cycle=0 if arm=='W0' else 1,read_chunk=kk,pos_tokens=(kk+1)*1024,win=None,arm=arm,condition='OUT',iv=iv,hits=hits,csa=csa,shadow=None));tim[f'{arm}@{kk}']=time.monotonic()-t0
    read('W0',0,rt.base);lg(f'  W0 read: {tim["W0@0"]:.0f}s')
    for arm in arms:
        A=json.load(open(out/f'E57_main_{arm}.arms.json'))
        for k_str,st in sorted(A['states'].items(),key=lambda x:int(x[0])):
            m=rd.load_state(st['file'],expect_hash=st['hash']);dt=M.inner_dtypes(LD,rd,m);assert dt==live,('precision',arm,k_str,dt,live)
            read(arm,int(k_str),m);del m;lg(f'  {arm} read at chunk {k_str}: {tim[f"{arm}@{k_str}"]:.0f}s')
        (out/'E57_curve_scores_partial.json').write_text(json.dumps(dict(done_arms=[arm],rows=rows),default=float))
    exp_ok=all(len([r for r in rows if r['arm']==arm and r['question']==it['id']])==len(POSITIONS) for arm in arms for it in items);assert exp_ok,'a state x question row is missing'
    (out/'E57_curve_scores.json').write_text(json.dumps(dict(meta=dict(bank=bsrc,context='OUT (8192-token SQuAD tail of job 261)',positions=POSITIONS,arms=list(arms),timings=tim,note='post-hoc; rows as job 261 (n_k = exposure at the read position)'),rows=rows),default=float));lg('wrote E57_curve_scores.json')
def selftest():
    assert POSITIONS[0]==19 and POSITIONS[-1]==279 and len(POSITIONS)==14 and all(b-a==20 for a,b in zip(POSITIONS,POSITIONS[1:]))
    assert set(ENGINE)=={'NAT','NATADAMH','ORC2','RAND2'} and RANK=={'ORC2':7,'RAND2':8}
    print('e57_curve_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    POD['masks']=_pop('--pod-masks');POD['ntg']=_pop('--ntg-from')
    import e57_main_run as M
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(RANK);M.stage_arms=curve_arms;M.stage_reads=curve_reads
    if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='arms':assert POD['masks'] and POD['ntg']
    M.main()
if __name__=='__main__':main()
