"""E57 T1/T2 cycle-2 experiment (GPU; post-hoc, exploratory): the arms ORC2 (ORC v2 rule R1+R2, partial forms only inside the articles of the question event, mask layout (window + current chunk, per-query keep, local guard 0) = job 282) and RAND2 (same hidden token counts, random fragments) are written for TWO cycles (560 chunks) from W0 and the weights are saved at the END OF CYCLE 1 (chunk 279) and the END OF CYCLE 2 (chunk 559). NAT: the saved states of job 261 at chunks 279 / 559 are the 2-cycle NAT run (the NAT update at chunk k depends on the weights before k only; the cycle-1 canary / update norms of the one-cycle re-runs agreed with job 261 to 1.7e-4), so NAT is not written again. The reads: the frozen 150 T1/T2 frames (e57_t1t2_v1.json, content hash e3897ebd...), OUT only (the 8192-token SQuAD tail of job 261, forbid list of bank v1), W0 and the six states (NAT / ORC2 / RAND2 at the end of cycle 1 and 2), with the scoring code and the row format of job 276 (e57_t1t2_run.rows_for_state): per item and state cand_lp (summed log-prob), cand_ntok, cand_surface, cand_kinds, cand_mean, group, event, id. Only raw scores are written; no rule, no threshold. A completeness check (every state x every item x non-empty lists of equal length) is part of the selftest and runs before the file is written.
Why the arms are written again from W0: the saved states hold the weights only (no Adam moments, no suffix KV state), so the cycle-1 states of job 282 cannot be continued exactly.
  python3 e57_t1t2c2_run.py --bundle B --out O --cycles 2 --npos 1 --canary-npos 4 --arms ORC2,RAND2 --pod-masks O/e57_orca2_masks.npz --ntg-from E57_main_NAT.arms.json --nat-arms E57_main_NAT.arms.json --frozen F --stage arms | reads      |  --selftest | --cpu-check --frozen F --nat-arms A --out O"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
POD={'masks':None,'ntg':None,'nat':None,'frozen':None,'guard':0}
RANK={'ORC2':7,'RAND2':8}
def _pop(flag,default=None):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
    return default
def check_rows(rows,items,labels):
    """completeness: every label x every item exactly once, with non-empty candidate lists of equal length (raw scores are the deliverable)"""
    ids=[it['id'] for it in items];seen={}
    for r in rows:seen.setdefault(r['state'],{})[r['id']]=r
    for lab in labels:
        assert lab in seen,('state missing',lab);assert set(seen[lab])==set(ids) and len(seen[lab])==len(ids),('items missing in state',lab)
        for it in items:
            r=seen[lab][it['id']];n=len(it['cands'])
            for k in ('cand_lp','cand_ntok','cand_surface','cand_kinds','cand_mean'):assert len(r[k])==n and n>0,('saved list missing / wrong length',lab,it['id'],k)
            assert r['group'] and r['event'] and r['id']==it['id'] and r['cand_surface']==it['cands'] and r['cand_kinds']==it['kinds'],('row metadata',lab,it['id'])
            assert all(np.isfinite(r['cand_lp'])),('non-finite log-prob',lab,it['id'])
    return True
def c2_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import e55_uv as UVm,e57_mask_stage as MS,e57_orc2 as O2,e57_main_run as M
    CH=1024;assert N==2 and nch==2*n and set(arms)<=set(RANK),(N,nch,arms)
    M.install_segmented_prefix(rt,seg_chunks=a.seg_chunks,ctx_chunks=a.ctx_chunks,log=lg)
    ends=O2.sentence_ends_v2(UVm.tokenizer(tfile),ids_all);masks={k:v for k,v in np.load(POD['masks']).items()}
    ntg=json.load(open(POD['ntg']))['norm_targets'];assert ntg and len(ntg)>=nch;ntg=list(ntg[:nch]);dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_t1t2c2_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0=M.record_segmented(L0,rt,s3,LD,ids_all[:nch*CH],seg_chunks=a.seg_chunks,ctx_chunks=a.ctx_chunks,log=lg)
    rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V of the 2-cycle stream recorded ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    def mk(arm,log):return MS.SelMaskAll('SELC1',RANK[arm],masks,ends,UVm,log=log,guard=POD['guard'])
    if a.smoke>0:
        for arm in arms:
            Ls=C.Ladder55(rt,save_dir=out/'states'/f'E57_t1t2c2_smoke_{arm}',canary=None);Ls.rec_cache=rc
            rs=Ls.run_stream('SELC1',1,ids_all,nch=a.smoke,tag=f'smoke{arm}',log=lambda *x:None,chunks_meta=dummy_meta,norm_targets=ntg,uv=mk(arm,lambda *x:None))
            lg(f'  smoke {arm}: {a.smoke} chunks ok, mean update norm {np.mean([e["update_norm"] for e in rs["log"]]):.3g}, mean ce {np.mean([e["ce"] for e in rs["log"]]):.3f}');del Ls,rs;gc.collect()
    for arm in arms:
        fnA=out/f'E57_main_{arm}.arms.json'
        if fnA.exists() and not a.rerun:print('exists, reused:',fnA);continue
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_main_{arm}',canary=None);L.rec_cache=rc;traj=[]
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),cycle=k//n+1,canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream('SELC1',1,ids_all,nch=nch,tag=f'c2{arm}',log=lg,read_hook=hook,hook_chunks=set(c for c in cch if c<nch),read_chunks=set(c for c in rch if c<nch),chunks_meta=dummy_meta,norm_targets=ntg,uv=mk(arm,lg));log=res['log']
        assert set(res['reads'])>={n-1,2*n-1},('weights at the end of cycle 1 and 2 were not saved',sorted(res['reads']))
        fnA.write_text(json.dumps(dict(arm=arm,engine='SELC1',mask_rank=RANK[arm],guard=POD['guard'],bank=bsrc,cycles=N,n_chunks=nch,canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','uv_hidden_tokens','uv_hidden_chunk_tokens') if k_ in e} for e in log],read_positions_in_cycle=kks),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def build_states(out,arms,nat_arms):
    """-> [(label, file, hash)] : NAT from job 261 (chunks 279 / 559), ORC2 / RAND2 from this run"""
    st=[];n1,n2='279','559'
    A=json.load(open(nat_arms))
    for c,k in ((1,n1),(2,n2)):st.append((f'NAT_c{c}',A['states'][k]['file'],A['states'][k]['hash']))
    for arm in arms:
        B=json.load(open(pathlib.Path(out)/f'E57_main_{arm}.arms.json'))
        for c,k in ((1,n1),(2,n2)):st.append((f'{arm}_c{c}',B['states'][k]['file'],B['states'][k]['hash']))
    return st
def c2_reads(a,out,items,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    import e55_gpu as RG,e57_t1t2_run as T,e57_reads as RD
    frozen=T.load_frozen(POD['frozen']);new=T.make_items(frozen,out);states=build_states(out,arms,POD['nat'])
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,new);assert not bad,bad
    forbid=sorted({s for it in items for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)})
    out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);GROUP=40;groups=[];o=0
    for g0 in range(0,len(new),GROUP):
        sub=new[g0:g0+GROUP];nr=sum(len(it['cands']) for it in sub);groups.append((sub,rd.packed(raw[o:o+nr],out_tail)));o+=nr
    assert o==len(raw);lg(f'OUT tail: {len(out_tail)} tokens; sha {hashlib.sha256(np.asarray(out_tail,np.int64).tobytes()).hexdigest()[:16]}; scorers built for {len(new)} frames')
    def score(m):
        m32=rd.to_fp32(m);res=[]
        with LD.jax.default_matmul_precision('highest'):
            for sub,P in groups:res+=RD.split_scores(np.asarray(P.r1(m32)),sub)
        return res
    t0=time.monotonic();w0rows=T.rows_for_state(new,score(rt.base),ntok,'W0');allrows=list(w0rows);tim={'W0':time.monotonic()-t0};lg(f'  W0 read: {tim["W0"]:.0f}s')
    part=pathlib.Path(out)/'E57_t1t2c2_scores_partial.json'
    def save_partial():part.write_text(json.dumps(dict(done=sorted({r['state'] for r in allrows}),rows=allrows),default=float))
    save_partial();live_dt=T.inner_dtypes_(LD,rd,rt.base)
    for lab,f,h in states:
        t0=time.monotonic();m=rd.load_state(str(f),expect_hash=h);dt=T.inner_dtypes_(LD,rd,m);assert dt==live_dt,('saved weights have another precision than the training state',lab,dt,live_dt)
        allrows+=T.rows_for_state(new,score(m),ntok,lab,w0=w0rows);del m;tim[lab]=time.monotonic()-t0;save_partial();lg(f'  {lab} read: {tim[lab]:.0f}s (partial rows saved)')
    check_rows(allrows,new,['W0']+[s[0] for s in states]);lg('completeness check passed: every state x every item has cand_lp / cand_ntok / cand_surface / cand_kinds')
    meta=dict(frozen_sha256=frozen['sha256'],bank_v1_sha256=frozen['bank_v1_sha256'],states={lab:h for lab,_,h in states},context='OUT (8192-token SQuAD tail of job 261)',cycle_ends='chunk 279 = end of cycle 1, chunk 559 = end of cycle 2; NAT = states of job 261',primary_score='cand_mean = cand_lp / cand_ntok (raw score columns only; no rule, no threshold)',n_items=len(new),timings=tim,saved_weights_precision=live_dt)
    pathlib.Path(out).mkdir(parents=True,exist_ok=True);(pathlib.Path(out)/'E57_t1t2c2_scores.json').write_text(json.dumps(dict(meta=meta,rows=allrows),default=float))
    kn=T.key_numbers(allrows,meta);(pathlib.Path(out)/'E57_t1t2c2_score_keys.json').write_text(json.dumps(kn,indent=1,default=float));lg('wrote E57_t1t2c2_scores.json and E57_t1t2c2_score_keys.json')
def selftest():
    import e57_t1t2_run as T
    items=[dict(id='a',source='s',kind_form='T1-question',event='e',type='PERSON',stratum='T1-question',cands=['g','x','y'],kinds=['gold','easy','hard'],gold_idx=0,hard_idx=[2],all_idx=[0,1,2],pool_idx=[2]),dict(id='b',source='s2',kind_form='T2-cloze',event='e2',type='PERSON',stratum='T2-cloze',cands=['g2','x2'],kinds=['gold','easy'],gold_idx=0,hard_idx=[],all_idx=[0,1],pool_idx=[])]
    sc=lambda d:[np.array([-3.0+d,-6.0,-5.0]),np.array([-4.0,-5.0+d])];nt=[[2,3,1],[2,2]]
    w0=T.rows_for_state(items,sc(0),nt,'W0');rows=w0+T.rows_for_state(items,sc(1),nt,'NAT_c1',w0=w0)+T.rows_for_state(items,sc(2),nt,'ORC2_c2',w0=w0)
    assert check_rows(rows,items,['W0','NAT_c1','ORC2_c2'])
    for bad in ('missing_state','missing_item','empty_list','nan'):
        r2=[dict(x) for x in rows]
        try:
            if bad=='missing_state':check_rows(r2,items,['W0','RAND2_c1'])
            elif bad=='missing_item':check_rows([x for x in r2 if not (x['state']=='NAT_c1' and x['id']=='b')],items,['W0','NAT_c1'])
            elif bad=='empty_list':
                r2[3]['cand_lp']=[];check_rows(r2,items,['W0','NAT_c1'])
            else:
                r2[3]['cand_lp']=[float('nan')]*len(r2[3]['cand_lp']);check_rows(r2,items,['W0','NAT_c1'])
        except AssertionError:continue
        raise SystemExit('completeness check did not catch: '+bad)
    print('e57_t1t2c2_run selftest ok')
def cpu_check():
    import argparse,e57_t1t2_run as T
    ap=argparse.ArgumentParser();ap.add_argument('--frozen');ap.add_argument('--nat-arms');ap.add_argument('--out');ap.add_argument('--tokenizer');a=ap.parse_args()
    frozen=T.load_frozen(a.frozen);items=T.make_items(frozen,a.out);print('frozen hash ok',frozen['sha256'][:16],'| items',len(items),'| rows',sum(len(it['cands']) for it in items),flush=True)
    A=json.load(open(a.nat_arms))
    for k in ('279','559'):
        s=A['states'][k];p=pathlib.Path(s['file']);print(f'NAT state chunk {k}: {p} exists={p.exists()} size={p.stat().st_size if p.exists() else None}')
        assert p.exists()
    assert len(A['norm_targets'])>=560
    selftest()
def main():
    if '--selftest' in sys.argv:selftest();return
    if '--cpu-check' in sys.argv:sys.argv.remove('--cpu-check');cpu_check();return
    POD['masks']=_pop('--pod-masks');POD['ntg']=_pop('--ntg-from');POD['nat']=_pop('--nat-arms');POD['frozen']=_pop('--frozen');g=_pop('--guard');POD['guard']=int(g) if g is not None else 0
    import e57_main_run as M
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(RANK);M.stage_arms=c2_arms;M.stage_reads=c2_reads
    if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='arms':assert POD['masks'] and POD['ntg'],'--pod-masks and --ntg-from are required for the arms stage'
    if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='reads':assert POD['nat'] and POD['frozen'],'--nat-arms and --frozen are required for the reads stage'
    M.main()
if __name__=='__main__':main()
