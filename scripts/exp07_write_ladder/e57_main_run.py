"""E57 MAIN run (GPU; E57 section 39): the natural news stream in date order repeated for --cycles cycles (all independent lead articles of the bundle, real tokenizer, whole 1024-token chunks per cycle; cycle c+1 starts where cycle c ended), one inner update per chunk, arms
  NAT      plain SGD write (the natural reading);
  NATADAMH Adam direction, update norm = 0.5 x the NAT norm of the same chunk, NO mask (the same-size update control of the masked arms; arm added at run time to e55_swap.ADAM);
  ORC      Adam direction x 0.5 NAT norm + ORACLE answer mask: every window sentence that holds the gold answer (surface-form match, any question of the bank) is hidden from the attention of the write (stored mask file e57_main_oracle_masks.npz, e55_selmask.SelMask arm SELC1; labels are used: oracle);
  RAND     the same hidden-token count per chunk as ORC, random window sentences (e55_uv.UV arm RSELC1).
Reads of the FROZEN bank (bank_v1.json) at W0 and at --npos (default 1 = the end of every cycle) read positions per cycle, fixed by stream position before the run: OUT = neutral SQuAD tail of exactly 8192 tokens (no gold string of the bank in it); IN = the natural window (the last 8192 stream tokens before the read; window_info present / absent by gold surface-form match in the decoded window). Reads are read-only (fp32 COPY of the weights). W0 is read with the same contexts as every later read. Canary (pg11 generation NLL, pg1342 tail NLL) at W0 and at --canary-npos (10) positions per cycle; the first chunk at which the rise of either canary NLL over W0 reaches --stop-rise  Log rows as in e57_reads.log_rows (per arm, cycle, read position, condition, window_info, question, n_k, shadow, gold log-prob, ranks over all / hard-only / full hard pool, accuracy, per-candidate log-probs). Results: E57_main_<ARM>.json (rewritten after every read position) and E57_main_W0.json.
  python3 e57_main_run.py --bundle OUT_news --out DIR [--bank F] [--cycles 2] [--npos 1] [--canary-npos 10] [--arms NAT,NATADAMH,ORC,RAND] [--stop-rise 0.10] --stage arms|reads [--check-only] [--selftest]"""
import sys,json,os,time,argparse,hashlib,pathlib,gc
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_reads as RD
CH=1024;NW=8;ARMS_ALL=('NAT','NATADAMH','ORC','RAND')
EXCLUDE_OVERLAP=('may-person-01','may-person-03')       # question wording overlaps an article sentence (leakcheck): reported separately in the report script
def read_positions(n,npos):return [round((j+1)*n/npos)-1 for j in range(npos)]
def oracle_masks(ends,ids_all,tk,items,nch,UVm):
    """stored hidden window sentences of the ORACLE answer mask (selmask format, rank 1): a window sentence is hidden when it holds the gold of any question of the bank"""
    import e57_bank as B
    ms=[B.ans_matcher(it['q'])[0] for it in items];memo={};out={}
    for k in range(1,nch):
        wsid,wst,wlen=UVm.window_sents(ends,k,CH,NW)
        if len(wlen)==0:continue
        base=(k-NW)*CH;hid=np.zeros(len(wlen),bool)
        for i,(st,ln) in enumerate(zip(wst,wlen)):
            a=int(st)+base
            if a not in memo:
                txt=tk.decode([int(x) for x in ids_all[a:a+int(ln)]]);memo[a]=any(m(txt) for m in ms)
            hid[i]=memo[a]
        out[f'h1_{k}']=hid;out[f'wlen{k}']=np.asarray(wlen,np.int32)
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--cycles',type=int,default=2);ap.add_argument('--npos',type=int,default=1,help='read positions per cycle (1 = the end of every cycle)');ap.add_argument('--canary-npos',type=int,default=10,help='canary positions per cycle (canary only, no reads)');ap.add_argument('--arms',default=','.join(ARMS_ALL));ap.add_argument('--stop-rise',type=float,default=None,help='unused (the canary limit was abolished by Hans, E57 section 50); kept for old command lines')
    ap.add_argument('--stage',default=None);ap.add_argument('--pair-tol',type=float,default=1e-3);ap.add_argument('--pfx-chunks',type=int,default=560);ap.add_argument('--pfx-seg',type=int,default=140);ap.add_argument('--pfx-seg2',type=int,default=70);ap.add_argument('--ctx-chunks',type=int,default=192);ap.add_argument('--seg-chunks',type=int,default=280);ap.add_argument('--pfx-tol',type=float,default=0.02);ap.add_argument('--noise-factor',type=float,default=2.0);ap.add_argument('--check-only',action='store_true');ap.add_argument('--stub',action='store_true');ap.add_argument('--tokenizer');ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    ROOT=HERE.parents[1];import gen_world_ctl as G
    tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.StubTok() if a.stub else G.load_enc(tfile)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json'
    import e57_trial_run as TR
    if bank.exists():items=RD.load_items(bank);bsrc=str(bank)
    else:
        assert a.check_only,f'frozen bank {bank} missing (exit 7)';out0=pathlib.Path(a.out);out0.mkdir(parents=True,exist_ok=True);tmp=out0/'_pseudo_bank.json';tmp.write_text(json.dumps(TR.pseudo_bank(a.bundle)));items=RD.load_items(tmp);bsrc='PSEUDO (check only)'
    ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH;N=a.cycles;nch=N*n;shtab=RD.shadow_table(iv,hits,n*CH);ids_all=np.tile(ids,N)
    kks=read_positions(n,a.npos);assert min(kks)>=NW-1,kks;rch=sorted(c*n+kk for c in range(N) for kk in kks);cpos=read_positions(n,a.canary_npos);cch=sorted(set(rch)|{c*n+kk for c in range(N) for kk in cpos})
    arms=a.arms.split(',');assert all(x in ARMS_ALL for x in arms),arms
    from tokenizers import Tokenizer;tk=None if a.stub else Tokenizer.from_file(tfile)
    print(f'bank {bsrc}: {len(items)} questions; stream {n} chunks x {N} cycles = {nch}; read positions in a cycle (chunk): {kks}; arms {arms}; ids sha {hashlib.sha256(ids.tobytes()).hexdigest()[:16]}',flush=True)
    print('duplicates removed from the scored candidate lists:',{i['id']:i['removed_duplicates'] for i in items if i['removed_duplicates']} or 'none',flush=True)
    def gwin(k):e=(k+1)*CH;return max(0,e-8192),e
    if tk is not None:
        raw,ntok,bad=RD.build_rows(tk,items);assert not bad,bad;print(f'rows: {len(raw)}',flush=True)
        import e55_uv as UVm,e55_selmask as SM
        ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids_all);ms=oracle_masks(ends,ids_all,tk,items,nch,UVm)
        hid_tok=[int(sum(int(l) for l,h in zip(ms[f"wlen{k}"],ms[f"h1_{k}"]) if h)) for k in range(NW+1,nch,max(1,nch//12)) if f"wlen{k}" in ms]
        print(f'oracle answer mask: hidden window tokens at sampled chunks {hid_tok} (of ~8192)',flush=True)
        for kk in kks[:2]+kks[-1:]:
            wp=RD.window_present(items,tk.decode([int(x) for x in ids_all[gwin(n+kk)[0]:gwin(n+kk)[1]]]));print(f'  read chunk {kk} (cycle 2 window): gold of {sum(wp.values())} of {len(items)} questions in the window',flush=True)
    if a.check_only:print('main run check-only OK');return
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert a.stage in ('pfx','noise','pair','arms','reads'),'--stage pair | arms | reads (separate processes: the GPU memory of the full-stream prefix output and of the fp32 reader do not fit together)'
    import e55_swap as E
    E.ADAM['NATADAMH']=('NAT',0.5,False);E.ARMS.append('NATADAMH')               # BEFORE e55_core is imported (lab() below): IMPLEMENTED / CHUNK_META_ARMS are built from E.ADAM at import
    import e55_run as R,e55_selmask as SM
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True)
    cw0=can(rt.base);print('canary W0',cw0,flush=True)
    if a.stage=='pfx':stage_pfx(a,out,items,bsrc,iv,hits,shtab,ids,ids_all,n,kks,tk,LD,s3,C,rt,can,cw0,lg,R)
    elif a.stage=='noise':stage_noise(a,out,items,bsrc,iv,hits,shtab,ids,ids_all,n,kks,tk,LD,s3,C,rt,can,cw0,lg,R)
    elif a.stage=='pair':stage_pair(a,out,items,bsrc,iv,hits,shtab,ids,n,kks,tk,LD,s3,C,rt,can,cw0,lg,R)
    elif a.stage=='arms':stage_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM)
    else:stage_reads(a,out,items,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can)
def stage_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    """STAGE 1 (no fp32 reader, no scorers on the GPU): the arms; canary at the end of every cycle; the weights at the end of every cycle are saved (state files of 1.4 GB, e55_core save_state) for stage 2"""
    import e55_uv as UVm
    install_segmented_prefix(rt,seg_chunks=a.seg_chunks,ctx_chunks=a.ctx_chunks,log=lg)
    ntg=None;ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids_all)
    mfile=out/'e57_main_oracle_masks.npz';ms=oracle_masks(ends,ids_all,tk,items,nch,UVm);np.savez_compressed(mfile,**ms)
    rs_t=None;dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}                                  # rc: ONE record of the natural K,V of the stream (about 400 GiB host for 6 cycles) shared by all arms
    # the natural K,V record of the whole stream needs a 44 GiB working array on the GPU; run_stream records AFTER it has computed the prefix, and the (cached) prefix output of 8 GiB plus the compiled segment programs left too little room (job 261, 08:57Z): record FIRST, while only the model is on the GPU, and hand the result to run_stream through its record cache (same key as run_stream builds)
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_main_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V of the stream recorded before the arms ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s');del L0,_h;gc.collect();LD.jax.clear_caches()
    for arm in arms:
        fnA=out/f'E57_main_{arm}.arms.json'
        if fnA.exists() and not a.rerun:
            prev=json.load(open(fnA))
            if arm=='NAT':ntg=prev['norm_targets']
            if arm=='ORC':rs_t=prev['rs_targets']
            print('exists, reused:',fnA);continue
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_main_{arm}',canary=None);L.rec_cache=rc
        eff=nch;uvo=None;engine={'NAT':'NAT','NATADAMH':'NATADAMH','ORC':'SELC1','RAND':'RSELC1'}[arm]
        if arm in ('NATADAMH','ORC','RAND'):assert ntg is not None,'NAT must run first (norm targets)'
        if arm=='ORC':uvo=SM.SelMask('SELC1',1,{k:v for k,v in np.load(mfile).items()},ends,None,log=lg)
        if arm=='RAND':assert rs_t is not None,'ORC must run before RAND (hidden-token counts)';uvo=UVm.UV('RSELC1',ends,None,L=L,rt=rt,s3=s3,rs_targets=rs_t,log=lg)
        traj=[];hold={'stopped':None}
        def hook(k,m,arm=arm,traj=traj,hold=hold):
            cyc=k//n+1;cn=can(m);traj.append(dict(chunk=int(k),cycle=cyc,canary=cn))
            rg=cn['_gen_nll']-cw0['_gen_nll'];rt_=cn['_tail_nll']-cw0['_tail_nll'];lg(f'  {arm} end of cycle {cyc} (chunk {k}): canary rise gen {rg:+.4f} tail {rt_:+.4f} (recorded only; no limit, no monitoring)')
            return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream(engine,1,ids_all,nch=eff,tag=f'main{arm}',log=lg,read_hook=hook,hook_chunks=set(c for c in cch if c<eff),read_chunks=set(c for c in rch if c<eff),chunks_meta=dummy_meta,norm_targets=ntg,uv=uvo)
        log=res['log']
        if arm=='NAT':ntg=[e['update_norm'] for e in log]
        if arm=='ORC':rs_t=[e.get('n_masked_keys',0) for e in log]
        fnA.write_text(json.dumps(dict(arm=arm,engine=engine,bank=bsrc,cycles=N,n_chunks=nch,canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys') if k_ in e} for e in log],norm_targets=ntg if arm=='NAT' else None,rs_targets=rs_t if arm=='ORC' else None,read_positions_in_cycle=kks,),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def install_segmented_prefix(rt,seg_chunks=280,ctx_chunks=192,log=print):
    """run_stream computes the prefix (frozen first layers) output of the WHOLE stream in one jitted call: the working memory grows with the stream length (about 46 MB per chunk: 12.9 GiB for 280 chunks, 77 GiB for the 1680 chunks of 6 cycles, which fails). The wrapper computes it in segments of seg_chunks chunks, each with ctx_chunks chunks of preceding context (24 prefix layers x window 8192 tokens = 192 chunks: the largest distance one position can depend on), keeps the segment part, and caches the result for the other arms (it depends on the token ids and the base weights only). RoPE is relative (and computed on the fly, rope_fly), so the segment may start at position 0."""
    import jax.numpy as jnp
    orig=rt.prefix;cache={}
    def seg_prefix(m,seq):
        T=int(seq.input_ids.shape[0])
        if T<=(seg_chunks+ctx_chunks)*CH:return orig(m,seq)
        key=(hashlib.sha256(np.asarray(seq.input_ids).tobytes()).hexdigest(),id(m))
        if key in cache:log('  prefix output of the stream taken from the cache');return cache[key]
        outs=[];t0=time.monotonic()
        for a in range(0,T,seg_chunks*CH):
            b=min(T,a+seg_chunks*CH);lo=max(0,a-ctx_chunks*CH)
            sub=type(seq)(input_ids=seq.input_ids[lo:b],target_tokens=seq.target_tokens[lo:b],loss_masks=seq.loss_masks[lo:b]);h=orig(m,sub);outs.append(h[a-lo:b-lo]);del h
        res=jnp.concatenate(outs,axis=0);del outs;cache.clear();cache[key]=res;log(f'  prefix output of the stream computed in segments of {seg_chunks} chunks (+{ctx_chunks} context): {T//CH} chunks, {time.monotonic()-t0:.0f}s');return res
    rt.prefix=seg_prefix;rt._orig_prefix=orig;return seg_prefix
def stage_pfx(a,out,items,bsrc,iv,hits,shtab,ids,ids_all,n,kks,tk,LD,s3,C,rt,can,cw0,lg,R):
    """EQUIVALENCE CHECK of the segmented prefix (before the long run), two parts, exit 9 if either fails (the criteria are fixed and are not relaxed after seeing the numbers):
    (1) the prefix output of the first --pfx-chunks (560) chunks in ONE call against the segmented computation with --pfx-seg (140) chunk segments: relative L2 error <= --pfx-tol (0.02);
    (2) the NAT arm over ONE cycle (280 chunks, fits in one call) is run twice, with the one-call prefix and with the segmented prefix (segments of --pfx-seg2 = 70 chunks + context, so that the segmentation is really used); at the end of the cycle the 84 read rows (OUT 42 + IN 42) and the canary of both weights are compared: cand_lp max abs difference <= --pair-tol (1e-3), no rank changed, canary gen and tail difference <= --pair-tol."""
    out=pathlib.Path(out);rep={}
    T=a.pfx_chunks*CH;ids_=ids_all[:T];seq=rt.seq(ids_,mask=np.ones(T,bool));t0=time.monotonic();orig=rt.prefix
    with rt.mesh:
        full=np.asarray(orig(rt.base,seq)).astype(np.float32);lg('full prefix computed',full.shape,f'{time.monotonic()-t0:.0f}s')
        install_segmented_prefix(rt,seg_chunks=a.pfx_seg,ctx_chunks=a.ctx_chunks,log=lg);segd=np.asarray(rt.prefix(rt.base,seq)).astype(np.float32);rt.prefix=orig
    d=np.abs(full-segd);rel=float(np.linalg.norm(full-segd)/np.linalg.norm(full));mx=float(d.max())
    per=[float(np.linalg.norm((full-segd)[c*CH:(c+1)*CH])/np.linalg.norm(full[c*CH:(c+1)*CH])) for c in range(0,a.pfx_chunks,max(1,a.pfx_chunks//14))]
    ok1=rel<=a.pfx_tol;rep['part1_prefix']=dict(chunks=a.pfx_chunks,segment_chunks=a.pfx_seg,context_chunks=a.ctx_chunks,rel_l2=rel,max_abs=mx,rel_l2_by_sampled_chunk=per,tolerance=a.pfx_tol,ok=bool(ok1))
    del full,segd,d;gc.collect();lg('PFX part 1',json.dumps(rep['part1_prefix']))
    import e55_gpu as RG
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)})
    out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);P_out=rd.packed(raw,out_tail)
    s_,e_=max(0,n*CH-8192),n*CH;P_in=rd.packed(raw,[int(x) for x in ids[s_:e_]]);wp=RD.window_present(items,tk.decode([int(x) for x in ids[s_:e_]]));csa={it['id']:it['q'].get('c_SA') for it in items}
    def score(P,m):
        m32=rd.to_fp32(m)
        with LD.jax.default_matmul_precision('highest'):return RD.split_scores(np.asarray(P.r1(m32)),items)
    def read(arm,m):
        rows=RD.log_rows(items,score(P_out,m),ntok,cycle=1,read_chunk=n-1,pos_tokens=n*CH,win=None,arm=arm,condition='OUT',iv=iv,hits=hits,csa=csa,shadow=shtab)
        rows+=RD.log_rows(items,score(P_in,m),ntok,cycle=1,read_chunk=n-1,pos_tokens=n*CH,win=(s_,e_),arm=arm,condition='IN',iv=iv,hits=hits,winpresent=wp,csa=csa,shadow=shtab);return rows
    res2={}
    for tag,seg in (('full',None),('segmented',a.pfx_seg2)):
        rt.prefix=orig
        if seg:install_segmented_prefix(rt,seg_chunks=seg,ctx_chunks=a.ctx_chunks,log=lg)
        got={}
        def hook(k,m):got['rows']=read('NAT_'+tag,m);got['canary']=can(m);return {'ok':True}
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_pfx_{tag}',canary=None);t1=time.monotonic()
        L.run_stream('NAT',1,ids,nch=n,tag='pfx'+tag,log=lg,read_hook=hook,hook_chunks={n-1});res2[tag]=dict(rows=got['rows'],canary=got['canary'],sec=time.monotonic()-t1);del L;gc.collect();rt.prefix=orig
    A,B=res2['full'],res2['segmented'];dl=[abs(x-y) for r1,r2 in zip(A['rows'],B['rows']) for x,y in zip(r1['cand_lp'],r2['cand_lp'])];rk=sum(1 for r1,r2 in zip(A['rows'],B['rows']) if (r1['rank_all'],r1['rank_hard'],r1['rank_pool'])!=(r2['rank_all'],r2['rank_hard'],r2['rank_pool']))
    cd=dict(gen=B['canary']['_gen_nll']-A['canary']['_gen_nll'],tail=B['canary']['_tail_nll']-A['canary']['_tail_nll'])
    ok2=max(dl)<=a.pair_tol and rk==0 and max(abs(cd['gen']),abs(cd['tail']))<=a.pair_tol
    rep['part2_nat_one_cycle']=dict(rows=len(A['rows']),max_abs_diff_cand_lp=max(dl),mean_abs_diff_cand_lp=float(np.mean(dl)),rows_with_changed_rank=rk,canary_full=A['canary'],canary_segmented=B['canary'],canary_diff=cd,segment_chunks=a.pfx_seg2,context_chunks=a.ctx_chunks,tolerance=a.pair_tol,ok=bool(ok2),sec_full=A['sec'],sec_segmented=B['sec'])
    rep['ok']=bool(ok1 and ok2);rep['sec']=time.monotonic()-t0
    (out/'E57_pfx_check.json').write_text(json.dumps(rep,indent=1,default=float));print('PFX CHECK',json.dumps({k:(v if k!='part2_nat_one_cycle' else {kk:vv for kk,vv in v.items() if not kk.startswith('canary_')}) for k,v in rep.items()},default=float),flush=True)
    if not rep['ok']:print('segmented prefix check FAILED (exit 9): the differences are reported above; the long run is not started and Hans decides (the criteria are not relaxed)',flush=True);sys.exit(9)
class StopRun(Exception):pass
def stage_noise(a,out,items,bsrc,iv,hits,shtab,ids,ids_all,n,kks,tk,LD,s3,C,rt,can,cw0,lg,R):
    """NOISE-FLOOR CONTROL for the segmented prefix (amended criterion, decided by Hans after seeing the first check, recorded as a post-hoc change): the SAME NAT arm over one cycle (280 chunks) with the prefix of a 560-chunk stream computed in one call (the first 280 chunks of the update use hidden states computed in a call of another shape; mathematically the same computation). Compared with the one-call 280-chunk run of job 256 (rows of E57_trial_NAT.json cycle 1, reproduced bit for bit in job 259) this gives the numerical noise floor of 'another call shape'. The segmented run of E57_pfx_check.json (job 259) PASSES the amended criterion when: no rank changed, canary gen and tail differences <= --pair-tol (1e-3), and its max and mean |delta cand_lp| are <= --noise-factor (2.0) x the noise floor max and mean."""
    import e55_gpu as RG
    pf=json.load(open(out/'E57_pfx_check.json'));seg=pf['part2_nat_one_cycle'];trial=json.load(open(out/'..'/'E57_trial_NAT.json')) if False else None
    tf=pathlib.Path(a.bundle).parent/'E57_trial_NAT.json';trial=json.load(open(tf));A=[r for r in trial['rows'] if r['cycle']==1]
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)})
    out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);P_out=rd.packed(raw,out_tail)
    s_,e_=max(0,n*CH-8192),n*CH;P_in=rd.packed(raw,[int(x) for x in ids[s_:e_]]);wp=RD.window_present(items,tk.decode([int(x) for x in ids[s_:e_]]));csa={it['id']:it['q'].get('c_SA') for it in items}
    def score(P,m):
        m32=rd.to_fp32(m)
        with LD.jax.default_matmul_precision('highest'):return RD.split_scores(np.asarray(P.r1(m32)),items)
    def read(arm,m):
        rows=RD.log_rows(items,score(P_out,m),ntok,cycle=1,read_chunk=n-1,pos_tokens=n*CH,win=None,arm=arm,condition='OUT',iv=iv,hits=hits,csa=csa,shadow=shtab)
        rows+=RD.log_rows(items,score(P_in,m),ntok,cycle=1,read_chunk=n-1,pos_tokens=n*CH,win=(s_,e_),arm=arm,condition='IN',iv=iv,hits=hits,winpresent=wp,csa=csa,shadow=shtab);return rows
    got={}
    def hook(k,m):got['rows']=read('NAT_prefix560',m);got['canary']=can(m);raise StopRun()
    L=C.Ladder55(rt,save_dir=out/'states'/'E57_noise',canary=None);t0=time.monotonic()
    try:L.run_stream('NAT',1,ids_all[:2*n*CH],nch=2*n,tag='noise',log=lg,read_hook=hook,hook_chunks={n-1})
    except StopRun:pass
    Bc=got['rows'];key=lambda r:(r['condition'],r['question'])
    amap={key(r):r for r in A};dl=[abs(x-y) for r in Bc for x,y in zip(r['cand_lp'],amap[key(r)]['cand_lp'])];rk=sum(1 for r in Bc if (r['rank_all'],r['rank_hard'],r['rank_pool'])!=(amap[key(r)]['rank_all'],amap[key(r)]['rank_hard'],amap[key(r)]['rank_pool']))
    cd=dict(gen=got['canary']['_gen_nll']-trial['trajectory'][-1]['canary']['_gen_nll'],tail=got['canary']['_tail_nll']-trial['trajectory'][-1]['canary']['_tail_nll'])
    floor=dict(rows=len(Bc),max_abs_diff_cand_lp=max(dl),mean_abs_diff_cand_lp=float(np.mean(dl)),rows_with_changed_rank=rk,canary_diff=cd)
    f=a.noise_factor;ok=(seg['rows_with_changed_rank']==0 and abs(seg['canary_diff']['gen'])<=a.pair_tol and abs(seg['canary_diff']['tail'])<=a.pair_tol and seg['max_abs_diff_cand_lp']<=f*floor['max_abs_diff_cand_lp'] and seg['mean_abs_diff_cand_lp']<=f*floor['mean_abs_diff_cand_lp'])
    rep=dict(noise_floor=floor,segmented_vs_full=dict(max_abs_diff_cand_lp=seg['max_abs_diff_cand_lp'],mean_abs_diff_cand_lp=seg['mean_abs_diff_cand_lp'],rows_with_changed_rank=seg['rows_with_changed_rank'],canary_diff=seg['canary_diff']),noise_factor=f,pair_tol=a.pair_tol,ok=bool(ok),note='amended criterion, decided after seeing the first check of job 259 (post-hoc change)',sec=time.monotonic()-t0)
    (out/'E57_noise_floor.json').write_text(json.dumps(rep,indent=1,default=float));print('NOISE FLOOR',json.dumps(rep,default=float),flush=True)
    if not ok:print('amended segmented-prefix criterion FAILED (exit 9)',flush=True);sys.exit(9)
def inner_dtypes(LD,rd,m):
    """precision of the inner (written) parameters: {dtype: number of leaves} -> the saved weights must have the precision of the training state"""
    import collections as _c
    return dict(_c.Counter(str(x.dtype) for x in LD.jax.tree_util.tree_leaves(LD.eqx.partition(m,rd.f)[0])))
def stage_pair(a,out,items,bsrc,iv,hits,shtab,ids,n,kks,tk,LD,s3,C,rt,can,cw0,lg,R):
    """PAIRED CHECK (cheap, before the long run): the NAT arm over ONE cycle (280 chunks, the stream of job 256), the weights at the end of the cycle are (a) read INLINE in the continuing run (fp32 reader resident, as in job 256) and (b) saved and read again FROM THE SAVED FILE; the cand_lp of every row, the ranks and the canary must agree (tolerance --pair-tol); the saved weights must have the precision of the training state. Exit 9 on disagreement."""
    import e55_gpu as RG
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)})
    out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);P_out=rd.packed(raw,out_tail);kk=kks[-1]
    s_,e_=max(0,(n)*CH-8192),n*CH;P_in=rd.packed(raw,[int(x) for x in ids[s_:e_]]);wp=RD.window_present(items,tk.decode([int(x) for x in ids[s_:e_]]))
    csa={it['id']:it['q'].get('c_SA') for it in items}
    def score(P,m):
        m32=rd.to_fp32(m)
        with LD.jax.default_matmul_precision('highest'):return RD.split_scores(np.asarray(P.r1(m32)),items)
    def read(arm,m):
        rows=RD.log_rows(items,score(P_out,m),ntok,cycle=1,read_chunk=n-1,pos_tokens=n*CH,win=None,arm=arm,condition='OUT',iv=iv,hits=hits,csa=csa,shadow=shtab)
        rows+=RD.log_rows(items,score(P_in,m),ntok,cycle=1,read_chunk=n-1,pos_tokens=n*CH,win=(s_,e_),arm=arm,condition='IN',iv=iv,hits=hits,winpresent=wp,csa=csa,shadow=shtab);return rows
    got={}
    def hook(k,m):
        got['rows']=read('PAIR_inline',m);got['canary']=can(m);got['dtypes']=inner_dtypes(LD,rd,m);return {'ok':True}
    L=C.Ladder55(rt,save_dir=out/'states'/'E57_pair',canary=None);t0=time.monotonic()
    res=L.run_stream('NAT',1,ids,nch=n,tag='pair',log=lg,read_hook=hook,hook_chunks={n-1},read_chunks={n-1})
    st=res['reads'][n-1];m2=rd.load_state(st['_file'],expect_hash=st['_hash']);rows2=read('PAIR_saved',m2);can2=can(m2);dt2=inner_dtypes(LD,rd,m2)
    d=[abs(x-y) for r1,r2 in zip(got['rows'],rows2) for x,y in zip(r1['cand_lp'],r2['cand_lp'])];rk=sum(1 for r1,r2 in zip(got['rows'],rows2) if (r1['rank_all'],r1['rank_hard'])!=(r2['rank_all'],r2['rank_hard']))
    cd=dict(gen=can2['_gen_nll']-got['canary']['_gen_nll'],tail=can2['_tail_nll']-got['canary']['_tail_nll'])
    ok=max(d)<=a.pair_tol and rk==0 and max(abs(cd['gen']),abs(cd['tail']))<=a.pair_tol and dt2==got['dtypes']
    rep=dict(rows=len(got['rows']),max_abs_diff_cand_lp=max(d),mean_abs_diff_cand_lp=float(np.mean(d)),rows_with_changed_rank=rk,canary_inline=got['canary'],canary_saved=can2,canary_diff=cd,precision_training_state=got['dtypes'],precision_saved_weights=dt2,tolerance=a.pair_tol,state_file=st['_file'],state_hash=st['_hash'],ok=bool(ok),sec=time.monotonic()-t0)
    (out/'E57_pair_check.json').write_text(json.dumps(rep,indent=1,default=float));print('PAIR CHECK',json.dumps({k:v for k,v in rep.items() if k not in('canary_inline','canary_saved')},default=float),flush=True)
    if not ok:print('pair check FAILED (exit 9)',flush=True);sys.exit(9)
def stage_reads(a,out,items,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    """STAGE 2: reads of the frozen bank at W0 and at the end of every cycle from the saved weights (read-only; fp32 copy)"""
    import e55_gpu as RG
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)})
    out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);P_out=rd.packed(raw,out_tail);P_in={}
    def gwin(k):e=(k+1)*CH;return max(0,e-8192),e
    def score(P,m):
        m32=rd.to_fp32(m)
        with LD.jax.default_matmul_precision('highest'):return RD.split_scores(np.asarray(P.r1(m32)),items)
    csa={it['id']:it['q'].get('c_SA') for it in items}
    def pin(kk):
        if kk not in P_in:s_,e_=gwin(n+kk);P_in[kk]=(rd.packed(raw,[int(x) for x in ids_all[s_:e_]]),RD.window_present(items,tk.decode([int(x) for x in ids_all[s_:e_]])),(s_,e_))
        return P_in[kk]
    def read(arm,cycle,kk,m,store):
        t0=time.monotonic();pos=(kk+1)*CH
        store.extend(RD.log_rows(items,score(P_out,m),ntok,cycle=cycle,read_chunk=cycle*n+kk if cycle>0 else kk,pos_tokens=pos,win=None,arm=arm,condition='OUT',iv=iv,hits=hits,csa=csa,shadow=shtab))
        P,wp,(s_,e_)=pin(kk);store.extend(RD.log_rows(items,score(P,m),ntok,cycle=cycle,read_chunk=cycle*n+kk if cycle>0 else kk,pos_tokens=pos,win=(s_,e_),arm=arm,condition='IN',iv=iv,hits=hits,winpresent=wp,csa=csa,shadow=shtab))
        return time.monotonic()-t0
    live_dt=inner_dtypes(LD,rd,rt.base);lg('inner parameter precision of the training state:',live_dt)
    fn0=out/'E57_main_W0.json'
    if not fn0.exists() or a.rerun:
        rows0=[];tim0={}
        for kk in kks:tim0[kk]=read('W0',0,kk,rt.base,rows0);lg(f'  W0 reads at chunk {kk}: {tim0[kk]:.0f}s')
        fn0.write_text(json.dumps(dict(arm='W0',bank=bsrc,canary_w0=cw0,rows=rows0,timings=tim0),default=float))
    for arm in arms:
        fnA=out/f'E57_main_{arm}.arms.json';fnF=out/f'E57_main_{arm}.json'
        if not fnA.exists():print('stage 1 file missing, skipped:',fnA);continue
        if fnF.exists() and not a.rerun:print('exists, reused:',fnF);continue
        A=json.load(open(fnA));rows=[];tim={};can_chk=[];by_chunk={t['chunk']:t['canary'] for t in A['trajectory']}
        for k_str,st in sorted(A['states'].items(),key=lambda x:int(x[0])):
            k=int(k_str);cyc=k//n+1;kk=k%n;m=rd.load_state(st['file'],expect_hash=st['hash'])
            dt=inner_dtypes(LD,rd,m);assert dt==live_dt,('saved weights have another precision than the training state',dt,live_dt)
            cs=can(m);c1=by_chunk.get(k);can_chk.append(dict(chunk=k,canary_from_saved=cs,canary_in_run=c1,diff_gen=None if c1 is None else cs['_gen_nll']-c1['_gen_nll'],diff_tail=None if c1 is None else cs['_tail_nll']-c1['_tail_nll']))
            tim[k]=read(arm,cyc,kk,m,rows);del m;lg(f'  {arm} read at the end of cycle {cyc} (chunk {k}): {tim[k]:.0f}s; canary from the saved weights vs in the run: {can_chk[-1]["diff_gen"]},{can_chk[-1]["diff_tail"]}')
        fnF.write_text(json.dumps(dict(arm=arm,engine=A['engine'],bank=bsrc,cycles=A['cycles'],n_chunks=A['n_chunks'],run_chunks=A['n_chunks'],canary_w0=cw0,trajectory=A['trajectory'],rows=rows,timings={str(k):v for k,v in tim.items()},sec_total=A['sec_total'],update_log=A['update_log'],read_positions_in_cycle=A['read_positions_in_cycle'],canary_saved_vs_run=can_chk,saved_weights_precision=live_dt),default=float));print('wrote',fnF,len(rows),'rows',flush=True)
def selftest():
    assert read_positions(280,10)==[27,55,83,111,139,167,195,223,251,279]
    import e55_uv as UVm
    ends=np.zeros(4*CH,bool);ends[100::100]=True
    class TK:
        def decode(self,ids):return ' '.join(str(i) for i in ids)
    class M:
        def __init__(s):s.q={}
    import e57_bank as B
    it=[dict(q=dict(type='NUMBER',canonical='5',variants=['5'],S='x',event='e'),id='q')]
    ids=np.arange(4*CH);ms=oracle_masks(ends,ids,TK(),it,4,UVm)
    assert all(k.startswith('h1_') or k.startswith('wlen') for k in ms) and any(ms[k].any() for k in ms if k.startswith('h1_'))
    print('e57_main_run selftest ok')
if __name__=='__main__':main()
