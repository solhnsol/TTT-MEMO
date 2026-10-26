"""E57 MAIN run (GPU; E57 section 39): the natural news stream in date order repeated for --cycles cycles (all independent lead articles of the bundle, real tokenizer, whole 1024-token chunks per cycle; cycle c+1 starts where cycle c ended), one inner update per chunk, arms
  NAT      plain SGD write (the natural reading);
  NATADAMH Adam direction, update norm = 0.5 x the NAT norm of the same chunk, NO mask (the same-size update control of the masked arms; arm added at run time to e55_swap.ADAM);
  ORC      Adam direction x 0.5 NAT norm + ORACLE answer mask: every window sentence that holds the gold answer (surface-form match, any question of the bank) is hidden from the attention of the write (stored mask file e57_main_oracle_masks.npz, e55_selmask.SelMask arm SELC1; labels are used: oracle);
  RAND     the same hidden-token count per chunk as ORC, random window sentences (e55_uv.UV arm RSELC1).
Reads of the FROZEN bank (bank_v1.json) at W0 and at --npos (default 1 = the end of every cycle) read positions per cycle, fixed by stream position before the run: OUT = neutral SQuAD tail of exactly 8192 tokens (no gold string of the bank in it); IN = the natural window (the last 8192 stream tokens before the read; window_info present / absent by gold surface-form match in the decoded window). Reads are read-only (fp32 COPY of the weights). W0 is read with the same contexts as every later read. Canary (pg11 generation NLL, pg1342 tail NLL) at W0 and at --canary-npos (10) positions per cycle; the first chunk at which the rise of either canary NLL over W0 reaches --stop-rise (default 0.10, the E55 limit) is RECORDED (the runs are short, R = 2, and continue). Log rows as in e57_reads.log_rows (per arm, cycle, read position, condition, window_info, question, n_k, shadow, gold log-prob, ranks over all / hard-only / full hard pool, accuracy, per-candidate log-probs). Results: E57_main_<ARM>.json (rewritten after every read position) and E57_main_W0.json.
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
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--cycles',type=int,default=2);ap.add_argument('--npos',type=int,default=1,help='read positions per cycle (1 = the end of every cycle)');ap.add_argument('--canary-npos',type=int,default=10,help='canary positions per cycle (canary only, no reads)');ap.add_argument('--arms',default=','.join(ARMS_ALL));ap.add_argument('--stop-rise',type=float,default=0.10)
    ap.add_argument('--stage',default=None);ap.add_argument('--check-only',action='store_true');ap.add_argument('--stub',action='store_true');ap.add_argument('--tokenizer');ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
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
    assert a.stage in ('arms','reads'),'--stage arms | reads (the two stages run in separate processes: the GPU memory of the full-stream prefix output and of the fp32 reader do not fit together)'
    import e55_swap as E
    E.ADAM['NATADAMH']=('NAT',0.5,False);E.ARMS.append('NATADAMH')               # BEFORE e55_core is imported (lab() below): IMPLEMENTED / CHUNK_META_ARMS are built from E.ADAM at import
    import e55_run as R,e55_selmask as SM
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True)
    cw0=can(rt.base);print('canary W0',cw0,flush=True)
    if a.stage=='arms':stage_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM)
    else:stage_reads(a,out,items,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg)
def stage_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    """STAGE 1 (no fp32 reader, no scorers on the GPU): the arms; canary at the end of every cycle; the weights at the end of every cycle are saved (state files of 1.4 GB, e55_core save_state) for stage 2"""
    import e55_uv as UVm
    ntg=None;ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids_all)
    mfile=out/'e57_main_oracle_masks.npz';ms=oracle_masks(ends,ids_all,tk,items,nch,UVm);np.savez_compressed(mfile,**ms)
    rs_t=None;dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}                                  # rc: ONE record of the natural K,V of the stream (about 400 GiB host for 6 cycles) shared by all arms
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
            rg=cn['_gen_nll']-cw0['_gen_nll'];rt_=cn['_tail_nll']-cw0['_tail_nll'];lg(f'  {arm} end of cycle {cyc} (chunk {k}): canary rise gen {rg:+.4f} tail {rt_:+.4f}')
            if max(rg,rt_)>=a.stop_rise and hold['stopped'] is None:hold['stopped']=int(k);lg(f'  {arm}: the end-of-cycle canary rise reached the limit {a.stop_rise} at chunk {k}; this cycle and every later cycle are exploratory in the report')
            return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream(engine,1,ids_all,nch=eff,tag=f'main{arm}',log=lg,read_hook=hook,hook_chunks=set(c for c in cch if c<eff),read_chunks=set(c for c in rch if c<eff),chunks_meta=dummy_meta,norm_targets=ntg,uv=uvo)
        log=res['log']
        if arm=='NAT':ntg=[e['update_norm'] for e in log]
        if arm=='ORC':rs_t=[e.get('n_masked_keys',0) for e in log]
        fnA.write_text(json.dumps(dict(arm=arm,engine=engine,bank=bsrc,cycles=N,n_chunks=nch,first_chunk_over_canary_limit=hold['stopped'],canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys') if k_ in e} for e in log],norm_targets=ntg if arm=='NAT' else None,rs_targets=rs_t if arm=='ORC' else None,read_positions_in_cycle=kks,stop_rise=a.stop_rise),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def stage_reads(a,out,items,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg):
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
    fn0=out/'E57_main_W0.json'
    if not fn0.exists() or a.rerun:
        rows0=[];tim0={}
        for kk in kks:tim0[kk]=read('W0',0,kk,rt.base,rows0);lg(f'  W0 reads at chunk {kk}: {tim0[kk]:.0f}s')
        fn0.write_text(json.dumps(dict(arm='W0',bank=bsrc,canary_w0=cw0,rows=rows0,timings=tim0),default=float))
    for arm in arms:
        fnA=out/f'E57_main_{arm}.arms.json';fnF=out/f'E57_main_{arm}.json'
        if not fnA.exists():print('stage 1 file missing, skipped:',fnA);continue
        if fnF.exists() and not a.rerun:print('exists, reused:',fnF);continue
        A=json.load(open(fnA));rows=[];tim={}
        for k_str,st in sorted(A['states'].items(),key=lambda x:int(x[0])):
            k=int(k_str);cyc=k//n+1;kk=k%n;m=rd.load_state(st['file'],expect_hash=st['hash']);tim[k]=read(arm,cyc,kk,m,rows);del m;lg(f'  {arm} read at the end of cycle {cyc} (chunk {k}): {tim[k]:.0f}s')
        fnF.write_text(json.dumps(dict(arm=arm,engine=A['engine'],bank=bsrc,cycles=A['cycles'],n_chunks=A['n_chunks'],run_chunks=A['n_chunks'],first_chunk_over_canary_limit=A['first_chunk_over_canary_limit'],canary_w0=cw0,trajectory=A['trajectory'],rows=rows,timings={str(k):v for k,v in tim.items()},sec_total=A['sec_total'],update_log=A['update_log'],read_positions_in_cycle=A['read_positions_in_cycle'],stop_rise=A['stop_rise']),default=float));print('wrote',fnF,len(rows),'rows',flush=True)
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
