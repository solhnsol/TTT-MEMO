"""E55 stage-0 driver (server; pairless fact world, design section 10). Subcommands:
  dry   world + stream hashes + read points + plan + host-memory estimate; NO model, NO GPU (CPU_ONLY dry run of the job).
  unit  GPU unit checks (U1-U5 of e55_units.py on the real model and tokenizer); exit 9 on failure.
  run   per arm: stream the world ONCE (e55_core.Ladder55.run_stream), save weights at the read chunks (appearance 2, 4, 8, 16), canary (+0.10 nat limit) at W0 and every read chunk. Result files are restart-safe (an existing one is reused); never overwritten (exit 8 unless --rerun).
  read  candidate scores (16 candidates, question frame = fact id mod 3) of every saved read-point state of the given arms and of W0: 'out' (neutral SQuAD tail padded to 8192 tokens, all items) and optionally 'in_g<g>' (the real-stream context of group g, E55 10.2) at the appearances of --in-apps; one JSON record per arm. Weights are never updated while reading.
World: gen_facts.build_fact_world(seed, **CELLS[cell], n_app=16, real tokenizer, default_bg); decoy stream = world['decoy']. The written chunk is always the real chunk.
STATUS: the CPU parts and the swap / composition / reader logic are unit-tested on a tiny random e2e model (e55_units.py); everything here that needs the 3B checkpoint, the real tokenizer, the lab data or the GPU has NEVER been run."""
import os,sys,json,argparse,time,subprocess,hashlib,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
import numpy as np
import gen_facts as G,e55_read as R,e55_swap as E
CH=1024
def git_rev():
    try:return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    except Exception:return None
def parse_apps(x,default=()):
    """'all' -> 1..16; '2,4,8,16' -> [2,4,8,16]; None -> default"""
    if x is None:return list(default)
    return list(range(1,17)) if x=='all' else [int(v) for v in x.split(',') if v]
def seed_plan(a,seeds):
    """{seed: (out-read appearances, in-window appearances, state-save appearances)}: defaults from --read-apps / --in-apps / --save-apps, overridden per seed by --plan '55=all|1,2,4,8,16;56=2,4,8,16|16' (out | in-window; both lists may be 'all')"""
    base=(parse_apps(a.read_apps,()),parse_apps(a.in_apps,()),parse_apps(a.save_apps,(2,4,8,16)));plans={s_:base for s_ in seeds}
    if a.plan:
        for part in a.plan.split(';'):
            sd,spec=part.split('=');o,_,i=spec.partition('|');plans[int(sd)]=(parse_apps(o),parse_apps(i),base[2])
    return plans
def decoy_kinds(arms,decoy):
    """fake-window streams the arms need: SWD / SWDS / SWDO -> `decoy` ('names' = the real chunks with fake names / places, 'world' = another fact world), SWB -> 'nofact'"""
    k=[]
    if any(x in ('SWD','SWDS','SWDO') for x in arms):k.append(decoy)
    if 'SWB' in arms:k.append('nofact')
    return tuple(k)
def build(a,seed=None,read_apps=None,arms=None):
    seed=a.seed if seed is None else seed;arms=(a.arms.split(',') if arms is None else arms)
    if getattr(a,'stub',False):enc=G.StubTok();bg=G._synth_bg(9000);tf=__file__                       # CPU test without tokenizers / lab data
    else:tf=a.tokenizer_file or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);bg=G.default_bg()
    kw={} if read_apps is None else {'read_apps':tuple(sorted(read_apps))}
    w=G.build_fact_world(seed,n_app=16,enc=enc,bg=bg,decoys=decoy_kinds(arms,getattr(a,'decoy','names')) or ('names',),**G.CELLS[a.cell],**kw);return enc,w,tf
def streams(w,enc=None):
    """-> {'A': (ids, sha) real stream, 'decoy': stream of the fake window of SWD (kind names | world), 'bg': background-only stream}; ids = concatenated chunk ids (n_chunks x 1024)"""
    out={};ch=G.realize(w);out['A']=(np.asarray(G.stream_ids(ch),np.int64),G.stream_sha(ch))
    for key,kind in (('decoy',[x for x in ('names','world') if x in w['decoys']][:1]),('bg',['nofact'] if 'nofact' in w['decoys'] else [])):
        if kind:ch=G.realize(w,kind[0]);assert all(len(c['ids'])==CH for c in ch);out[key]=(np.asarray(G.stream_ids(ch),np.int64),G.stream_sha(ch),kind[0])
    return out
def host_gib(nl_prefix,hidden,T,nstreams,bytes_=2):return nl_prefix*T*hidden*bytes_*2*nstreams/2**30     # K and V of the prefix layers, merged heads, per stream
def dry(a):
    enc,w,tf=build(a);S=streams(w,enc);P=R.plan(w,enc)
    info={'cell':a.cell,'seed':a.seed,'k':w['k'],'n':w['n'],'p':w['p'],'n_chunks':w['n_chunks'],'read_chunks':w['read_chunks'],'stream_sha':{k:v[1] for k,v in S.items()},'decoy_kinds':w['decoys'],'tokenizer_sha12':hashlib.sha256(Path(tf).read_bytes()).hexdigest()[:12],'code':git_rev(),
          'items':len(P['items']),'rows':len(P['rows']),'in_window_contexts':{str(k):sorted(v['ctx']) for k,v in P['read'].items()},'min_units_per_chunk':w['min_units_per_chunk'],
          'host_gib_per_stream_3B(24 prefix layers, hidden 2560, bf16)':round(host_gib(24,2560,w['n_chunks']*CH,1),1)}
    print(json.dumps(info,indent=1));return info
def lab():
    """import the lab modules (server only)"""
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS',''),'XLA_FLAGS must contain --xla_gpu_deterministic_ops=true'
    import ladder as LD,s3,e55_core as C
    return LD,s3,C
class Canary55:
    """gen_nll / tail_nll of ladder.Ladder without its old-world Readouts (held-out pg11 text and the name-scrubbed pg1342 tail)"""
    def __init__(s,LD,rt):
        import readouts as RD
        s.LD=LD;s.rt=rt;tail,_,_=RD.neutral_tail(rt.tok,rt.bos);s.T=types.SimpleNamespace(tok=rt.tok,tail_ids=tail)
    def _gen_setup(s):return s.LD.Ladder._gen_setup(s)
    def __call__(s,m):return {'_gen_nll':s.LD.Ladder.gen_nll(s,m),'_tail_nll':s.LD.Ladder.tail_nll(s,m)}
def unit_checks(rt,L,s3,ids,dec,n,tol_ce,tol_upd):
    """GPU unit checks on the first n chunks of world A: U1 NAT write path vs native_step, U2 decoy := real stream -> SWD == NAT, U3 write ids are the real chunk, U4 a real decoy changes exactly the slots of each arm and moves CE / update, U5 p=9 plan -> SWDS == NAT. Mirrors e55_units.py (CPU, tiny model); also called by it (U12)."""
    import jax,jax.numpy as jnp
    res={}
    def cos(x,y):return float(jnp.vdot(x,y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y)))
    def delta(newp,m):return jnp.concatenate([(p_-q_).ravel().astype(jnp.float32) for p_,q_ in zip(jax.tree.leaves(newp),jax.tree.leaves(m.inner_parameters()))])
    def run_(arm,p,dd,nch=n):return L.run_stream(arm,p,ids[:nch*CH],decoy_ids=None if dd is None else dd[:nch*CH],nch=nch,tag='unit',log=lambda *x:None)
    with rt.mesh:
        k=min(9,n-1);seq=L._stream_seq(ids[:n*CH],n);pref=rt.prefix(rt.base,seq);Kh,Vh,_=L.record(ids[:n*CH]);Kd,Vd,_=L.record(dec[:n*CH]);Sd=L._stream_seq(dec[:n*CH],n);pd=rt.prefix(rt.base,Sd)
        m=rt.base;st_all=s3.clone_state(rt.state_all);sts={'real':s3.clone_state(rt.state_suffix),'decoy':s3.clone_state(rt.state_suffix)}
        for i in range(k):
            sts['real']=L.read_state(m,sts['real'],rt.chunk(seq,i),pref[i*CH:(i+1)*CH]);sts['decoy']=L.read_state(m,sts['decoy'],rt.chunk(Sd,i),pd[i*CH:(i+1)*CH])
        recs={'real':(Kh,Vh),'decoy':(Kd,Vd)};c=rt.chunk(seq,k);ref={}
        for arm in ('NAT','SWD','SWDS','SWDO','MSK'):
            Kc,Vc,keep,st_w,pl=L.window_inputs(arm,k,2,recs,sts);pw=L.prefix_masked(m,jnp.asarray(ids[k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep));newp,gn,coef,ce=L.write_step(m,st_all,st_w,c,pw,jnp.asarray(keep));ref[arm]=dict(ce=float(ce),gn=float(gn),dl=delta(newp,m),Kc=np.asarray(Kc.astype(jnp.float32)))
        newp2,_st,gn2,coef2,ce2=L.native_step(m,st_all,sts['real'],c,pref[k*CH:(k+1)*CH]);d2=delta(newp2,m)
        res['U1']={'chunk':k,'ce_write_path':ref['NAT']['ce'],'ce_native_step':float(ce2),'abs_ce_diff':abs(ref['NAT']['ce']-float(ce2)),'rel_update_diff':float(jnp.linalg.norm(ref['NAT']['dl']-d2)/jnp.linalg.norm(d2)),'cos_update':cos(ref['NAT']['dl'],d2),'gnorm_write':ref['NAT']['gn'],'gnorm_native':float(gn2)}
        res['U4']={'chunk':k,**{f'ce_{x}':ref[x]['ce'] for x in ref},'cos_update_NAT_vs_SWD':cos(ref['NAT']['dl'],ref['SWD']['dl']),'cos_update_NAT_vs_MSK':cos(ref['NAT']['dl'],ref['MSK']['dl'])}
        slots=lambda x:[j for j in range(1,9) if np.abs(ref[x]['Kc'][:,(8-j)*CH:(9-j)*CH]-ref['NAT']['Kc'][:,(8-j)*CH:(9-j)*CH]).max()>0];res['U4']['prefix_window_slots_changed']={x:slots(x) for x in ('SWD','SWDS','SWDO')}
    seen=[];orig=L.prefix_masked
    def spy(m_,ids_,*x,**kw):seen.append(np.asarray(ids_).copy());return orig(m_,ids_,*x,**kw)
    L.prefix_masked=spy;nat=run_('NAT',2,None);L.prefix_masked=orig
    res['U3']={'write_ids_equal_real':all(np.array_equal(seen[i],ids[i*CH:(i+1)*CH]) for i in range(n)) and len(seen)==n}
    swd_same=run_('SWD',2,ids);swds9=run_('SWDS',9,dec)
    res['U2']={'final_hash_equal':nat['final_hash']==swd_same['final_hash'],'max_abs_ce_diff':max(abs(x['ce']-y['ce']) for x,y in zip(nat['log'],swd_same['log'])),'swapped_slots':sum(e['n_swapped_slots'] for e in swd_same['log'])}
    res['U5']={'final_hash_equal':nat['final_hash']==swds9['final_hash'],'max_abs_ce_diff':max(abs(x['ce']-y['ce']) for x,y in zip(nat['log'],swds9['log'])),'swapped_slots':sum(e['n_swapped_slots'] for e in swds9['log'])}
    res['timing_sec_per_chunk_NAT']=float(np.median([e['sec'] for e in nat['log']]))
    ok=(res['U1']['abs_ce_diff']<=tol_ce and res['U1']['rel_update_diff']<=tol_upd and res['U3']['write_ids_equal_real'] and (res['U2']['final_hash_equal'] or res['U2']['max_abs_ce_diff']<=tol_ce) and (res['U5']['final_hash_equal'] or res['U5']['max_abs_ce_diff']<=tol_ce)
        and res['U4']['prefix_window_slots_changed']=={'SWD':list(range(1,9)),'SWDS':[2,4,6,8],'SWDO':[1,3,5,7]} and abs(res['U4']['ce_SWD']-res['U4']['ce_NAT'])>0 and res['U4']['cos_update_NAT_vs_SWD']<1-1e-6 and res['U2']['swapped_slots']>0 and res['U5']['swapped_slots']==0)
    res['PASS']=bool(ok);res['tolerances']={'ce':tol_ce,'rel_update':tol_upd};return res
def unit(a):
    """GPU unit checks (real tokenizer, 3B model, first --unit-chunks chunks of world A); exit 9 if any fails"""
    enc,w,tf=build(a);S=streams(w,enc);LD,s3,C=lab();rt=LD.Runtime();L=C.Ladder55(rt,save_dir=Path(a.out or '.')/'states'/'unit',canary=None)
    res=unit_checks(rt,L,s3,S['A'][0],S['decoy'][0],a.unit_chunks,a.tol_ce,a.tol_upd);print(json.dumps(res,indent=1,default=float))
    if a.out:Path(a.out).mkdir(parents=True,exist_ok=True);json.dump(res,open(Path(a.out)/f'E55_unit_{a.tag}.json','w'),indent=1,default=float)
    if not res['PASS']:print('E55 UNIT CHECKS FAILED',flush=True);sys.exit(9)
    print('E55 UNIT CHECKS PASS',flush=True)
def rows_for(LD,rt,items):
    import stage2
    tok=rt.tok;slot=stage2.make_slot(tok,None);raw=[]
    for it in items:
        for c in it['cands']:ids,span=stage2.item_ids(tok,slot,it['prefix'],c);raw.append((ids,(span[0]-1,span[1]-1)))
    return raw
def forbid_list(w):return [x['X'] for x in w['facts']+w['ctrl']]+[x['P'] for x in w['facts']]+list(w['unexposed'])+[x['X'] for k_ in ('names','decoy') if k_ in w for x in w[k_]['facts']]+[x['P'] for k_ in ('names','decoy') if k_ in w for x in w[k_]['facts']]
def run(a):
    """stream the world ONCE per arm and seed; inline frozen-weight reads (design 11c) after the chunks of the seed's plan, states saved only at --save-apps. Weights, optimizer, carried KV window are never touched by a read (U15 checks bit-identity)."""
    OUT=Path(a.out or os.environ.get('EXP07_OUT','.'));OUT.mkdir(parents=True,exist_ok=True);arms=a.arms.split(',');seeds=[int(x) for x in a.seeds.split(',')] if a.seeds else [a.seed];plans=seed_plan(a,seeds)
    armseed=dict(x.split(':') for x in a.arm_seeds.split(';')) if a.arm_seeds else {}                  # e.g. 'SWB:55' = arm SWB only for seed 55
    if a.smoke_chunks:assert 'smoke' in a.tag,'a smoke run needs a tag containing smoke'
    for ar in arms:assert ar in ('NAT','SWD','SWB','SWDS','SWDO','MSK'),ar
    LD,s3,C=lab();rt=LD.Runtime();assert int(rt.cfg.model.sliding_window_size)==8192;can=None if a.no_canary else Canary55(LD,rt)
    inline=any(plans[s_][0] or plans[s_][1] for s_ in seeds);rd=None
    if inline:
        import e55_gpu as RG;rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    for sd in seeds:
        my=[ar for ar in arms if ar not in armseed or str(sd) in armseed[ar].split(',')]
        if not my:continue
        ro,ri,sv=plans[sd];enc,w,tf=build(a,sd,read_apps=set(ro)|set(ri)|set(sv),arms=my);S=streams(w,enc);nch=a.smoke_chunks or w['n_chunks'];p=w['p']
        est=host_gib(24,2560,nch*CH,1+len(set(S)-{'A'}));print(f'seed {sd}: host memory for the recordings (3B estimate): {est:.1f} GiB (limit {a.host_gib_limit})',flush=True);assert est<=a.host_gib_limit,'recordings would exceed --host-gib-limit'
        P=R.plan(w,enc);items=P['items'];reads=[a_*p-1 for a_ in sv] if not a.smoke_chunks else [nch-1];canary_chunks=[a_*p-1 for a_ in parse_apps(a.canary_apps,sv)] if not a.smoke_chunks else [nch-1]
        R_=None
        if inline and (ro or ri):
            tok=rt.tok;raw=rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=out_tail_8192(LD,tok,rt.bos,forbid_list(w))
            R_=RG.Reads(rd,raw,len(items),G.NCAND,P['read'],out_tail,rt.base,ro,ri,log=lambda *x:print(*x,flush=True));print(f'seed {sd}: W0 read done',flush=True)
        for ar in my:
            fn=OUT/f'E55_{a.tag}_{ar}_s{sd}.saved.json'
            if fn.exists() and not a.rerun:print('exists, reused:',fn,flush=True);continue
            sdir=OUT/'states'/f'E55_{a.tag}_{ar}_s{sd}';L=C.Ladder55(rt,save_dir=sdir,canary=can)
            ids,sha=S['A'];assert sha==G.stream_sha(G.realize(w)),'stream hash mismatch'
            got={}
            def partial(k,m,ar=ar,sd=sd):
                x=R_.at((k+1)//p,m);got[(k+1)//p]=x
                try:ms=LD.jax.devices()[0].memory_stats();print(f'  [mem] after read at appearance {(k+1)//p}: in use {ms["bytes_in_use"]/2**30:.1f} GiB, peak {ms["peak_bytes_in_use"]/2**30:.1f} GiB',flush=True)
                except Exception:pass                                   # diagnostic only (device memory should stay flat from read to read)
                rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':a.cell,'seed':sd,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':False})[ar];json.dump(rec,open(OUT/f'E55_{a.tag}_{ar}_s{sd}.read.partial.json','w'))
                return {c_:'ok' for c_ in x}
            hook_chunks=[a_*p-1 for a_ in sorted(set(ro)|set(ri))] if R_ else []
            t0=time.monotonic();print(f'== seed {sd} {ar}: {nch} chunks, inline reads (out) {ro}, (in-window) {ri}, states saved {sv}',flush=True)
            r=L.run_stream(ar,p,ids,decoy_ids=S['decoy'][0] if 'decoy' in S else None,bg_ids=S['nofact'][0] if 'nofact' in S else (S['bg'][0] if 'bg' in S else None),read_chunks=reads,canary_chunks=canary_chunks,nch=nch,tag=ar+f'_s{sd}',log=lambda *x:print(*x,flush=True),read_hook=partial if R_ else None,hook_chunks=hook_chunks);r.pop('model',None);r.pop('inline',None)
            lg=r['log'];r['summary']={'mean_update_norm':float(np.mean([e['update_norm'] for e in lg])),'clip_bind_frac':r['clip_bind_frac'],'cum_update_norm':r['cum_update_norm'],'mean_ce':float(np.mean([e['ce'] for e in lg])),'sec_inline_reads':float(sum(e.get('sec_inline_read',0) for e in lg)),'n_inline_reads':sum(1 for e in lg if 'sec_inline_read' in e)}
            print(f'  {ar} seed {sd}: update norm mean {r["summary"]["mean_update_norm"]:.4g}, clip fraction {r["summary"]["clip_bind_frac"]:.2f}, mean CE {r["summary"]["mean_ce"]:.3f}, inline reads {r["summary"]["n_inline_reads"]} in {r["summary"]["sec_inline_reads"]:.0f}s',flush=True)
            r.update(stream_sha=sha,decoy_sha={k_:v_[1] for k_,v_ in S.items() if k_!='A'},decoy_kinds=w['decoys'],world_seed=sd,cell=a.cell,read_apps_out=ro,read_apps_in=ri,save_apps=sv,sec_total=time.monotonic()-t0,code=git_rev(),k=w['k'],n=w['n'],bank=w['bank_kind'])
            if can:
                w0=r['canary_w0'];lim=0.10
                for k,e in r['reads'].items():
                    if 'canary' not in e:continue
                    for key in ('_gen_nll','_tail_nll'):
                        if e['canary'][key]-w0[key]>lim:print(f'WARNING canary {key} at chunk {k}: +{e["canary"][key]-w0[key]:.3f} nat over W0 (limit {lim})',flush=True)
            if R_:
                import e55_gpu as RG
                rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':a.cell,'seed':sd,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'code':git_rev(),'complete':True,'place_len':w['place_len']})[ar];fr=OUT/f'E55_{a.tag}_{ar}_s{sd}.read.json'
                if fr.exists() and not a.rerun:print('exists, not overwritten:',fr)
                else:json.dump(rec,open(fr,'w'));print('wrote',fr)
            json.dump(r,open(fn,'w'));print('wrote',fn,f'{time.monotonic()-t0:.0f}s',flush=True)
def out_tail_8192(LD,tok,bos,forbid):
    """neutral out-of-window context of exactly 8192 tokens: BOS + SQuAD dev paragraphs (as ladder.squad_tail, which stops at 7168), none containing a name / place of the worlds"""
    d=json.load(open(LD.ROOT/'data/squad/dev-v1.1.json'));paras=[p['context'] for x in d['data'] for p in x['paragraphs']]
    paras=[' '.join(x.split()) for x in paras[:400]];paras=[x for x in paras if not any(nm in x for nm in forbid)];text='\n\n'.join(paras);ids=[bos]+LD.stage2.enc(tok,text).ids
    for nm in forbid:assert nm not in text,nm
    assert len(ids)>=8192;return ids[:8192]
def read(a):
    OUT=Path(a.out or os.environ.get('EXP07_OUT','.'));enc,w,tf=build(a);arms=a.arms.split(',');LD,s3,C=lab()
    import e55_gpu as RG,stage2
    fnout=lambda ar:OUT/f'E55_{a.tag}_{ar}_s{a.seed}.read.json'
    todo=[ar for ar in arms if a.rerun or not fnout(ar).exists()]
    if not todo:print('all records exist');return
    sf={}
    for ar in todo:
        sf[ar]={};r=json.load(open(OUT/f'E55_{a.tag}_{ar}_s{a.seed}.saved.json'))
        for ai,k in zip(w['read_apps'],w['read_chunks']):
            e=r['reads'].get(str(k))
            if e is not None:sf[ar][ai]=(e['_file'],e['_hash'])
    rt=LD.Runtime();rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    P=R.plan(w,enc);items=P['items'];tok=rt.tok;slot=stage2.make_slot(tok,None);raw=[]
    for it in items:
        for c in it['cands']:ids,span=stage2.item_ids(tok,slot,it['prefix'],c);raw.append((ids,(span[0]-1,span[1]-1)))
    pl=RG.check_rows(raw,len(items),G.NCAND);print('candidate span length (tokens):',pl,'world place_len:',w['place_len'],flush=True)
    forbid=[x['X'] for x in w['facts']+w['ctrl']+w['decoy']['facts']]+[x['P'] for x in w['facts']+w['decoy']['facts']]
    out_tail=out_tail_8192(LD,tok,rt.bos,forbid);t0=time.monotonic()
    which=('out','in') if a.in_apps else ('out',);in_apps=[int(x) for x in a.in_apps.split(',')] if a.in_apps else None
    only=[int(x) for x in a.apps.split(',')] if a.apps else None
    read_,W0,idx=RG.read_world(rd,raw,len(items),G.NCAND,P['read'],out_tail,{ar:sf[ar] for ar in todo},log=lambda *x:print(*x,flush=True),only_apps=only,which_ctx=which,w0_model=rt.base,in_apps=in_apps)
    info={'cell':a.cell,'seed':a.seed,'k':w['k'],'n':w['n'],'p':w['p'],'bank':w['bank_kind'],'tag':a.tag,'code':git_rev(),'sec_read':time.monotonic()-t0,'place_len':w['place_len'],'out_tail_sha12':hashlib.sha256(np.asarray(out_tail,np.int64).tobytes()).hexdigest()[:12]}
    for ar,rec in RG.make_records(read_,W0,G.item_meta(items),P['read'],idx,info).items():
        if fnout(ar).exists() and not a.rerun:print('exists, not overwritten:',fnout(ar));continue
        json.dump(rec,open(fnout(ar),'w'));print('wrote',fnout(ar))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['dry','run','read','unit']);ap.add_argument('--cell',default='base',choices=list(G.CELLS));ap.add_argument('--seed',type=int,default=55);ap.add_argument('--arms',default='NAT,SWD');ap.add_argument('--tag',default='st0')
    ap.add_argument('--out',default=None);ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--smoke-chunks',type=int,default=0);ap.add_argument('--no-canary',action='store_true');ap.add_argument('--rerun',action='store_true');ap.add_argument('--stub',action='store_true',help='dry only: stub tokenizer and synthetic background (CPU test)');ap.add_argument('--host-gib-limit',type=float,default=100.0)
    ap.add_argument('--save-apps',default=None,help='run: appearances at which states are saved (default 2,4,8,16)');ap.add_argument('--canary-apps',default=None,help='run: appearances with a canary read (default = save-apps; W0 always)');ap.add_argument('--apps',default=None,help='read: appearances to read (default all saved)');ap.add_argument('--in-apps',default=None,help='read: comma list of appearances at which the in-window contexts are also read (e.g. 16); default none')
    ap.add_argument('--seeds',default=None,help='run: comma list of world seeds (one model load for all)');ap.add_argument('--plan',default=None,help="run: per-seed read plan '55=all|1,2,4,8,16;56=2,4,8,16|16' (out-of-window appearances | in-window appearances)");ap.add_argument('--read-apps',default=None,help='run: default out-of-window inline reads (appearances, or all)');ap.add_argument('--arm-seeds',default=None,help="run: restrict an arm to seeds, e.g. 'SWB:55'");ap.add_argument('--decoy',default='names',choices=['names','world'],help='fake window of SWD: names = same chunks with fake names / places (primary), world = another fact world');ap.add_argument('--unit-chunks',type=int,default=12);ap.add_argument('--tol-ce',type=float,default=1e-3);ap.add_argument('--tol-upd',type=float,default=2e-3)
    a=ap.parse_args();{'dry':dry,'run':run,'read':read,'unit':unit}[a.cmd](a)
