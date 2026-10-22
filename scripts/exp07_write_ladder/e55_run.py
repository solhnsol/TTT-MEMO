"""E55 stage-0 driver (server). Subcommands:
  dry   world + stream hashes + read points + plan + host-memory estimate; NO model, NO GPU (CPU_ONLY dry run of the job).
  run   per arm and world (A, B): stream the world ONCE (e55_core.Ladder55.run_stream), save weights at the read chunks (appearance 2, 4, 8, 16), canary (+0.10 nat limit) at W0 and every read chunk. Result files are restart-safe (an existing one is reused); never overwritten (exit 8 unless --rerun).
  read  candidate scores (16 candidates, 3 question frames) of every saved read-point state of the given arms and of W0, 'out' (neutral SQuAD tail) and optionally 'in' (real-stream tails e, e-1, e-3, e-7) contexts (e55_gpu.read_world); one JSON record per arm.
World: gen_world_ctl.build_ctl_world(seed, **CELLS[cell], n_app=16, real tokenizer, default_bg); world B = names swapped inside pairs; decoy stream = world['decoy'] (one stream for A and B). The written chunk is always the real chunk of world A/B.
STATUS: the CPU parts and the swap / composition / reader logic are unit-tested on a tiny random e2e model (e55_units.py); everything here that needs the 3B checkpoint, the real tokenizer, the lab data or the GPU has NEVER been run."""
import os,sys,json,argparse,time,subprocess,hashlib,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
import numpy as np
import gen_world_ctl as G,e55_read as R,e55_swap as E
CH=1024
def git_rev():
    try:return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    except Exception:return None
def build(a):
    if getattr(a,'stub',False):enc=G.StubTok();bg=G._synth_bg(6000);tf=__file__                       # CPU test without tokenizers / lab data
    else:tf=a.tokenizer_file or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);bg=G.default_bg()
    w=G.build_ctl_world(a.seed,n_app=16,enc=enc,bg=bg,**G.CELLS[a.cell]);return enc,w,tf
def streams(w,enc):
    """-> {'A': (ids, sha), 'B': ..., 'decoy': ...}; ids = concatenated chunk ids (n_chunks x 1024)"""
    out={}
    for key,(swap,dec) in {'A':(False,False),'B':(True,False),'decoy':(False,True)}.items():
        ch=G.realize(w,swap,enc,decoy=dec);assert all(len(c['ids'])==CH for c in ch);out[key]=(np.asarray([t for c in ch for t in c['ids']],np.int64),G.stream_sha(ch))
    return out
def host_gib(nl_prefix,hidden,T,nstreams,bytes_=2):return nl_prefix*T*hidden*bytes_*2*nstreams/2**30     # K and V of the prefix layers, merged heads, per stream
def dry(a):
    enc,w,tf=build(a);S=streams(w,enc);P=R.plan(w,enc)
    info={'cell':a.cell,'seed':a.seed,'M':w['M'],'s':w['s'],'p':w['p'],'N':w['N'],'n_chunks':w['n_chunks'],'read_chunks':w['read_chunks'],'stream_sha':{k:v[1] for k,v in S.items()},'tokenizer_sha12':hashlib.sha256(Path(tf).read_bytes()).hexdigest()[:12],'code':git_rev(),
          'items':len(P['items']),'rows':len(P['rows']),'in_window_contexts':{str(k):sorted(v['ctx']) for k,v in P['read'].items()},
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
def run(a):
    OUT=Path(a.out or os.environ.get('EXP07_OUT','.'));OUT.mkdir(parents=True,exist_ok=True)
    enc,w,tf=build(a);S=streams(w,enc);arms=a.arms.split(',');nch=a.smoke_chunks or w['n_chunks']
    if a.smoke_chunks:assert 'smoke' in a.tag,'a smoke run needs a tag containing smoke'
    for ar in arms:assert ar in ('NAT','SWD','SWDS','SWDO','MSK'),ar
    est=host_gib(24,2560,nch*CH,1+int(any(x in ('SWD','SWDS','SWDO') for x in arms)));print(f'host memory for the recordings (3B estimate): {est:.1f} GiB (limit {a.host_gib_limit})',flush=True);assert est<=a.host_gib_limit,'recordings would exceed --host-gib-limit'
    LD,s3,C=lab();rt=LD.Runtime();assert int(rt.cfg.model.sliding_window_size)==8192;can=None if a.no_canary else Canary55(LD,rt)
    sel=[int(x) for x in a.save_apps.split(',')] if a.save_apps else list(G.READ_APPS);ck=[int(x) for x in a.canary_apps.split(',')] if a.canary_apps else sel
    reads=[k_ for ai,k_ in zip(G.READ_APPS,w['read_chunks']) if ai in sel] if not a.smoke_chunks else [nch-1];canary_chunks=[k_ for ai,k_ in zip(G.READ_APPS,w['read_chunks']) if ai in ck] if not a.smoke_chunks else [nch-1]
    for ar in arms:
        for wld in 'AB':
            fn=OUT/f'E55_{a.tag}_{ar}_s{a.seed}_{wld}.saved.json'
            if fn.exists() and not a.rerun:print('exists, reused:',fn,flush=True);continue
            sd=OUT/'states'/f'E55_{a.tag}_{ar}_s{a.seed}'/wld;L=C.Ladder55(rt,save_dir=sd,canary=can)
            ids,sha=S[wld];assert sha==G.stream_sha(G.realize(w,wld=='B',enc)),'stream hash mismatch'
            t0=time.monotonic();print(f'== {ar} world {wld}: {nch} chunks, reads after chunks {reads}',flush=True)
            r=L.run_stream(ar,w['p'],ids,decoy_ids=S['decoy'][0],read_chunks=reads,canary_chunks=canary_chunks,nch=nch,tag=f'{ar}_{wld}',log=lambda *x:print(*x,flush=True));r.pop('model',None)
            r.update(stream_sha=sha,stream_sha_gen=sha,decoy_sha=S['decoy'][1],world_seed=a.seed,cell=a.cell,read_apps=[ai for ai in G.READ_APPS if ai in sel],sec_total=time.monotonic()-t0,code=git_rev(),M=w['M'],s=w['s'],bank=w['bank_kind'])
            if can:
                w0=r['canary_w0'];lim=0.10
                for k,e in r['reads'].items():
                    if 'canary' not in e:continue
                    for key in ('_gen_nll','_tail_nll'):
                        if e['canary'][key]-w0[key]>lim:print(f'WARNING canary {key} at chunk {k}: +{e["canary"][key]-w0[key]:.3f} nat over W0 (limit {lim})',flush=True)
            json.dump(r,open(fn,'w'));print('wrote',fn,f'{time.monotonic()-t0:.0f}s',flush=True)
def read(a):
    OUT=Path(a.out or os.environ.get('EXP07_OUT','.'));enc,w,tf=build(a);arms=a.arms.split(',');LD,s3,C=lab()
    import e55_gpu as RG,stage2
    fnout=lambda ar:OUT/f'E55_{a.tag}_{ar}_s{a.seed}.read.json'
    todo=[ar for ar in arms if a.rerun or not fnout(ar).exists()]
    if not todo:print('all records exist');return
    sf={}
    for ar in todo:
        sf[ar]={}
        for wld in 'AB':
            r=json.load(open(OUT/f'E55_{a.tag}_{ar}_s{a.seed}_{wld}.saved.json'))
            for ai,k in zip(G.READ_APPS,w['read_chunks']):
                e=r['reads'].get(str(k));
                if e is None:continue
                sf[ar].setdefault(ai,{})[wld]=(e['_file'],e['_hash'])
    rt=LD.Runtime();rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    P=R.plan(w,enc);items=P['items'];tok=rt.tok;slot=stage2.make_slot(tok,None);raw=[]
    for it in items:
        for c in it['cands']:ids,span=stage2.item_ids(tok,slot,it['prefix'],c);raw.append((ids,(span[0]-1,span[1]-1)))
    pl=RG.check_rows(raw,len(items),G.NCAND);print('candidate span length (tokens):',pl,'world place_len:',w['place_len'],flush=True)
    forbid=[x['X'] for x in w['persons']+w['ctrl']+w['decoy']['persons']]+[x['P'] for x in w['persons']+w['ctrl']+w['decoy']['persons']]
    out_tail=LD.squad_tail(tok,rt.bos,forbid);t0=time.monotonic()
    which=('out','in') if a.in_apps else ('out',);in_apps=[int(x) for x in a.in_apps.split(',')] if a.in_apps else None
    only=[int(x) for x in a.apps.split(',')] if a.apps else None
    read_,W0=RG.read_world(rd,raw,len(items),G.NCAND,P['read'],out_tail,{ar:sf[ar] for ar in todo},log=lambda *x:print(*x,flush=True),only_apps=only,which_ctx=which,w0_model=rt.base,in_apps=in_apps)
    info={'cell':a.cell,'seed':a.seed,'M':w['M'],'s':w['s'],'p':w['p'],'bank':w['bank_kind'],'tag':a.tag,'code':git_rev(),'sec_read':time.monotonic()-t0,'place_len':w['place_len'],'squad_tail_sha12':hashlib.sha256(np.asarray(out_tail,np.int64).tobytes()).hexdigest()[:12]}
    for ar,rec in RG.make_records(read_,W0,R.item_meta(items),P['read'],info).items():
        if fnout(ar).exists() and not a.rerun:print('exists, not overwritten:',fnout(ar));continue
        json.dump(rec,open(fnout(ar),'w'));print('wrote',fnout(ar))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['dry','run','read','unit']);ap.add_argument('--cell',default='base',choices=list(G.CELLS));ap.add_argument('--seed',type=int,default=55);ap.add_argument('--arms',default='NAT,SWD');ap.add_argument('--tag',default='st0')
    ap.add_argument('--out',default=None);ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--smoke-chunks',type=int,default=0);ap.add_argument('--no-canary',action='store_true');ap.add_argument('--rerun',action='store_true');ap.add_argument('--stub',action='store_true',help='dry only: stub tokenizer and synthetic background (CPU test)');ap.add_argument('--host-gib-limit',type=float,default=100.0)
    ap.add_argument('--save-apps',default=None,help='run: appearances at which states are saved (default 2,4,8,16)');ap.add_argument('--canary-apps',default=None,help='run: appearances with a canary read (default = save-apps; W0 always)');ap.add_argument('--apps',default=None,help='read: appearances to read (default all saved)');ap.add_argument('--in-apps',default=None,help='read: comma list of appearances at which the in-window contexts are also read (e.g. 16); default none')
    ap.add_argument('--unit-chunks',type=int,default=12);ap.add_argument('--tol-ce',type=float,default=1e-3);ap.add_argument('--tol-upd',type=float,default=2e-3)
    a=ap.parse_args();{'dry':dry,'run':run,'read':read,'unit':unit}[a.cmd](a)
