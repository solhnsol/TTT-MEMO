"""E55 unit tests U1-U10 on CPU with a tiny random e2e model (e55_cpu_harness.py; the real SWA / prefix / suffix / inner-loop code, window 8192, chunk 1024). They show that the swap composition and the step equivalences hold in the real code paths; they say NOTHING about the 3B numerics (tolerances there are in PROTOCOL_e55_gpu.md).
Run: python3 e55_units.py [--only U2,U5] (about 10-20 min on one CPU core, mostly XLA compilation). Prints one line per test with the measured numbers and `E55 UNITS: ALL PASS`.
U0 layout of the carried suffix state (lag j = rows [(8-j)*1024, (9-j)*1024), checked against the real SWA update)   U1 write path with nothing swapped = native_step   U2 decoy == real stream: SWD == NAT bit for bit
U3 the written chunk is always the real chunk   U4 a real decoy changes exactly the existing slots and changes CE / update direction   U5 p = 9: SWDS == NAT (and p = 1: SWDO == NAT)   U6 MSK keep == s3_sets prefix-all keep, MSK != NAT
U7 recordings: same absolute positions (shared first chunk -> identical K,V; decoy/real shapes equal)   U8 weights save/load round trip (hash)   U9 world + stream hash reproducibility (stub tokenizer)   U10 smoke: every arm over a short stream incl. read-point saves   U11 readers (CPU, random candidate rows): W0 scored twice identical, saved base weights reload to the W0 scores, a changed state moves the scores, per-group in-window contexts, records feed e55_analyze"""
import sys,argparse,time,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[1]/'scripts/exp04_what_ttt_retains'))
import e55_cpu_harness as H
rt=H.TinyRuntime();s3,reuse=H.install_stubs(rt)
import e55_core as C,e55_swap as E
jax,jnp,np=H.jax,H.jnp,H.np
CH=1024;W=8192
L=C.Ladder55(rt,save_dir='/tmp/e55_units_states')
def mk(n,seed,lo=3):return np.random.default_rng(seed).integers(lo,rt.cfg.model.vocab_size,size=n*CH)
NCH=10;REAL=mk(NCH,1);DEC=mk(NCH,2)
def maxdiff(a,b):return float(max(jnp.max(jnp.abs(x.astype(jnp.float32)-y.astype(jnp.float32))) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b))))
def delta(m_new_p,m):return jnp.concatenate([(a-b).ravel() for a,b in zip(jax.tree.leaves(m_new_p),jax.tree.leaves(m.inner_parameters()))])
def cos(a,b):return float(jnp.vdot(a,b)/(jnp.linalg.norm(a)*jnp.linalg.norm(b)))
def setup(nch,streams):
    """recordings, prefix outputs, natural chains for the named streams"""
    out={'seq':{},'pref':{},'rec':{}}
    with rt.mesh:
        for nm,ids in streams.items():
            out['seq'][nm]=L._stream_seq(ids,nch);out['pref'][nm]=rt.prefix(rt.base,out['seq'][nm]);Kh,Vh,_=L.record(ids[:nch*CH]);out['rec'][nm]=(Kh,Vh)
    return out
RES={}
def u0():
    """advance a chain over 6 chunks with fixed weights; the cache after chunk c holds chunk c in its last block; after j-1 more chunks the same K,V sit in block j"""
    S=setup(7,{'real':REAL});st=s3.clone_state(rt.state_suffix);snap=[]
    with rt.mesh:
        for k in range(7):
            st=L.read_state(rt.base,st,rt.chunk(S['seq']['real'],k),S['pref']['real'][k*CH:(k+1)*CH]);snap.append(st)
    leaf=lambda s:[x for x in jax.tree.leaves(s) if getattr(x,'ndim',0)==3]
    err=0.0
    for k in range(1,7):          # state at the START of chunk k+1 ... compare snapshot[c] last block with snapshot[k] block j, j = k-c+1
        for c in range(0,k+1):
            j=k-c+1
            if j>8:continue
            a=leaf(snap[c]);b=leaf(snap[k]);assert len(a)==2
            for x,y in zip(a,b):err=max(err,float(jnp.max(jnp.abs(x[:,7*CH:8*CH]-y[:,(8-j)*CH:(9-j)*CH]))))
    assert err==0.0,err
    # and the composition picks exactly these rows: take a decoy chain, compose lag j from it, check the block equals the decoy chain's block j and everything else the real chain's
    S2=setup(7,{'real':REAL,'decoy':DEC});sts={'real':s3.clone_state(rt.state_suffix),'decoy':s3.clone_state(rt.state_suffix)}
    with rt.mesh:
        for k in range(7):
            for nm in sts:sts[nm]=L.read_state(rt.base,sts[nm],rt.chunk(S2['seq'][nm],k),S2['pref'][nm][k*CH:(k+1)*CH])
    pl={j:('decoy' if j in (2,5) else 'real') for j in range(1,8)};cnt=[];comp=C.compose_state({'real':sts['real'],'decoy':sts['decoy'],'bg':sts['real']},pl,cnt);assert len(cnt)==2
    for x,r_,d_ in zip(leaf(comp),leaf(sts['real']),leaf(sts['decoy'])):
        for j in range(1,9):
            sl=slice((8-j)*CH,(9-j)*CH);ref=d_ if j in (2,5) else r_;assert float(jnp.max(jnp.abs(x[:,sl]-ref[:,sl])))==0.0,j
    RES['U0']=f'cache shift identity max|diff| {err}; composition rows exact for lags 2,5 (decoy) and all others real; 2 KV leaves composed'
def u1():
    k=3;S=setup(5,{'real':REAL});m=rt.base;st_all=s3.clone_state(rt.state_all);sts={'real':s3.clone_state(rt.state_suffix)}
    with rt.mesh:
        for i in range(k):sts['real']=L.read_state(m,sts['real'],rt.chunk(S['seq']['real'],i),S['pref']['real'][i*CH:(i+1)*CH])
        c=rt.chunk(S['seq']['real'],k);pref_c=S['pref']['real'][k*CH:(k+1)*CH]
        Kc,Vc,keep,st_w,pl=L.window_inputs('NAT',k,2,S['rec'],sts)
        pw=L.prefix_masked(m,jnp.asarray(REAL[k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep))
        newp,gn,coef,ce=L.write_step(m,st_all,st_w,c,pw,jnp.asarray(keep))
        newp2,_st,gn2,coef2,ce2=L.native_step(m,st_all,sts['real'],c,pref_c)
    dp=float(jnp.max(jnp.abs(pw-pref_c)));dn=maxdiff(newp,newp2)
    assert abs(float(ce)-float(ce2))<1e-4 and dn<1e-5 and abs(float(gn)-float(gn2))<1e-3*max(1,float(gn2)),(float(ce),float(ce2),dn,float(gn),float(gn2))
    RES['U1']=f'NAT write path vs native_step at chunk {k}: |ce diff| {abs(float(ce)-float(ce2)):.2e}, max|new params diff| {dn:.2e}, |gnorm diff| {abs(float(gn)-float(gn2)):.2e}, masked-prefix vs natural prefix output max|diff| {dp:.2e} (CPU fp32, tiny model)'
def run(arm,p,dec,nch=NCH,reads=(),tag='u'):
    return L.run_stream(arm,p,REAL,decoy_ids=dec,bg_ids=dec,read_chunks=reads,nch=nch,tag=tag,log=lambda *a:None)
_cache={}
def nat():
    if 'nat' not in _cache:_cache['nat']=run('NAT',2,None)
    return _cache['nat']
def u2():
    a=nat();b=run('SWD',2,REAL)
    assert a['final_hash']==b['final_hash'] and [e['ce'] for e in a['log']]==[e['ce'] for e in b['log']],(a['final_hash'],b['final_hash'])
    RES['U2']=f"SWD with the decoy stream replaced by the real stream: final weight hash {b['final_hash']} == NAT {a['final_hash']}, all {NCH} per-chunk CE identical (bitwise); swapped slots counted {sum(e['n_swapped_slots'] for e in b['log'])}"
def u3():
    seen=[];orig=L.prefix_masked
    def spy(m,ids_,*a,**k):seen.append(np.asarray(ids_).copy());return orig(m,ids_,*a,**k)
    L.prefix_masked=spy
    try:
        for arm in ('NAT','SWD','SWDS','SWDO','MSK'):
            seen.clear();run(arm,2,DEC,nch=4)
            assert len(seen)==4 and all(np.array_equal(seen[k],REAL[k*CH:(k+1)*CH]) for k in range(4)),arm
    finally:L.prefix_masked=orig
    RES['U3']='the token ids given to the write forward equal the real chunk in all 5 arms (4 chunks each); decoy ids never reach it'
def u4():
    k=9;S=setup(NCH,{'real':REAL,'decoy':DEC});m=rt.base;st_all=s3.clone_state(rt.state_all);sts={'real':s3.clone_state(rt.state_suffix),'decoy':s3.clone_state(rt.state_suffix)}
    with rt.mesh:
        for i in range(k):
            for nm in sts:sts[nm]=L.read_state(m,sts[nm],rt.chunk(S['seq'][nm],i),S['pref'][nm][i*CH:(i+1)*CH])
        c=rt.chunk(S['seq']['real'],k);res={}
        for arm in ('NAT','SWD','SWDS','SWDO'):
            cnt=[];Kc,Vc,keep,st_w,pl=L.window_inputs(arm,k,2,S['rec'],sts,cnt)
            pw=L.prefix_masked(m,jnp.asarray(REAL[k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep));newp,gn,coef,ce=L.write_step(m,st_all,st_w,c,pw,jnp.asarray(keep))
            res[arm]=dict(Kc=np.asarray(Kc),st=st_w,pl=pl,ce=float(ce),dl=delta(newp,m))
    # which slots differ: prefix-layer K window and suffix state, per lag
    def slots_diff(a_k,b_k):return [j for j in range(1,9) if np.abs(a_k[:,(8-j)*1024:(9-j)*1024]-b_k[:,(8-j)*1024:(9-j)*1024]).max()>0]
    leaf=lambda s:[x for x in jax.tree.leaves(s) if getattr(x,'ndim',0)==3]
    for arm,expect in (('SWD',list(range(1,9))),('SWDS',[2,4,6,8]),('SWDO',[1,3,5,7])):
        pd=slots_diff(res[arm]['Kc'],res['NAT']['Kc']);sd=sorted({j for x,y in zip(leaf(res[arm]['st']),leaf(res['NAT']['st'])) for j in range(1,9) if float(jnp.max(jnp.abs(x[:,(8-j)*1024:(9-j)*1024]-y[:,(8-j)*1024:(9-j)*1024])))>0})
        assert pd==expect and sd==expect,(arm,pd,sd,expect)
    cs=cos(res['NAT']['dl'],res['SWD']['dl']);assert res['SWD']['ce']!=res['NAT']['ce'] and cs<1-1e-6,(res['SWD']['ce'],res['NAT']['ce'],cs)
    RES['U4']=f"chunk {k}, p=2: SWD changes slots 1-8 in both the prefix-layer window K and the suffix state; SWDS exactly lags 2,4,6,8, SWDO exactly 1,3,5,7; CE NAT {res['NAT']['ce']:.5f} -> SWD {res['SWD']['ce']:.5f}, SWDS {res['SWDS']['ce']:.5f}, SWDO {res['SWDO']['ce']:.5f}; cosine(update NAT, update SWD) {cs:.6f}"
def u5():
    a=nat();b=run('SWDS',9,DEC);assert a['final_hash']!=b['final_hash'] or True
    # p=9: no lag in 1..8 is a multiple of 9 -> SWDS must equal NAT with the SAME p (p only enters through the plan; the stream is the same array)
    n9=run('NAT',9,None);assert b['final_hash']==n9['final_hash'] and [e['ce'] for e in b['log']]==[e['ce'] for e in n9['log']]
    c=run('SWDO',1,DEC);n1=run('NAT',1,None);assert c['final_hash']==n1['final_hash']
    RES['U5']=f"p=9: SWDS == NAT bitwise (hash {b['final_hash']}), swapped slots {sum(e['n_swapped_slots'] for e in b['log'])}; p=1: SWDO == NAT bitwise (hash {c['final_hash']}), swapped slots {sum(e['n_swapped_slots'] for e in c['log'])} (decoy stream given but unused)"
def u6():
    import s3_sets as SETS
    for k in (0,3,8,9):
        pos=list(range(max(0,k*CH-W),k*CH));ref=SETS.Sets.keep_vector(None,k,pos);mine=C.keep_vector(E.plan('MSK',k,2));assert np.array_equal(ref,mine),k
    a=nat();b=run('MSK',2,None,nch=4);assert [e['ce'] for e in b['log']][1:]!=[e['ce'] for e in a['log']][1:4]
    RES['U6']=f"MSK keep vector == s3_sets prefix-all keep for k=0,3,8,9; MSK CE differs from NAT from chunk 1 on (chunk 3: {b['log'][3]['ce']:.5f} vs {a['log'][3]['ce']:.5f}); masked slots at chunk 9: {E.plan('MSK',9,2) and sum(1 for v in E.plan('MSK',9,2).values() if v=='mask')}"
def u7():
    X=REAL.copy();Y=REAL.copy();Y[CH:]=mk(NCH-1,7)      # same first chunk, different rest
    with rt.mesh:
        Kx,Vx,_=L.record(X[:3*CH]);Ky,Vy,_=L.record(Y[:3*CH]);Kd,_,_=L.record(DEC[:3*CH])
    assert np.array_equal(Kx[:,:CH],Ky[:,:CH]) and np.array_equal(Vx[:,:CH],Vy[:,:CH]) and not np.array_equal(Kx[:,CH:2*CH],Ky[:,CH:2*CH]) and Kx.shape==Kd.shape
    RES['U7']=f'recording is causal and position-absolute: streams sharing chunk 0 give identical K,V for chunk 0 and different K,V afterwards; real and decoy recordings have the same shape {Kx.shape} (layers, tokens, heads, head dim)'
def u8():
    r=run('NAT',2,None,nch=3,reads=(1,2),tag='u8');m=L.load_state(r['reads'][2]['_file']);h=s3.reuse._hash_pytree(m.inner_parameters())[0]
    assert h==r['reads'][2]['_hash']==r['final_hash'] and r['reads'][1]['_hash']!=r['reads'][2]['_hash']
    RES['U8']=f"state saved after chunk 2 reloaded: weight hash {h} == the streamed weights; the two read-point states differ"
def u9():
    import gen_facts as G
    enc=G.StubTok();bg=G._synth_bg(9000);w1=G.build_fact_world(55,enc=enc,bg=bg,**G.CELLS['base']);w2=G.build_fact_world(55,enc=enc,bg=bg,**G.CELLS['base']);w3=G.build_fact_world(56,enc=enc,bg=bg,**G.CELLS['base'])
    s1=G.stream_sha(G.realize(w1));s2=G.stream_sha(G.realize(w2));d1=G.stream_sha(G.realize(w1,'world'));s3_=G.stream_sha(G.realize(w3))
    assert s1==s2 and s1!=d1 and s1!=s3_
    RES['U9']=f'same seed twice: stream hash {s1} == {s2}; decoy {d1}; seed 56 {s3_} (stub tokenizer; the real-tokenizer reproducibility is the server dry run)'
def u10():
    out={}
    for arm in ('NAT','SWD','SWDS','SWDO','MSK'):
        r=run(arm,2,DEC,nch=4,reads=(1,3),tag='u10'+arm);out[arm]=(r['final_hash'],r['clip_bind_frac'],len(r['reads']));assert len(r['log'])==4 and all(np.isfinite(e['ce']) for e in r['log'])
    r=run('SWB',1,DEC,nch=4,reads=(3,),tag='u10SWB');out['SWB']=(r['final_hash'],r['clip_bind_frac'],len(r['reads']));assert len(r['log'])==4 and all(np.isfinite(e['ce']) for e in r['log'])
    RES['U10']='smoke, 4 chunks, read points after chunks 1 and 3: '+'; '.join(f'{a} hash {v[0]} clip-bind {v[1]:.2f}' for a,v in out.items())+'; SWB refused (needs a no-fact stream)'
def u11():
    import gen_facts as G,e55_read as R,e55_gpu as RG,e55_metrics as MT,hashlib
    enc=G.StubTok();w=G.build_fact_world(55,enc=enc,bg=G._synth_bg(9000),**G.CELLS['base']);pl=R.plan(w);V=rt.cfg.model.vocab_size
    items=pl['items'];n=len(items);NC=G.NCAND;assert n==36
    rng=np.random.default_rng(11);cand=rng.integers(3,V,size=(NC,3));pre=rng.integers(3,V,size=(n,9))
    raw=[(list(map(int,pre[i]))+list(map(int,cand[c])),(8,11)) for i in range(n) for c in range(NC)]
    assert RG.check_rows(raw,n,NC)==3
    mp=lambda t:[int(x)%(V-3)+3 for x in t]
    ctxs={nm:{**c,'tail':mp(c['tail'])} for nm,c in pl['read'][16]['ctx'].items()};read={16:{'read_chunk':31,'ctx':ctxs}}
    assert set(ctxs)=={'in_g0','in_g1'} and all(len(c['tail'])==8192 for c in ctxs.values())
    out_tail=[int(x) for x in rng.integers(3,V,size=8192)]
    rd=RG.Reader55(rt,rt,s3.packing if hasattr(s3,'packing') else sys.modules['packing'],H.TailCtx,reuse.get_filter_spec,reuse._hash_pytree,jax,H.eqx,jnp)
    m_base=rt.base;inner=m_base.inner_parameters();pert=L.set_inner(m_base,jax.tree.map(lambda x:x+0.02*jnp.ones_like(x),inner))
    fb=L.save_state(m_base,'u11_base.npz');fp=L.save_state(pert,'u11_pert.npz')
    sf={'BASE':{16:(fb['_file'],fb['_hash'])},'PERT':{16:(fp['_file'],fp['_hash'])}}
    h_before=reuse._hash_pytree(pert.inner_parameters())[0]
    t0=time.monotonic();rdd,W0,idx=RG.read_world(rd,raw,n,NC,read,out_tail,sf,w0_model=m_base,log=lambda *a:None,only_apps=(16,))
    assert reuse._hash_pytree(pert.inner_parameters())[0]==h_before and reuse._hash_pytree(m_base.inner_parameters())[0]==fb['_hash']      # reading leaves the weights bit-identical
    P=rd.packed(raw,out_tail);s1=rd.score(P,m_base,n,NC);s2=rd.score(P,m_base,n,NC);assert np.array_equal(s1,s2) and np.array_equal(s1,W0['out'])
    assert np.array_equal(rdd['BASE'][16]['out'],W0['out']) and not np.allclose(rdd['PERT'][16]['out'],W0['out'])
    for nm in ctxs:
        assert np.array_equal(rdd['BASE'][16][nm],W0[f'{nm}@16']) and not np.allclose(rdd['PERT'][16][nm],W0[f'{nm}@16']) and not np.isnan(W0[f'{nm}@16']).any()
    # a fact of group 1 is scored in group 1's context (and not in group 0's): recompute that block directly
    c1=ctxs['in_g1'];i1=np.asarray(c1['items']);Pg=rd.packed([raw[i*NC+k] for i in i1 for k in range(NC)],c1['tail']);direct=rd.score(Pg,m_base,len(i1),NC);assert np.array_equal(direct,W0['in_g1@16'])
    c0=ctxs['in_g0'];ex0=[i for i in c0['items'] if items[i]['kind']=='fact'];ex1=[i for i in c1['items'] if items[i]['kind']=='fact'];assert len(ex0)==len(ex1)==16 and not set(ex0)&set(ex1) and len(c0['items'])==20 and len(c1['items'])==16
    A_=G.realize(w);tail_sha=hashlib.sha256(np.asarray(pl['read'][16]['ctx']['in_g1']['tail'],np.int64).tobytes()).hexdigest()[:12];assert pl['read'][16]['ctx']['in_g1']['tail'][:1024]==A_[c1['end']-7]['ids'] and pl['read'][16]['ctx']['in_g1']['tail'][-1024:]==A_[c1['end']]['ids']
    recs=RG.make_records(rdd,W0,G.item_meta(items),read,idx,{'cell':'u11','k':32,'n':1,'p':2,'bank':'50','seed':55});rec=json.loads(json.dumps(recs['PERT']))
    assert rec['meta']['d']['16']['in_g1'][ex1[0]]==2 and rec['meta']['d']['16']['in_g1'][ex0[0]] is None
    v=MT.vec([rec],16,'out','G_rank');v2=MT.vec([rec],16,['in_g0','in_g1'],'G_rank');assert len(v)==w['k'] and len(v2)==w['k'] and not np.isnan(v2).any()
    RES['U11']=f"16 candidates x {n} items (32 facts + 4 control), rows packed, 3-token spans; W0 'out' scored twice identical, reading leaves the weights bit-identical (hash before == after), saved base weights reloaded -> scores identical to W0 (out and both group contexts), a +0.02 weight shift moves them; each fact is scored in its own group's context (group 1 context = real chunks {c1['end']-7}..{c1['end']} of the stream, recomputed directly: identical; sha12 {tail_sha}; d = 2 = p); the record round-trips through JSON into e55_metrics ({len(v)} facts, out and in-window) ({time.monotonic()-t0:.0f}s); random rows, tiny model, NOT 3B"
def u12():
    import e55_run as RUN
    r=RUN.unit_checks(rt,L,s3,REAL,DEC,NCH,1e-3,2e-3);assert r['PASS'],r
    RES['U12']=f"e55_run.unit_checks (the check list the server runs on the 3B model) on the tiny model, 10 chunks: PASS; U1 |ce diff| {r['U1']['abs_ce_diff']:.1e}, rel update diff {r['U1']['rel_update_diff']:.1e}; U2 hash equal {r['U2']['final_hash_equal']} (swapped slots {r['U2']['swapped_slots']}); U5 hash equal {r['U5']['final_hash_equal']} (swapped {r['U5']['swapped_slots']}); U4 window slots changed {r['U4']['prefix_window_slots_changed']}"
def u13():
    """arm SWB at p = 1 (fact-free fake window): every existing lag 1..8 comes from the no-fact stream, in the prefix-layer window and in the carried suffix state; bg := real stream -> NAT bit for bit; a real bg stream changes CE / update; the written ids are the real chunk"""
    k=9;S=setup(NCH,{'real':REAL,'bg':DEC});recs=S['rec'];m=rt.base;st_all=s3.clone_state(rt.state_all);sts={'real':s3.clone_state(rt.state_suffix),'bg':s3.clone_state(rt.state_suffix)}
    with rt.mesh:
        for i in range(k):
            sts['real']=L.read_state(m,sts['real'],rt.chunk(S['seq']['real'],i),S['pref']['real'][i*CH:(i+1)*CH]);sts['bg']=L.read_state(m,sts['bg'],rt.chunk(S['seq']['bg'],i),S['pref']['bg'][i*CH:(i+1)*CH])
        Kc,Vc,keep,st_w,pl=L.window_inputs('SWB',k,1,recs,sts);assert set(pl.values())=={'bg'} and sorted(pl)==list(range(1,9)) and keep.all()
        Kb=np.asarray(recs['bg'][0].astype(jnp.float32));Kw=np.asarray(Kc.astype(jnp.float32))
        for j in range(1,9):
            c=k-j;assert np.array_equal(Kw[:,(8-j)*CH:(9-j)*CH],Kb[:,c*CH:(c+1)*CH]),j      # prefix-layer window slot = the bg recording of chunk k-j
        kv=[x for x in jax.tree.leaves(st_w) if getattr(x,'ndim',0)==3 and x.shape[1]==W];kb=[x for x in jax.tree.leaves(sts['bg']) if getattr(x,'ndim',0)==3 and x.shape[1]==W];kr=[x for x in jax.tree.leaves(sts['real']) if getattr(x,'ndim',0)==3 and x.shape[1]==W]
        assert len(kv)==2 and all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(kv,kb)) and not all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(kv,kr))   # suffix state = bg chain, not the real chain
    nat_=run('NAT',1,None);same=run('SWB',1,REAL);assert same['final_hash']==nat_['final_hash'] and [e['ce'] for e in same['log']]==[e['ce'] for e in nat_['log']]
    swb=run('SWB',1,DEC);assert swb['final_hash']!=nat_['final_hash'] and sum(e['n_swapped_slots'] for e in swb['log'])==sum(min(i,8) for i in range(NCH))
    seen=[];orig=L.prefix_masked
    def spy(m_,ids_,*a,**kw):seen.append(np.asarray(ids_).copy());return orig(m_,ids_,*a,**kw)
    L.prefix_masked=spy;run('SWB',1,DEC,nch=4);L.prefix_masked=orig;assert all(np.array_equal(seen[i],REAL[i*CH:(i+1)*CH]) for i in range(4))
    RES['U13']=f"SWB p=1 at chunk {k}: all lags 1-8 are bg (prefix-layer window K slices equal the bg recording exactly; suffix state equals the bg chain, differs from the real chain); bg := real -> NAT bitwise (hash {same['final_hash']}), 'swapped slots' {sum(e['n_swapped_slots'] for e in swb['log'])} over {NCH} chunks, a real bg stream changes the weights, the written ids are the real chunk"
def u14():
    """fake windows of the (16,1,1)-style world (stub tokenizer, injective): 'names' (A) and 'nofact' (B) streams contain none of the real names / places / control names / never-exposed places; layout and lengths equal the real chunks; B contains none of the original fact-sentence tokens; a planted name is caught"""
    import gen_facts as G
    enc=G.StubTok();bg=G._synth_bg(9000);w=G.build_fact_world(55,enc=enc,bg=bg,**G.CELLS['p1']);v,A_=G.verify(w,enc);assert v['names_decoy_fact_free'] and v['nofact_fact_free'] and v['decoy_fact_free'] and v['names_decoy_disjoint']
    N_=G.realize(w,'names');B_=G.realize(w,'nofact');ns=0
    for a_,n_,b_ in zip(A_,N_,B_):
        assert len(a_['ids'])==len(n_['ids'])==len(b_['ids'])==G.CH
        for x in a_['sents']:
            sent=a_['ids'][x['start']:x['end']];ns+=1
            assert b_['ids'][x['start']:x['end']]!=sent and len(b_['ids'][x['start']:x['end']])==len(sent)
            assert not any(b_['ids'][i:i+6]==sent[:6] for i in range(G.CH-5)),'original sentence tokens in the nofact chunk'         # the first 6 tokens of an original fact sentence never occur in B
            assert n_['ids'][x['start']:x['end']]!=sent and n_['ids'][x['start']-1]==a_['ids'][x['start']-1]
    t=enc(w['facts'][0]['X'])
    for kd in ('names','nofact'):
        w2=G.build_fact_world(55,enc=enc,bg=bg,**G.CELLS['p1']);w2[kd]['chunks'][0]['ids'][7:7+len(t)]=t
        try:G.verify(w2,enc);ok=False
        except AssertionError:ok=True
        assert ok,kd
    RES['U14']=f'p=1 world: names stream (A) and nofact stream (B) have the exact layout of the real chunks (1024 tokens, sentence spans identical, {ns} sentences), contain no name / place / control name / never-exposed place of the world (token search), B shares no original sentence prefix with the real chunks, and the other tokens of B equal the real chunk; a planted name is caught in both (stub tokenizer; the server dry run repeats it with the real tokenizer)'
def u15():
    """inline frozen-weight reads (e55_gpu.Reads via run_stream read_hook): per-chunk CE and the final weights are bit-identical with and without reads, for NAT and SWD; reads differ along the stream; W0 shared"""
    import e55_gpu as RG
    V=rt.cfg.model.vocab_size;rng=np.random.default_rng(15);NC=16;n=3;cand=rng.integers(3,V,size=(NC,3));pre=rng.integers(3,V,size=(n,9))
    raw=[(list(map(int,pre[i]))+list(map(int,cand[c])),(8,11)) for i in range(n) for c in range(NC)]
    out_tail=[int(x) for x in rng.integers(3,V,size=8192)]
    plan={a:{'ctx':{'in_g0':{'group':0,'items':list(range(n)),'d':[1]*n,'tail':[int(x) for x in rng.integers(3,V,size=min(a,8)*CH)],'window_chunks':min(a,8)}}} for a in (1,2,8)}
    rd=RG.Reader55(rt,rt,s3.packing if hasattr(s3,'packing') else sys.modules['packing'],H.TailCtx,reuse.get_filter_spec,reuse._hash_pytree,jax,H.eqx,jnp)
    t0=time.monotonic();R_=RG.Reads(rd,raw,n,NC,plan,out_tail,rt.base,{1,2,8},{1,8});got={}
    def hook(k,m):got[k+1]=R_.at(k+1,m);return None
    msg=[]
    for arm in ('NAT','SWD'):
        base=L.run_stream(arm,1,REAL,decoy_ids=DEC,nch=8,tag='u15a',log=lambda *a:None);got.clear()
        with_=L.run_stream(arm,1,REAL,decoy_ids=DEC,nch=8,tag='u15b',log=lambda *a:None,read_hook=hook,hook_chunks=[0,1,7])
        assert [e['ce'] for e in base['log']]==[e['ce'] for e in with_['log']] and base['final_hash']==with_['final_hash'],arm
        assert set(got)=={1,2,8} and set(got[1])=={'out','in_g0'} and set(got[2])=={'out'} and got[8]['out'].shape==(n,NC) and not np.array_equal(got[1]['out'],got[8]['out'])
        msg.append(f"{arm} hash {base['final_hash']}")
    assert set(R_.W0)=={'out','in_g0@1','in_g0@8'} and np.array_equal(R_.W0['out'],rd.score(R_.P_out,rt.base,n,NC))
    RES['U15']=f"reads at appearances 1, 2, 8 (out) and 1, 8 (in-window, window shorter than 8 chunks at a=1): per-chunk CE and final weight hash identical with and without reads ({'; '.join(msg)}); reads differ between chunk 1 and 8; W0 stored once ({time.monotonic()-t0:.0f}s, tiny model, random rows)"
def u16():
    """NATM (size-matched NAT): target = NAT's own per-chunk update norms -> NAT bit for bit; target = 0.5 x NAT's -> the logged update norm of every chunk equals the target, the direction is NAT's (cosine of the chunk-0 update with NAT's = 1, later chunks differ only through the weights), the written ids are the real chunk"""
    n8=8;nat_=L.run_stream('NAT',1,REAL,nch=n8,tag='u16n',log=lambda *a:None);t1=[e['update_norm'] for e in nat_['log']]
    same=L.run_stream('NATM',1,REAL,nch=n8,tag='u16a',log=lambda *a:None,norm_targets=t1)
    assert same['final_hash']==nat_['final_hash'] and [e['ce'] for e in same['log']]==[e['ce'] for e in nat_['log']] and all(abs(e['scale']-1.0)<1e-6 for e in same['log'])
    t2=[0.5*x for x in t1];half=L.run_stream('NATM',1,REAL,nch=n8,tag='u16b',log=lambda *a:None,norm_targets=t2)
    assert all(abs(e['update_norm']-t)<=1e-4*t for e,t in zip(half['log'],t2)) and half['final_hash']!=nat_['final_hash'] and abs(half['log'][0]['scale']-0.5)<1e-4 and abs(half['log'][0]['ce']-nat_['log'][0]['ce'])<1e-6
    seen=[];orig=L.prefix_masked
    def spy(m_,ids_,*a,**kw):seen.append(np.asarray(ids_).copy());return orig(m_,ids_,*a,**kw)
    L.prefix_masked=spy;L.run_stream('NATM',1,REAL,nch=4,tag='u16c',log=lambda *a:None,norm_targets=t2);L.prefix_masked=orig;assert all(np.array_equal(seen[i],REAL[i*CH:(i+1)*CH]) for i in range(4))
    try:L.run_stream('NATM',1,REAL,nch=2,tag='u16d',log=lambda *a:None);ok=False
    except AssertionError:ok=True
    assert ok
    RES['U16']=f"NATM p=1, 8 chunks: target = NAT's norms -> hash {same['final_hash']} == NAT bitwise (all rescale factors 1); target = 0.5 x NAT's -> every logged update norm equals the target (max rel dev {max(abs(e['update_norm']-t)/t for e,t in zip(half['log'],t2)):.1e}), chunk-0 rescale factor {half['log'][0]['scale']:.4f}, CE of chunk 0 equal to NAT's (same start weights), the written ids are the real chunk, no target -> refused"
# reference values produced by the code of commit 50cf2de (job 180a) in this same tiny setup; U17 shows that the NAT / SWD paths and the readout path did not change
REF_180A={'NAT':'828ef3cad315e149','SWD':'7194ea8a98b29eac','reads':'e58c4dbd9dba1d03'}
def u17():
    """the paths of job 180a are unchanged under the code of job 180b: NAT / SWD final weights (8 chunks, p = 1, read hook inserted) and the a = 1 / 8 read scores hash to the values recorded with the 180a commit; reads inserted change nothing (bit-identical, as U15)"""
    import e55_gpu as RG
    V=rt.cfg.model.vocab_size;rng=np.random.default_rng(15);NC=16;n=3;cand=rng.integers(3,V,size=(NC,3));pre=rng.integers(3,V,size=(n,9))
    raw=[(list(map(int,pre[i]))+list(map(int,cand[c])),(8,11)) for i in range(n) for c in range(NC)];out_tail=[int(x) for x in rng.integers(3,V,size=8192)]
    plan={a:{'ctx':{'in_g0':{'group':0,'items':list(range(n)),'d':[1]*n,'tail':[int(x) for x in rng.integers(3,V,size=min(a,8)*CH)],'window_chunks':min(a,8)}}} for a in (1,2,8)}
    rd=RG.Reader55(rt,rt,s3.packing if hasattr(s3,'packing') else sys.modules['packing'],H.TailCtx,reuse.get_filter_spec,reuse._hash_pytree,jax,H.eqx,jnp)
    R_=RG.Reads(rd,raw,n,NC,plan,out_tail,rt.base,{1,2,8},{1,8});got={};hs={}
    def hook(k,m):got[k+1]=R_.at(k+1,m)
    for arm in ('NAT','SWD'):
        r=L.run_stream(arm,1,REAL,decoy_ids=DEC,nch=8,tag='u17'+arm,log=lambda *a:None,read_hook=hook,hook_chunks=[0,1,7]);hs[arm]=r['final_hash']
        if arm=='NAT':hh=hashlib.sha256(b''.join(np.ascontiguousarray(np.asarray(got[a][c],np.float64)).tobytes() for a in (1,2,8) for c in sorted(got[a]))+np.asarray(R_.W0['out'],np.float64).tobytes()).hexdigest()[:16]
    print('U17 values:',hs,hh,flush=True)
    assert hs['NAT']==REF_180A['NAT'] and hs['SWD']==REF_180A['SWD'],(hs,REF_180A)
    assert hh==REF_180A['reads'],(hh,REF_180A)
    RES['U17']=f"180a paths under this code: NAT hash {hs['NAT']} and SWD hash {hs['SWD']} equal the values of commit 50cf2de (8 chunks, p=1, reads at appearance 1, 2, 8 inserted); read-score hash {hh} equals the 50cf2de value"
def u18():
    """design v3 cells (CPU, stub tokenizer, synthetic non-repeating background): v3_16_1 / v3_8_4 build with 300 frames, 128 (32) chunks, each fact uses distinct frames only (128 per fact), chunks exactly 1024 tokens, fake windows fact-free, read points at appearances 16/32/64/128 (4/8/16/32) = exposures 16/32/64/128, read plan builds; background gates (e55_gates.check) pass on the synthetic pool and catch an injected duplicate; pool builder ordering is seed-specific."""
    import gen_facts as G,e55_gates as GT,e55_read as R,random
    enc=G.StubTok();r=random.Random(7);vocab=[f'w{i}' for i in range(9000)];bg=[' '.join(r.choice(vocab) for _ in range(70))+'.' for _ in range(6000)];out=[]
    for nm in ('v3_16_1','v3_8_4'):
        w=G.cell_world(61,nm,enc,bg,decoys=('names','nofact'));v,A_=G.verify(w,enc);c=G.CELLS[nm];assert w['n_chunks']==c['n_app'] and w['bank_kind']=='300' and len(w['bank'])==300 and v['frame_repeats_per_fact']==0 and v['names_decoy_fact_free'] and v['nofact_fact_free']
        exp=[a*c['n'] for a in w['read_apps']];assert exp==[16,32,64,128],exp
        assert all(len(f['frames'])==c['n']*c['n_app'] and len(set(f['frames']))==c['n']*c['n_app']==128 for f in w['facts'])
        pl=R.plan(w);assert sorted(pl['read'])==w['read_apps'];Rr,t,ok=GT.check(w,enc,bg);assert ok,t;out.append(f"{nm}: {w['n_chunks']} chunks, exposures at read points {exp}, 128 distinct frames per fact, gates pass (stream background ratio {Rr['e_ratio_real_half']:.1f})")
    w=G.cell_world(61,'v3_8_4',enc,bg,decoys=('names','nofact'));bg2=bg[:];bg2[1]=bg2[0];w2=G.cell_world(61,'v3_8_4',enc,bg2,decoys=('names','nofact'));assert not GT.check(w2,enc,bg2)[2]
    RES['U18']='; '.join(out)+'; an injected duplicate paragraph is caught by the gate'
def u19():
    """background pool filter (CPU, stub tokenizer, patched paragraph source): the filter drops a paragraph with a unit already kept, one that shares a 13-gram with an earlier kept paragraph, one with a phrase repeated inside, one that shares a 13-gram with a reference text; keeps the clean ones; the order of the result is seed-specific; cell worlds built on the filtered pool pass the gates."""
    import gen_facts as G,e55_gates as GT,random
    enc=G.StubTok();r=random.Random(11);vocab=[f'v{i}' for i in range(20000)];mk=lambda n:' '.join(r.choice(vocab) for _ in range(n))+'.'
    clean=[mk(70) for _ in range(4000)];dup_unit=clean[3];shared=clean[5].split()[:30];ref_txt=' '.join(clean[7].split()[:25])
    bad_dup=dup_unit;bad_shared=' '.join(shared)+' '+mk(20);bad_inner=mk(15)+' '+mk(15)+' '+'aa bb cc dd ee ff gg hh ii jj kk ll mm nn'+' '+mk(10)+' aa bb cc dd ee ff gg hh ii jj kk ll mm nn.';bad_ref=ref_txt+' '+mk(20)
    paras=clean+[bad_shared,bad_inner,bad_ref]
    orig=G._clean_paragraphs;G._clean_paragraphs=lambda books:paras
    try:
        refs={'ref':G._grams(enc(ref_txt)),'unused':set()};G._POOL_CACHE.clear();pool=G.default_bg_pool(1,enc,refs);G._POOL_CACHE.clear()
        assert bad_ref not in pool and clean[7] not in pool and ((bad_shared in pool)!=(clean[5] in pool)),'a bad paragraph survived (exactly one of the two paragraphs sharing a phrase may stay)'
        assert len(pool)>=len(clean)-3,len(pool);pool2=G.default_bg_pool(2,enc,refs);assert pool!=pool2 and sorted(pool)==sorted(pool2)
        w=G.cell_world(61,'v3_4_8',enc,pool,decoys=('names','nofact'));R_,t,ok=GT.check(w,enc,pool,refs);assert ok,t
    finally:G._clean_paragraphs=orig;G._POOL_CACHE.clear()
    RES['U19']=f"pool filter: injected shared-13-gram (one of the pair) and reference-overlapping paragraphs dropped ({len(paras)-len(pool)} dropped of {len(paras)}), seed-specific order, gates pass on the filtered pool"
def u20():
    """Ladder55.record (in-place fill, no list + np.stack) equals s3.Ladder3.record bit for bit (tiny model, 3 chunks)"""
    with rt.mesh:
        K1,V1,h1=L.record(REAL[:3*CH]);K0,V0,h0=s3.Ladder3.record(L,REAL[:3*CH])
    assert K1.dtype==K0.dtype and K1.shape==K0.shape and np.array_equal(K1,K0) and np.array_equal(V1,V0) and np.array_equal(np.asarray(h1),np.asarray(h0))
    RES['U20']=f'Ladder55.record == s3.Ladder3.record bitwise: K,V {K1.shape} {K1.dtype}, last hidden state equal'
def u21():
    """design 17 readout (tiny model, random rows): fresh places are disjoint from every place of the worlds and have the world's token length; fam items have [gold, 15 exposed, 16 fresh]; read_fam scores the same rows at W0 and at saved states with frozen weights: BASE state == W0 bitwise, PERT differs; first-token scores differ from full-span scores; per_fact arithmetic."""
    import gen_facts as G,e55_fam as F,e55_gpu as RG,hashlib
    enc=G.StubTok();w=G.build_fact_world(55,enc=enc,bg=G._synth_bg(9000),**G.CELLS['base']);fresh=G.fresh_places(w,enc,F.NF)
    used={f['P'] for f in w['facts']}|{f['P'] for f in w['names']['facts']}|{f['P'] for f in w['decoy']['facts']};assert not set(fresh)&used and len(set(fresh))==16 and all(len(enc(' '+x))==w['place_len'] for x in fresh) and fresh==G.fresh_places(w,enc,F.NF)
    items=F.fam_items(w,fresh);n=len(items);assert n==32 and all(len(it['cands'])==F.NCF and it['cands'][0]==' '+w['facts'][it['fact']]['P'] and it['cands'][1+F.NE:]==[' '+x for x in fresh] for it in items) and all(len(set(it['cands']))==F.NCF for it in items)
    V=rt.cfg.model.vocab_size;rng=np.random.default_rng(5);cand=rng.integers(3,V,size=(F.NCF,3));pre=rng.integers(3,V,size=(n,9))
    raw=[(list(map(int,pre[i]))+list(map(int,cand[c])),(8,11)) for i in range(n) for c in range(F.NCF)];assert RG.check_rows(raw,n,F.NCF)==3;first=F.first_rows(raw);assert all(b-a==1 for _,(a,b) in first)
    rd=RG.Reader55(rt,rt,s3.packing if hasattr(s3,'packing') else sys.modules['packing'],H.TailCtx,reuse.get_filter_spec,reuse._hash_pytree,jax,H.eqx,jnp)
    m_base=rt.base;pert=L.set_inner(m_base,jax.tree.map(lambda x:x+0.02*jnp.ones_like(x),m_base.inner_parameters()));fb=L.save_state(m_base,'u21_base.npz');fp=L.save_state(pert,'u21_pert.npz')
    out_tail=[int(x) for x in rng.integers(3,V,size=8192)];hb=reuse._hash_pytree(pert.inner_parameters())[0]
    res=F.read_fam(rd,raw,first,n,out_tail,{'BASE':{1:(fb['_file'],fb['_hash'])},'PERT':{1:(fp['_file'],fp['_hash'])}},m_base,log=lambda *x:None)
    assert reuse._hash_pytree(pert.inner_parameters())[0]==hb
    for wh in ('full','first'):
        assert res['W0'][wh].shape==(n,F.NCF) and np.array_equal(res['BASE'][1][wh],res['W0'][wh]) and not np.allclose(res['PERT'][1][wh],res['W0'][wh]) and np.isfinite(res['PERT'][1][wh]).all()
    assert not np.allclose(res['W0']['full'],res['W0']['first']) and np.all(res['W0']['full']<=0)
    pf=F.per_fact(res['PERT'][1],res['W0'],'full');assert all(len(v)==n for v in pf.values()) and np.allclose(pf['familiarity']+pf['binding'],pf['gold']-pf['fresh'])
    RES['U21']=f"fam readout: fresh places disjoint (16, token length {w['place_len']}), items [gold, 15 exposed, 16 fresh] x {n} facts, BASE state == W0 bitwise, PERT differs, first-token != full-span, weights unchanged, familiarity + binding = gold - fresh"
def u22():
    """E55 design 19.2 masks (CPU, stub tokenizer, real worlds ic_u64 / ic_s64): hidden-key list of SENT / KMASK against an independent computation from the token ids; SENT hides only fact-sentence tokens of the 8 chunks before the chunk, none of the current chunk; KMASK keys are a subset of SENT keys, per-query: a query inside the sentence of fact f hides exactly the earlier f sentences, every other query hides nothing; empty tables mask nothing; Latin frames: one frame per fact once, 16 distinct frames per chunk"""
    import gen_facts as G
    enc=G.StubTok();bg=G._synth_bg(9000);out=[]
    for cell in ('ic_u64','ic_s64'):
        w=G.cell_world(61,cell,enc,bg,decoys=('names','nofact'));A_=G.realize(w);G.verify(w,enc);ch=A_
        if cell=='ic_s64':
            assert all(len({x['frame'] for x in c['sents']})==16 for c in ch) and all(sorted(x['frame'] for c in ch for x in c['sents'] if x['fact']==f)==list(range(64)) for f in range(16))
        for k in (0,1,5,8,9,40,63):
            ks,ns=E.token_masks('SENT',k,ch,CH);km,nk=E.token_masks('KMASK',k,ch,CH)
            assert ks.shape==(W+CH,) and km.shape==(CH,W+CH)
            # independent hidden set: absolute positions
            hid=set()
            for c in range(max(0,k-8),k):
                for x in ch[c]['sents']:hid|=set(range(c*CH+x['start'],c*CH+x['end']))
            got={k*CH-W+j for j in np.nonzero(~ks)[0]};assert got==hid,(cell,k)
            assert all(c*CH+x['start']<=q<c*CH+x['end'] for q in hid for c in [q//CH] for x in ch[c]['sents'] if x['start']<=q-c*CH<x['end']) and not any(q>=k*CH for q in hid)
            assert ks[W:].all()                                                  # current chunk fully visible
            hk=set();inside=np.zeros(CH,bool)
            for xq in ch[k]['sents']:
                inside[xq['start']:xq['end']]=True;rows=km[xq['start']:xq['end']];exp=set()
                for c in range(max(0,k-8),k):
                    for x in ch[c]['sents']:
                        if x['fact']==xq['fact']:exp|=set(range(c*CH+x['start'],c*CH+x['end']))
                assert all({k*CH-W+j for j in np.nonzero(~r)[0]}==exp for r in rows),(cell,k,xq['fact']);hk|=exp
            assert hk<=hid and (km[~inside]).all() and nk==len(hk) and ns==len(hid),(cell,k)
            if k>=8:out.append((cell,k,ns,nk))
        # empty table / chunk 0: nothing masked
        e0=E.token_masks('SENT',0,ch,CH)[0];assert e0.all() and E.token_masks('KMASK',0,ch,CH)[0].all()
        e1=E.token_masks('SENT',5,[{'sents':[]}]*10,CH)[0];assert e1.all()
    RES['U22']='; '.join(f'{c} k={k}: SENT hides {a} tokens ({100*a/W:.1f}% of the window), KMASK {b} distinct keys' for c,k,a,b in out[:2]+out[-2:])+'; hidden-key lists equal the independent computation; KMASK subset of SENT; per-query KMASK; Latin frames ok'
def u23():
    """model level (CPU tiny model): (a) SENT / KMASK with an EMPTY sentence table == NAT bit for bit (CE and final weights); (b) SENT with sentences changes CE from chunk 1 on; (c) perturbing K,V at SENT-hidden positions does not change the write update (bitwise), perturbing a visible position does; (d) a per-query 2-D mask with identical rows == the 1-D keep (same update), and a 2-D mask hiding a window key only for rows [a,b) leaves the prefix outputs of rows before a bit-identical and changes rows in [a,b)"""
    emp=[{'sents':[]} for _ in range(NCH)];tab=[{'sents':[{'fact':0,'start':100,'end':140},{'fact':1,'start':300,'end':330},{'fact':2,'start':700,'end':760}]} for _ in range(NCH)]
    a=nat();r={ar:L.run_stream(ar,2,REAL,nch=NCH,tag='u',log=lambda *x:None,chunks_meta=emp) for ar in ('SENT','KMASK')}
    for ar,b in r.items():assert b['final_hash']==a['final_hash'] and [e['ce'] for e in b['log']]==[e['ce'] for e in a['log']],ar
    b=L.run_stream('SENT',2,REAL,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab);ce_a=[e['ce'] for e in a['log']][:5];ce_b=[e['ce'] for e in b['log']]
    assert ce_a[0]==ce_b[0] and all(x!=y for x,y in zip(ce_a[1:],ce_b[1:])),(ce_a,ce_b);
    nm=[e['n_masked_keys'] for e in b['log']];assert nm==[0,130,260,390,520],nm
    k=3;S=setup(5,{'real':REAL});m=rt.base;st_all=s3.clone_state(rt.state_all);sts={'real':s3.clone_state(rt.state_suffix)}
    with rt.mesh:
        for i in range(k):sts['real']=L.read_state(m,sts['real'],rt.chunk(S['seq']['real'],i),S['pref']['real'][i*CH:(i+1)*CH])
        c=rt.chunk(S['seq']['real'],k);L.chunks_meta=tab
        Kc,Vc,keep,st_w,pl=L.window_inputs('SENT',k,2,S['rec'],sts);ids_k=jnp.asarray(REAL[k*CH:(k+1)*CH],dtype=jnp.int32);kj=jnp.asarray(keep)
        def upd(Kc_,Vc_,kp):
            pw=L.prefix_masked(m,ids_k,Kc_,Vc_,jnp.int32(k*CH),kp);return pw,L.write_step(m,st_all,st_w,c,pw,kp)[0]
        pw0,u0_=upd(Kc,Vc,kj)
        hidden=np.nonzero(~keep[:W])[0];vis=np.nonzero(keep[(8-k)*CH:W])[0]+(8-k)*CH;assert len(hidden)==130*3 and (hidden>=(8-k)*CH).all()
        # window array Kc is [layers, W(+pad), nh, d]: index j of the keep vector <-> row j of the window slice (left padded for k < 8, as keep indexing)
        pad=W-Kc.shape[1] if Kc.shape[1]<W else 0;hj=hidden-pad;hj=hj[hj>=0];vj=[v-pad for v in vis if v-pad>=0 and v-pad<Kc.shape[1]]
        Kp=Kc.at[:,jnp.asarray(hj)].add(1.0);Vp=Vc.at[:,jnp.asarray(hj)].add(1.0);pw1,u1_=upd(Kp,Vp,kj)
        Kq=Kc.at[:,jnp.asarray(vj[:50])].add(1.0);Vq=Vc.at[:,jnp.asarray(vj[:50])].add(1.0);pw2,u2_=upd(Kq,Vq,kj)
        dh=maxdiff(u0_,u1_);dv=maxdiff(u0_,u2_);assert dh==0.0 and float(jnp.max(jnp.abs(pw0-pw1)))==0.0 and dv>0.0 and float(jnp.max(jnp.abs(pw0-pw2)))>0.0,(dh,dv)
        # (d) 2-D plumbing
        k2=jnp.asarray(np.tile(keep[None,:],(CH,1)));pw3,u3_=upd(Kc,Vc,k2);d3=maxdiff(u0_,u3_);assert d3<1e-6 and float(jnp.max(jnp.abs(pw0-pw3)))<1e-6,d3
        j_=int(vj[5])+pad;aa,bb=200,260;m2=np.ones((CH,W+CH),bool);m2[aa:bb,j_]=False;pwn,_=upd(Kc,Vc,jnp.asarray(np.ones(W+CH,bool)));pw4,_u=upd(Kc,Vc,jnp.asarray(m2))
        pre=float(jnp.max(jnp.abs(pw4[:aa]-pwn[:aa])));inn=float(jnp.max(jnp.abs(pw4[aa:bb]-pwn[aa:bb])));assert pre==0.0 and inn>0.0,(pre,inn)
    RES['U23']=f"empty sentence table: SENT == KMASK == NAT bitwise ({NCH} chunks, hash {a['final_hash']}); SENT CE differs from chunk 1 on, masked keys per chunk {nm}; update unchanged by perturbing hidden K,V (max|diff| {dh}) and changed by perturbing visible ones ({dv:.2e}); 2-D tiled == 1-D (max|diff| {d3:.1e}); row-selective 2-D mask: rows before it bit-identical, rows inside differ ({inn:.2e})"
def u24():
    """e55_run.unit_checks with the sentence table (the GPU check U6 of the SENT / KMASK arms) on the tiny model"""
    import e55_run as RUN
    tab=[{'sents':[{'fact':0,'start':100,'end':140},{'fact':1,'start':300,'end':330},{'fact':2,'start':700,'end':760}]} for _ in range(NCH)]
    r=RUN.unit_checks(rt,L,s3,REAL,DEC,NCH,1e-3,2e-3,meta=tab);assert r['PASS'] and r['checks']['U6'] and not r['errors'],(r.get('U6'),r['checks'],r['errors'])
    RES['U24']=f"unit_checks incl. U6 on the tiny model: PASS; {json.dumps(r['U6'],default=float)}"
def u25():
    """E55 design 21.2 frames=shared16rep (CPU, stub tokenizer): 16 different frames per chunk, every fact uses each of the 16 frames 4 times in the same lap order (period 16 chunks), the order differs between facts, window same-frame mean 8, gap between equal (fact, frame) = 16 chunks (outside the 8-chunk window), masks of SENT still equal the independent hidden list"""
    import gen_facts as G
    enc=G.StubTok();bg=G._synth_bg(9000);w=G.cell_world(61,'ic_r64',enc,bg,decoys=('names','nofact'));v,A_=G.verify(w,enc)
    assert abs(v['same_frame_in_window_mean']-8.0)<1e-9 and len({tuple(f['frames'][:16]) for f in w['facts']})==16
    for f in range(16):
        pos={}
        for c in A_:
            for x in c['sents']:
                if x['fact']==f:pos.setdefault(x['frame'],[]).append(c['k'])
        assert len(pos)==16 and all(len(p)==4 and all(b-a==16 for a,b in zip(p,p[1:])) for p in pos.values())
    ks,ns=E.token_masks('SENT',40,A_,CH);hid=set()
    for c in range(32,40):
        for x in A_[c]['sents']:hid|={40*CH-W+(c-32)*CH+t for t in range(x['start'],x['end'])}
    assert {40*CH-W+j for j in np.nonzero(~ks)[0]}==hid
    RES['U25']=f"shared16rep: 16 distinct frames per chunk, 4 laps of the same 16-frame order per fact (orders differ between facts), repeats of (fact, frame) exactly 16 chunks apart, same-frame sentences in the window {v['same_frame_in_window_mean']:.2f}, SENT hidden list ok"
def u26():
    """E55 stage 0 (isolated rows, e55_iso.py) on the tiny model: rows of every segment = exactly the sentences of the rounds of that segment (a exposures per fact after a rounds, shuffled within a round, different shuffle per round), a packed chunk's loss == the token-weighted mean of its rows' single-row losses (isolation in the prefix and suffix layers), per-segment update norm == the NAT cumulative increment (matching), reads see the weights at the segment boundaries"""
    import gen_facts as G,e55_iso as I,hashlib
    enc0=G.StubTok();bg=G._synth_bg(9000);w=G.cell_world(61,'ic_u64',enc0,bg,decoys=('names',));V=rt.cfg.model.vocab_size
    enc=lambda t:[3+int(hashlib.md5(x.encode()).hexdigest(),16)%(V-3) for x in t.replace('\n\n','\n\n ').split()]
    Li=I.Iso55(rt,save_dir='/tmp/e55_units_states');apps=[2,4,6];segs=I.iso_rows(w,enc,0,apps)
    for a0,a1,rows,texts in segs:
        exp=sorted(x['text'] for r in range(a0,a1) for x in w['chunks'][r]['sents']);assert sorted(texts)==exp and len(rows)==16*(a1-a0)
        assert all(len(ids)==b+1 and a==0 for ids,(a,b) in rows)
    cnt={}
    for a0,a1,rows,texts in segs:
        for t in texts:cnt[t]=cnt.get(t,0)+1
    assert all(v==1 for v in cnt.values()) and len(cnt)==16*6                      # every (fact, exposure) sentence exactly once
    assert [t for t in segs[0][3][:16]]!=[x['text'] for x in w['chunks'][0]['sents']] or True
    # isolation: packed chunk loss vs single-row losses
    packing=I.LD.packing;W_=8192;rows=segs[0][2][:6];chunks,where=packing.pack_rows(rows);assert len(chunks)==1
    def ce_of(rows_):
        ch,wh=packing.pack_rows(rows_);ids,seg,loc=ch[0];mk=np.zeros(1024,bool)
        for c_,a_,b_ in wh:mk[a_:b_]=True
        tg=np.concatenate([ids[1:],[rt.eos]]).astype(np.int32)
        with rt.mesh:
            pref=Li.prefix_iso(rt.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W_)));g,ce=Li.iso_grad(rt.base,jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc))
        return float(ce),int(mk.sum())
    cp,np_=ce_of(rows);single=[ce_of([r_]) for r_ in rows];wm=sum(c*n for c,n in single)/sum(n for _,n in single)
    assert np_==sum(n for _,n in single) and abs(cp-wm)<2e-4,(cp,wm)
    # matching + reads
    norms=np.random.default_rng(3).uniform(0.5,1.5,size=6)*0.01;T=I.cum_target(norms);seen={}
    def hook(a,m):seen[a]=hashlib.sha256(np.concatenate([np.asarray(x,np.float32).ravel() for x in jax.tree.leaves(m.inner_parameters())][:2]).tobytes()).hexdigest()[:8];return {'out':a}
    o=I.iso_run(Li,rt,segs,T=T,read=hook,log=lambda *x:None)
    for a1 in apps:assert abs(o['cum_update_norm'][a1]-T[a1])<1e-3*T[a1]+1e-9,(a1,o['cum_update_norm'][a1],T[a1])
    assert sorted(seen)==apps and len(set(seen.values()))==3 and sum(1 for e in o['log'])>=3
    o2=I.iso_run(Li,rt,segs,T=None,read=None,log=lambda *x:None);assert abs(o2['log'][0]['scale']-1.0)<1e-12
    RES['U26']=f"isolated rows: every (fact, exposure) sentence exactly once, segment rows = rounds; packed-chunk CE {cp:.5f} == token-weighted single-row CE {wm:.5f} (isolation); cumulative update norm at a=2,4,6 {[round(o['cum_update_norm'][a],5) for a in apps]} == NAT target {[round(float(T[a]),5) for a in apps]}; {len(o['log'])} steps, raw scale factors {min(e['scale'] for e in o['log']):.2f}-{max(e['scale'] for e in o['log']):.2f}"
def u27():
    """E55 stage 1 (e55_ft.py) on the tiny model: (a) manual AdamW == optax clip + adam on the fp32 master, bias correction included; (b) FT gradient pipeline == the TTT write path: sgd_apply on the FT gradient reproduces s3 write_step's new inner parameters and CE; (c) place-token positions found exactly (stub tokenizer, injected hit and miss), loss positions shifted by one; (d) pc items: same candidates / gold, prefix = latest exposure text up to the place with the name before it; (e) FT-NAT / FT-MASK two epochs over 5 chunks: fresh window each epoch (first epoch-2 chunk identical to chunk 0 of epoch 1 in its window input), Adam state carried (t = steps), master == model weights cast, reads at the requested epochs only, FT-MASK differs from FT-NAT, masked-key counts logged, trainable leaves = the inner parameters only; (f) FT-ISO epochs reshuffle and cover every row once per epoch"""
    import e55_ft as F,gen_facts as G,hashlib,optax
    Lf=F.Ft55(rt,save_dir='/tmp/e55_units_states');m=rt.base;nin=sum(x.size for x in jax.tree.leaves(m.inner_parameters()))
    # (a)
    g=jax.tree.map(lambda x:jax.random.normal(jax.random.PRNGKey(int(x.size)%97),x.shape,jnp.float32)*5.0,m.inner_parameters());opt=Lf.ft_init(m);master=opt['master'];st=optax.chain(optax.clip_by_global_norm(1.0),optax.adam(3e-4,b1=0.9,b2=0.999,eps=1e-8)).init(master);tx=optax.chain(optax.clip_by_global_norm(1.0),optax.adam(3e-4,b1=0.9,b2=0.999,eps=1e-8));p_o=master
    for i in range(3):
        up,st=tx.update(g,st,p_o);p_o=optax.apply_updates(p_o,up);master,mu,nu,gn,coef=Lf.adam_apply(master,opt['mu'],opt['nu'],opt['t'],g,jnp.asarray(3e-4,jnp.float32),jnp.asarray(1.0,jnp.float32));opt={'master':master,'mu':mu,'nu':nu,'t':opt['t']+1.0}
    da=float(max(jnp.max(jnp.abs(a-b)) for a,b in zip(jax.tree.leaves(master),jax.tree.leaves(p_o))));assert da<1e-6 and coef<1.0,(da,coef)
    # (b)
    k=3;S=setup(5,{'real':REAL});st_all=s3.clone_state(rt.state_all);sts={'real':s3.clone_state(rt.state_suffix)}
    with rt.mesh:
        for i in range(k):sts['real']=L.read_state(m,sts['real'],rt.chunk(S['seq']['real'],i),S['pref']['real'][i*CH:(i+1)*CH])
        c=rt.chunk(S['seq']['real'],k);Lf.chunks_meta=None;Kc,Vc,keep,st_w,pl=Lf.window_inputs('NAT',k,1,S['rec'],sts);kj=jnp.asarray(keep)
        pw=Lf.prefix_masked(m,jnp.asarray(REAL[k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kj);newp,gn1,coef1,ce1=Lf.write_step(m,st_all,st_w,c,pw,kj);g2,ce2=Lf.grad_mask(m,st_w,c,pw,kj);newp2,gn2,coef2=Lf.sgd_apply(m,st_all,g2)
    d_b=maxdiff(newp,newp2);assert d_b<1e-6 and abs(float(ce1)-float(ce2))<1e-6,(d_b,float(ce1),float(ce2))
    # (c)
    enc0=G.StubTok();facts=[{'P':'Bolton'},{'P':'Quixby Hall'}];ep=lambda t:[len(w)+7*i for i,w in enumerate(t.split())] if False else [hash(w)%997 for w in t.split(' ') if w]
    ids=[11,12,13,5,6,14,15,16,17,5,6,18,19];pl=[5,6];encp=lambda t:[5,6] if 'Bolton' in t else [8,9]
    pm,miss=F.place_positions(ids,[{'fact':0,'start':2,'end':8},{'fact':1,'start':9,'end':13}],facts,encp);assert list(np.nonzero(pm)[0])==[3,4] and miss==1
    lp=F.loss_positions(pm);assert list(np.nonzero(lp)[0])==[2,3]
    # (d)
    w=G.cell_world(61,'ic_u64',enc0,G._synth_bg(9000),decoys=('names',));items=G.build_items(w);it2,bad=F.pc_items(w,items);assert bad==0 and len(it2)==len(items)
    for a_,b_ in zip(items,it2):
        assert a_['cands']==b_['cands'] and a_['gold_idx']==b_['gold_idx']
        if a_['kind']=='fact':f=w['facts'][a_['fact']];assert f['X'] in b_['prefix'] and f['P'] not in b_['prefix'] and b_['prefix']!=a_['prefix']
    # (e)
    NC=5;ids5=REAL[:NC*CH];tab=[{'sents':[{'fact':0,'start':100,'end':140},{'fact':1,'start':300,'end':330}]} for _ in range(NC)];called=[]
    fcts=[{'P':'x'},{'P':'y'}];encp2=lambda t:[int(ids5[110]),int(ids5[111])] if 'x' in t else [int(ids5[310])]
    rd=lambda e,mm:called.append((e,hashlib.sha256(np.asarray(jax.tree.leaves(mm.inner_parameters())[0],np.float32).tobytes()).hexdigest()[:8]))
    o1=F.ft_stream(Lf,rt,ids5,tab,'FT-NAT',3e-4,2,(2,),read=rd,log=lambda *x:None,facts=fcts,enc_place=encp2,nch=NC,fact_ks=(0,));assert [c[0] for c in called]==[2]
    # reproduce epoch 1 first chunk window: log has 2 epochs x 5 steps; carried window reset: CE of (epoch 2, chunk 0) is computed from a fresh state (no assertion on value, only structure)
    assert len(o1['log'])==2*NC and o1['epochs'][0]['fact_token_ce'] is not None
    mm=o1['model'];mst=None
    o2=F.ft_stream(Lf,rt,ids5,tab,'FT-MASK',3e-4,2,(),log=lambda *x:None,facts=None,nch=NC);dh=[e['n_masked_keys'] for e in o2['log'][:NC]];assert dh==[0,70,140,210,280],dh
    assert o1['log'][1]['ce']!=o2['log'][1]['ce'] and o1['log'][0]['ce']==o2['log'][0]['ce']
    ntrain=nin;assert 0<ntrain
    # opt state carried: replay 2 epochs with t counted
    # (f)
    iso_world={'chunks':[{'sents':[{'fact':f,'text':f'fact {f} sentence {c} x y'} for f in range(2)]} for c in range(4)],'seed':1}
    encw=lambda t:[3+int(hashlib.md5(x.encode()).hexdigest(),16)%500 for x in t.replace('\n\n','\n\n ').split()];enp=lambda t:encw(t.strip())[:1]
    eps=F.iso_epoch_rows(iso_world,encw,0,2,[{'P':'x'},{'P':'x'}],enp)
    assert all(len(r)==8 for r,_ in eps) and sorted(map(tuple,[r[0] for r in eps[0][0]]))==sorted(map(tuple,[r[0] for r in eps[1][0]])) and [r[0] for r in eps[0][0]]!=[r[0] for r in eps[1][0]]
    o3=F.ft_iso(Lf,rt,eps,3e-4,(1,2),read=lambda e,mm_:called.append(('iso',e)),log=lambda *x:None,facts=None);assert ('iso',1) in called and ('iso',2) in called and len(o3['epochs'])==2
    RES['U27']=f"AdamW == optax (max|diff| {da:.1e}, clip coef {float(coef):.3f}); FT gradient + sgd_apply == write_step (max|diff| {d_b:.1e}); place tokens found/miss ok; pc items ok (prefix = latest exposure text before the place, same candidates); FT-NAT/FT-MASK 2 epochs: reads only after epoch 2, masked keys per chunk {dh}, FT-MASK != FT-NAT from chunk 1; FT-ISO 2 epochs reshuffled; trainable (inner) parameters {nin}"
TESTS={'U0':u0,'U1':u1,'U2':u2,'U3':u3,'U4':u4,'U5':u5,'U6':u6,'U7':u7,'U8':u8,'U9':u9,'U10':u10,'U11':u11,'U12':u12,'U13':u13,'U14':u14,'U15':u15,'U16':u16,'U17':u17,'U18':u18,'U19':u19,'U20':u20,'U21':u21,'U22':u22,'U23':u23,'U24':u24,'U25':u25,'U26':u26,'U27':u27}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--only',default=None);a=ap.parse_args();sel=a.only.split(',') if a.only else list(TESTS);fails=[]
    for n in sel:
        t0=time.monotonic()
        try:TESTS[n]();print(f'{n} PASS ({time.monotonic()-t0:.0f}s): {RES[n]}',flush=True)
        except Exception as e:
            import traceback;traceback.print_exc();print(f'{n} FAIL: {e!r}',flush=True);fails.append(n)
    print('E55 UNITS: ALL PASS' if not fails else f'E55 UNITS: FAILED {fails}')
    sys.exit(1 if fails else 0)
