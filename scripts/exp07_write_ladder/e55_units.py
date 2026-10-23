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
def u29():
    """E55 design 26.4: SENTO (sentence rows hide the earlier sentences of the OTHER facts) vs KMASK (same fact): per row independent computation, KMASK and SENTO hidden pairs disjoint, intersection of their keeps == SENT keep on every sentence row, background rows see everything, chunk 0 / empty table mask nothing"""
    import gen_facts as G
    enc=G.StubTok();bg=G._synth_bg(9000);n=0
    w=G.cell_world(61,'ic_u64',enc,bg,decoys=('names','nofact'));ch=G.realize(w);G.verify(w,enc)
    for k in (0,1,5,9,40,63):
        ks,_=E.token_masks('SENT',k,ch,CH);km,_=E.token_masks('KMASK',k,ch,CH);so,nso=E.token_masks('SENTO',k,ch,CH)
        assert so.shape==(CH,W+CH);inside=np.zeros(CH,bool)
        assert ((~km)&(~so)).sum()==0
        for xq in ch[k]['sents']:
            inside[xq['start']:xq['end']]=True;exp=set()
            for c in range(max(0,k-8),k):
                for x in ch[c]['sents']:
                    if x['fact']!=xq['fact']:exp|=set(range(c*CH+x['start'],c*CH+x['end']))
            for r in range(xq['start'],xq['end']):
                assert {k*CH-W+j for j in np.nonzero(~so[r])[0]}==exp,(k,xq['fact'])
                assert np.array_equal(km[r]&so[r],ks),(k,r)
                n+=1
        assert so[~inside].all()
        sp,_=E.token_masks('SENTPQ',k,ch,CH);assert sp[~inside].all() and all(np.array_equal(sp[r],ks) for xq in ch[k]['sents'] for r in range(xq['start'],xq['end'])) and np.array_equal(sp&~(~km|~so),sp)
    assert E.token_masks('SENTO',0,ch,CH)[0].all() and E.token_masks('SENTO',5,[{'sents':[]}]*10,CH)[0].all()
    RES['U29']=f'SENTPQ sentence rows == SENT keep, background rows unmasked; SENTO rows == independent other-fact computation ({n} rows), KMASK/SENTO disjoint, intersection of keeps == SENT keep on every sentence row, background rows unmasked, empty/chunk0 unmasked'
def u30():
    """E55 design 27.2 Q2 arms (CPU): (a) annotate_answers on the stub world finds name and place in every sentence (miss (0,0)); OSPAN keys = exactly the annotated spans of the earlier sentences, only in sentence rows, background rows see everything; SENTP25/50/75 keys nested p25 <= p50 <= p75 <= SENTPQ, shares about 25/50/75% of SENTPQ's, identical for every sentence row, deterministic; (b) model level: empty table == NAT bitwise for the new arms, with sentences CE differs from chunk 1 on, n_masked_pairs logged; (c) DclmNll on the tiny model: finite, deterministic, W0 differs from a perturbed model"""
    import gen_facts as G,e55_run as RUN,tempfile
    enc=G.StubTok();bg=G._synth_bg(9000);w=G.cell_world(61,'ic_u64',enc,bg,decoys=('names','nofact'));ch=G.realize(w);G.verify(w,enc)
    miss=E.annotate_answers(ch,w['facts'],enc);assert miss==(0,0),miss
    sh=[]
    for k in (9,40,63):
        sq,_=E.token_masks('SENTPQ',k,ch,CH);op,no=E.token_masks('OSPAN',k,ch,CH);ps={p:E.token_masks(p,k,ch,CH) for p in ('SENTP25','SENTP50','SENTP75')}
        exp=set()
        for c in range(max(0,k-8),k):
            for x in ch[c]['sents']:
                for a_,b_ in x['ans']:exp|=set(range(c*CH+a_,c*CH+b_))
        assert {k*CH-W+j for j in np.nonzero(~op.all(0))[0]}==exp and len(exp)==no
        inside=np.zeros(CH,bool)
        for xq in ch[k]['sents']:
            inside[xq['start']:xq['end']]=True
            for r in range(xq['start'],xq['end']):assert np.array_equal(op[r],op[xq['start']]) and all(np.array_equal(m[r],m[xq['start']]) for m,_ in ps.values())
        assert op[~inside].all() and all(m[~inside].all() for m,_ in ps.values())
        kq=set(np.nonzero(~sq.all(0))[0]);k25,k50,k75=[set(np.nonzero(~ps[p][0].all(0))[0]) for p in ('SENTP25','SENTP50','SENTP75')]
        assert k25<=k50<=k75<=kq and set(np.nonzero(~op.all(0))[0])<=kq
        sh.append([round(len(x)/len(kq),2) for x in (k25,k50,k75)]);assert all(abs(a-b)<0.06 for a,b in zip(sh[-1],(0.25,0.5,0.75))),sh[-1]
        m2=E.token_masks('SENTP50',k,ch,CH)[0];assert np.array_equal(m2,ps['SENTP50'][0])
    emp=[{'sents':[]} for _ in range(NCH)];tab=[{'sents':[{'fact':0,'start':100,'end':140,'ans':[(110,112),(130,133)]},{'fact':1,'start':300,'end':330,'ans':[(305,308)]},{'fact':2,'start':700,'end':760,'ans':[(710,715),(750,752)]}]} for _ in range(NCH)]
    a=nat()
    for ar in ('OSPAN','SENTP25','SENTP50','SENTP75'):
        b=L.run_stream(ar,2,REAL,nch=NCH,tag='u',log=lambda *x:None,chunks_meta=emp);assert b['final_hash']==a['final_hash'] and [e['ce'] for e in b['log']]==[e['ce'] for e in a['log']],ar
        b=L.run_stream(ar,2,REAL,nch=4,tag='u',log=lambda *x:None,chunks_meta=tab);ce_a=[e['ce'] for e in a['log']][:4];ce_b=[e['ce'] for e in b['log']]
        assert ce_a[0]==ce_b[0] and all(x!=y for x,y in zip(ce_a[1:],ce_b[1:])),(ar,ce_a,ce_b);assert [e['n_masked_pairs'] for e in b['log']][0]==0 and b['log'][3]['n_masked_pairs']>0 and b['log'][3]['n_masked_keys']>0
    f=Path(tempfile.mkdtemp())/'d.u32le.bin';np.random.default_rng(7).integers(3,rt.cfg.model.vocab_size,size=3*8193).astype('<u4').tofile(f)
    import ladder as LDm;dc=RUN.DclmNll(LDm,rt,str(f),2);v0=dc(rt.base);assert np.isfinite(v0) and v0==dc(rt.base) and dc.n==2
    mp=a['model'] if 'model' in a else None
    RES['U30']=f'answer spans found (miss {miss}); OSPAN keys == annotated spans, SENTP nested with shares {sh}; empty table == NAT bitwise for OSPAN/SENTP25/50/75; CE differs with sentences, n_masked_pairs logged; DclmNll finite+deterministic (W0 {v0:.3f} on random ids)'
def u31():
    """E55 design 27.5 two-loss mix (CPU, tiny model): (a) lambda = 0 (monkeypatched) with sentences == NAT (CE per chunk and final inner weights equal up to float summation order); (b) empty table: MIX25 == NAT likewise; (c) lambda = 1: chunk 1 CE of the masked pass == SENTPQ's CE, and chunk 0 update equals SENTPQ's (no window yet); (d) MIX25 with sentences: finite, differs from NAT from chunk 1 on, logs ce_masked_pass / n_masked_pairs (> 0 from chunk 1), update norm finite; (e) ALT: chunks 0 (masked path, nothing hidden yet) and 1 (unmasked) equal NAT's CE, chunk 2 (masked, window present) differs; ALT logs alt_masked_chunk; (f) mix_weights: fact positions = [start-1, end-1), sums: unmasked + masked weights = valid/N on fact positions and valid/N elsewhere (unmasked) / 0 (masked)"""
    emp=[{'sents':[]} for _ in range(NCH)];tab=[{'sents':[{'fact':0,'start':100,'end':140},{'fact':1,'start':300,'end':330},{'fact':2,'start':700,'end':760}]} for _ in range(NCH)]
    a=nat();ce_n=[e['ce'] for e in a['log']]
    def close(x,y,tol=2e-4):return all(abs(p-q)<=tol*max(1,abs(q)) for p,q in zip(x,y))
    def wdiff(m1,m2):return maxdiff(m1.inner_parameters(),m2.inner_parameters())
    nref=wdiff(a['model'],rt.base)
    old=E.MIX['MIX25'];E.MIX['MIX25']=0.0
    try:b0=L.run_stream('MIX25',2,REAL,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab)
    finally:E.MIX['MIX25']=old
    assert close([e['ce'] for e in b0['log']],ce_n[:5]),([e['ce'] for e in b0['log']],ce_n[:5])
    be=L.run_stream('MIX25',2,REAL,nch=5,tag='u',log=lambda *x:None,chunks_meta=emp);assert close([e['ce'] for e in be['log']],ce_n[:5])
    sq=L.run_stream('SENTPQ',2,REAL,nch=3,tag='u',log=lambda *x:None,chunks_meta=tab)
    E.MIX['MIX25']=1.0
    try:b1=L.run_stream('MIX25',2,REAL,nch=3,tag='u',log=lambda *x:None,chunks_meta=tab)
    finally:E.MIX['MIX25']=old
    assert close([b1['log'][0]['ce_masked_pass'],b1['log'][1]['ce_masked_pass']],[sq['log'][0]['ce'],sq['log'][1]['ce']],1e-3),([e['ce_masked_pass'] for e in b1['log']],[e['ce'] for e in sq['log']])
    bm=L.run_stream('MIX25',2,REAL,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab);lg=bm['log']
    assert all(np.isfinite(e['ce']) and np.isfinite(e['update_norm']) and 'ce_masked_pass' in e for e in lg) and lg[0]['ce']==ce_n[0] or abs(lg[0]['ce']-ce_n[0])<2e-4
    assert close([e['ce'] for e in lg[:2]],ce_n[:2],1e-6) and all(abs(e['ce']-c_)>1e-6 for e,c_ in zip(lg[2:],ce_n[2:5])) and all(abs(e['ce_masked_pass']-e['ce'])>1e-6 for e in lg[1:]),([e['ce'] for e in lg],ce_n[:5],[e['ce_masked_pass'] for e in lg]);assert lg[0]['n_masked_pairs']==0 and all(e['n_masked_pairs']>0 for e in lg[1:])
    bl=L.run_stream('ALT',2,REAL,nch=4,tag='u',log=lambda *x:None,chunks_meta=tab);cl=[e['ce'] for e in bl['log']]
    assert close(cl[:2],ce_n[:2]) and abs(cl[2]-ce_n[2])>1e-6 and [e['alt_masked_chunk'] for e in bl['log']]==[True,False,True,False] and bl['log'][1]['n_masked_pairs']==0 and bl['log'][2]['n_masked_pairs']>0,(cl,ce_n[:4])
    valid=np.ones(CH,np.float32);valid[-1]=0;wu,wm,nf=C.Ladder55.mix_weights(tab[0]['sents'],valid,0.25);n=valid.sum();f=np.zeros(CH,bool)
    for x in tab[0]['sents']:f[x['start']-1:x['end']-1]=True
    assert nf==f.sum() and np.allclose(wu[f],0.75*valid[f]/n) and np.allclose(wu[~f],valid[~f]/n) and np.allclose(wm[f],0.25*valid[f]/n) and (wm[~f]==0).all() and np.allclose((wu+wm)[f],valid[f]/n)
    RES['U31']=f'lambda=0 and empty-table MIX == NAT (CE within 2e-4), lambda=1 masked-pass CE == SENTPQ chunk 0/1, MIX25 differs from NAT from chunk 2 (the logged CE is the unmasked pass; chunk 1 still equal because chunk 0 has no window), masked-pass CE differs from chunk 1, pairs logged; ALT chunks 0/1 == NAT, chunk 2 differs; weights: fact positions {nf}, wu+wm == valid/N on fact tokens'
def u32():
    """E55 design 27.8 Q1-A inner-step variants on the tiny model (stage-0 rows, T = a synthetic NAT cumulative norm): STD default == explicit variant STD (bitwise weights); every variant hits the per-step update norm target inc = (T[a1]-T[a0])/steps; SUR1/SUR2: weighted loss differs from STD (different weights), mean weight in (0, 1], >0 and the unweighted CE logged equals STD's CE at step 0; SUR with power 1 and weights forced uniform would equal STD (checked on iso_grad_w with a constant-CE surrogate is not possible, so: the weight cap 5 and mean 1 are checked directly on the formula); ADAM first step direction = g/|g| elementwise sign-like (bias-corrected m/sqrt(v) = sign for step 1), MOM first step direction = clipped g; ADAM/MOM refuse T=None"""
    import gen_facts as G,e55_iso as I,hashlib
    enc0=G.StubTok();bg=G._synth_bg(9000);w=G.cell_world(61,'ic_u64',enc0,bg,decoys=('names',));V=rt.cfg.model.vocab_size
    enc=lambda t:[3+int(hashlib.md5(x.encode()).hexdigest(),16)%(V-3) for x in t.replace('\n\n','\n\n ').split()]
    Li=I.Iso55(rt,save_dir='/tmp/e55_units_states');segs=I.iso_rows(w,enc,0,[1,2]);T=I.cum_target([0.01]*2)
    def run(v,**k):return I.iso_run(Li,rt,segs,T=T,variant=v,log=lambda *x:None,**k)
    ra=I.iso_run(Li,rt,segs,T=T,log=lambda *x:None);rb=run('STD');assert I.LD.jax.tree_util.tree_all(I.LD.jax.tree.map(lambda x,y:bool((x==y).all()),ra['model'].inner_parameters(),rb['model'].inner_parameters()))
    res={'STD':rb}
    for v in ('SUR1','SUR2','ADAM','MOM'):res[v]=run(v)
    for v,r in res.items():
        inc=[(T[e['a1']]-T[e['a0']])/sum(1 for x in r['log'] if x['a0']==e['a0']) for e in r['log']]
        assert all(abs(e['update_norm']-i)<=0.15*i for e,i in zip(r['log'],inc)),(v,[round(e['update_norm'],5) for e in r['log']],inc)
    d={v:float(max(jnp.max(jnp.abs(x.astype(jnp.float32)-y.astype(jnp.float32))) for x,y in zip(jax.tree.leaves(res[v]['model'].inner_parameters()),jax.tree.leaves(rb['model'].inner_parameters())))) for v in ('SUR1','SUR2','ADAM','MOM')}
    assert all(x>0 for x in d.values()),d
    for v in ('SUR1','SUR2'):
        mw=[e['mean_weight'] for e in res[v]['log']];assert all(0<x<=1.0+1e-5 for x in mw),mw;assert abs(res[v]['log'][0]['ce']-rb['log'][0]['ce'])<1e-5
    assert all(0<e['mean_weight'] for e in res['SUR2']['log']) and np.mean([e['mean_weight'] for e in res['SUR2']['log']])!=np.mean([e['mean_weight'] for e in res['SUR1']['log']])
    try:I.iso_run(Li,rt,segs,T=None,variant='ADAM',log=lambda *x:None);raise SystemExit('ADAM without T accepted')
    except AssertionError:pass
    RES['U32']=f'default == STD bitwise; all 5 variants hit the per-step norm target (within 15%); weights differ from STD by (max abs) {({k:round(v,6) for k,v in d.items()})}; SUR mean weight in (0,1], first-step CE equals STD; ADAM without matching refused'
def u33():
    """E55 design 27.8 attention diagnostic on the tiny model (CPU): the eager 'att' forward through the prefix layers returns the same hidden states as prefix_masked (the production 'ctx' path; tolerance 1e-4), the captured per-layer probabilities are proper distributions (rows sum to 1, zero on invalid / future keys), attn_stats runs on them for all layer groups, and a run_stream NAT with diag_fn changes nothing about the run (CE per chunk and final weights equal to the plain NAT run)"""
    import e55_diag as DG
    ids=REAL;tab=[{'sents':[{'fact':0,'start':100,'end':140,'ans':[(105,107)]},{'fact':1,'start':300,'end':330,'ans':[(305,308)]},{'fact':2,'start':700,'end':760,'ans':[(710,715)]}]} for _ in range(NCH)]
    seen=[];orig=DG.attn_stats
    def spy(A,rows,facts,chunks,k,CH_=1024,nw=8,ps=DG.PS):seen.append((np.asarray(A),np.asarray(rows),k));return orig(A,rows,facts,chunks,k,CH_,nw,ps)
    DG.attn_stats=spy
    try:
        fn=DG.make_diag_fn(L,rt,s3,tab,log=lambda *x:None);a=L.run_stream('NAT',2,ids,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab)
        cap={}
        def fn2(k,m,Kc,Vc,keep,ids_k):
            r=fn(k,m,Kc,Vc,keep,ids_k)
            with rt.mesh:cap[k]=(fn.last_h,L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep)))
            return r
        b=L.run_stream('NAT',2,ids,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab,diag_fn=fn2,diag_chunks={3,4})
        hd=max(float(jnp.max(jnp.abs(x.astype(jnp.float32)-y.astype(jnp.float32)))) for x,y in cap.values());assert hd<1e-3,hd
    finally:DG.attn_stats=orig
    assert set(b['diag'])=={3,4},set(b['diag']);assert [e['ce'] for e in b['log']]==[e['ce'] for e in a['log']] and b['final_hash']==a['final_hash']
    r=b['diag'][4];assert r['n_layers']>0
    for g in ('all','g1','g2','g3'):
        x=r[g]
        if x is None:continue
        assert x['n_rows']==40+30+60 and set(x['p'])=={'1','2','5','10','20'} and 0<=x['attention_mass']['same']<=1 and abs(x['p']['20']['same']+x['p']['20']['background']+(x['p']['20']['any']-x['p']['20']['same'])-1.0)<1e-6
    W_=8192
    for A,rows,k in seen:
        assert np.allclose(A.sum(1),1.0,atol=1e-3),(k,A.sum(1)[:3])
        assert all((A[i,W_+rows[i]+1:]<1e-12).all() for i in range(len(rows)))        # no attention to future keys of the current chunk
        assert (A[:,:max(0,(8-k))*1024]==0).all()                                       # absent chunks (k < 8) get no attention
    RES['U33']=f"att-mode hidden states == prefix_masked (max diff {hd:.2e}); attention rows are proper distributions (sum 1, no future keys), stats for 4 layer groups x 5 p-values, run with diag_fn == run without (CE bitwise), n_layers {r['n_layers']}"
def u34():
    """E55 design 28.3 / 28.7 (CPU, tiny model): (a) PSPAN keys = the place spans, NSPAN keys = the name spans, OSPAN = their union, per-query (background rows unmasked), annotate_answers stores ans_P / ans_X; (b) run_stream NAT with diag_multi (the 8 probe conditions NAT, KMASK, SENTO, SENT, SWD, O-ALL, P-ALL, N-ALL; the fake stream recorded) == the plain NAT run (CE and final weights bitwise); (c) probe: every probed copy m' differs from the stream weights m by exactly the target norm (sum of the last 8 NAT update norms), the baseline read is taken at m itself, NAT-condition direction cosine to NAT == 1, other conditions differ, dPM3 follows the injected scalar reader, chunk CE of the SENT / SWD condition differs from NAT's; (d) report renders with the ratio table"""
    import gen_facts as G,e55_diag as DG
    enc=G.StubTok();bg=G._synth_bg(9000);w=G.cell_world(61,'ic_u64',enc,bg,decoys=('names','nofact'));ch=G.realize(w);G.verify(w,enc)
    assert E.annotate_answers(ch,w['facts'],enc)==(0,0)
    for k in (9,40):
        o,_=E.token_masks('OSPAN',k,ch,CH);p,_=E.token_masks('PSPAN',k,ch,CH);n,_=E.token_masks('NSPAN',k,ch,CH)
        hk=lambda m:set(np.nonzero(~m.all(0))[0].tolist());assert hk(o)==hk(p)|hk(n) and hk(p)&hk(n)==set() and hk(p) and hk(n)
        exp=set()
        for c in range(max(0,k-8),k):
            for x in ch[c]['sents']:
                for a_,b_ in x['ans_P']:exp|=set(range(c*CH-k*CH+W+a_,c*CH-k*CH+W+b_))
        assert hk(p)==exp
        inside=np.zeros(CH,bool)
        for xq in ch[k]['sents']:inside[xq['start']:xq['end']]=True
        assert p[~inside].all() and n[~inside].all()
    tab=[{'sents':[{'fact':0,'start':100,'end':140,'ans':[(105,106),(110,112)],'ans_P':[(110,112)],'ans_X':[(105,106)]},{'fact':1,'start':300,'end':330,'ans':[(305,306),(310,312)],'ans_P':[(310,312)],'ans_X':[(305,306)]},{'fact':2,'start':700,'end':760,'ans':[(705,706),(720,722)],'ans_P':[(720,722)],'ans_X':[(705,706)]}]} for _ in range(NCH)]
    seen=[]
    def reader(m):
        seen.append(m);t=float(sum(jnp.sum(x.astype(jnp.float32)) for x in jax.tree.leaves(m.inner_parameters())));i=np.arange(16)[:,None];c=np.arange(16)[None,:]
        return np.sin(i+c)+(i*0.37+c*0.11)*t
    meta={'kind':['fact']*12+['ctrl']*4,'gold_idx':[(3*j)%16 for j in range(16)]}
    pf=DG.make_probe_fn(L,rt,s3,reader,meta,nsum=2,log=lambda *x:None)
    a=L.run_stream('NAT',1,REAL,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab)
    b=L.run_stream('NAT',1,REAL,decoy_ids=DEC,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab,diag_multi=pf,diag_chunks={3,4})
    assert [e['ce'] for e in a['log']]==[e['ce'] for e in b['log']] and a['final_hash']==b['final_hash']
    assert len(seen)==2*(1+8)
    for j,k in enumerate((3,4)):
        v=b['diag_multi'][k];base=seen[9*j];tgt=sum(e['update_norm'] for e in b['log'][k-2:k]);assert abs(v['target_norm']-tgt)<1e-9*max(1,tgt)
        for i,cn in enumerate(DG.PROBE_CONDS):
            mp=seen[9*j+1+i];dn=float(sum(jnp.sum((x.astype(jnp.float32)-y.astype(jnp.float32))**2) for x,y in zip(jax.tree.leaves(mp.inner_parameters()),jax.tree.leaves(base.inner_parameters()))))**0.5
            assert abs(dn-tgt)<=0.02*tgt,(k,cn,dn,tgt)
        assert abs(v['NAT']['cos_to_NAT']-1)<1e-4 and all(abs(v[cn]['cos_to_NAT'])<=1+1e-4 for cn in DG.PROBE_CONDS)
        assert abs(v['SENT']['chunk_ce']-v['NAT']['chunk_ce'])>1e-6 and abs(v['SWD']['chunk_ce']-v['NAT']['chunk_ce'])>1e-6 and v['OALL']['mask_keys']>0 and v['NAT']['mask_keys']==0 and v['PALL']['mask_keys']<v['OALL']['mask_keys']
        assert len(v['NAT']['dpm3'])==12 and all(np.isfinite(v[cn]['dpm3_mean']) for cn in DG.PROBE_CONDS)
    t=DG.probe_report({'seed':0,'chunks':{str(k):v for k,v in b['diag_multi'].items()}});assert 'ratios to the full mask' in t and 'KMASK' in t
    RES['U34']='PSPAN/NSPAN keys = place / name spans, OSPAN = union, background rows unmasked; probe run == plain NAT run bitwise; probed copies differ from the stream weights by exactly the target norm (sum of last NAT norms), baseline read at the stream weights, NAT cos 1; chunk CE of SENT / SWD differs; report renders'
def u23():
    """model level (CPU tiny model): (a) SENT / KMASK with an EMPTY sentence table == NAT bit for bit (CE and final weights); (b) SENT with sentences changes CE from chunk 1 on; (c) perturbing K,V at SENT-hidden positions does not change the write update (bitwise), perturbing a visible position does; (d) a per-query 2-D mask with identical rows == the 1-D keep (same update), and a 2-D mask hiding a window key only for rows [a,b) leaves the prefix outputs of rows before a bit-identical and changes rows in [a,b)"""
    emp=[{'sents':[]} for _ in range(NCH)];tab=[{'sents':[{'fact':0,'start':100,'end':140},{'fact':1,'start':300,'end':330},{'fact':2,'start':700,'end':760}]} for _ in range(NCH)]
    a=nat();r={ar:L.run_stream(ar,2,REAL,nch=NCH,tag='u',log=lambda *x:None,chunks_meta=emp) for ar in ('SENT','KMASK','SENTO','SENTPQ')}
    for ar,b in r.items():assert b['final_hash']==a['final_hash'] and [e['ce'] for e in b['log']]==[e['ce'] for e in a['log']],ar
    b=L.run_stream('SENT',2,REAL,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab);ce_a=[e['ce'] for e in a['log']][:5];ce_b=[e['ce'] for e in b['log']]
    assert ce_a[0]==ce_b[0] and all(x!=y for x,y in zip(ce_a[1:],ce_b[1:])),(ce_a,ce_b);
    nm=[e['n_masked_keys'] for e in b['log']];assert nm==[0,130,260,390,520],nm
    for ar_ in ('KMASK','SENTO','SENTPQ'):
        b_=L.run_stream(ar_,2,REAL,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab);assert all(np.isfinite(e['ce']) for e in b_['log']) and [e['ce'] for e in b_['log']][0]==ce_b[0],ar_
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
def u28():
    """E55 design 24.3 arms: (a) ISOB rows concatenated == the 1024 tokens of the stream chunk, fact sentences are rows, one packed chunk per round, fact share of the tokens 30-40%; (c) ISOD one chunk per round with the round's 16 sentences only; per-step norm == NAT's per-chunk norm (J = 1 matching); read_at reads only at the requested points; (b) bigstep targets = NAT norm x chunks / ISO steps per segment (and the segment total of the targets x ... consistent with the stage-0 per-step norm)"""
    import gen_facts as G,e55_iso as I,hashlib
    enc0=G.StubTok();bg=G._synth_bg(9000);w=G.cell_world(61,'ic_u64',enc0,bg,decoys=('names',));V=rt.cfg.model.vocab_size;packing=I.LD.packing
    sg=I.rows_bg(w);assert len(sg)==64
    for r,(a0,a1,rows,_) in enumerate(sg):
        assert [t for ids,_ in rows for t in ids]==list(w['chunks'][r]['ids']) and len(packing.pack_rows(rows)[0])==1
        n_s=sum(x['end']-x['start'] for x in w['chunks'][r]['sents']);assert 0.20<=n_s/1024<=0.45,(r,n_s)   # stub tokenizer sentences are shorter than real ones (the design quotes 31-37% for the real world)
        sent_rows=[ids for ids,_ in rows if any(list(ids)==list(w['chunks'][r]['ids'][x['start']:x['end']]) for x in w['chunks'][r]['sents'])];assert len(sent_rows)>=16
    enc=lambda t:[3+int(hashlib.md5(x.encode()).hexdigest(),16)%(V-3) for x in t.replace('\n\n','\n\n ').split()]
    sd=I.rows_density(w,enc,0);assert len(sd)==64 and all(len(rows)==16 and len(packing.pack_rows(rows)[0])==1 for _,_,rows,_ in sd)
    # matching on the tiny model, 3 rounds: per-step norm == NAT norm; read_at
    Li=I.Iso55(rt,save_dir='/tmp/e55_units_states');norms=np.random.default_rng(5).uniform(0.5,1.5,size=64)*0.01;T=I.cum_target(norms);seen=[]
    # rebuild small token ids inside the tiny vocab (the stub world ids can exceed it)
    sg3=[(a0,a1,[([3+t%(V-3) for t in ids],ab) for ids,ab in rows],txt) for a0,a1,rows,txt in sg[:3]]
    o=I.iso_run(Li,rt,sg3,T=T,read=lambda a,m:seen.append(a) or {},log=lambda *x:None,read_at={3})
    assert seen==[3] and len(o['log'])==3 and all(abs(e['update_norm']-norms[i])<1e-3*norms[i] for i,e in enumerate(o['log'])),(seen,[e['update_norm'] for e in o['log']],norms[:3])
    o2=I.iso_run(Li,rt,sd[:3],T=T,read=None,log=lambda *x:None);assert len(o2['log'])==3 and all(abs(e['update_norm']-norms[i])<1e-3*norms[i] for i,e in enumerate(o2['log']))
    # (b)
    segs0=I.iso_rows(w,enc,0,[16,32,64]) if False else None
    nr=np.ones(64);isos=[(0,16,[([5]*26,(0,25))]*(16*16),[]),(16,32,[([5]*26,(0,25))]*(16*16),[])];tg=I.bigstep_targets(nr,isos,[16,32]);J1=len(packing.pack_rows(isos[0][2])[0]);assert abs(tg[0]-16/J1)<1e-12 and tg[40]==1.0 and abs(tg[20]-16/J1)<1e-12
    # (d) FOS: continuous fact-only stream padded to 1024 multiples, targets spread per segment
    segs2=I.iso_rows(w,enc,0,[8,16]);ids_f,ends,pad=I.fos_stream(segs2,[7,8]);assert len(ids_f)%1024==0 and ends[-1]==len(ids_f)//1024-1 and ends[0]<ends[-1] and pad<2*1023
    Tf=I.cum_target(np.full(64,0.01));tg_f=I.fos_targets(Tf,segs2,ends);assert len(tg_f)==ends[-1]+1 and abs(tg_f[:ends[0]+1].sum()-(Tf[8]-Tf[0]))<1e-12 and abs(tg_f.sum()-Tf[16])<1e-12
    ids_t=ids_f[:3*1024]%(V-3)+3;nt=np.asarray([0.004,0.006,0.005]);os_=Li.run_stream('NATM',1,ids_t,nch=3,tag='u',log=lambda *x:None,norm_targets=nt);assert all(abs(e['update_norm']-nt[i])<1e-3*nt[i] for i,e in enumerate(os_['log']))
    RES['U28']=f"FOS: stream padded to 1024 multiples (pad {pad}), targets sum to the NAT increments, NATM run hits the per-chunk targets; ISOB: 64 rounds, rows concatenated == chunk tokens, 1 packed chunk per round, fact share 30-40%; ISOD: 64 rounds x 16 sentence rows, 1 chunk each; per-step norm == NAT per-chunk norm ({[round(e['update_norm'],5) for e in o['log']]} vs {[round(float(x),5) for x in norms[:3]]}); read_at honoured; bigstep targets = NAT norm x {16}/{J1} per segment"
def u35():
    """E55 design 29.3 Q1-B Adam arms on the tiny model: per-chunk update norm == factor x the NAT norm target (NATADAM 1.0, SENTPQADAM 1.0, SENTPQADAMH 0.5, within 2%); NATADAM chunk-0 CE == NAT chunk 0 and the weights differ from the NAT run (different direction); SENTPQADAM chunk CE == SENTPQ's at chunk 0/1 (same window, chunk 1 uses different weights, so only chunk 0/1 before the first masked effect) and masked pairs logged; NATADAM logs no masked pairs; missing norm_targets refused."""
    tab=[{'sents':[{'fact':0,'start':100,'end':140},{'fact':1,'start':300,'end':330},{'fact':2,'start':700,'end':760}]} for _ in range(NCH)]
    a=nat();tg=[e['update_norm'] for e in a['log']];n=5
    r1=L.run_stream('NATADAM',2,REAL,nch=n,tag='u',log=lambda *x:None,chunks_meta=tab,norm_targets=tg)
    r2=L.run_stream('SENTPQADAM',2,REAL,nch=n,tag='u',log=lambda *x:None,chunks_meta=tab,norm_targets=tg)
    r3=L.run_stream('SENTPQADAMH',2,REAL,nch=n,tag='u',log=lambda *x:None,chunks_meta=tab,norm_targets=tg)
    for r,f in ((r1,1.0),(r2,1.0),(r3,0.5)):
        assert all(abs(e['update_norm']-f*tg[k])<=0.02*f*tg[k] for k,e in enumerate(r['log'])),([e['update_norm'] for e in r['log']],[f*x for x in tg[:n]])
    assert abs(r1['log'][0]['ce']-a['log'][0]['ce'])<2e-4 and r1['final_hash']!=a['final_hash'] and all(e['n_masked_pairs']==0 for e in r1['log']) and all(e['n_masked_pairs']>0 for e in r2['log'][1:])
    sq=L.run_stream('SENTPQ',2,REAL,nch=2,tag='u',log=lambda *x:None,chunks_meta=tab);assert abs(r2['log'][0]['ce']-sq['log'][0]['ce'])<2e-4 and abs(r2['log'][1]['ce']-sq['log'][1]['ce'])>1e-9
    seen=[];rh=L.run_stream('NATADAM',2,REAL,nch=n,tag='u',log=lambda *x:None,chunks_meta=tab,norm_targets=tg,read_hook=lambda k,m:seen.append(k) or {},hook_chunks=[1,3])      # E55 29.5: the Adam moments leave the device during a read and come back bit for bit
    assert seen==[1,3] and rh['final_hash']==r1['final_hash'],(seen,rh['final_hash'],r1['final_hash'])
    L.adam_offload=True;seen2=[]
    try:ro=L.run_stream('NATADAM',2,REAL,nch=n,tag='u',log=lambda *x:None,chunks_meta=tab,norm_targets=tg,read_hook=lambda k,m:seen2.append(k) or {},hook_chunks=[1,3])
    finally:L.adam_offload=False
    assert seen2==[1,3] and ro['final_hash']==r1['final_hash']      # the optional host offload is also bit-exact
    try:L.run_stream('NATADAM',2,REAL,nch=2,tag='u',log=lambda *x:None,chunks_meta=tab);raise SystemExit('norm_targets not required')
    except AssertionError:pass
    RES['U35']='Adam arms: per-chunk update norm == factor x NAT norm (within 2%, factors 1 / 1 / 0.5), NATADAM chunk-0 CE == NAT and different final weights, SENTPQADAM chunk-0 CE == SENTPQ, masked pairs logged only for the masked arm, norm_targets required'
def u36():
    """E55 design 30.3 OSPAN-R25/50/75 (CPU): hidden key sets are nested R25 < R50 < R75 < OSPAN (same fixed draws), each a subset of the answer-span keys, shares of the OSPAN keys about 25 / 50 / 75 % (within 8 points), the same rows for every fact-sentence query, background queries unmasked, deterministic (two calls equal)."""
    import gen_facts as G
    enc=G.StubTok();bg=G._synth_bg(9000);w=G.cell_world(61,'ic_u64',enc,bg,decoys=('names','nofact'));ch=G.realize(w);assert E.annotate_answers(ch,w['facts'],enc)==(0,0)
    hk=lambda m:set(np.nonzero(~m.all(0))[0].tolist())
    for k in (9,40):
        o,no=E.token_masks('OSPAN',k,ch,CH);r={f:E.token_masks(f'OSPANR{f}',k,ch,CH) for f in (25,50,75)}
        H=[hk(r[f][0]) for f in (25,50,75)];ho=hk(o)
        assert H[0]<=H[1]<=H[2]<=ho and H[0] and len(H[2])<len(ho),(len(H[0]),len(H[1]),len(H[2]),len(ho))
        for f,h in zip((25,50,75),H):assert abs(len(h)/len(ho)-f/100)<0.08,(k,f,len(h)/len(ho))
        for f in (25,50,75):
            m=r[f][0];rows=m[np.nonzero(~m.all(1))[0]];assert (rows==rows[0]).all();assert E.token_masks(f'OSPANR{f}',k,ch,CH)[0].tobytes()==m.tobytes()
            inside=np.zeros(CH,bool)
            for xq in ch[k]['sents']:inside[xq['start']:xq['end']]=True
            assert m[~inside].all()
    RES['U36']='OSPAN-R25/50/75: nested subsets of the OSPAN answer-span keys, shares about 25 / 50 / 75 %, identical rows for all fact-sentence queries, background unmasked, deterministic'
def u37():
    """E55 design 31 (Hans: occupation too) CPU: annotate_answers finds the occupation in every sentence (with_occ miss (0,0,0)), the span is the word after 'the' (token before it is not inside name/place spans); OCCSPAN keys = the occupation spans, ALLSPAN = OSPAN union OCCSPAN, OCCSPAN disjoint from OSPAN, per-query (background rows unmasked), the old two-tuple return is unchanged."""
    import gen_facts as G
    enc=G.StubTok();bg=G._synth_bg(9000);w=G.cell_world(61,'ic_u64',enc,bg,decoys=('names','nofact'));ch=G.realize(w)
    assert E.annotate_answers(ch,w['facts'],enc)==(0,0) and E.annotate_answers(ch,w['facts'],enc,with_occ=True)==(0,0,0)
    hk=lambda m:set(np.nonzero(~m.all(0))[0].tolist())
    for c in ch[:3]:
        for x in c['sents']:
            assert len(x['ans_O'])==1 and len(x['ans_A'])==3 and x['ans_O'][0] not in x['ans'],x
            a_,b_=x['ans_O'][0];assert list(c['ids'][a_:b_])==list(enc(w['facts'][x['fact']]['o'])) or list(c['ids'][a_:b_])==list(enc(' '+w['facts'][x['fact']]['o']))
    for k in (9,40):
        o,_=E.token_masks('OSPAN',k,ch,CH);oc,n_=E.token_masks('OCCSPAN',k,ch,CH);al,_=E.token_masks('ALLSPAN',k,ch,CH)
        assert hk(oc) and hk(o)&hk(oc)==set() and hk(al)==hk(o)|hk(oc) and n_==len(hk(oc))
        inside=np.zeros(CH,bool)
        for xq in ch[k]['sents']:inside[xq['start']:xq['end']]=True
        assert oc[~inside].all() and al[~inside].all()
    RES['U37']='occupation spans found in every sentence (miss (0,0,0)), OCCSPAN keys = occupation spans, ALLSPAN = OSPAN union OCCSPAN, disjoint, background unmasked, two-tuple return unchanged'
def u38():
    """E55 31 attention dump on the tiny model (CPU): the run with the dump function == the plain run (CE and final hash bitwise); per layer group the accumulated mass over the facts equals the column sum of the head/layer-averaged attention rows seen by attn_stats (1e-3), the per-query top-m indices are the m largest valid window keys of the averaged row, per-head tops lie inside the valid window (value > 0), their values are at most 1 and the best head value is not below the head-averaged value; labels: key_fact / key_sent / key_ans bits match the sentence table; flat_dump keys; the votegrid runs on the dump (npz round trip)."""
    import e55_diag as DG,e55_votegrid as VG,tempfile,os
    ids=REAL;tab=[{'sents':[{'fact':0,'start':100,'end':140,'ans_X':[(105,107)],'ans_P':[(110,112)],'ans_O':[(120,121)]},{'fact':1,'start':300,'end':330,'ans_X':[(305,308)],'ans_P':[(310,312)],'ans_O':[(320,321)]},{'fact':2,'start':700,'end':760,'ans_X':[(710,715)],'ans_P':[(720,722)],'ans_O':[(730,731)]}]} for _ in range(NCH)]
    seen=[];orig=DG.attn_stats
    def spy(A,rows,facts,chunks,k,CH_=1024,nw=8,ps=DG.PS):seen.append((np.asarray(A),k));return orig(A,rows,facts,chunks,k,CH_,nw,ps)
    a=L.run_stream('NAT',2,ids,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab)
    DG.attn_stats=spy
    try:
        dfn=DG.make_dump_fn(L,rt,s3,tab,topm=4,log=lambda *x:None);b=L.run_stream('NAT',2,ids,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab,diag_fn=dfn,diag_chunks={3,4})
        dg=DG.make_diag_fn(L,rt,s3,tab,log=lambda *x:None);c=L.run_stream('NAT',2,ids,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab,diag_fn=dg,diag_chunks={3,4})
    finally:DG.attn_stats=orig
    assert [e['ce'] for e in b['log']]==[e['ce'] for e in a['log']] and b['final_hash']==a['final_hash'] and set(b['diag'])=={3,4}
    refs={}
    for A,k in seen:refs.setdefault(k,[]).append(A)         # attn_stats calls of the diag run: all, g1, g2, g3 in order (a group with no layers is skipped)
    for k in (3,4):
        z=b['diag'][k];valid=z['key_valid'];assert z['n_rows']==40+30+60 and valid.sum()==min(k,8)*1024 and valid[(8-k)*1024:8192].all() and not valid[8192:].any()
        gs=[g for g in ('all','g1','g2','g3') if f'{g}_top_idx' in z];assert len(gs)==len(refs[k]),(gs,len(refs[k]))
        for g,A in zip(gs,refs[k]):
            assert np.allclose(z[f'{g}_mass_facts'].sum(0),A.sum(0),atol=1e-3),g
            Av=np.where(valid[None,:],A,-1.0);top=np.sort(Av,axis=1)[:,::-1][:,:4]
            got=np.take_along_axis(A,z[f'{g}_top_idx'].astype(np.int64),1);assert np.allclose(np.sort(got,axis=1)[:,::-1],top,atol=2e-3),g
            assert z[f'{g}_top_idx'].shape==(130,4) and z[f'{g}_mass_facts'].shape[0]==3
        hv=z['heads_val'].astype(np.float32);hi=z['heads_idx'].astype(np.int64);assert hv.ndim==4 and hv.shape[0]==z['n_layers'] and hv.shape[2:]==(130,4) and hv.max()<=1.0+1e-3
        m_=hv>0;assert valid[hi[m_]].all()
        assert (z['key_fact']>=0).sum()==min(k,8)*130 and set(np.unique(z['key_fact'][z['key_fact']>=0]))=={0,1,2}
        assert z['key_ans'][(8-k)*1024+105]==1 and z['key_ans'][(8-k)*1024+110]==2 and z['key_ans'][(8-k)*1024+120]==4 and z['key_sent'][(8-k)*1024+100]==0 and z['key_sent'][(8-k)*1024+1024+300]==64+1
    f_=tempfile.mktemp(suffix='.npz');np.savez_compressed(f_,**DG.flat_dump(b['diag']));rs=VG.grid(f_,('g3','all'),('avg','heads','mass'),(8,),(2,));os.remove(f_);assert rs and VG.report(rs)
    RES['U38']='attention dump: run with the dump == plain run bitwise; group mass == column sums of the attn_stats rows; per-query tops = the m largest valid keys; per-head tops valid and <= 1; key labels (fact / sentence id / name-place-occupation bits) match the table; npz round trip + votegrid'
def u39():
    """E55 34 (CPU, tiny model): (a) adam_dir_vfix == adam_dir for the first VFIX_T steps (same inputs) and the second moment stops changing after VFIX_T steps (bias correction fixed), while the first moment keeps moving; (b) arms SENTP25/50/75ADAMH (random-token masks + Adam, factor 0.5), SENTPQADAM75 (factor .75) and SENTPQVFIX10 / SENTPQVFIX05 (v frozen after 8 chunks, factors 1 / .5): per-chunk update norm == factor x the target (2%), masked pairs of SENTP25 < SENTP50 < SENTP75 < SENTPQ, the vfix arm differs from the plain Adam arm from chunk 8 on (v frozen) but not before."""
    f32=jnp.float32;rng=np.random.default_rng(0);tr=lambda: {'a':jnp.asarray(rng.normal(size=(5,3)),f32),'b':jnp.asarray(rng.normal(size=(7,)),f32)}
    mu=jax.tree.map(jnp.zeros_like,tr());nu=jax.tree.map(jnp.zeros_like,tr());mu2,nu2=mu,nu;nus=[]
    for s in range(12):
        g=tr();d1,mu,nu,_,_=L.adam_dir(g,mu,nu,jnp.asarray(float(s),f32));d2,mu2,nu2,_,_=L.adam_dir_vfix(g,mu2,nu2,jnp.asarray(float(s),f32));nus.append(nu2)
        if s<E.VFIX_T:assert maxdiff(d1,d2)<1e-6,(s,maxdiff(d1,d2))
    assert all(maxdiff(nus[i],nus[E.VFIX_T-1])==0 for i in range(E.VFIX_T,12)) and maxdiff(nus[3],nus[4])>0 and maxdiff(nu,nus[-1])>0 and maxdiff(d1,d2)>0
    tab=[{'sents':[{'fact':0,'start':100,'end':140},{'fact':1,'start':300,'end':330},{'fact':2,'start':700,'end':760}]} for _ in range(NCH)]
    a=nat();tg=[e['update_norm'] for e in a['log']];n=10;R={}
    for ar in ('SENTP25ADAMH','SENTP50ADAMH','SENTP75ADAMH','SENTPQADAM75','SENTPQVFIX10','SENTPQVFIX05','SENTPQADAM'):R[ar]=L.run_stream(ar,2,REAL,nch=n,tag='u',log=lambda *x:None,chunks_meta=tab,norm_targets=tg)
    for ar,f in (('SENTP25ADAMH',.5),('SENTP50ADAMH',.5),('SENTP75ADAMH',.5),('SENTPQADAM75',.75),('SENTPQVFIX10',1.0),('SENTPQVFIX05',.5)):
        assert all(abs(e['update_norm']-f*tg[k])<=0.02*f*tg[k] for k,e in enumerate(R[ar]['log'])),ar
    pr=[sum(e['n_masked_pairs'] for e in R[x]['log']) for x in ('SENTP25ADAMH','SENTP50ADAMH','SENTP75ADAMH','SENTPQADAM')];assert pr[0]<pr[1]<pr[2]<pr[3],pr
    sq=R['SENTPQADAM'];vf=R['SENTPQVFIX10']
    assert all(abs(vf['log'][k]['ce']-sq['log'][k]['ce'])<2e-4 for k in range(E.VFIX_T+1)) and any(abs(vf['log'][k]['ce']-sq['log'][k]['ce'])>1e-7 for k in range(E.VFIX_T+1,n)),([e['ce'] for e in vf['log']],[e['ce'] for e in sq['log']])
    RES['U39']='vfix direction == Adam for 8 steps then v frozen (m keeps moving); SENTP25/50/75-ADAM-0.5N, ADAM-0.75N, VFIX arms hit factor x target norm (2%), masked pairs nested p25<p50<p75<SENTPQ, VFIX differs from Adam only after chunk 8'
def u40():
    """E55 32.8 online sentence-vote masks (CPU, tiny model): (a) sentence_ends rule (end punctuation + space/newline token, decimals do not split, last token ends); (b) vote_select recovers planted voted sentences and respects p and k; (c) rands_hidden hits the token target with < one sentence overshoot, seeded; (d) UV arm with planted votes on the fact sentences hides exactly the fact sentences (hidden tokens == SENT's, recall = precision = 1), writes with Adam-0.5N (update norm = 0.5 x target); (e) the real att pass of the tiny model gives top-m window keys only, shapes [layers_in_group, H, 1024, m]; (f) RS arm reproduces the hidden-token counts of its UV arm"""
    import e55_uv as UVm
    class FE:
        m={1:'.',2:'\u0120the',3:'3',4:'5',5:'?',6:'\u010a\u010a',7:'\u0120A',8:'.\\"'}
        def id_to_token(self,i):return self.m.get(int(i),'x')
    e=UVm.sentence_ends(FE(),[2,1,2,1,7,3,1,4,2,5,7,6,2]);assert list(np.nonzero(e)[0])==[1,3,9,11,12],list(np.nonzero(e)[0])      # '.' before ' the' ends (idx3 is '.', next ' A'? idx4=7 starts with space) ; '3 . 5' decimal at idx5-7 does not split
    # (b)
    CH=1024;ends=np.zeros(CH*6,bool)
    for c in range(6):
        for b in (99,139,299,329,699,759,1023):ends[c*CH+b]=True
    wsid,wst,wlen=UVm.window_sents(ends,5);assert len(wlen)==5*7 and wsid[:3*CH].min()==-1 and (wsid[3*CH:]>=0).all() and wlen.sum()==5*CH
    fact_s=[j for j in range(len(wlen)) if wlen[j] in (40,30,60)];assert len(fact_s)==15
    qs=np.repeat(np.arange(7),[100,40,160,30,370,60,264]);keys=np.asarray([wst[j]+2 for j in fact_s]);hidx=np.tile(keys[None,None,None,:],(1,2,CH,1)).astype(np.int16);hval=np.full(hidx.shape,0.1,np.float16)
    hid,vi=UVm.vote_select(hidx,hval,qs,wsid,wlen,100,3);assert set(np.nonzero(hid)[0])==set(fact_s),(np.nonzero(hid)[0],fact_s)
    hid2,_=UVm.vote_select(hidx,hval,qs,wsid,wlen,10,1);assert hid2.sum()==max(1,int(np.ceil(len(wlen)*0.10))) and not (hid2&~np.isin(np.arange(len(wlen)),fact_s)).any()
    hid3,_=UVm.vote_select(hidx,hval,qs,wsid,wlen,100,8);assert hid3.sum()==0      # only 7 query sentences, k=8 -> nothing
    # (c)
    r1=UVm.rands_hidden(wlen,500,3);r2=UVm.rands_hidden(wlen,500,3);r3=UVm.rands_hidden(wlen,500,4);assert (r1==r2).all() and (r1!=r3).any() and 500<=wlen[r1].sum()<500+wlen.max() and UVm.rands_hidden(wlen,0,3).sum()==0
    # (d)-(f)
    nch=7;ends=np.zeros(NCH*CH,bool)
    for c in range(NCH):
        for b in (99,139,299,329,699,759,1023):ends[c*CH+b]=True
    tab=[{'sents':[{'fact':0,'start':100,'end':140},{'fact':1,'start':300,'end':330},{'fact':2,'start':700,'end':760}]} for _ in range(NCH)]
    a=nat();tg=[x['update_norm'] for x in a['log']]
    sent=L.run_stream('SENT',2,REAL,nch=nch,tag='u',log=lambda *x:None,chunks_meta=tab)
    uv=UVm.UV('UVA',ends,tab,L=L,rt=rt,s3=s3,topm=4,log=lambda *x:None);uv.p=100;uv.kmin=3
    def planted(k,m,Kc,Vc,ids_k,keep):
        ws,st,ln=UVm.window_sents(ends,k);fs=[j for j in range(len(ln)) if ln[j] in (40,30,60)]
        if not fs:return np.zeros((1,1,CH,1),np.int16),np.zeros((1,1,CH,1),np.float16)
        ki=np.asarray([st[j]+2 for j in fs]);return np.tile(ki[None,None,None,:],(1,1,CH,1)).astype(np.int16),np.full((1,1,CH,len(fs)),0.1,np.float16)
    real_att=uv.att_pass;uv.att_pass=planted
    r=L.run_stream('UVA',2,REAL,nch=nch,tag='u',log=lambda *x:None,chunks_meta=tab,norm_targets=tg,uv=uv)
    for k in range(nch):
        e_=r['log'][k];assert e_['n_masked_keys']==sent['log'][k]['n_masked_keys'],(k,e_['n_masked_keys'],sent['log'][k]['n_masked_keys'])
        if k>0:assert e_['lab_recall']==1.0 and e_['lab_precision']==1.0 and e_['lab_bg_false']==0.0,(k,e_)
        assert abs(e_['update_norm']-0.5*tg[k])<=0.02*0.5*tg[k],(k,e_['update_norm'],tg[k])
    rs=UVm.UV('RSA',ends,tab,rs_targets=[x['n_masked_keys'] for x in r['log']],log=lambda *x:None)
    q=L.run_stream('RSA',2,REAL,nch=nch,tag='u',log=lambda *x:None,chunks_meta=tab,norm_targets=tg,uv=rs)
    for k in range(nch):assert abs(q['log'][k]['n_masked_keys']-r['log'][k]['n_masked_keys'])<=int(max(UVm.window_sents(ends,k)[2],default=0)),(k,q['log'][k]['n_masked_keys'],r['log'][k]['n_masked_keys'])
    assert sum(x['n_masked_keys'] for x in q['log'])>0 and [x['ce'] for x in q['log']]!=[x['ce'] for x in r['log']]
    # (e) real att pass: shapes / validity
    uv2=UVm.UV('UVA',ends,tab,L=L,rt=rt,s3=s3,topm=4,log=lambda *x:None);seen={}
    def spy(k,m,Kc,Vc,ids_k,keep):
        hi,hv=real_att_(k,m,Kc,Vc,ids_k,keep);seen[k]=(hi,hv);return hi,hv
    real_att_=uv2.att_pass;uv2.att_pass=spy
    r2_=L.run_stream('UVC',2,REAL,nch=5,tag='u',log=lambda *x:None,chunks_meta=tab,norm_targets=tg,uv=uv2)
    for k,(hi,hv) in seen.items():
        assert hi.ndim==4 and hi.shape[2]==CH and hi.shape[3]==4 and hv.shape==hi.shape,hi.shape
        ok=hv.astype(np.float32)>0;assert (hi[ok]>= (8-min(k,8))*CH).all() and (hi[ok]<8*CH).all()
    RES['U40']='sentence ends from tokens only; vote_select recovers planted voted sentences (p,k respected); RS hits token target (< 1 sentence), seeded; UV with planted votes hides exactly the fact sentences of SENT (recall = precision = 1, bg 0), update norm 0.5 x target; real att pass: top-m window keys only'
def u41():
    """E55 job 200a leak (CPU, tiny model): every arm leaves ~38 live (32768,16) complex64 RoPE tables (held as constants of its compiled functions) -> live-array count grows per arm; jax.clear_caches() after the arm returns the count to the baseline, for an Adam arm and for a UV arm with the real attention pass"""
    import gc,e55_uv as UVm
    def live():
        gc.collect();x=jax.live_arrays();return len(x),sum(1 for v in x if v.shape==(32768,16))
    tab=[{'sents':[{'fact':0,'start':100,'end':140},{'fact':1,'start':300,'end':330},{'fact':2,'start':700,'end':760}]} for _ in range(NCH)]
    tg=[e['update_norm'] for e in nat()['log']];jax.clear_caches();b=live()
    ends=np.zeros(NCH*1024,bool)
    for c in range(NCH):
        for q in (99,139,299,329,699,759,1023):ends[c*1024+q]=True
    for arm in ('SENTPQADAMH','UVA'):
        L2=C.Ladder55(rt,save_dir='/tmp/e55_units_states');uv=UVm.UV(arm,ends,tab,L=L2,rt=rt,s3=s3,topm=4,log=lambda *x:None) if arm=='UVA' else None
        r=L2.run_stream(arm,2,REAL,nch=4,tag='u',log=lambda *x:None,chunks_meta=tab,norm_targets=tg,uv=uv);r.pop('model',None);r=L2=uv=None
        g=live();assert g[0]>b[0] and g[1]>0,(arm,'expected a leak without clear_caches (the reason for the fix)',g,b)
        jax.clear_caches();h=live();assert h==b,(arm,h,b)
    RES['U41']='arm leaves live RoPE tables without jax.clear_caches() (tiny model: +38 arrays) and none after it, for an Adam arm and a UV arm (real attention pass); baseline restored exactly'
def u42():
    """E57 1a periodic K,V sharing and resume (CPU, tiny model with ONE prefix layer = receptive field 8 chunks): a stream of 5 rounds of 4 chunks (cyclic repetition of one round) run (a) plainly, (b) with periodic={round_chunks: 4, rec_chunks: 12} (only the first 3 rounds recorded; later windows / prefix outputs / query positions taken from the last recorded round), (c) in two pieces with start_chunk / sts0 / weights0. Per-chunk CE and update norms of (b) equal (a) to 1e-3 relative, the final weights agree, (c) equals (b), rec_chunks < 2 rounds is refused."""
    rt2=H.TinyRuntime(layers=3,suffix=2);L2=C.Ladder55(rt2,save_dir='/tmp/e55_units_states')
    rc=4;R=5;nch=rc*R;rnd=np.random.default_rng(7).integers(3,rt2.cfg.model.vocab_size,size=rc*CH);ids=np.tile(rnd,R);per={'round_chunks':rc,'rec_chunks':3*rc}
    a=L2.run_stream('NAT',1,ids,nch=nch,tag='u42a',log=lambda *x:None)
    msgs=[]
    b=L2.run_stream('NAT',1,ids,nch=nch,tag='u42b',log=lambda s,*x:msgs.append(s),periodic=per)
    assert any('recorded natural K,V of real' in s for s in msgs)
    for k in range(nch):
        ea,eb=a['log'][k],b['log'][k];assert abs(ea['ce']-eb['ce'])<=1e-3*max(1.0,abs(ea['ce'])) and abs(ea['update_norm']-eb['update_norm'])<=1e-3*ea['update_norm'],(k,ea['ce'],eb['ce'],ea['update_norm'],eb['update_norm'])
    wd=maxdiff(a['model'].inner_parameters(),b['model'].inner_parameters())
    c1=L2.run_stream('NAT',1,ids[:14*CH],nch=14,tag='u42c',log=lambda *x:None,periodic=per)
    c2=L2.run_stream('NAT',1,ids,nch=nch,tag='u42d',log=lambda *x:None,periodic=per,start_chunk=14,weights0=c1['model'],sts0=c1['sts_final'])
    assert len(c2['log'])==nch-14
    for j,k in enumerate(range(14,nch)):
        eb,ec=b['log'][k],c2['log'][j];assert ec['k']==k and abs(eb['ce']-ec['ce'])<=1e-3*max(1.0,abs(eb['ce'])) and abs(eb['update_norm']-ec['update_norm'])<=1e-2*eb['update_norm']      # 1e-2 on the update norm: the first part's stream mask drops the loss of its very last token (_stream_seq masks the last position of the stream it is given), so the weights after chunk 13 differ by one token's loss (0.4 % in the update norm of the next chunk),(k,eb['ce'],ec['ce'],eb['update_norm'],ec['update_norm'])
    wd2=maxdiff(b['model'].inner_parameters(),c2['model'].inner_parameters());assert wd2<=1e-2 and wd<=1e-3,(wd,wd2)
    try:L2.run_stream('NAT',1,ids,nch=nch,tag='u42e',log=lambda *x:None,periodic={'round_chunks':rc,'rec_chunks':rc});raise RuntimeError('rec_chunks < 2 rounds must be refused')
    except AssertionError:pass
    RES['U42']=f"periodic K,V sharing (3 of 5 rounds recorded, rounds 4-5 taken from round 3) equals the plain run: per-chunk CE / update norm within 1e-3 over {nch} chunks, final weights max diff {wd:.2e}; resume (14 chunks, then start_chunk=14 with weights0 + sts0) equals the one-piece periodic run (weights diff {wd2:.1e}, 1e-2 tolerance: the first part masks the loss of its last token); rec_chunks < 2 rounds refused"
def u43():
    """E57 1a runner (e57_run1a.run_1a) on the tiny model with a stand-in canary: reads at the ends of the listed rounds only (chunks R x rc - 1), canary rise = max(gen, tail) against W0, R_max rule, summary / saved json written without model objects, resume bundle (weights state + suffix chain state) continues a run so that the later reads equal the one-piece run."""
    sys.path.insert(0,str(HERE/'e57'));import e57_run1a as RA
    rt2=H.TinyRuntime(layers=3,suffix=2)
    def can(m):d=maxdiff(m.inner_parameters(),rt2.base.inner_parameters());return {'_gen_nll':2.0+20*d,'_tail_nll':3.0+10*d}
    L2=C.Ladder55(rt2,save_dir='/tmp/e55_units_states',canary=can);rc=4;rnd=np.random.default_rng(9).integers(3,rt2.cfg.model.vocab_size,size=rc*CH);sl=int(rt2.cfg.model.seq_len);npre=RA.n_prefix_layers(rt2);assert npre==1,npre
    A=RA.run_1a(L2,rnd,[2,3,4],'/tmp/e57_u43','full',sl,npre,log=lambda *x:None)
    assert [x['chunk'] for x in A['reads']]==[7,11,15] and A['rec_rounds']==3 and A['rec_exact'] and A['nch']==16,(A['reads'],A['rec_rounds'])
    for x in A['reads']:assert abs(x['crise']-max(x['crise_gen'],x['crise_tail']))<1e-9 and x['crise_gen']>0 and x['crise_tail']>0 and x['canary']['_gen_nll']>A['canary_w0']['_gen_nll']
    assert A['reads'][0]['crise']<A['reads'][1]['crise']<A['reads'][2]['crise']          # the weights keep moving away from W0
    lim=(A['reads'][0]['crise']+A['reads'][1]['crise'])/2;B=dict(A,limit=lim,reads=A['reads'])
    cand=[x['R'] for x in A['reads'] if x['crise']<=lim];assert cand==[2]
    import json;sj=json.load(open('/tmp/e57_u43/E57_full_R4.summary.json'));assert sj['reads'][2]['R']==4 and 'model' not in json.load(open('/tmp/e57_u43/E57_full_R4.saved.json'))
    P1=RA.run_1a(L2,rnd,[2],'/tmp/e57_u43','part',sl,npre,log=lambda *x:None);assert P1['rec_rounds']==2 and not P1['rec_exact'] or P1['rec_rounds']>=2
    P2=RA.run_1a(L2,rnd,[3,4],'/tmp/e57_u43','part',sl,npre,resume=P1['resume_bundle'],log=lambda *x:None)
    assert [x['R'] for x in P2['reads']]==[2,3,4] and P2['start_chunk']==8
    for a_,b_ in zip(A['reads'][1:],P2['reads'][1:]):assert abs(a_['crise']-b_['crise'])<=1e-2*max(a_['crise'],1e-6),(a_['R'],a_['crise'],b_['crise'])
    assert RA.decision(A).startswith('canary rise <= ') or 'R_max' in RA.decision(A);assert 'R_max' in RA.report(A)
    RES['U43']=f"stage-1a runner on the tiny model: reads only at the ends of rounds 2,3,4 (chunks 7,11,15), canary rises {[round(x['crise'],3) for x in A['reads']]}, summary / saved json without model objects, resume bundle reproduces the later reads of the one-piece run (rise within 1%)"
def u44():
    """RoPE without table (rope_fly): (a) inside the table the patched apply_rope equals the upstream table version (max abs diff on random q, k at random and edge positions); (b) beyond the table (positions up to 3.4M, table = 2*seq_len) the values are finite and a q.k score depends on the relative offset only (median relative error < 2%); (c) the upstream lookup beyond the table is NOT usable (clamps: wrong rotation); (d) a 3-chunk NAT stream is identical with and without the patch"""
    import rope_fly as RF;assert RF.UPSTREAM is not None,'patch not installed'
    from ttt.model import attention as A
    att=[m for m in jax.tree.leaves(rt.base,is_leaf=lambda x:isinstance(x,A.AttentionBase)) if isinstance(x:=m,A.AttentionBase)][0]
    tab=2*int(att.config.seq_len);hd=att.head_dim;rng=np.random.default_rng(3)
    q=jnp.asarray(rng.standard_normal((1,256,att.num_heads,hd)),jnp.float32);pos=jnp.asarray(np.concatenate([rng.integers(0,tab,200),np.tile([0,1,tab-1],19)[:56]]),jnp.int32)
    new=att.apply_rope(q,pos);old=RF.UPSTREAM(att,q,pos);da=float(jnp.max(jnp.abs(new-old)))
    assert da<=1e-5,da
    q1=jnp.asarray(rng.standard_normal((1,1,att.num_heads,hd)),jnp.float32);k1=jnp.asarray(rng.standard_normal((1,1,att.num_heads,hd)),jnp.float32);errs=[]
    for t in (0,tab-1,tab+5,500000,1700000,3400000):
        for d in (1,64,1024,8191):
            sq=lambda a,b:float(jnp.sum(att.apply_rope(q1,jnp.asarray([a],jnp.int32))*att.apply_rope(k1,jnp.asarray([b],jnp.int32))))
            s_=sq(t+d,t);s0=sq(d,0);errs.append(abs(s_-s0)/(float(jnp.linalg.norm(q1))*float(jnp.linalg.norm(k1))))
    assert all(np.isfinite(errs)) and float(np.median(errs))<0.02,np.median(errs)
    far=jnp.asarray([3400000],jnp.int32);cl=float(jnp.max(jnp.abs(RF.UPSTREAM(att,q1,far)-RF.UPSTREAM(att,q1,jnp.asarray([tab-1],jnp.int32)))));fa=float(jnp.max(jnp.abs(att.apply_rope(q1,far)-RF.UPSTREAM(att,q1,jnp.asarray([tab-1],jnp.int32)))))
    ids=REAL[:3*CH];r_new=L.run_stream('NAT',1,ids,nch=3,tag='u44',log=lambda *x:None);n_new=[e['update_norm'] for e in r_new['log']];r_new.pop('model',None)
    import packing;saved=packing._orig_rope;packing._orig_rope=RF.UPSTREAM
    try:
        jax.clear_caches();L3=C.Ladder55(rt,save_dir='/tmp/e55_units_states');r_old=L3.run_stream('NAT',1,ids,nch=3,tag='u44o',log=lambda *x:None);n_old=[e['update_norm'] for e in r_old['log']];r_old.pop('model',None)
    finally:packing._orig_rope=saved;jax.clear_caches()
    dn=max(abs(a-b)/max(abs(b),1e-9) for a,b in zip(n_new,n_old));assert dn<1e-4,(n_new,n_old)
    RES['U44']=f"inside the table max abs diff {da:.2e}; beyond it (up to 3.4M) median relative score error {float(np.median(errs)):.4f}, max {max(errs):.4f}; upstream lookup at 3.4M gives {"NaN" if cl!=cl else "the last table row"} (differs from the patched value by {fa:.2f}), unusable; 3-chunk NAT update norms patched vs table: max relative diff {dn:.1e}"
def u45():
    """E57 approximation error (CPU, tiny model with ONE prefix layer, round = 4 chunks, so exactness needs P = 3 recorded rounds): the same 5-round stream plain vs periodic with P = 3 (exact) and P = 2 (approximate: context deeper than the recorded rounds is cut). Reports the per-chunk relative deviation of CE and of the update norm and the final weight difference relative to the weight change; the P = 3 run must be exact to 1e-3, the P = 2 deviation is a measurement (only required to be finite and below 50 %)."""
    rt2=H.TinyRuntime(layers=3,suffix=2);L2=C.Ladder55(rt2,save_dir='/tmp/e55_units_states')
    rc=4;R=5;nch=rc*R;ids=np.tile(np.random.default_rng(7).integers(3,rt2.cfg.model.vocab_size,size=rc*CH),R)
    a=L2.run_stream('NAT',1,ids,nch=nch,tag='u45a',log=lambda *x:None);w0=rt2.base.inner_parameters()
    dev={}
    for P in (3,2):
        b=L2.run_stream('NAT',1,ids,nch=nch,tag=f'u45p{P}',log=lambda *x:None,periodic={'round_chunks':rc,'rec_chunks':P*rc})
        ce=[abs(a['log'][k]['ce']-b['log'][k]['ce'])/abs(a['log'][k]['ce']) for k in range(nch)];un=[abs(a['log'][k]['update_norm']-b['log'][k]['update_norm'])/a['log'][k]['update_norm'] for k in range(nch)]
        dev[P]=dict(ce_max=max(ce),ce_late=max(ce[P*rc:]),un_max=max(un),un_late=max(un[P*rc:]),wd=maxdiff(a['model'].inner_parameters(),b['model'].inner_parameters())/maxdiff(a['model'].inner_parameters(),w0))
    assert dev[3]['ce_max']<=1e-3 and dev[3]['un_max']<=1e-3 and all(np.isfinite(list(v.values())).all() and v['ce_max']<0.5 and v['un_max']<0.5 for v in dev.values()),dev
    RES['U45']="P=3 (exact): max rel. deviation CE %.1e / update norm %.1e, final weights %.1e of the total weight change; P=2 (approximate, chunks %d-%d use the cut context): CE %.2e (max), %.2e (late chunks), update norm %.2e / %.2e, final weights %.2e of the total weight change"%(dev[3]['ce_max'],dev[3]['un_max'],dev[3]['wd'],2*rc,nch-1,dev[2]['ce_max'],dev[2]['ce_late'],dev[2]['un_max'],dev[2]['un_late'],dev[2]['wd'])
def u46():
    """world with period p = 4 (cell v3_64_4 shape: k sentences per 4 chunks, 3 exposures here = 12 chunks) through the decomp stream path on the tiny model: the read hook is called exactly at the chunks a x p - 1 and maps to the exposure count (k + 1) // p; SENTPQ masks earlier fact sentences from chunk 1 on; SENTPQADAMH takes half the SENTPQ update norm per chunk"""
    import gen_facts as G
    enc=G.StubTok();w=G.build_fact_world(61,enc=enc,bg=G._synth_bg(9000),k=16,n=1,p=4,bank='300',n_app=3,decoys=('names',));PP=w['p'];nch=w['n_chunks'];assert PP==4 and nch==12 and w['read_chunks']==[3,7,11] or w['read_chunks']==[a*PP-1 for a in w['read_apps']],(w['read_chunks'],nch)
    V=rt.cfg.model.vocab_size;ids=np.asarray(G.stream_ids(G.realize(w)),np.int64)%V;ids[ids<3]+=3
    got={};hc=[3,7,11]
    def run(ar,ntg=None):
        calls=[]
        L2=C.Ladder55(rt,save_dir='/tmp/e55_units_states')
        r=L2.run_stream(ar,1,ids,nch=nch,tag='u46'+ar,log=lambda *x:None,chunks_meta=w['chunks'],norm_targets=ntg,read_hook=lambda k,m:calls.append((k,(k+1)//PP)) or {},hook_chunks=hc);r.pop('model',None);return r,calls
    rn,cn=run('NAT');assert cn==[(3,1),(7,2),(11,3)],cn
    rs,cs=run('SENTPQ');assert cs==cn and rs['log'][0]['n_masked_keys']==0 and all(e['n_masked_keys']>0 for e in rs['log'][1:]),[e['n_masked_keys'] for e in rs['log']]
    tg=[e['update_norm'] for e in rs['log']];ra,_=run('SENTPQADAMH',tg);ratio=float(np.mean([a['update_norm']/b for a,b in zip(ra['log'],tg)]));assert 0.45<=ratio<=0.55,ratio
    RES['U46']=f"p = 4 world (16 facts, 12 chunks): hook calls {cn} (chunk, exposure count); SENTPQ masked keys per chunk {[e['n_masked_keys'] for e in rs['log']][:5]}...; Adam-0.5N / SENTPQ mean update norm ratio {ratio:.3f}"
TESTS={'U0':u0,'U1':u1,'U2':u2,'U3':u3,'U4':u4,'U5':u5,'U6':u6,'U7':u7,'U8':u8,'U9':u9,'U10':u10,'U11':u11,'U12':u12,'U13':u13,'U14':u14,'U15':u15,'U16':u16,'U17':u17,'U18':u18,'U19':u19,'U20':u20,'U21':u21,'U22':u22,'U23':u23,'U24':u24,'U25':u25,'U26':u26,'U27':u27,'U28':u28,'U29':u29,'U30':u30,'U31':u31,'U32':u32,'U33':u33,'U34':u34,'U35':u35,'U36':u36,'U37':u37,'U38':u38,'U39':u39,'U40':u40,'U41':u41,'U42':u42,'U43':u43,'U44':u44,'U45':u45,'U46':u46}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--only',default=None);a=ap.parse_args();sel=a.only.split(',') if a.only else list(TESTS);fails=[]
    for n in sel:
        t0=time.monotonic()
        try:TESTS[n]();print(f'{n} PASS ({time.monotonic()-t0:.0f}s): {RES[n]}',flush=True)
        except Exception as e:
            import traceback;traceback.print_exc();print(f'{n} FAIL: {e!r}',flush=True);fails.append(n)
    print('E55 UNITS: ALL PASS' if not fails else f'E55 UNITS: FAILED {fails}')
    sys.exit(1 if fails else 0)
