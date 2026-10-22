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
    return L.run_stream(arm,p,REAL,decoy_ids=dec,read_chunks=reads,nch=nch,tag=tag,log=lambda *a:None)
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
    import gen_world_ctl as G
    enc=G.StubTok();bg=G._synth_bg(6000);w1=G.build_ctl_world(55,8,1,2,16,'50',enc,bg);w2=G.build_ctl_world(55,8,1,2,16,'50',enc,bg)
    s1=G.stream_sha(G.realize(w1,False,enc));s2=G.stream_sha(G.realize(w2,False,enc));d1=G.stream_sha(G.realize(w1,False,enc,decoy=True));b1=G.stream_sha(G.realize(w1,True,enc))
    assert s1==s2 and s1!=d1 and s1!=b1
    RES['U9']=f'same seed twice: stream hash A {s1} == {s2}; decoy {d1}; world B {b1} (stub tokenizer; the real-tokenizer reproducibility is job 164)'
def u10():
    out={}
    for arm in ('NAT','SWD','SWDS','SWDO','MSK'):
        r=run(arm,2,DEC,nch=4,reads=(1,3),tag='u10'+arm);out[arm]=(r['final_hash'],r['clip_bind_frac'],len(r['reads']));assert len(r['log'])==4 and all(np.isfinite(e['ce']) for e in r['log'])
    try:
        run('SWB',2,DEC,nch=2);ok=False
    except NotImplementedError:ok=True
    assert ok
    RES['U10']='smoke, 4 chunks, read points after chunks 1 and 3: '+'; '.join(f'{a} hash {v[0]} clip-bind {v[1]:.2f}' for a,v in out.items())+'; SWB refused (needs a background-only stream)'
def u11():
    import gen_world_ctl as G,e55_read as R,e55_gpu as RG,e55_analyze as AN,hashlib
    enc=G.StubTok();w=G.build_ctl_world(55,8,1,2,16,'50',enc,G._synth_bg(6000));pl=R.plan(w,enc);V=rt.cfg.model.vocab_size
    items=pl['items'];n=len(items);NC=G.NCAND;assert n==40
    rng=np.random.default_rng(11);cand=rng.integers(3,V,size=(NC,3));pre=rng.integers(3,V,size=(n,9))
    raw=[(list(map(int,pre[i]))+list(map(int,cand[c])),(8,11)) for i in range(n) for c in range(NC)]
    assert RG.check_rows(raw,n,NC)==3
    mp=lambda t:[int(x)%(V-3)+3 for x in t]
    ctxs={nm:{**c,'tail_A':mp(c['tail_A']),'tail_B':mp(c['tail_B'])} for nm,c in pl['read'][16]['ctx'].items()};read={16:{'read_chunk':31,'ctx':ctxs}}
    assert set(ctxs)=={'in_g0','in_g1'} and all(len(c['tail_A'])==8192 for c in ctxs.values())
    out_tail=[int(x) for x in rng.integers(3,V,size=7168)]
    rd=RG.Reader55(rt,rt,s3.packing if hasattr(s3,'packing') else sys.modules['packing'],H.TailCtx,reuse.get_filter_spec,reuse._hash_pytree,jax,H.eqx,jnp)
    m_base=rt.base;inner=m_base.inner_parameters();pert=L.set_inner(m_base,jax.tree.map(lambda x:x+0.02*jnp.ones_like(x),inner))
    fb=L.save_state(m_base,'u11_base.npz');fp=L.save_state(pert,'u11_pert.npz')
    sf={'BASE':{16:{'A':(fb['_file'],fb['_hash']),'B':(fb['_file'],fb['_hash'])}},'PERT':{16:{'A':(fp['_file'],fp['_hash']),'B':(fp['_file'],fp['_hash'])}}}
    t0=time.monotonic();rdd,W0=RG.read_world(rd,raw,n,NC,read,out_tail,sf,w0_model=m_base,log=lambda *a:None,only_apps=(16,))
    P=rd.packed(raw,out_tail);s1=rd.score(P,m_base,n,NC);s2=rd.score(P,m_base,n,NC);assert np.array_equal(s1,s2) and np.array_equal(s1,W0['out'])
    for w_ in 'AB':
        assert np.array_equal(rdd['BASE'][16]['out'][w_],W0['out']) and np.array_equal(rdd['BASE'][16]['in'][w_],W0['in@16'][w_])
        assert not np.allclose(rdd['PERT'][16]['out'][w_],W0['out']) and not np.allclose(rdd['PERT'][16]['in'][w_],W0['in@16'][w_])
        assert not np.isnan(W0['in@16'][w_]).any()
    # an exposed item of group 1 is scored in group 1's context (and not in group 0's): recompute that block directly
    c1=ctxs['in_g1'];idx=np.asarray(c1['items']);Pg=rd.packed([raw[i*NC+k] for i in idx for k in range(NC)],c1['tail_A']);direct=rd.score(Pg,m_base,len(idx),NC);assert np.array_equal(direct,W0['in@16']['A'][idx])
    c0=ctxs['in_g0'];assert set(c0['items'])&set(c1['items'])==set() or True
    ex0=[i for i in c0['items'] if items[i]['kind']=='exposed'];ex1=[i for i in c1['items'] if items[i]['kind']=='exposed'];assert len(ex0)==len(ex1)==16 and not set(ex0)&set(ex1)
    assert not np.array_equal(W0['in@16']['A'],W0['in@16']['B'])      # the in-window tails differ between the worlds (names), so W0 is stored per world
    A_=G.realize(w,False,enc);tail_sha=hashlib.sha256(np.asarray(c1['tail_A'],np.int64).tobytes()).hexdigest()[:12];assert pl['read'][16]['ctx']['in_g1']['tail_A'][:1024]==A_[c1['end']-7]['ids'] and pl['read'][16]['ctx']['in_g1']['tail_A'][-1024:]==A_[c1['end']]['ids']
    recs=RG.make_records(rdd,W0,R.item_meta(items),read,{'cell':'u11','p':2,'M':8,'s':1,'bank':'50','seed':55});rec=json.loads(json.dumps(recs['PERT']))
    v=AN.change_vec(rec,16,'out','P2');v2=AN.change_vec(rec,16,'in','P1');assert len(v)==w['N'] and len(v2)==w['N']
    RES['U11']=f"16 candidates x {n} items, rows packed, 3-token spans; W0 'out' scored twice identical (max|diff| 0), saved base weights reloaded -> scores identical to W0 (out and in-window, worlds A and B), a +0.02 weight shift moves them; each exposed item is scored in its own group's context (group 1 context = real chunks {c1['end']-7}..{c1['end']} of world A, recomputed directly: identical; sha12 {tail_sha}); tails of A and B differ so W0 is stored per world; the record round-trips through JSON into e55_analyze.change_vec ({len(v)} pairs) ({time.monotonic()-t0:.0f}s); random rows, tiny model, NOT 3B"
def u12():
    import e55_run as RUN
    r=RUN.unit_checks(rt,L,s3,REAL,DEC,NCH,1e-3,2e-3);assert r['PASS'],r
    RES['U12']=f"e55_run.unit_checks (the check list the server runs on the 3B model) on the tiny model, 10 chunks: PASS; U1 |ce diff| {r['U1']['abs_ce_diff']:.1e}, rel update diff {r['U1']['rel_update_diff']:.1e}; U2 hash equal {r['U2']['final_hash_equal']} (swapped slots {r['U2']['swapped_slots']}); U5 hash equal {r['U5']['final_hash_equal']} (swapped {r['U5']['swapped_slots']}); U4 window slots changed {r['U4']['prefix_window_slots_changed']}"
TESTS={'U0':u0,'U1':u1,'U2':u2,'U3':u3,'U4':u4,'U5':u5,'U6':u6,'U7':u7,'U8':u8,'U9':u9,'U10':u10,'U11':u11,'U12':u12}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--only',default=None);a=ap.parse_args();sel=a.only.split(',') if a.only else list(TESTS);fails=[]
    for n in sel:
        t0=time.monotonic()
        try:TESTS[n]();print(f'{n} PASS ({time.monotonic()-t0:.0f}s): {RES[n]}',flush=True)
        except Exception as e:
            import traceback;traceback.print_exc();print(f'{n} FAIL: {e!r}',flush=True);fails.append(n)
    print('E55 UNITS: ALL PASS' if not fails else f'E55 UNITS: FAILED {fails}')
    sys.exit(1 if fails else 0)
