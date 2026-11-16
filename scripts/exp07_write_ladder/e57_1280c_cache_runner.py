"""E57 job 1280-cache (job 1302; GPU; new file; e57_1280_runner.py and every registered 1280 file are imported and NOT edited): CACHE-PATH ONE-STEP MARGINS at the 1280 branch chunks. Seeds 101..108 only: the natural NATS trajectory is re-run exactly as job 1280c ran it (same arguments, K0 = 8 irrelevant for NATS, deterministic XLA ops) and the SAME branch chunks of each seed (taken from the stored results_s<seed>.json) are used. At every branch chunk k, from the weights W_t before the write of chunk k, ONE update step is taken with each of four update-path masks on the cached window K,V (the prefix layers are not recomputed):
  NATS      natural keep (reproduces the logged gnorm / coef of the trajectory: isolation check)
  NATMASK   ALLQ256 band mask (as 1280)
  CACHE_A   cached oracle: OracleUV keep (a sentence row of fact f does not see the earlier sentences of fact f in the window), the mask of the NLL condition A of 1280
  CACHE_AP  cached placebo: PlaceboOracleUV keep (condition A_P of 1280)
and the primary (logsumexp) and secondary (largest wrong) margins of the facts of chunk k are read after each step with the same reader, candidate rows and neutral tail as 1280. The recompute path (B, B_P, BAND), t3, gain_noprior and p16 are NOT repeated. Per chunk the NLL arrays of NC, A, A_P on the F tokens are also stored (identity check against 1280: nc, a, a_P).
Output: <out>/results_cache_s<seed>.json. Identity check (CPU, e57_1280c_cache_analyze.py --check): isolation gnorm / coef against the trajectory log, NATS and NATMASK margins and the NC / A / A_P NLL arrays against the stored 1280 results; any difference aborts the job (exit 13).
  python3 e57_1280c_cache_runner.py --seed S --out O --branches 10,12,... [--smoke N] | --selftest"""
import sys,os,json,time,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_1280_runner as R
import e57_1280_margin as MG
CH=R.CH;NW=R.NW;WIN=R.WIN;STORE=R.STORE
def make_factory(cfg):
    import e57_oracle_masks as OM,e57_closure_masks as CM
    def factory(L,LD,s3,rt,holder,chunks,log=print):
        jax,jnp,eqx=LD.jax,LD.jnp,LD.eqx;L.rec_cache={};STORE['L']=L;orig=L.run_stream
        def rs(arm,p,real_ids,*a,**kw):STORE['ids']=np.asarray(real_ids);STORE['nofact']=kw.get('bg_ids');return orig(arm,p,real_ids,*a,**kw)
        L.run_stream=rs
        @eqx.filter_jit
        def logp_mask(m,st,c,pref,kmask):
            s3.S3SUF['kmask']=kmask
            try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
            finally:s3.S3SUF.clear()
            lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
        elig=R.eligible_chunks(chunks,cfg['nch']);br=cfg['branches'];assert br,'--branches (the 1280 branch chunks of the seed) is required';STORE['branches']=list(br);STORE['elig']=sorted(elig);STORE['results']={}
        assert all(k in elig for k in br),('a requested branch chunk is not eligible',br,sorted(elig))
        log(f'  1280-cache branch chunks (eligible {len(elig)}): {list(br)}')
        cfg_br=set(br)
        def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
            if k not in cfg_br:return {}
            t0=time.monotonic();base=np.asarray(keep,bool);c=ctx['c'];st_w=ctx['st_w'];q0=jnp.int32(k*CH);lm=np.asarray(c.loss_masks,bool)
            F=[t for t in elig[k] if lm[t]];res=dict(chunk=k,n_F=len(F));fa=sorted({x['fact'] for x in chunks[k]['sents']})
            def lp_of(kmask):
                with rt.mesh:
                    pref=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,q0,jnp.asarray(kmask));lp=np.asarray(logp_mask(m,st_w,c,pref,jnp.asarray(kmask)),np.float64)
                return lp,pref
            keepA,_=OM.OracleUV(chunks,8,lambda *a:None).keep(k,m,Kc,Vc,ids_k,base);keepP,_=OM.PlaceboOracleUV(chunks,8,lambda *a:None).keep(k,m,Kc,Vc,ids_k,base);keepBand=CM.band_keep(np.ones(WIN+CH,bool),256)
            lp={};pref={};lp['NC'],pref['NC']=lp_of(base);lp['A'],pref['A']=lp_of(keepA);lp['A_P'],pref['A_P']=lp_of(keepP)
            with rt.mesh:pref['BAND']=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,q0,jnp.asarray(keepBand))
            res['nll']={cn:(-lp[cn][F]).tolist() for cn in ('NC','A','A_P')};res['n_hidden']=dict(cached_oracle=int(R.hidden_from_keep(base,keepA,k).__len__()),placebo=int(R.hidden_from_keep(base,keepP,k).__len__()))
            items=STORE['items'];raw=STORE['raw'];tail=STORE['tail'];rd=STORE['rd'];rows=[r for f in fa for r in raw[f*16:(f+1)*16]]
            with rt.mesh:P=rd.packed(rows,tail)
            st_all=s3.clone_state(rt.state_all)
            def step(prefx,keepx):
                with rt.mesh:newp,gn,coef,ce=L.write_step(m,st_all,st_w,c,prefx,jnp.asarray(keepx))
                return newp,float(gn),float(coef),float(ce)
            upd={'NATS':step(pref['NC'],base),'NATMASK':step(pref['BAND'],keepBand),'CACHE_A':step(pref['A'],keepA),'CACHE_AP':step(pref['A_P'],keepP)}
            res['isolation']=dict(gnorm_here=upd['NATS'][1],coef_here=upd['NATS'][2],ce_here=upd['NATS'][3]);res['steps']={an:dict(gnorm=v[1],coef=v[2],ce=v[3]) for an,v in upd.items()}
            M={};Ms={};g=[items[f]['gold_idx'] for f in fa]
            for an,(newp,gn,coef,ce) in upd.items():
                with rt.mesh:S=rd.score(P,L.set_inner(m,newp),len(fa),16)
                M[an]=MG.margin_primary(S,g).tolist();Ms[an]=MG.margin_secondary(S,g).tolist()
            res['margin_primary']=M;res['margin_secondary']=Ms;res['facts']=fa;res['sec']=time.monotonic()-t0;STORE['results'][k]=res
            log(f'  1280-cache branch chunk {k}: n_F {len(F)}, E_cache(mean A-NC) {np.mean(res["nll"]["A"])-np.mean(res["nll"]["NC"]):.4f}, {res["sec"]:.0f}s');return {}
        fn.wants_ctx=True;return fn
    return factory
def build_seed_outputs(seed,out,cfg,log_entries):
    res=STORE['results'];ks=sorted(res);out=pathlib.Path(out)
    iso=[(abs(res[k]['isolation']['gnorm_here']-log_entries[k]['gnorm'])<=1e-6*max(1.0,abs(log_entries[k]['gnorm'])) and abs(res[k]['isolation']['coef_here']-log_entries[k]['coef'])<=1e-6) for k in ks]
    json.dump(dict(seed=seed,branches=ks,isolation_ok=bool(all(iso)),results=res),open(out/f'results_cache_s{seed}.json','w'),default=float);return 'results_cache_s%d.json'%seed
def selftest():
    assert callable(make_factory) and R.CH==1024 and R.NW==8
    f=make_factory(dict(nch=64,branches=[10,12],seed=101));assert callable(f)
    print('e57_1280c_cache_runner selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    R.make_factory=make_factory;R.build_seed_outputs=build_seed_outputs;R.main()
if __name__=='__main__':main()
