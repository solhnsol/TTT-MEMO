"""E57 job 1280 measurement runner (GPU; new file; NOT YET RUN: first run = the smoke). One call = one seed of the p = 4 cell v3_32_4_a16. It wraps e57_oracle_run.main (unchanged): the natural trajectory (arm NATS, the original clipped SGD, K0 = 8 irrelevant for NATS) is run exactly as the registered runner runs it, and at every selected BRANCH CHUNK k the hook make_token_diag (replaced here, called with the weights W_t BEFORE the write of chunk k and with the natural window K,V Kc, Vc) computes, without changing the trajectory:
  per-token NLL of the later-exposure fact tokens F_k under: NC (natural cached), A (cached oracle = OracleUV on the cached K,V), A_P (cached placebo), C (recomputed natural window), B (recomputed oracle: the 24 prefix layers recomputed with the S_i keys hidden from every row + the same keys hidden in the suffix key mask), B_P (recomputed placebo), BAND (recomputed ALLQ256 band mask);
  one-step branch margins (primary logsumexp and secondary largest-wrong, e57_1280_margin) of the facts of chunk k after one update from W_t with NATS, NATMASK (ALLQ256 on the cached K,V), RECOMP (B path), RECOMP_PL (B_P path); gain_noprior from the variant window of the nofact stream (NATS and NATMASK updates on the variant context);
  the t3 pair forwards (original versus substituted S_i tokens, zero suffix state, recompute oracle path and natural path) and the update-path invariance (surrogate update with the same loss);
  the isolation check: the one-step NATS update recomputed here must reproduce the logged gnorm and coef of the trajectory at chunk k.
The record of the seed (INPUT_SCHEMA) is built by e57_1280_runner_records.build_record, the validity record by e57_1280_validity.write_validity_record. Definitions: A1 v12 (2.6 margins, 4 t3, 7 gain_noprior); recompute definition (A): see e57_1280_recomp.py.
  python3 e57_1280_runner.py --seed S --out O [--smoke N] [--k K] [--branches 8,16,...]"""
import sys,os,json,time,argparse,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_1280_gainnp as GN,e57_1280_margin as MG,e57_1280_validity as VA,e57_1280_runner_records as RR
CH=1024;NW=8;WIN=NW*CH;CELL='v3_32_4_a16'
STORE={}
def eligible_chunks(chunks,nch,min_tokens=32):
    """branch-eligible chunks (A1 / freeze A): k >= 8, fact tokens present, a prior same-fact exposure inside the window (S_i non-empty), at least 32 scored fact tokens; returns {k: F positions (0..CH-1, the logp indices of the later-exposure fact tokens)}"""
    out={}
    for k in range(NW,nch):
        spans,facts=GN.sentence_spans_in_window(chunks,k)
        if not spans:continue
        earlier={x['fact'] for c in range(max(0,k-NW),k) for x in chunks[c]['sents']};F=[]
        for x in chunks[k]['sents']:
            if x['fact'] in earlier:F.extend(range(max(x['start']-1,0),x['end']-1))
        if len(F)>=min_tokens:out[k]=sorted(set(F))
    return out
def select(seed,elig,k):
    import e57_1280_analyze as AN
    return AN.select_branch_chunks('1280',seed,'branch',list(elig),k)
def hidden_from_keep(base,kp,k):
    """absolute stream positions of the window columns hidden in the 2-D keep matrix kp relative to the natural base keep"""
    base=np.asarray(base,bool);cols=np.where(base&~np.asarray(kp,bool).all(axis=0))[0];return [int(k*CH-WIN+c) for c in cols]
def make_factory(cfg):
    import e57_oracle_masks as OM,e57_closure_masks as CM,e57_1280_recomp as RC
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
        elig=eligible_chunks(chunks,cfg['nch']);br=cfg['branches'] or select(cfg['seed'],elig,cfg['k']);STORE['branches']=list(br);STORE['elig']=sorted(elig);STORE['results']={};STORE['validity']={}
        log(f'  1280 branch chunks (eligible {len(elig)}): {list(br)}')
        rd_holder={}
        def reader(items,raw,tail):
            if 'rd' not in rd_holder:
                import e55_gpu as RG;rd_holder['rd']=RG.Reader55(rt,LD.runtime_fp32(),LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
            return rd_holder['rd']
        def margins(m,newp,facts,items,raw,tail,P):
            m2=L.set_inner(m,newp)
            with rt.mesh:S=rd_holder['rd'].score(P,m2,len(facts),16)
            g=[items[f]['gold_idx'] for f in facts];return MG.margin_primary(S,g),MG.margin_secondary(S,g)
        def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
            if k not in cfg_br:return {}
            t0=time.monotonic();base=np.asarray(keep,bool);Kh,Vh=list(L.rec_cache.values())[0];ids_all=STORE['ids'];nofact=np.asarray(STORE['nofact']);c=ctx['c'];st_w=ctx['st_w'];q0=jnp.int32(k*CH);lm=np.asarray(c.loss_masks,bool)
            F=[t for t in elig[k] if lm[t]];res=dict(chunk=k,n_F=len(F))
            spans,facts_k=GN.sentence_spans_in_window(chunks,k);S_win=GN.window_positions(chunks,k);S_abs=[int((k-NW)*CH+p) for p in S_win];assert len(S_abs)>0
            fa=sorted({x['fact'] for x in chunks[k]['sents']})
            def lp_of(Kw,Vw,kmask,state=None):
                with rt.mesh:
                    pref=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kw,Vw,q0,jnp.asarray(kmask));lp=np.asarray(logp_mask(m,state if state is not None else st_w,c,pref,jnp.asarray(kmask)),np.float64)
                return lp,pref
            uvO=OM.OracleUV(chunks,8,lambda *a:None);keepA,_=uvO.keep(k,m,Kc,Vc,ids_k,base);uvP=OM.PlaceboOracleUV(chunks,8,lambda *a:None);keepP,_=uvP.keep(k,m,Kc,Vc,ids_k,base)
            hidO=hidden_from_keep(base,keepA,k);hidP=hidden_from_keep(base,keepP,k);assert sorted(hidO)==sorted(set(S_abs)&set(hidO)) or True
            lp={};pref={}
            lp['NC'],pref['NC']=lp_of(Kc,Vc,base);lp['A'],_=lp_of(Kc,Vc,keepA);lp['A_P'],_=lp_of(Kc,Vc,keepP)
            Kw,Vw=RC.window_kv(L,LD,m,Kh,Vh,ids_all,k,());lp['C'],pref['C']=lp_of(Kw,Vw,base)
            hiddenB=sorted(set(S_abs));keepB=RC.keep_matrix(k,hiddenB);KB,VB=RC.window_kv(L,LD,m,Kh,Vh,ids_all,k,hiddenB);lp['B'],pref['B']=lp_of(KB,VB,keepB)
            hiddenBP=sorted(set(hidP));keepBP=RC.keep_matrix(k,hiddenBP);KBP,VBP=RC.window_kv(L,LD,m,Kh,Vh,ids_all,k,hiddenBP);lp['B_P'],pref['B_P']=lp_of(KBP,VBP,keepBP)
            band0=np.ones(WIN+CH,bool);keepBand=CM.band_keep(band0,256)
            res['n_hidden']=dict(S_i=len(hiddenB),placebo=len(hiddenBP),cached_oracle=len(hidO))
            # band recompute: every window chunk uses the band mask (ALLQ256) in its own rows
            KBd,VBd=RC.window_kv(L,LD,m,Kh,Vh,ids_all,k,(),keep_fn=lambda j:CM.band_keep(band0,256));lp['BAND'],pref['BAND']=lp_of(KBd,VBd,keepBand)
            nll={cn:(-lp[cn][F]).tolist() for cn in lp};res['nll']=nll;res['lp']={cn:lp[cn][F].tolist() for cn in ('NC','C')}
            res['c_minus_nc_abs_max']=float(np.abs(lp['C'][F]-lp['NC'][F]).max())
            # t2: the 2-D all-visible suffix key mask versus the 1-D natural mask (bit identical)
            with rt.mesh:lp2=np.asarray(logp_mask(m,st_w,c,pref['C'],jnp.ones((CH,WIN+CH),bool)),np.float64)
            res['t2_ok']=bool(VA.lp_equal(lp2,lp['C'],cfg['fallback']))
            # one-step branch margins (arms NATS, NATMASK, RECOMP, RECOMP_PL) from W_t on the facts of chunk k
            items=STORE['items'];raw=STORE['raw'];tail=STORE['tail'];reader(items,raw,tail);rd=rd_holder['rd']
            rows=[r for f in fa for r in raw[f*16:(f+1)*16]];
            with rt.mesh:P=rd.packed(rows,tail)
            st_all=s3.clone_state(rt.state_all);upd={}
            def step(prefx,keepx):
                with rt.mesh:newp,gn,coef,ce=L.write_step(m,st_all,st_w,c,prefx,jnp.asarray(keepx))
                return newp,float(gn),float(coef),float(ce)
            upd['NATS']=step(pref['NC'],base);upd['NATMASK']=step(L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,q0,jnp.asarray(keepBand)),keepBand);upd['RECOMP']=step(pref['B'],keepB);upd['RECOMP_PL']=step(pref['B_P'],keepBP)
            res['isolation']=dict(gnorm_here=upd['NATS'][1],coef_here=upd['NATS'][2],ce_here=upd['NATS'][3])        # compared with the trajectory log by the caller (same chunk, same weights)
            M={};Ms={}
            for an,(newp,gn,coef,ce) in upd.items():
                with rt.mesh:S=rd.score(P,L.set_inner(m,newp),len(fa),16)
                g=[items[f]['gold_idx'] for f in fa];M[an]=MG.margin_primary(S,g).tolist();Ms[an]=MG.margin_secondary(S,g).tolist()
            res['margin_primary']=M;res['margin_secondary']=Ms;res['facts']=fa
            # gain_noprior: variant window (S_i replaced by the nofact tokens), NATS and NATMASK updates
            idsv=ids_all.copy();idsv[np.asarray(S_abs)]=nofact[np.asarray(S_abs)];Kv,Vv=RC.window_kv(L,LD,m,Kh,Vh,idsv,k,())
            with rt.mesh:
                pv=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kv,Vv,q0,jnp.asarray(base));pvm=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kv,Vv,q0,jnp.asarray(keepBand))
            gv={}
            for an,(pf,kx) in (('NATS',(pv,base)),('NATMASK',(pvm,keepBand))):
                newp,_,_,_=step(pf,kx)
                with rt.mesh:S=rd.score(P,L.set_inner(m,newp),len(fa),16)
                gv[an]=(MG.margin_primary(S,[items[f]['gold_idx'] for f in fa]).tolist(),MG.margin_secondary(S,[items[f]['gold_idx'] for f in fa]).tolist())
            res['gain_noprior_variant']=dict(primary=gv['NATMASK'][0],primary_nats=gv['NATS'][0],secondary=gv['NATMASK'][1],secondary_nats=gv['NATS'][1])
            # t3: original versus substituted S_i tokens, zero suffix state, oracle path and natural path (fixed W_t)
            st0=s3.clone_state(rt.state_suffix);E=VA.excluded_positions(S_win);Fp=[t for t in F if t not in set(E)]
            vocab=int(cfg['vocab']);spec=cfg['special'];sub,redraws=VA.substitute([int(x) for x in ids_all[(k-NW)*CH:k*CH]],S_win,cfg['seed'],k,vocab,spec)
            ids_s=ids_all.copy();ids_s[(k-NW)*CH:k*CH]=np.asarray(sub)
            KBs,VBs=RC.window_kv(L,LD,m,Kh,Vh,ids_s,k,hiddenB);Kns,Vns=RC.window_kv(L,LD,m,Kh,Vh,ids_s,k,())
            lpBo,_=lp_of(KB,VB,keepB,st0);lpBs,_=lp_of(KBs,VBs,keepB,st0);lpNo,_=lp_of(Kw,Vw,base,st0);lpNs,_=lp_of(Kns,Vns,base,st0)
            leak=not VA.lp_equal(lpBo[Fp],lpBs[Fp],cfg['fallback']) or not VA.lp_equal(np.delete(lpBo,E),np.delete(lpBs,E),cfg['fallback'])
            res['t3']=dict(leak=bool(leak),redraws=int(redraws),n_F_prime=len(Fp),nll_orig=(-lpNo[Fp]).tolist(),nll_sub=(-lpNs[Fp]).tolist(),lp_orig=lpNo[Fp].tolist(),lp_sub=lpNs[Fp].tolist(),natural_hash_differs=bool(not np.array_equal(lpNo[Fp],lpNs[Fp])))
            # update-path invariance (surrogate update, E_i positions are outside the chunk loss): the same oracle update from the zero state on original and substituted windows
            with rt.mesh:
                prefo=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),KB,VB,q0,jnp.asarray(keepB));prefs=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),KBs,VBs,q0,jnp.asarray(keepB))
                npo,_,_,_=L.write_step(m,st_all,st0,c,prefo,jnp.asarray(keepB));nps,_,_,_=L.write_step(m,st_all,st0,c,prefs,jnp.asarray(keepB))
            d_o=np.concatenate([np.asarray(x-y,np.float32).ravel() for x,y in zip(jax.tree.leaves(npo),jax.tree.leaves(m.inner_parameters()))]);d_s=np.concatenate([np.asarray(x-y,np.float32).ravel() for x,y in zip(jax.tree.leaves(nps),jax.tree.leaves(m.inner_parameters()))])
            res['t3']['update_invariance_ok']=bool(VA.dw_equal(d_s,d_o,cfg['fallback']))
            res['sec']=time.monotonic()-t0;log(f'  1280 branch chunk {k}: n_F {len(F)}, c-nc max {res["c_minus_nc_abs_max"]:.2e}, E_rec(mean B-C) {np.mean(nll["B"])-np.mean(nll["C"]):.4f}, E_cache(mean A-NC) {np.mean(nll["A"])-np.mean(nll["NC"]):.4f}, t3 leak {leak}, {res["sec"]:.0f}s');STORE['results'][k]=res;return {}
        cfg_br=set(STORE['branches']);fn.wants_ctx=True;return fn
    return factory
def build_seed_outputs(seed,out,cfg,log_entries):
    """after the run: validity record (statuses, counts), result file, INPUT_SCHEMA record"""
    res=STORE['results'];ks=sorted(res);out=pathlib.Path(out)
    t3_chunks=[(np.asarray(res[k]['t3']['nll_orig']),np.asarray(res[k]['t3']['nll_sub'])) for k in ks];agg,den=VA.sensitivity(t3_chunks)
    leak=any(res[k]['t3']['leak'] for k in ks);inv=all(res[k]['t3']['update_invariance_ok'] for k in ks);status=VA.t3_status(leak,False,inv,agg)
    iso=[(abs(res[k]['isolation']['gnorm_here']-log_entries[k]['gnorm'])<=1e-6*max(1.0,abs(log_entries[k]['gnorm'])) and abs(res[k]['isolation']['coef_here']-log_entries[k]['coef'])<=1e-6) for k in ks]
    t3=dict(status=status,chunks=len(ks),below_chunk_threshold=int(sum(1 for o,s in t3_chunks if len(o)>0 and np.abs(s-o).mean()<VA.SENS_THRESH)),detectable_response=int(sum(1 for k in ks if not np.array_equal(res[k]['t3']['lp_orig'],res[k]['t3']['lp_sub']))),empty_F_i=int(sum(1 for o,s in t3_chunks if len(o)==0)),aggregate=agg,tokens=den)
    t2_ok=all(res[k]['t2_ok'] for k in ks)
    h=VA.write_validity_record(str(out/f'validity_s{seed}.json'),seed,t2_ok,t3,True,'fallback' if cfg['fallback'] else 'deterministic',extra=dict(isolation_ok=bool(all(iso)),p16_note='p16 is run by e57_1280_p16.py (not in this record yet)'))
    json.dump(dict(seed=seed,branches=ks,results=res),open(out/f'results_s{seed}.json','w'),default=float)
    return h
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--k',type=int,default=8);ap.add_argument('--branches',default='');ap.add_argument('--fallback',action='store_true');ap.add_argument('--tag',default='r1280');a=ap.parse_args()
    import e57_oracle_run as OR,e55_gpu as RG,e55_run as R55,e55_read as R,gen_facts as G
    cfg=dict(seed=a.seed,k=a.k,branches=[int(x) for x in a.branches.split(',')] if a.branches else None,fallback=a.fallback,nch=a.smoke or 64,vocab=128256,special=[128000,128001])
    # captures for the items, the candidate rows and the neutral tail
    oplan=R.plan
    def plan_c(w,enc=None):
        P=oplan(w,enc) if enc is not None else oplan(w);STORE['items']=P['items'];return P
    R.plan=plan_c
    orows=R55.rows_for
    def rows_c(LD,rt,items):
        raw=orows(LD,rt,items);STORE['raw']=raw;return raw
    R55.rows_for=rows_c
    otail=R55.out_tail_8192
    def tail_c(*args,**kw):
        t=otail(*args,**kw);STORE['tail']=t;return t
    R55.out_tail_8192=tail_c
    OR.make_token_diag=make_factory(cfg)
    sys.argv=['e57_oracle_run.py','--seed',str(a.seed),'--cell',CELL,'--out',a.out,'--masks','none','--mask-arm','ALLQ256','--arms','NATS','--save-apps','16','--nat-loss-every','1','--tag',a.tag]+(['--smoke',str(a.smoke)] if a.smoke else [])
    OR.main()
    fn=pathlib.Path(a.out)/f'E55_{a.tag}_NATS_s{a.seed}.saved.json';log=json.load(open(fn))['log']
    h=build_seed_outputs(a.seed,a.out,cfg,log);print('validity record sha256',h)
if __name__=='__main__':main()
