"""E59 write-signal decomposition, E2E 3B (GPU; new file; thread-server 11/18; EXPLORATORY; nothing existing edited; uses e55_core / e55_iso / e57_natmask_run engine / e57_qband_run / e55_run helpers as libraries). Arms (E2E v3_16_1 worlds, seeds 71 / 72; helpers of e59_srv_core.py):
  NAT     the existing carried-KV NAT write (ENGINE SGDSEL with no mask), ORIGINAL rule (SGD lr 1, clip 1, no rescaling); its applied per-chunk update norms are the REFERENCE norms of the seed (saved in its json: log[k].update_norm).
  NATF    NAT context, loss ONLY on the fact tokens (mean over the fact target tokens); direction-only rescale to the reference norm
  CHK     the chunk alone: no earlier-chunk key is visible (window keys hidden for every row, current chunk causal), loss over all targets; rescaled to the reference norm
  CHKF    CHK context, loss on the fact tokens only; rescaled
  FACT    the 16 fact sentences of chunk r (sep + sentence ids, stream order) concatenated into ONE row (mutual causal attention, no general text), one step per round; rescaled
  ISO     the 16 fact sentences as 16 ISOLATED rows (sep + sentence ids) packed in one 1024-token chunk, segment mask (no cross-sentence attention), ONE step per round, loss = one mean over all target tokens; rescaled
  NATREF  NAT with the update rescaled to its own reference norms (reproduction / ratio-1 check), ISOORIG ISO with the original rule (no rescaling)
Round r = chunk r = exposure r; 128 steps; reads (out-of-window neutral 8192 tail, 16 candidates; in-window for the stream arms) at exposures 16/32/64/128; canary after the run; fp32 inner weights. Context policy of every stream arm = that of the existing NAT (e55_core.run_stream: the 24 prefix layers are computed once with the base weights, the recorded K,V of the window for the prefix layers, the suffix-layer SWA state is carried from the chunk-by-chunk pass and is a constant in the backward pass; only the current chunk is differentiated); FACT / ISO use the isolated-row path of e55_iso (prefix_iso with base weights, empty suffix state). Per-step records (arm json 'steps'): k, raw norm, reference norm, ratio raw / reference (+ 'noise_direction' flag when the raw norm < 1e-3 x the reference), applied norm, net displacement ||W_k - W_0||, running path length, per-leaf squared-norm occupancy of the update, fact-token mean loss (nll of the chunk at the weights BEFORE the update), all-token mean loss, fact hash (sha256 of the fact token ids of the chunk), chunk hash; the fixed coordinate sketch of every update (262144 random coordinates, seed 0, same in every arm) goes to <arm>_s<seed>.sketch.npy [steps, 262144] for cosines between arms.
Divergence flag (reported, nothing excluded here): non-finite ce / gnorm / update norm, last-8-chunk mean CE > 4.0, final canary change > +1.0.
  python3 e59_srv_run.py --seed 71 --out O --arms NAT,NATF,CHK,CHKF,FACT,ISO [--ref-dir D] [--tag e59] [--smoke N] [--rerun] | --selftest"""
import sys,os,json,time,gc,hashlib,argparse,pathlib,math
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e59_srv_core as K
CH=1024;NS=262144
ARMS=('NAT','NATF','CHK','CHKF','FACT','ISO','NATREF','ISOORIG','BAND1024','MASK1024','SAME','SAME_U1','SAME_U2','SAME_U1U2','RANDTOK','BGSENT')
STREAM={'NAT':(False,False,False),'BAND1024':(False,False,False),'MASK1024':(False,False,False),'SAME':(False,False,False),'SAME_U1':(False,False,False),'SAME_U2':(False,False,False),'SAME_U1U2':(False,False,False),'RANDTOK':(False,False,False),'BGSENT':(False,False,False),'NATREF':(False,False,True),'NATF':(True,False,True),'CHK':(False,True,True),'CHKF':(True,True,True)}   # (fact-only loss, chunk alone, rescale to reference)
class ChunkOnlyUV:
    """uv object of e55_core (E.UVARMS path): every window key hidden (1-D keep, index j = absolute position k*CH-WIN+j), the current chunk causal as always"""
    def __init__(self,WIN,log):self.WIN=WIN;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        kp=np.asarray(keep_all,bool).copy();kp[:self.WIN]=False;return kp,{'uv_arm':'CHK','uv_hidden_tokens':int(self.WIN),'q_banded':0}
def tools(L,rt):
    import jax,jax.numpy as jnp,equinox as eqx
    f32=jnp.float32;inner=rt.base.inner_parameters();leaves,td=jax.tree_util.tree_flatten(inner);paths=[jax.tree_util.keystr(p) for p,_ in jax.tree_util.tree_flatten_with_path(inner)[0]]
    sizes=[int(np.prod(x.shape)) for x in leaves];tot=sum(sizes);rng=np.random.default_rng(0);idx=[]
    for s in sizes:c=max(1,int(round(NS*s/tot)));idx.append(np.sort(rng.choice(s,size=min(c,s),replace=False)).astype(np.int32))
    idx_dev=[jnp.asarray(i) for i in idx];nsk=sum(len(i) for i in idx)
    @eqx.filter_jit
    def stats(newp,old):
        nl=jax.tree_util.tree_leaves(newp);ol=jax.tree_util.tree_leaves(old);d=[a.astype(f32)-b.astype(f32) for a,b in zip(nl,ol)]
        sq=jnp.stack([jnp.sum(x*x) for x in d]);sk=jnp.concatenate([x.reshape(-1)[i] for x,i in zip(d,idx_dev)]);return sq,sk
    @eqx.filter_jit
    def disp(newp,base):
        return jnp.sqrt(sum(jnp.sum((a.astype(f32)-b.astype(f32))**2) for a,b in zip(jax.tree_util.tree_leaves(newp),jax.tree_util.tree_leaves(base))))
    return dict(stats=stats,disp=disp,paths=paths,nsk=nsk)
def finalize(L,T,m,newp,ref,st,k):
    """direction-only rescale (ref not None), record the step; -> new inner params"""
    old=m.inner_parameters();raw=L.norm_between(newp,m)
    if not np.isfinite(raw) or raw==0:raise FloatingPointError(f'zero / non-finite raw update at step {k}')
    flag=None
    if ref is not None:
        newp=L.rescale(newp,m,ref/raw)
        if raw<1e-3*ref:flag='noise_direction'
    un=L.norm_between(newp,m)
    if ref is not None:assert abs(un-ref)<=1e-4*ref+1e-9,('rescaled norm differs from the reference (fp32 tolerance 1e-4)',un,ref)
    sq,sk=T['stats'](newp,old);sq=np.asarray(sq,np.float64);dis=float(T['disp'](newp,st['W0']))
    st['sketch'].append(np.asarray(sk,np.float32));st['path']+=un
    st['steps'].append(dict(k=k,raw_norm=float(raw),ref_norm=None if ref is None else float(ref),ratio=None if ref is None else float(raw/ref),flag=flag,applied_norm=float(un),norm_rel_err=None if ref is None else float(abs(un-ref)/ref),net_displacement=dis,path_length=float(st['path']),occupancy={p:float(v) for p,v in zip(T['paths'],sq/max(sq.sum(),1e-300))}))
    return newp
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    import e55_run as R55,e55_read as R,e55_gpu as RG,e55_iso as ISO,gen_facts as G,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;cell='v3_16_1'
    if a.u12:import e59_u12 as U12;U12.install();cell=U12.CELL          # 32-fact world, 16 facts outside the read candidates (U1 / U2 hiding experiment)
    else:
        try:import e59_u12 as U12
        except Exception:U12=None
    if a.fq_worlds:import fq_run as FQR;FQR.install(a.fq_worlds);cell=f'fq{seed}'          # FictionalQA worlds (labels 211..218): same patches as fq_run.py (register_cell / build / build_items)
    ci=OR.register_cell(cell);ro=list(ci['read_apps']);ri=[ro[-1]];assert ro==[16,32,64,128],ro
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);nch=a.smoke or w['n_chunks'];ids,sha=S['A'];assert sha==G.stream_sha(G.realize(w))
    P=R.plan(w,enc);items=P['items'];tok=rt.tok;raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w))
    R_s=RG.Reads(rd,raw,len(items),G.NCAND,P['read'],out_tail,rt.base,ro,ri,log=lg);R_i=RG.Reads(rd,raw,len(items),G.NCAND,P['read'],out_tail,rt.base,ro,(),log=lg);lg(f'seed {seed}: W0 read done')
    RR_s=RR_i=items_r=None
    if a.reverse:        # reverse-question readout (e59_rev.py): same 16-candidate span margin, place -> person; weights only read
        import e59_rev as RV;items_r=RV.rev_items(w);raw_r=R55.rows_for(LD,rt,items_r);RG.check_rows(raw_r,len(items_r),G.NCAND);P_r=RV.rev_plan(P)['read']
        RR_s=RG.Reads(rd,raw_r,len(items_r),G.NCAND,P_r,out_tail,rt.base,ro,ri,log=lg);RR_i=RG.Reads(rd,raw_r,len(items_r),G.NCAND,P_r,out_tail,rt.base,ro,(),log=lg);lg(f'seed {seed}: reverse W0 read done')
    sep=list(w['sep']);W=int(rt.cfg.model.sliding_window_size);ref_dir=pathlib.Path(a.ref_dir or a.out)
    fmasks=[K.target_mask(K.fact_spans(w['chunks'][k]['sents'])) for k in range(nch)];fhash=[K.ids_hash(K.fact_ids(w['chunks'][k])) for k in range(nch)]
    for ar in a.arms.split(','):
        assert ar in ARMS,ar;fn=out/f'E59_{a.tag}_{ar}_s{seed}.json'
        if fn.exists() and not a.rerun:lg('exists, reused:',fn);continue
        ref=None
        if ar not in ('NAT','ISOORIG','BAND1024','MASK1024'):
            if a.ref_nats:rf=pathlib.Path(a.ref_nats.replace('{seed}',str(seed)));assert rf.exists(),('reference norms (the NATS run of this seed, E55 saved.json) missing',rf);ref=[e['update_norm'] for e in json.load(open(rf))['log']][:nch]       # FictionalQA: the applied per-chunk norms of the existing NATS run (job 1313)
            else:rf=ref_dir/f'E59_{a.tag}_NAT_s{seed}.json';assert rf.exists(),('reference norms (NAT of this seed) missing',rf);ref=[e['applied_norm'] for e in json.load(open(rf))['steps']][:nch]
            assert len(ref)>=nch
        L=ISO.Iso55(rt,save_dir=out/'states'/f'E59_{a.tag}_{ar}_s{seed}',canary=can);T=tools(L,rt);st=dict(steps=[],sketch=[],path=0.0,W0=rt.base.inner_parameters());got={};gotr={};t0=time.monotonic();lg(f'== seed {seed} {ar}: {nch} rounds, reference norms {"original rule" if ref is None else "NAT applied norms"}')
        if ar in STREAM:
            fonly,chunk_alone,resc=STREAM[ar];uv=ChunkOnlyUV(W,lg) if chunk_alone else (QR.SelMaskQ('ALLQ1024',{'band_ALLQ1024':np.asarray(1024),'all_ALLQ1024':np.asarray(1)},lg) if ar in ('BAND1024','MASK1024') else (U12.uv_for(ar,w['chunks'],lg) if ar in U12.ARMS else (__import__('e59_match').uv_for(ar,w['chunks'],seed,lg) if ar in ('RANDTOK','BGSENT') else QR.SelMaskQ('NATQ',{},lg))));cnt=[0];nllrec=[]     # MASK1024 (= BAND1024; name used from job 1321 on): band-1024 mask on every row, in ALL 32 layers (e55_core passes the same keep to prefix_masked and write_step), ORIGINAL write rule (no rescaling, as jobs 1297c / 1308 / 1313), K0 = 0
            @eqx.filter_jit
            def grad_fn(m,st_suf,c,pref_,kmask,wt):
                def total(mm):
                    s3.S3SUF['kmask']=kmask
                    try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
                    finally:s3.S3SUF.clear()
                    lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
                (_l,nll),g=eqx.filter_value_and_grad(total,has_aux=True)(m);return g.inner_parameters(),nll
            def ws(m,st_all,st_suf,c,pref,kmask,fonly=fonly,resc=resc):
                k=cnt[0];cnt[0]+=1;valid=np.asarray(c.loss_masks,np.float32);fm=fmasks[k].astype(np.float32)
                if fonly:assert set(np.flatnonzero(fm)).issubset(set(np.flatnonzero(valid))),'fact positions outside the loss mask';wt=fm/fm.sum()
                else:wt=valid/valid.sum()
                g,nll=grad_fn(m,st_suf,c,pref,kmask,jnp.asarray(wt));nll=np.asarray(nll,np.float64);newp,gn,coef=L.sgd_apply(m,st_all,g);del g
                newp=finalize(L,T,m,newp,ref[k] if (resc and ref is not None) else None,st,k);s_=st['steps'][-1];s_.update(fact_ce=float((nll*fm).sum()/fm.sum()),all_ce=float((nll*valid).sum()/valid.sum()),n_loss_tokens=int(wt.astype(bool).sum()),fact_hash=fhash[k],chunk_hash=hashlib.sha256(np.asarray(ids[k*CH:(k+1)*CH],np.int64).tobytes()).hexdigest()[:16]);ce=s_['all_ce']
                return newp,gn,coef,jnp.asarray(ce,jnp.float32)
            L.write_step=ws
            def partial(k,m,got=got,gotr=gotr):
                h0=LD.reuse._hash_pytree(m.inner_parameters())[0];x=R_s.at(k+1,m);assert h0==LD.reuse._hash_pytree(m.inner_parameters())[0],'weights changed during the read';got[k+1]=x;gotr[k+1]=RR_s.at(k+1,m) if RR_s is not None else None;return {c_:'ok' for c_ in x}
            r=L.run_stream(NM.ENGINE,1,ids,read_chunks=[a_-1 for a_ in ro],canary_chunks=[a_-1 for a_ in ro],nch=nch,tag=f'{ar}_s{seed}',log=lg,read_hook=partial,hook_chunks=[a_-1 for a_ in ro],chunks_meta=w['chunks'],uv=uv);r.pop('model',None);R_=R_s;RRx=RR_s
        else:
            m=rt.base;st_all=s3.clone_state(rt.state_all);r={'log':[],'reads':{}};eos=rt.eos;cw0=can(m) if can else None;packing=LD.packing
            for k in range(nch):
                ch=w['chunks'][k]
                if ar in ('ISO','ISOORIG'):ids_,seg,loc,mk=K.isolated_mask(K.iso_rows(ch,sep),CH)
                else:
                    rid,rm=K.fact_row(ch,sep);n=len(rid);assert n<=CH,n;ids_=np.asarray(rid+[rid[-1]]*(CH-n),np.int32);seg=np.asarray([0]*n+[-1]*(CH-n),np.int32);loc=np.asarray(list(range(n))+list(range(CH-n)),np.int32);mk=np.zeros(CH,bool);mk[:n]=rm
                sp0=K.fact_spans(ch['sents'])[0][0]==0                    # FictionalQA worlds: a sentence at token 0 of the chunk has no in-chunk predecessor, NAT/target_mask does not train on its first token; drop it here too (token parity with NAT)
                if sp0:mk[len(sep)-1]=False
                tg=np.concatenate([ids_[1:],[eos]]).astype(np.int32);assert set(np.flatnonzero(mk))>=set() and int(mk.sum())==int(fmasks[k].sum()),'target token count differs from the fact tokens'
                pref=L.prefix_iso(rt.base,jnp.asarray(ids_),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)));g,ce=L.iso_grad(m,jnp.asarray(ids_),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc))
                newp,gn,coef=L.sgd_apply(m,st_all,g);del g;newp=finalize(L,T,m,newp,None if ar=='ISOORIG' else ref[k],st,k)
                tgt=[int(tg[p]) for p in np.flatnonzero(mk)];exp_=[t for s in K.fact_ids(ch) for t in s][(1 if sp0 else 0):];assert tgt==exp_,'target token ids differ from the fact token ids'
                s_=st['steps'][-1];s_.update(fact_ce=float(ce),all_ce=float(ce),n_loss_tokens=int(mk.sum()),fact_hash=fhash[k],input_hash=K.ids_hash([ids_[:int((seg>=0).sum())].tolist()]),clip_coef=float(coef),grad_norm=float(gn))
                un=L.norm_between(newp,m);m=L.set_inner(m,newp);r['log'].append(dict(k=k,ce=float(ce),gnorm=float(gn),coef=float(coef),update_norm=un))
                if k+1 in ro:
                    got[k+1]=R_i.at(k+1,m)
                    if RR_i is not None:gotr[k+1]=RR_i.at(k+1,m)
            r['canary_w0']=cw0;r['canary_end']=can(m) if can else None;R_=R_i;RRx=RR_i
        lgs=r['log'];nf=all(math.isfinite(v) for e in lgs for kk,v in e.items() if kk in ('ce','gnorm','update_norm') and isinstance(v,(int,float)));last8=float(np.mean([e['ce'] for e in lgs[-8:]]));cw0=r.get('canary_w0');cend=r.get('canary_end')
        if cend is None and r.get('reads'):
            lk=max(r['reads']);e=r['reads'][lk];cend=e.get('canary')
        can_ch=None if (cw0 is None or cend is None) else (cend['_gen_nll']-cw0['_gen_nll'] if isinstance(cend,dict) else cend-cw0)
        div=bool((not nf) or last8>4.0 or (can_ch is not None and can_ch>1.0))
        rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':cell,'seed':seed,'k':w['k'],'n':w['n'],'p':1,'bank':w['bank_kind'],'tag':a.tag,'complete':True,'place_len':w['place_len']})[ar];json.dump(rec,open(out/f'E59_{a.tag}_{ar}_s{seed}.read.json','w'))
        if a.reverse:
            recr=RG.make_records({ar:gotr},RRx.W0,G.item_meta(items_r),P_r,RRx.idx,{'cell':cell,'seed':seed,'k':w['k'],'n':w['n'],'p':1,'bank':w['bank_kind'],'tag':a.tag,'complete':True,'direction':'reverse (place -> person)','place_len':w['place_len']})[ar];json.dump(recr,open(out/f'E59_{a.tag}_{ar}_s{seed}.rev.read.json','w'))
        np.save(out/f'E59_{a.tag}_{ar}_s{seed}.sketch.npy',np.stack(st['sketch']).astype(np.float32))
        r.update(arm=ar,seed=seed,steps=st['steps'],stream_sha=sha,sec=time.monotonic()-t0,reference='original rule' if ref is None else 'NAT applied norms',diverged=div,last8_ce=last8,finite=nf,canary_change_gen=can_ch,sketch_dim=T['nsk'],args=vars(a),code=os.popen('git rev-parse HEAD 2>/dev/null').read().strip())
        json.dump(r,open(fn,'w'),default=float);lg('wrote',fn,f'{time.monotonic()-t0:.0f}s','DIVERGED' if div else '');gc.collect();LD.jax.clear_caches()
def selftest():
    K.selftest()
    class A:pass
    import ast;ast.parse(open(__file__).read());assert set(STREAM)<=set(ARMS)
    print('e59_srv_run selftest ok (syntax; arms',ARMS,')')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--arms',default='NAT');ap.add_argument('--ref-dir',default='');ap.add_argument('--tag',default='e59');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--rerun',action='store_true');ap.add_argument('--reverse',action='store_true');ap.add_argument('--u12',action='store_true');ap.add_argument('--fq-worlds',default='');ap.add_argument('--ref-nats',default='');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    run(a)
if __name__=='__main__':main()
