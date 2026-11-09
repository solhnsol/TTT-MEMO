"""E57 synthetic world: original-NAT (plain clipped SGD) write arms with the ALLQ mask and/or a step-size multiplier (GPU, one seed per call; new file, no existing file edited; design by Hans 11/09 'step size from the original paper'). Same write run, reads, canary and saved state as e57_syn_qrun.py, but the update is the ORIGINAL inner rule (clipped SGD, lr x ilr, NO Adam, NO norm matching) instead of engine SELC1:
  NATS    no mask, multiplier 1 (reproduces the native NAT: control of the write path)      NATSx2 / NATSx4    no mask, lr x 2 / x 4
  NATMASKC / NATMASKD (Council 11/09): same number of hidden keys per row as ALLQ, random keys (C) / most recent keys (D); selected by arm name, band from --mask-arm.
  RANDMASK random-row mask (stored arm --rand-arm, default RANDQ256P50), lr x 1 (Council 11/09: control for the mask effect)
  NATMASK ALLQ mask (band from the mask file, every row from chunk K0), lr x 1               NATMASKx2 / NATMASKx4   ALLQ mask, lr x 2 / x 4
The mask arm is the UV-style SelMaskQ (key `ALLQ<band>` of the mask file); the engine is the arm SGDSEL below, registered here as a UV arm that is NOT in E.ADAM, so e55_core takes the SGD path (write_step) with the masked keep. The multiplier is applied after the clip: new = old + mult x (SGD step). update_norm in the log is measured after the multiplier; 'scale' in the log stays 1.0 and the multiplier is stored as r['lr_mult'].
  python3 e57_natmask_run.py --seed 64 --out O --masks M.npz --mask-arm ALLQ256 --arms NATS,NATMASK,NATMASKx2,... [--tag synng] [--smoke N] [--rerun] | --selftest      (XLA_FLAGS=--xla_gpu_deterministic_ops=true)"""
import sys,os,json,time,gc,re,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_qband_run as QR
ENGINE='SGDSEL'
CH=1024;WIN=8192;K0=8
class KeyMatchedQ:
    """Council 11/09 controls with the SAME number of hidden keys per row as the band arm (ALLQ<band>), chosen differently. kind 'C': random keys among the causal natural-visible keys (distance >= 1); kind 'D': the n_i MOST RECENT causal keys (distance 1..n_i), n_i = number of natural-visible keys at distance >= band (what ALLQ hides). The row's own key (distance 0) stays visible. Applied only to the update-path keep (like SelMaskQ); chunks < K0: natural. Per-row hidden counts are logged."""
    def __init__(self,kind,band,seed=0,log=print):
        assert kind in ('C','D');self.kind=kind;self.band=int(band);self.seed=seed;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<K0:return base,{'uv_arm':'KEY'+self.kind,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        d=np.arange(CH)[:,None]+WIN-np.arange(WIN+CH)[None,:];vis=np.broadcast_to(base,(CH,WIN+CH))
        hidden_allq=vis&(d>=self.band);n=hidden_allq.sum(1)                      # keys ALLQ hides per row
        cand=vis&(d>=1)
        if self.kind=='D':hide=cand&(d<=n[:,None])
        else:
            rng=np.random.default_rng((self.seed,k));sc=np.where(cand,rng.random((CH,WIN+CH),dtype=np.float32),2.0)
            rank=np.argsort(np.argsort(sc,axis=1),axis=1);hide=cand&(rank<n[:,None])
        kp=vis&~hide;nh=hide.sum(1)
        assert (nh<=n).all() and (nh>=np.minimum(n,cand.sum(1))).all() and kp[np.arange(CH),np.arange(CH)+WIN].all()
        if k%16==0:self.log(f'  KEY{self.kind} chunk {k}: hidden keys/row mean {nh.mean():.0f} (ALLQ {n.mean():.0f}), band {self.band}')
        return kp,{'uv_arm':'KEY'+self.kind,'q_banded':CH,'q_band':self.band,'hidden_keys_row_mean':float(nh.mean()),'hidden_keys_row_min':int(nh.min()),'hidden_keys_row_max':int(nh.max()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def _tq(df,q):
    from scipy import stats
    return float(stats.t.ppf(q,df))
def ci(x,level):
    """t interval of the mean of per-seed values x -> (mean, lo, hi, sd)"""
    x=np.asarray(x,float);n=len(x);mu=x.mean();sd=x.std(ddof=1) if n>1 else float('nan');h=_tq(n-1,0.5+level/2)*sd/np.sqrt(n) if n>1 else float('nan');return float(mu),float(mu-h),float(mu+h),float(sd)
def verdict_rules(nat,allq,arm):
    """Council 11/09 verdict for a control arm (C = NATMASKC or D = NATMASKD) against ALLQ (= NATMASK) and NATS, per-seed read accuracies (equal-length lists, same seeds):
    g = ALLQ - NATS (per seed); gate: the 90% CI of g must exclude 0, else 'not attributable'. diff = arm - ALLQ (per seed).
    EQUIVALENT (the amount of hidden keys is the cause) = the 90% CI of mean(diff) lies inside +-0.25|g| (TOST). SPLIT (which keys matter) = the 95% CI of diff excludes 0 AND |mean diff| > 2 SD_diff AND |mean diff| >= 0.5|g|. Anything else = undecided -> repeat once with 5 seeds, then report rho = dC/dALLQ with its CI."""
    nat,allq,arm=map(lambda v:np.asarray(v,float),(nat,allq,arm));assert len(nat)==len(allq)==len(arm)>=3,'seeds >= 3'
    g=allq-nat;gm,glo,ghi,_=ci(g,0.90)
    if glo<=0<=ghi:return {'verdict':'not attributable (90% CI of g contains 0)','g':gm,'g_ci90':(glo,ghi)}
    diff=arm-allq;dm,l90,h90,sd=ci(diff,0.90);_,l95,h95,_=ci(diff,0.95);ga=abs(gm)
    if l90>=-0.25*ga and h90<=0.25*ga:v='equivalent (TOST): the AMOUNT of hidden keys explains the gain'
    elif (l95>0 or h95<0) and abs(dm)>2*sd and abs(dm)>=0.5*ga:v='split: WHICH keys are hidden matters'
    else:v='undecided: rerun with 5 seeds once, then report rho'
    rho=float(np.mean(arm-nat)/np.mean(allq-nat)) if np.mean(allq-nat)!=0 else float('nan')
    return {'verdict':v,'arm':arm.mean().item(),'g':gm,'g_ci90':(glo,ghi),'diff_mean':dm,'diff_ci90':(l90,h90),'diff_ci95':(l95,h95),'diff_sd':sd,'rho':rho}
def register_engine():
    """SGDSEL = UV arm on the SGD path: in E.ARMS and E.UVARMS, not in E.ADAM (e55_core then uses write_step, the clipped SGD rule, with the keep from the uv object)."""
    import e55_swap as E
    if ENGINE not in E.ARMS:E.ARMS.append(ENGINE)
    if ENGINE not in E.UVARMS:E.UVARMS=tuple(E.UVARMS)+(ENGINE,)
    assert ENGINE not in E.ADAM
def parse_arm(name,mask_arm,rand_arm='RANDQ256P50'):
    """NATS[xN] -> (no mask 'NATQ', N); NATMASK[xN] -> (mask_arm, N); RANDMASK[xN] -> (rand_arm, N)  (ALLQ masks every row, so a random mask of the SAME ratio is identical; RANDMASK uses the stored random-row arm instead, default 50% of the rows)"""
    m=re.fullmatch(r'(NATS|NATMASK|RANDMASK|NATMASKC|NATMASKD)(?:x(\d+(?:\.\d+)?))?',name);assert m,('arm name',name)
    return {'NATS':'NATQ','NATMASK':mask_arm,'RANDMASK':rand_arm,'NATMASKC':'KEYC:'+mask_arm,'NATMASKD':'KEYD:'+mask_arm}[m.group(1)],(float(m.group(2)) if m.group(2) else 1.0)
def mult_step(orig,mult):
    """wrap Ladder.write_step: the returned inner parameters old + mult x (new - old) in fp32, cast back; (gnorm, coef, ce) unchanged"""
    import jax,jax.numpy as jnp
    def ws(m,*a,**k):
        newp,gn,coef,ce=orig(m,*a,**k)
        if mult==1.0:return newp,gn,coef,ce
        f32=jnp.float32;old=m.inner_parameters()
        newp=jax.tree.map(lambda n,o:(o.astype(f32)+(n.astype(f32)-o.astype(f32))*mult).astype(n.dtype),newp,old)
        return newp,gn,coef,ce
    return ws
def selftest():
    QR.selftest()
    assert parse_arm('NATS','ALLQ256')==('NATQ',1.0) and parse_arm('NATSx4','ALLQ256')==('NATQ',4.0)
    assert parse_arm('RANDMASKx4','ALLQ256')==('RANDQ256P50',4.0) and parse_arm('NATMASK','ALLQ256')==('ALLQ256',1.0) and parse_arm('NATMASKx2','ALLQ128')==('ALLQ128',2.0)
    for bad in ('NATQ','NATMASKx','NATx2'):
        try:parse_arm(bad,'ALLQ256');raise SystemExit('parse_arm accepted '+bad)
        except AssertionError:pass
    register_engine();import e55_swap as E
    assert ENGINE in E.ARMS and ENGINE in E.UVARMS and ENGINE not in E.ADAM
    nat=np.ones(WIN+CH,bool);nat[:50]=False;band=256
    for kind in ('C','D'):
        kp,info=KeyMatchedQ(kind,band,0,lambda *x:None).keep(9,None,None,None,None,nat);assert kp.shape==(CH,WIN+CH)
        d=np.arange(CH)[:,None]+WIN-np.arange(WIN+CH)[None,:];n=(np.broadcast_to(nat,(CH,WIN+CH))&(d>=band)).sum(1);cand=np.broadcast_to(nat,(CH,WIN+CH))&(d>=1)
        assert np.array_equal((cand&~kp).sum(1),n) and kp[np.arange(CH),np.arange(CH)+WIN].all() and not (kp&~np.broadcast_to(nat,(CH,WIN+CH))).any()
        if kind=='D':assert not kp[100,WIN+100-1] and kp[100,WIN+100-n[100]-1] if n[100]>0 and WIN+100-n[100]-1>=50 else True
    k0,_=KeyMatchedQ('C',band,0).keep(3,None,None,None,None,nat);assert k0.ndim==1
    nt=[0.10,0.12,0.11,0.09];aq=[0.30,0.33,0.31,0.29]
    assert verdict_rules(nt,aq,[0.30,0.32,0.31,0.30])['verdict'].startswith('equivalent'),verdict_rules(nt,aq,[0.30,0.32,0.31,0.30])
    assert verdict_rules(nt,aq,[0.11,0.13,0.10,0.12])['verdict'].startswith('split')
    assert verdict_rules(nt,[0.10,0.13,0.10,0.11],aq)['verdict'].startswith('not attributable')
    assert verdict_rules(nt,aq,[0.25,0.33,0.20,0.34])['verdict'].startswith('undecided')
    # multiplier wrapper on a toy 'model'
    import jax.numpy as jnp
    class M_:
        def inner_parameters(self):return {'w':jnp.ones((3,),jnp.float32)}
    orig=lambda m,*a,**k:({'w':jnp.full((3,),0.9,jnp.float32)},1.0,1.0,0.0)
    n2=mult_step(orig,2.0)(M_())[0]['w'];assert np.allclose(np.asarray(n2),0.8,atol=1e-6),n2
    n1=mult_step(orig,1.0)(M_())[0]['w'];assert np.allclose(np.asarray(n1),0.9,atol=1e-6)
    print('e57_natmask_run selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--masks');ap.add_argument('--mask-arm',default='ALLQ256');ap.add_argument('--rand-arm',default='RANDQ256P50');ap.add_argument('--arms');ap.add_argument('--tag',default='synng');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e55_read as R,e55_gpu as RG,gen_facts as G
    import e57_syn_pass as SP
    register_engine()
    seed=a.seed;OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True);arms=a.arms.split(',');ro=[16,32,64,128];ri=[128];sv=[128];CH=1024
    masks={k:v for k,v in np.load(a.masks,allow_pickle=True).items()};spec={ar:parse_arm(ar,a.mask_arm,a.rand_arm) for ar in arms}
    for ar,(ma,_) in spec.items():assert ma=='NATQ' or f'band_{ma[5:] if ma.startswith("KEY") else ma}' in masks,('mask arm not in the mask file',ar,ma)
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    lg=lambda *x:print(*x,flush=True)
    enc,w,tf=R55.build(SP.synth_args(seed),seed,read_apps=set(ro)|set(ri)|set(sv),arms=['NAT']);S=R55.streams(w,enc);nch=a.smoke or w['n_chunks'];p=w['p'];assert p==1
    ids,sha=S['A'];assert sha==G.stream_sha(G.realize(w)),'stream hash mismatch'
    P=R.plan(w,enc);items=P['items'];reads=[a_*p-1 for a_ in sv] if not a.smoke else [nch-1];canary_chunks=reads
    tok=rt.tok;raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w))
    R_=RG.Reads(rd,raw,len(items),G.NCAND,P['read'],out_tail,rt.base,ro,ri,log=lambda *x:lg(*x));lg(f'seed {seed}: W0 read done')
    for ar in arms:
        fn=OUT/f'E55_{a.tag}_{ar}_s{seed}.saved.json'
        if fn.exists() and not a.rerun:lg('exists, reused:',fn);continue
        ma,mult=spec[ar];uv=(KeyMatchedQ(ma[3],masks[f'band_{ma[5:]}'],seed,lg) if ma.startswith('KEY') else QR.SelMaskQ(ma,masks,lg));L=C.Ladder55(rt,save_dir=OUT/'states'/f'E55_{a.tag}_{ar}_s{seed}',canary=can);got={}
        L.write_step=mult_step(L.write_step,mult)
        def partial(k,m,ar=ar,got=got):
            x=R_.at((k+1)//p,m);got[(k+1)//p]=x
            rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':'v3_16_1','seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':False})[ar];json.dump(rec,open(OUT/f'E55_{a.tag}_{ar}_s{seed}.read.partial.json','w'))
            return {c_:'ok' for c_ in x}
        hook_chunks=[a_*p-1 for a_ in sorted(set(ro)|set(ri))] if not a.smoke else [nch-1]
        t0=time.monotonic();lg(f'== seed {seed} {ar} (mask {ma}, lr x{mult:g}): {nch} chunks, reads out {ro} in {ri}')
        r=L.run_stream(ENGINE,p,ids,decoy_ids=S['decoy'][0] if 'decoy' in S else None,bg_ids=S['nofact'][0] if 'nofact' in S else (S['bg'][0] if 'bg' in S else None),read_chunks=reads,canary_chunks=canary_chunks,nch=nch,tag=ar+f'_s{seed}',log=lg,read_hook=partial,hook_chunks=hook_chunks,chunks_meta=w['chunks'],uv=uv);r.pop('model',None)
        lgs=r['log'];r['summary']={'mean_update_norm':float(np.mean([e['update_norm'] for e in lgs])),'clip_bind_frac':r['clip_bind_frac'],'cum_update_norm':r['cum_update_norm'],'mean_ce':float(np.mean([e['ce'] for e in lgs])),'mean_gnorm':float(np.mean([e['gnorm'] for e in lgs])),'mean_coef':float(np.mean([e['coef'] for e in lgs])),'median_update_norm':float(np.median([e['update_norm'] for e in lgs])),'clip_frac_coef_lt1':float(np.mean([e['coef']<1.0 for e in lgs])),'mean_masked_row_frac':float(np.mean([e.get('q_banded',0) for e in lgs])/CH),'mean_hidden_keys_row':float(np.mean([e['hidden_keys_row_mean'] for e in lgs if 'hidden_keys_row_mean' in e] or [0]))}
        r.update(stream_sha=sha,world_seed=seed,cell='v3_16_1',read_apps_out=ro,read_apps_in=ri,save_apps=sv,sec_total=time.monotonic()-t0,k=w['k'],n=w['n'],bank=w['bank_kind'],engine='SGD(clip)',mask_arm=ma,band=(int(masks[f'band_{ma[5:] if ma.startswith("KEY") else ma}']) if ma!='NATQ' else None),lr_mult=mult,arm=ar)
        w0=r['canary_w0']
        for k,e in r['reads'].items():
            if 'canary' in e:
                for key in ('_gen_nll','_tail_nll'):lg(f'  canary {key} at chunk {k}: {e["canary"][key]-w0[key]:+.4f}')
        rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':'v3_16_1','seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':True,'place_len':w['place_len']})[ar];fr=OUT/f'E55_{a.tag}_{ar}_s{seed}.read.json';json.dump(rec,open(fr,'w'));json.dump(r,open(fn,'w'),default=float)
        lg('wrote',fn,f'{time.monotonic()-t0:.0f}s');gc.collect();LD.jax.clear_caches()
if __name__=='__main__':main()
