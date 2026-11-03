"""E57 synthetic world: M2h / M3 write arms against ALLQ256 (GPU; one seed of cell v3_16_1, 128 chunks, window 8 chunks; new file; thread 'ALLQ보다 나은 가림 방법 찾기', design confirmed by Hans 11/03 09:39 KST). Same run as e57_syn_qrun.py (e55_run.run with inline frozen-weight reads after appearances 16 / 32 / 64 / 128, canary at the last chunk, state saved at appearance 128, engine SELC1 = Adam direction x norm target) with two new mask / loss rules on top of ALLQ256 (every query row sees only the previous 256 tokens, as in the arm ALLQ256 of job 360b):
  deciding rows D(k) = the top 30% of the eligible rows of every chunk k >= 8 by window gain g = nll_band256 - nll_win of the W0 pass (e57_syn_mmasks.py, qb_TOP30_<k>)
  M2H64  (Hans' replacement of M2H, 11/03 09:55 KST) the rows of D(k) see only the previous 64 tokens, all other rows are ALLQ256
  M2H    the rows of D(k) see NO window key at all (the 8192 keys before the chunk are hidden; their chunk keys within the band stay), all other rows are ALLQ256
  M3W<w> ALLQ256 mask for every row, the loss of the rows of D(k) is weighted by w (w = 2, 4; loss = sum_i valid_i x (w if i in D else 1) x nll_i / N_valid, i.e. exactly lm_loss for w = 1)
  update norm: suffix N05 = 0.5 x the NAT norm of the chunk (as ALLQ256 of job 360b), suffix NALL = the norm ALLQ256 really used in the same chunk (E55_synq_ALLQ256_s64.saved.json); arm names M2HN05, M2HNALL, M3W2N05, M3W2NALL, M3W4N05, M3W4NALL (no underscore: file names E55_synq_<ARM>_s<seed>.saved.json / .read.json, picked up by e57_syn_tokread.py).
  python3 e57_syn_mrun.py --seed 64 --out O --m30 syn_s64_m30_masks.npz --ntg-nat E55_v3a_NAT_s64.saved.json --ntg-allq E55_synq_ALLQ256_s64.saved.json --arms M2HN05,... [--smoke N] [--tag synq] | --selftest   (XLA_FLAGS=--xla_gpu_deterministic_ops=true)"""
import sys,os,json,time,gc,hashlib,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_qband_run as QR
import e57_closure_masks as CM
CH=1024;WIN=8*CH;K0=8
ARMS={'M2H64N05':('M2H64',1,'N05'),'M2HN05':('M2H',1,'N05'),'M2HNALL':('M2H',1,'NALL'),'M3W2N05':('M3',2,'N05'),'M3W2NALL':('M3',2,'NALL'),'M3W4N05':('M3',4,'N05'),'M3W8N05':('M3',8,'N05'),'M3W16N05':('M3',16,'N05'),'M3NMW8N05':('M3NM',8,'N05'),'M3W4NALL':('M3',4,'NALL')}
class UVDeciding:
    """UV-compatible keep object: ALLQ256 for every row from chunk K0; M2H: the deciding rows additionally lose the window keys; for M3 only the deciding-row vector of the chunk is published in holder['dec'] for the weighted gradient"""
    def __init__(self,kind,masks,holder,log=print):self.kind=kind;self.m=masks;self.holder=holder;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool);self.holder['dec']=np.zeros(CH,np.float32)
        if k<K0:return base,{'uv_arm':self.kind,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        dec=CM.up(self.m[f'qb_TOP30_{k}'],CH);self.holder['dec']=dec.astype(np.float32)
        if self.kind=='M3NM':return base,{'uv_arm':self.kind,'q_banded':0,'n_deciding':int(dec.sum()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}     # no mask at all: natural window for every row, only the loss weight on the deciding rows
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();bk=CM.band_keep(base,256);kp[:]=bk
        if self.kind=='M2H':kp[dec,:WIN]=False
        if self.kind=='M2H64':b64=CM.band_keep(base,64);kp[dec]=b64[dec]       # deciding rows see only the previous 64 tokens (chunk and window keys), all other rows ALLQ256
        if k%16==0:self.log(f'  {self.kind} chunk {k}: deciding rows {int(dec.sum())}, window-less {int(dec.sum()) if self.kind=="M2H" else 0}')
        return kp,{'uv_arm':self.kind,'q_banded':CH,'q_band':256,'n_deciding':int(dec.sum()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def install_weighted_grad(L,s3,holder,weight):
    """replace L.grad_mask (used by the Adam engines) by the gradient of the loss with the weight `weight` on the deciding rows"""
    import jax,jax.numpy as jnp,equinox as eqx
    S3SUF=s3.S3SUF
    @eqx.filter_jit
    def grad_w(m,st_suf,c,pref_,kmask,w):
        def total(mm):
            S3SUF['kmask']=kmask
            try:out=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
            finally:S3SUF.clear()
            lp=jax.nn.log_softmax(out.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(w*nll),nll
        (_l,nll),grads=eqx.filter_value_and_grad(total,has_aux=True)(m);valid=c.loss_masks.astype(jnp.float32);ce=jnp.sum(valid*nll)/jnp.maximum(jnp.sum(valid),1e-10);return grads.inner_parameters(),ce
    def grad_mask(m,st_suf,c,pref_,kmask):
        valid=jnp.asarray(c.loss_masks,jnp.float32);dec=jnp.asarray(holder['dec'],jnp.float32);w=valid*(1.0+(weight-1.0)*dec)/jnp.maximum(valid.sum(),1e-10);return grad_w(m,st_suf,c,pref_,kmask,w)
    L.grad_mask=grad_mask
def selftest():
    QR.selftest()
    rng=np.random.default_rng(0);dec=rng.random(CH)<.3;holder={};masks={'qb_TOP30_9':CM.pk(dec)};u=UVDeciding('M2H',masks,holder,log=lambda *x:None);nat=np.ones(WIN+CH,bool);kp,info=u.keep(9,None,None,None,None,nat)
    assert kp.shape==(CH,WIN+CH) and not kp[dec,:WIN].any() and info['n_deciding']==int(dec.sum())
    kp2,_=UVDeciding('M3',masks,{},log=lambda *x:None).keep(9,None,None,None,None,nat);allq=CM.band_keep(nat,256);assert np.array_equal(kp2,allq)
    # M2H differs from ALLQ256 only on the deciding rows whose band reaches into the window (position < 255)
    diff=(kp!=allq).any(1);assert diff.sum()==int((dec&(np.arange(CH)<255)).sum())
    kp3,_=UVDeciding('M2H64',masks,{},log=lambda *x:None).keep(9,None,None,None,None,nat);b64=CM.band_keep(nat,64);assert np.array_equal(kp3[dec],b64[dec]) and np.array_equal(kp3[~dec],allq[~dec]) and kp3[dec].sum(1).max()==64
    print('e57_syn_mrun selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--m30');ap.add_argument('--ntg-nat');ap.add_argument('--ntg-allq');ap.add_argument('--arms');ap.add_argument('--tag',default='synq');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e55_read as R,e55_gpu as RG,gen_facts as G
    import e57_syn_pass as SP
    seed=a.seed;OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True);arms=a.arms.split(',');ro=[16,32,64,128];ri=[128];sv=[128];CH_=1024
    for ar in arms:assert ar in ARMS,ar
    masks={k:v for k,v in np.load(a.m30,allow_pickle=True).items()};ntgN=[e['update_norm'] for e in json.load(open(a.ntg_nat))['log']];ntgA=[e['update_norm']/0.5 for e in json.load(open(a.ntg_allq))['log']];assert len(ntgN)>=128 and len(ntgA)>=128
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
        kind,weight,nrm=ARMS[ar];ntg=ntgN if nrm=='N05' else ntgA;holder={'dec':np.zeros(CH_,np.float32)}
        uv=UVDeciding(kind,masks,holder,lg);L=C.Ladder55(rt,save_dir=OUT/'states'/f'E55_{a.tag}_{ar}_s{seed}',canary=can);got={}
        if kind in ('M3','M3NM'):install_weighted_grad(L,s3,holder,float(weight))
        def partial(k,m,ar=ar,got=got):
            x=R_.at((k+1)//p,m);got[(k+1)//p]=x
            rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':'v3_16_1','seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':False})[ar];json.dump(rec,open(OUT/f'E55_{a.tag}_{ar}_s{seed}.read.partial.json','w'))
            return {c_:'ok' for c_ in x}
        hook_chunks=[a_*p-1 for a_ in sorted(set(ro)|set(ri))] if not a.smoke else [nch-1]
        t0=time.monotonic();lg(f'== seed {seed} {ar} ({kind}, weight {weight}, norms {nrm}): {nch} chunks, reads out {ro} in {ri}')
        r=L.run_stream('SELC1',p,ids,decoy_ids=S['decoy'][0] if 'decoy' in S else None,bg_ids=S['nofact'][0] if 'nofact' in S else (S['bg'][0] if 'bg' in S else None),read_chunks=reads,canary_chunks=canary_chunks,nch=nch,tag=ar+f'_s{seed}',log=lg,read_hook=partial,hook_chunks=hook_chunks,norm_targets=ntg,chunks_meta=w['chunks'],uv=uv);r.pop('model',None)
        lgs=r['log'];r['summary']={'mean_update_norm':float(np.mean([e['update_norm'] for e in lgs])),'clip_bind_frac':r['clip_bind_frac'],'cum_update_norm':r['cum_update_norm'],'mean_ce':float(np.mean([e['ce'] for e in lgs])),'mean_scale':float(np.mean([e.get('scale',1.0) for e in lgs]))}
        r.update(stream_sha=sha,world_seed=seed,cell='v3_16_1',read_apps_out=ro,read_apps_in=ri,save_apps=sv,sec_total=time.monotonic()-t0,k=w['k'],n=w['n'],bank=w['bank_kind'],engine='SELC1',mrun=True,arm=ar,mkind=kind,weight=weight,norm_source=nrm)
        w0=r['canary_w0']
        for k,e in r['reads'].items():
            if 'canary' in e:
                for key in ('_gen_nll','_tail_nll'):lg(f'  canary {key} at chunk {k}: {e["canary"][key]-w0[key]:+.4f}')
        rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':'v3_16_1','seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':True,'place_len':w['place_len']})[ar];fr=OUT/f'E55_{a.tag}_{ar}_s{seed}.read.json';json.dump(rec,open(fr,'w'));json.dump(r,open(fn,'w'),default=float)
        lg('wrote',fn,f'{time.monotonic()-t0:.0f}s');gc.collect();LD.jax.clear_caches()
if __name__=='__main__':main()
