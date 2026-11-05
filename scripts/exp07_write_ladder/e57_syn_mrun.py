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
import e57_longcell      # registers the cell v3_16_1_L1024 (long synthetic stream)
import e57_syn_shuf      # pair-shuffled place tokens when E57_SYN_SHUF is set (job 960); otherwise no effect
CH=1024;WIN=8*CH;K0=8
ARMS={'M2H64N05':('M2H64',1,'N05'),'M2HN05':('M2H',1,'N05'),'M2HNALL':('M2H',1,'NALL'),'M3W2N05':('M3',2,'N05'),'M3W2NALL':('M3',2,'NALL'),'M3W4N05':('M3',4,'N05'),'M3W8N05':('M3',8,'N05'),'M3W16N05':('M3',16,'N05'),'M3NMW8N05':('M3NM',8,'N05'),'M3W1N05':('M3',1,'N05'),'M4W8N05':('M4',8,'N05'),'KL16W8N05':('KL16',8,'N05'),'M3NMW1N05':('M3NM',1,'N05'),'M3PB05W8N05':('M3PB05',8,'N05'),'M3PB10W8N05':('M3PB10',8,'N05'),'M3PBC05W8N05':('M3PBC05',8,'N05'),'M3PBC10W8N05':('M3PBC10',8,'N05'),'M3D30W8N05':('PD30',8,'N05'),'M3D50W8N05':('PD50',8,'N05'),'M3R30W8N05':('PR30',8,'N05'),'NATSGD':('NATSGD',1,'-'),'M3B128W1N05':('M3B128',1,'N05'),'KL64W8N05':('KL64',8,'N05'),'M3W4NALL':('M3',4,'NALL'),'M3CGW8N05':('PCG',8,'N05'),'M3CGSHW8N05':('PCGSH',8,'N05'),'M3CWW8N05':('PCW',8,'N05'),'M3CBW8N05':('PCB',8,'N05'),'M3CGW10N05':('PCG',10,'N05'),'M3CGSHW10N05':('PCGSH',10,'N05'),'M3CWW10N05':('PCW',10,'N05'),'M3CBW10N05':('PCB',10,'N05'),'M3T15W8N05':('M3T15',8,'N05'),'M3T50W8N05':('M3T50',8,'N05'),'M3W32N05':('M3',32,'N05'),'M3GWW10N05':('M3GW',10,'N05'),'M3GWW20N05':('M3GW',20,'N05'),'M3GEW8N05':('M3GE',8,'N05'),'M3GEW16N05':('M3GE',16,'N05'),'M3T05W8N05':('M3T05',8,'N05'),'M3T05W16N05':('M3T05',16,'N05'),'M3T10W8N05':('M3T10',8,'N05'),'M3T10W16N05':('M3T10',16,'N05'),'M3T15W16N05':('M3T15',16,'N05'),'M3TA05W16N05':('M3TA05',16,'N05'),'M3TA10W16N05':('M3TA10',16,'N05'),'M3TA05W8N05':('M3TA05',8,'N05'),'M3TA10W8N05':('M3TA10',8,'N05')}
CONT=('PCG','PCGSH','PCW','PCB')     # continuous arms (job 1000): s in [0,1] per row from e57_syn_cmasks.py; band b = 8192 x (16/8192)^s (s = 0: natural window of 8192), loss weight 1 + (w - 1) s
def cont_keep(base,s,band=True):
    """-> bool [CH, WIN+CH]: row i sees the previous b_i tokens (b_i = round(8192 (16/8192)^s_i), window AND chunk keys) intersected with the natural keep; rows with s = 0 keep the natural window unchanged"""
    kp=np.broadcast_to(base,(CH,WIN+CH)).copy()
    if not band:return kp
    b=np.maximum(np.round(WIN*(16.0/WIN)**np.asarray(s,np.float64)).astype(np.int64),16);d=np.arange(CH)[:,None]+WIN-np.arange(WIN+CH)[None,:];rows=np.where(np.asarray(s)>0)[0]
    kp[rows]=(base[None,:]&(d[rows]>=0)&(d[rows]<b[rows][:,None]));return kp
class UVDeciding:
    """UV-compatible keep object: ALLQ256 for every row from chunk K0; M2H: the deciding rows additionally lose the window keys; for M3 only the deciding-row vector of the chunk is published in holder['dec'] for the weighted gradient"""
    def __init__(self,kind,masks,holder,log=print):self.kind=kind;self.m=masks;self.holder=holder;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool);self.holder['dec']=np.zeros(CH,np.float32)
        if k<K0:return base,{'uv_arm':self.kind,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        if self.kind in ('M3T05','M3T10','M3TA05','M3TA10','M3T15','M3T50','M3GW','M3GE'):          # grid job 1020: ALLQ256 on every row; loss weight on the top 15 / 50% rows by g (M3T15 / M3T50) or GRADED by the continuous s of e57_syn_cmasks.py (M3GW: w_i = 1 + (w - 1) s_i)
            dec=(np.asarray(self.m[f'qc_s_{k}' if self.kind=='M3GW' else f'qe_f_{k}'],np.float32) if self.kind in ('M3GW','M3GE') else CM.up(self.m[f'qt{self.kind[3:]}_{k}'],CH).astype(np.float32));self.holder['dec']=dec;kp=np.broadcast_to(base,(CH,WIN+CH)).copy();kp[:]=CM.band_keep(base,256)
            return kp,{'uv_arm':self.kind,'q_banded':CH,'q_band':256,'n_deciding':int((dec>0).sum()),'mean_dec':float(dec.mean()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        if self.kind in CONT:               # job 1000: continuous mapping from the window gain (PCG), the same s permuted among the eligible rows (PCGSH), weight only with the natural window (PCW), band only with weight 1 (PCB)
            s=np.asarray(self.m[('qcsh_s_%d' if self.kind=='PCGSH' else 'qc_s_%d')%k],np.float32);self.holder['dec']=(np.zeros(CH,np.float32) if self.kind=='PCB' else s);kpp=cont_keep(base,s,band=self.kind!='PCW')
            return kpp,{'uv_arm':self.kind,'q_banded':int((s>0).sum()) if self.kind!='PCW' else 0,'mean_s':float(s.mean()),'n_deciding':int((s>0).sum()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        if self.kind in ('PD30','PD50','PR30'):          # partial masking: ONLY the selected rows get the ALLQ256 band, all other rows keep the natural window; the selected rows carry the loss weight
            dec=CM.up(self.m[f'qp_{self.kind[1:]}_{k}'],CH);self.holder['dec']=dec.astype(np.float32);kpp=np.broadcast_to(base,(CH,WIN+CH)).copy();kpp[dec]=CM.band_keep(base,256)[dec]
            return kpp,{'uv_arm':self.kind,'q_banded':int(dec.sum()),'q_band':256,'n_deciding':int(dec.sum()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        dec=CM.up(self.m[f'qb_TOP30_{k}'],CH);self.holder['dec']=dec.astype(np.float32)
        if self.kind=='M3NM':return base,{'uv_arm':self.kind,'q_banded':0,'n_deciding':int(dec.sum()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}     # no mask at all: natural window for every row, only the loss weight on the deciding rows
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();bk=CM.band_keep(base,128 if self.kind=='M3B128' else 256);kp[:]=bk
        if self.kind=='M2H':kp[dec,:WIN]=False
        if self.kind in ('M2H64','M4'):b64=CM.band_keep(base,64);kp[dec]=b64[dec]       # deciding rows see only the previous 64 tokens (chunk and window keys), all other rows ALLQ256
        if self.kind in ('KL16','KL64'):           # key-level: the deciding rows see the natural window minus their top-m ALLV far keys (W0 key-mass dump), all other rows ALLQ256
            rows=np.where(dec)[0];idx=self.m[f'qk_ALLV{self.kind[2:]}_{k}'];assert idx.shape[0]==len(rows);kp[rows]=base[None,:]
            for r,ix in zip(rows,idx):kp[r,ix[ix>=0].astype(np.int64)]=False
        if self.kind.startswith('M3PB'):             # contrast term: second suffix forward in which every row sees only its own position (band 1); prefix outputs of that forward computed here (frozen prefix layers)
            import jax.numpy as jnp
            kp2=CM.band_keep(base,1);L=self.holder['L'];self.holder['kmask2']=jnp.asarray(kp2);self.holder['pref2']=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(kp2))
        if k%16==0:self.log(f'  {self.kind} chunk {k}: deciding rows {int(dec.sum())}, window-less {int(dec.sum()) if self.kind=="M2H" else 0}')
        return kp,{'uv_arm':self.kind,'q_banded':CH,'q_band':128 if self.kind=='M3B128' else 256,'n_deciding':int(dec.sum()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
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
def install_contrast_grad(L,s3,holder,weight,beta,log=print,cap=None):
    """after install_weighted_grad: the loss  sum_i w_i nll_i - beta * sum_{i in D} w_i nll'_i  with w_i = valid (1 + (weight - 1) dec_i) / n_valid and nll' = the NLL of the same rows in a second suffix forward in which each row sees only its own position (band 1, holder['kmask2'], holder['pref2'] set by UVDeciding.keep); chunks without a contrast mask (k < K0) use the plain weighted loss. The mean nll' over the deciding rows is stored in holder['nll2']."""
    import jax,jax.numpy as jnp,equinox as eqx
    S3SUF=s3.S3SUF;base_gm=L.grad_mask
    @eqx.filter_jit
    def grad_c(m,st_suf,c,pref_,kmask,pref2,kmask2,w,w2):
        def fwd(mm,pf,km):
            S3SUF['kmask']=km
            try:out=mm.language_model.suffix_call(prefix_outputs=pf,state=st_suf,seq=c)
            finally:S3SUF.clear()
            lp=jax.nn.log_softmax(out.logits.astype(jnp.float32),axis=-1);return -jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
        def total(mm):
            nll=fwd(mm,pref_,kmask);nll2=fwd(mm,pref2,kmask2);nll2c=nll2 if cap is None else jnp.minimum(nll2,cap);return jnp.sum(w*nll)-beta*jnp.sum(w2*nll2c),(nll,nll2)
        (_l,(nll,nll2)),grads=eqx.filter_value_and_grad(total,has_aux=True)(m);valid=c.loss_masks.astype(jnp.float32);ce=jnp.sum(valid*nll)/jnp.maximum(jnp.sum(valid),1e-10);m2=jnp.sum(w2*nll2)/jnp.maximum(jnp.sum(w2),1e-30)
        return grads.inner_parameters(),ce,m2
    def grad_mask(m,st_suf,c,pref_,kmask):
        if holder.get('pref2') is None:holder['nll2']=None;return base_gm(m,st_suf,c,pref_,kmask)
        valid=jnp.asarray(c.loss_masks,jnp.float32);dec=jnp.asarray(holder['dec'],jnp.float32);w=valid*(1.0+(weight-1.0)*dec)/jnp.maximum(valid.sum(),1e-10)
        g,ce,m2=grad_c(m,st_suf,c,pref_,kmask,holder['pref2'],holder['kmask2'],w,w*dec);m2=float(m2);holder['nll2']=m2;holder.setdefault('clog',[]).append(m2)
        if not np.isfinite(m2) or m2>holder.get('nll2_max',100.0):raise RuntimeError(f"contrast NLL' blew up: mean nll' over the deciding rows {m2}")
        return g,ce
    L.grad_mask=grad_mask
def selftest():
    QR.selftest()
    rng=np.random.default_rng(0);dec=rng.random(CH)<.3;holder={};masks={'qb_TOP30_9':CM.pk(dec)};u=UVDeciding('M2H',masks,holder,log=lambda *x:None);nat=np.ones(WIN+CH,bool);kp,info=u.keep(9,None,None,None,None,nat)
    assert kp.shape==(CH,WIN+CH) and not kp[dec,:WIN].any() and info['n_deciding']==int(dec.sum())
    kp2,_=UVDeciding('M3',masks,{},log=lambda *x:None).keep(9,None,None,None,None,nat);allq=CM.band_keep(nat,256);assert np.array_equal(kp2,allq)
    # M2H differs from ALLQ256 only on the deciding rows whose band reaches into the window (position < 255)
    diff=(kp!=allq).any(1);assert diff.sum()==int((dec&(np.arange(CH)<255)).sum())
    kp3,_=UVDeciding('M2H64',masks,{},log=lambda *x:None).keep(9,None,None,None,None,nat);b64=CM.band_keep(nat,64);assert np.array_equal(kp3[dec],b64[dec]) and np.array_equal(kp3[~dec],allq[~dec]) and kp3[dec].sum(1).max()==64
    s=np.linspace(0,1,CH).astype(np.float32);kc=cont_keep(nat,s);assert np.array_equal(kc[0],nat) and kc[-1].sum()==16 and (np.diff(kc.sum(1).astype(np.int64))<=0).sum()>0
    assert np.array_equal(cont_keep(nat,s,band=False),np.broadcast_to(nat,(CH,WIN+CH))) and (kc.sum(1)>=16).all()
    z=np.zeros(CH,np.float32);assert np.array_equal(cont_keep(nat,z),np.broadcast_to(nat,(CH,WIN+CH)))
    m2={'qc_s_9':s};h2={};kp5,_=UVDeciding('PCG',m2,h2,log=lambda *x:None).keep(9,None,None,None,None,nat);assert np.array_equal(kp5,kc) and np.array_equal(h2['dec'],s)
    h3={};UVDeciding('PCB',m2,h3,log=lambda *x:None).keep(9,None,None,None,None,nat);assert h3['dec'].sum()==0
    ms={'qt15_9':CM.pk(dec),'qc_s_9':np.linspace(0,1,CH).astype(np.float32)};hh={};kq,_=UVDeciding('M3T15',ms,hh,log=lambda *x:None).keep(9,None,None,None,None,nat);assert np.array_equal(kq,allq) and np.array_equal(hh['dec'],dec.astype(np.float32))
    hh={};kq,_=UVDeciding('M3GW',ms,hh,log=lambda *x:None).keep(9,None,None,None,None,nat);assert np.array_equal(kq,allq) and np.array_equal(hh['dec'],ms['qc_s_9'])
    me={'qe_f_9':np.linspace(0,1,CH).astype(np.float32)**2};hh={};kq,_=UVDeciding('M3GE',me,hh,log=lambda *x:None).keep(9,None,None,None,None,nat);assert np.array_equal(kq,allq) and np.array_equal(hh['dec'],me['qe_f_9'])
    print('e57_syn_mrun selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--cell',default='v3_16_1');ap.add_argument('--save-apps',default='128');ap.add_argument('--no-inline',action='store_true');ap.add_argument('--segmented',action='store_true');ap.add_argument('--out');ap.add_argument('--m30');ap.add_argument('--ntg-nat');ap.add_argument('--ntg-allq');ap.add_argument('--arms');ap.add_argument('--tag',default='synq');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e55_read as R,e55_gpu as RG,gen_facts as G
    import e57_syn_pass as SP
    seed=a.seed;OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True);arms=a.arms.split(',');ro=[16,32,64,128];ri=[128];sv=[int(x) for x in a.save_apps.split(',')];CH_=1024
    for ar in arms:assert ar in ARMS,ar
    masks={k:v for k,v in np.load(a.m30,allow_pickle=True).items()};ntgN=None if a.ntg_nat=='auto' else [e['update_norm'] for e in json.load(open(a.ntg_nat))['log']];ntgA=[e['update_norm']/0.5 for e in json.load(open(a.ntg_allq))['log']] if a.ntg_allq else None;assert (ntgN is None or len(ntgN)>=128) and (ntgA is None or len(ntgA)>=128)
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);can_w0=can(rt.base)
    if a.segmented:import e57_pqa2_run as P2;P2.patch_segmented()         # segmented K,V record / prefix for the 1024-chunk stream
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    lg=lambda *x:print(*x,flush=True)
    enc,w,tf=R55.build(SP.synth_args(seed,a.cell),seed,read_apps=set(ro)|set(ri)|{128},arms=['NAT']);S=R55.streams(w,enc);nch=a.smoke or w['n_chunks'];p=w['p'];
    ids,sha=S['A'];assert sha==G.stream_sha(G.realize(w)),'stream hash mismatch'
    P=R.plan(w,enc);items=P['items'];reads=[a_*p-1 for a_ in sv] if not a.smoke else [nch-1];canary_chunks=reads
    tok=rt.tok;raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w))
    R_=None if a.no_inline else RG.Reads(rd,raw,len(items),G.NCAND,P['read'],out_tail,rt.base,ro,ri,log=lambda *x:lg(*x));lg(f'seed {seed}: W0 read done' if R_ else f'seed {seed}: inline reads off')
    rc={}
    if a.segmented:             # long stream: record the natural K,V once (segmented) and share it between the arms
        L0=C.Ladder55(rt,save_dir=OUT/'states'/f'rec_{a.tag}_s{seed}',canary=None);t_rec=time.monotonic()
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids[:nch*CH])
        import hashlib as _H;rc[(_H.sha256(np.asarray(ids[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);del L0,_h;gc.collect();LD.jax.clear_caches();lg(f'  natural K,V recorded once in {time.monotonic()-t_rec:.0f}s')
    for ar in arms:
        fn=OUT/f'E55_{a.tag}_{ar}_s{seed}.saved.json'
        if fn.exists() and not a.rerun:lg('exists, reused:',fn);continue
        kind,weight,nrm=ARMS[ar];ntg=None if kind=='NATSGD' else (ntgN if nrm=='N05' else ntgA);engine='NAT' if kind=='NATSGD' else 'SELC1';assert kind=='NATSGD' or nrm!='N05' or ntgN is not None,'NATSGD (norm source) must run first with --ntg-nat auto';holder={'dec':np.zeros(CH_,np.float32)}
        uv=None if kind=='NATSGD' else UVDeciding(kind,masks,holder,lg);L=C.Ladder55(rt,save_dir=OUT/'states'/f'E55_{a.tag}_{ar}_s{seed}',canary=can);got={}
        if rc:L.rec_cache=rc
        if kind in ('M3','M3NM','M3B128','M4','KL16','KL64','PD30','PD50','PR30','PCG','PCGSH','PCW','PCB','M3T05','M3T10','M3TA05','M3TA10','M3T15','M3T50','M3GW','M3GE') or kind.startswith('M3PB'):install_weighted_grad(L,s3,holder,float(weight))
        if kind.startswith('M3PB'):
            holder['L']=L;install_contrast_grad(L,s3,holder,float(weight),0.5 if kind.endswith('05') else 1.0,cap=8.0 if kind.startswith('M3PBC') else None)
            _cn=L.canary
            def canary_checked(mm,_cn=_cn):
                cn=_cn(mm)
                if cn['_gen_nll']-can_w0['_gen_nll']>0.5:raise RuntimeError(f"canary generation NLL rose by {cn['_gen_nll']-can_w0['_gen_nll']:+.3f} > 0.5")
                return cn
            L.canary=canary_checked
        def partial(k,m,ar=ar,got=got):
            x=R_.at((k+1)//p,m);got[(k+1)//p]=x
            rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':a.cell,'seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':False})[ar];json.dump(rec,open(OUT/f'E55_{a.tag}_{ar}_s{seed}.read.partial.json','w'))
            return {c_:'ok' for c_ in x}
        hook_chunks=[] if a.no_inline else ([a_*p-1 for a_ in sorted(set(ro)|set(ri))] if not a.smoke else [nch-1])
        t0=time.monotonic();lg(f'== seed {seed} {ar} ({kind}, weight {weight}, norms {nrm}): {nch} chunks, reads out {ro} in {ri}')
        try:
            r=L.run_stream(engine,p,ids,decoy_ids=S['decoy'][0] if 'decoy' in S else None,bg_ids=S['nofact'][0] if 'nofact' in S else (S['bg'][0] if 'bg' in S else None),read_chunks=reads,canary_chunks=canary_chunks,nch=nch,tag=ar+f'_s{seed}',log=lg,read_hook=None if a.no_inline else partial,hook_chunks=hook_chunks,norm_targets=ntg,chunks_meta=w['chunks'],uv=uv);r.pop('model',None)
        except RuntimeError as e_:
            if not kind.startswith('M3PB'):raise
            lg('ABORTED',ar,str(e_));json.dump(dict(arm=ar,aborted=str(e_),contrast_log=holder.get('clog')),open(fn,'w'));gc.collect();LD.jax.clear_caches();continue
        if kind=='NATSGD':ntgN=[e['update_norm'] for e in r['log']]
        lgs=r['log'];r['summary']={'mean_update_norm':float(np.mean([e['update_norm'] for e in lgs])),'clip_bind_frac':r['clip_bind_frac'],'cum_update_norm':r['cum_update_norm'],'mean_ce':float(np.mean([e['ce'] for e in lgs])),'mean_scale':float(np.mean([e.get('scale',1.0) for e in lgs]))}
        r.update(stream_sha=sha,world_seed=seed,cell=a.cell,read_apps_out=ro,read_apps_in=ri,save_apps=sv,sec_total=time.monotonic()-t0,k=w['k'],n=w['n'],bank=w['bank_kind'],contrast_log=holder.get('clog'),engine=engine,mrun=True,arm=ar,mkind=kind,weight=weight,norm_source=nrm)
        w0=r['canary_w0']
        for k,e in r['reads'].items():
            if 'canary' in e:
                for key in ('_gen_nll','_tail_nll'):lg(f'  canary {key} at chunk {k}: {e["canary"][key]-w0[key]:+.4f}')
        if a.no_inline:rec={'meta':G.item_meta(items),'read':{},'note':'inline reads off (--no-inline); the per-token reads of e57_syn_tokread.py use this meta'};fr=OUT/f'E55_{a.tag}_{ar}_s{seed}.read.json';json.dump(rec,open(fr,'w'));json.dump(r,open(fn,'w'),default=float)
        else:rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':a.cell,'seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':True,'place_len':w['place_len']})[ar];fr=OUT/f'E55_{a.tag}_{ar}_s{seed}.read.json';json.dump(rec,open(fr,'w'));json.dump(r,open(fn,'w'),default=float)
        lg('wrote',fn,f'{time.monotonic()-t0:.0f}s');gc.collect();LD.jax.clear_caches()
if __name__=='__main__':main()
