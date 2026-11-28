"""New oracle stream P1d (job 1362 'p1d'; GPU; NEW file, wrapper of e59_kvmulti.py which stays UNCHANGED; implementation thread 11/28; design: E59 [S2628-DORACLE], Hans 11/28 04:28 KST "새 오라클로 수치부터 뽑아서 바인딩 차이가 생기는지부터 확인해보자"; EXPLORATORY).
P1d = the 1348 P1s stream (C0 keep for the current-chunk query rows, clipped SGD, natural clip, same lr, same reads) with the CARRIED window recorded at processing time with the D rule of jobs 1359 / 1360 (e59_relay.keep_Drec): no row sees any fact-sentence key of earlier chunks or of its own chunk (a sentence's own rows keep their own sentence), positions kept = the earlier fact sentences are deleted from everything that is carried.
  - prefix layers (frozen at W0): the carried K,V are the chunk-by-chunk D re-record (same jax.lax.scan graph as the engine prefix_masked, e59_relay2.py recorder; weight-independent, computed once per seed);
  - suffix layers: the carried suffix state is advanced at the end of every chunk with the weights at the start of that chunk (engine rule) over the D prefix outputs with keep_Drec as the suffix key mask.
The write of chunk k is the P1s write (C0 keep on the current chunk, loss over the whole chunk) on top of that carried window. Within the current chunk the background rows still see the chunk's own sentences (the design keeps it: "청크 사이 간접 경로를 지운다").
Everything else is e59_kvmulti.run with ARMS = {'P1d': 'C0'}: R1 16 x 16 at 8/16/32/64/96/128, R2 at 128 (1-chunk context, kept only because kvmulti writes it; the 8-chunk reads are a later read-only job on the saved final weights), diagnostics, --save-final.
Checks (--check): chunk-0 write equals P1s of 1348 (gradient norm, CE; nothing is carried into chunk 0); chunk-1 write differs (the carried window changed); the recorder K at layer 0 equals the natural record up to bf16 rounding.
  python3 e59_p1d.py --seed 231 --out O [--nch 128] [--check] | --selftest"""
import sys,os,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
from e59_relay import keep_Drec,keep_B,sent_mask_chunk,earlier_cols,CH,NW,WIN
NREAD=16;FSETS={'P1d':'all','Dr':'read','De':'extra'}      # S2628-REDO step 1 (Hans 11/28 13:46 KST): Dr = only the 16 read facts' sentences deleted from the carried window, De = only the 16 extra facts'
def in_set(f,fset):return fset=='all' or (fset=='read' and int(f)<NREAD) or (fset=='extra' and int(f)>=NREAD)
def keep_Dset(chunks,k,base,fset):
    """keep_Drec restricted to the sentences of the facts in fset: no row sees an earlier-chunk or same-chunk sentence key of those facts (a sentence's own rows keep their own sentence); fset 'all' = keep_Drec"""
    if fset=='all':return keep_Drec(chunks,k,base)
    b2=np.broadcast_to(np.asarray(base,bool),(CH,WIN+CH));kp=b2.copy();fid=earlier_cols(chunks,k);ec=np.where(fid>=0)[0];ec=ec[[in_set(fid[c],fset) for c in ec]];kp[:,ec]=False
    for x in chunks[k]['sents']:
        if in_set(x['fact'],fset):a,b=int(x['start']),int(x['end']);kp[:,WIN+a:WIN+b]=False;kp[a:b,WIN+a:WIN+b]=b2[a:b,WIN+a:WIN+b]
    return kp
def install(log=print,CHK=None):
    """patch e55_iso.Iso55 so that every stream run records the carried prefix K,V with keep_Drec (chunk-by-chunk scan recorder) and advances the suffix state over the D prefix outputs with keep_Drec in the suffix layers"""
    import jax,jax.numpy as jnp,equinox as eqx,e55_iso as ISO,s3
    from ttt.model.data import Batch
    SWA=s3.SWA;_cur=SWA.full_sw_attention
    def _full_rec(self,hs,seq,state):
        if s3.S3.get('mode')=='ctxrec':
            q0=s3.S3['q0'];n=hs.shape[0];_q,xk,xv=self.get_attention_input(hs,position_ids=q0+jnp.arange(n,dtype=jnp.int32));s3.S3['rec']=(xk,xv);s3.S3['mode']='ctx'
            try:return _cur(self,hs,seq,state)
            finally:s3.S3['mode']='ctxrec'
        return _cur(self,hs,seq,state)
    SWA.full_sw_attention=_full_rec
    @eqx.filter_jit
    def prefix_rec(m,ids_,Kst,Vst,q0,keep):
        h=m.language_model.wte_call(ids_);blocks_=m.language_model.model.h.prefix_blocks
        s3.S3['mode']='ctxrec';s3.S3['q0']=q0;s3.S3['keep']=keep
        try:
            seq_=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool))
            def body(x,xs):
                blk,Kc,Vc=xs;y,_=blk(x,(Kc,Vc),seq_,is_prefix=True);K,V=s3.S3['rec'];return y,(K,V)
            h,(Ks,Vs)=jax.lax.scan(body,h,(blocks_,Kst,Vst))
        finally:s3.S3.clear()
        return h,Ks,Vs
    @eqx.filter_jit
    def read_state_m(m,st,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:return m.language_model.suffix_call(prefix_outputs=pref_,state=st,seq=c).new_state
        finally:s3.S3SUF.clear()
    base_of=lambda k:(k*CH-WIN+np.arange(WIN+CH))>=0
    orig=ISO.Iso55.__init__
    def init(self,*a,**k):
        orig(self,*a,**k);rt=self.rt;nat_record=self.record;self._rk=[0];self._fset='all';orun=self.run_stream
        def record_D(ids,progress=None):
            Kn,Vn,_=nat_record(ids);ch=self.chunks_meta;nch=len(ids)//CH;Kh=np.zeros_like(Kn);Vh=np.zeros_like(Vn);PD={}
            for kk in range(nch):
                Kc,Vc=self.ctx_kv(Kh,Vh,kk);ids_k=jnp.asarray(np.asarray(ids[kk*CH:(kk+1)*CH]),dtype=jnp.int32);h,Ks,Vs=prefix_rec(rt.base,ids_k,Kc,Vc,jnp.int32(kk*CH),jnp.asarray(keep_Dset(ch,kk,base_of(kk),self._fset)))
                Kh[:,kk*CH:(kk+1)*CH]=np.asarray(jax.device_get(Ks)).reshape(Kh[:,kk*CH:(kk+1)*CH].shape);Vh[:,kk*CH:(kk+1)*CH]=np.asarray(jax.device_get(Vs)).reshape(Vh[:,kk*CH:(kk+1)*CH].shape);PD[kk]=h
            if CHK is not None:
                f32=lambda x:np.asarray(x,np.float32)
                CHK['K_scale']=float(np.abs(f32(Kn[:,:4*CH])).max());CHK['chunk0_K_maxabs_D_vs_natural_layer0']=float(np.abs(f32(Kh[0,:CH])-f32(Kn[0,:CH])).max())
            self._PD=PD;self._rk[0]=0;log(f'  {self._arm}: carried prefix K,V recorded with the D rule, fact set {self._fset!r}, over {nch} chunks');return Kh,Vh,None
        def read_state_D(m,st,c,pref_):
            kk=self._rk[0];self._rk[0]+=1;return read_state_m(m,st,c,self._PD[kk],jnp.asarray(keep_Dset(self.chunks_meta,kk,base_of(kk),self._fset)))
        def run_w(*aa,**kk):
            self._arm=kk['tag'].split('_',1)[1];self._fset=FSETS[self._arm];self.rec_cache=None      # the record cache of e59_kvmulti.run is shared across arms: each arm records its own carried window
            return orun(*aa,**kk)
        self.record=record_D;self.read_state=read_state_D;self.run_stream=run_w
    ISO.Iso55.__init__=init
def selftest():
    chunks=[{'sents':[dict(fact=0,start=10,end=20),dict(fact=20,start=50,end=60)]} for _ in range(12)]
    D=keep_Drec(chunks,0,(0*CH-WIN+np.arange(WIN+CH))>=0);assert not D[0:10][:,WIN+10:WIN+20].any() and D[10:20][:,WIN+10:WIN+20].all()
    base=np.ones(WIN+CH,bool);ch3=[{'sents':[dict(fact=0,start=10,end=20),dict(fact=20,start=50,end=60)]} for _ in range(12)];k=3;off=lambda c:(c-k+NW)*CH
    assert (keep_Dset(ch3,k,base,'all')==keep_Drec(ch3,k,base)).all()
    R=keep_Dset(ch3,k,base,'read');assert not R[:,off(0)+10:off(0)+20].any() and R[:,off(0)+50:off(0)+60].all() and not R[0:10][:,WIN+10:WIN+20].any() and R[0:10][:,WIN+50:WIN+60].all() and R[10:20][:,WIN+10:WIN+20].all()
    X=keep_Dset(ch3,k,base,'extra');assert X[:,off(0)+10:off(0)+20].all() and not X[:,off(0)+50:off(0)+60].any() and not X[0:50][:,WIN+50:WIN+60].any() and X[50:60][:,WIN+50:WIN+60].all()
    print('e59_p1d selftest ok: keep_Drec (from e59_relay), fact-set variants all / read / extra')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--nch',type=int,default=128);ap_.add_argument('--expo',default='8,16,32,64,96,128');ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--check',action='store_true');ap_.add_argument('--arms',default='P1d');ap_.add_argument('--base-dir',default='experiments/exp07_write_ladder/kvmulti');a=ap_.parse_args()
    if a.selftest:selftest()
    else:
        import e59_kvmulti as KV
        out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);CHK={};install(lambda *x:print(*x,flush=True),CHK);KV.ARMS={x:'C0' for x in a.arms.split(',')}
        KV.run(argparse.Namespace(seed=a.seed,out=str(out),nch=a.nch,expo=a.expo,save_final=True,selftest=False))
        r=json.load(open(out/f'kvmulti_s{a.seed}.json'));b=json.load(open(pathlib.Path(a.base_dir)/f'kvmulti_s{a.seed}.json'));sb=b['arms']['P1s']['steps']
        for arm in a.arms.split(','):
            s=r['arms'][arm]['steps'];CHK[arm]=dict(chunk0_gnorm=s[0]['gnorm'],chunk0_gnorm_P1s_1348=sb[0]['gnorm'],chunk0_ce_absdiff=abs(s[0]['ce']-sb[0]['ce']),chunk1_gnorm_absdiff=abs(s[1]['gnorm']-sb[1]['gnorm']))
        json.dump(CHK,open(out/f'p1d_checks_s{a.seed}.json','w'));print('P1D CHECKS',CHK,flush=True)
        if a.check:
            for arm in a.arms.split(','):
                c_=CHK[arm];assert abs(c_['chunk0_gnorm']-c_['chunk0_gnorm_P1s_1348'])<=1e-5*c_['chunk0_gnorm_P1s_1348'] and c_['chunk0_ce_absdiff']<1e-5,('chunk-0 write differs from P1s',arm,c_)
                assert c_['chunk1_gnorm_absdiff']>0,('chunk-1 write identical to P1s: the carried window did not change',arm,c_)
            assert CHK['chunk0_K_maxabs_D_vs_natural_layer0']<=0.02*CHK['K_scale'],('layer-0 K of chunk 0 differs beyond bf16 rounding',CHK)
