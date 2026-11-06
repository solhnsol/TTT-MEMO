"""E57 online g128 rule for the QA runner (new file; job 1160): the same idea as UVOnline / install_online_dec of e57_syn_mrun.py (job 1150) for streams in which the band-128 mask (ALLQ128) is applied to every row from chunk 0 (QA): keep() returns the ALLQ128 keep and stashes the prefix outputs of the NATURAL window; the wrapper of L.grad_mask (outermost, after install_weighted_grad) computes g128 = nll_band128 - nll_win with the CURRENT fast weights (the band-128 forward is the training forward itself: pref_ / kmask of the write step), takes the eligible rows (rows_eligible(lm, 0) & lm, as in e57_pqa16_run.deciding_masks) with g > tau and publishes them in holder['dec'] before the weighted gradient reads it.
  selftest: python3 e57_online_uv.py --selftest"""
import sys,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
CH=1024;WIN=8*CH;BAND=128
class UVOnlineBand:
    def __init__(self,holder,log=print,band=BAND):self.holder=holder;self.log=log;self.band=band
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        import jax.numpy as jnp
        base=np.asarray(keep_all,bool);self.holder['dec']=np.zeros(CH,np.float32);L=self.holder['L']
        self.holder['stash']=dict(pn=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(base)),nat=jnp.asarray(base),k=k)
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();kp[:]=CM.band_keep(base,self.band)
        return kp,{'uv_arm':'M3OA2','q_banded':CH,'q_band':self.band,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def install_online_dec_band(L,s3,holder,tau,log=print):
    import jax,jax.numpy as jnp,equinox as eqx
    S3SUF=s3.S3SUF;inner=L.grad_mask
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    def gm(m,st_w,c,pref_,kmask,*rest,**kw):
        S=holder.get('stash')
        if S is not None:
            nw=-np.asarray(logp_mask(m,st_w,c,S['pn'],S['nat']),np.float64);nb=-np.asarray(logp_mask(m,st_w,c,pref_,kmask),np.float64);g=nb-nw;lm=np.asarray(c.loss_masks,bool)
            el=CM.rows_eligible(lm,np.zeros(CH,np.int32))&lm;dec=(el&(g>tau)).astype(np.float32);holder['dec']=dec;holder.setdefault('shares',[]).append(float(dec.sum()/max(1,el.sum())));holder.setdefault('g_mean',[]).append(float(g[el].mean()))
            if S['k']%16==0:log(f"  online ABS tau {tau} chunk {S['k']}: selected rows {int(dec.sum())} ({holder['shares'][-1]:.3f} of eligible), mean g {holder['g_mean'][-1]:+.3f}")
        return inner(m,st_w,c,pref_,kmask,*rest,**kw)
    L.grad_mask=gm
def selftest():
    h={};u=UVOnlineBand(h);assert u.band==128;print('e57_online_uv selftest ok')
if __name__=='__main__':
    selftest() if '--selftest' in sys.argv else print(__doc__)
