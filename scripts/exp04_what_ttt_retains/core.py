"""Shared native TTT-E2E 3B streaming primitives for Exp04 (no edits to e2e/ or earlier code)."""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(ROOT/'scripts/exp02a4_ttt_e2e_binding'))
import run_matched_lexical as reuse
jax=reuse.jax;jnp=reuse.jnp;eqx=reuse.eqx;np=reuse.np
Batch=reuse.Batch;MetaModel=reuse.MetaModel;clone_state=reuse.clone_state
CH=1024

def retrying(fn,tries=40,wait=30):
    """The A100 is shared with other tenants whose usage fluctuates; calls are functional (inputs untouched on failure),
    so on RESOURCE_EXHAUSTED wait and retry the identical call instead of losing the stream."""
    def w(*a,**k):
        for i in range(tries):
            try:return fn(*a,**k)
            except Exception as e:
                if 'RESOURCE_EXHAUSTED' not in str(e) or i==tries-1:raise
                print('OOM retry',i,flush=True);time.sleep(wait)
    return w

class Runtime:
    def __init__(self):
        self.cfg,self.mesh,self.base,self.state_all,self.state_prefix,self.state_suffix=reuse.load()
        cfg=self.cfg;self.eos=int(cfg.model.eos_token_id);self.bos=int(cfg.model.bos_token_id)
        assert int(cfg.model.sliding_window_size)==8192 and int(cfg.model.mini_batch_size)==CH
        self.tok=reuse.load_tokenizer()
        f=reuse.get_filter_spec(self.base,cfg.training.spec_inner,'inner parameters')
        self.fast_hash=reuse._hash_pytree(self.base.inner_parameters())[0]
        self.frozen_hash=reuse._hash_pytree(eqx.partition(self.base,f)[1])[0]
        state_prefix=self.state_prefix
        @eqx.filter_jit
        def prefix(m,seq):
            return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(seq.input_ids),clone_state(state_prefix),seq).last_hidden_state
        @eqx.filter_jit
        def update(m,opt,state_tuple,seq,pref):return MetaModel.inner_loop_step(m,opt,state_tuple,seq,pref)
        @eqx.filter_jit
        def chunk_logp(m,st,seq,pref):
            """per-token logp of the chunk's targets under weights m and suffix KV st; returns (new_state, logp[CH])"""
            result=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=seq)
            lp=jax.nn.log_softmax(result.logits.astype(jnp.float32),axis=-1)
            return result.new_state,jnp.take_along_axis(lp,seq.target_tokens[:,None],axis=-1)[:,0]
        self.prefix=retrying(prefix);self.update=retrying(update);self.chunk_logp=retrying(chunk_logp)

    def seq(self,ids,mask=None,start=0):
        ids=jnp.asarray(ids,dtype=jnp.int32)
        targets=jnp.concatenate([ids[1:],jnp.asarray([self.eos],dtype=jnp.int32)])
        if mask is None:mask=jnp.ones(len(ids),dtype=bool)
        return Batch(input_ids=ids,target_tokens=targets,loss_masks=jnp.asarray(mask,dtype=bool))

    def chunk(self,seq,k):
        s=slice(k*CH,(k+1)*CH);return Batch(input_ids=seq.input_ids[s],target_tokens=seq.target_tokens[s],loss_masks=seq.loss_masks[s])

    def pad(self,ids):
        n=(len(ids)+CH-1)//CH*CH;return list(ids)+[self.eos]*(n-len(ids))

    def stream(self,ids,on,hooks=None,loss_mask=None,collect=True,update_mask=None):
        """Stream ids (len multiple of CH after padding) chunk by chunk.
        Returns per-token pre-update logp [n]. ON: native inner update after each chunk (NLL measured BEFORE
        the chunk's own update). OFF: identical forward and KV carry, no update. hooks[k](rt,m,suffix_state)
        is called after chunk k has been consumed (post-update for ON) for readouts."""
        with self.mesh:return self._stream(ids,on,hooks,loss_mask,collect,update_mask)

    def _stream(self,ids,on,hooks,loss_mask,collect,update_mask=None):
        n=len(ids);padded=self.pad(ids);nch=len(padded)//CH
        mask=np.zeros(len(padded),dtype=bool);mask[:n-1]=True
        if loss_mask is not None:mask[:n-1]&=np.asarray(loss_mask[:n-1],dtype=bool)
        seq=self.seq(padded,mask=mask);pref=self.prefix(self.base,seq);jax.block_until_ready(pref)
        m=self.base;lps=[]
        if on:
            opt=m.inner_optimizer(clone_state(self.state_all)).init(m.inner_parameters());state_tuple=(clone_state(self.state_all),clone_state(self.state_suffix))
            st=state_tuple[1]
        else:st=clone_state(self.state_suffix)
        for k in range(nch):
            c=self.chunk(seq,k);p=pref[k*CH:(k+1)*CH]
            if on and update_mask is not None and not update_mask[k]:
                # forward-only chunk inside an ON stream: weights frozen, native KV carry continues (no-update control)
                new_st,lp=self.chunk_logp(m,state_tuple[1],c,p);state_tuple=(state_tuple[0],new_st);st=new_st
            elif on:
                _,lp=self.chunk_logp(m,state_tuple[1],c,p)
                r=self.update(m,opt,state_tuple,c,p);jax.block_until_ready(r.metrics)
                m,opt,state_tuple=r.new_model,r.new_optimizer_state,r.new_state;st=state_tuple[1];del r
            else:
                st,lp=self.chunk_logp(m,st,c,p)
            if collect:lps.append(np.asarray(lp))
            if hooks and k in hooks:hooks[k](self,m,st)
        del pref
        return (np.concatenate(lps)[:n] if collect else None),m

    def score_probe(self,m,st,ids,spans):
        """Teacher-forced logp of token spans inside a fresh final chunk run from carried suffix KV st.
        ids: list (<=CH). spans: list of (a,b) token index ranges (b exclusive) to score. No update. Prefix
        is recomputed over the short probe chunk only (probes are local-context short prompts)."""
        padded=ids+[self.eos]*(CH-len(ids));seq=self.seq(padded)
        with self.mesh:
            pref=self.prefix(m,seq)
            _,lp=self.chunk_logp(m,clone_state(st),seq,pref)
        lp=np.asarray(lp);return [float(lp[a-1:b-1].sum()) for a,b in spans]

def _score_probe_full(self,m,st,full_ids,spans):
    """full_ids = exactly 8192 tokens: preceding stream tail (7168) + final probe chunk (1024). The frozen prefix is
    recomputed causally over the 8192 window; the suffix reads the carried native KV state st, then scores the
    final chunk. No update. Returns summed logp of each (a,b) span (absolute indices within full_ids)."""
    assert len(full_ids)==8192
    with self.mesh:
        seq=self.seq(full_ids);pref=self.prefix(m,seq)
        last=Batch(input_ids=seq.input_ids[-CH:],target_tokens=seq.target_tokens[-CH:],loss_masks=seq.loss_masks[-CH:])
        _,lp=self.chunk_logp(m,clone_state(st),last,pref[-CH:])
    lp=np.asarray(lp);base=8192-CH
    return [float(lp[a-1-base:b-1-base].sum()) for a,b in spans]
Runtime.score_probe_full=_score_probe_full
