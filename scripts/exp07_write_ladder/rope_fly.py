"""RoPE without the position table (E57 / E55 long streams). Upstream AttentionBase.apply_rope looks positions up in a table of 2*seq_len entries built as a compile-time constant of every jitted function: positions >= 2*seq_len (a 512-chunk stream has 524288) index out of range and every compiled function carries the table as a constant. This computes cos / sin of position x frequency at the requested positions with the SAME float32 arithmetic as precompute_freqs_cis (outer product, then sin / cos, then complex64), so the values equal the table entries inside the table, and are defined beyond it. Install once after the e2e attention module (and packing.py, which wraps apply_rope) is imported: install() replaces the original function that packing.py wraps, or apply_rope itself when packing is not loaded. ROPE_TABLE=1 in the environment keeps the upstream table (install() does nothing)."""
import os,sys
UPSTREAM=None      # the table version (for tests)
def _fly_factory(A):
    import jax,jax.numpy as jnp
    def apply_rope(self,xis,position_ids):
        dim=self.head_dim
        freqs=1.0/(self.config.rope_theta**(jnp.arange(0,dim,2)[:(dim//2)].astype(jnp.float32)/dim))
        ang=jnp.outer(jnp.asarray(position_ids),freqs).astype(jnp.float32)
        fc=jnp.complex64(jnp.cos(ang)+1j*jnp.sin(ang))
        fn=A.maybe_double_remat(A.apply_rotary_emb,prevent_cse=True,policy_remat=self.config.remat_rms,policy_remat_bwd=self.config.remat_rms_bwd)
        return jax.tree.map(lambda x:fn(x,fc),xis)
    apply_rope._rope_fly=True
    return apply_rope
def install():
    """-> True when patched now, False when already patched or disabled"""
    if os.environ.get('ROPE_TABLE'):return False
    from ttt.model import attention as A
    global UPSTREAM
    fly=_fly_factory(A);pk=sys.modules.get('packing')
    if pk is not None and hasattr(pk,'_orig_rope'):
        if getattr(pk._orig_rope,'_rope_fly',False):return False
        UPSTREAM=pk._orig_rope;pk._orig_rope=fly;return True
    if getattr(A.AttentionBase.apply_rope,'_rope_fly',False):return False
    UPSTREAM=A.AttentionBase.apply_rope;A.AttentionBase.apply_rope=fly;return True
