"""CPU unit test of the loss-mask hook for outer-loop masking (E2E official loss.py on CPU jax + numpy references). Usage: outer_mask_test.py [--e2e DIR]  (DIR = a checkout of the official e2e repo, default <repo>/e2e or $E2E_ROOT).
Checks: (1) shift convention (position p <-> loss_masks[p-1]); (2) the official cross_entropy_loss_and_accuracy equals the numpy reference with the mask, denominator = #valid; (3) the gradient w.r.t. the logits is EXACTLY zero at masked positions and equals the unmasked-reference gradient rescaled by 1/#valid elsewhere (this is the write gradient the inner step differentiates); (4) all-masked chunk: loss 0, gradient 0, no NaN; (5) token_log_probs / token-wise NLL is not mask-dependent (the masked positions still appear in the NLL metric); (6) the patched _to_batch keeps the bos rule and removes exactly round(frac*T) more targets, deterministic; (7) the outer loss (mean of chunk losses) changes when a mask changes the denominator.
NOT tested here: the full MetaModel inner loop / cuDNN prefix (needs the GPU server): the smoke job 150 runs the real model with MASK_FRAC > 0."""
import os,sys,argparse,importlib.util,types
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
import outer_mask as OM
def load_loss(e2e):
    p=Path(e2e)/'ttt/model/loss.py';assert p.exists(),f'official loss.py not found: {p}'
    spec=importlib.util.spec_from_file_location('ttt_loss',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--e2e',default=os.environ.get('E2E_ROOT',str(ROOT/'e2e')));a=ap.parse_args()
    import jax,jax.numpy as jnp
    L=load_loss(a.e2e);rng=np.random.default_rng(0);V=11;T=16
    # (1) shift
    m=OM.apply_target_mask(np.ones(T,bool),[0,1,5,T,T+3]);assert (~m).nonzero()[0].tolist()==[0,4,T-1],(~m).nonzero()[0]   # p=1 -> j=0, p=5 -> j=4, p=T -> j=T-1; p=0 and p>T ignored
    # (2) loss vs numpy reference
    logits=rng.normal(size=(T,V)).astype(np.float32);tg=rng.integers(0,V,T);valid=OM.apply_target_mask(np.ones(T,bool),[2,3,7,8,9])
    lo,_=L.cross_entropy_loss_and_accuracy(jnp.asarray(logits),jnp.asarray(tg),jnp.asarray(valid));ref=OM.reference_chunk_loss(logits,tg,valid);assert abs(float(lo)-ref)<1e-5,(float(lo),ref)
    # (3) gradient wrt logits
    g=np.asarray(jax.grad(lambda x:L.cross_entropy_loss_and_accuracy(x,jnp.asarray(tg),jnp.asarray(valid))[0])(jnp.asarray(logits)))
    assert np.all(g[~valid]==0.0),'masked positions must get exactly zero gradient'
    p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);oh=np.eye(V)[tg];gref=(p-oh)/valid.sum();assert np.abs(g[valid]-gref[valid]).max()<1e-6,np.abs(g[valid]-gref[valid]).max()
    # (4) all masked
    z=np.zeros(T,bool);lz,_=L.cross_entropy_loss_and_accuracy(jnp.asarray(logits),jnp.asarray(tg),jnp.asarray(z));gz=np.asarray(jax.grad(lambda x:L.cross_entropy_loss_and_accuracy(x,jnp.asarray(tg),jnp.asarray(z))[0])(jnp.asarray(logits)))
    assert float(lz)==0.0 and np.all(gz==0.0) and np.isfinite(gz).all()
    # (5) token-wise NLL ignores the mask
    t1=-np.asarray(L.token_log_probs(jnp.asarray(logits),jnp.asarray(tg)));assert t1.shape==(T,) and np.all(t1>0)
    # (6) patched _to_batch with a stand-in Batch (dataclass), bos rule kept
    import dataclasses
    @dataclasses.dataclass(frozen=True)
    class B:
        input_ids:object;target_tokens:object;loss_masks:object
    def orig(data,*,bos_token_id,eos_token_id):
        t=np.asarray(data);return B(t[:-1],t[1:],t[1:]!=bos_token_id)
    data=rng.integers(1,50,1025);data[100]=0   # bos = 0 at sample position 100
    b0=orig(data,bos_token_id=0,eos_token_id=1);assert (~b0.loss_masks).sum()==1 and not b0.loss_masks[99]
    P=OM.make_patched_to_batch(orig,0.25,seed=3);b1=P(data,bos_token_id=0,eos_token_id=1);b2=P(data,bos_token_id=0,eos_token_id=1)
    assert np.array_equal(b1.loss_masks,b2.loss_masks) and not b1.loss_masks[99] and (~b1.loss_masks).sum()>=256 and (~b1.loss_masks).sum()<=257,(~b1.loss_masks).sum()
    assert np.array_equal(b1.input_ids,b0.input_ids) and OM.make_patched_to_batch(orig,0.0) is orig
    # (7) outer loss (mean over chunks) depends on the mask through the denominator
    C=8;lg=rng.normal(size=(2*C,V)).astype(np.float32);tt=rng.integers(0,V,2*C);v=np.ones(2*C,bool);v[:3]=False
    ch=lambda mm:np.mean([OM.reference_chunk_loss(lg[i*C:(i+1)*C],tt[i*C:(i+1)*C],mm[i*C:(i+1)*C]) for i in range(2)])
    assert abs(ch(v)-ch(np.ones(2*C,bool)))>1e-6 and OM.chunk_valid_counts(v,C).tolist()==[C-3,C]
    print('outer_mask_test OK (jax',jax.__version__,'CPU; official loss.py from',a.e2e,')')
if __name__=='__main__':main()
