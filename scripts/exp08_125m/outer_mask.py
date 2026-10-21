"""Outer-loop masking helpers (pure numpy, no jax / ttt import; used by outer_smoke.py and outer_mask_test.py).
Convention of the official data path (e2e ttt/dataloader/lm_dataset.py _to_batch): a sample is tokens[0 .. T] (T+1 tokens); input_ids = tokens[:-1], target_tokens = tokens[1:], loss_masks = (tokens[1:] != bos).
So loss_masks[j] belongs to the target tokens[j+1]: excluding the loss of the token at sample position p (p >= 1) means loss_masks[p-1] = False. Position 0 is never a target.
The same loss_masks feed the inner (write) loss of every 1024-token chunk AND the outer loss (loss_for_sequence = mean over chunks of the masked CE), so one mask removes the position from the write gradient and from the meta objective.
cross_entropy_loss_and_accuracy (ttt/model/loss.py) divides the summed masked NLL of a chunk by the number of valid targets of that chunk (clipped to 1e-10): a masked-out position changes the denominator, an all-masked chunk gives loss 0."""
import dataclasses
import numpy as np
def apply_target_mask(loss_masks,positions):
    """loss_masks: bool array [T] (target index j <-> token position j+1). positions: iterable of SAMPLE token positions whose loss is excluded. Returns a new bool array; position 0 and out-of-range positions are ignored (not targets)."""
    m=np.array(loss_masks,dtype=bool,copy=True);T=len(m)
    for p in positions:
        if 1<=p<=T:m[p-1]=False
    return m
def random_position_mask(tokens,frac,seed):
    """loss_masks for a sample of T+1 tokens with a fraction `frac` of the targets removed uniformly at random (exact count round(frac*T) among the targets whose default mask is True); deterministic in (tokens, seed)."""
    tokens=np.asarray(tokens)   # the caller ANDs the result with the bos rule
    T=len(tokens)-1;rng=np.random.default_rng([int(seed),int(np.sum(tokens[:16].astype(np.int64)))%(2**31)]);n=int(round(frac*T))
    out=np.ones(T,bool);out[rng.choice(T,size=n,replace=False)]=False;return out
def make_patched_to_batch(orig,frac,seed=0):
    """wraps the official lm_dataset._to_batch: loss_masks = default AND random mask of the given fraction (frac == 0: returns orig unchanged). The Batch is an equinox Module (a dataclass): dataclasses.replace builds the new one."""
    if frac<=0:return orig
    def _to_batch(data,*,bos_token_id,eos_token_id):
        b=orig(data,bos_token_id=bos_token_id,eos_token_id=eos_token_id)
        return dataclasses.replace(b,loss_masks=np.asarray(b.loss_masks)&random_position_mask(np.asarray(data),frac,seed))
    return _to_batch
def chunk_valid_counts(loss_masks,chunk=1024):
    m=np.asarray(loss_masks,bool);assert len(m)%chunk==0;return m.reshape(-1,chunk).sum(1)
def reference_chunk_loss(logits,targets,valid):
    """numpy re-implementation of cross_entropy_loss_and_accuracy for one chunk: sum of masked NLL / max(#valid, 1e-10)."""
    lg=np.asarray(logits,np.float64);lg=lg-lg.max(-1,keepdims=True);lp=lg-np.log(np.exp(lg).sum(-1,keepdims=True))
    tl=-lp[np.arange(len(targets)),targets];v=np.asarray(valid,np.float64);return float((tl*v).sum()/max(v.sum(),1e-10))
