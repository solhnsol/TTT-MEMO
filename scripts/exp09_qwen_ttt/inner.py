"""Exp09 inner rule on LoRA fast weights: one plain SGD step per chunk, mean token NTP loss over the chunk's own
tokens, global-norm clip (optax.clip_by_global_norm semantics: coef = 1 if ||g|| < clip else clip/||g||), step
lr * scale * coef * g. Three ways to build the chunk's loss:
  native  : chunk k with the previous `win` chunks of the stream as context (sliding window, recomputed each step, no KV
            cache); loss on chunk k's positions only (the last position predicts the next chunk's first token, as in the
            JAX stream); gradient flows through the whole forward, context positions included.
  masked  : chunk k alone, positions from 0; loss on targets inside the chunk (the last position has none).
  packed  : isolated rows (L1) packed greedily into <= CH-token steps; per-row position restart, which makes
            transformers' Qwen3 mask block-diagonal (masking_utils.find_packed_sequence_indices; checked by
            test_l1_packing.py); loss on each row's (a,b) positions, mean over all of them.
Logits are formed only at loss positions (lm_head on selected hidden states), in fp32."""
import torch,torch.nn.functional as F
from qwen_runtime import parts,lora_params

def _ce(model,ids,pos,sel,tgt):
    """mean CE of logits at positions `sel` (1-D long) against `tgt`; ids/pos: 1-D long tensors (one row)."""
    dec,head=parts(model)
    hs=dec(input_ids=ids[None],position_ids=pos[None],use_cache=False).last_hidden_state[0]
    lg=head(hs[sel]).float()
    return F.cross_entropy(lg,tgt,reduction='mean')

def native_batch(stream,k,CH,win,device):
    """(ids, pos, sel, tgt) for chunk k of `stream` (python list) with `win` previous chunks as context."""
    n=len(stream);s=max(0,(k-win)*CH);c0=k*CH;e=min((k+1)*CH,n);assert c0<n
    ids=torch.tensor(stream[s:e],dtype=torch.long,device=device);pos=torch.arange(e-s,device=device)
    last=min(e,n-1)   # positions c0..last-1 have a target in the stream
    sel=torch.arange(c0-s,last-s,device=device);tgt=torch.tensor(stream[c0+1:last+1],dtype=torch.long,device=device)
    return ids,pos,sel,tgt

def masked_batch(stream,k,CH,device):
    n=len(stream);c0=k*CH;e=min((k+1)*CH,n)
    ids=torch.tensor(stream[c0:e],dtype=torch.long,device=device);pos=torch.arange(e-c0,device=device)
    sel=torch.arange(0,e-c0-1,device=device);tgt=ids[1:].clone()
    return ids,pos,sel,tgt

def pack_rows(rows,CH):
    """greedy first-fit in order (exp05 packing.pack_rows). rows: [(ids,(a,b))]. Returns per step (ids, pos, sel, tgt)
    as python lists; sel = positions whose next token (inside the same row) is a target."""
    steps=[];cur=[];pos=[];sel=[];tgt=[]
    def flush():steps.append((cur,pos,sel,tgt))
    for ids,(a,b) in rows:
        assert len(ids)<=CH,(len(ids),CH)
        if cur and len(cur)+len(ids)>CH:flush();cur,pos,sel,tgt=[],[],[],[]
        off=len(cur);cur=cur+list(ids);pos=pos+list(range(len(ids)))
        sel=sel+[off+i for i in range(a,b)];tgt=tgt+[ids[i+1] for i in range(a,b)]
    if cur:flush()
    return steps

def packed_batch(step,device):
    ids,pos,sel,tgt=step;T=lambda x:torch.tensor(x,dtype=torch.long,device=device)
    return T(ids),T(pos),T(sel),T(tgt)

def loss_of(model,batch):return _ce(model,*batch)

def sgd_step(model,batch,lr,scale=1.0,clip=1.0):
    """one inner step in place on the LoRA params; returns stats (loss before the step, grad norm, clip coef, update norm)."""
    ps=[p for _,p in lora_params(model)]
    loss=loss_of(model,batch)
    g=torch.autograd.grad(loss,ps)
    gn=torch.sqrt(sum((x.float()**2).sum() for x in g))
    coef=torch.where(gn<clip,torch.ones_like(gn),clip/torch.clamp(gn,min=1e-30))
    s=lr*scale*coef
    un=torch.zeros((),device=gn.device)
    with torch.no_grad():
        for p,x in zip(ps,g):
            d=(s*x).to(p.dtype);p.sub_(d);un+=(d.float()**2).sum()
    return {'ce':float(loss.detach()),'grad_norm':float(gn),'clip_coef':float(coef),'update_norm':float(un.sqrt())}
