"""Exp09 inner rule on LoRA fast weights: one plain SGD step per chunk, mean token NTP loss over the chunk's own
tokens, global-norm clip (optax.clip_by_global_norm semantics: coef = 1 if ||g|| < clip else clip/||g||), step
lr * scale * coef * g. Ways to build the chunk's loss:
  native detached (default, TTT-E2E structure; class DetachedWriter): chunk k is ONE forward of its own tokens with the
            carried KV of the previous <= `win` chunks as past_key_values. The carried KV is detached (no gradient into
            it) and was computed with the weights in force when each chunk was written (pre-update weights of that
            chunk's step). After the step, chunk k's KV from that same forward is appended and entries beyond `win`
            chunks are dropped. Window positions stay in [0, (win+1)*CH): the oldest cached chunk always starts at
            position 0 and chunk k sits at the end, so when the window slides the cached (post-RoPE) keys are
            re-rotated by the position shift (RoPE angle shift), always from the key as originally computed (one
            rotation per use, no accumulated rounding). Values carry no position and are not touched.
  native recompute (--native-ctx recompute): chunk k with the previous `win` chunks of the stream as context,
            recomputed with the current weights each step, no KV cache; gradient flows through the whole forward.
  Both: loss on chunk k's positions only (the last position predicts the next chunk's first token, as in the JAX stream).
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
    return sgd_step_fn(model,lambda:loss_of(model,batch),lr,scale,clip)

def sgd_step_fn(model,loss_fn,lr,scale=1.0,clip=1.0):
    ps=[p for _,p in lora_params(model)]
    loss=loss_fn()
    g=torch.autograd.grad(loss,ps)
    gn=torch.sqrt(sum((x.float()**2).sum() for x in g))
    coef=torch.where(gn<clip,torch.ones_like(gn),clip/torch.clamp(gn,min=1e-30))
    s=lr*scale*coef
    un=torch.zeros((),device=gn.device)
    with torch.no_grad():
        for p,x in zip(ps,g):
            d=(s*x).to(p.dtype);p.sub_(d);un+=(d.float()**2).sum()
    return {'ce':float(loss.detach()),'grad_norm':float(gn),'clip_coef':float(coef),'update_norm':float(un.sqrt())}

# ---------------- native write, TTT-E2E structure: detached rolling KV cache ----------------
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import rotate_half

class DetachedWriter:
    """rolling, detached KV of the last `win` chunks; see the module docstring. One instance per pass (reset())."""
    def __init__(self,model,CH,win,device):
        self.model=model;self.CH=CH;self.win=win;self.device=device;self.chunks=[]
        self.rot=parts(model)[0].rotary_emb
        assert float(getattr(self.rot,'attention_scaling',1.0))==1.0,'re-rotation assumes plain RoPE (attention_scaling 1)'
    def reset(self):self.chunks=[]
    def rotate(self,k,shift):
        """RoPE angle shift of post-RoPE keys k [B,H,L,D] by `shift` positions (fp32 math, cast back)."""
        if shift==0:return k
        pid=torch.tensor([[shift]],device=k.device)
        cos,sin=self.rot(k.float(),pid)          # [1,1,D]
        cos=cos[:,None].float();sin=sin[:,None].float();kf=k.float()
        return (kf*cos+rotate_half(kf)*sin).to(k.dtype)
    def past(self):
        """DynamicCache of the cached chunks at window positions [0, n*CH) (oldest first) and the offset n*CH."""
        if not self.chunks:return DynamicCache(),0
        nl=len(self.chunks[0]['kv']);data=[]
        rk=[[self.rotate(c['kv'][l][0],i*self.CH-c['pos']) for i,c in enumerate(self.chunks)] for l in range(nl)]
        for l in range(nl):
            data.append((torch.cat(rk[l],dim=2),torch.cat([c['kv'][l][1] for c in self.chunks],dim=2)))
        return DynamicCache(ddp_cache_data=data),sum(c['kv'][0][0].shape[2] for c in self.chunks)
    def forward(self,ids):
        """one forward of chunk ids (1-D long) after the cached window. Returns (hidden [L,d], new kv, its position)."""
        dec,_=parts(self.model);cache,off=self.past();L=ids.shape[0]
        pos=(off+torch.arange(L,device=ids.device))[None]
        hs=dec(input_ids=ids[None],position_ids=pos,past_key_values=cache,use_cache=True).last_hidden_state[0]
        kv=[(l.keys[:,:,-L:].detach(),l.values[:,:,-L:].detach()) for l in cache.layers]
        return hs,kv,off
    def push(self,kv,pos):
        self.chunks.append({'kv':kv,'pos':pos})
        while len(self.chunks)>self.win:self.chunks.pop(0)
    def step(self,stream,k,lr,scale=1.0,clip=1.0):
        """inner step on chunk k of `stream` (python list), then append chunk k's KV (pre-update weights). Returns stats
        (None if the chunk has no target)."""
        CH=self.CH;n=len(stream);c0=k*CH;e=min((k+1)*CH,n);last=min(e,n-1)
        ids=torch.tensor(stream[c0:e],dtype=torch.long,device=self.device)
        tgt=torch.tensor(stream[c0+1:last+1],dtype=torch.long,device=self.device);box={}
        def lf():
            hs,kv,off=self.forward(ids);box['kv']=kv;box['pos']=off
            return F.cross_entropy(parts(self.model)[1](hs[:last-c0]).float(),tgt)
        st=sgd_step_fn(self.model,lf,lr,scale,clip) if last>c0 else None
        if 'kv' not in box:
            with torch.no_grad():_,box['kv'],box['pos']=self.forward(ids)
        self.push(box['kv'],box['pos']);return st
