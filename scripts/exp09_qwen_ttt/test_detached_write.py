"""Tests 9a-d: native write with the TTT-E2E structure (inner.DetachedWriter: one forward of chunk k after the detached,
carried KV of the previous 7 chunks, re-rotated to window positions [0, 8*CH)).
(a) frozen weights: rolling cache + re-rotation == explicit concatenation of the 7 previous chunks + chunk k from
    position 0. Exact only for a 1-layer model: with more layers the carried KV of chunk k-7 was computed while it
    attended to chunks before the window (depth receptive field; same in TTT-E2E's carried SWA KV), so a recomputed
    window differs by design. For the 4-layer model the re-rotation is checked against the same rolling cache kept at
    ABSOLUTE positions (no re-rotation; RoPE is relative, so logits must agree).
(b) no gradient flows into the cache; grads equal those with explicitly .detach().clone()d cache tensors.
(c) changing a chunk older than the window: 1-layer model, frozen weights -> chunk k loss and grad unchanged (exact);
    4-layer model: the indirect (depth) dependence is measured and reported (expected non-zero, see (a)).
(d) when weights change between chunks, the cached KV of chunk j equals the KV computed with the weights in force at
    chunk j's step (pre-update), not with the current weights."""
from _testutil import *
import torch.nn.functional as F
import inner as IN
from transformers import DynamicCache

CH=32;WIN=7;NCHK=12
g=torch.Generator().manual_seed(9);stream=torch.randint(0,151000,(NCHK*CH,),generator=g).tolist()
def chunk(st,k):return torch.tensor(st[k*CH:(k+1)*CH])
def model(layers):
    m=QR.tiny_qwen(seed=0,layers=layers);pm,_=QR.wrap_lora(m,16,32,('gate_proj','up_proj','down_proj'),min(2,layers));randomize_lora_(pm,std=0.02);return pm
def roll(pm,st,upto):
    """frozen weights: run the writer forward-only through chunks 0..upto-1, return the writer."""
    Wr=IN.DetachedWriter(pm,CH,WIN,DEV)
    with torch.no_grad():
        for k in range(upto):_,kv,off=Wr.forward(chunk(st,k));Wr.push(kv,off)
    return Wr
def head_logits(pm,hs):return QR.parts(pm)[1](hs).float()
def explicit(pm,st,k):
    x=torch.tensor(st[(k-WIN)*CH:(k+1)*CH])[None]
    with torch.no_grad():return pm(input_ids=x).logits[0,-CH:].float()
def absolute_ref(pm,st,k):
    """rolling cache of the last WIN chunks kept at absolute positions (no re-rotation)."""
    dec,_=QR.parts(pm);ent=[]
    with torch.no_grad():
        for j in range(k+1):
            data=[(torch.cat([e[l][0] for e in ent],2),torch.cat([e[l][1] for e in ent],2)) for l in range(len(ent[0]))] if ent else None
            cache=DynamicCache(ddp_cache_data=data) if data else DynamicCache()
            hs=dec(input_ids=chunk(st,j)[None],position_ids=(j*CH+torch.arange(CH))[None],past_key_values=cache,use_cache=True).last_hidden_state[0]
            ent.append([(l.keys[:,:,-CH:],l.values[:,:,-CH:]) for l in cache.layers]);ent=ent[-WIN:]
    return head_logits(pm,hs)

# ---- (a) ----
for layers in (1,4):
    pm=model(layers)
    for k in (9,11):
        Wr=roll(pm,stream,k);n=len(Wr.chunks)
        with torch.no_grad():hs,_,off=Wr.forward(chunk(stream,k))
        lg=head_logits(pm,hs);de=float((lg-explicit(pm,stream,k)).abs().max());da=float((lg-absolute_ref(pm,stream,k)).abs().max())
        pos_ok=off==WIN*CH and n==WIN and [c['pos'] for c in Wr.chunks]==[min(j,WIN)*CH for j in range(k-WIN,k)]
        if layers==1:check(f'9a 1-layer, chunk {k} (window slid {k-WIN}x): rolling detached KV + re-rotation == explicit concatenation from position 0',de<=1e-4 and pos_ok,f'max abs diff {de:.2e}; chunk k at positions [{off},{off+CH})')
        check(f'9a {layers}-layer, chunk {k}: re-rotated window == same cache at absolute positions (RoPE shift exact)',da<=1e-4 and pos_ok,f'max abs diff {da:.2e}')
        if layers==4:print(f'   info: 4-layer chunk {k}: carried-KV logits vs recomputed-window logits differ by {de:.2e} (depth receptive field, expected)',flush=True)
# ---- (b) ----
pm=model(4);ps=[p for _,p in QR.lora_params(pm)];Wr=IN.DetachedWriter(pm,CH,WIN,DEV)
for k in range(9):Wr.step(stream,k,lr=5e-2)
nograd=all((not t.requires_grad) and t.grad_fn is None for c in Wr.chunks for kv in c['kv'] for t in kv)
k=9;tgt=torch.tensor(stream[k*CH+1:(k+1)*CH+1])
def grad_with(W_):
    hs,_,_=W_.forward(chunk(stream,k));loss=F.cross_entropy(head_logits(pm,hs),tgt);return [x.detach().clone() for x in torch.autograd.grad(loss,ps)],float(loss.detach())
g1,l1=grad_with(Wr)
W2=IN.DetachedWriter(pm,CH,WIN,DEV);W2.chunks=[{'pos':c['pos'],'kv':[(a.detach().clone(),b.detach().clone()) for a,b in c['kv']]} for c in Wr.chunks]
g2,l2=grad_with(W2)
check('9b cached KV tensors carry no graph (requires_grad False, grad_fn None)',nograd,f'{len(Wr.chunks)} chunks x {len(Wr.chunks[0]["kv"])} layers')
check('9b chunk-k grads identical with explicitly .detach().clone()d cache; grads non-zero',all(torch.equal(a,b) for a,b in zip(g1,g2)) and l1==l2 and max(float(x.abs().max()) for x in g1)>0)
# the cache past() hands to the forward is also graph-free
cache,_=Wr.past();check('9b the DynamicCache built for the forward holds graph-free tensors',all(not l.keys.requires_grad and not l.values.requires_grad for l in cache.layers))
# ---- (c) ----
for layers in (1,4):
    pm=model(layers);ps=[p for _,p in QR.lora_params(pm)];k=10
    st2=list(stream)
    for i in range((k-WIN-1)*CH,(k-WIN)*CH):st2[i]=(st2[i]*7+13)%151000    # chunk k-8: just outside the window
    res=[]
    for st_ in (stream,st2):
        Wr=roll(pm,st_,k);hs,_,_=Wr.forward(chunk(st_,k));loss=F.cross_entropy(head_logits(pm,hs),torch.tensor(st_[k*CH+1:(k+1)*CH+1]))
        res.append((float(loss.detach()),[x.detach() for x in torch.autograd.grad(loss,ps)]))
    dl=abs(res[0][0]-res[1][0]);dg=max(float((a-b).abs().max()) for a,b in zip(res[0][1],res[1][1]))
    if layers==1:check('9c 1-layer, frozen weights: changing chunk k-8 (outside the window) leaves chunk k loss and grad unchanged',dl==0 and dg==0,f'|dloss| {dl:.1e} |dgrad| {dg:.1e}')
    else:check('9c 4-layer: chunk k-8 reaches chunk k only through the carried deep-layer KV of chunk k-7 (expected, reported)',dl>0,f'|dloss| {dl:.2e} |dgrad| {dg:.2e}')
# ---- (d) ----
pm=model(4);Wr=IN.DetachedWriter(pm,CH,WIN,DEV);snaps=[]
for k in range(4):snaps.append(QR.lora_state(pm));Wr.step(stream,k,lr=0.5)
final=QR.lora_state(pm);dec,_=QR.parts(pm);ent=[];md=0.0
with torch.no_grad():
    for j in range(4):
        QR.load_lora_state_(pm,snaps[j])
        data=[(torch.cat([e[l][0] for e in ent],2),torch.cat([e[l][1] for e in ent],2)) for l in range(len(ent[0]))] if ent else None
        cache=DynamicCache(ddp_cache_data=data) if data else DynamicCache()
        dec(input_ids=chunk(stream,j)[None],position_ids=(j*CH+torch.arange(CH))[None],past_key_values=cache,use_cache=True)
        ent.append([(l.keys[:,:,-CH:],l.values[:,:,-CH:]) for l in cache.layers])
        md=max(md,max(float((a-c).abs().max()) for (a,b),(c,d_) in zip(Wr.chunks[j]['kv'],ent[j])),max(float((b-d_).abs().max()) for (a,b),(c,d_) in zip(Wr.chunks[j]['kv'],ent[j])))
    QR.load_lora_state_(pm,final);cache=DynamicCache()
    dec(input_ids=chunk(stream,0)[None],position_ids=torch.arange(CH)[None],past_key_values=cache,use_cache=True)
    dnow=float((cache.layers[-1].values-Wr.chunks[0]['kv'][-1][1]).abs().max())
wmoved=max(float((final[n]-snaps[0][n]).abs().max()) for n in final)
check('9d cached KV of chunks 0-3 == KV computed with the weights in force at each chunk\'s step (pre-update)',md<=1e-5,f'max abs diff {md:.2e}')
check('9d ... and differs from the KV the current (final) weights would give',dnow>1e-4 and wmoved>0,f'last-layer value diff {dnow:.2e}')
done()
