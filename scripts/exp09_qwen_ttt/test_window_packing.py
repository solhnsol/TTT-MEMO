"""Tests 4-5: sliding-window native step == explicit concatenation (labels only on chunk k, nothing before the window
matters), masked write sees no earlier token; L1 packing == each row alone (no cross-row leakage, positions restart)."""
from _testutil import *
import torch.nn.functional as F
import inner as IN

CH=32;WIN=7
g=torch.Generator().manual_seed(5);stream=torch.randint(0,151000,(10*CH+11,),generator=g).tolist()   # 11 chunks, last partial
pm=tiny(seed=0);randomize_lora_(pm)
ps=[p for _,p in QR.lora_params(pm)]
def grads(loss):return [x.detach().clone() for x in torch.autograd.grad(loss,ps)]
def explicit(k):
    """loss of chunk k computed independently: plain HF forward on stream[s:e] (no position_ids), full logits,
    CE on the logits of chunk k's positions against the next stream tokens."""
    s=max(0,(k-WIN)*CH);c0=k*CH;e=min((k+1)*CH,len(stream))
    x=torch.tensor(stream[s:e])[None];lg=pm(input_ids=x).logits[0].float()
    tgt=stream[c0+1:e+1];lg=lg[c0-s:c0-s+len(tgt)]
    return F.cross_entropy(lg,torch.tensor(tgt)),len(tgt),c0-s
n=(len(stream)+CH-1)//CH
for k in (0,3,8,n-1):
    b=IN.native_batch(stream,k,CH,WIN,DEV);l1=IN.loss_of(pm,b);g1=grads(l1);l1=l1.detach()
    l2,nt,ctx=explicit(k);g2=grads(l2);l2=l2.detach()
    gd=max(float((a-c).abs().max()/(c.abs().max()+1e-30)) for a,c in zip(g1,g2))
    lab_ok=int(b[2].min())==ctx and b[2].numel()==nt and b[3].tolist()==stream[k*CH+1:k*CH+1+nt]
    ctxlen=len(b[0])-(min((k+1)*CH,len(stream))-k*CH)
    check(f'4 native chunk {k}: loss/grad == explicit concatenation (context {ctxlen} tok = {ctxlen//CH} chunks, {nt} labels, all inside chunk {k})',
          abs(float(l1)-float(l2))<=1e-5 and gd<=1e-4 and lab_ok,f'|dloss| {abs(float(l1)-float(l2)):.1e}, grad rel {gd:.1e}')
check('4 last chunk is partial and its last token has no label',IN.native_batch(stream,n-1,CH,WIN,DEV)[2].numel()==len(stream)-(n-1)*CH-1)
# tokens before the window do not matter; tokens inside the window do
k=9;b=IN.native_batch(stream,k,CH,WIN,DEV);l0=float(IN.loss_of(pm,b))
st2=list(stream);st2[(k-WIN)*CH-1]=(st2[(k-WIN)*CH-1]+1)%151000;l_out=float(IN.loss_of(pm,IN.native_batch(st2,k,CH,WIN,DEV)))
st3=list(stream);st3[(k-WIN)*CH+3]=(st3[(k-WIN)*CH+3]+1)%151000;l_in=float(IN.loss_of(pm,IN.native_batch(st3,k,CH,WIN,DEV)))
check('4 window: a token just before the 7-chunk window does not change the loss; one inside does',l_out==l0 and l_in!=l0,f'out {abs(l_out-l0):.1e}, in {abs(l_in-l0):.1e}')
# masked write: chunk alone
k=5;bm=IN.masked_batch(stream,k,CH,DEV);gm=grads(IN.loss_of(pm,bm))
st4=list(stream)
for i in range(0,k*CH):st4[i]=(st4[i]*7+13)%151000
gm2=grads(IN.loss_of(pm,IN.masked_batch(st4,k,CH,DEV)))
check('4 masked write: changing every earlier chunk leaves the gradient bit-identical',all(torch.equal(a,c) for a,c in zip(gm,gm2)))
x=torch.tensor(stream[k*CH:(k+1)*CH])[None];lr_=F.cross_entropy(pm(input_ids=x).logits[0,:-1].float(),x[0,1:])
check('4 masked write == chunk alone from position 0, CH-1 in-chunk labels',abs(float(IN.loss_of(pm,bm))-float(lr_))<=1e-6 and bm[2].numel()==CH-1 and int(bm[1][0])==0)

# ---- 5. packing ----
def rows_random(seed,n,ctx=False):
    g=torch.Generator().manual_seed(seed);out=[]
    for i in range(n):
        L=int(torch.randint(5,40,(1,),generator=g));ids=torch.randint(0,151000,(L,),generator=g).tolist()
        a=int(torch.randint(1,L-2,(1,),generator=g)) if ctx else 0;out.append((ids,(a,L-1)))
    return out
for tag,rows,CHP in (('plain rows',rows_random(1,9),128),('ctx-style rows (loss on a suffix)',rows_random(2,9,True),128),('full 1024-token step',rows_random(3,60),1024)):
    steps=IN.pack_rows(rows,CHP);s0=steps[0];nrow=0;tot=0
    # which rows went into step 0
    for ids,(a,b) in rows:
        if tot+len(ids)>len(s0[0]):break
        tot+=len(ids);nrow+=1
    assert tot==len(s0[0]) and nrow>=3,(tot,nrow)
    lp=IN.loss_of(pm,IN.packed_batch(s0,DEV));gp=grads(lp)
    N=sum(b-a for _,(a,b) in rows[:nrow]);acc=0.0;ga=[torch.zeros_like(p) for p in ps]
    for ids,(a,b) in rows[:nrow]:
        x=torch.tensor(ids)[None];lg=pm(input_ids=x).logits[0].float()   # each row alone, plain HF forward
        l=F.cross_entropy(lg[a:b],x[0,a+1:b+1],reduction='sum')/N;acc+=float(l)
        for t,gg in zip(ga,torch.autograd.grad(l,ps)):t+=gg
    rl=abs(float(lp)-acc)/abs(acc);rg=max(float((a-c).abs().max()/(c.abs().max()+1e-30)) for a,c in zip(gp,ga))
    check(f'5 packed step == mean over rows run alone ({tag}: {nrow} rows, {len(s0[0])} tokens)',rl<=1e-4 and rg<=1e-4,f'loss rel {rl:.1e}, grad rel {rg:.1e}')
# negative control: positions NOT restarted -> rows see each other -> different loss (the test is sensitive)
rows=rows_random(1,9);s0=IN.pack_rows(rows,128)[0];ids,pos,sel,tgt=IN.packed_batch(s0,DEV)
leak=float(IN.loss_of(pm,(ids,torch.arange(len(ids)),sel,tgt)));ok=float(IN.loss_of(pm,(ids,pos,sel,tgt)))
check('5 negative control: continuous positions (no restart) change the loss',abs(leak-ok)>1e-3,f'{abs(leak-ok):.2e}')
# eager attention gives the same packed loss as sdpa
pe=tiny(seed=0,attn='eager');QR.load_lora_state_(pe,QR.lora_state(pm))
le=float(IN.loss_of(pe,IN.packed_batch(s0,DEV)));check('5 packed loss identical under eager attention',abs(le-ok)<=1e-5,f'{abs(le-ok):.1e}')
done()
