"""Test 10: band mask of the Qwen3 write (CPU, tiny random Qwen3): (a) mask tensor, (b) hook acts only on the LoRA layers and only while active (hidden states of earlier layers identical), (c) band >= whole context gives the natural step (update equal to DetachedWriter), (d) a small band changes the update, (e) the pushed KV is natural (equal to DetachedWriter's), (f) band 0 = DetachedWriter."""
from _testutil import *
import torch,copy
import bandmask as BM,inner as IN
from qwen_runtime import parts,lora_params
CHk=16
def mk(seed=0):
    m=tiny(seed=seed,last_n=2);randomize_lora_(m,seed=3);m.eval();return m
def state(m):return [p.detach().clone() for _,p in lora_params(m)]
a=BM.band_allowed(4,3,2,'cpu')[0,0];ref=torch.zeros(4,7,dtype=torch.bool)
for i in range(4):
    for j in range(7):ref[i,j]=(0<=(3+i)-j<2)
check('10a band_allowed matches the definition (query pos off+i, keys at distance 0..band-1)',bool((a==ref).all()))
m=mk();dec=parts(m)[0];ids=torch.arange(1,CHk+1);w=BM.BandWriter(m,CHk,3,DEV,band=3);check('10b hooks only on the LoRA layers',w.layer_ids==[len(dec.layers)-2,len(dec.layers)-1],str(w.layer_ids))
stream=[int(x) for x in torch.randint(1,50,(CHk*5+1,),generator=torch.Generator().manual_seed(1))]
with torch.no_grad():
    for k in range(2):
        hs,kv,off=w.forward(torch.tensor(stream[k*CHk:(k+1)*CHk]));w.push(kv,off)
    x=torch.tensor(stream[2*CHk:3*CHk]);cache,off=w.past();pos=(off+torch.arange(CHk))[None]
    def run(active):
        cache,off=w.past();w._off=off;w.active=active
        try:o=dec(input_ids=x[None],position_ids=pos,past_key_values=cache,use_cache=True,output_hidden_states=True)
        finally:w.active=False
        return o.hidden_states
    h0=run(False);h1=run(True);nL=len(dec.layers);first=w.layer_ids[0]
    same=all(torch.equal(h0[i],h1[i]) for i in range(first+1))        # hidden_states[i] = input of layer i; layers < first are natural; the input of the first masked layer too
    diff=not torch.equal(h0[-1],h1[-1])
check('10b earlier layers identical, masked layers change the output when active',same and diff)
w.band=10**6
with torch.no_grad():hb=run(True)
check('10c band >= context: masked forward equals natural',torch.allclose(h0[-1],hb[-1],atol=1e-6))
w.close()
def run_steps(kind,band=0,nst=4):
    m=mk();W=(IN.DetachedWriter(m,CHk,3,DEV) if kind=='plain' else BM.BandWriter(m,CHk,3,DEV,band));st=[]
    for k in range(nst):st.append(W.step(stream,k,1e-2))
    out=(state(m),[c['kv'] for c in W.chunks],st)
    if kind!='plain':W.close()
    return out
sp,kp,stp=run_steps('plain');_,kp1,_=run_steps('plain',nst=1);_,ks1,_=run_steps('band',3,nst=1);sb,kb,stb=run_steps('band',10**6);ss,ks,sts=run_steps('band',3);s0,k0,st0=run_steps('band',0)
check('10c big band: LoRA weights after 4 steps equal the plain step',all(torch.allclose(x,y,atol=1e-6) for x,y in zip(sp,sb)))
check('10d small band: weights differ from the plain write',any(not torch.allclose(x,y,atol=1e-6) for x,y in zip(sp,ss)))
check('10d small band: gradient norms differ (the update norm itself is clipped)',abs(stp[1]['grad_norm']-sts[1]['grad_norm'])>1e-6,f"chunk0 {stp[0]['grad_norm']:.6f}/{sts[0]['grad_norm']:.6f}, chunk1 {stp[1]['grad_norm']:.6f}/{sts[1]['grad_norm']:.6f}")
check('10f band 0 = plain write',all(torch.equal(x,y) for x,y in zip(sp,s0)))
# pushed KV natural: after ONE step the chunk-0 KV (pre-update weights) must equal the plain writer's
eq0=all(torch.allclose(a[0],b[0],atol=1e-6) and torch.allclose(a[1],b[1],atol=1e-6) for a,b in zip(kp1[0],ks1[0]));print('max KV diff chunk 0',max(float((a[0]-b[0]).abs().max()) for a,b in zip(kp1[0],ks1[0])))
check('10e pushed KV of chunk 0 is natural (equal to the plain writer, small band)',eq0)
done()
