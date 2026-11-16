"""Exp09 band mask (job 1301 line, EXPLORATORY; new file, no existing exp09 file edited): the ALLQ<band> update-path mask of the
JAX runs (e57_closure_masks.band_keep) for the Qwen3 SWA + NTP-TTT write. In the step's gradient forward every query row of the
chunk sees only the previous `band` tokens (itself and band - 1 before it, window keys and chunk keys), in the attention of the layers that carry
fast weights (the LoRA layers = the 'suffix' of the JAX model); the layers before them stay natural (JAX: natural prefix outputs).
The carried KV that is pushed after the step and every read stay NATURAL: BandWriter computes the KV in a separate no-grad forward with
the pre-update weights and no mask (the JAX mask acts on the update path only). band = 0 or None: no mask (arm NATS).
  BandWriter(model, CH, win, device, band)  - drop-in for inner.DetachedWriter (step(stream, k, lr, scale, clip))"""
import torch,torch.nn.functional as F
import inner as IN
from qwen_runtime import parts,lora_params
def band_allowed(L,off,band,device):
    """bool [1,1,L,off+L]: query i (position off+i) may attend to key j iff 0 <= (off+i)-j < band"""
    q=(off+torch.arange(L,device=device))[:,None];k=torch.arange(off+L,device=device)[None,:];d=q-k
    return ((d>=0)&(d<int(band)))[None,None]
def lora_layer_ids(model):
    import re;ids=set()
    for n,_ in lora_params(model):
        m=re.search(r'layers\.(\d+)\.',n)
        if m:ids.add(int(m.group(1)))
    assert ids,'no LoRA layer found';return sorted(ids)
class BandWriter(IN.DetachedWriter):
    def __init__(self,model,CH,win,device,band):
        super().__init__(model,CH,win,device);self.band=int(band or 0);self.active=False;self._off=0;self.layer_ids=lora_layer_ids(model);dec=parts(model)[0];self.hooks=[]
        for i in self.layer_ids:self.hooks.append(dec.layers[i].self_attn.register_forward_pre_hook(self._hook,with_kwargs=True))
    def _hook(self,mod,args,kwargs):
        if not (self.active and self.band>0):return None
        L=kwargs['hidden_states'].shape[1] if 'hidden_states' in kwargs else args[0].shape[1]
        kwargs=dict(kwargs);kwargs['attention_mask']=band_allowed(L,self._off,self.band,kwargs['hidden_states'].device if 'hidden_states' in kwargs else args[0].device);return args,kwargs
    def close(self):
        for h in self.hooks:h.remove()
        self.hooks=[]
    def forward_masked(self,ids):
        """gradient forward with the band mask on the LoRA layers (hidden states [L,d]); the KV of this forward is NOT used"""
        dec,_=parts(self.model);cache,off=self.past();L=ids.shape[0];pos=(off+torch.arange(L,device=ids.device))[None];self._off=off;self.active=True
        try:hs=dec(input_ids=ids[None],position_ids=pos,past_key_values=cache,use_cache=True).last_hidden_state[0]
        finally:self.active=False
        return hs
    def step(self,stream,k,lr,scale=1.0,clip=1.0):
        if self.band<=0:return super().step(stream,k,lr,scale,clip)
        CH=self.CH;n=len(stream);c0=k*CH;e=min((k+1)*CH,n);last=min(e,n-1)
        ids=torch.tensor(stream[c0:e],dtype=torch.long,device=self.device);tgt=torch.tensor(stream[c0+1:last+1],dtype=torch.long,device=self.device)
        with torch.no_grad():_,kv,pos=self.forward(ids)                    # natural KV, pre-update weights
        st=None
        if last>c0:
            def lf():return F.cross_entropy(parts(self.model)[1](self.forward_masked(ids)[:last-c0]).float(),tgt)
            st=IN.sgd_step_fn(self.model,lf,lr,scale,clip)
        self.push(kv,pos);return st
