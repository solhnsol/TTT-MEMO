"""OLMo-3-7B port of the detached-window TTT write (new file; thread-server 11/19; EXPLORATORY; Hans: 'OLMo-3-7B, last 8 layers, mask 32 layers'). Base: allenai/Olmo-3-1025-7B (Apache-2.0): 32 layers = 24 sliding_attention (window 4096) + 8 full_attention (layers 3,7,...,31), MHA 32 heads, hidden 4096; RoPE per LAYER TYPE (sliding: default theta 5e5, full: yarn factor 8, attention_factor 1.2079), q_norm / k_norm over all heads before RoPE.
ONE configuration object (LayerPlan) says which layers are updated (fast: last 8 = 24..31, full-MLP gate/up/down fp32) and which layers the write-path attention mask hits (MASK arm: ALL 32 layers); the writer reads these lists from it and asserts them (the Qwen band mask hit only the 9 fast layers because the two lists were separate).
OlmoWriter = exp09 inner.DetachedWriter (rolling detached KV of the previous `win` chunks + the current chunk, one inner step per chunk) with the OLMo changes: (1) cached keys are re-rotated to window positions with the RoPE of THEIR layer type (inv_freq of that type, pure rotation; the cached keys already carry the type's attention_scaling because HF stores post-RoPE keys, so the shift must NOT scale again), (2) the write-path mask is built per layer from the model's own allowed set (causal, and the 4096 sliding window for sliding layers) intersected with the band / own-chunk rule, (3) it does not assert attention_scaling == 1.
  allowed(layer_type, L, off, window, band=None, own_chunk=False) -> bool [1,1,L,off+L]
  python3 olmo_port.py --selftest   (CPU, tiny random Olmo3: cached == full forward incl. sliding layers and window shorter than the stream, mask == original when band is None, boundary tests at band 1024 / sliding window 4096 scaled to the tiny model: distance B-1 changes the output, B does not)"""
import sys,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/exp09_qwen_ttt'))
import torch,torch.nn.functional as F
import qwen_runtime as QR
from qwen_runtime import parts,lora_params
from transformers import DynamicCache
from transformers.models.olmo3.modeling_olmo3 import rotate_half
class LayerPlan:
    """single source of truth for the layer lists (read by the writer, asserted by the arms)"""
    def __init__(self,config,fast_n=8,mask_arm=False):
        self.n=config.num_hidden_layers;self.layer_types=list(config.layer_types);self.window=int(config.sliding_window)
        self.fast_layers=list(range(self.n-fast_n,self.n));self.mask_layers=list(range(self.n)) if mask_arm else []
    def check(self,model,band=None,own_chunk=False):
        assert len(self.layer_types)==self.n==model.config.num_hidden_layers
        if self.mask_layers:
            assert self.mask_layers==list(range(self.n)),'the MASK arm must mask ALL layers'
            assert set(self.fast_layers)<=set(self.mask_layers),'fast layers must be inside the mask layers'
            if band is not None:assert band<=self.window,('band must lie inside the sliding window',band,self.window)
        names={nm for nm,_ in lora_params(model)};assert names=={f'model.layers.{i}.mlp.{t}.weight' for i in self.fast_layers for t in ('gate_proj','up_proj','down_proj')},'updated parameters differ from the plan'
        return True
def allowed(layer_type,L,off,window,band=None,own_chunk=False,device='cpu'):
    q=(off+torch.arange(L,device=device))[:,None];k=torch.arange(off+L,device=device)[None,:];d=q-k;m=d>=0
    if layer_type=='sliding_attention':m=m&(d<window)
    if band is not None:m=m&(d<band)
    if own_chunk:m=m&(k>=off)
    return m[None,None]
class OlmoWriter:
    def __init__(self,model,CH,win,device,band=None,own_chunk=False,mask_all=False,fast_n=8):
        self.model=model;self.CH=CH;self.win=win;self.device=device;self.chunks=[];dec=parts(model)[0];self.rot=dec.rotary_emb;self.cfg=model.config
        self.plan=LayerPlan(self.cfg,fast_n,mask_arm=mask_all);self.band=band;self.own=own_chunk;self.active=False;self._off=0;self.hooks=[];self.plan.check(model,band,own_chunk)
        for i in self.plan.mask_layers:self.hooks.append(dec.layers[i].self_attn.register_forward_pre_hook(self._mk_hook(self.plan.layer_types[i]),with_kwargs=True))
    def _mk_hook(self,lt):
        def hook(mod,args,kwargs):
            if not self.active:return None
            hs=kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0];kw=dict(kwargs);kw['attention_mask']=allowed(lt,hs.shape[1],self._off,self.plan.window,self.band,self.own,hs.device);return args,kw
        return hook
    def close(self):
        for h in self.hooks:h.remove()
        self.hooks=[]
    def reset(self):self.chunks=[]
    def rotate(self,k,shift,layer_type):
        """pure RoPE rotation of post-RoPE keys [B,H,L,D] by `shift` positions with the inv_freq of the layer type (no attention scaling: the cached keys already carry it)"""
        if shift==0 or getattr(self,'abs_mode',False):return k
        inv=getattr(self.rot,f'{layer_type}_inv_freq').float().to(k.device);ang=shift*inv;emb=torch.cat([ang,ang],-1);cos=emb.cos()[None,None,None,:];sin=emb.sin()[None,None,None,:]
        kf=k.float();return (kf*cos+rotate_half(kf)*sin).to(k.dtype)
    def past(self):
        if not self.chunks:return DynamicCache(),0
        nl=len(self.chunks[0]['kv']);data=[]
        for l in range(nl):
            lt=self.plan.layer_types[l];ks=[self.rotate(c['kv'][l][0],i*self.CH-c['pos'],lt) for i,c in enumerate(self.chunks)];data.append((torch.cat(ks,dim=2),torch.cat([c['kv'][l][1] for c in self.chunks],dim=2)))
        return DynamicCache(ddp_cache_data=data),sum(c['kv'][0][0].shape[2] for c in self.chunks)
    def forward(self,ids,masked=False):
        dec,_=parts(self.model);cache,off=self.past();L=ids.shape[0];base=(self.abs_pos if getattr(self,'abs_mode',False) else off);pos=(base+torch.arange(L,device=ids.device))[None];self._off=off;self.active=bool(masked)
        try:hs=dec(input_ids=ids[None],position_ids=pos,past_key_values=cache,use_cache=True).last_hidden_state[0]
        finally:self.active=False
        kv=[(l.keys[:,:,-L:].detach(),l.values[:,:,-L:].detach()) for l in cache.layers];return hs,kv,off
    def push(self,kv,pos):
        self.chunks.append({'kv':kv,'pos':pos})
        while len(self.chunks)>self.win:self.chunks.pop(0)
def tiny_olmo3(seed=0,layers=8,hidden=64,heads=4,inter=128,window=16,vocab=300,dtype=torch.float32,fast_n=2,layer_types=None):
    from transformers import AutoConfig,Olmo3ForCausalLM
    import glob
    src=[p for p in glob.glob(str(Path.home()/'.cache/huggingface/hub/models--allenai--Olmo-3-1025-7B/snapshots/*'))]
    c=AutoConfig.from_pretrained(src[0] if src else '/tmp/claude-1147/olmo')
    c.num_hidden_layers=layers;c.hidden_size=hidden;c.intermediate_size=inter;c.num_attention_heads=heads;c.num_key_value_heads=heads;c.sliding_window=window;c.vocab_size=vocab;c.pad_token_id=0;c.eos_token_id=1
    c.layer_types=list(layer_types) if layer_types is not None else [('sliding_attention' if (i%4)!=3 else 'full_attention') for i in range(layers)]
    for lt in c.rope_parameters:
        if c.rope_parameters[lt].get('rope_type')=='yarn':c.rope_parameters[lt]['original_max_position_embeddings']=64
    torch.manual_seed(seed);m=Olmo3ForCausalLM(c).to(dtype).eval();m,_=QR.wrap_fullmlp(m,last_n=fast_n);return m
def selftest():
    torch.manual_seed(0);m=tiny_olmo3(layers=8,window=16);CH=32;stream=torch.randint(2,299,(6*CH,))
    full=lambda a,b,p0=0:parts(m)[0](input_ids=stream[a:b][None],position_ids=(p0+torch.arange(b-a))[None],use_cache=False).last_hidden_state[0]
    # (1) cached forward == full forward when the window covers the whole stream (no rotation needed); and, with a window SHORTER than the stream, the re-rotated cache (window-relative positions, per-layer-type RoPE) gives the same hidden states as the same carried KV used at ABSOLUTE positions (RoPE is relative: sliding + yarn layers exercise the rotation)
    W=OlmoWriter(m,CH,8,torch.device('cpu'),mask_all=False,fast_n=2)
    with torch.no_grad():
        for k in range(5):
            c=stream[k*CH:(k+1)*CH];hs,kv,off=W.forward(c);ref=full(0,(k+1)*CH)[-CH:];assert float((hs-ref).abs().max())<2e-4,('full cover',k,float((hs-ref).abs().max()));W.push(kv,off)
        for win in (1,2,3):
            Wr=OlmoWriter(m,CH,win,torch.device('cpu'),mask_all=False,fast_n=2);Wa=OlmoWriter(m,CH,win,torch.device('cpu'),mask_all=False,fast_n=2);Wa.abs_mode=True
            for k in range(5):
                c=stream[k*CH:(k+1)*CH];hr,kvr,offr=Wr.forward(c);Wa.abs_pos=k*CH;ha,kva,offa=Wa.forward(c);d=float((hr-ha).abs().max())
                assert d<2e-4,('rotation',win,k,d);Wr.push(kvr,offr);Wa.push(kva,k*CH)
        assert any(W.plan.layer_types[l]=='full_attention' for l in range(8)) and any(W.plan.layer_types[l]=='sliding_attention' for l in range(8))
    # (2) our explicit 'original allowed' masks on ALL layers (band None) == the model's own masks
    W=OlmoWriter(m,CH,3,torch.device('cpu'),band=None,mask_all=True,fast_n=2)
    with torch.no_grad():
        W2=OlmoWriter(m,CH,3,torch.device('cpu'),mask_all=False,fast_n=2)
        for k in range(4):
            c=stream[k*CH:(k+1)*CH];h1,kv1,o1=W.forward(c,masked=True);h2,kv2,o2=W2.forward(c,masked=False);assert float((h1-h2).abs().max())<1e-5,(k,float((h1-h2).abs().max()));W.push(kv2,o2);W2.push(kv2,o2)
    # (3) boundary tests on one-layer models (sensitivity of the last position to a changed token at distance d): band B and sliding window S
    for lt,expect in (('sliding_attention',None),('full_attention',None)):
        m1=tiny_olmo3(layers=1,window=24,fast_n=1,layer_types=[lt])
        T=60;x=torch.randint(2,299,(T,));dec=parts(m1)[0]
        def last_hs(tok,band=None):
            W1=OlmoWriter(m1,T,1,torch.device('cpu'),band=band,mask_all=True,fast_n=1) if True else None
            with torch.no_grad():h,_,_=W1.forward(tok,masked=True)
            W1.close();return h[-1]
        base=last_hs(x,band=None);B=10
        for band in (None,B):
            lim=band if band is not None else (24 if lt=='sliding_attention' else T)
            b0=last_hs(x,band)
            for d in (0,1,lim-1,lim):
                if d>=T:continue
                y=x.clone();pos=T-1-d;y[pos]=(y[pos]+7)%297+2
                if d==0:continue
                ch=float((last_hs(y,band)-b0).abs().max())>1e-6
                want=d<lim;assert ch==want,(lt,band,d,'changes' if ch else 'unchanged','expected','change' if want else 'no change')
    print('olmo_port selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
