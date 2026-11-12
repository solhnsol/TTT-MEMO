"""Exp09 readout: per-item margins (gold minus foil summed log-prob of the candidate tokens) after a fixed context
(R1 = pg1342 neutral tail, R1-alt = SQuAD tail), no weight update; canaries = mean NTP NLL of the pg11 text and of
the tail. The context is encoded ONCE per evaluation with the current weights (its KV depends on the LoRA layers),
then all probe rows are scored in batches against copies of that KV cache (positions continue after the context).
test_readout.py checks cached == full-concatenation scoring. Logits are fp32 (lm_head output upcast before
log_softmax); with readout_dtype='fp32' the frozen base is also cast to fp32 for the readout (exact round trip)."""
import torch,torch.nn.functional as F
from transformers import DynamicCache
from qwen_runtime import parts

@torch.no_grad()
def encode_context(model,ctx_ids,device):
    dec,_=parts(model);ids=torch.tensor(ctx_ids,dtype=torch.long,device=device)[None]
    cache=DynamicCache()
    dec(input_ids=ids,position_ids=torch.arange(ids.shape[1],device=device)[None],past_key_values=cache,use_cache=True)
    return [(l.keys,l.values) for l in cache.layers],ids.shape[1]

def _span_lp(head,hs,a,b):
    """sum log p(tokens a..b-1) from hidden states hs (positions a-1..b-2 predict them)."""
    lg=head(hs[a-1:b-1]).float();return lg.log_softmax(-1)
@torch.no_grad()
def score_rows(model,ctx_kv,ctx_len,rows,device,batch=8,pad_id=0):
    """rows: [(ids,(a,b))] with candidate tokens ids[a:b] (a >= 1). Returns per-row summed log-prob (float64 list)."""
    dec,head=parts(model);out=[]
    for g in range(0,len(rows),batch):
        grp=rows[g:g+batch];B=len(grp);L=max(len(r[0]) for r in grp)
        ids=torch.full((B,L),pad_id,dtype=torch.long,device=device)
        for i,(x,_) in enumerate(grp):ids[i,:len(x)]=torch.tensor(x,dtype=torch.long,device=device)
        pos=(ctx_len+torch.arange(L,device=device))[None].expand(B,L)
        if ctx_kv is None:cache=None
        else:cache=DynamicCache(ddp_cache_data=[(k.expand(B,-1,-1,-1),v.expand(B,-1,-1,-1)) for k,v in ctx_kv])
        hs=dec(input_ids=ids,position_ids=pos,past_key_values=cache,use_cache=cache is not None).last_hidden_state
        for i,(x,(a,b)) in enumerate(grp):   # right padding: real tokens never see the pads (causal)
            assert a>=1
            lp=_span_lp(head,hs[i],a,b);t=torch.tensor(x[a:b],dtype=torch.long,device=device)
            out.append(float(lp.gather(1,t[:,None]).double().sum()))
        del hs,cache
    return out

@torch.no_grad()
def score_full(model,ctx_ids,row,device):
    """reference: one forward on context + row, no cache."""
    dec,head=parts(model);x,(a,b)=row;full=list(ctx_ids)+list(x);o=len(ctx_ids)
    ids=torch.tensor(full,dtype=torch.long,device=device)[None]
    hs=dec(input_ids=ids,position_ids=torch.arange(len(full),device=device)[None],use_cache=False).last_hidden_state[0]
    lp=_span_lp(head,hs,o+a,o+b);t=torch.tensor(x[a:b],dtype=torch.long,device=device)
    return float(lp.gather(1,t[:,None]).double().sum())

@torch.no_grad()
def nll(model,ids,device,piece=1024):
    """mean NLL of ids[1:] given the prefix (one forward, lm_head in pieces); = JAX lp[:-1].mean() over the 8 chunks."""
    dec,head=parts(model);x=torch.tensor(ids,dtype=torch.long,device=device)[None]
    hs=dec(input_ids=x,position_ids=torch.arange(len(ids),device=device)[None],use_cache=False).last_hidden_state[0]
    tot=0.0;n=len(ids)-1
    for s in range(0,n,piece):
        e=min(s+piece,n);lg=head(hs[s:e]).float()
        tot+=float(F.cross_entropy(lg,x[0,s+1:e+1],reduction='sum').double())
    return tot/n

class Evaluator:
    """margins for all items under the R1 and R1-alt contexts + canaries, in the exp07 log format."""
    def __init__(self,world,rows,device,batch=8,pad_id=0,readout_dtype=None):
        self.W=world;self.rows=rows;self.device=device;self.batch=batch;self.pad=pad_id;self.readout_dtype=readout_dtype
    def margins(self,model,ctx_ids):
        kv,L=encode_context(model,ctx_ids,self.device)
        sc=score_rows(model,kv,L,self.rows,self.device,self.batch,self.pad);del kv
        mg=[g-f for g,f in zip(sc[0::2],sc[1::2])];out={};i=0
        for f,its in self.W.fam.items():out[f]=mg[i:i+len(its)];i+=len(its)
        assert i==len(mg)
        return out
    def evaluate(self,model):
        from qwen_runtime import cast_base_
        was=model.training;model.eval();orig=None
        if self.readout_dtype is not None:
            orig=next(p.dtype for n,p in model.named_parameters() if 'lora' not in n);cast_base_(model,self.readout_dtype)
        try:
            out=self.margins(model,self.W.tail_ids);alt=self.margins(model,self.W.alt_ids)
            out['_gen_nll']=nll(model,self.W.gen_ids,self.device);out['_tail_nll']=nll(model,self.W.tail_ids,self.device);out['_alt']=alt
        finally:
            if orig is not None:cast_base_(model,orig)
            model.train(was)
        return out
