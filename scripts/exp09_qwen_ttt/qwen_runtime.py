"""Exp09 runtime: Qwen3 base (bf16) + PEFT LoRA fast weights (fp32) on the MLP projections of the last N layers.
Patterns from scripts/exp02a2_semantic_v3/model_runtime.py (LoRA fp32 over a bf16 base, frozen-weight watch, LoRA state
save/restore), with the TTT-E2E-like placement: gate/up/down_proj of the last 9 of 36 layers, r=16, alpha=32, B=0 init."""
from __future__ import annotations
import glob,hashlib,os
from pathlib import Path
import torch
from peft import LoraConfig,TaskType,get_peft_model

HF_CACHE=Path(os.environ.get('HF_HOME',Path.home()/'.cache/huggingface'))/'hub'
QWEN_ID='Qwen/Qwen3-4B-Instruct-2507'

def qwen_snapshot_dir(model_id=QWEN_ID):
    d=sorted(glob.glob(str(HF_CACHE/('models--'+model_id.replace('/','--'))/'snapshots'/'*')))
    if not d:raise FileNotFoundError(f'{model_id} not in the HF cache {HF_CACHE}')
    return Path(d[-1])

def is_lora(name):return 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name

def load_base(model_dir,device,dtype=torch.bfloat16,attn='sdpa'):
    from transformers import AutoModelForCausalLM
    m=AutoModelForCausalLM.from_pretrained(str(model_dir),dtype=dtype,low_cpu_mem_usage=True,attn_implementation=attn)
    dts={p.dtype for p in m.parameters()}
    if any('float8' in str(d) for d in dts):raise RuntimeError(f'FP8 weights loaded: {dts}')
    return m.to(device)

def tiny_qwen(model_dir=None,seed=0,layers=4,hidden=128,inter=256,heads=4,kv=2,head_dim=32,attn='sdpa',dtype=torch.float32):
    """randomly initialised small Qwen3 with the real vocabulary/rope/norm settings (tests and smoke runs only)."""
    from transformers import AutoConfig,Qwen3ForCausalLM
    cfg=AutoConfig.from_pretrained(str(model_dir or qwen_snapshot_dir()))
    cfg.num_hidden_layers=layers;cfg.hidden_size=hidden;cfg.intermediate_size=inter;cfg.num_attention_heads=heads
    cfg.num_key_value_heads=kv;cfg.head_dim=head_dim;cfg.max_window_layers=layers
    if hasattr(cfg,'layer_types') and cfg.layer_types is not None:cfg.layer_types=cfg.layer_types[:layers]
    cfg._attn_implementation=attn
    torch.manual_seed(seed);m=Qwen3ForCausalLM(cfg).to(dtype)
    return m.eval()

def wrap_lora(model,r=16,alpha=32,targets=('gate_proj','up_proj','down_proj'),last_n=9,dropout=0.0):
    """PEFT LoRA on `targets` of the last `last_n` decoder layers; LoRA params fp32, everything else frozen."""
    n=model.config.num_hidden_layers;layers=list(range(n-last_n,n))
    cfg=LoraConfig(r=r,lora_alpha=alpha,lora_dropout=dropout,bias='none',target_modules=list(targets),
                   layers_to_transform=layers,layers_pattern='layers',task_type=TaskType.CAUSAL_LM,inference_mode=False)
    pm=get_peft_model(model,cfg)
    for name,p in pm.named_parameters():
        if is_lora(name):p.data=p.data.to(torch.float32);p.requires_grad_(True)
        else:p.requires_grad_(False)
    pm.config.use_cache=False
    names=[nm for nm,_ in pm.named_parameters() if is_lora(nm)]
    got=sorted({int(nm.split('.layers.')[1].split('.')[0]) for nm in names})
    if got!=layers:raise RuntimeError(f'LoRA on layers {got}, expected {layers}')
    for t in targets:
        k=sum(1 for nm in names if f'.{t}.lora_A' in nm)
        if k!=last_n:raise RuntimeError(f'{t}: LoRA on {k} layers, expected {last_n}')
    for nm,p in pm.named_parameters():
        if 'lora_B' in nm and p.abs().max().item()!=0.0:raise RuntimeError('LoRA B is not zero at init')
    return pm,{'r':r,'alpha':alpha,'scaling':alpha/r,'targets':list(targets),'layers':layers,'dropout':dropout,
               'n_tensors':len(names),'n_params':sum(p.numel() for nm,p in pm.named_parameters() if is_lora(nm)),'dtype':'float32'}

def lora_params(model):
    """ordered (name, param) list of the fast weights."""
    out=[(n,p) for n,p in model.named_parameters() if is_lora(n)]
    if not out:raise RuntimeError('no LoRA parameters')
    return out

def parts(model):
    """(decoder, lm_head) of a (PEFT-wrapped) causal LM: losses project only the needed positions through lm_head."""
    m=model
    if hasattr(m,'get_base_model'):m=m.get_base_model()
    return m.model,m.lm_head

def lora_state(model):return {n:p.detach().cpu().clone() for n,p in lora_params(model)}
def load_lora_state_(model,state):
    with torch.no_grad():
        ps=dict(lora_params(model))
        if set(ps)!=set(state):raise RuntimeError('LoRA state keys differ')
        for n,p in ps.items():p.copy_(state[n].to(device=p.device,dtype=p.dtype))
def tensor_hash(named):
    hh=hashlib.sha256()
    for n,t in named:
        t=t.detach().contiguous().cpu();hh.update(n.encode());hh.update(str(tuple(t.shape)).encode());hh.update(str(t.dtype).encode())
        hh.update(t.view(torch.uint8).numpy().tobytes() if t.dtype!=torch.bfloat16 else t.view(torch.int16).numpy().tobytes())
    return hh.hexdigest()
def lora_hash(model):return tensor_hash(lora_params(model))

def watch_names(model,all_frozen=False):
    """frozen tensors whose hash is checked: all of them (tests), or a small cross-section (layer 0 / first LoRA layer /
    last layer q_proj and MLP base weights, final norm) so the check stays cheap on the 4B model."""
    names=[n for n,_ in model.named_parameters() if not is_lora(n)]
    if all_frozen:return names
    n=model.config.num_hidden_layers;keys=[f'layers.0.self_attn.q_proj.weight',f'layers.{n-1}.self_attn.q_proj.weight',
        f'layers.{n-1}.mlp.down_proj.base_layer.weight',f'layers.{n-1}.mlp.gate_proj.base_layer.weight','model.norm.weight','layers.0.mlp.up_proj.weight']
    out=[x for x in names if any(x.endswith(k) for k in keys)]
    if len(out)<5:raise RuntimeError(f'frozen watch set too small: {out}')
    return out
def frozen_hash(model,names):
    ps=dict(model.named_parameters());return tensor_hash([(n,ps[n]) for n in names])

def cast_base_(model,dtype):
    """cast every frozen floating parameter in place (LoRA stays fp32). bf16 -> fp32 -> bf16 is exact, so the fp32
    readout can borrow the training model without a second copy; the frozen hash is re-checked after casting back."""
    with torch.no_grad():
        for n,p in model.named_parameters():
            if not is_lora(n) and p.is_floating_point() and p.dtype!=dtype:p.data=p.data.to(dtype)
