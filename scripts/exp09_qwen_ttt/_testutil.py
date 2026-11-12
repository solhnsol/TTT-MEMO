"""shared helpers for the Exp09 CPU tests (tiny random Qwen3, real Qwen tokenizer). Refuses to run with a visible GPU."""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
assert os.environ.get('CUDA_VISIBLE_DEVICES','x')=='','run the tests with CUDA_VISIBLE_DEVICES= (the GPU is busy)'
import torch
torch.set_num_threads(min(8,os.cpu_count() or 1))
import qwen_runtime as QR
OUT=ROOT/'experiments/exp09_qwen_ttt/tests';OUT.mkdir(parents=True,exist_ok=True)
DEV=torch.device('cpu')
def tiny(seed=0,lora=True,last_n=2,attn='sdpa'):
    m=QR.tiny_qwen(seed=seed,attn=attn)
    if not lora:return m
    pm,cfg=QR.wrap_lora(m,16,32,('gate_proj','up_proj','down_proj'),last_n);return pm
def randomize_lora_(model,seed=1,std=0.05):
    """non-zero B (and A) so gradients and updates are generic (tests only)."""
    g=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _,p in QR.lora_params(model):p.copy_(torch.randn(p.shape,generator=g)*std)
_RES=[]
def check(name,ok,detail=''):
    line=f"{'PASS' if ok else 'FAIL'}: {name}"+(f' ({detail})' if detail else '');print(line,flush=True);_RES.append((name,ok))
def done():
    bad=[n for n,ok in _RES if not ok];print(f'{len(_RES)-len(bad)}/{len(_RES)} passed',flush=True);sys.exit(1 if bad else 0)
