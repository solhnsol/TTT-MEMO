"""OLMo-3-7B runner (new file; thread-server 11/19; EXPLORATORY; wrapper of e59_qwen_run.run / e59_qwen_mask; nothing existing edited): the E59 arms on allenai/Olmo-3-1025-7B with the 4-way set NAT, MASK1024 (= band-1024 mask in ALL 32 layers), CHK (chunk alone, all layers), ISO (16 isolated sentences, one step), the fixed per-chunk update norm T of exp09 (--step-norm), fast weights = full gate / up / down MLP matrices of the LAST 8 layers (fp32, layers 24..31; OlmoWriter.LayerPlan asserts the lists), world seeds from the synthetic v3_16_1 world rebuilt with the OLMo tokenizer (names / places 2 tokens, 128 chunks of 1024 tokens), reads at exposures 16/32/64/128 (out = neutral tail --tail-len, in = the last 7 chunks), reverse-question readout with --reverse, canary. Implementation: patches in this process only: qwen_ladder.build -> OLMo loader (+ wrap_fullmlp(last_n = --olmo-fast-n)), inner.DetachedWriter -> olmo_port.OlmoWriter (NAT), e59_qwen_mask.MaskWriter -> OlmoMaskAdapter (MASK1024).
  python3 olmo_run.py --world-seed 221 --model <snapshot dir> --fast fullmlp --olmo-fast-n 8 --lr 1.0 --clip 1.0 --step-norm T --arms NAT,MASK1024,CHK,ISO --reads 16,32,64,128 --out O [--reverse] | --selftest"""
import sys,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,ROOT/'scripts/exp09_qwen_ttt',ROOT/'scripts/exp07_write_ladder'):sys.path.insert(0,str(p))
import torch
import qwen_ladder as QL,qwen_runtime as QR,world_text as WT,inner as IN
import e59_qwen_run as E,e59_qwen_mask as M
import olmo_port as OP
class OlmoMaskAdapter:
    """interface of e59_qwen_mask.MaskWriter for the OLMo writer: forward(ids) natural (no grad needed), forward_masked(ids) with the write-path mask of the arm, push, close"""
    def __init__(self,model,CH,win,device,mode,all_layers=False):
        assert mode=='band' and all_layers,'OLMo MASK arm = band 1024 in ALL layers'
        self.w=OP.OlmoWriter(model,CH,win,device,band=1024,mask_all=True,fast_n=OLMO_FAST_N[0]);self.model=model
    def forward(self,ids):return self.w.forward(ids,masked=False)
    def forward_masked(self,ids):return self.w.forward(ids,masked=True)[0]
    def push(self,kv,pos):self.w.push(kv,pos)
    def close(self):self.w.close()
OLMO_FAST_N=[8]
def build(a):
    dev=torch.device(a.device);OLMO_FAST_N[0]=a.olmo_fast_n;mdir=Path(a.model)
    tok=WT.load_tokenizer(QR.tokenizer_file(mdir));bos={'endoftext':tok.token_to_id('<|endoftext|>'),'none':None}[a.bos]
    if a.tiny:base=OP.tiny_olmo3(seed=a.tiny_seed,layers=8,vocab=100278,window=a.olmo_tiny_window,dtype=torch.float32,fast_n=a.olmo_fast_n)
    else:
        from transformers import AutoModelForCausalLM
        base=AutoModelForCausalLM.from_pretrained(str(mdir),dtype={'bf16':torch.bfloat16,'fp32':torch.float32}[a.base_dtype],low_cpu_mem_usage=True,attn_implementation=a.attn).to(dev)
        base,_=QR.wrap_fullmlp(base,tuple(a.lora_targets.split(',')),a.olmo_fast_n)
    model=base.to(dev);lcfg={'mode':'fullmlp','layers':list(range(model.config.num_hidden_layers-a.olmo_fast_n,model.config.num_hidden_layers)),'model':'olmo3'}
    model.config.use_cache=False;return dev,mdir,tok,bos,model,lcfg
def install():
    QL.build=build;IN.DetachedWriter=lambda model,CH,win,device:OP.OlmoWriter(model,CH,win,device,fast_n=OLMO_FAST_N[0])
    M.install();M.MaskWriter=OlmoMaskAdapter
def parser():
    ap=E.parser();ap.add_argument('--olmo-fast-n',type=int,default=8);ap.add_argument('--olmo-tiny-window',type=int,default=1024);return ap
def selftest():
    sys.argv=[sys.argv[0]];import tempfile,shutil,glob
    snap=glob.glob(str(Path.home()/'.cache/huggingface/hub/models--allenai--Olmo-3-1025-7B/snapshots/*'))[0]
    install();td=Path(tempfile.mkdtemp())
    a=parser().parse_args(['--tiny','--device','cpu','--model',snap,'--max-chunks','3','--chunk','1024','--reads','1,3','--tail-len','256','--readout-batch','16','--threads','8','--olmo-fast-n','2','--olmo-tiny-window','4096','--world-seed','221','--lr','1e-2','--rerun','--step-norm','0.25','--reverse','--out',str(td/'o'),'--arms','NAT,MASK1024,CHK,ISO']);torch.manual_seed(0);E.run(a)
    R={x:json.load(open(td/'o'/f'e59q_{x}_s221.json')) for x in ('NAT','MASK1024','CHK','ISO')}
    for x,r in R.items():
        s=r['steps'];assert len(s)==3 and all(abs(e['applied_norm']-0.25)<=1e-4*0.25+1e-7 for e in s),x
        assert set(r['reads'])=={'1','3'} and set(r['reads_rev'])=={'1','3'} and len(r['reads_rev']['3']['out'])==16,x
    assert [e['fact_hash'] for e in R['MASK1024']['steps']]==[e['fact_hash'] for e in R['NAT']['steps']]
    assert abs(R['MASK1024']['steps'][0]['ce']-R['NAT']['steps'][0]['ce'])<1e-4,'chunk 0 has no window: MASK1024 == NAT'
    assert R['MASK1024']['steps'][1]['ce']!=R['NAT']['steps'][1]['ce'],'MASK must differ from NAT once a window exists'
    shutil.rmtree(td,ignore_errors=True);print('olmo_run selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        install();a=parser().parse_args();torch.manual_seed(0);E.run(a)
