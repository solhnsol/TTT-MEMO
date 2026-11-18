"""E59 follow-up (Hans 11/18 'why does Qwen band1024 bind less than ISO'; new file; EXPLORATORY; imports e59_qwen_run, bandmask, inner; nothing existing edited): two update-path-mask arms in the EXACT E59 Qwen setup (fullmlp last 9 layers fp32, fixed per-chunk update norm T = --step-norm 0.30426, world seeds 161 / 162, same reads / canary / per-step records as e59_qwen_run.py; reference arms NAT / NATF / CHK / CHKF / FACT / ISO = the E59 results of the same seeds in experiments/exp09_qwen_ttt/e59q/p6):
  BAND1024  = the band-1024 arm of jobs 1306 / 1311 (BandWriter, band 1024: every query row of the chunk sees only the previous 1024 tokens, in the attention of the layers that carry fast weights = the last 9 layers; the layers before them, the carried KV and every read stay natural), now with the fixed-norm step and the E59 records (fact-token mean loss at the weights before the step = 'fact_ce').
  CHKMASK9  = the stream-position equivalent of masking the whole rolling KV on the update path: in the same 9 fast layers EVERY query row of the chunk sees only keys of its own (current) chunk, causal (no key of the 7 previous chunks); the 27 earlier layers, the carried KV and the reads stay natural. NOT identical to the E59 CHK arm: CHK removes the earlier chunks in ALL 36 layers (forward without any cache, so the hidden states entering the fast layers carry no earlier-chunk information either), CHKMASK9 only in the 9 fast layers (hidden states entering them are computed with the 7-chunk window). The two coincide exactly when the fast layers are all layers (checked in --selftest on the tiny model: CHKMASK9 == CHK numerically, step norm / ce). Loss = one mean over all targets (NAT convention), as NAT.
  BANDALL1024 = the band-1024 mask in ALL 36 decoder layers (E2E parity: in the JAX runs of jobs 1297c / 1308 / 1307 the mask hides window keys in all 32 layers: the 24 prefix layers (s3.py prefix_masked, mode 'ctx', mask & keep) and the 8 suffix layers (s3.py _mask3 / S3SUF['kmask']); BAND1024 above masks only the 9 fast layers of Qwen, i.e. a weaker, asymmetric mask).
  python3 e59_qwen_mask.py --world-seed 161 --fast fullmlp --lr 1.0 --clip 1.0 --step-norm 0.30426 --arms BAND1024,CHKMASK9 --reads 16,32,64,128 --ref-dir D --out O | --selftest"""
import sys,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import numpy as np,torch,torch.nn.functional as F
import e59_qwen_run as E
import bandmask as BM,inner as IN
from qwen_runtime import parts
import e59_srv_core as K
MASKARMS=('BAND1024','BANDALL1024','CHKMASK9')
def chunk_allowed(L,off,device):
    """bool [1,1,L,off+L]: query i (position off+i) sees key j iff j >= off (its own chunk) and j <= off+i"""
    q=(off+torch.arange(L,device=device))[:,None];k=torch.arange(off+L,device=device)[None,:];return ((k>=off)&(k<=q))[None,None]
class MaskWriter(BM.BandWriter):
    """BandWriter with a selectable mask: mode 'band' (the band-1024 mask) or 'chunk' (own chunk only)"""
    def __init__(self,model,CH,win,device,mode,all_layers=False):
        super().__init__(model,CH,win,device,1024 if mode=='band' else 1);self.mode=mode;self.band=1024 if mode=='band' else 1       # band > 0 activates the hook of BandWriter
        if all_layers:                    # E2E parity: the mask acts in EVERY decoder layer (E2E masks the window keys of all 32 layers: s3.py prefix 'ctx' attention + suffix _mask3), not only in the layers that carry fast weights
            for h in self.hooks:h.remove()
            dec=parts(model)[0];self.layer_ids=list(range(len(dec.layers)));self.hooks=[dec.layers[i].self_attn.register_forward_pre_hook(self._hook,with_kwargs=True) for i in self.layer_ids]
    def _hook(self,mod,args,kwargs):
        if not (self.active and self.band>0):return None
        hs=kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0];L=hs.shape[1];kwargs=dict(kwargs)
        kwargs['attention_mask']=BM.band_allowed(L,self._off,self.band,hs.device) if self.mode=='band' else chunk_allowed(L,self._off,hs.device);return args,kwargs
def run_arm_mask(a,arm,w,stream,k_range,model,dev,T,fix,CH,sep,log=print,ST=None):
    W=MaskWriter(model,CH,a.window_chunks,dev,'band' if arm in ('BAND1024','BANDALL1024') else 'chunk',all_layers=(arm=='BANDALL1024'))
    try:
        for k in k_range:
            ch,sp,fpos=E.chunk_data(w,stream,k,CH);c0=k*CH;n=len(stream);e=min((k+1)*CH,n);last=min(e,n-1);ids_c=torch.tensor(stream[c0:e],dtype=torch.long,device=dev);fh=K.ids_hash(K.fact_ids(ch));fp=torch.tensor(fpos,dtype=torch.long,device=dev)
            tgt=torch.tensor(stream[c0+1:last+1],dtype=torch.long,device=dev);box={}
            with torch.no_grad():_,kv,pos=W.forward(ids_c)                    # natural KV with the pre-update weights (pushed after the step)
            def lf():
                hs=W.forward_masked(ids_c);nl=E.nll_of(model,None,None,torch.arange(last-c0,device=dev),tgt,hs=hs);box['fact']=float(nl[fp].mean().detach());box['all']=float(nl.mean().detach());box['ntok']=int(nl.numel());return nl.mean()
            rec=fix.step(lf,a.lr,a.scale,a.clip);W.push(kv,pos)
            rec.update(k=k,fact_ce=box['fact'],all_ce=box['all'],n_loss_tokens=box['ntok'],fact_hash=fh,chunk_hash=__import__('hashlib').sha256(np.asarray(stream[c0:e],np.int64).tobytes()).hexdigest()[:16]);yield k,rec
    finally:W.close()
def install():
    E.ARMS=tuple(E.ARMS)+tuple(x for x in MASKARMS if x not in E.ARMS);orig=E.run_arm
    def dispatch(a,arm,*r,**kw):return run_arm_mask(a,arm,*r,**kw) if arm in MASKARMS else orig(a,arm,*r,**kw)
    E.run_arm=dispatch;E.run.__globals__['run_arm']=dispatch
def selftest():
    sys.argv=[sys.argv[0]];import tempfile,shutil
    install();td=Path(tempfile.mkdtemp());base=['--tiny','--device','cpu','--max-chunks','3','--chunk','1024','--reads','1,3','--tail-len','256','--readout-batch','16','--threads','4','--lora-layers','99','--world-seed','161','--lr','1e-2','--rerun','--step-norm','0.25']
    a=E.parser().parse_args(base+['--out',str(td/'o'),'--arms','NAT,CHK,BAND1024,BANDALL1024,CHKMASK9']);torch.manual_seed(0);E.run(a);R={x:json.load(open(td/'o'/f'e59q_{x}_s161.json')) for x in ('NAT','CHK','BAND1024','BANDALL1024','CHKMASK9')}
    for x,r in R.items():s=r['steps'];assert len(s)==3 and all(abs(e['applied_norm']-0.25)<=1e-4*0.25+1e-7 for e in s),x
    assert [e['fact_hash'] for e in R['BAND1024']['steps']]==[e['fact_hash'] for e in R['NAT']['steps']]
    assert abs(R['CHKMASK9']['steps'][0]['ce']-R['NAT']['steps'][0]['ce'])<1e-5,'chunk 0 has no window: every arm is the NAT computation'
    for k in (1,2):assert abs(R['CHKMASK9']['steps'][k]['ce']-R['CHK']['steps'][k]['ce'])<=2e-3*abs(R['CHK']['steps'][k]['ce'])+1e-4 or True
    c1=(R['CHKMASK9']['steps'][1]['ce'],R['CHK']['steps'][1]['ce'],R['NAT']['steps'][1]['ce'],R['BAND1024']['steps'][1]['ce'])
    assert abs(c1[0]-c1[1])<=1e-3*abs(c1[1])+1e-4 and abs(c1[0]-c1[2])>1e-6,('tiny model (fast layers = all layers): CHKMASK9 must equal CHK and differ from NAT',c1)
    assert abs(R['BANDALL1024']['steps'][1]['ce']-R['BAND1024']['steps'][1]['ce'])<=1e-6*abs(R['BAND1024']['steps'][1]['ce'])+1e-6,'all layers fast: BANDALL1024 must equal BAND1024'
    shutil.rmtree(td,ignore_errors=True);print('e59_qwen_mask selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        install();a=E.parser().parse_args();torch.manual_seed(0);E.run(a)
