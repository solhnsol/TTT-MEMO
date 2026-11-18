"""Exp09 cell band ladder with a FIXED per-chunk update norm (job 1310 proposal; EXPLORATORY; new file, nothing existing edited; imports cell_band_ladder and inner): the NATS / NATMASK<band> arms of cell_band_ladder.py (SWA structure of exp09: detached rolling KV of 7 chunks + the current chunk, one inner step per 1024-token chunk, band mask on the update path) with the size of every inner step fixed: with --step-norm T the step of every chunk is  new = old - T * g / ||g||  (the direction of the gradient, the global update norm is exactly T in every chunk; no clip, no lr), i.e. a normalised-gradient step. Without --step-norm the run is the unchanged cell_band_ladder run (clipped SGD). inner.sgd_step_fn is replaced inside this process only; DetachedWriter.step (NATS) and BandWriter.step (band arms) call that one function, so both arms get the same rule.
Purpose (Hans 11/18): write sizes with the E2E relative size: T = (cumulative relative path) x ||W_fast|| / 128 with ||W_fast|| = 649.09 (the full gate / up / down matrices of the last 9 layers of Qwen3-4B-Base, --fast fullmlp): 2.0% -> 0.10142, 6.0% -> 0.30426, 12.05% (the E2E NATS path: 87.48 / 725.93 over 128 chunks) -> 0.61110 per chunk. Qwen is a model that was not trained to write at test time, so a size that does not diverge has to be FOUND: for post-hoc judgement of divergence every run stores in its json (a) the per-chunk log (ce, grad_norm, clip_coef, update_norm, raw_update_norm = the norm the clipped lr x scale step would have had) of the unchanged runner, so the maximum loss, the maximum gradient norm and non-finite values can be read afterwards, and (b) the canary NLL (pg11 text, first canary-len tokens, nat / token) after EVERY read exposure in reads[<exposure>]['canary'] (the end-of-run value stays in 'canary'). No divergence rule is applied here.
Output {out}/cband_<arm>_s<seed>.json as cell_band_ladder.py (+ reads[<e>]['canary'] and args.step_norm); W0 files are reused from the out directory when present.
  python3 cell_band_norm.py --step-norm T --world-seed S --arms NATS,NATMASK1024,NATMASK256 --fast fullmlp --reads 16,32,64,128 --out O | --selftest"""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import numpy as np,torch
import cell_band_ladder as CB
import qwen_ladder as QL,qwen_runtime as QR,inner as IN,world_text as WT
from qwen_runtime import lora_params
ST={'T':None,'canids':None,'tok':None,'bos':None,'each':False}
_orig_step=IN.sgd_step_fn;_orig_read=CB.read_all
def sgd_step_norm(model,loss_fn,lr,scale=1.0,clip=1.0):
    T=ST['T']
    if T is None:return _orig_step(model,loss_fn,lr,scale,clip)
    ps=[p for _,p in lora_params(model)];loss=loss_fn();g=torch.autograd.grad(loss,ps)
    gn=torch.sqrt(sum((x.float()**2).sum() for x in g));coef=torch.where(gn<clip,torch.ones_like(gn),clip/torch.clamp(gn,min=1e-30))
    s=(T/gn) if float(gn)>0 else torch.zeros_like(gn);un=torch.zeros((),device=gn.device)
    with torch.no_grad():
        for p,x in zip(ps,g):
            d=(s*x).to(p.dtype);p.sub_(d);un+=(d.float()**2).sum()
    return {'ce':float(loss.detach()),'grad_norm':float(gn),'clip_coef':float(coef),'raw_update_norm':float(lr*scale*coef*gn),'update_norm':float(un.sqrt())}
def read_all_canary(model,ctxs,rows,dev,batch,pad,rd_dtype):
    out=_orig_read(model,ctxs,rows,dev,batch,pad,rd_dtype)
    if ST['each']:out['canary']=float(CB.canary(model,ST['canids'],dev,rd_dtype))
    return out
def install(step_norm,each=True):
    ST['T']=step_norm;ST['each']=each;IN.sgd_step_fn=sgd_step_norm;CB.read_all=read_all_canary
def parser():
    ap=CB.parser();ap.add_argument('--step-norm',type=float,default=0.0,help='fixed global update norm of every inner step (0 = the unchanged clipped SGD rule)');ap.add_argument('--no-canary-each',action='store_true');return ap
def run(a):
    if a.threads:torch.set_num_threads(a.threads)
    install(a.step_norm if a.step_norm>0 else None,each=not a.no_canary_each)
    mdir=Path(a.model) if a.model else (QR.cached_snapshot(QR.MODEL_IDS.get(a.model_id,a.model_id)) if a.tiny else QR.resolve_model(QR.MODEL_IDS.get(a.model_id,a.model_id)))
    tok=WT.load_tokenizer(QR.tokenizer_file(mdir));bos={'endoftext':tok.token_to_id('<|endoftext|>'),'none':None}[a.bos];ST['canids']=WT.gen_ids(tok,bos,a.canary_len)
    CB.run(a)
def selftest():
    sys.argv=[sys.argv[0]]
    import tempfile,shutil
    td=Path(tempfile.mkdtemp())
    base=['--tiny','--device','cpu','--max-chunks','3','--chunk','1024','--reads','1,3','--tail-len','256','--readout-batch','16','--threads','4','--lora-layers','2','--world-seed','161','--lr','1e-2','--rerun','--arms','NATS,NATMASK256']
    a=parser().parse_args(base+['--out',str(td/'n'),'--step-norm','0.25']);torch.manual_seed(0);run(a)
    d={x:json.load(open(td/'n'/f'cband_{x}_s161.json')) for x in ('NATS','NATMASK256')}
    for arm,r in d.items():
        un=[e['update_norm'] for e in r['log']];assert len(un)==3 and all(abs(u-0.25)/0.25<0.01 for u in un),(arm,un)                   # every chunk has exactly the target norm
        assert all(isinstance(r['reads'][e]['canary'],float) for e in ('1','3')) and r['args']['step_norm']==0.25 and set(r['reads']['3'])=={'out','in','canary'}
        assert all(np.isfinite(e['ce']) and e['grad_norm']>0 and 'raw_update_norm' in e for e in r['log'])
    assert abs(d['NATS']['log'][0]['update_norm']-d['NATMASK256']['log'][0]['update_norm'])<0.01 and (d['NATS']['log'][2]['ce']!=d['NATMASK256']['log'][2]['ce'] or d['NATS']['log'][1]['ce']!=d['NATMASK256']['log'][1]['ce'])
    # the unchanged rule without --step-norm: the original clipped SGD (update norm = lr x coef x ||g||)
    a2=parser().parse_args(base+['--out',str(td/'o'),'--arms','NATS']);torch.manual_seed(0);run(a2);r2=json.load(open(td/'o'/'cband_NATS_s161.json'))
    assert all(abs(e['update_norm']-0.01*e['clip_coef']*e['grad_norm'])<1e-6+1e-3*abs(e['update_norm']) for e in r2['log']) and 'canary' in r2['reads']['1']
    IN.sgd_step_fn=_orig_step;CB.read_all=_orig_read;shutil.rmtree(td,ignore_errors=True);print('cell_band_norm selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        a=parser().parse_args();torch.manual_seed(0);run(a)
