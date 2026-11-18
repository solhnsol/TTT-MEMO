"""E59 write-signal decomposition on Qwen3-4B-Base (new file; thread-server 11/18; EXPLORATORY; nothing existing edited; imports cell_band_ladder / inner / qwen_runtime and the E2E helpers e59_srv_core.py of scripts/exp07_write_ladder). Fast weights = the full gate / up / down MLP matrices of the last 9 layers (--fast fullmlp, fp32), world seeds 161 / 162 (cell v3_16_1, Qwen tokens, 128 chunks, one inner step per chunk = exposure = round), every step is a normalised-gradient step of the FIXED global norm T (--step-norm, default 0.30426 = 6.0% cumulative path of ||W_fast|| = 649.09 over 128 chunks; 2% = 0.10142 for the sensitivity run), i.e. every arm takes the DIRECTION of its gradient only (the clipped lr-x-gradient of the original rule is only logged as 'raw_norm' = lr scale coef ||g||; ratio raw / T) -- the same size rule for every arm, no reference file needed (the NAT arm is the reference, its applied norm is T in every chunk).
Arms: NAT (the existing NATS write: detached rolling KV of 7 chunks + the current chunk, loss over all targets), NATF (NAT context, loss on the fact target tokens only), CHK (chunk alone: position ids from 0, no earlier chunk, loss over all targets), CHKF (chunk alone, fact targets only), FACT (the chunk's 16 fact sentences, each sep + sentence ids, in stream order, ONE row, causal), ISO (the 16 sentences as isolated rows packed in one chunk with per-row position restart = block-diagonal attention (transformers packed-sequence mask; checked in --selftest: the logits of a row do not change with the other rows), loss = ONE mean over all target tokens).
Target set of F / FACT / ISO = the fact sentence tokens (e59_srv_core.target_mask / fact_row / isolated_mask; sep is an input only; fact token ids are sliced from the stream chunk once, hash logged per step). Loss of every arm = ONE mean over its target tokens. Per step: ce (all-target mean at the weights before the step, NAT-F/CHK-F: mean over the loss tokens), fact_ce (mean nll of the fact target tokens), raw_norm, ratio raw / T (flag noise_direction when raw < 1e-3 T, the step is NOT skipped), applied norm, net displacement ||W_k - W_0|| and path length, squared-norm occupancy per parameter tensor, fact hash, and a fixed coordinate sketch of the update (262144 coordinates, seed 0) in <arm>_s<seed>.sketch.npy for cosines between arms. Reads (out-of-window neutral tail / in-window of the last 7 chunks, 16 candidates, summed span log-prob) at exposures 16 / 32 / 64 / 128 and the canary after every read, W0 reads reused from --ref-dir (cband_W0_s<seed>.json) or computed. Divergence flag (reported, nothing excluded): non-finite, last-8 mean ce > 4.0, final canary change > +1.0.
  python3 e59_qwen_run.py --world-seed 161 --fast fullmlp --lr 1.0 --clip 1.0 --step-norm 0.30426 --arms NAT,NATF,CHK,CHKF,FACT,ISO --reads 16,32,64,128 --out O [--ref-dir D] | --selftest"""
import sys,json,time,hashlib,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
import numpy as np,torch,torch.nn.functional as F
import cell_band_ladder as CB
import qwen_ladder as QL,qwen_runtime as QR,inner as IN,world_text as WT
from qwen_runtime import parts,lora_params
import e59_srv_core as K
ARMS=('NAT','NATF','CHK','CHKF','FACT','ISO');NS=262144
def parser():
    ap=CB.parser();ap.add_argument('--step-norm',type=float,default=0.30426);ap.add_argument('--ref-dir',default='');return ap
class Fixed:
    """fixed-norm step with the per-step records; one instance per arm run"""
    def __init__(s,model,T,dev):
        s.model=model;s.T=T;s.ps=[p for _,p in lora_params(model)];s.names=[n for n,_ in lora_params(model)];s.W0=[p.detach().clone() for p in s.ps];g=np.random.default_rng(0);tot=sum(p.numel() for p in s.ps);s.idx=[]
        for p in s.ps:c=min(p.numel(),max(1,int(round(NS*p.numel()/tot))));s.idx.append(torch.tensor(np.sort(g.choice(p.numel(),size=c,replace=False)),dtype=torch.long,device=p.device))
        s.sketch=[];s.steps=[];s.path=0.0
    def step(s,loss_fn,lr,scale,clip):
        loss=loss_fn();g=torch.autograd.grad(loss,s.ps);gn=torch.sqrt(sum((x.float()**2).sum() for x in g))
        if not torch.isfinite(gn) or float(gn)==0:raise FloatingPointError('zero / non-finite gradient')
        coef=float(torch.where(gn<clip,torch.ones_like(gn),clip/torch.clamp(gn,min=1e-30)));raw=lr*scale*coef*float(gn);f=s.T/float(gn);sq=[];sk=[]
        with torch.no_grad():
            for p,x,i in zip(s.ps,g,s.idx):
                d=(f*x).to(p.dtype);p.sub_(d);df=d.float();sq.append(float((df**2).sum()));sk.append(df.reshape(-1)[i].cpu())
            disp=math.sqrt(sum(float(((p.float()-w.float())**2).sum()) for p,w in zip(s.ps,s.W0)))
        un=math.sqrt(sum(sq));s.path+=un;s.sketch.append(torch.cat(sk).numpy().astype(np.float32));tot=sum(sq)
        rec=dict(ce=float(loss.detach()),grad_norm=float(gn),clip_coef=coef,raw_norm=raw,ratio=raw/s.T,flag='noise_direction' if raw<1e-3*s.T else None,applied_norm=un,net_displacement=disp,path_length=s.path,occupancy={n:v/tot for n,v in zip(s.names,sq)});s.steps.append(rec);return rec
def nll_of(model,ids,pos,sel,tgt,hs=None):
    dec,head=parts(model)
    if hs is None:hs=dec(input_ids=ids[None],position_ids=pos[None],use_cache=False).last_hidden_state[0]
    return F.cross_entropy(head(hs[sel]).float(),tgt,reduction='none')
def chunk_data(w,stream,k,CH):
    ch=w['chunks'][k];sp=K.fact_spans(ch['sents']);fpos=np.flatnonzero(K.target_mask(sp,CH));return ch,sp,fpos
def run_arm(a,arm,w,stream,k_range,model,dev,T,fix,CH,sep,log=print,ST=None):
    """one arm over chunks -> yields (k, rec) after every step; reads are done by the caller"""
    W=IN.DetachedWriter(model,CH,a.window_chunks,dev) if arm in ('NAT','NATF') else None
    for k in k_range:
        ch,sp,fpos=chunk_data(w,stream,k,CH);c0=k*CH;n=len(stream);e=min((k+1)*CH,n);last=min(e,n-1);ids_c=torch.tensor(stream[c0:e],dtype=torch.long,device=dev);fh=K.ids_hash(K.fact_ids(ch));fp=torch.tensor(fpos,dtype=torch.long,device=dev)
        if arm in ('NAT','NATF'):
            tgt=torch.tensor(stream[c0+1:last+1],dtype=torch.long,device=dev);box={}
            def lf():
                hs,kv,off=W.forward(ids_c);box['kv']=kv;box['pos']=off;nl=nll_of(model,None,None,torch.arange(last-c0,device=dev),tgt,hs=hs);box['fact']=float(nl[fp].mean().detach());box['all']=float(nl.mean().detach());box['ntok']=int(fp.numel()) if arm=='NATF' else int(nl.numel())
                return nl[fp].mean() if arm=='NATF' else nl.mean()
            rec=fix.step(lf,a.lr,a.scale,a.clip)
            if 'kv' not in box:
                with torch.no_grad():_,box['kv'],box['pos']=W.forward(ids_c)
            W.push(box['kv'],box['pos'])
        elif arm in ('CHK','CHKF'):
            ids=ids_c;pos=torch.arange(len(ids),device=dev);tgt=torch.tensor(stream[c0+1:last+1],dtype=torch.long,device=dev);box={}
            def lf():
                nl=nll_of(model,ids,pos,torch.arange(last-c0,device=dev),tgt);box['fact']=float(nl[fp].mean().detach());box['all']=float(nl.mean().detach());box['ntok']=int(fp.numel()) if arm=='CHKF' else int(nl.numel());return nl[fp].mean() if arm=='CHKF' else nl.mean()
            rec=fix.step(lf,a.lr,a.scale,a.clip)
        else:
            if arm=='FACT':rid,rm=K.fact_row(ch,sep);ids_l=list(rid);pos_l=list(range(len(rid)));sel_l=list(np.flatnonzero(rm))
            else:
                rows=K.iso_rows(ch,sep);ii,seg,loc,mk=K.isolated_mask(rows,CH);m_=int((seg>=0).sum());ids_l=[int(x) for x in ii[:m_]];pos_l=[int(x) for x in loc[:m_]];sel_l=list(np.flatnonzero(mk[:m_]))
            assert [ids_l[p+1] for p in sel_l]==[t for s_ in K.fact_ids(ch) for t in s_],'target token ids differ from the fact token ids'
            ids=torch.tensor(ids_l,dtype=torch.long,device=dev);pos=torch.tensor(pos_l,dtype=torch.long,device=dev);sel=torch.tensor(sel_l,dtype=torch.long,device=dev);tgt=ids[sel+1];box={}
            def lf():
                nl=nll_of(model,ids,pos,sel,tgt);box['fact']=float(nl.mean().detach());box['all']=box['fact'];box['ntok']=int(nl.numel());return nl.mean()
            rec=fix.step(lf,a.lr,a.scale,a.clip)
        rec.update(k=k,fact_ce=box['fact'],all_ce=box['all'],n_loss_tokens=box['ntok'],fact_hash=fh,chunk_hash=hashlib.sha256(np.asarray(stream[c0:e],np.int64).tobytes()).hexdigest()[:16]);yield k,rec
def run(a):
    if a.threads:torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);dev,mdir,tok,bos,model,lcfg=QL.build(a);CH=a.chunk;reads=[int(x) for x in a.reads.split(',')];seed=a.world_seed
    tf=QR.tokenizer_file(mdir);enc,w,stream,sha,items=CB.build_world(tf,seed,a.cell,reads);rows,nt=CB.item_rows(tok,items);assert all(len(set(n))==1 for n in nt);pad=tok.token_to_id('<|endoftext|>');canids=WT.gen_ids(tok,bos,a.canary_len)
    out_ids=WT.neutral_tail(tok,bos,a.tail_len)[0];nch=len(stream)//CH if not a.max_chunks else a.max_chunks;stream=stream[:nch*CH];sep=list(w['sep'])
    rd={'bf16':None,'fp32':torch.float32}[a.readout_dtype] if not a.tiny else None;init=QR.lora_state(model)
    meta=dict(kind=[it['kind'] for it in items],fact=[it['fact'] for it in items],gold_idx=[it['gold_idx'] for it in items],n_tok=nt,stream_sha=sha,n_chunks=nch,reads=reads,cell=a.cell,seed=seed)
    def ctxs(e):return {'out':out_ids,'in':stream[max(0,e*CH-a.window_chunks*CH):e*CH]}
    w0o=out/f'cband_W0_s{seed}.json';w0f=Path(a.ref_dir)/f'cband_W0_s{seed}.json' if a.ref_dir else None
    if not w0o.exists():
        if w0f is not None and w0f.exists():
            w0=json.load(open(w0f));assert w0['meta']['stream_sha']==sha;json.dump(w0,open(w0o,'w'))
        else:
            QR.load_lora_state_(model,init);W0={str(e):CB.read_all(model,ctxs(e),rows,dev,a.readout_batch,pad,rd) for e in reads if e<=nch};json.dump(dict(meta=meta,reads=W0,canary=CB.canary(model,canids,dev,rd)),open(w0o,'w'))
    w0c=json.load(open(w0o))['canary']
    for arm in a.arms.split(','):
        assert arm in ARMS,arm;fn=out/f'e59q_{arm}_s{seed}.json'
        if fn.exists() and not a.rerun:print('exists',fn);continue
        QR.load_lora_state_(model,init);fix=Fixed(model,a.step_norm,dev);R={};t0=time.monotonic();log=[];err=None
        try:
            for k,rec in run_arm(a,arm,w,stream,range(nch),model,dev,a.step_norm,fix,CH,sep):
                log.append(rec)
                if (k+1) in reads:
                    R[str(k+1)]=CB.read_all(model,ctxs(k+1),rows,dev,a.readout_batch,pad,rd);R[str(k+1)]['canary']=float(CB.canary(model,canids,dev,rd));print(f'{arm} s{seed}: exposure {k+1} read ({time.monotonic()-t0:.0f}s)',flush=True)
        except FloatingPointError as ex:err=repr(ex)
        can=float(CB.canary(model,canids,dev,rd));steps=fix.steps;last8=float(np.mean([e['ce'] for e in steps[-8:]])) if steps else float('nan');nf=err is None and all(math.isfinite(e[kk]) for e in steps for kk in ('ce','grad_norm','applied_norm'))
        div=bool((not nf) or last8>4.0 or can-w0c>1.0)
        if fix.sketch:np.save(out/f'e59q_{arm}_s{seed}.sketch.npy',np.stack(fix.sketch))
        json.dump(dict(arm=arm,seed=seed,meta=meta,steps=steps,log=steps,reads=R,canary=can,canary_w0=w0c,step_norm=a.step_norm,diverged=div,error=err,last8_ce=last8,sec=time.monotonic()-t0,args=vars(a)),open(fn,'w'),default=float);print('wrote',fn,f'{time.monotonic()-t0:.0f}s','DIVERGED' if div else '',flush=True)
    for h in []:pass
def selftest():
    sys.argv=[sys.argv[0]];import shutil,tempfile
    td=Path(tempfile.mkdtemp());base=['--tiny','--device','cpu','--max-chunks','3','--chunk','1024','--reads','1,3','--tail-len','256','--readout-batch','16','--threads','4','--lora-layers','2','--world-seed','161','--lr','1e-2','--rerun','--step-norm','0.25']
    a=parser().parse_args(base+['--out',str(td/'o'),'--arms',','.join(ARMS)]);torch.manual_seed(0);run(a);R={x:json.load(open(td/'o'/f'e59q_{x}_s161.json')) for x in ARMS}
    for x,r in R.items():
        s=r['steps'];assert len(s)==3,x;assert all(abs(e['applied_norm']-0.25)<=1e-4*0.25+1e-7 for e in s),(x,[e['applied_norm'] for e in s]);assert all(math.isfinite(e['ratio']) and e['n_loss_tokens']>0 and e['fact_ce']>0 for e in s)
        assert abs(sum(s[0]['occupancy'].values())-1)<1e-5 and set(r['reads'])=={'1','3'} and 'canary' in r['reads']['3'] and s[2]['net_displacement']<=s[2]['path_length']+1e-6 and (np.load(td/'o'/f'e59q_{x}_s161.sketch.npy').shape[0]==3)
    for x in ARMS[1:]:assert [e['fact_hash'] for e in R[x]['steps']]==[e['fact_hash'] for e in R['NAT']['steps']],'fact hash differs'
    assert abs(R['NAT']['steps'][0]['raw_norm']-R['CHK']['steps'][0]['raw_norm'])<=1e-12+1e-7*abs(R['NAT']['steps'][0]['raw_norm']) and abs(R['NAT']['steps'][0]['ce']-R['CHK']['steps'][0]['ce'])<1e-6,'chunk 0: NAT and CHK must be the same computation'
    assert R['NAT']['steps'][1]['ce']!=R['CHK']['steps'][1]['ce'],'CHK must differ from NAT once a window exists'
    # no cross-sentence leakage in the packed ISO row: logits of row 1 alone == logits of row 1 inside the packed row (tiny model)
    dev=torch.device('cpu');a2=parser().parse_args(base+['--out',str(td/'o2'),'--arms','NAT']);dv,mdir,tok,bos,model,lcfg=QL.build(a2);model.eval()
    A_=list(range(11,31));B_=list(range(40,52));dec,head=parts(model)
    with torch.no_grad():
        la=head(dec(input_ids=torch.tensor([A_],device=dev),position_ids=torch.arange(len(A_),device=dev)[None],use_cache=False).last_hidden_state[0]).float()
        ids=torch.tensor([A_+B_],device=dev);pos=torch.tensor([list(range(len(A_)))+list(range(len(B_)))],device=dev);lb=head(dec(input_ids=ids,position_ids=pos,use_cache=False).last_hidden_state[0]).float()
        lb2=head(dec(input_ids=torch.tensor([B_],device=dev),position_ids=torch.arange(len(B_),device=dev)[None],use_cache=False).last_hidden_state[0]).float()
    assert float((lb[:len(A_)]-la).abs().max())<1e-3 and float((lb[len(A_):]-lb2).abs().max())<1e-3,'packed ISO rows leak across rows (position-restart packing is not block diagonal)'
    shutil.rmtree(td,ignore_errors=True);print('e59_qwen_run selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        a=parser().parse_args();torch.manual_seed(0);run(a)
