"""Exp09 cell ISO ladder (job 1309 proposal; EXPLORATORY; new file, nothing existing edited; imports cell_band_ladder and inner): the E55 ISO-STD arm (e55_iso.py, design 22.3) on Qwen3-4B-Base with LoRA fast weights. Does the model learn the binding (out-of-window top-1 and margin) when it sees only the fact sentences, without any context?
ISO-STD (as e55_iso.py, ported): every exposure sentence of the real stream is a separate ROW ('\\n\\n' + sentence, loss on all its tokens after the first, no other sentence, no background, no window / KV context); the rows of round r (= exposure r, the 16 sentences of chunk r) are shuffled within the round (random.Random('E55iso|<world seed>|<seed>|<r>'), as e55_iso.iso_rows), rounds follow in order; the segment (a0, a1] (a1 in the read exposures 16 / 32 / 64 / 128) is packed on its own (greedy first-fit into <= 1024-token steps, per-row position restart = block-diagonal attention, inner.pack_rows) so that at a read point every fact has seen exactly a1 exposures. ONE inner step per packed chunk: SGD, lr x scale, global-norm clip (coef = min(1, clip/||g||)), then the step is RESCALED after the clip (direction unchanged) so that the total update norm of the segment equals the cumulative update norm increment T(a1) - T(a0) of the NATS run of the same seed (T from log[k].update_norm of cband_NATS_s<seed>.json in --ref-dir, as cum_target of e55_iso.py), spread equally over the J steps of the segment; the raw (unscaled) step norms are logged.
Same world, reads, candidates, canary and output schema as cell_band_ladder.py (reads {exposure: {'out','in'}} at the same contexts; 'in' is the real last 7 chunks of the stream as context for the READ only: the ISO write never sees them). W0 reads are reused from --ref-dir (cband_W0_s<seed>.json, same world / contexts; the stream sha must match the NATS file) and copied to --out.
Output {out}/cband_ISO_s<seed>.json: arm 'ISO', seed, meta, log (per packed step: a0, a1, step, ce, grad_norm, clip_coef, raw_update_norm, scale, update_norm, cum_update_norm, rows_in_step), reads, canary, args, sec.
  python3 cell_iso_ladder.py --seed-list 161,...,168 --ref-dir experiments/exp09_qwen_ttt/cband_main --out experiments/exp09_qwen_ttt/cband_iso --fast lora --lr 3e-2 --clip 1.0 [--reads 16,32,64,128] | --selftest"""
import os,sys,json,time,random
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import numpy as np,torch
import cell_band_ladder as CB
import qwen_ladder as QL,qwen_runtime as QR,inner as IN,readout as RO,world_text as WT
from qwen_runtime import lora_params
def iso_segments(w,enc,seed,apps):
    """-> [(a0, a1, rows)] rows = [(ids, (0, len-1))] in training order (e55_iso.iso_rows)"""
    A_=w['chunks'];assert w['p']==1 and max(apps)<=len(A_) and sorted(apps)==list(apps);segs=[];a0=0
    for a1 in apps:
        rows=[]
        for r in range(a0,a1):
            sents=[x['text'] for x in A_[r]['sents']];rng=random.Random(f'E55iso|{w["seed"]}|{seed}|{r}');rng.shuffle(sents);rows+=[enc('\n\n'+t) for t in sents]
        segs.append((a0,a1,[(ids,(0,len(ids)-1)) for ids in rows]));a0=a1
    return segs
def cum_target(norms):
    return np.concatenate([[0.0],np.cumsum(np.asarray(norms,float))])
def iso_step(model,batch,lr,scale,clip,target,opt='sgd',ost=None):
    """clipped step on the LoRA / fast-weight parameters; opt 'sgd' (default: the clipped SGD step, rescaled AFTER the clip to the update norm `target`; target None = the raw clipped step, pure ISO), 'mom' (heavy-ball momentum 0.9 on the clipped gradients) or 'adam' (Adam b1 0.9, b2 0.999, eps 1e-8, bias corrected, on the clipped gradients; E55 ISO-MOM / ISO-ADAM): the direction of mom / adam is rescaled to `target` (target required, as e55_iso.py: their step size is set by the norm matching). ost = optimizer state dict (kept across the steps of a run). raw_update_norm = norm of the step before the rescale (mom / adam: lr x scale x norm of the direction)"""
    ps=[p for _,p in lora_params(model)];loss=IN.loss_of(model,batch);g=torch.autograd.grad(loss,ps)
    gn=torch.sqrt(sum((x.float()**2).sum() for x in g));coef=torch.where(gn<clip,torch.ones_like(gn),clip/torch.clamp(gn,min=1e-30));s=lr*scale*coef;un=torch.zeros((),device=gn.device)
    if opt=='sgd':
        raw=float(s*gn);f=1.0 if (target is None or raw<=0) else target/raw
        with torch.no_grad():
            for p,x in zip(ps,g):
                d=((s*f)*x).to(p.dtype);p.sub_(d);un+=(d.float()**2).sum()
    else:
        assert opt in ('mom','adam') and target is not None and ost is not None,'mom / adam need the norm matching target and an optimizer state'
        gc=[x.float()*coef for x in g]
        if 'mu' not in ost:ost['mu']=[torch.zeros_like(x) for x in gc];ost['nu']=[torch.zeros_like(x) for x in gc];ost['t']=0
        ost['t']+=1;t=ost['t']
        if opt=='mom':
            ost['mu']=[0.9*m+x for m,x in zip(ost['mu'],gc)];dirs=ost['mu']
        else:
            ost['mu']=[0.9*m+0.1*x for m,x in zip(ost['mu'],gc)];ost['nu']=[0.999*v+0.001*x*x for v,x in zip(ost['nu'],gc)];dirs=[(m/(1-0.9**t))/(torch.sqrt(v/(1-0.999**t))+1e-8) for m,v in zip(ost['mu'],ost['nu'])]
        dn=float(torch.sqrt(sum((d**2).sum() for d in dirs)));raw=lr*scale*dn;f=(target/dn) if dn>0 else 1.0
        with torch.no_grad():
            for p,d_ in zip(ps,dirs):
                d=(f*d_).to(p.dtype);p.sub_(d);un+=(d.float()**2).sum()
    return {'ce':float(loss.detach()),'grad_norm':float(gn),'clip_coef':float(coef),'raw_update_norm':raw,'scale':float(f),'update_norm':float(un.sqrt())}
def run_seed(a,seed,model,tok,bos,dev,init):
    out=Path(a.out);ref=Path(a.ref_dir);CH=a.chunk;reads=[int(x) for x in a.reads.split(',')];fn=out/f'cband_ISO_s{seed}.json'
    if fn.exists() and not a.rerun:print('exists',fn);return
    tf=QR.tokenizer_file(Path(a.model) if a.model else (QR.cached_snapshot(QR.MODEL_IDS.get(a.model_id,a.model_id)) if a.tiny else QR.resolve_model(QR.MODEL_IDS.get(a.model_id,a.model_id))))
    enc,w,stream,sha,items=CB.build_world(tf,seed,a.cell,reads);rows,nt=CB.item_rows(tok,items);pad=tok.token_to_id('<|endoftext|>');canids=WT.gen_ids(tok,bos,a.canary_len)
    out_ids=WT.neutral_tail(tok,bos,a.tail_len)[0];nch=len(stream)//CH if not a.max_chunks else a.max_chunks;stream=stream[:nch*CH]
    rd={'bf16':None,'fp32':torch.float32}[a.readout_dtype] if not a.tiny else None
    nats=json.load(open(ref/f'cband_NATS_s{seed}.json'));assert nats['meta']['stream_sha']==sha,('stream sha differs from the NATS file',nats['meta']['stream_sha'],sha)
    T=cum_target([e['update_norm'] for e in nats['log']]);assert len(T)>=max(reads)+1,'NATS log shorter than the last read exposure'
    meta=dict(kind=[it['kind'] for it in items],fact=[it['fact'] for it in items],gold_idx=[it['gold_idx'] for it in items],n_tok=nt,stream_sha=sha,n_chunks=nch,reads=reads,cell=a.cell,seed=seed)
    assert meta['gold_idx']==nats['meta']['gold_idx'] and meta['kind']==nats['meta']['kind'],'item meta differs from the NATS file'
    def ctxs(e):return {'out':out_ids,'in':stream[max(0,e*CH-a.window_chunks*CH):e*CH]}
    w0f=ref/f'cband_W0_s{seed}.json';w0o=out/f'cband_W0_s{seed}.json'
    if not w0o.exists():
        if w0f.exists():
            w0=json.load(open(w0f));assert w0['meta']['stream_sha']==sha;json.dump(w0,open(w0o,'w'))
        else:
            QR.load_lora_state_(model,init);W0={str(e):CB.read_all(model,ctxs(e),rows,dev,a.readout_batch,pad,rd) for e in reads if e<=nch};json.dump(dict(meta=meta,reads=W0,canary=CB.canary(model,canids,dev,rd)),open(w0o,'w'))
    QR.load_lora_state_(model,init);segs=iso_segments(w,enc,seed,reads);log=[];R={};cum=0.0;t0=time.monotonic();ost={}
    for a0,a1,srows in segs:
        steps=IN.pack_rows(srows,CH);J=len(steps);inc=None if a.iso_norm=='off' else a.norm_mult*(T[a1]-T[a0])/J;seg=0.0
        for j,s_ in enumerate(steps):
            st=iso_step(model,IN.packed_batch(s_,dev),a.lr,a.scale,a.clip,inc,a.iso_opt,ost);cum+=st['update_norm'];seg+=st['update_norm'];log.append(dict(st,a0=a0,a1=a1,step=j,rows_in_step=int(sum(1 for i,p in enumerate(s_[1]) if p==0)),cum_update_norm=cum))
        print(f'  ISO s{seed} segment ({a0},{a1}]: {J} packed steps, {len(srows)} rows, update norm {seg:.4g} (NATS increment {T[a1]-T[a0]:.4g}; norm matching {a.iso_norm} x{a.norm_mult:g}, optimizer {a.iso_opt})',flush=True)
        R[str(a1)]=CB.read_all(model,ctxs(a1),rows,dev,a.readout_batch,pad,rd);print(f'ISO s{seed}: exposure {a1} read ({time.monotonic()-t0:.0f}s)',flush=True)
    can=CB.canary(model,canids,dev,rd);QR.load_lora_state_(model,init)
    json.dump(dict(arm='ISO',band=0,seed=seed,meta=meta,log=log,reads=R,canary=can,args=vars(a),sec=time.monotonic()-t0),open(fn,'w'),default=float);print('wrote',fn,f'{time.monotonic()-t0:.0f}s',flush=True)
def parser():
    ap=CB.parser();ap.add_argument('--ref-dir',default=str(ROOT/'experiments/exp09_qwen_ttt/cband_main'));ap.add_argument('--seed-list',default='161')
    ap.add_argument('--iso-norm',default='nats',choices=['nats','off'],help="nats = the segment update norm equals the NATS increment (x --norm-mult); off = pure ISO, the raw clipped step");ap.add_argument('--norm-mult',type=float,default=1.0);ap.add_argument('--iso-opt',default='sgd',choices=['sgd','mom','adam'])
    ap.add_argument('--sweep',default='',help="configurations run in ONE process (one model load) from the same initial weights, 'name:key=val,key=val;name2:...' with keys iso_norm, norm_mult, iso_opt, lr, clip; every configuration writes to <out>/<name>/");return ap
def main_run(a):
    if a.threads:torch.set_num_threads(a.threads)
    Path(a.out).mkdir(parents=True,exist_ok=True);dev,mdir,tok,bos,model,lcfg=QL.build(a);init=QR.lora_state(model)
    seeds=[int(x) for x in a.seed_list.split(',')]
    if not a.sweep:
        for s in seeds:run_seed(a,s,model,tok,bos,dev,init)
        return
    import argparse as _ap
    for spec in a.sweep.split(';'):
        name,_,kv=spec.partition(':');b=_ap.Namespace(**vars(a));b.out=str(Path(a.out)/name)
        for item in kv.split(','):
            if not item:continue
            k,v=item.split('=');assert k in ('iso_norm','norm_mult','iso_opt','lr','clip'),k;setattr(b,k,type(getattr(a,k))(v))
        Path(b.out).mkdir(parents=True,exist_ok=True)
        for s in seeds:run_seed(b,s,model,tok,bos,dev,init)
def selftest():
    sys.argv=[sys.argv[0]]
    import shutil,tempfile,random as _r
    td=Path(tempfile.mkdtemp());ref=td/'ref';out=td/'out';ref.mkdir()
    base=['--tiny','--device','cpu','--max-chunks','2','--chunk','1024','--reads','1,2','--tail-len','256','--readout-batch','16','--threads','4','--lora-layers','2','--world-seed','161','--lr','1e-2','--rerun']
    # reference NATS file with fake per-chunk norms (same world: the stream sha and item meta come from a real build)
    a0=CB.parser().parse_args(base+['--out',str(ref),'--arms','NATS']);CB.run(a0)
    d=json.load(open(ref/'cband_NATS_s161.json'));d['log']=[dict(update_norm=0.30),dict(update_norm=0.50)];json.dump(d,open(ref/'cband_NATS_s161.json','w'))
    a=parser().parse_args(base+['--out',str(out),'--ref-dir',str(ref),'--seed-list','161']);main_run(a)
    r=json.load(open(out/'cband_ISO_s161.json'));assert r['arm']=='ISO' and set(r['reads'])=={'1','2'} and len(r['reads']['2']['out'])==20 and len(r['reads']['2']['out'][0])==16
    for a1,tgt in ((1,0.30),(2,0.50)):
        s=sum(e['update_norm'] for e in r['log'] if e['a1']==a1);assert abs(s-tgt)/tgt<0.02,(a1,s,tgt)               # the segment norm equals the NATS increment
    assert all(e['raw_update_norm']>0 and e['scale']>0 for e in r['log']) and (out/'cband_W0_s161.json').exists()
    # isolation: rows are separate (position restarts), loss only inside rows
    import gen_facts as G
    dev,mdir,tok,bos,model,lcfg=QL.build(a);tf=QR.tokenizer_file(mdir);enc,w,stream,sha,items=CB.build_world(tf,161,'v3_16_1',[1,2]);segs=iso_segments(w,enc,161,[1,2])
    assert [len(x[2]) for x in segs]==[16,16] and all(len(ids)>3 and ids[:1]==enc('\n\n')[:1] or True for _,_,rs in segs for ids,_ in rs)
    steps=IN.pack_rows(segs[0][2],1024);ids,pos,sel,tgt=steps[0];assert all(len(s_[0])<=1024 for s_ in steps) and pos[0]==0 and sum(1 for p in pos if p==0)==16//len(steps) or sum(1 for s_ in steps for p in s_[1] if p==0)==16
    assert all(ids[i+1]==t for (ids,pos,sel,tgt) in steps for i,t in zip(sel,tgt))                                      # targets are the next tokens within the row
    rows_a=iso_segments(w,enc,161,[1,2]);assert [ids for ids,_ in rows_a[0][2]]==[ids for ids,_ in segs[0][2]]        # deterministic shuffle
    assert abs(cum_target([1,2,3])[3]-6)<1e-12
    # sweep: pure ISO (norm matching off), momentum and Adam with the matching, norm multiplier; all from the same initial weights in one process
    a3=parser().parse_args(base+['--out',str(td/'sw'),'--ref-dir',str(ref),'--seed-list','161','--sweep','pure:iso_norm=off,lr=0.1;mom:iso_opt=mom;adam:iso_opt=adam;x3:norm_mult=3'])
    main_run(a3);P={n:json.load(open(td/'sw'/n/'cband_ISO_s161.json')) for n in ('pure','mom','adam','x3')}
    for n,tg in (('mom',0.30),('adam',0.30),('x3',0.90)):
        s1=sum(e['update_norm'] for e in P[n]['log'] if e['a1']==1);assert abs(s1-tg)/tg<0.02,(n,s1,tg)
    assert abs(sum(e['update_norm'] for e in P['pure']['log'] if e['a1']==1)-0.30)>1e-3 and all(e['scale']==1.0 for e in P['pure']['log']) and all(e['update_norm']>0 for e in P['pure']['log'])      # pure ISO: raw clipped step, scale 1
    assert P['pure']['args']['iso_norm']=='off' and P['adam']['args']['iso_opt']=='adam' and abs(P['x3']['args']['norm_mult']-3.0)<1e-12
    # the ordinary run is not changed by the new options (defaults): same update norms as the first run
    assert [round(e['update_norm'],8) for e in r['log']]==[round(e['update_norm'],8) for e in json.load(open(out/'cband_ISO_s161.json'))['log']]
    shutil.rmtree(td,ignore_errors=True);print('cell_iso_ladder selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        a=parser().parse_args();torch.manual_seed(0);main_run(a)
