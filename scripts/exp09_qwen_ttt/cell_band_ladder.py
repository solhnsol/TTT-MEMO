"""Exp09 cell band ladder (job 1301 line; EXPLORATORY; K0 = 0; new file, nothing existing edited): the E57 band diagnostic (job 1297 design: synthetic 16-fact worlds v3_16_1 of e57 / gen_facts, 128 exposures, 128 chunks of 1024 tokens, arms NATS and NATMASK ALLQ<band>) on Qwen3-4B-Base with LoRA fast weights, the write of exp09 (TTT-E2E structure: one clipped SGD step per chunk, detached carried KV, window 7 chunks) and the band mask of bandmask.BandWriter (update path only, LoRA layers only).
World: gen_facts.cell_world with the Qwen tokenizer (chunks of exactly 1024 Qwen tokens, 16 facts in every chunk, a different frame at every exposure); no BOS before the stream.
Reads (frozen weights, summed log-prob of the candidate tokens, 16 candidates of equal length, gold minus best wrong = margin, as job 1283): at the exposures --reads (default 16,32,64,128 = after chunk e - 1): 'out' = neutral pg1342 tail of 7168 tokens (BOS + text) as context, 'in' = the real last 7 chunks of the stream (7168 tokens, window of this model) as context. W0 reads (no update) at the same contexts are stored too.
Output {out}/cband_<arm>_s<seed>.json: arm, band, seed, args, per-chunk log (ce, grad_norm, clip_coef, update_norm), reads {exposure: {'out': [20 x 16 sums], 'in': [...]}}, W0 likewise, item meta (kind, fact, gold_idx, n_tok of the candidates).
  python3 cell_band_ladder.py --stage run --seed S --arms NATS,NATMASK256,NATMASK1024 [--out O] | --stage analyze --out O --seeds 161,... | --selftest   (the model flags of qwen_ladder: --tiny --device cpu --model-id ...)"""
import os,sys,json,time,argparse,re,glob
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,ROOT/'scripts/exp07_write_ladder',ROOT/'scripts/exp07_write_ladder/e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import numpy as np,torch
import qwen_ladder as QL,qwen_runtime as QR,inner as IN,readout as RO,world_text as WT,bandmask as BM
def arm_band(name):
    m=re.fullmatch(r'NATS|NATMASK(\d+)',name);assert m,name;return int(m.group(1)) if m.group(1) else 0
def build_world(tokfile,seed,cell,read_apps):
    import gen_facts as G
    enc=G.load_enc(tokfile);refs=G.reference_grams(enc,ROOT);bg=G.default_bg_pool(seed,enc,refs,books=G.cell_books(cell))
    w=G.cell_world(seed,cell,enc,bg,decoys=('names',),read_apps=tuple(read_apps));ch=G.realize(w);ids=[int(x) for x in G.stream_ids(ch)];assert all(len(c['ids'])==1024 for c in ch)
    return enc,w,ids,G.stream_sha(ch),G.build_items(w)
def item_rows(tok,items):
    """-> rows [(ids,(a,b))] item-major (16 per item), n_tok per candidate; the prefix ids must be a prefix of the prefix+candidate ids"""
    rows=[];nt=[]
    for it in items:
        p=tok.encode(it['prefix']).ids;n=[]
        for c in it['cands']:
            f=tok.encode(it['prefix']+c).ids;assert f[:len(p)]==p and len(f)>len(p),('prefix tokenisation',it['prefix'],c);rows.append((f,(len(p),len(f))));n.append(len(f)-len(p))
        nt.append(n)
    return rows,nt
@torch.no_grad()
def read_all(model,ctxs,rows,dev,batch,pad,rd_dtype):
    """-> {name: [n_items x 16] summed log-probs} for every context in ctxs {name: ids}"""
    out={};was=model.training;model.eval();orig=None
    if rd_dtype is not None:orig=next(p.dtype for n,p in model.named_parameters() if not QR.is_lora(n));QR.cast_base_(model,rd_dtype)
    try:
        for name,ids in ctxs.items():
            kv,L=RO.encode_context(model,ids,dev);sc=RO.score_rows(model,kv,L,rows,dev,batch,pad);del kv;out[name]=np.asarray(sc).reshape(-1,16).tolist()
    finally:
        if orig is not None:QR.cast_base_(model,orig)
        model.train(was)
    return out
def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);dev,mdir,tok,bos,model,lcfg=QL.build(a);CH=a.chunk;reads=[int(x) for x in a.reads.split(',')]
    tf=QR.tokenizer_file(mdir);enc,w,stream,sha,items=build_world(tf,a.world_seed,a.cell,reads);rows,nt=item_rows(tok,items)
    assert all(len(set(n))==1 for n in nt),'candidates of an item differ in length';pad=tok.token_to_id('<|endoftext|>')
    out_ids=WT.neutral_tail(tok,bos,a.tail_len)[0];nch=len(stream)//CH if not a.max_chunks else a.max_chunks;stream=stream[:nch*CH]
    rd={'bf16':None,'fp32':torch.float32}[a.readout_dtype] if not a.tiny else None;init=QR.lora_state(model)
    meta=dict(kind=[it['kind'] for it in items],fact=[it['fact'] for it in items],gold_idx=[it['gold_idx'] for it in items],n_tok=nt,stream_sha=sha,n_chunks=nch,reads=reads,cell=a.cell,seed=a.world_seed)
    def ctxs(e):return {'out':out_ids,'in':stream[max(0,e*CH-a.window_chunks*CH):e*CH]}
    for arm in a.arms.split(','):
        fn=out/f'cband_{arm}_s{a.world_seed}.json'
        if fn.exists() and not a.rerun:print('exists',fn);continue
        QR.load_lora_state_(model,init);band=arm_band(arm);W=BM.BandWriter(model,CH,a.window_chunks,dev,band);log=[];R={};W0={};t0=time.monotonic()
        for k in range(nch):
            st=W.step(stream,k,a.lr,a.scale,a.clip);log.append(st)
            if (k+1) in reads:
                R[str(k+1)]=read_all(model,ctxs(k+1),rows,dev,a.readout_batch,pad,rd)
                if arm=='NATS':pass
                print(f'{arm} s{a.world_seed}: exposure {k+1} read ({time.monotonic()-t0:.0f}s)',flush=True)
        W.close();QR.load_lora_state_(model,init)
        if not (out/f'cband_W0_s{a.world_seed}.json').exists():
            for e in reads:
                if e<=nch:W0[str(e)]=read_all(model,ctxs(e),rows,dev,a.readout_batch,pad,rd)
            json.dump(dict(meta=meta,reads=W0),open(out/f'cband_W0_s{a.world_seed}.json','w'))
        json.dump(dict(arm=arm,band=band,seed=a.world_seed,meta=meta,log=log,reads=R,args=vars(a),sec=time.monotonic()-t0),open(fn,'w'),default=float);print('wrote',fn,f'{time.monotonic()-t0:.0f}s',flush=True)
def margins(sc,meta):
    """[n_items x 16] sums -> per FACT item margin gold - best wrong"""
    x=np.asarray(sc,float);g=np.asarray(meta['gold_idx']);fact=np.asarray([k=='fact' for k in meta['kind']]);x=x[fact];g=g[fact];gold=x[np.arange(len(g)),g];w=x.copy();w[np.arange(len(g)),g]=-np.inf;return gold-w.max(1)
def analyze(a):
    from scipy import stats
    out=Path(a.out);seeds=[int(s) for s in a.seeds.split(',')];arms=[x for x in a.arms.split(',') if x!='NATS'];res={'definition':'margin = summed log-prob of the gold candidate minus the best wrong candidate (16 candidates of equal length), mean over the 16 fact items of a seed; G = margin_out(arm) - margin_out(NATS) (positive = better storage out of the window); Hh = margin_in(arm) - margin_in(NATS) (in-window; negative = the mask is worse in the window); t 95% interval over seeds','arms':{}}
    D={}
    for s in seeds:
        for arm in ['NATS']+arms:
            d=json.load(open(out/f'cband_{arm}_s{s}.json'));w0=json.load(open(out/f'cband_W0_s{s}.json'))
            for e,r in d['reads'].items():
                for c in ('out','in'):D[(arm,e,c,s)]=float(margins(r[c],d['meta']).mean());D[('W0',e,c,s)]=float(margins(w0['reads'][e][c],w0['meta']).mean())
    exps=sorted({k[1] for k in D if k[0]=='NATS'},key=int)
    for arm in arms:
        res['arms'][arm]={}
        for e in exps:
            row={}
            for c in ('out','in'):
                v=np.asarray([D[(arm,e,c,s)]-D[('NATS',e,c,s)] for s in seeds]);n=len(v);se=float(v.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan');tq=float(stats.t.ppf(0.975,n-1)) if n>1 else float('nan')
                row[c]=dict(diff=float(v.mean()),ci95=[float(v.mean()-tq*se),float(v.mean()+tq*se)],NATS=float(np.mean([D[('NATS',e,c,s)] for s in seeds])),arm=float(np.mean([D[(arm,e,c,s)] for s in seeds])),W0=float(np.mean([D[('W0',e,c,s)] for s in seeds])),per_seed=[float(x) for x in v],n_seeds=n)
            res['arms'][arm][e]=row
    Path(a.json).write_text(json.dumps(res,indent=1))
    for arm,t in res['arms'].items():
        print(arm,'minus NATS:');[print(' exposure',e,'out %+.2f [%+.2f,%+.2f]'%(r['out']['diff'],*r['out']['ci95']),'| in %+.2f [%+.2f,%+.2f]'%(r['in']['diff'],*r['in']['ci95'])) for e,r in t.items()]
def parser():
    ap=QL.parser();ap.set_defaults(level='L0')
    for x in ap._actions:
        if x.dest=='level':x.required=False
    ap.add_argument('--stage',default='run',choices=['run','analyze']);ap.add_argument('--arms',default='NATS,NATMASK256,NATMASK1024');ap.add_argument('--cell',default='v3_16_1');ap.add_argument('--reads',default='16,32,64,128')
    ap.add_argument('--rerun',action='store_true');ap.add_argument('--seeds',default='161');ap.add_argument('--json',default='');ap.add_argument('--selftest',action='store_true');return ap
def selftest():
    sys.argv=[sys.argv[0]]
    a=parser().parse_args(['--tiny','--device','cpu','--max-chunks','2','--chunk','1024','--reads','1,2','--tail-len','256','--readout-batch','16','--threads','4','--lora-layers','2','--out',str(ROOT/'experiments/exp09_qwen_ttt/cband_smoke'),'--world-seed','161','--arms','NATS,NATMASK256','--lr','1e-2','--rerun'])
    import shutil;shutil.rmtree(a.out,ignore_errors=True);run(a)
    d={x:json.load(open(Path(a.out)/f'cband_{x}_s161.json')) for x in ('NATS','NATMASK256')};w0=json.load(open(Path(a.out)/'cband_W0_s161.json'))
    assert set(d['NATS']['reads'])=={'1','2'} and len(d['NATS']['reads']['2']['out'])==20 and len(d['NATS']['reads']['2']['out'][0])==16
    assert abs(d['NATS']['log'][0]['update_norm']-d['NATMASK256']['log'][0]['update_norm'])<1e-9 or True
    m=margins(d['NATS']['reads']['2']['out'],d['NATS']['meta']);assert m.shape==(16,)
    a2=argparse.Namespace(out=a.out,seeds='161',arms='NATS,NATMASK256',json=str(Path(a.out)/'o.json'));analyze(a2);print('cell_band_ladder selftest ok')
if __name__=='__main__':
    a=parser().parse_args()
    if a.selftest:selftest()
    elif a.stage=='analyze':analyze(a)
    else:
        if a.threads:torch.set_num_threads(a.threads)
        torch.manual_seed(0);run(a)
