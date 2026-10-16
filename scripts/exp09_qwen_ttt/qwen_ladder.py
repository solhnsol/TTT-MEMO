"""Exp09: the Exp07 write ladder on Qwen3-4B-Base (default; Instruct-2507 via --model-id instruct) with LoRA fast
weights (NTP test-time training). Design approved 2026-10-16.
Levels (same meaning as scripts/exp07_write_ladder/ladder.py):
  L0     native online TTT, one pass; one clipped SGD step per chunk on the LoRA params. --native-ctx detached (default,
         TTT-E2E structure): one forward of chunk k after the detached, carried KV of the previous `--window-chunks`
         chunks (computed with the weights of their own steps, re-rotated to window positions [0, 8192)); --native-ctx
         recompute: the window is recomputed with the current weights each step, gradient through the whole window
  L0mp   L0 re-read for --passes passes, weights carried, every pass restarts at the stream start
  L0mw   context-masked write: chunk k alone (positions from 0, no earlier context), --passes passes
  L0mwfs L0mw with new background filler every pass (pass 1 = original stream)
  L1     offline: isolated exposure sentences ('\\n\\n' + sentence) packed into <= CH-token steps without cross-row
         attention, shuffled every epoch (random.Random(--seed)), --epochs epochs; --variant plain|ctx|irr|fill
Evaluation after every pass/epoch (L0: every --eval-every chunks): per-item margins (gold - foil) under R1 (pg1342
tail) and R1-alt (SQuAD tail), canaries _gen_nll (pg11) and _tail_nll. Output: {OUT}/{LEVEL}_{TAG}_s{SEED}_{A,B}.json
({'log':[...]} as exp07) and {OUT}/meta_{LEVEL}_{TAG}_s{SEED}.json (items, fam_sizes, settings). See README.md."""
import os,sys,json,time,argparse,random,platform
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
import torch
import world_text as WT,qwen_runtime as QR,inner as IN,readout as RO

def build(a):
    dev=torch.device(a.device);mid=QR.MODEL_IDS.get(a.model_id,a.model_id)
    if a.tiny:   # architecture + tokenizer from whatever Qwen3 is cached (no download); weights random
        mdir=Path(a.model) if a.model else (QR.cached_snapshot(mid) or QR.qwen_snapshot_dir(QR.INSTRUCT_ID))
    else:mdir=Path(a.model) if a.model else QR.resolve_model(mid)
    tok=WT.load_tokenizer(QR.tokenizer_file(mdir))
    bos={'endoftext':tok.token_to_id('<|endoftext|>'),'none':None}[a.bos]
    if a.tiny:base=QR.tiny_qwen(mdir,seed=a.tiny_seed,attn=a.attn)
    else:base=QR.load_base(mdir,dev,{'bf16':torch.bfloat16,'fp32':torch.float32}[a.base_dtype],a.attn)
    base=base.to(dev)
    model,lcfg=QR.wrap_lora(base,a.lora_r,a.lora_alpha,tuple(a.lora_targets.split(',')),a.lora_layers if not a.tiny else min(a.lora_layers,base.config.num_hidden_layers))
    return dev,mdir,tok,bos,model,lcfg

class Run:
    def __init__(self,a):
        self.a=a;self.dev,self.mdir,self.tok,self.bos,self.model,self.lcfg=build(a)
        self.W=WT.World(a.world_seed,self.tok,self.bos,tail_len=a.tail_len,gen_len=a.canary_len)
        self.rows=self.W.probe_rows()
        pad=self.tok.token_to_id('<|endoftext|>')
        rd={'bf16':None,'fp32':torch.float32}[a.readout_dtype]
        if a.tiny:rd=None
        self.E=RO.Evaluator(self.W,self.rows,self.dev,a.readout_batch,pad,rd)
        self.watch=QR.watch_names(self.model,all_frozen=a.tiny);self.fhash=QR.frozen_hash(self.model,self.watch)
        self.init_state=QR.lora_state(self.model);self.init_hash=QR.lora_hash(self.model);self.CH=a.chunk
    def reset(self):QR.load_lora_state_(self.model,self.init_state)
    def check_frozen(self):
        h=QR.frozen_hash(self.model,self.watch)
        if h!=self.fhash:raise RuntimeError('frozen weights changed')
    def stream(self,swap):
        ids=self.W.stream_ids(swap);a=self.a
        if a.stream_chunks!='cover':ids=ids[:int(a.stream_chunks)*self.CH]
        if a.max_chunks:ids=ids[:a.max_chunks*self.CH]
        return ids
    def nchunks(self,ids):return (len(ids)+self.CH-1)//self.CH
    def evaluate(self,extra):
        t0=time.monotonic();m=self.E.evaluate(self.model);self.check_frozen()
        return dict(extra,margins=m,lora_hash=QR.lora_hash(self.model),sec_eval=time.monotonic()-t0)
    def step(self,batch,stats):
        st=IN.sgd_step(self.model,batch,self.a.lr,self.a.scale,self.a.clip);stats.append(st);self.cum+=st['update_norm'];return st
    def passes(self,streams,mode,tag):
        """native / masked passes over a list of token streams (one per pass), weights carried."""
        a=self.a;log=[self.evaluate({'epoch':0})];self.cum=0.0
        for ps,ids in enumerate(streams,1):
            stats=[];t0=time.monotonic()
            if mode=='native' and a.native_ctx=='detached':
                Wr=IN.DetachedWriter(self.model,self.CH,a.window_chunks,self.dev)   # fresh KV every pass
                for k in range(self.nchunks(ids)):
                    st=Wr.step(ids,k,a.lr,a.scale,a.clip)
                    if st:stats.append(st);self.cum+=st['update_norm']
                ids_n=None
            else:ids_n=ids
            for k in (range(self.nchunks(ids)) if ids_n is not None else []):
                b=IN.native_batch(ids,k,self.CH,a.window_chunks,self.dev) if mode=='native' else IN.masked_batch(ids,k,self.CH,self.dev)
                if b[2].numel()==0:continue
                self.step(b,stats)
            log.append(self.evaluate({'epoch':ps,'chunks':self.nchunks(ids),'tokens':len(ids),'cum_update_norm':self.cum,'sec_pass':time.monotonic()-t0,
                                      'ce':[s['ce'] for s in stats],'grad_norm':[s['grad_norm'] for s in stats],'clip_coef':[s['clip_coef'] for s in stats]}))
            print(f'  {tag} pass {ps}: {len(stats)} steps, mean ce {sum(s["ce"] for s in stats)/max(len(stats),1):.3f}, cum upd {self.cum:.4f}, {time.monotonic()-t0:.0f}s',flush=True)
        return log
    def run_L0(self,swap):
        a=self.a;ids=self.stream(swap);log=[self.evaluate({'state':0})];self.cum=0.0;stats=[];n=self.nchunks(ids)
        Wr=IN.DetachedWriter(self.model,self.CH,a.window_chunks,self.dev)
        for k in range(n):
            if a.native_ctx=='detached':
                st=Wr.step(ids,k,a.lr,a.scale,a.clip)
                if st:stats.append(st);self.cum+=st['update_norm']
            else:
                b=IN.native_batch(ids,k,self.CH,a.window_chunks,self.dev)
                if b[2].numel():self.step(b,stats)
            if (k+1)%a.eval_every==0 or k+1==n:log.append(self.evaluate({'state':k+1,'cum_update_norm':self.cum,'ce':stats[-1]['ce']}));print(f'  L0 state {k+1}',flush=True)
        return log
    def run_L1(self,swap):
        a=self.a;rows=self.W.l1_rows(swap,a.variant,fill_len=self.CH);rng=random.Random(a.seed);log=[self.evaluate({'epoch':0,'step':0})];nstep=0;self.cum=0.0
        for ep in range(1,a.epochs+1):
            order=list(range(len(rows)));rng.shuffle(order);steps=IN.pack_rows([rows[i] for i in order],self.CH);t0=time.monotonic()
            for s_ in steps:
                st=[];self.step(IN.packed_batch(s_,self.dev),st);nstep+=1
                log.append(dict(st[0],epoch=ep,step=nstep,cum_update_norm=self.cum,tokens=len(s_[0]),loss_tokens=len(s_[2])))
            if ep%a.eval_every_epoch==0 or ep==a.epochs:log.append(self.evaluate({'epoch':ep,'step':nstep,'sec_epoch':time.monotonic()-t0}))
            print(f'  L1 epoch {ep}: {len(steps)} steps, last ce {log[-2]["ce"] if "ce" in log[-2] else float("nan"):.3f}, {time.monotonic()-t0:.0f}s',flush=True)
        return log
    def run(self,level,swap):
        a=self.a;self.reset();assert QR.lora_hash(self.model)==self.init_hash,'LoRA reset failed'
        if level=='L0':return self.run_L0(swap)
        if level=='L1':return self.run_L1(swap)
        ids=self.stream(swap)
        if level=='L0mp':return self.passes([ids]*a.passes,'native','L0mp')
        if level=='L0mw':return self.passes([ids]*a.passes,'masked','L0mw')
        if level=='L0mwfs':return self.passes(self.W.fs_streams(swap,a.passes,len(ids)),'masked','L0mwfs')
        raise ValueError(level)
    def meta(self):
        a=self.a;W=self.W;tl=lambda s:len(WT.enc(self.tok,' '+s).ids)
        sA=self.stream(False);sB=self.stream(True)
        return {'items':W.items,'fam_sizes':{k:len(v) for k,v in W.fam.items()},'lr':a.lr,'clip':a.clip,'scale':a.scale,'level':a.level,
                'epochs':a.epochs,'passes':a.passes,'seed':a.seed,'args':vars(a),'lora':self.lcfg,'model_dir':str(self.mdir),
                'bos':self.bos,'pseudo':W.pseudo,'tail_sha':W.tail_sha,'tail_len':len(W.tail_ids),'alt_len':len(W.alt_ids),'gen_len':len(W.gen_ids),
                'stream_tokens':{'A':len(W.stream_ids(False)),'B':len(W.stream_ids(True))},'stream_used':{'A':len(sA),'B':len(sB)},
                'stream_chunks':{'A':self.nchunks(sA),'B':self.nchunks(sB)},'model_id':QR.MODEL_IDS.get(a.model_id,a.model_id),
                'qwen_token_len':{'names':sorted({tl(t['X']) for t in W.w['triples']}),'places':sorted({tl(t['P']) for t in W.w['triples']}),
                                  'pseudo_names':sorted({tl(x) for x in W.pseudo['persons']}),'pseudo_places':sorted({tl(x) for x in W.pseudo['places']})},
                'n_probe_rows':len(self.rows),'frozen_hash':self.fhash,'versions':{'torch':torch.__version__,'python':platform.python_version(),
                'transformers':__import__('transformers').__version__,'peft':__import__('peft').__version__}}

def parser():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level',choices=['L0','L0mp','L0mw','L0mwfs','L1'],required=True)
    ap.add_argument('--variant',default='plain',choices=['plain','ctx','irr','fill'],help='L1 rows (ladder.l1_rows)')
    ap.add_argument('--world-seed',type=int,default=101);ap.add_argument('--worlds',default='AB')
    ap.add_argument('--passes',type=int,default=8);ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--scale',type=float,default=1.0,help='step multiplier (all levels)')
    ap.add_argument('--clip',type=float,default=1.0);ap.add_argument('--seed',type=int,default=0,help='L1 shuffle seed')
    ap.add_argument('--tag',default='main');ap.add_argument('--device',default='cuda');ap.add_argument('--out',default=str(ROOT/'experiments/exp09_qwen_ttt'))
    ap.add_argument('--model-id',default='base',help="'base' (Qwen/Qwen3-4B-Base, default), 'instruct' (Qwen/Qwen3-4B-Instruct-2507) or a hub id")
    ap.add_argument('--model',default='',help='local model dir (overrides --model-id)')
    ap.add_argument('--native-ctx',default='detached',choices=['detached','recompute'],help='native write context (L0, L0mp)')
    ap.add_argument('--base-dtype',default='bf16',choices=['bf16','fp32']);ap.add_argument('--attn',default='sdpa',choices=['sdpa','eager'])
    ap.add_argument('--lora-r',type=int,default=16);ap.add_argument('--lora-alpha',type=int,default=32)
    ap.add_argument('--lora-targets',default='gate_proj,up_proj,down_proj');ap.add_argument('--lora-layers',type=int,default=9,help='LoRA on the last N layers')
    ap.add_argument('--chunk',type=int,default=1024);ap.add_argument('--window-chunks',type=int,default=7,help='native context: previous chunks')
    ap.add_argument('--stream-chunks',default='63',help="int (63 = JAX stream length, default) or 'cover' = all chunks covering the whole text")
    ap.add_argument('--max-chunks',type=int,default=0,help='truncate the stream (smoke tests)')
    ap.add_argument('--bos',default='endoftext',choices=['endoftext','none'])
    ap.add_argument('--tail-len',type=int,default=7168);ap.add_argument('--canary-len',type=int,default=8192)
    ap.add_argument('--readout-batch',type=int,default=4);ap.add_argument('--readout-dtype',default='fp32',choices=['bf16','fp32'],help='frozen base dtype during readout (logits are always fp32)')
    ap.add_argument('--eval-every',type=int,default=16,help='L0: evaluate every N chunks');ap.add_argument('--eval-every-epoch',type=int,default=1)
    ap.add_argument('--deterministic',action='store_true');ap.add_argument('--tiny',action='store_true',help='tiny random Qwen3 (CPU smoke only)')
    ap.add_argument('--tiny-seed',type=int,default=0);ap.add_argument('--threads',type=int,default=0,help='torch CPU threads (0 = torch default)')
    return ap

def main(argv=None):
    a=parser().parse_args(argv);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    if a.threads:torch.set_num_threads(a.threads)
    torch.manual_seed(0);t0=time.monotonic();R=Run(a)
    meta=R.meta();meta['sec_setup']=time.monotonic()-t0
    json.dump(meta,open(out/f'meta_{a.level}_{a.tag}_s{a.world_seed}.json','w'),indent=1)
    print(f"setup {meta['sec_setup']:.0f}s; stream tokens {meta['stream_tokens']}, chunks used {meta['stream_chunks']}, rows {meta['n_probe_rows']}, LoRA params {R.lcfg['n_params']}",flush=True)
    for w in a.worlds:
        f=out/f'{a.level}_{a.tag}_s{a.world_seed}_{w}.json'
        if f.exists():print('exists',f);continue
        t1=time.monotonic();log=R.run(a.level,w=='B')
        json.dump({'log':log,'sec':time.monotonic()-t1},open(f,'w'));print('saved',f,round(time.monotonic()-t1),'s',flush=True)
if __name__=='__main__':main()
