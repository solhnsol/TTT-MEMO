"""Qwen3-4B-Base reduced ladder (job 1367 'qwen_ladder'; GPU; NEW file; implementation thread 11/28; design E59 [S2821-GENERAL] 2b, Hans 11/28 22:33 KST "사다리를 qwen에서도 재현해 보는 게 먼저"; EXPLORATORY; imports e59_qwen_run / e59_qwen_hide and the E2E mask code, nothing existing edited).
World U12 (32 facts = 16 read + 16 extra, one sentence per fact and chunk, 128 chunks of 1024 Qwen tokens; seeds 231-238), read items of the E2E ladder (e59_kvpos2.items2: 16 x 16 candidates = 12 exposed + 4 never-exposed fresh places). Write = the E59 Qwen rule of jobs 1322 / 1334 (fast weights = gate/up/down of the last 9 layers, fp32; fixed per-chunk update norm T = --step-norm, direction of the gradient only). Window = --window-chunks 8 (the Hans 8-chunk rule; the earlier Qwen ports used 7). Every mask acts in all 36 decoder layers.
Arms (E2E names):
  P0s   SAMEVIS: sentence rows cannot see earlier-chunk sentences of OTHER facts (e59_hidearms OTHERONLY); carried KV natural; loss = whole chunk.
  P1s   C0: sentence rows cannot see any earlier-chunk fact sentence (e59_u12 SAME_U1U2 = the E2E C0 keep, checked); carried KV natural; loss = whole chunk.
  P1d   P1s write + the carried KV recorded at processing time with the D rule (e59_relay.keep_Drec: no row sees any fact-sentence key of earlier chunks or of its own chunk; positions kept), with the weights at the start of the chunk.
  L2b   P1d window + loss over the 16 read-fact sentence tokens only + current chunk reduced to the 16 read sentences, each isolated (e59_early1.keep_L 'L2b').
  ISO1  the 16 read-fact sentences of the chunk as isolated rows, one fixed-norm step (the E59 Qwen ISO arm on the read facts only; no window).
Reads (e59_qwen_run): R1 out = the neutral tail, in = the last 8 chunks (8-chunk window), at --reads, 16 x 16 summed span log-probs, canary.
  python3 e59_qwen_ladder.py --world-seed 231 --cell u12_k32 --fast fullmlp --lr 1.0 --clip 1.0 --step-norm 0.30426 --window-chunks 8 --arms P0s,P1s,P1d,L2b,ISO1 --reads 8,16,32,64,96,128 --readout-dtype fp32 --out O | --selftest | --mask-check"""
import sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'exp07_write_ladder'));sys.path.insert(0,str(HERE.parent/'exp07_write_ladder'/'e57'))
import numpy as np
LADDER=('P0s','P1s','P1d','L2b','ISO1');NREAD=16;CH=1024
# C0 = e59_u12 SAME_U1U2 (e59_hidearms BOTH is SAME_U1: it leaves the 16 extra facts visible in U12)
WRITE_UV={'P0s':'OTHERONLY','P1s':'C0','P1d':'C0','L2b':'C0'};D_KV=('P1d','L2b')
def read_only(w):
    """copy of the world whose chunks list only the 16 read facts' sentences (ISO1 rows)"""
    return dict(w,chunks=[dict(c,sents=[s for s in c['sents'] if int(s['fact'])<NREAD]) for c in w['chunks']])
class KeepD:
    """uv object: keep() = e59_relay.keep_Drec (the D record rule)"""
    def __init__(self,chunks):self.chunks=chunks;self.name='D'
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        from e59_relay import keep_Drec;return keep_Drec(self.chunks,k,np.asarray(keep_all,bool)),{'masked_pairs':0}
class KeepNat:
    """uv object: the natural keep (everything in the window visible; with the sliding limit of LadderWriter = the E2E natural attention)"""
    name='NAT'
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):return np.broadcast_to(np.asarray(keep_all,bool),(CH,8*CH+CH)).copy(),{'masked_pairs':0}
class KeepL2b:
    def __init__(self,inner,chunks):self.inner=inner;self.chunks=chunks;self.name='L2b'
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        from e59_early1 import keep_L;kp,info=self.inner.keep(k,m,Kc,Vc,ids_k,keep_all);return keep_L(np.asarray(kp,bool),self.chunks[k],'L2b'),info
def read_loss_positions(chunk):
    m=np.zeros(CH,bool)
    for x in chunk['sents']:
        if int(x['fact'])<NREAD:a,b=int(x['start']),int(x['end']);m[max(a-1,0):b-1]=True
    return np.flatnonzero(m)
LAST={}
def install():
    import torch,e59_qwen_run as E,e59_qwen_hide as QH,e59_hidearms as H,cell_band_ladder as CB,e59_u12 as U
    U.install()
    _bw=CB.build_world
    def build_world(tokfile,seed,cell,read_apps):
        import gen_facts as G
        from e59_kvpos2 import items2
        if cell!=U.CELL:return _bw(tokfile,seed,cell,read_apps)
        enc=G.load_enc(tokfile);refs=G.reference_grams(enc,E.ROOT if hasattr(E,'ROOT') else CB.ROOT);bg=G.default_bg_pool(seed,enc,refs,books=G.cell_books(cell))
        w=G.cell_world(seed,cell,enc,bg,decoys=('names',),read_apps=tuple(read_apps));ch=G.realize(w);ids=[int(x) for x in G.stream_ids(ch)];assert all(len(c['ids'])==1024 for c in ch)
        UX=G.fresh_places(w,enc,4);LAST['w']=w;return enc,w,ids,G.stream_sha(ch),items2(w,UX)
    CB.build_world=build_world;E.CB.build_world=build_world
    import world_text as WT_
    _nt=WT_.neutral_tail
    def neutral_tail(tok,bos,length=None):
        """E2E parity (e55_run.out_tail_8192): the R1 context is BOS + the SQuAD dev paragraphs [:400] without any name / place of the world (e55_run.forbid_list), exactly 8192 tokens"""
        w=LAST.get('w')
        if w is None:return _nt(tok,bos,length) if length else _nt(tok,bos)
        fb=[x['X'] for x in w['facts']+w['ctrl']]+[x['P'] for x in w['facts']]+list(w['unexposed'])+[x['X'] for k_ in ('names','decoy') if k_ in w for x in w[k_]['facts']]+[x['P'] for k_ in ('names','decoy') if k_ in w for x in w[k_]['facts']]
        text=WT_.squad_text(fb);ids=WT_._pre(bos)+WT_.enc(tok,text).ids;assert len(ids)>=8192;return ids[:8192],text,'squad8192'
    WT_.neutral_tail=neutral_tail;E.WT.neutral_tail=neutral_tail
    class LadderWriter(QH.HideWriter):
        def __init__(self,model,CH_,win,device,uv,seed,uv_kv=None):super().__init__(model,CH_,win,device,uv,seed);self.uv_kv=uv_kv if uv_kv is not None else KeepNat()
        def allowed(self,L,off,device):
            """the arm keep AND the E2E sliding limit (key j visible to query i only if (off+i) - j < 8 x CH, as the E2E prefix / suffix attention with sliding_window_size 8192)"""
            a_=super().allowed(L,off,device);qi=(off+torch.arange(L,device=device))[:,None];kj=torch.arange(off+L,device=device)[None,:];return a_&((qi-kj)<8*self.CH)[None,None]
        def forward_kv(self,ids):
            """carried KV of the chunk recorded with uv_kv (KeepNat = natural, KeepD = the D rule) and the sliding limit in all layers (no gradient)"""
            uv0=self.uv;self.uv=self.uv_kv;self._mk=None
            try:
                dec,_=E.parts(self.model);cache,off=self.past();L=ids.shape[0];pos=(off+torch.arange(L,device=ids.device))[None];self._off=off;self.active=True
                try:hs=dec(input_ids=ids[None],position_ids=pos,past_key_values=cache,use_cache=True).last_hidden_state[0]
                finally:self.active=False
                kv=[(l.keys[:,:,-L:].detach(),l.values[:,:,-L:].detach()) for l in cache.layers];return hs,kv,off
            finally:self.uv=uv0;self._mk=None
    E.LadderWriter=LadderWriter
    def run_arm_ladder(a,arm,w,stream,k_range,model,dev,T,fix,CH_,sep,log=print,ST=None):
        if arm=='ISO1':
            yield from E._orig_run_arm(a,'ISO',read_only(w),stream,k_range,model,dev,T,fix,CH_,sep,log,ST);return
        ch_all=w['chunks'];uv=U.uv_for('SAME_U1U2',ch_all,log) if WRITE_UV[arm]=='C0' else H.uv_for(WRITE_UV[arm],ch_all,int(a.world_seed),log)
        if arm=='L2b':uv=KeepL2b(uv,ch_all)
        assert a.window_chunks==8,'the ladder uses the 8-chunk window';W=LadderWriter(model,CH_,a.window_chunks,dev,uv,a.world_seed,uv_kv=KeepD(ch_all) if arm in D_KV else None)
        try:
            for k in k_range:
                ch,sp,fpos=E.chunk_data(w,stream,k,CH_);c0=k*CH_;n=len(stream);e=min((k+1)*CH_,n);last=min(e,n-1);ids_c=torch.tensor(stream[c0:e],dtype=torch.long,device=dev);fh=E.K.ids_hash(E.K.fact_ids(ch));fp=torch.tensor(fpos,dtype=torch.long,device=dev)
                rp=torch.tensor(read_loss_positions(ch),dtype=torch.long,device=dev);tgt=torch.tensor(stream[c0+1:last+1],dtype=torch.long,device=dev);box={};W.set_chunk(k)
                with torch.no_grad():_,kv,pos=W.forward_kv(ids_c)
                W.set_chunk(k)
                def lf():
                    hs=W.forward_masked(ids_c);nl=E.nll_of(model,None,None,torch.arange(last-c0,device=dev),tgt,hs=hs);box['fact']=float(nl[fp].mean().detach());box['read_fact']=float(nl[rp].mean().detach());box['all']=float(nl.mean().detach())
                    if arm=='L2b':box['ntok']=int(rp.numel());return nl[rp].mean()
                    box['ntok']=int(nl.numel());return nl.mean()
                rec=fix.step(lf,a.lr,a.scale,a.clip);W.push(kv,pos)
                rec.update(k=k,fact_ce=box['fact'],read_fact_ce=box['read_fact'],all_ce=box['all'],n_loss_tokens=box['ntok'],fact_hash=fh,chunk_hash=hashlib.sha256(np.asarray(stream[c0:e],np.int64).tobytes()).hexdigest()[:16],hook_layers=len(W.layer_ids),kv_rule=('D' if arm in D_KV else 'natural'));yield k,rec
        finally:W.close()
    E._orig_run_arm=E.run_arm;E.ARMS=tuple(E.ARMS)+tuple(x for x in LADDER if x not in E.ARMS)
    def dispatch(a,arm,*r,**kw):return run_arm_ladder(a,arm,*r,**kw) if arm in LADDER else E._orig_run_arm(a,arm,*r,**kw)
    E.run_arm=dispatch;E.run.__globals__['run_arm']=dispatch;return E
def mask_check(seeds=(231,)):
    """CPU (no model): BOTH == the E2E C0 keep (SAME_U1U2) on U12; the Qwen key-axis mapping of HideWriter.allowed with an 8-chunk cache keeps every E2E column; L2b / D keeps as in E2E"""
    import torch,e59_hidearms as H,e59_u12 as U,e59_qwen_hide as QH,e59_shadow as SH,qwen_runtime as QR
    from e59_kvpos2 import CONDS
    E=install();WIN=8*CH;tf=QR.tokenizer_file(QR.qwen_snapshot_dir(QR.BASE_ID))
    for seed in seeds:
        enc,w,ids,sha,items=E.CB.build_world(tf,seed,U.CELL,(8,16,32,64,96,128));ch=w['chunks'];assert len(items)==16 and all(len(it['cands'])==16 for it in items) and all(len(c['ids'])==CH for c in ch);c0=SH.mk_uv(CONDS['C0'],w,seed,lambda *x:None);both=U.uv_for('SAME_U1U2',ch,lambda *x:None);sv=H.uv_for('OTHERONLY',ch,seed,lambda *x:None);sv_e2e=SH.mk_uv(CONDS['SAMEVIS'],w,seed,lambda *x:None)
        for k in (1,8,9,64,127):
            base=(k*CH-WIN+np.arange(WIN+CH))>=0;A=np.asarray(c0.keep(k,None,None,None,None,base)[0],bool);B=np.asarray(both.keep(k,None,None,None,None,base)[0],bool);assert (A==B).all(),('C0 differs',seed,k);assert (np.asarray(sv.keep(k,None,None,None,None,base)[0],bool)==np.asarray(sv_e2e.keep(k,None,None,None,None,base)[0],bool)).all(),('SAMEVIS differs',seed,k)
            for uv in (both,sv,KeepL2b(both,ch),KeepD(ch),KeepNat()):
                Wr=object.__new__(E.LadderWriter);Wr.CH=CH;Wr.win=8;Wr.uv=uv;Wr._mk=None;Wr.k=k;off=min(k,8)*CH
                M=Wr.allowed(CH,off,torch.device('cpu'))[0,0].numpy();kp=np.asarray(uv.keep(k,None,None,None,None,base)[0],bool);tri=(np.arange(off+CH)[None,:]-off)<=np.arange(CH)[:,None]
                sl=(off+np.arange(CH))[:,None]-np.arange(off+CH)[None,:]<WIN;ref=np.concatenate([kp[:,WIN-off:WIN],kp[:,WIN:]],axis=1)&tri&sl;assert (M==ref).all(),(seed,k,uv.name)      # = the E2E attention mask (keep, causal, distance < 8192) on the Qwen key axis
        rp=read_loss_positions(ch[5]);assert len(rp)==sum(int(x['end'])-max(int(x['start'])-1,0)-1 for x in ch[5]['sents'] if int(x['fact'])<NREAD)
        ro=read_only(w);assert all(len(c['sents'])==NREAD for c in ro['chunks']) and len(w['chunks'][0]['sents'])==32
    print('e59_qwen_ladder mask_check ok: C0 and SAMEVIS keeps == E2E on the Qwen U12 world, 8-chunk key mapping of BOTH / L2b / D keeps, read loss positions, read-only ISO world',seeds)
def selftest():
    """tiny Qwen-shaped model on CPU, U12 world with the Qwen tokenizer: every ladder arm runs 3 chunks; chunk 0 of P1s == P1d (nothing carried yet); chunk 1 differs (D carried KV); L2b loss tokens = read-fact positions; ISO1 rows = 16"""
    import json,tempfile,shutil,math,torch
    E=install();sys.argv=[sys.argv[0]];td=Path(tempfile.mkdtemp())
    base=['--tiny','--device','cpu','--max-chunks','3','--chunk','1024','--reads','1,3','--tail-len','256','--readout-batch','16','--threads','8','--lora-layers','99','--world-seed','231','--cell','u12_k32','--lr','1e-2','--rerun','--step-norm','0.25','--window-chunks','8']
    a=E.parser().parse_args(base+['--out',str(td/'o'),'--arms',','.join(LADDER)]);torch.manual_seed(0);E.run(a);R={x:json.load(open(td/'o'/f'e59q_{x}_s231.json')) for x in LADDER}
    for x,r in R.items():s=r['steps'];assert len(s)==3 and all(abs(e['applied_norm']-0.25)<=1e-4*0.25+1e-7 and math.isfinite(e['ce']) for e in s),x;assert np.asarray(r['reads']['3']['out']).shape==(16,16) and np.asarray(r['reads']['3']['in']).shape==(16,16),x
    assert abs(R['P1s']['steps'][0]['ce']-R['P1d']['steps'][0]['ce'])<1e-5,'chunk 0: P1d = P1s'
    assert abs(R['P1s']['steps'][1]['ce']-R['P1d']['steps'][1]['ce'])>1e-7,'chunk 1: the D carried KV must change the write'
    assert len(json.load(open(td/'o'/'cband_W0_s231.json'))['reads']['1']['out'])==16
    assert R['L2b']['steps'][0]['n_loss_tokens']<R['P1d']['steps'][0]['n_loss_tokens'] and R['L2b']['steps'][0]['kv_rule']=='D' and R['P0s']['steps'][0]['kv_rule']=='natural'
    shutil.rmtree(td,ignore_errors=True);print('e59_qwen_ladder selftest ok (tiny model, U12 Qwen world, 5 arms)')
if __name__=='__main__':
    if '--mask-check' in sys.argv:mask_check(tuple(range(231,239)))
    elif '--selftest' in sys.argv:selftest()
    else:
        E=install();import torch;a=E.parser().parse_args();torch.manual_seed(0);E.run(a)
