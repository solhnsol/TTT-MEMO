"""E55 32.11 label-free sentence masks from the sentence-grid candidates (the top candidates of the 213 grid on seed 61), as a WRITE-experiment input.
Why a separate mask stage: the attention the candidates use is the attention of the PREFIX layers, which are frozen in TTT-E2E (only the suffix is written), so the selection of window sentences for chunk k depends on the stream tokens only and NOT on the write state. The masks for all chunks are therefore computed ONCE per seed with a forward-only pass (no write step) and shared by every write arm (same masks that an online pass before each write would give; checked on the tiny model by U48).
Candidate configurations are fixed in the candidate file of the 213 report (seed 61, top 10); only the token-vote kinds are supported here ('tok' with heads by sharp / copy / induct, optional rare-token query filter, sink / previous-chunk exclusion, k-times trigger; V6 and V7 'tok' configurations): the lean pass stores the maximum window mass and its key index per (layer, head, query token) and nothing else. A configuration that needs the sentence-level arrays (AL / AV / q_ce) is refused with its name.
Nothing of the fact table enters the selection; the fact table is used ONLY for the per-chunk label statistics (recall, precision, hidden share, background false-mask) that are logged and written to the stage report."""
import numpy as np
import e55_uv as UVm,e55_sentsel as SEL
CH=1024;NW=8;WIN=NW*CH
def supported(cfg):
    return cfg.get('sel')=='tok' and not cfg.get('av') and cfg.get('qf') in (None,'rare') and (cfg.get('v6') or cfg.get('v7')) and cfg.get('sel')!='str'
def choose(d,cfg):
    """hidden window sentences [nW] bool of one configuration on the lean arrays of one chunk"""
    assert supported(cfg),('configuration needs arrays the lean mask stage does not keep',cfg.get('name'))
    return (SEL.select_v7 if cfg.get('v7') else SEL.select_heads)(d,cfg)
def lean_arrays(k,ends,all_ids,ids_k,tmax,tkey):
    """dict for e55_sentsel from the lean pass: tmax float16 / tkey int16 [L,H,CH]; window sentence layout and token ids from the stream"""
    q,wsid,wst,wlen=UVm.chunk_sents(ends,k,CH),*UVm.window_sents(ends,k,CH,NW)
    wid=np.full(WIN,-1,np.int32)
    for c_ in range(max(0,k-NW),k):
        off=(c_-k+NW)*CH;wid[off:off+CH]=np.asarray(all_ids[c_*CH:(c_+1)*CH],np.int32)
    key=np.asarray(tkey).astype(np.int64);tw=np.where(wsid[key]>=0,wsid[key],-1).astype(np.int16)
    return {'T_max':np.asarray(tmax),'T_key':np.asarray(tkey),'T_wsid':tw,'q_start':np.asarray([a for a,_ in q],np.int32),'q_len':np.asarray([b-a for a,b in q],np.int32),'w_start':np.asarray(wst,np.int32),'w_len':np.asarray(wlen,np.int32),'w_chunk':(k-NW+np.asarray(wst,np.int32)//CH).astype(np.int32),'win_ids':wid,'q_ids':np.asarray(ids_k,np.int32)}
def lean_pass(L,rt,s3,m,Kc,Vc,ids_k,k):
    """forward-only prefix pass with all chunk tokens as queries and the UNMASKED window -> (tmax float16 [nL,H,CH], tkey int16 [nL,H,CH])"""
    import jax,jax.numpy as jnp
    s3.S3.clear();s3.S3['mode']='att';s3.S3['q0']=jnp.int32(k*CH);s3.S3['keep']=jnp.asarray(np.ones(WIN+CH,bool));s3.S3['att_rows']=jnp.arange(CH,dtype=jnp.int32);s3.S3['att_out']=[];s3.S3['att_noout']=True;s3.S3['att_lean']=[]
    try:
        ids=jnp.asarray(ids_k,dtype=jnp.int32)
        with rt.mesh:
            h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=L.Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
            for l in range(Kc.shape[0]):
                blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
        lean=s3.S3['att_lean']
    finally:s3.S3.clear()
    return np.stack([x[0] for x in lean]),np.stack([x[1] for x in lean])
def compute(L,rt,s3,ends,all_ids,cands,ranks,chunks_meta=None,nch=128,k0=1,log=print,passfn=None):
    """-> (hidden {k: {rank: bool [nW]}}, window sentence lengths {k: [nW]}, stats {rank: pooled label counts}); passfn(k, ids_k) -> (tmax, tkey) replaces the GPU pass in the CPU test"""
    import jax.numpy as jnp
    for r in ranks:assert supported(cands[r-1]),(r,cands[r-1].get('name'))
    hid={};wl={};cnt={r:np.zeros(6) for r in ranks};per={r:{} for r in ranks}
    if passfn is None:
        with rt.mesh:Kh,Vh,_=L.record(np.asarray(all_ids)[:nch*CH])
    for k in range(k0,nch):
        ids_k=np.asarray(all_ids[k*CH:(k+1)*CH])
        if passfn is None:
            with rt.mesh:Kc,Vc=L.ctx_kv(Kh,Vh,k)
            tmax,tkey=lean_pass(L,rt,s3,rt.base,Kc,Vc,ids_k,k)
        else:tmax,tkey=passfn(k,ids_k)
        d=lean_arrays(k,ends,all_ids,ids_k,tmax,tkey);hid[k]={};wl[k]=d['w_len']
        if len(d['w_len'])==0:continue
        for r in ranks:
            h=choose(d,cands[r-1]);hid[k][r]=h
            if chunks_meta is not None:
                import e55_sentdump_labels as LB,e55_sentgrid as G
                lb=LB.labels(chunks_meta,ends,k)
                if k>=NW:c=G.counts(h,d,lb);cnt[r]+=c;per[r][k]=G.metrics(c)
        if k%16==0:log(f'  mask stage chunk {k}: '+', '.join(f'c{r} hidden sentences {int(hid[k][r].sum())}/{len(d["w_len"])}' for r in ranks if r in hid[k]))
    return hid,wl,cnt,per
def save(path,hid,wl):
    d={}
    for k,v in hid.items():
        d[f'wlen{k}']=np.asarray(wl[k],np.int32)
        for r,h in v.items():d[f'h{r}_{k}']=np.asarray(h,bool)
    np.savez_compressed(path,**d)
class SelMask(UVm.UV):
    """UV-compatible object (.keep) that applies the stored hidden window sentences of candidate `rank` (arm SELC<rank>); label-free, the fact table only feeds the per-chunk log"""
    def __init__(self,arm,rank,masks,ends,chunks_meta=None,CH=CH,nw=NW,log=print):
        self.vlog={};self.arm=arm;self.rank=rank;self.masks=masks;self.ends=np.asarray(ends);self.meta=chunks_meta;self.CH=CH;self.nw=nw;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        CH,nw=self.CH,self.nw;W=nw*CH;wsid,wst,wlen=UVm.window_sents(self.ends,k,CH,nw);keep=np.ones(W+CH,bool);info={'uv_arm':self.arm,'uv_window_sents':int(len(wlen)),'sel_rank':self.rank}
        key=f'h{self.rank}_{k}';hid=np.zeros(len(wlen),bool)
        if len(wlen):
            assert key in self.masks and len(self.masks[key])==len(wlen) and np.array_equal(self.masks[f'wlen{k}'],wlen),('mask file does not match this stream at chunk',k)
            hid=np.asarray(self.masks[key],bool)
        hk=np.zeros(W,bool)
        if len(wlen):hk=(wsid>=0)&hid[np.clip(wsid,0,len(wlen)-1)]
        keep[:W]=~hk
        if self.meta is not None:info.update({'lab_'+a:b for a,b in UVm.label_stats(hk,self.meta,k,CH,nw).items()})
        info['uv_hidden_tokens']=int(hk.sum())
        if k%16==0:self.log(f'  {self.arm} chunk {k}: hidden window tokens {int(hk.sum())} ({hk.sum()/max((wsid>=0).sum(),1):.3f} of window)'+(f", fact recall {info['lab_recall']:.2f} precision {info['lab_precision'] if info['lab_precision'] is None else round(info['lab_precision'],2)}" if 'lab_recall' in info else ''))
        return keep,info
