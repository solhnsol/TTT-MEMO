"""E55 design 22.1 stage 1: fine-tuning upper-bound probe. Only the FAST weights (the TTT inner-update targets, last quarter of the layers' MLPs) are trained, everything else is frozen; AdamW (weight decay 0, global-norm clip 1.0, fp32 master weights + fp32 Adam state, learning rate = argument), ONE optimizer step per 1024-token chunk, loss = next-token loss over the whole chunk (the same lm_loss and the same write path as the TTT inner step, so FT-NAT is 'the TTT stream with Adam instead of the inner SGD rule'). Arms: FT-NAT (real window), FT-MASK (earlier fact sentences hidden from attention in the training forward only: exactly the SENT mask of job 183), FT-ISO (the isolated rows of stage 0, same Adam). The 64-chunk stream is repeated epoch after epoch; state (weights, master, Adam moments) carries on; at the START of every epoch the carried suffix KV window is reset (a fresh start of the same stream, no cross-epoch window). Reads (frozen weights, no further training) after the epochs in --read-epochs: out-of-window + in-window (the last 8 chunks) + POSITIVE CONTROL (the same 16-candidate question asked in the frame of the fact's latest exposure sentence that has the name before the place, i.e. a frame the model was trained on). The loss of the place tokens (the fact tokens) is logged in every epoch (chunk k % 8 == 7 for the stream arms, all rows for FT-ISO) before the update of that chunk.
CPU tested with a tiny model (e55_units.py U27); the 3B GPU path is NOT run."""
import time,random,math
import numpy as np
import e55_core as C,e55_iso as I,e55_swap as E
import s3
jax,jnp,eqx=s3.jax,s3.jnp,s3.eqx
LD=s3.LD
CH=1024;NW=E.NW
class Ft55(I.Iso55):
    def build_ft(self):
        rt=self.rt;MetaModel=LD.MetaModel;clone_state=LD.clone_state;Batch=self.Batch
        B1,B2,EPS=0.9,0.999,1e-8
        @eqx.filter_jit
        def adam_apply(master,mu,nu,t,g,lr,clip):
            """AdamW (wd 0) on the fp32 master: global-norm clip first, bias-corrected moments"""
            f32=jnp.float32;g=jax.tree.map(lambda x:x.astype(f32),g);gn=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(g)));coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));g=jax.tree.map(lambda x:x*coef,g)
            mu=jax.tree.map(lambda a,x:B1*a+(1-B1)*x,mu,g);nu=jax.tree.map(lambda a,x:B2*a+(1-B2)*x*x,nu,g);t1=t+1.0;c1=1-B1**t1;c2=1-B2**t1
            master=jax.tree.map(lambda p,a,b:p-lr*(a/c1)/(jnp.sqrt(b/c2)+EPS),master,mu,nu);return master,mu,nu,gn,coef
        @eqx.filter_jit
        def place_ce_chunk(m,st_suf,c,pref_,kmask,lmask):
            """forward-only CE of the chunk with the loss restricted to `lmask` (positions i that predict a place token), same write path / mask as the training forward"""
            c2=eqx.tree_at(lambda b:b.loss_masks,c,lmask)
            s3.S3SUF['kmask']=kmask
            try:(_l,(ce,_n,_s))=MetaModel.lm_loss(m,c2,st_suf,prefix_outputs=pref_)
            finally:s3.S3SUF.clear()
            return ce
        @eqx.filter_jit
        def place_ce_iso(m,ids_,tg_,mk_,pref_,seg_,loc_):
            packing=LD.packing;packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
            try:
                seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_);(_l,(ce,_n,_s))=MetaModel.lm_loss(m,seq,clone_state(rt.state_suffix),prefix_outputs=pref_)
            finally:packing.PACK_SUF.clear()
            return ce
        self.adam_apply=adam_apply;self.place_ce_chunk=place_ce_chunk;self.place_ce_iso=place_ce_iso
    def __init__(self,*a,**k):
        super().__init__(*a,**k);self.build_ft()
    def ft_init(self,m):
        f32=jnp.float32;master=jax.tree.map(lambda x:x.astype(f32),m.inner_parameters());z=jax.tree.map(jnp.zeros_like,master);return {'master':master,'mu':z,'nu':jax.tree.map(jnp.zeros_like,master),'t':jnp.asarray(0.0,f32)}
    def ft_update(self,m,opt,g,lr,clip=1.0):
        """one AdamW step on the fp32 master; -> (new model with the master cast to the inner parameter dtype, opt, grad norm, clip coef, update norm in the model dtype)"""
        master,mu,nu,gn,coef=self.adam_apply(opt['master'],opt['mu'],opt['nu'],opt['t'],g,jnp.asarray(lr,jnp.float32),jnp.asarray(clip,jnp.float32))
        newp=jax.tree.map(lambda x,p:x.astype(p.dtype),master,m.inner_parameters());un=self.norm_between(newp,m);opt={'master':master,'mu':mu,'nu':nu,'t':opt['t']+1.0}
        return self.set_inner(m,newp),opt,float(gn),float(coef),un
def place_positions(ids,sents,facts,enc_place):
    """boolean [len(ids)]: True at the tokens of the place name inside every fact sentence of the chunk (token sequence enc_place(' ' + P) searched inside the sentence span; first occurrence). -> (mask over token positions, number of sentences where the place was not found)"""
    mask=np.zeros(len(ids),bool);miss=0;ids=list(ids)
    for x in sents:
        pl=enc_place(' '+facts[x['fact']]['P']);L=len(pl);seg=ids[x['start']:x['end']];hit=None
        for i in range(len(seg)-L+1):
            if seg[i:i+L]==pl:hit=x['start']+i;break
        if hit is None:miss+=1;continue
        mask[hit:hit+L]=True
    return mask,miss
def loss_positions(tokmask):
    """position i predicts token i+1: loss mask over positions for a mask over target tokens"""
    lm=np.zeros(len(tokmask),bool);lm[:-1]=tokmask[1:];return lm
def pc_items(world,items):
    """positive-control items: same candidates / gold as the standard items, question prefix = the text of the fact's LATEST exposure sentence up to the place (the name must stand before the place, otherwise the next older exposure is used): a frame the model was trained on. -> (items, n_facts_without_valid_frame)"""
    A_=world['chunks'];facts={f['id']:f for f in world['facts']};out=[];bad=0
    for it in items:
        it2=dict(it)
        if it['kind']=='fact':
            f=facts[it['fact']];pre=None
            for c in reversed(A_):
                for x in c['sents']:
                    if x['fact']!=f['id']:continue
                    t=x['text'];i=t.find(f['P'])
                    if i>0 and f['X'] in t[:i]:pre=t[:i].rstrip();break
                if pre is not None:break
            if pre is None:bad+=1;pre=it['prefix']
            it2['prefix']=pre
        out.append(it2)
    return out,bad
def ft_stream(L,rt,ids,chunks_meta,arm,lr,epochs,read_epochs,read=None,log=print,facts=None,enc_place=None,nch=None,clip=1.0,weights0=None,fact_ks=None):
    """FT-NAT / FT-MASK over the stream `ids` (nch chunks of 1024 tokens), `epochs` passes, Adam state carried. read(epoch, m) is called after the epochs in read_epochs. -> {'log': per step, 'epochs': per epoch summary, 'model': final}"""
    assert arm in ('FT-NAT','FT-MASK');n=nch or len(ids)//CH;ids=np.asarray(ids)[:n*CH];ar_='SENT' if arm=='FT-MASK' else 'NAT'
    fact_ks=set(range(7,n,8)) if fact_ks is None else set(fact_ks);L.chunks_meta=chunks_meta;L.n_masked_keys=0;m=weights0 if weights0 is not None else rt.base;opt=L.ft_init(m);st_all=LD.clone_state(rt.state_all);out={'log':[],'epochs':[],'arm':arm,'lr':lr};miss_total=0
    with rt.mesh:
        seq=L._stream_seq(ids,n);pref=rt.prefix(rt.base,seq);Kh,Vh,_=L.record(ids);recs={'real':(Kh,Vh)}
        for ep in range(1,epochs+1):
            sts={'real':LD.clone_state(rt.state_suffix)};fact_ce=[];ces=[];t_ep=time.monotonic()           # fresh carried window at every epoch start
            for k in range(n):
                t0=time.monotonic();old=m;c=rt.chunk(seq,k);Kc,Vc,keep,st_w,pl=L.window_inputs(ar_,k,1,recs,sts)
                kj=jnp.asarray(keep);pw=L.prefix_masked(m,jnp.asarray(ids[k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kj)
                if facts is not None and k in fact_ks and chunks_meta[k]['sents']:
                    pm,ms=place_positions(ids[k*CH:(k+1)*CH],chunks_meta[k]['sents'],facts,enc_place);miss_total+=ms
                    if pm.any():fact_ce.append(float(L.place_ce_chunk(m,st_w,c,pw,kj,jnp.asarray(loss_positions(pm)))))
                g,ce=L.grad_mask(m,st_w,c,pw,kj);m,opt,gn,coef,un=L.ft_update(m,opt,g,lr,clip);ces.append(float(ce))
                sts['real']=L.read_state(old,sts['real'],c,pref[k*CH:(k+1)*CH])
                out['log'].append({'epoch':ep,'k':k,'ce':float(ce),'gnorm':gn,'coef':coef,'update_norm':un,'n_masked_keys':int(L.n_masked_keys),'sec':time.monotonic()-t0})
            es={'epoch':ep,'mean_ce':float(np.mean(ces)),'fact_token_ce':float(np.mean(fact_ce)) if fact_ce else None,'n_fact_ce':len(fact_ce),'place_not_found':miss_total,'sec':time.monotonic()-t_ep,'cum_update_norm':float(sum(e['update_norm'] for e in out['log']))};out['epochs'].append(es)
            log(f'  {arm} lr {lr:g} epoch {ep}/{epochs}: mean CE {es["mean_ce"]:.4f}, place-token CE {es["fact_token_ce"]}, {es["sec"]:.0f}s')
            if read is not None and ep in read_epochs:read(ep,m)
    out['model']=m;return out
def ft_iso(L,rt,segs_by_epoch,lr,read_epochs,read=None,log=print,facts=None,enc_place=None,clip=1.0,weights0=None):
    """FT-ISO: epoch e = one pass over all isolated rows (segs_by_epoch[e-1] = packed-row list, reshuffled every epoch), Adam as in ft_stream. place-token CE over all rows of the epoch (before the update of the chunk)."""
    packing=LD.packing;W=int(rt.cfg.model.sliding_window_size);m=weights0 if weights0 is not None else rt.base;opt=L.ft_init(m);out={'log':[],'epochs':[],'arm':'FT-ISO','lr':lr}
    with rt.mesh:
        for ep,(rows,place_rows) in enumerate(segs_by_epoch,1):
            chunks,where=packing.pack_rows(rows);mks=[np.zeros(1024,bool) for _ in chunks];pms=[np.zeros(1024,bool) for _ in chunks]
            for c_,a_,b_ in where:mks[c_][a_:b_]=True
            if place_rows is not None:
                for (c_,a_,b_),pr in zip(where,place_rows):pms[c_][a_+pr[0]:a_+pr[1]]=True      # pr = (first, last+1) place target positions inside the row
            fce=[];ces=[];t_ep=time.monotonic()
            for (ids,seg,loc),mk,pm in zip(chunks,mks,pms):
                t0=time.monotonic();tg=np.concatenate([ids[1:],[rt.eos]]).astype(np.int32);pref=L.prefix_iso(rt.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
                if pm.any():fce.append(float(L.place_ce_iso(m,jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(pm),pref,jnp.asarray(seg),jnp.asarray(loc))))
                g,ce=L.iso_grad(m,jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc));m,opt,gn,coef,un=L.ft_update(m,opt,g,lr,clip);ces.append(float(ce))
                out['log'].append({'epoch':ep,'ce':float(ce),'gnorm':gn,'coef':coef,'update_norm':un,'sec':time.monotonic()-t0})
            es={'epoch':ep,'mean_ce':float(np.mean(ces)),'fact_token_ce':float(np.mean(fce)) if fce else None,'steps':len(chunks),'sec':time.monotonic()-t_ep,'cum_update_norm':float(sum(e['update_norm'] for e in out['log']))};out['epochs'].append(es)
            log(f'  FT-ISO lr {lr:g} epoch {ep}: {len(chunks)} packed chunks, mean CE {es["mean_ce"]:.4f}, place-token CE {es["fact_token_ce"]}, {es["sec"]:.0f}s')
            if read is not None and ep in read_epochs:read(ep,m)
    out['model']=m;return out
def iso_epoch_rows(world,enc,seed,epochs,facts,enc_place):
    """rows of every epoch for FT-ISO: all 64 rounds' sentences ('\\n\\n' + text), reshuffled per epoch (seeded); -> list of (rows, place_rows) with place_rows[i] = (first, last+1) positions that PREDICT the place tokens (position i predicts token i+1) inside row i; None if the place tokens are not found"""
    A_=world['chunks'];sents=[(x['text'],x['fact']) for c in A_ for x in c['sents']];out=[]
    for e in range(epochs):
        order=list(range(len(sents)));random.Random(f'E55ftiso|{world["seed"]}|{seed}|{e}').shuffle(order);rows=[];pr=[]
        for i in order:
            t,f=sents[i];ids=enc('\n\n'+t);rows.append((ids,(0,len(ids)-1)));pl=enc_place(' '+facts[f]['P']);L=len(pl);hit=None
            for j in range(len(ids)-L+1):
                if ids[j:j+L]==pl:hit=j;break
            pr.append((max(hit-1,0),hit-1+L) if hit is not None else (0,0))
        out.append((rows,pr))
    return out
