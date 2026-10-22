"""E55 design 27.8 attention diagnostic (forward only, no backward, no weight change): at chosen chunks of the NAT stream, how well does the attention that the FACT-SENTENCE query tokens of the current chunk put on the 8-chunk window keys single out the earlier same-fact sentence tokens?
For the queries (every token inside a fact sentence of chunk k) the head- and layer-averaged attention probability over the in-window keys (earlier 8 chunks, current chunk excluded) is ranked; for the top p % of the in-window keys (p = 1, 2, 5, 10, 20) the share of keys that are (a) tokens of earlier sentences of the SAME fact, (b) tokens of any earlier fact sentence, (c) background, and the recall of the answer-span tokens (name and place) of the same fact's earlier sentences, are reported against the random baseline (the share of that category among the valid window keys; recall p %). Prefix layers only (the layers that use the window K,V through s3 mode 'att'); layer groups: all / first third / middle third / last third.
CPU tested with the tiny model (e55_units.py U33); the 3B GPU path is NOT run."""
import numpy as np,json
import e55_swap as E
PS=(1,2,5,10,20)
def key_categories(chunks,k,qfact,CH=1024,nw=8):
    """-> (valid, same, anyf, ans_same) bool arrays over the WIN+CH key axis of chunk k for a query of fact `qfact` (index j <-> position k*CH-nw*CH+j; keys of the current chunk are never valid here)"""
    W=nw*CH;valid=np.zeros(W+CH,bool);same=np.zeros(W+CH,bool);anyf=np.zeros(W+CH,bool);ans=np.zeros(W+CH,bool)
    for c in range(max(0,k-nw),k):
        off=(c-k+nw)*CH;valid[off:off+CH]=True
        for x in chunks[c]['sents']:
            anyf[off+x['start']:off+x['end']]=True
            if x['fact']==qfact:
                same[off+x['start']:off+x['end']]=True
                for a_,b_ in x.get('ans',[]):ans[off+a_:off+b_]=True
    return valid,same,anyf,ans
def attn_stats(A,rows,facts,chunks,k,CH=1024,nw=8,ps=PS):
    """A [n_rows, WIN+CH] attention probabilities of the query rows (head/layer averaged), rows[i] = token index of the query in chunk k, facts[i] = its fact id. -> dict: per p: mean share same / any / background among the top-p % valid keys, baselines (category share of the valid keys), answer recall (same-fact answer tokens in the top set / all of them), attention mass on the categories, number of rows"""
    out={'n_rows':int(len(rows)),'p':{}};mass=np.zeros(3);base=np.zeros(3);rec_rows={p:[] for p in ps};sh={p:np.zeros(3) for p in ps};cnt=0
    cache={}
    for i,f in enumerate(facts):
        if f not in cache:cache[f]=key_categories(chunks,k,f,CH,nw)
        valid,same,anyf,ans=cache[f];nv=int(valid.sum())
        if nv==0:continue
        a=np.where(valid,A[i],-1.0);order=np.argsort(-a,kind='stable')[:nv];cnt+=1
        bgm=valid&~anyf
        mass+=[A[i][valid&same].sum(),A[i][valid&anyf].sum(),A[i][bgm].sum()];base+=[same.sum()/nv,anyf.sum()/nv,bgm.sum()/nv]
        for p in ps:
            n=max(1,int(np.ceil(nv*p/100)));top=order[:n];m_=np.zeros(len(a),bool);m_[top]=True
            sh[p]+=[(m_&same).sum()/n,(m_&anyf).sum()/n,(m_&bgm).sum()/n]
            if ans.sum()>0:rec_rows[p].append((m_&ans).sum()/ans.sum())
    cnt=max(cnt,1);out['attention_mass']={'same':float(mass[0]/cnt),'any':float(mass[1]/cnt),'background':float(mass[2]/cnt)};out['baseline_share']={'same':float(base[0]/cnt),'any':float(base[1]/cnt),'background':float(base[2]/cnt)}
    for p in ps:out['p'][str(p)]={'same':float(sh[p][0]/cnt),'any':float(sh[p][1]/cnt),'background':float(sh[p][2]/cnt),'answer_recall':float(np.mean(rec_rows[p])) if rec_rows[p] else None,'recall_baseline':p/100}
    return out
def make_diag_fn(L,rt,s3,chunks,groups=3,log=print):
    """-> fn(k, m, Kc, Vc, keep, ids_k): eager forward of the prefix layers of chunk k through s3 mode 'att' (weights m = the weights BEFORE the write of chunk k, window K,V Kc/Vc = [layers, WIN, ...] as given to prefix_masked), attention of the fact-sentence query rows collected per layer; returns the stats dict per layer group ('all', 'g1', 'g2', 'g3') -- nothing is written back"""
    import jax,jax.numpy as jnp
    CH=1024
    def fn(k,m,Kc,Vc,keep,ids_k):
        sents=chunks[k]['sents'];rows=np.asarray([t for x in sents for t in range(x['start'],x['end'])],np.int32);facts=[x['fact'] for x in sents for _ in range(x['start'],x['end'])]
        if len(rows)==0 or k==0:return {'n_rows':0}
        s3.S3.clear();s3.S3['mode']='att';s3.S3['q0']=jnp.int32(k*CH);s3.S3['keep']=jnp.asarray(keep);s3.S3['att_rows']=jnp.asarray(rows);s3.S3['att_out']=[]
        try:
            ids=jnp.asarray(ids_k,dtype=jnp.int32)
            with rt.mesh:
                h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=L.Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
                for l in range(Kc.shape[0]):
                    blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
            per=s3.S3['att_out'];fn.last_h=h
        finally:s3.S3.clear()
        nL=len(per);assert nL==Kc.shape[0],(nL,Kc.shape[0]);res={'n_layers':nL}
        bounds=[0,nL//3,2*nL//3,nL]
        for name,(a_,b_) in (('all',(0,nL)),('g1',(bounds[0],bounds[1])),('g2',(bounds[1],bounds[2])),('g3',(bounds[2],bounds[3]))):
            if b_<=a_:res[name]=None;continue         # fewer than 3 prefix layers (tiny test model)
            A=np.mean(per[a_:b_],axis=0);res[name]=attn_stats(A,rows,facts,chunks,k,CH)
        log(f'  attention diagnostic at chunk {k}: {len(rows)} fact query rows, {nL} prefix layers');return res
    fn.last_h=None
    return fn
def diag_report(path_or_dict,out=None):
    d=json.load(open(path_or_dict)) if not isinstance(path_or_dict,dict) else path_or_dict
    L=[f"# E55 attention diagnostic (design 27.8), seed {d.get('seed')}, NAT stream, prefix layers only",'','Queries: every token inside a fact sentence of the chunk (chunk index from 0). Keys: the 8-chunk window before the chunk (current chunk excluded). For each query the head/layer-averaged attention probability ranks the window keys; the top p % are compared with a random choice of p % of the valid window keys: share of same-fact earlier sentence tokens, share of any earlier fact sentence tokens, share of background, and recall of the name and place tokens of the same fact (random baseline = p %). Layer groups: all prefix layers / first / middle / last third. Exploratory (one seed, 16 facts).','']
    for ck,v in sorted(d['chunks'].items(),key=lambda x:int(x[0])):
        if not v.get('all'):L.append(f'## chunk {ck}: no window or no fact sentences');continue
        for g in ('all','g1','g2','g3'):
            x=v.get(g)
            if not x:continue
            b=x['baseline_share'];L+=[f"## chunk {ck}, layers {g} ({x['n_rows']} query rows)",'',f"attention mass on same-fact earlier tokens {x['attention_mass']['same']:.3f} (random {b['same']:.4f}), any fact {x['attention_mass']['any']:.3f} (random {b['any']:.3f}), background {x['attention_mass']['background']:.3f} (random {b['background']:.3f})",'','| top p % of window keys | same-fact share (random) | any-fact share (random) | background share (random) | answer-token recall (random) |','|---|---|---|---|---|']
            for p,y in x['p'].items():L.append(f"| {p} | {y['same']:.3f} ({b['same']:.4f}) | {y['any']:.3f} ({b['any']:.3f}) | {y['background']:.3f} ({b['background']:.3f}) | "+(f"{y['answer_recall']:.3f}" if y['answer_recall'] is not None else 'n/a')+f" ({y['recall_baseline']:.2f}) |")
            L.append('')
    t='\n'.join(L)
    if out:open(out,'w').write(t)
    return t
def key_labels(chunks,k,CH=1024,nw=8):
    """E55 31 per-key labels over the WIN+CH key axis of chunk k (index j <-> position k*CH-nw*CH+j): valid (window key of an existing earlier chunk), fact (fact id of the fact sentence the key lies in, -1 = background), sent (global sentence id c*64 + index of the sentence in its chunk, -1 = none), ans (bit 1 = name token, 2 = place token, 4 = occupation token; needs annotate_answers with the spans ans_X / ans_P / ans_O)"""
    W=nw*CH;valid=np.zeros(W+CH,bool);fact=np.full(W+CH,-1,np.int16);sent=np.full(W+CH,-1,np.int32);ans=np.zeros(W+CH,np.uint8)
    for c in range(max(0,k-nw),k):
        off=(c-k+nw)*CH;valid[off:off+CH]=True
        for si,x in enumerate(chunks[c]['sents']):
            sl=slice(off+x['start'],off+x['end']);fact[sl]=x['fact'];sent[sl]=c*64+si
            for key,bit in (('ans_X',1),('ans_P',2),('ans_O',4)):
                for a_,b_ in x.get(key,[]):ans[off+a_:off+b_]|=bit
    return valid,fact,sent,ans
def make_dump_fn(L,rt,s3,chunks,topm=8,log=print):
    """E55 31 attention dump (forward only, same eager 'att' path as make_diag_fn): at a chunk k of the NAT stream, for the fact-sentence query rows: (1) per layer group (all / g1 / g2 / g3) the head- and layer-averaged attention row -> per query the top-m window keys (index int16, value float16) and the accumulated attention mass per fact [n_facts_in_chunk, WIN+CH] (float32; summing over the facts gives the mass vector M of all queries); (2) per layer and head the top-m window keys per query ([layers, heads, rows, m], int16 / float16; entries with value 0 are padding when fewer than m window keys exist); (3) the per-key labels (key_labels) and the query rows / facts. Returns a dict of numpy arrays (flatten with flat_dump). Nothing is written back."""
    import jax,jax.numpy as jnp
    CH=1024
    def fn(k,m,Kc,Vc,keep,ids_k):
        sents=chunks[k]['sents'];rows=np.asarray([t for x in sents for t in range(x['start'],x['end'])],np.int32);facts=np.asarray([x['fact'] for x in sents for _ in range(x['start'],x['end'])],np.int16)
        if len(rows)==0 or k==0:return {'n_rows':0}
        lo=(8-min(k,8))*CH
        s3.S3.clear();s3.S3['mode']='att';s3.S3['q0']=jnp.int32(k*CH);s3.S3['keep']=jnp.asarray(keep);s3.S3['att_rows']=jnp.asarray(rows);s3.S3['att_out']=[];s3.S3['att_topk']=topm;s3.S3['att_lo']=lo;s3.S3['att_heads']=[]
        try:
            ids=jnp.asarray(ids_k,dtype=jnp.int32)
            with rt.mesh:
                h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=L.Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
                for l in range(Kc.shape[0]):
                    blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
            per=s3.S3['att_out'];heads=s3.S3['att_heads'];fn.last_h=h
        finally:s3.S3.clear()
        nL=len(per);assert nL==Kc.shape[0]==len(heads),(nL,Kc.shape[0],len(heads))
        valid,kfact,ksent,kans=key_labels(chunks,k,CH);out={'n_rows':int(len(rows)),'n_layers':nL,'rows':rows,'qfact':facts,'key_valid':valid,'key_fact':kfact,'key_sent':ksent,'key_ans':kans,'heads_idx':np.stack([x[0] for x in heads]),'heads_val':np.stack([x[1] for x in heads])}
        ufacts=np.unique(facts);out['mass_facts']=ufacts.astype(np.int16)
        bounds=[0,nL//3,2*nL//3,nL]
        for name,(a_,b_) in (('all',(0,nL)),('g1',(bounds[0],bounds[1])),('g2',(bounds[1],bounds[2])),('g3',(bounds[2],bounds[3]))):
            if b_<=a_:continue
            A=np.mean(per[a_:b_],axis=0);Av=np.where(valid[None,:],A,-1.0);idx=np.argsort(-Av,axis=1,kind='stable')[:,:topm]
            out[f'{name}_top_idx']=idx.astype(np.int16);out[f'{name}_top_val']=np.take_along_axis(A,idx,1).astype(np.float16);out[f'{name}_mass_facts']=np.stack([A[facts==f].sum(0) for f in ufacts]).astype(np.float32)
            out[f'{name}_layers']=np.asarray([a_,b_],np.int16)
        log(f'  attention dump at chunk {k}: {len(rows)} fact query rows, {nL} prefix layers, top-{topm}');return out
    fn.last_h=None
    return fn
def flat_dump(d):
    """{chunk: dict of arrays} -> flat dict for np.savez_compressed: 'c<k>/<name>'"""
    out={}
    for k,v in d.items():
        for n,a in v.items():out[f'c{k}/{n}']=np.asarray(a)
    return out
PROBE_CONDS=('NAT','KMASK','SENTO','SENT','SWD','OALL','PALL','NALL')
PROBE_ARM={'NAT':'NAT','KMASK':'KMASK','SENTO':'SENTO','SENT':'SENT','SWD':'SWD','OALL':'OSPAN','PALL':'PSPAN','NALL':'NSPAN'}
def make_probe_fn(L,rt,s3,reader,meta,conds=PROBE_CONDS,nsum=8,log=print):
    """E55 28.7 one-step binding probe (forward-looking, nothing is committed): at the chunks of diag_multi, starting from the NAT weights m BEFORE the write of the chunk, for every window condition the write gradient step of the chunk is computed (L.write_step with the condition's window inputs; direction d = new - m), rescaled to ONE fixed norm = the sum of the last `nsum` NAT update norms of the run (the 8-chunk cumulative NAT norm), applied to a copy m' = m + s d/|d| and the out-of-window read of the 16 candidates on the unexposed question frames is scored; per fact PM3 = score(gold) - mean of the other 15; reported: PM3(m') - PM3(m) per fact (the weights m itself are the baseline read once per chunk), the direction cosine to the NAT direction, the chunk CE under the condition (free), the unrescaled step norm. reader(m) -> (n_items, 16) scores (Reads.at(.)['out']); meta = item meta (kind, gold_idx). The weights of the stream are never touched."""
    import jax,jax.numpy as jnp,e55_metrics as MT
    kind=np.asarray(meta['kind']);gold=np.asarray(meta['gold_idx']);isf=kind=='fact'
    def pm3(S):return MT.item_stats(np.asarray(S,float)[isf],gold[isf])['pm3']
    st_all=s3.clone_state(rt.state_all)
    def tdot(x,y):return float(sum(jnp.sum(a_.astype(jnp.float32)*b_.astype(jnp.float32)) for a_,b_ in zip(jax.tree.leaves(x),jax.tree.leaves(y))))
    def fn(k,m,cond_in,ids_k,c,nat_log):
        CH=1024;ids_j=jnp.asarray(ids_k,dtype=jnp.int32);tgt=float(sum(e['update_norm'] for e in nat_log[-nsum:])) if nat_log else None
        assert tgt and tgt>0,'the probe needs the update norms of earlier NAT chunks'
        base=pm3(reader(m));res={'target_norm':tgt,'pm3_base':[float(x) for x in base]};dirs={}
        for cn in conds:
            Kx,Vx,kx,sx=cond_in[PROBE_ARM[cn]]
            with rt.mesh:
                pref=L.prefix_masked(m,ids_j,Kx,Vx,jnp.int32(k*CH),jnp.asarray(kx));newp,gn,coef,ce=L.write_step(m,st_all,sx,c,pref,jnp.asarray(kx))
            d=jax.tree.map(lambda a_,b_:a_.astype(jnp.float32)-b_.astype(jnp.float32),newp,m.inner_parameters());dn=tdot(d,d)**0.5;dirs[cn]=d
            sc=tgt/dn if dn>0 else 0.0;mp=L.set_inner(m,jax.tree.map(lambda b_,x:(b_.astype(jnp.float32)+sc*x).astype(b_.dtype),m.inner_parameters(),d))
            S=reader(mp);p=pm3(S);res[cn]={'dpm3':[float(x) for x in (p-base)],'dpm3_mean':float(np.mean(p-base)),'raw_step_norm':dn,'chunk_ce':float(ce),'grad_norm':float(gn),'cos_to_NAT':None,'mask_keys':int((~kx).sum()) if np.ndim(kx)==1 else int((~kx.all(0)).sum())}
            log(f'  probe chunk {k} {cn}: dPM3 {res[cn]["dpm3_mean"]:+.4f}, raw step norm {dn:.4g}, chunk CE {float(ce):.3f}')
        if 'NAT' in dirs:
            nn=tdot(dirs['NAT'],dirs['NAT'])**0.5
            for cn in conds:res[cn]['cos_to_NAT']=tdot(dirs[cn],dirs['NAT'])/(tdot(dirs[cn],dirs[cn])**0.5*nn+1e-30)
        return res
    fn.conds=tuple(PROBE_ARM[c_] for c_ in conds);fn.cond_names=tuple(conds);return fn
def probe_report(d,out=None):
    d=json.load(open(d)) if not isinstance(d,dict) else d;conds=list(PROBE_CONDS)
    def ci(x):
        x=np.asarray(x,float);r=np.random.default_rng(5);b=x[r.integers(0,len(x),(2000,len(x)))].mean(1);return float(x.mean()),*np.percentile(b,[2.5,97.5]).tolist()
    L=[f"# E55 one-step binding probe (design 28.7), seed {d.get('seed')}, NAT weights before the write of each chunk",'','For each window condition the write gradient step of the chunk is computed from the NAT weights, rescaled to ONE fixed norm (the sum of the last 8 NAT update norms, shown) and applied to a COPY (nothing is committed); dPM3 = change of PM3 (gold score minus the mean of the other 15 candidates, nats) on the 16 facts, out-of-window read, unexposed question frames, relative to the unmodified weights; mean [95% fact-bootstrap CI]. cos = cosine of the step direction to the NAT step; chunk CE = CE of the chunk under that window condition (free by-product). O-ALL / P-ALL / N-ALL = name+place / place / name tokens of ALL earlier fact sentences hidden from fact-sentence queries; SWD = window K,V of the fake stream (fake places outside the candidates). One seed, 16 facts; exploratory.','']
    gains={}
    for ck,v in sorted(d['chunks'].items(),key=lambda x:int(x[0])):
        L+=[f"## chunk {ck} (target step norm {v['target_norm']:.4g})",'','| condition | dPM3 | dPM3 - NAT | cos to NAT | raw step norm | chunk CE | hidden keys |','|---|---|---|---|---|---|---|']
        nat=np.asarray(v['NAT']['dpm3'])
        for cn in conds:
            x=v[cn];dp=np.asarray(x['dpm3']);gains.setdefault(cn,{})[ck]=float(np.mean(dp-nat));f=lambda t:f'{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]'
            L.append(f"| {cn} | {f(ci(dp))} | {f(ci(dp-nat))} | {x['cos_to_NAT']:.3f} | {x['raw_step_norm']:.4g} | {x['chunk_ce']:.3f} | {x['mask_keys']} |")
        L.append('')
    L+=['## ratios to the full mask (SENT) of the gain over NAT (dPM3 - NAT), per chunk and mean over chunks','','| condition | '+' | '.join(f'chunk {c}' for c in sorted(gains['NAT'],key=int))+' | mean of ratios |','|---|'+'---|'*(len(gains['NAT'])+1)]
    for cn in ('KMASK','SENTO','OALL','PALL','NALL','SWD'):
        rs=[gains[cn][c]/gains['SENT'][c] if abs(gains['SENT'][c])>1e-9 else float('nan') for c in sorted(gains['NAT'],key=int)];L.append(f'| {cn} | '+' | '.join(f'{r:.2f}' for r in rs)+f' | {np.nanmean(rs):.2f} |')
    rk=[gains['KMASK'][c]/gains['SENT'][c] for c in gains['NAT'] if abs(gains['SENT'][c])>1e-9];m_=float(np.mean(rk)) if rk else float('nan')
    L+=['',f"analogue of the pre-set rule of the place-CE version (design 28.6: ratio >= 0.3 'P-2R', <= 0.15 'P-LC', otherwise undetermined) applied to KMASK / SENT of this probe: mean ratio {m_:.2f} -> "+('undetermined (no data)' if m_!=m_ else 'P-2R' if m_>=0.3 else 'P-LC' if m_<=0.15 else 'undetermined')+' (the rule for THIS probe is the design thread\'s to fix)']
    t='\n'.join(L)
    if out:open(out,'w').write(t)
    return t
if __name__=='__main__':
    import sys
    if '--selftest' in sys.argv:
        rng=np.random.default_rng(0);CH=1024;ch=[{'sents':[{'fact':f,'start':100+f*60,'end':140+f*60,'ans':[(105+f*60,108+f*60)]} for f in range(4)]} for _ in range(10)];k=9
        A=rng.random((8,CH*9));A/=A.sum(1,keepdims=True);rows=list(range(100,104))+list(range(160,164));facts=[0]*4+[1]*4
        r=attn_stats(A,rows,facts,ch,k);assert abs(r['p']['20']['same']-r['baseline_share']['same'])<0.05
        # planted attention: boost the same-fact keys -> top shares rise far above baseline and recall -> high
        A2=A.copy()
        for i,f in enumerate(facts):
            v,s,a_,an=key_categories(ch,k,f);A2[i][s]*=500
        A2/=A2.sum(1,keepdims=True);r2=attn_stats(A2,rows,facts,ch,k);assert r2['p']['1']['same']>0.9 and r2['p']['20']['answer_recall']>0.9 and r2['attention_mass']['same']>0.9
        print('e55_diag SELFTEST OK')
