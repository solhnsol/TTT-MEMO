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
