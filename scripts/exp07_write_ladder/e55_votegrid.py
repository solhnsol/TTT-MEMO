"""E55 31 voting-mask grid on the CPU from the attention dump (e55_run.py attdump -> e55_attdump_s<seed>.npz).
Definition (design 31.1): the per-query attention row of a layer group (head-averaged) gives the top-m window keys as triggers; every trigger j votes for the interval [j-n, j+n]; C[x] = number of votes of position x summed over all fact-sentence queries of the chunk; the key x is hidden when C[x] >= c x (the votes a uniformly random trigger placement would give per window key, V (2n+1) / n_valid with V = number of triggers); the mask is per key and common to all queries.
Sources: 'avg' = triggers from the layer-group head-averaged rows (the design definition), 'heads' = triggers from every (layer, head) top-m of the group (many more votes; same threshold rule), 'mass' = the CPU approximation of 31.1: the accumulated attention mass M[x] of all queries, box-filtered over 2n+1 keys, hidden when >= c x its mean over the valid window keys.
Metrics per chunk (mean over the dumped chunks): (a) hidden share of the valid window keys, (b) coverage of fact-sentence tokens, (c) coverage of the answer tokens (name / place / occupation / any), (d) precision = fact-sentence tokens among the hidden keys (random level = fact share of the window), (e) share of the fact sentences with >= 50% of their tokens hidden, (f) coverage of the earlier sentences of the SAME fact (mean over the query facts of the chunk). Pass rule (fixed before the results, design 31.3): some (n, c) with coverage (b) >= 0.6, share (a) <= 0.25, precision (d) >= 0.6 and (e) >= 0.6."""
import numpy as np,json,argparse,sys
def votes(idx,val,n,W):
    """idx, val: any shape [..., m]; entries with val <= 0 are padding. -> (C [W] int32, number of triggers)"""
    i=np.asarray(idx).astype(np.int64).ravel();v=np.asarray(val).astype(np.float32).ravel();i=i[v>0]
    lo=np.maximum(i-n,0);hi=np.minimum(i+n,W-1);d=np.zeros(W+1,np.int64);np.add.at(d,lo,1);np.add.at(d,hi+1,-1);return np.cumsum(d)[:W],int(len(i))
def hide_mask(src,z,c,k,g,n,cc):
    """z: dict of the arrays of chunk k; -> bool [W] hidden keys, or None when the group is missing"""
    valid=z[f'c{k}/key_valid'];W=len(valid)
    if src=='mass':
        if f'c{k}/{g}_mass_facts' not in z:return None
        M=z[f'c{k}/{g}_mass_facts'].sum(0).astype(np.float64);cs=np.concatenate([[0],np.cumsum(M)]);lo=np.maximum(np.arange(W)-n,0);hi=np.minimum(np.arange(W)+n,W-1)+1;S=(cs[hi]-cs[lo])/(2*n+1)
        lvl=M[valid].sum()/max(valid.sum(),1);return valid&(S>=cc*lvl)
    if src=='avg':
        if f'c{k}/{g}_top_idx' not in z:return None
        C,V=votes(z[f'c{k}/{g}_top_idx'],z[f'c{k}/{g}_top_val'],n,W)
    else:
        if f'c{k}/{g}_layers' not in z:return None
        a_,b_=[int(x) for x in z[f'c{k}/{g}_layers']];C,V=votes(z[f'c{k}/heads_idx'][a_:b_],z[f'c{k}/heads_val'][a_:b_],n,W)
    thr=max(1.0,cc*V*(2*n+1)/max(valid.sum(),1));return valid&(C>=thr)
def metrics(z,k,hid):
    valid=z[f'c{k}/key_valid'];fact=z[f'c{k}/key_fact'];sent=z[f'c{k}/key_sent'];ans=z[f'c{k}/key_ans'];ft=valid&(fact>=0);nv=max(int(valid.sum()),1);r={'share':hid.sum()/nv,'cov_fact':(hid&ft).sum()/max(ft.sum(),1),'precision':(hid&ft).sum()/max(hid.sum(),1) if hid.sum() else float('nan'),'random_precision':ft.sum()/nv}
    for nm,bit in (('name',1),('place',2),('occ',4)):
        s=valid&((ans&bit)>0);r['cov_'+nm]=(hid&s).sum()/max(s.sum(),1)
    s=valid&(ans>0);r['cov_ans']=(hid&s).sum()/max(s.sum(),1)
    ids=np.unique(sent[ft]);r['sent50']=float(np.mean([(hid&(sent==i)).sum()/max((sent==i).sum(),1)>=0.5 for i in ids])) if len(ids) else float('nan')
    qf=np.unique(z[f'c{k}/qfact']);r['same_fact']=float(np.mean([(hid&valid&(fact==f)).sum()/max((valid&(fact==f)).sum(),1) for f in qf])) if len(qf) else float('nan')
    return r
def grid(npz,groups=('g3',),sources=('avg','heads','mass'),ns=(8,16,32),cs=(2,4)):
    z=dict(np.load(npz))
    ks=sorted({int(x.split('/')[0][1:]) for x in z if x.endswith('/key_valid')});rows=[]
    for src in sources:
        for g in groups:
            for n in ns:
                for cc in cs:
                    ms=[]
                    for k in ks:
                        h=hide_mask(src,z,None,k,g,n,cc)
                        if h is None:break
                        ms.append(metrics(z,k,h))
                    if len(ms)!=len(ks):continue
                    m={q:float(np.nanmean([x[q] for x in ms])) for q in ms[0]};m.update(src=src,group=g,n=n,c=cc,chunks=ks,ok=bool(m['cov_fact']>=0.6 and m['share']<=0.25 and m['precision']>=0.6 and m['sent50']>=0.6));m['per_chunk_share']=[float(x['share']) for x in ms];rows.append(m)
    return rows
def report(rows,out=None):
    L=['# E55 31 voting-mask grid from the attention dump (CPU)','',__doc__.split('\n')[0],'','source = avg (design definition) / heads / mass (approximation); n = half-width of the voted interval in tokens; c = threshold as a multiple of the random vote level.','','| source | layers | n | c | hidden share | fact-token coverage | answer coverage (any / name / place / occ) | precision (random) | sentences >= 50% covered | same-fact earlier sentences covered | pass |','|---|---|---|---|---|---|---|---|---|---|---|']
    for m in rows:L.append(f"| {m['src']} | {m['group']} | {m['n']} | {m['c']} | {m['share']:.3f} | {m['cov_fact']:.3f} | {m['cov_ans']:.3f} / {m['cov_name']:.3f} / {m['cov_place']:.3f} / {m['cov_occ']:.3f} | {m['precision']:.3f} ({m['random_precision']:.3f}) | {m['sent50']:.3f} | {m['same_fact']:.3f} | {'PASS' if m['ok'] else 'no'} |")
    ok=[m for m in rows if m['ok']];L+=['',f"pass rule (all four): fact-token coverage >= 0.6, hidden share <= 0.25, precision >= 0.6, sentences >= 50% covered >= 0.6: {len(ok)} of {len(rows)} grid cells pass"+('' if not ok else ': '+', '.join(f"{m['src']}/{m['group']} n={m['n']} c={m['c']}" for m in ok))]
    L+=['','chunks averaged: '+(str(rows[0]['chunks']) if rows else 'n/a')]
    t='\n'.join(L)+'\n'
    if out:open(out,'w').write(t)
    return t
def selftest():
    import tempfile,os
    rng=np.random.default_rng(0);W=9216;d={};k=16;facts=list(range(16))
    valid=np.zeros(W,bool);valid[:8192]=True;fact=np.full(W,-1,np.int16);sent=np.full(W,-1,np.int32);ans=np.zeros(W,np.uint8)
    starts=np.arange(16)*500+100
    for f,s in enumerate(starts):fact[s:s+24]=f;sent[s:s+24]=f;ans[s+3:s+5]|=1;ans[s+8:s+9]|=2;ans[s+12:s+13]|=4
    rows=np.arange(16*24);qf=np.repeat(np.arange(16),24);idx=np.zeros((len(rows),8),np.int16);val=np.zeros((len(rows),8),np.float16);M=np.zeros((16,W),np.float32)
    for r,f in enumerate(qf):
        idx[r]=rng.integers(starts[f],starts[f]+24,8);val[r]=0.02;M[f,idx[r]]+=0.02          # attention concentrates inside the sentence of the query fact (a planted signal)
    d.update({f'c{k}/key_valid':valid,f'c{k}/key_fact':fact,f'c{k}/key_sent':sent,f'c{k}/key_ans':ans,f'c{k}/qfact':qf.astype(np.int16),f'c{k}/g3_top_idx':idx,f'c{k}/g3_top_val':val,f'c{k}/g3_mass_facts':M,f'c{k}/g3_layers':np.asarray([16,24],np.int16)})
    d[f'c{k}/heads_idx']=np.tile(idx[None,None],(24,4,1,1));d[f'c{k}/heads_val']=np.tile(val[None,None],(24,4,1,1))
    f_=tempfile.mktemp(suffix='.npz');np.savez(f_,**d);rs=grid(f_,('g3',),('avg','heads','mass'),(8,16),(2,4));os.remove(f_)
    assert len(rs)==3*2*2 and all(0<=m['share']<=1 for m in rs)
    a=[m for m in rs if m['src']=='avg' and m['n']==8 and m['c']==2][0]
    assert a['precision']>0.5 and a['cov_fact']>0.5 and a['share']<0.25 and a['sent50']>0.5,a      # planted signal -> a clear mask exists
    t=report(rs);assert 'PASS' in t and 'grid cells pass' in t
    print('e55_votegrid SELFTEST OK')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--npz');ap.add_argument('--out');ap.add_argument('--groups',default='g3,all,g1,g2');ap.add_argument('--sources',default='avg,heads,mass');ap.add_argument('--ns',default='8,16,32');ap.add_argument('--cs',default='2,4');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    rows=grid(a.npz,tuple(a.groups.split(',')),tuple(a.sources.split(',')),tuple(int(x) for x in a.ns.split(',')),tuple(int(x) for x in a.cs.split(',')));t=report(rows,a.out);print(t)
