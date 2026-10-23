"""E55 32.10 label-free sentence selection variants V1-V5 on the sentence dump (CPU, numpy). INPUT: only the arrays of the dump (attention mass A[g,h,q,w], query features q_len / q_ce, window features w_len / w_chunk, token ids). This module must not import e55_sentdump_labels and must not open any label file (checked by e55_sentgrid.py's self test: no label name in this file's source).
Pipeline of one configuration cfg = dict(group, heads, gate, ngate, nvote, rule, rparam, ce, rare):
  S[q,w] = attention mass of query sentence q on window sentence w for the layer group ('g1' / 'g2' / 'g3' / 'all' = mean of the three), heads summed ('sum') or each head voting separately ('head');
  candidates = window sentences with >= MIN_TOK tokens;
  V1 gate (cfg gate='sharp', ngate = share of queries kept): queries ranked by the maximum of their row over the candidates (sharpness), the top share is kept;
  V4 (ce = share): queries ranked by their mean token CE, the highest share is kept (surprising sentences carry the information to be written);
  every kept query votes for its top-nvote candidates (V1 votes; 'head' = per head);
  V5 (rare=True): the votes of a window sentence are multiplied by R_j = (number of distinct tokens of j that occur in 2..RARE_MAX window sentences) / (distinct tokens of j) (rare but repeated tokens, from the token ids only);
  hide rule: 'rank' = the top rparam % of the candidates with a positive score (V2), 'z' = score > mean + rparam x std over the candidates (V3).
-> bool [nW] hidden window sentences."""
import numpy as np
MIN_TOK=4;RARE_MAX=20
def mass(A,group,heads):
    """A float16 [3,H,nQ,nW] -> float32 [H,nQ,nW] (heads='head') or [1,nQ,nW] (heads='sum' = sum over the heads)"""
    A=A.astype(np.float32);X=A.mean(0) if group=='all' else A[{'g1':0,'g2':1,'g3':2}[group]]
    return X if heads=='head' else X.sum(0,keepdims=True)
def rare_repeat(win_ids,w_start,w_len):
    """R_j per window sentence from the token ids only: share of the distinct tokens of the sentence that occur in 2..RARE_MAX different window sentences"""
    sets=[set(win_ids[s:s+n].tolist())-{-1} for s,n in zip(w_start,w_len)];df={}
    for st in sets:
        for t in st:df[t]=df.get(t,0)+1
    return np.asarray([sum(1 for t in st if 2<=df[t]<=RARE_MAX)/max(len(st),1) for st in sets],np.float32)
def select(d,cfg):
    """d: dict of the dump arrays of ONE (state, chunk); cfg as in the module doc -> hidden bool [nW]"""
    S=mass(d['A'],cfg.get('group','g3'),cfg.get('heads','sum'));w_len=np.asarray(d['w_len']);cand=w_len>=MIN_TOK;nW=len(w_len);nQ=S.shape[1]
    if not cand.any():return np.zeros(nW,bool)
    keepq=np.ones(nQ,bool);Sc=np.where(cand[None,None,:],S,-1.0)
    if cfg.get('gate')=='sharp':
        sh=Sc.sum(0).max(1) if S.shape[0]>1 else Sc[0].max(1);n=max(1,int(round(nQ*cfg.get('ngate',0.5))));k_=np.zeros(nQ,bool);k_[np.argsort(-sh,kind='stable')[:n]]=True;keepq&=k_
    if cfg.get('ce'):
        ce=np.nan_to_num(np.asarray(d['q_ce'],np.float64),nan=-1e9);n=max(1,int(round(nQ*cfg['ce'])));k_=np.zeros(nQ,bool);k_[np.argsort(-ce,kind='stable')[:n]]=True;keepq&=k_
    nv=cfg.get('nvote',1);votes=np.zeros(nW,np.float64)
    for hm in Sc:
        order=np.argsort(-hm,axis=1,kind='stable')[:,:nv]
        for q in np.nonzero(keepq)[0]:
            for j in order[q]:
                if cand[j] and hm[q,j]>0:votes[j]+=1
    if cfg.get('rare'):votes=votes*rare_repeat(np.asarray(d['win_ids']),np.asarray(d['w_start']),w_len)
    hid=np.zeros(nW,bool);pos=cand&(votes>0)
    if not pos.any():return hid
    if cfg.get('rule','rank')=='rank':
        nc=int(cand.sum());n=max(1,int(np.ceil(nc*cfg.get('rparam',30)/100.0)));idx=np.nonzero(pos)[0];top=idx[np.argsort(-votes[idx],kind='stable')[:n]];hid[top]=True
    else:
        v=votes[cand];hid=cand&(votes>v.mean()+cfg.get('rparam',1.5)*v.std())&pos
    return hid

def head_rank(d,group,pct):
    """label-free head selection by token sharpness: heads (layer, head) of the layer group ranked by the mean over the query tokens (inside sentences) of the maximum window mass T_max; returns bool [L,H] of the top pct %"""
    Tm=np.asarray(d['T_max'],np.float32);nL,H=Tm.shape[:2]
    sharp=Tm.mean(-1);lay=np.zeros(nL,bool)
    b={'g1':(0,nL//3),'g2':(nL//3,2*nL//3),'g3':(2*nL//3,nL),'all':(0,nL)}[group];lay[b[0]:b[1]]=True
    idx=np.argwhere(lay[:,None]&np.ones((1,H),bool));n=max(1,int(round(len(idx)*pct/100.0)));val=sharp[idx[:,0],idx[:,1]]
    top=idx[np.argsort(-val,kind='stable')[:n]];sel=np.zeros((nL,H),bool);sel[top[:,0],top[:,1]]=True;return sel
def select_heads(d,cfg):
    """V6: sharpest-head selection + vote trigger. cfg: sel 'tok' (a query token votes for the window sentence of its max key when T_max >= thr) or 'hs' (a query sentence votes per head for its top-nvote window sentences of the per-layer sentence mass AL); hpct, hgroup, thr, nvote; trig 'q' (distinct voting query sentences) or 'h' (distinct voting heads) with the count k; a window sentence with >= MIN_TOK tokens and votes >= k is hidden. No labels."""
    w_len=np.asarray(d['w_len']);nW=len(w_len);cand=w_len>=MIN_TOK;nQ=len(d['q_len'])
    sel=head_rank(d,cfg.get('hgroup','all'),cfg.get('hpct',25));li,hi=np.nonzero(sel);V=np.zeros((len(li),nQ,nW),bool)
    if cfg['sel']=='tok':
        qs=np.full(np.asarray(d['T_max']).shape[-1],-1,np.int64)
        for i,(a,n) in enumerate(zip(np.asarray(d['q_start']),np.asarray(d['q_len']))):qs[a:a+n]=i
        Tm=np.asarray(d['T_max'],np.float32)[li,hi];Tw=np.asarray(d['T_wsid'])[li,hi].astype(np.int64)
        ok=(Tm>=cfg.get('thr',0.2))&(qs[None,:]>=0)&(Tw>=0);r,t=np.nonzero(ok);V[r,qs[t],Tw[r,t]]=True
    else:
        AL=np.asarray(d['AL'])[li,hi].astype(np.float32);AL=np.where(cand[None,None,:],AL,-1.0)
        order=np.argsort(-AL,axis=2,kind='stable')[:,:,:cfg.get('nvote',1)]
        for j in range(order.shape[2]):
            r,q=np.nonzero(np.take_along_axis(AL,order[:,:,j:j+1],2)[:,:,0]>0);V[r,q,order[r,q,j]]=True
    cnt=V.any(0).sum(0) if cfg.get('trig','q')=='q' else V.any(1).sum(0)
    return cand&(cnt>=cfg.get('k',2))
def head_score(d,kind):
    """label-free head scores [L,H] from the token statistics: 'sharp' = mean over the query tokens (inside sentences) of T_max; 'copy' = mean of T_max x [token id of the max key == token id of the query token] (copy / same-token heads); 'induct' = mean of T_max x [token before the max key == token before the query token] (induction heads); the max key must be a real window key"""
    ck=(id(d),kind)
    if ck in _HS:return _HS[ck]
    r=_head_score(d,kind);_HS.clear() if len(_HS)>64 else None;_HS[ck]=r;return r
_HS={}
def _head_score(d,kind):
    Tm=np.asarray(d['T_max'],np.float32);qs=np.zeros(Tm.shape[-1],bool)
    for a,n in zip(np.asarray(d['q_start']),np.asarray(d['q_len'])):qs[a:a+n]=True
    if kind=='sharp':return Tm[:,:,qs].mean(-1)
    key=np.asarray(d['T_key']).astype(np.int64);wid=np.asarray(d['win_ids']);qid=np.asarray(d['q_ids']);t=np.nonzero(qs)[0]
    if kind=='copy':same=(wid[key[:,:,t]]==qid[t][None,None,:])&(wid[key[:,:,t]]>=0)
    else:
        t=t[t>0];kp=np.maximum(key[:,:,t]-1,0);same=(wid[kp]==qid[t-1][None,None,:])&(wid[kp]>=0)&(key[:,:,t]>0)
    return (Tm[:,:,t]*same).mean(-1)
def pick_heads(d,kind,group,pct):
    sc=head_score(d,kind);nL,H=sc.shape;b={'g1':(0,nL//3),'g2':(nL//3,2*nL//3),'g3':(2*nL//3,nL),'all':(0,nL)}[group]
    idx=np.argwhere((np.arange(nL)[:,None]>=b[0])&(np.arange(nL)[:,None]<b[1])&np.ones((1,H),bool));n=max(1,int(round(len(idx)*pct/100.0)))
    top=idx[np.argsort(-sc[idx[:,0],idx[:,1]],kind='stable')[:n]];return top[:,0],top[:,1]
def qtok_mask(d,qf):
    """label-free query-token filter [CH]: 'ce' = query tokens whose token CE (write forward) is in the top 30 % of the sentence tokens (surprising tokens carry the new fact); 'rare' = tokens whose id occurs in 2..RARE_MAX different window sentences (rare but repeated, from the token ids only)"""
    qid=np.asarray(d['q_ids']);qs=np.zeros(len(qid),bool)
    for a,n in zip(np.asarray(d['q_start']),np.asarray(d['q_len'])):qs[a:a+n]=True
    if qf=='ce':
        ce=np.nan_to_num(np.asarray(d['q_tce'],np.float32),nan=-1e9);thr=np.percentile(ce[qs],70) if qs.any() else 0;return qs&(ce>=thr)
    wid=np.asarray(d['win_ids']);df={}
    for s_,l_ in zip(np.asarray(d['w_start']),np.asarray(d['w_len'])):
        for t in set(wid[s_:s_+l_].tolist())-{-1}:df[t]=df.get(t,0)+1
    return qs&np.asarray([2<=df.get(int(t),0)<=RARE_MAX for t in qid])
def select_v7(d,cfg):
    """V7 (no labels): heads chosen by head_score kind (sharp / copy / induct) top hpct % of group hgroup; sel 'tok' = query tokens with T_max (x relative V-norm of the key when av) >= thr vote for the window sentence of their max key, keys in the first nsink window tokens and in the last norecent chunks excluded; sel 'hsagg' = per query sentence the per-head sentence mass (src AL or AV) is aggregated over the selected heads by agg (sum / max / top5 = mean of the 5 largest) and the query votes for its top sentence (window sentences of the first nsink tokens / last norecent chunks excluded); trig 'q' = distinct voting query sentences >= k, 'rank' = the top rparam % of the candidates with votes. -> hidden bool [nW]"""
    w_len=np.asarray(d['w_len']);w_st=np.asarray(d['w_start']);w_ch=np.asarray(d['w_chunk']);nW=len(w_len);nQ=len(d['q_len'])
    cand=w_len>=MIN_TOK
    if cfg.get('nsink',0):cand&=w_st>=cfg['nsink']
    if cfg.get('norecent',0):cand&=w_ch<w_ch.max()-cfg['norecent']+1
    li,hi=pick_heads(d,cfg.get('kind','sharp'),cfg.get('hgroup','all'),cfg.get('hpct',10));V=np.zeros((len(li),nQ,nW),bool)
    if cfg['sel']=='tok':
        Tm=np.asarray(d['T_max'],np.float32)[li,hi];key=np.asarray(d['T_key']).astype(np.int64)[li,hi];Tw=np.asarray(d['T_wsid'])[li,hi].astype(np.int64)
        if cfg.get('av'):
            vn=np.asarray(d['VN'],np.float32);vn=vn/np.maximum(vn.mean(-1,keepdims=True),1e-9);kvh=hi*vn.shape[1]//np.asarray(d['T_max']).shape[1];Tm=Tm*np.take_along_axis(vn[li,kvh],key,axis=1)
        qs=np.full(Tm.shape[-1],-1,np.int64)
        for i,(a,n) in enumerate(zip(np.asarray(d['q_start']),np.asarray(d['q_len']))):qs[a:a+n]=i
        ok=(Tm>=cfg.get('thr',0.2))&(qs[None,:]>=0)&(Tw>=0)&(key>=cfg.get('nsink',0))
        if cfg.get('qf'):ok&=qtok_mask(d,cfg['qf'])[None,:]
        if cfg.get('norecent',0):ok&=key<8192-cfg['norecent']*1024
        r,t=np.nonzero(ok);V[r,qs[t],Tw[r,t]]=True
    else:
        X=np.asarray(d[cfg.get('src','AL')])[li,hi].astype(np.float32);X=np.where(cand[None,None,:],X,-1.0);agg=cfg.get('agg','sum')
        S=X.sum(0) if agg=='sum' else X.max(0) if agg=='max' else np.sort(X,axis=0)[-5:].mean(0)
        top=S.argmax(1);q=np.nonzero(S[np.arange(nQ),top]>0)[0]
        for h in range(1):V[0,q,top[q]]=True
    cnt=V.any(0).sum(0)
    if cfg.get('trig','q')=='q':return cand&(cnt>=cfg.get('k',2))
    pos=cand&(cnt>0);hid=np.zeros(nW,bool)
    if pos.any():
        n=max(1,int(np.ceil(cand.sum()*cfg.get('rparam',30)/100.0)));idx=np.nonzero(pos)[0];hid[idx[np.argsort(-cnt[idx],kind='stable')[:n]]]=True
    return hid
def select_str(d,cfg):
    """string-match baseline (label-free, token ids only): a window sentence is hidden if it shares a run of >= n identical consecutive tokens with the write chunk (the E50 style exact string overlap), n = cfg['n']"""
    n=cfg.get('n',8);qid=np.asarray(d['q_ids']);wid=np.asarray(d['win_ids']);w_st=np.asarray(d['w_start']);w_len=np.asarray(d['w_len'])
    grams={tuple(qid[i:i+n].tolist()) for i in range(len(qid)-n+1)};hid=np.zeros(len(w_len),bool)
    for j,(s,l) in enumerate(zip(w_st,w_len)):
        if l>=MIN_TOK:
            seg=wid[s:s+l]
            hid[j]=any(tuple(seg[i:i+n].tolist()) in grams for i in range(l-n+1))
    return hid
