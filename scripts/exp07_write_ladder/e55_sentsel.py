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
