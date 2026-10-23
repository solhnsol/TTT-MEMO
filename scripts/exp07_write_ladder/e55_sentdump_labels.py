"""E55 32.10 evaluation labels of the sentence dump (separate file on purpose: the selection code e55_sentsel.py must never import this module or open the label file). Per sentence of the write chunk and of the window (the same punctuation sentences as e55_sentdump): fact id (majority fact of its tokens, -1 = background), fact-token share, and for window sentences the chunk number (= round of the fact in p = 1 worlds)."""
import numpy as np
import e55_sentdump as SD
def token_fact(chunks_meta,c,CH=1024):
    f=np.full(CH,-1,np.int16)
    for x in chunks_meta[c]['sents']:f[x['start']:x['end']]=x['fact']
    return f
def sent_label(tf,a,b):
    t=tf[a:b];fact=t[t>=0]
    if len(fact)==0:return -1,0.0
    vals,cnt=np.unique(fact,return_counts=True);share=len(fact)/max(b-a,1)
    return (int(vals[np.argmax(cnt)]) if share>=0.5 else -1),float(share)
def labels(chunks_meta,ends,k,CH=1024,nw=8):
    q,wsid,wst,wlen=SD.layout(ends,k,CH,nw);out={}
    tq=token_fact(chunks_meta,k,CH);ql=[sent_label(tq,a,b) for a,b in q];out['q_fact']=np.asarray([x[0] for x in ql],np.int16);out['q_fact_share']=np.asarray([x[1] for x in ql],np.float32)
    wl=[]
    for s,n in zip(wst,wlen):
        c=k-nw+int(s)//CH;tf=token_fact(chunks_meta,c,CH);o=int(s)-(c-k+nw)*CH;wl.append(sent_label(tf,o,o+int(n)))
    out['w_fact']=np.asarray([x[0] for x in wl],np.int16);out['w_fact_share']=np.asarray([x[1] for x in wl],np.float32);out['w_chunk']=(k-nw+np.asarray(wst)//CH).astype(np.int32);return out
