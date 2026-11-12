"""W0 key second moment C for the KEYSTAT screen's c_hi / c_lo cells (PROTOCOL 2026-10-17 KEYSTAT/PoE v1). Generic text: ~1M tokens
from Gutenberg books that no other text of the study uses (asserted: not the canary pg11, the long-NLL pg2160 or its candidates
pg1399/pg1023, the R1 tail pg1342, the 4 background books pg98/pg1260/pg768/pg766). Each 1024-token chunk is read alone
(positions from 0, the screen's chunk condition) by the fp32 W0 model; per suffix layer, C_in = mean x_in x_in^T (w1/w3 input,
2560^2) and C_mid = mean x_mid x_mid^T (w2 input, 5632^2). Output: DUMP/keystat_C.npz + keystat_C.json (source, sha, tokens).
Usage: python keystat_C.py"""
import os,sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ladder as LD
from ladder import *
import taps as TP
BOOKS=['pg1184.txt','pg2701.txt','pg345.txt'];NTOK=1_000_000
EXCL={'pg11.txt','pg2160.txt','pg1399.txt','pg1023.txt','pg1342.txt','pg98.txt','pg1260.txt','pg768.txt','pg766.txt'}
DUMP=Path(os.environ.get('DUMP','/home/solhnsol/workspace/TTT-MEMO-dumps'))
def main():
    assert not set(BOOKS)&EXCL
    rt=Runtime();L=LD.Ladder(rt,101,readout=True);rt32=L.rt32;m0=rt32.base;Batch=L.Batch;W=int(rt.cfg.model.sliding_window_size)
    ids=[];src={}
    for b in BOOKS:
        f=LD.ROOT/'data/narrative_gutenberg'/b;raw=f.read_bytes();src[b]=hashlib.sha256(raw).hexdigest()
        t=raw.decode(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t;t=' '.join(t.replace('\r','').split())
        ids+=[rt.bos]+stage2.enc(L.T.tok,t).ids
        if len(ids)>=NTOK:break
    nch=NTOK//CH+1;assert len(ids)>=nch*CH,len(ids);ids=ids[:nch*CH]
    deltas,acts,NL,dm,dff=TP.make_taps(m0,rt32,Batch)
    seg=jnp.asarray(np.zeros(CH,np.int32));loc=jnp.asarray(np.arange(CH,dtype=np.int32));pm=jnp.asarray(packing.prefix_mask(np.zeros(CH,np.int32),W))
    @eqx.filter_jit
    def pref(m,ids_):
        packing.PACK_PRE['mask']=pm
        try:
            sq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=loc)
            return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),clone_state(rt32.state_prefix),sq).last_hidden_state
        finally:packing.PACK_PRE.clear()
    @jax.jit
    def acc(Ci,Cm,xi,xm):return Ci+jnp.einsum('ltd,lte->lde',xi,xi),Cm+jnp.einsum('ltd,lte->lde',xm,xm)
    Ci=jnp.zeros((NL,dm,dm),jnp.float32);Cm=jnp.zeros((NL,dff,dff),jnp.float32);t0=time.monotonic()
    with rt32.mesh,jax.default_matmul_precision('highest'):
        for k in range(nch):
            c=np.asarray(ids[k*CH:(k+1)*CH],np.int32);tg=np.concatenate([c[1:],[rt32.eos]]).astype(np.int32)
            xi,xm,_,_=acts(jnp.asarray(c),jnp.asarray(tg),jnp.ones(CH,bool),pref(m0,jnp.asarray(c)),seg,loc);Ci,Cm=acc(Ci,Cm,xi,xm)
            if k%100==0:print('chunk',k,'of',nch,round(time.monotonic()-t0),'s',flush=True)
    n=nch*CH;DUMP.mkdir(parents=True,exist_ok=True)
    np.savez(DUMP/'keystat_C.npz',C_in=np.asarray(Ci/n),C_mid=np.asarray(Cm/n))
    meta={'books':BOOKS,'sha256':src,'tokens':n,'chunks':nch,'excluded':sorted(EXCL),'condition':'chunk alone, positions from 0, fp32 W0, highest precision',
          'mean_eig_in':[float(jnp.trace(Ci[l])/dm/n) for l in range(NL)],'mean_eig_mid':[float(jnp.trace(Cm[l])/dff/n) for l in range(NL)],'sec':time.monotonic()-t0}
    json.dump(meta,open(DUMP/'keystat_C.json','w'),indent=1);json.dump(meta,open(LD.OUT/'keystat_C.json','w'),indent=1);print(json.dumps(meta)[:400],flush=True)
if __name__=='__main__':main()
