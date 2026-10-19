"""2x2 power flag (PROTOCOL 2x2 amendment 03:20 KST, item 3): at W0, per seed and world, rho = ||sum_k c_k v_k|| / ||v_screen||.
v_k = gP(chunk) - gP(c2-swapped chunk) at W0 (the stream's per-chunk c2 term, a chunk-mean gradient), c_k = the clip coefficient
of the NTP gradient g_k at W0 for the native write (carried KV, W0 throughout) and for the masked write (chunk alone).
v_screen norm = onestep2_screen_checks_s{seed}.json signal_norms c2ch. No update is applied; no D is computed.
Also reported: the screen's c_w (clip coefficient of the isolated NTP gradient) for the c_w-adjusted variant.
Usage: python rho2x2.py 101 102 103   -> OUT/rho2x2.json"""
import os,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import stream2x2 as S2
from stream2x2 import *
def main():
    res={}
    rt=Runtime()
    for sd in map(int,sys.argv[1:]):
        L=S2.Ladder2(rt,sd,readout=False);L.build2();chk=json.load(open(S2.OUT/f'onestep2_screen_checks_s{sd}.json'));res[sd]={}
        W=int(rt.cfg.model.sliding_window_size);seg=np.zeros(CH,np.int32);loc=np.arange(CH,dtype=np.int32);pm=jnp.asarray(packing.prefix_mask(seg,W))
        samenext=np.concatenate([np.ones(CH-1,bool),[False]]);m=rt.base;st_all=clone_state(rt.state_all)
        for w in 'AB':
            sw=w=='B';ids=np.asarray(L.T.P[sw].ids)
            rX,_=SP.chunk_rows(L.T.w,sw,L.T.tok,rt.bos,rt.eos,'X');rY,_=SP.chunk_rows(L.T.w,sw,L.T.tok,rt.bos,rt.eos,'Y');rows={x[3]['k']:(x[0],x[1],y[0],y[1]) for x,y in zip(rX,rY)}
            zero=jax.tree.map(jnp.zeros_like,m.inner_parameters());acc_n=None;acc_m=None;cn=[];cm=[]
            with rt.mesh:
                mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);pref=rt.prefix(rt.base,seq);st_suf=clone_state(rt.state_suffix)
                def gp(ii,mm):
                    tg=np.concatenate([ii[1:],[rt.eos]]).astype(np.int32);pf=L.prefix_iso(rt.base,jnp.asarray(ii),jnp.asarray(loc),pm)
                    return L.gP(m,jnp.asarray(ii),jnp.asarray(tg),jnp.asarray(mm),pf,jnp.asarray(seg),jnp.asarray(loc))
                for k in range(NCH):
                    _,st_suf,gn_n,c_n,_=L.nat_step(m,st_all,st_suf,rt.chunk(seq,k),pref[k*CH:(k+1)*CH],zero,jnp.float32(0.0))   # W0 native read (update discarded)
                    c=ids[k*CH:(k+1)*CH];tg=np.concatenate([ids[k*CH+1:(k+1)*CH+1],[rt.eos]])[:CH].astype(np.int32);pf=L.prefix_iso(rt.base,jnp.asarray(c),jnp.asarray(loc),pm)
                    _,gn_m,c_m,_=L.msk_step(m,st_all,jnp.asarray(c),jnp.asarray(tg),jnp.asarray(samenext),pf,jnp.asarray(seg),jnp.asarray(loc),zero,jnp.float32(0.0),jnp.float32(1.0))
                    cn.append(float(c_n));cm.append(float(c_m))
                    if k in rows:
                        iX,mX,iY,mY=rows[k];v=jax.tree.map(lambda a,b:a-b,gp(iX,mX),gp(iY,mY))
                        acc_n=jax.tree.map(lambda x:x*c_n,v) if acc_n is None else jax.tree.map(lambda a,x:a+x*c_n,acc_n,v)
                        acc_m=jax.tree.map(lambda x:x*c_m,v) if acc_m is None else jax.tree.map(lambda a,x:a+x*c_m,acc_m,v);del v
            vs=float(chk['signal_norms']['c2ch'][w]);nn=float(optax.global_norm(acc_n));nm=float(optax.global_norm(acc_m))
            res[sd][w]={'rho_native':nn/vs,'rho_masked':nm/vs,'v_screen_norm':vs,'n_chunks_v':len(rows),'screen_c_w':float(chk['coef'][w]),
                        'clip_bind_frac_native_W0':sum(x<1 for x in cn)/len(cn),'clip_bind_frac_masked_W0':sum(x<1 for x in cm)/len(cm)}
            print(sd,w,json.dumps(res[sd][w]),flush=True);del acc_n,acc_m
        del L
    json.dump(res,open(S2.OUT/'rho2x2.json','w'),indent=1)
if __name__=='__main__':main()
