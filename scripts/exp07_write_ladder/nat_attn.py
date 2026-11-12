"""E33 attention-mask cells on a natural-text stream (9105 spread): masked write (S3 write forward) where the masked set S_k = the per-target top-k head-max attention union read at FIXED W0.
Cells (attn8, attn16, attn32, attn64) are cut from ONE score extraction per world: cell 'sel' reads the pass-1 stream of world A / B at W0 (natural chain, no update; same capture / candidates / ties as nat_signal.py) and saves
the per-chunk, per-target top-256 key order (uint16 [CH,256] per chunk, ORDER ONLY, no score values) to OUT/nat_attn_sel_s{stream}_{layout}_{w}.npz. The set of a cell is the union over targets of the first k keys (nested in k).
The same positional sets are used in passes 1-3 (pass-2/3 filler keeps every exposure unit at the same token position, drift <= 2); only the pass-1 stream is read for scores.
Candidates: the 8K window before the chunk minus the first 4 stream positions and the last 128 tokens. The write step is Ladder3.write_step (masked write forward, natural read state advance), update rule / clip / lr as R (L0fs, L0mwfs).
Per chunk and pass the log keeps: n_masked, n_cand, win_len (min(8192, chunk start)), n_O, n_Oc (O inside the candidates), tp (masked O in the candidates), gnorm, clip coef, update norm, ce.
Cell nativerep = the R run's own native-fs call (run_L0_fillerswap, ONE pass, same code path as L0fs) on both worlds: the inner-weight hash of its pass-1 state is compared with the hash of the R run's L0fs pass-1 state (reuse check, files nat_attn_repro_s{stream}_{A,B}.json); the job script stops before the attention cells on a mismatch.
Runs through nat_ladder.main (hooks), so --mode save / eval, --worlds, the state files and the hash check are the R job's.
Usage (GPU): nat_attn.py --nat 9105:spread --nat-dir DIR --cells sel,attn8,attn16,attn32,attn64 --passes 3 --worlds AB --tag nat --mode save|eval [--keep]   |   --selftest (CPU)"""
import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_signal as NS
CH,WIN,NCH=NS.CH,NS.WIN,NS.NCH
KS=(8,16,32,64);TOPK=256
def sel_path(out,stream,layout,w):return Path(out)/f'nat_attn_sel_s{stream}_{layout}_{w}.npz'
def chunk_set(order,k,K):
    """absolute masked positions of cell K at chunk k from the saved order [CH,>=K]"""
    assert K<=order.shape[1]
    u=NS.union_sets(order,NS.cand_mask(k),[K])[K];return k*CH-WIN+np.nonzero(u)[0]
def stat_row(S,k,sets,gn=None):
    """masked / candidate / O counts for chunk k (the selection signal columns)"""
    candm=NS.cand_mask(k);k0=k*CH;O=sets.oracle(k);Oc={p for p in O if k0-WIN<=p<k0 and candm[p-(k0-WIN)]};Ss={int(x) for x in S};tp=len(Ss&Oc)
    return {'n_masked':int(len(Ss)),'n_cand':int(candm.sum()),'win_len':int(min(WIN,k0)),'n_O':len(O),'n_Oc':len(Oc),'tp':tp}
def selftest():
    rng=np.random.default_rng(0);k=9;cm=NS.cand_mask(k);M=rng.random((CH,WIN)).astype(np.float32);Mw=M.copy();Mw[:,~cm]=0
    order=np.argsort(-Mw,axis=1,kind='stable')[:,:TOPK].astype(np.uint16);prev=None
    for K in KS:
        S=chunk_set(order,k,K);assert len(S)==len(set(S.tolist())) and all(cm[p-(k*CH-WIN)] for p in S)
        if prev is not None:assert set(prev.tolist())<=set(S.tolist())
        prev=S
    full=NS.union_sets(np.argsort(-Mw,axis=1,kind='stable'),cm,[64])[64];assert (np.nonzero(full)[0]==chunk_set(order,k,64)-(k*CH-WIN)).all(),'top-256 truncation changed a k=64 set'
    assert chunk_set(order,0,64).size==0
    k=12;k0=k*CH;tab=[dict(idx=0,t=0,pair='p0',ord=1,label='name_only',place=[],pall=[k0-3000],names=[k0-3001]),dict(idx=1,t=0,pair='p0',ord=2,label='fact',place=[k0+10],pall=[k0+10],names=[k0+5])]
    sets=NS.NatSets(tab,0);assert sets.oracle(k)==[k0-3001,k0-3000];S=np.array([k0-3000,k0-50,k0-2]);r=stat_row(S,k,sets);assert (r['n_masked'],r['n_O'],r['n_Oc'],r['tp'],r['win_len'])==(3,2,2,1,WIN),r
    print('nat_attn SELFTEST OK',{K:len(chunk_set(order,k,K)) for K in KS})
def make_class():
    """GPU only (server s3_attn / s3_stage1 / ladder imports); returns NatLadder3 = NatLadder + the attention machinery"""
    import nat_ladder as NL
    import s3_attn as SA
    import s3 as S3M
    import ladder as LD
    from ladder import clone_state,jnp,stage2,reuse
    class NatLadder3(NL.NatLadder,SA.Ladder3A):
        def sel_order(self,swap):
            """pass-1 stream scores at W0 -> {k: uint16 [CH,TOPK]} (cached on disk, produced once per world by cell 'sel')"""
            f=sel_path(LD.OUT,NL.NAT['stream'],NL.NAT['layout'],'B' if swap else 'A')
            if f.exists():z=np.load(f);return {int(n[1:]):z[n] for n in z.files}
            rt=self.rt;m=rt.base;ids=np.asarray(self.T.P[swap].ids)[:NCH*CH];mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);out={};t0=time.monotonic()
            with rt.mesh:
                pref_nat=rt.prefix(m,seq);st=clone_state(rt.state_suffix)
                for k in range(NCH):
                    c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH];A,Bl,M=self.capture(m,st,c,pref_c);Mw=np.asarray(M)[:,:WIN].copy();Mw[:,~NS.cand_mask(k)]=0.0
                    out[k]=np.argsort(-Mw,axis=1,kind='stable')[:,:TOPK].astype(np.uint16);st=self.read_state(m,st,c,pref_c)
                    if k%8==0:print(f'  sel {"B" if swap else "A"} chunk {k} {time.monotonic()-t0:.0f}s',flush=True)
            np.savez_compressed(f,**{f'k{k:02d}':v for k,v in out.items()});print('saved',f,flush=True);return out
        def pass_sets(self,wp,swap,ids,ps):
            text=NS.NW.realize(wp,swap)[0];e=stage2.enc(self.T.tok,text);n=min(len(e.ids),len(ids)-1)
            assert list(ids[1:1+n])==list(e.ids[:n]),f'pass {ps}: the stream ids differ from the encoding of the realized pass world (the exposure-unit table would be wrong)'
            table,_=NS.build_table(wp,swap,list(e.offsets),NCH*CH);s=NS.NatSets(table,0);s.passno=ps;return s
        def run_attn(self,swap,K,passes):
            rt=self.rt;log=[];m=rt.base;st_all=clone_state(rt.state_all);streams,worlds=self.streams_worlds(swap,passes);cum=0.0;order=self.sel_order(swap)
            with rt.mesh:
                log.append({'epoch':0,'margins':self.evaluate(m)})
                for ps in range(1,passes+1):
                    ids=np.asarray(streams[ps-1])[:NCH*CH];stats=[];t_pass=time.monotonic()
                    mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);pref_nat=rt.prefix(rt.base,seq);st_suf=clone_state(rt.state_suffix)
                    t0=time.monotonic();Kh,Vh,_=self.record(ids);print(f'  recorded natural K,V in {time.monotonic()-t0:.0f}s',flush=True)
                    sets=self.pass_sets(worlds[ps-1],swap,ids,ps)
                    for k in range(NCH):
                        t0=time.monotonic();old=m;c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH];S=chunk_set(order[k],k,K);keep=sets.keep_vector(k,S.tolist());Kc,Vc=self.ctx_kv(Kh,Vh,k)
                        pref_w=self.prefix_masked(m,jnp.asarray(ids[k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep))
                        newp,gn,coef,ce=self.write_step(m,st_all,st_suf,c,pref_w,jnp.asarray(keep));st_suf=self.read_state(m,st_suf,c,pref_c);m=self.set_inner(m,newp)
                        un=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(NL.jax.tree.leaves(m.inner_parameters()),NL.jax.tree.leaves(old.inner_parameters())))**0.5;cum+=un
                        stats.append({'k':k,'gnorm':float(gn),'coef':float(coef),'update_norm':un,'ce':float(ce),'sec':time.monotonic()-t0,**stat_row(S,k,sets)})
                        if k%8==0:print(f'  attn{K} {"B" if swap else "A"} pass {ps} chunk {k+1}/{NCH} {stats[-1]["sec"]:.1f}s masked {stats[-1]["n_masked"]}',flush=True)
                    cs=[s_['coef'] for s_ in stats]
                    log.append({'epoch':ps,'cum_update_norm':cum,'chunks':stats,'clip_bind_frac':sum(c_<1 for c_ in cs)/len(cs),'mean_coef':sum(cs)/len(cs),'n_masked_total':sum(s_['n_masked'] for s_ in stats),
                                'stream_hash':hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),'inner_hash':reuse._hash_pytree(m.inner_parameters())[0],'sec_pass':time.monotonic()-t_pass,'margins':self.evaluate(m)})
                    print(f'  attn{K} {"B" if swap else "A"} pass {ps}: cum upd {cum:.3f}, clip binds {log[-1]["clip_bind_frac"]:.3f}, masked tokens {log[-1]["n_masked_total"]}, {log[-1]["sec_pass"]:.0f}s',flush=True)
                    del Kh,Vh
            return log
    return NatLadder3
def install_hooks():
    import nat_ladder as NL
    import ladder as LD
    def post(L,a):
        if a.mode=='save':L.build3();L.build1();L.build_attn()
    def cell_sel(L,swap,a):
        L.sel_order(swap);return [{'sel':str(sel_path(LD.OUT,NL.NAT['stream'],NL.NAT['layout'],'B' if swap else 'A'))}]
    def cell_nativerep(L,swap,a):
        log=L.run_L0_fillerswap(swap,1,None)   # the R run's native-fs call with one pass (same code path, same pass-1 stream)
        w='B' if swap else 'A';ref=LD.OUT/f'L0fs_{a.tag}_s{NL.NAT["stream"]}_{w}.saved.json'
        if ref.exists():
            rl=json.load(open(ref))['log'];h_new=log[1]['margins'].get('_hash');h_ref=rl[1]['margins'].get('_hash')
            rep={'world':w,'nativerep_pass1_hash':h_new,'R_L0fs_pass1_hash':h_ref,'identical':h_new==h_ref,'ref_file':str(ref)}
            json.dump(rep,open(LD.OUT/f'nat_attn_repro_s{NL.NAT["stream"]}_{w}.json','w'),indent=1);print('REUSE CHECK (native pass 1 vs R run):','IDENTICAL' if rep['identical'] else 'DIFFERENT',rep,flush=True)
        else:print('REUSE CHECK skipped: no',ref,flush=True)
        return log
    def mk(K):return lambda L,swap,a:L.run_attn(swap,K,a.passes)
    NL.NAT_HOOKS.update(cls=make_class,post=post,cells={'sel':cell_sel,'nativerep':cell_nativerep,**{f'attn{K}':mk(K) for K in KS}})
def main():
    if '--selftest' in sys.argv:selftest();return
    import nat_ladder as NL
    install_hooks();NL.main()
if __name__=='__main__':main()
