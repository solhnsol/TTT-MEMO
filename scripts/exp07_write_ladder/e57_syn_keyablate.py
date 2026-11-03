"""E57 synthetic s64, key-level masking, GPU DIAGNOSTIC (new file; GPU forward only, W0, NO weight update; method thread card 'M4 + 키 단위 가림' part (b), Hans '진단 먼저'): on the deciding rows (top 30% by window gain g = nll_band256 - nll_win, syn_s64_m30_masks.npz) of the chunks 16..127 the NLL when only a set of far keys (window keys beyond the band 256, i.e. 4 <= c <= i + 8192 - 256) is removed from the natural window of the row: ALLV top m, ALL top m (from the W0 key-mass dump, e57_syn_s64/syn_s64_dump_j*.npz), m uniformly random far keys, and ALL far keys of the SAME FACT (oracle, from the world table), m = 16 and 64. Removed gain = nll_masked - nll_win; fraction of g removed = sum(removed gain) / sum(g) over the rows. Natural-window rows are re-scored at every 16th chunk as a check of the masked-pass path (max |diff| to the stored nll_win). Output <out>/e57_syn_keyablate.npz (per variant [112,1024] NLL) + e57_syn_keyablate.json.
  python3 e57_syn_keyablate.py --out O [--chunks 16,127] | --selftest"""
import sys,json,time,gc,hashlib,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;WIN=8192;BAND=256;MS=(16,64);VARIANTS=[f'{n}{m}' for n in ('ALLV','ALL','RAND') for m in MS]+['ORACLE']
def key_sets(k,dec_rows,D,fact,rng):
    """-> {variant: list (per deciding row) of int arrays of window-key indices to remove}"""
    out={v:[] for v in VARIANTS};win0=(k-8)*CH
    for i in dec_rows:
        lim=min(WIN-1,i+WIN-BAND);far=np.arange(4,lim+1)
        for nm in ('ALLV','ALL'):
            idx=D[f'c{k}/{nm}_b{BAND}_idx'][i].astype(np.int64);idx=idx[idx>=0]
            for m in MS:out[f'{nm}{m}'].append(idx[:m])
        for m in MS:out[f'RAND{m}'].append(rng.choice(far,size=min(m,len(far)),replace=False))
        rf=fact[k*CH+i+1];out['ORACLE'].append(far[fact[np.clip(win0+far,0,len(fact)-1)]==rf] if rf>=0 else far[:0])
    return out
def masked_keep(keep1,dec_rows,sets):
    kp=np.broadcast_to(keep1,(CH,WIN+CH)).copy()
    for i,ks in zip(dec_rows,sets):kp[i,ks]=False
    return kp
def selftest():
    keep1=np.ones(WIN+CH,bool);kp=masked_keep(keep1,[3,5],[np.asarray([10,20]),np.asarray([],np.int64)]);assert not kp[3,10] and not kp[3,20] and kp[3,11] and kp[5].all() and kp[4].all()
    fact=np.full(40*CH,-1,np.int32);fact[(20-8)*CH+100:(20-8)*CH+110]=7;fact[20*CH+1+5]=7
    class DD(dict):pass
    D={'c20/ALLV_b256_idx':np.tile(np.arange(128,dtype=np.int16),(CH,1)),'c20/ALL_b256_idx':np.tile(np.arange(128,dtype=np.int16)[::-1],(CH,1))}
    s=key_sets(20,[5],D,fact,np.random.default_rng(0));assert len(s['ORACLE'][0])==10 and (s['ORACLE'][0]>=100).all() and len(s['ALLV16'][0])==16 and len(s['RAND64'][0])==64
    print('e57_syn_keyablate selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--chunks',default='16,127');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import jax,jax.numpy as jnp,equinox as eqx,types,e55_run as R55,e57_syn_pass as SP,e57_dwg_analyze as DW,e57_pqa2_stream as S
    k0,k1=[int(x) for x in a.chunks.split(',')];out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True)
    enc,w,tf=R55.build(SP.synth_args(64),64,arms=['NAT']);Sx=R55.streams(w,enc);ids,sha=Sx['A'];nch=128;ids=np.asarray(ids)[:nch*CH];fact=np.full(nch*CH+1,-1,np.int32)
    for c in w['chunks']:
        for s in c['sents']:fact[c['k']*CH+s['start']:c['k']*CH+s['end']]=s['fact']
    M=np.load(EXP/'e57_synq/syn_s64_m30_masks.npz');Z=np.load(EXP/'e57_syn_s64/syn_s64_nll.npz');dumps={}
    for j in range(8):
        D=np.load(EXP/f'e57_syn_s64/syn_s64_dump_j{j:02d}.npz');
        for k in [int(x) for x in D['chunks']]:dumps[k]={f'c{k}/{nm}_b256_idx':D[f'c{k}/{nm}_b256_idx'] for nm in ('ALLV','ALL')}
    LD,s3,C=R55.lab();rt=LD.Runtime()
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    t0=time.monotonic();L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
    del _h;rc={(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)};del L0;gc.collect();lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s')
    res={v:np.zeros((nch,CH),np.float32) for v in VARIANTS};nat_chk=[];nkeys={v:[] for v in VARIANTS};decm=np.zeros((nch,CH),bool)
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        keep=np.asarray(keep,bool);dec=CM.up(M[f'qb_TOP30_{k}'],CH);rows=np.where(dec)[0];decm[k]=dec;rng=np.random.default_rng(1000+k);sets=key_sets(k,rows,dumps[k],fact,rng)
        def nll(kp):
            pm=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(kp));return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pm,jnp.asarray(kp)),np.float64)
        with rt.mesh:
            for v in VARIANTS:
                res[v][k]=nll(masked_keep(keep,rows,sets[v])).astype(np.float32);nkeys[v].append(float(np.mean([len(x) for x in sets[v]])))
            if k%16==0:
                nat=nll(np.broadcast_to(keep,(CH,WIN+CH)).copy());nat_chk.append(float(np.abs(nat-Z['nll_win'][k]).max()))
        if k%16==0:lg(f'  chunk {k}: {time.monotonic()-t0:.0f}s, natural-path check max|diff| {nat_chk[-1]:.2e}')
        return dict(done=True)
    fn.wants_ctx=True
    L=C.Ladder55(rt,save_dir=out/'states'/'score',canary=None);L.rec_cache=rc
    _ws=L.write_step;L.write_step=lambda m,*a_,_ws=_ws,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]
    L.run_stream('NAT',1,ids,nch=nch,tag='keyablate',log=lambda *x:None,chunks_meta=[{'sents':[]} for _ in range(nch)],diag_fn=fn,diag_chunks=set(range(k0,k1+1)))
    np.savez_compressed(out/'e57_syn_keyablate.npz',dec=decm,**res)
    ch=list(range(k0,k1+1));g=(Z['nll_band'][0]-Z['nll_win']).astype(np.float64);summ={}
    rowfact=np.stack([fact[k*CH+np.arange(CH)+1] for k in range(nch)])
    for sub,name in ((decm,'deciding_rows'),(decm&(rowfact>=0),'deciding_rows_with_a_fact')):
        r={}
        for v in VARIANTS:
            dl=(res[v].astype(np.float64)-Z['nll_win'])[:,:];num=np.asarray([dl[k][sub[k]].sum() for k in ch]);den=np.asarray([g[k][sub[k]].sum() for k in ch]);ix=np.random.default_rng(1).integers(0,len(ch),(2000,len(ch)))
            fr=num.sum()/den.sum();bs=num[ix].sum(1)/den[ix].sum(1);nk=float(np.mean(nkeys[v]));r[v]=dict(fraction_of_g_removed=float(fr),ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],mean_removed_gain_per_row=float(num.sum()/sub[ch].sum()),mean_g_per_row=float(den.sum()/sub[ch].sum()),mean_keys_removed_per_deciding_row=nk,fraction_removed_per_key=float(fr/nk) if nk>0 else None)
        summ[name]=dict(n_rows=int(sub[ch].sum()),variants=r)
    (out/'e57_syn_keyablate.json').write_text(json.dumps(dict(chunks=[k0,k1],natural_path_check_max_abs_diff=nat_chk,summary=summ,sec=time.monotonic()-t0),indent=1));print(json.dumps(summ)[:1500])
if __name__=='__main__':main()
