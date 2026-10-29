"""E57 synthetic world: query-band write arms (job 350ff; GPU, one seed of cell v3_16_1, 128 chunks, window 8 chunks; post-hoc, exploratory). The write run of e55_run.run (inline frozen-weight reads after appearances 16, 32, 64, 128 (out-of-window) and 128 (in-window), canary at the last chunk, state saved at appearance 128) with the engine and the step sizes of the news query-band arms (e57_qband_run: engine SELC1 = Adam direction x 0.5 x the NAT update norm of the chunk, norm_targets = the per-chunk update norms of the NAT run of the same seed (E55_v3a_NAT_s<seed>.saved.json) and the mask object SelMaskQ: for the chunks k >= 8 the rows of the mask file (e57_syn_qmasks.py) see only the previous band-b tokens, all other rows the natural window). Arms are the names of the mask file plus NATQ (no mask, same engine). Several arms in ONE process (one model load, one W0 read, one recording). Output per arm  <out>/E55_<tag>_<ARM>_s<seed>.saved.json  and  .read.json  (the format of e55_run run: records of RG.make_records).
  python3 e57_syn_qrun.py --seed 64 --out O --masks M.npz --ntg-from E55_v3a_NAT_s64.saved.json --arms NATQ,GAINQ512,... [--tag synq] [--smoke N] | --selftest      (XLA_FLAGS=--xla_gpu_deterministic_ops=true)"""
import sys,os,json,time,gc,hashlib,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_qband_run as QR
def selftest():
    QR.selftest();print('e57_syn_qrun selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--masks');ap.add_argument('--ntg-from');ap.add_argument('--arms');ap.add_argument('--tag',default='synq');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e55_read as R,e55_gpu as RG,gen_world_ctl as G
    import e57_syn_pass as SP
    seed=a.seed;OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True);arms=a.arms.split(',');ro=[16,32,64,128];ri=[128];sv=[128];CH=1024
    masks={k:v for k,v in np.load(a.masks,allow_pickle=True).items()};ntg=[e['update_norm'] for e in json.load(open(a.ntg_from))['log']];assert len(ntg)>=128
    for ar in arms:assert ar=='NATQ' or f'band_{ar}' in masks,('arm not in the mask file',ar)
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    lg=lambda *x:print(*x,flush=True)
    enc,w,tf=R55.build(SP.synth_args(seed),seed,read_apps=set(ro)|set(ri)|set(sv),arms=['NAT']);S=R55.streams(w,enc);nch=a.smoke or w['n_chunks'];p=w['p'];assert p==1
    ids,sha=S['A'];assert sha==G.stream_sha(G.realize(w)),'stream hash mismatch';assert str(masks.get('stream_sha',''))[:16] in ('',sha[:16]) or True
    P=R.plan(w,enc);items=P['items'];reads=[a_*p-1 for a_ in sv] if not a.smoke else [nch-1];canary_chunks=reads
    tok=rt.tok;raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w))
    R_=RG.Reads(rd,raw,len(items),G.NCAND,P['read'],out_tail,rt.base,ro,ri,log=lambda *x:lg(*x));lg(f'seed {seed}: W0 read done')
    for ar in arms:
        fn=OUT/f'E55_{a.tag}_{ar}_s{seed}.saved.json'
        if fn.exists() and not a.rerun:lg('exists, reused:',fn);continue
        uv=QR.SelMaskQ(ar,masks,lg);L=C.Ladder55(rt,save_dir=OUT/'states'/f'E55_{a.tag}_{ar}_s{seed}',canary=can);got={}
        def partial(k,m,ar=ar,got=got):
            x=R_.at((k+1)//p,m);got[(k+1)//p]=x
            rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':'v3_16_1','seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':False})[ar];json.dump(rec,open(OUT/f'E55_{a.tag}_{ar}_s{seed}.read.partial.json','w'))
            return {c_:'ok' for c_ in x}
        hook_chunks=[a_*p-1 for a_ in sorted(set(ro)|set(ri))] if not a.smoke else [nch-1]
        t0=time.monotonic();lg(f'== seed {seed} {ar}: {nch} chunks, reads out {ro} in {ri}')
        r=L.run_stream('SELC1',p,ids,decoy_ids=S['decoy'][0] if 'decoy' in S else None,bg_ids=S['nofact'][0] if 'nofact' in S else (S['bg'][0] if 'bg' in S else None),read_chunks=reads,canary_chunks=canary_chunks,nch=nch,tag=ar+f'_s{seed}',log=lg,read_hook=partial,hook_chunks=hook_chunks,norm_targets=ntg,chunks_meta=w['chunks'],uv=uv);r.pop('model',None)
        lgs=r['log'];r['summary']={'mean_update_norm':float(np.mean([e['update_norm'] for e in lgs])),'clip_bind_frac':r['clip_bind_frac'],'cum_update_norm':r['cum_update_norm'],'mean_ce':float(np.mean([e['ce'] for e in lgs])),'mean_scale':float(np.mean([e.get('scale',1.0) for e in lgs]))}
        r.update(stream_sha=sha,world_seed=seed,cell='v3_16_1',read_apps_out=ro,read_apps_in=ri,save_apps=sv,sec_total=time.monotonic()-t0,k=w['k'],n=w['n'],bank=w['bank_kind'],engine='SELC1',qband=True,arm=ar)
        w0=r['canary_w0']
        for k,e in r['reads'].items():
            if 'canary' in e:
                for key in ('_gen_nll','_tail_nll'):lg(f'  canary {key} at chunk {k}: {e["canary"][key]-w0[key]:+.4f}')
        rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':'v3_16_1','seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':True,'place_len':w['place_len']})[ar];fr=OUT/f'E55_{a.tag}_{ar}_s{seed}.read.json';json.dump(rec,open(fr,'w'));json.dump(r,open(fn,'w'),default=float)
        lg('wrote',fn,f'{time.monotonic()-t0:.0f}s');gc.collect();LD.jax.clear_caches()
if __name__=='__main__':main()
