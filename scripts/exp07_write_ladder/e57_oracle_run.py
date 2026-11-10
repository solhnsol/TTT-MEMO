"""E57 oracle / half-oracle arms and distance x visibility cells (GPU; new file; core-claims thread 11/10; the registered e57_natmask_run.py is imported, not edited). Same write run, reads, canary and logs as e57_natmask_run.main (clipped SGD, engine SGDSEL, K0 = 8), with
  NATORACLE : e57_oracle_masks.OracleUV (a sentence row does not see the earlier sentences of its own fact in the window)
  NATHALF   : e57_oracle_masks.HalfOracleUV (every row sees the previous 256 tokens; sentence rows also see the earlier sentences of their own fact)
  plus the arms of e57_natmask_run (NATS, NATMASK, NATMASKC, DIR...). Cells: the registered ones (v3_16_1, v3_64_4, ...) and runtime cells 'v3_<k>_<p>_a<n_app>' (e.g. v3_32_4_a16: 32 facts, period p = 4 = the gap in chunks between consecutive exposures of a fact, 16 exposures, 64 chunks; background books as v3_64_4), registered in gen_facts.CELLS at run time (gen_facts.py is not edited); read points: the registered cell's read_apps, for a runtime cell only the last exposure n_app (out-of-window read + in-window read of group 0), saved state at the same point.
  python3 e57_oracle_run.py --seed S --cell C --out O --arms NATS,NATMASK,NATORACLE,NATHALF [--mask-arm ALLQ256] [--tag T] [--smoke N] [--rerun] | --selftest"""
import sys,os,json,time,gc,re,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_natmask_run as NM
import e57_qband_run as QR
import e57_oracle_masks as OM
from e57_natmask_run import ENGINE,K0,KeyMatchedQ,arm_mode,swap_step,with_base,make_nat_loss_diag,register_engine
def register_cell(cell):
    """runtime cell 'v3_<k>_<p>_a<n_app>' -> gen_facts.CELLS entry (k facts, n = 1, period p, bank 300, n_app exposures, read only at the last exposure)"""
    import gen_facts as G
    m=re.fullmatch(r'v3_(\d+)_(\d+)_a(\d+)',cell)
    if m and cell not in G.CELLS:
        k,p,na=map(int,m.groups());assert k>=p,'facts k >= groups p';G.CELLS[cell]=dict(k=k,n=1,p=p,bank='300',n_app=na,read_apps=(na,));G.CELL_BOOKS[cell]=G.CELL_BOOKS['v3_64_4']
    return G.CELLS[cell]
def parse_arm2(name,mask_arm,rand_arm='RANDQ256P50'):
    if name=='NATORACLE':return 'ORACLE',1.0
    if name=='NATHALF':return 'HALF',1.0
    return NM.parse_arm(name,mask_arm,rand_arm)
def selftest():
    NM.K0=8;NM.selftest();OM.selftest()
    assert parse_arm2('NATORACLE','ALLQ256')==('ORACLE',1.0) and parse_arm2('NATHALF','ALLQ256')==('HALF',1.0) and parse_arm2('NATMASK','ALLQ256')==('ALLQ256',1.0)
    import gen_facts as G
    c=register_cell('v3_32_16_a16');assert c==dict(k=32,n=1,p=16,bank='300',n_app=16,read_apps=(16,)) and G.CELLS['v3_32_16_a16']['p']==16
    try:register_cell('v3_8_16_a16');raise SystemExit('k<p accepted')
    except AssertionError:pass
    print('e57_oracle_run selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--cell',default='v3_16_1');ap.add_argument('--save-apps',default='128');ap.add_argument('--out');ap.add_argument('--masks');ap.add_argument('--mask-arm',default='ALLQ256');ap.add_argument('--rand-arm',default='RANDQ256P50');ap.add_argument('--arms');ap.add_argument('--tag',default='synng');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--rerun',action='store_true');ap.add_argument('--nat-loss-every',type=int,default=8);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    cellinfo=register_cell(a.cell)
    import e55_run as R55,e55_read as R,e55_gpu as RG,gen_facts as G
    import e57_syn_pass as SP
    register_engine()
    seed=a.seed;OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True);arms=a.arms.split(',');ro=list(cellinfo['read_apps']);ri=[ro[-1]];sv=[ro[-1]] if a.save_apps=='128' and a.cell!='v3_16_1' else [int(x) for x in a.save_apps.split(',')];CH=1024
    masks={k:v for k,v in np.load(a.masks,allow_pickle=True).items()} if a.masks and os.path.exists(a.masks) else {};masks.setdefault(f'band_{a.mask_arm}',np.asarray(int(re.sub(r'\D','',a.mask_arm))));masks.setdefault(f'all_{a.mask_arm}',np.asarray(1)) if a.mask_arm.startswith('ALLQ') else None;masks['band_INF']=np.asarray(10**6);masks['all_INF']=np.asarray(1);spec={ar:parse_arm2(ar,a.mask_arm,a.rand_arm) for ar in arms}
    for ar,(ma,_) in spec.items():assert ma in ('NATQ','ORACLE','HALF') or f'band_{ma[5:] if ma.startswith("KEY") else ma}' in masks,('mask arm not in the mask file',ar,ma)
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    lg=lambda *x:print(*x,flush=True)
    enc,w,tf=R55.build(SP.synth_args(seed,a.cell),seed,read_apps=set(ro)|set(ri)|set(sv),arms=['NAT']);S=R55.streams(w,enc);nch=a.smoke or w['n_chunks'];p=w['p']
    ids,sha=S['A'];assert sha==G.stream_sha(G.realize(w)),'stream hash mismatch'
    P=R.plan(w,enc);items=P['items'];reads=[a_*p-1 for a_ in sv] if not a.smoke else [nch-1];canary_chunks=reads
    tok=rt.tok;raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w))
    R_=RG.Reads(rd,raw,len(items),G.NCAND,P['read'],out_tail,rt.base,ro,ri,log=lambda *x:lg(*x));lg(f'seed {seed}: W0 read done')
    for ar in arms:
        fn=OUT/f'E55_{a.tag}_{ar}_s{seed}.saved.json'
        if fn.exists() and not a.rerun:lg('exists, reused:',fn);continue
        ma,mult=spec[ar];uv=(OM.OracleUV(w['chunks'],K0,lg) if ma=='ORACLE' else OM.HalfOracleUV(w['chunks'],256,K0,lg) if ma=='HALF' else KeyMatchedQ(ma[3],masks[f'band_{ma[5:]}'],seed,lg) if ma.startswith('KEY') else QR.SelMaskQ(ma,masks,lg));L=C.Ladder55(rt,save_dir=OUT/'states'/f'E55_{a.tag}_{ar}_s{seed}',canary=can);got={}
        mode=arm_mode(ar);hold={};logfull=ma!='NATQ';uv=with_base(uv,hold,L if (mode or logfull) else None,LD.jnp,full_every=a.nat_loss_every if logfull else 0,always_full=bool(mode))
        L.write_step=swap_step(L.write_step,mult,mode,hold,every=a.nat_loss_every if logfull else 0);dchunks={c_ for c_ in range(K0,nch,a.nat_loss_every)}|{nch-1} if a.nat_loss_every>0 else set();dfn=make_nat_loss_diag(L,LD,s3,rt,hold,lg) if dchunks else None
        def partial(k,m,ar=ar,got=got):
            x=R_.at((k+1)//p,m);got[(k+1)//p]=x
            rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':a.cell,'seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':False})[ar];json.dump(rec,open(OUT/f'E55_{a.tag}_{ar}_s{seed}.read.partial.json','w'))
            return {c_:'ok' for c_ in x}
        hook_chunks=[a_*p-1 for a_ in sorted(set(ro)|set(ri))] if not a.smoke else [nch-1]
        t0=time.monotonic();lg(f'== seed {seed} {ar} (mask {ma}, lr x{mult:g}): {nch} chunks, reads out {ro} in {ri}')
        r=L.run_stream(ENGINE,p,ids,decoy_ids=S['decoy'][0] if 'decoy' in S else None,bg_ids=S['nofact'][0] if 'nofact' in S else (S['bg'][0] if 'bg' in S else None),read_chunks=reads,canary_chunks=canary_chunks,nch=nch,tag=ar+f'_s{seed}',log=lg,read_hook=partial,hook_chunks=hook_chunks,chunks_meta=w['chunks'],uv=uv,diag_fn=dfn,diag_chunks=dchunks);r.pop('model',None);r['wlog']=hold.get('wlog',[]);r['swap_mode']=mode
        lgs=r['log'];r['summary']={'mean_update_norm':float(np.mean([e['update_norm'] for e in lgs])),'clip_bind_frac':r['clip_bind_frac'],'cum_update_norm':r['cum_update_norm'],'mean_ce':float(np.mean([e['ce'] for e in lgs])),'mean_gnorm':float(np.mean([e['gnorm'] for e in lgs])),'mean_coef':float(np.mean([e['coef'] for e in lgs])),'median_update_norm':float(np.median([e['update_norm'] for e in lgs])),'clip_frac_coef_lt1':float(np.mean([e['coef']<1.0 for e in lgs])),'mean_masked_row_frac':float(np.mean([e.get('q_banded',0) for e in lgs])/CH),'mean_hidden_keys_row':float(np.mean([e['hidden_keys_row_mean'] for e in lgs if 'hidden_keys_row_mean' in e] or [0]))}
        r.update(stream_sha=sha,world_seed=seed,cell=a.cell,read_apps_out=ro,read_apps_in=ri,save_apps=sv,sec_total=time.monotonic()-t0,k=w['k'],n=w['n'],bank=w['bank_kind'],engine='SGD(clip)',mask_arm=ma,band=(int(masks[f'band_{ma[5:] if ma.startswith("KEY") else ma}']) if ma not in ('NATQ','ORACLE','HALF') else None),lr_mult=mult,arm=ar)
        w0=r['canary_w0']
        for k,e in r['reads'].items():
            if 'canary' in e:
                for key in ('_gen_nll','_tail_nll'):lg(f'  canary {key} at chunk {k}: {e["canary"][key]-w0[key]:+.4f}')
        rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':a.cell,'seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':True,'place_len':w['place_len']})[ar];fr=OUT/f'E55_{a.tag}_{ar}_s{seed}.read.json';json.dump(rec,open(fr,'w'));json.dump(r,open(fn,'w'),default=float)
        lg('wrote',fn,f'{time.monotonic()-t0:.0f}s');gc.collect();LD.jax.clear_caches()
if __name__=='__main__':main()
