"""Exposure-DENSITY experiment (E2E 3B, synthetic world; GPU; new file; thread-server 11/19; EXPLORATORY; Hans: exposures per fact fixed at 16, only the repeat gap changes; arms NAT vs MASK1024; main metric = out-of-window forward margin after the last exposure (W0 recorded), plus the STEP-MATCHED CONTROL: every condition is padded with background-only chunks to the same total number of write steps).
Gap grid (16 facts, 16 exposures per fact; chunks of 1024 tokens): n2p1 (2 sentences of each fact per chunk, 8 chunks, mean gap ~500 tokens), n1p1 (16 chunks, ~1000), n1p2 (32, ~2050), n1p4 (64, ~4100), n1p8 (128, ~8200), n1p16 (256, ~16400). UNMATCHED = the original design (steps = chunks of the condition); MATCHED = the background-only chunks of a fill stream (the 'nofact' stream of an independent world, seed + 1000, so no fact name / place and different background order) are put BEFORE the condition's chunks so that all conditions have --total (256) steps and the read follows right after the last exposure (same lag to the read). Arms: NAT (original clipped SGD rule, K0 = 0) and MASK1024 (band-1024 mask on every row in ALL 32 layers, original rule) exactly as job 1313 / e59_srv_run; forward readout at the end (out neutral tail, 16 candidates, summed span margin); W0 margin per condition; per-step ce / update norm in the json.
  python3 e59_dens_run.py --seed 241 --out O [--conditions n2p1,n1p1,n1p2,n1p4,n1p8,n1p16] [--matched both|0|1] [--arms NAT,MASK1024] | --selftest"""
import sys,os,json,time,gc,argparse,pathlib,types
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;TOTAL=256
COND={'n2p1':(2,1,8),'n1p1':(1,1,16),'n1p2':(1,2,16),'n1p4':(1,4,16),'n1p8':(1,8,16),'n1p16':(1,16,16)}      # (sentences per fact and chunk, period in chunks, chunks with the fact = n_app)
def register():
    import gen_facts as G
    for k,(n,p,na) in COND.items():
        name=f'dens_{k}';G.CELLS[name]=dict(k=16,n=n,p=p,bank='300',n_app=na,read_apps=(na,));G.CELL_BOOKS[name]=G.CELL_BOOKS['v3_64_4']
    return G
def build_cond(R55,G,cond,seed):
    n,p,na=COND[cond];a=types.SimpleNamespace(cell=f'dens_{cond}',seed=seed,tokenizer_file=None,bg='pool',decoy='names',stub=False,arms='NAT')
    enc,w,tf=R55.build(a,seed,read_apps={na},arms=['NAT']);return enc,w
def build_fill(R55,G,seed,fill_offset=1000):
    a=types.SimpleNamespace(cell='dens_n1p16',seed=seed+fill_offset,tokenizer_file=None,bg='pool',decoy='names',stub=False,arms='SWB')
    enc,w,tf=R55.build(a,seed+fill_offset,read_apps={16},arms=['SWB']);ch=G.realize(w,'nofact');assert len(ch)>=TOTAL-8 and all(len(c['ids'])==CH for c in ch);return ch
def stream_ids(real_chunks,fill_chunks,total,matched):
    r=(total-len(real_chunks)) if matched else 0;assert r>=0
    ids=[t for c in fill_chunks[:r] for t in c['ids']]+[t for c in real_chunks for t in c['ids']];return ids,r
def run(a):
    import jax,e55_run as R55,e55_gpu as RG,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,gen_facts as G
    G=register();NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    LD,s3,C=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp);tok=rt.tok
    fill=build_fill(R55,G,seed);modes=[False,True] if a.matched=='both' else [a.matched=='1']
    for cond in a.conditions.split(','):
        enc,w=build_cond(R55,G,cond,seed);real=w['chunks'];items=G.build_items(w);raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w))
        R_=None
        for matched in modes:
            if matched and len(real)>=TOTAL:continue          # p16: identical to the unmatched run
            tag=f'{cond}_{"m" if matched else "u"}'
            for ar in a.arms.split(','):
                fn=out/f'DENS_{tag}_{ar}_s{seed}.json'
                if fn.exists() and not a.rerun:lg('exists, reused:',fn);continue
                ids,r=stream_ids(real,fill,TOTAL,matched);nch=len(ids)//CH;assert nch==len(real)+r
                if R_ is None:R_=RG.Reads(rd,raw,len(items),G.NCAND,{},out_tail,rt.base,(1,),(),log=lg);lg(f'{cond}: W0 read done')
                uv=QR.SelMaskQ('ALLQ1024',{'band_ALLQ1024':np.asarray(1024),'all_ALLQ1024':np.asarray(1)},lg) if ar=='MASK1024' else QR.SelMaskQ('NATQ',{},lg)
                L=ISO.Iso55(rt,save_dir=out/'states'/f'DENS_{tag}_{ar}_s{seed}',canary=None);got={};t0=time.monotonic();metas=[{'sents':[]} for _ in range(r)]+list(real)
                def hook(k,m,got=got):
                    got['out']=R_.at(1,m)['out'];return {'out':'ok'}
                lg(f'== seed {seed} {tag} {ar}: {nch} steps ({r} background-only first), gap mean tokens ~{CH*COND[cond][1]/COND[cond][0]:.0f}')
                res=L.run_stream(NM.ENGINE,1,ids,read_chunks=[],canary_chunks=[],nch=nch,tag=f'{tag}_{ar}_s{seed}',log=lg,read_hook=hook,hook_chunks=[nch-1],chunks_meta=metas,uv=uv);res.pop('model',None)
                meta=G.item_meta(items);S=np.asarray(got['out']);W0=np.asarray(R_.W0['out'])
                json.dump(dict(cond=cond,matched=bool(matched),arm=ar,seed=seed,n_steps=nch,n_bg_first=r,n_cond_chunks=len(real),mean_gap_tokens=CH*COND[cond][1]/COND[cond][0],read={'out':S.tolist()},W0={'out':W0.tolist()},meta={k:meta[k] for k in ('kind','gi')},log=[dict(k=e['k'],ce=e['ce'],update_norm=e['update_norm'],gnorm=e['gnorm']) for e in res['log']],sec=time.monotonic()-t0,code=os.popen('git rev-parse HEAD 2>/dev/null').read().strip()),open(fn,'w'),default=float)
                lg('wrote',fn,f'{time.monotonic()-t0:.0f}s');import shutil;shutil.rmtree(out/'states'/f'DENS_{tag}_{ar}_s{seed}',ignore_errors=True);gc.collect();LD.jax.clear_caches()
def selftest():
    import e55_run as R55,gen_facts as G
    from tokenizers import Tokenizer
    G=register();tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));fill=build_fill(R55,G,241);res={}
    for cond,(n,p,na) in COND.items():
        enc,w=build_cond(R55,G,cond,241);real=w['chunks'];assert len(real)==na*p//n if n==1 else len(real)==na,(cond,len(real))
        ex=[sum(1 for c in real for s in c['sents'] if s['fact']==f) for f in range(16)];assert set(ex)=={16},(cond,ex)
        for matched in (False,True):
            ids,r=stream_ids(real,fill,TOTAL,matched);assert len(ids)%CH==0 and (len(ids)//CH==TOTAL if matched or len(real)==TOTAL else len(ids)//CH==len(real))
        res[cond]=(len(real),TOTAL-len(real))
    names=[f['X'] for f in w['facts']]+[f['P'] for f in w['facts']]
    txt=''.join(tk.decode(c['ids']) for c in fill[:50])
    assert not any(x in txt for x in names),'fact names / places occur in the fill stream'
    items=G.build_items(w);assert len(items)==20 and all(len(it['cands'])==16 for it in items)
    print('e59_dens_run selftest ok: chunks/background-first per condition',res)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--conditions',default=','.join(COND));ap.add_argument('--matched',default='both',choices=['both','0','1']);ap.add_argument('--arms',default='NAT,MASK1024');ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    run(a)
if __name__=='__main__':main()
