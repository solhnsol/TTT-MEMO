"""E57 1a-1 style canary trajectory (GPU; NAT single arm; forward + one inner update per 1024-token chunk; NO reads, NO questions): the natural news stream of ONE cycle (all independent lead articles of the events of the bundle, global date order, lead sentences only, separated by a blank line, real llama3 tokenizer, cut to whole 1024-token chunks) is run through e55_core.Ladder55.run_stream('NAT'); at the chunks --points the held-out canary (pg11 generation NLL and the name-scrubbed pg1342 tail NLL, e55_run.Canary55) is measured, and at W0. Result: a trajectory of the canary rise against the number of updates (compare E55: +0.023 SGD at 128 updates, limit +0.10) and the per-chunk update log. The CPU part (--check-only, also with --stub) builds the stream and prints chunk count / hash.
  python3 e57_canary_run.py --bundle OUT/e57_bundle --out DIR [--points 15,31,...] [--check-only] [--stub] [--tokenizer F]"""
import sys,json,os,time,argparse,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'e57'))
CH=1024
def build_ids(bundle_parent,enc):
    import e57_bank as B
    st,idx=B.load(bundle_parent);items=[]
    for ev,E in st.items():
        for ai in range(len(E['rids'])):items.append((int(E['dates'][ai]),ev,E['rids'][ai],' '.join(E['sents'][ai][:max(1,E['kept'][ai])])))
    items.sort(key=lambda x:(x[0],x[1],x[2]));ids=[]
    for d,ev,rid,txt in items:ids+=list(enc(txt+'\n\n'))
    n=len(ids)//CH;arr=np.asarray(ids[:n*CH],np.int64);return arr,len(items),len(ids)
def points_default(n,every=24):return sorted({k for k in range(every-1,n,every)}|{n-1})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True,help='OUT dir that holds e57_bundle');ap.add_argument('--out',required=True);ap.add_argument('--points');ap.add_argument('--check-only',action='store_true');ap.add_argument('--stub',action='store_true');ap.add_argument('--tokenizer');ap.add_argument('--rerun',action='store_true');a=ap.parse_args()
    ROOT=HERE.parents[1]
    sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
    import gen_world_ctl as G
    enc=G.StubTok() if a.stub else G.load_enc(a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
    ids,narts,ntok=build_ids(a.bundle,enc);n=len(ids)//CH;pts=[int(x) for x in a.points.split(',')] if a.points else points_default(n)
    assert n>=2 and all(0<=p<n for p in pts),(n,pts);sha=hashlib.sha256(ids.tobytes()).hexdigest()[:16]
    print(f'stream: {narts} articles, {ntok} tokens, {n} chunks of {CH} (tail of {ntok-n*CH} tokens cut), canary at chunks {pts}, ids sha {sha}',flush=True)
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);fn=out/'E57_canary_traj_NAT.json'
    if a.check_only:print('canary trajectory check-only OK');sys.exit(0)
    if fn.exists() and not a.rerun:print('exists, not redone (exit 8):',fn);sys.exit(8)
    import e55_run as R
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);L=C.Ladder55(rt,save_dir=out/'states'/'E57_canary_traj',canary=can);t0=time.monotonic()
    r=L.run_stream('NAT',1,ids,nch=n,tag='canaryNAT',log=lambda *x:print(*x,flush=True),read_chunks=set(pts),canary_chunks=set(pts))
    traj=[{'chunk':int(k),'updates':int(k)+1,'canary':v.get('canary')} for k,v in sorted(r['reads'].items())]
    res=dict(arm='NAT',n_chunks=n,n_articles=narts,tokens=ntok,ids_sha=sha,canary_w0=r.get('canary_w0'),trajectory=traj,log=[{k:e[k] for k in ('k','ce','update_norm','coef','sec') if k in e} for e in r['log']],sec_total=time.monotonic()-t0,
        e55_reference='E55 canary rise at 128 updates: SGD +0.023, 0.5N +0.043, Adam +0.159; limit 0.10 (E57 doc 12.2)')
    fn.write_text(json.dumps(res,default=float));print('wrote',fn)
