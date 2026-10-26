"""E55 32.13 LEARNED head-weighting masks as a WRITE-experiment input (mask stage; GPU forward only, no write step). Same role as e55_run.py selmasks (job 231/232) but the window sentences are chosen by the logistic regression on the per-(layer, head) sentence attention mass trained on seed 61 (e55_sentgrid_ext.py; LABELS WERE USED to train the coefficients, on seed 61 only; they are FROZEN here and never refit): for every chunk k = 1..127 of the stream of --seed (cell v3_16_1) the prefix layers (frozen, independent of the write state) give AL[layer, head, query sentence, window sentence] for the write chunk against the unmasked real 8-chunk window; feature = log(sum over the query sentences of AL + 1e-4) per (layer, head), standardised with the stored mu / sd, logit = w.x + b; the highest-scoring candidate sentences (>= 4 tokens) are hidden until --frac of the window tokens is hidden (fixed budget; frac 0.33 is the mask share at which the learned ranking reaches recall .90 on the seed-61..64 dumps, 0.324-0.330, chosen with labels on those dumps). The fact table of the stream enters only the per-chunk label statistics of the report.
Output: <out>/e55_lmasks_s<seed>.npz (selmask format, rank 1: keys h1_<k>, wlen<k>) + e55_lmasks_report_s<seed>.md + .json; the write stage is the unchanged `e55_run.py decomp ... --arms SELC1,RSELC1 --sel-masks e55_lmasks_s{seed}.npz --sel-cands <candidate json>` (SELC1 = learned masks, RSELC1 = random sentences with the same hidden tokens per chunk).
  python3 e55_lmask_run.py --seed 62 --out DIR --coef e55_sentgrid_ext_s61_coef.npz [--frac 0.33] [--check-only] | --selftest"""
import sys,os,json,time,argparse,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[1]/'scripts/exp04_what_ttt_retains'))
import e55_sentgrid_ext as X
MODEL_KEY='NAT/sum/all/100'
def load_model(coef):
    Z=np.load(coef);return {nm:Z[f'{MODEL_KEY}/{nm}'] for nm in ('w','b','mu','sd')}
def learned_hidden(d,model,frac,seed=0):
    """d: arrays of ONE chunk (needs AL float16 [L,H,nQ,nW] and w_len) -> hidden bool [nW] (candidates with >= 4 tokens, highest logit first, until frac of ALL window tokens is hidden)"""
    w_len=np.asarray(d['w_len']);nW=len(w_len)
    if nW==0:return np.zeros(0,bool)
    Xf,cand=X.features(d,'sum','all');z=np.full(nW,-np.inf);z[cand]=X.predict_lr(model,Xf)
    return X.hide_to_fraction(z,w_len,frac,np.random.default_rng(seed))
def compute(L,rt,s3,ends,all_ids,model,frac,chunks_meta=None,nch=128,k0=1,log=print,passfn=None):
    """-> (hidden {k: {1: bool [nW]}}, window sentence lengths {k: [nW]}, pooled label counts, per-chunk metrics); passfn(k, ids_k) -> d replaces the GPU pass in the CPU test"""
    import e55_sentdump as SD,e55_sentgrid as G
    if passfn is None:
        with rt.mesh:Kh,Vh,_=L.record(np.asarray(all_ids)[:nch*SD.CH])
        fn=SD.make_sentdump_fn(L,rt,s3,ends,all_ids,log=lambda *x:None)
    hid={};wl={};cnt=np.zeros(6);per={}
    for k in range(k0,nch):
        ids_k=np.asarray(all_ids[k*SD.CH:(k+1)*SD.CH])
        if passfn is None:
            with rt.mesh:Kc,Vc=L.ctx_kv(Kh,Vh,k)
            d=fn(k,rt.base,Kc,Vc,None,ids_k,None)
        else:d=passfn(k,ids_k)
        h=learned_hidden(d,model,frac);hid[k]={1:h};wl[k]=np.asarray(d['w_len'])
        if chunks_meta is not None and k>=SD.NW and len(h):
            import e55_sentdump_labels as LB
            lb=LB.labels(chunks_meta,ends,k);c=G.counts(h,d,lb);cnt+=c;per[k]=G.metrics(c)
        if k%16==0:log(f'  learned mask chunk {k}: hidden sentences {int(h.sum())}/{len(h)}')
    return hid,wl,cnt,per
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--coef');ap.add_argument('--frac',type=float,default=0.33);ap.add_argument('--check-only',action='store_true');ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R,e55_uv as UVm,e55_selmask as SM,e55_sentgrid as G
    OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True);sd=a.seed;model=load_model(a.coef);assert model['w'].shape==(768,),model['w'].shape
    fn=OUT/f'e55_lmasks_s{sd}.npz'
    ns=argparse.Namespace(seed=sd,cell='v3_16_1',bg='pool',tokenizer_file=None,arms='NAT',decoy='names',stub=False)
    if a.check_only:
        enc,w,tf=R.build(ns,sd,arms=['NAT']);S=R.streams(w,enc);ids,_=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids)
        for k in (8,64,127):print(f'  chunk {k}: {len(UVm.chunk_sents(ends,k))} query sentences, {len(UVm.window_sents(ends,k)[1])} window sentences',flush=True)
        print(f'lmask check-only OK: seed {sd}, coefficients {a.coef} ({MODEL_KEY}, 768 weights), budget {a.frac}; output {fn} ({"exists" if fn.exists() else "to be written"})');return
    if fn.exists() and not a.rerun:print('exists, reused:',fn);return
    LD,s3,C=R.lab();rt=LD.Runtime();enc,w,tf=R.build(ns,sd,arms=['NAT']);assert w['p']==1 and w['n_chunks']==128;S=R.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids)
    L=C.Ladder55(rt,save_dir=OUT/'states'/f'E55_lmasks_s{sd}',canary=None);L.rec_cache={};t0=time.monotonic();lg=lambda *x:print(*x,flush=True)
    hid,wl,cnt,per=compute(L,rt,s3,ends,ids,model,a.frac,chunks_meta=w['chunks'],nch=128,log=lg);SM.save(fn,hid,wl)
    import e55_sentdump_labels as LB
    tiers={'8-31':range(8,32),'32-63':range(32,64),'64-127':range(64,128),'8-127':range(8,128)};name=f'LEARNED head weighting (frozen seed-61 coefficients {MODEL_KEY}, top {a.frac:.2f} of the window tokens)'
    Rp=[f'# E55 32.13 learned masks, label statistics (seed {sd}; LABELS WERE USED TO TRAIN THE COEFFICIENTS, on seed 61 only)','','| chunks | F1 | accuracy | recall | precision | hidden | bg false | chance |','|---|---|---|---|---|---|---|---|'];js={}
    for tn,rg in tiers.items():
        ks=[k for k in rg if k in per]
        if not ks:continue
        tot=np.zeros(6)
        for k in ks:
            d_={'w_len':wl[k]};lb=LB.labels(w['chunks'],ends,k);tot+=G.counts(hid[k][1],d_,lb)
        m=G.metrics(tot);js[tn]=m;Rp.append(f"| {tn} | {m['f1']:.3f} | {m['acc']:.3f} | {m['recall']:.3f} | {m['precision']:.3f} | {m['hidden']:.3f} | {m['bg_false']:.3f} | {m['chance']:.3f} |")
    (OUT/f'e55_lmasks_report_s{sd}.md').write_text('\n'.join(Rp)+'\n')
    json.dump({'seed':sd,'cell':'v3_16_1','stream_sha':sha,'candidate':name,'frac':a.frac,'coef':str(a.coef),'coef_sha12':hashlib.sha256(open(a.coef,'rb').read()).hexdigest()[:12],'metrics':js,'sec_total':time.monotonic()-t0,'npz_bytes':fn.stat().st_size},open(OUT/f'e55_lmasks_s{sd}.json','w'),default=float);print('\n'.join(Rp));print('wrote',fn,fn.stat().st_size,'bytes')
def selftest():
    cd={};labs={}
    for k in (1,2,3,4):cd[k],labs[k]=X.synth(k)
    fe={k:X.features(d,'sum','all') for k,d in cd.items()};Xa=np.concatenate([fe[k][0] for k in (1,2,3)]);y=np.concatenate([(np.asarray(labs[k]['w_fact'])[fe[k][1]]>=0).astype(float) for k in (1,2,3)]);m=X.fit_lr(Xa,y,10.0)
    d=cd[4];h=learned_hidden(d,m,0.25);wl=np.asarray(d['w_len']);fact=np.asarray(labs[4]['w_fact'])>=0
    assert 0.2<=wl[h].sum()/wl.sum()<=0.35 and wl[h&fact].sum()/wl[fact].sum()>0.9,(wl[h].sum()/wl.sum(),wl[h&fact].sum()/wl[fact].sum())
    hid,wls,cnt,per=compute(None,None,None,None,np.zeros(4096,np.int64),m,0.25,nch=4,k0=1,passfn=lambda k,ids:cd[k])
    assert set(hid)=={1,2,3} and all(set(v)=={1} for v in hid.values()) and (np.asarray(hid[2][1]).sum()>0)
    print('e55_lmask_run selftest ok')
if __name__=='__main__':main()
