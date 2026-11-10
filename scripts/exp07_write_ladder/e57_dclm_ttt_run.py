"""E57 DCLM preservation gate (GPU; new file; Council-fixed spec 11/10): does the masked update (NATMASK) hurt a general-text stream more than the original TTT (NATS = original NAT)? Online (prequential) TTT over DCLM validation text, nothing is trained on the facts.
Streams: DCLM validation docs 0..63 (data/dclm_val_subset/val_c0_200x8193.u32le.bin, 8193 tokens per doc), first 8192 tokens of each doc, joined in index order 16 at a time into 4 streams of 128 chunks (docs 0-15, 16-31, 32-47, 48-63); the selection is fixed. The fast weights run on through the stream (no reset at doc boundaries).
Arms (same engine, same hyper-parameters as e57_natmask_run.py; K0 = 8, ALLQ256 from chunk 8): W0 = fast weights frozen (update multiplier 0), NATS = original NAT (clipped SGD), NATMASK = ALLQ256 mask on the update path, NATMASKC = same number of hidden keys per row, random keys. (DIR arms later only if the first result calls for them.)
Metric: for every chunk k, BEFORE the write of the chunk, the chunk is scored with the current fast weights and the FULL natural local context (natural keep in prefix and suffix; make_nat_loss_diag of e57_natmask_run.py) -> mean NLL (nat/token) over the valid targets of the chunk = prequential NLL. NATS: this equals the update-path ce of the chunk (cross check).
Gate (analyze stage, CPU): mean over the 480 pairs (4 streams x chunks 8..127) of NLL(NATMASK) - NLL(NATS); chunk means are the primary metric; cluster bootstrap over the 4 streams for the CI (rough, 4 clusters), per-stream means, and the comparison against W0 frozen. First report = point estimate <= +0.01 nat with the CI upper bound shown.
  python3 e57_dclm_ttt_run.py --stage run --out O [--arms W0,NATS,NATMASK,NATMASKC] [--streams 0,1,2,3] [--smoke N] [--rerun] | --stage analyze --out O [--json J] | --selftest"""
import sys,os,json,time,gc,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_natmask_run as NM
import e57_qband_run as QR
CH=1024;NCH=128;DOCS=16;NSTREAM=4;K0=8;BIN=ROOT/'data/dclm_val_subset/val_c0_200x8193.u32le.bin';ARMS_ALL=['W0','NATS','NATMASK','NATMASKC']
def streams(path=BIN,n_streams=NSTREAM,docs=DOCS):
    """-> list of int64 id arrays (docs*8192 tokens each); stream s = docs s*16 .. s*16+15, first 8192 tokens of each doc"""
    a=np.fromfile(path,dtype='<u4');assert a.size%8193==0,a.size;a=a.reshape(-1,8193);assert a.shape[0]>=n_streams*docs
    return [a[s*docs:(s+1)*docs,:8192].reshape(-1).astype(np.int64) for s in range(n_streams)]
def arm_spec(arm):
    """-> (mask arm name for the uv object, update multiplier)"""
    return {'W0':('NATQ',0.0),'NATS':('NATQ',1.0),'NATMASK':('ALLQ256',1.0),'NATMASKC':('KEYC:ALLQ256',1.0)}[arm]
def run(a):
    import e55_run as R55
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);NM.register_engine();NM.K0=a.k0;QR.K0=a.k0
    LD,s3,C=R55.lab();rt=LD.Runtime();lg=lambda *x:print(*x,flush=True)
    S=streams();masks={'band_ALLQ256':np.asarray(256),'all_ALLQ256':np.asarray(1)}
    for si in [int(x) for x in a.streams.split(',')]:
        ids=S[si];nch=a.smoke or NCH;ids=ids[:nch*CH];assert len(ids)==nch*CH
        L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
        import hashlib
        del _h;rc={(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)};del L0;gc.collect()
        for arm in a.arms.split(','):
            fn=out/f'E57_dclmttt_{arm}_st{si}.json'
            if fn.exists() and not a.rerun:lg('exists, reused:',fn);continue
            ma,mult=arm_spec(arm);L=C.Ladder55(rt,save_dir=out/'states'/f'st{si}_{arm}',canary=None);L.rec_cache=rc
            uv=NM.KeyMatchedQ('C',256,si,lg) if ma.startswith('KEY') else QR.SelMaskQ(ma,masks,lg)
            hold={};uv=NM.with_base(uv,hold,None,None,full_every=0,always_full=False)
            L.write_step=NM.mult_step(L.write_step,mult);dfn=NM.make_nat_loss_diag(L,LD,s3,rt,hold,lambda *x:None)
            t0=time.monotonic();lg(f'== stream {si} {arm} (mask {ma}, lr x{mult:g}): {nch} chunks')
            r=L.run_stream(NM.ENGINE,1,ids,nch=nch,tag=f'dclm{si}{arm}',log=lambda *x:None,read_chunks=(),canary_chunks=None,chunks_meta=[{'sents':[]} for _ in range(nch)],uv=uv,diag_fn=dfn,diag_chunks=set(range(nch)));r.pop('model',None)
            nll=[float(r['diag'][k]['nat_nll_mean']) for k in range(nch)]
            rec=dict(arm=arm,stream=si,docs=[si*DOCS,si*DOCS+DOCS-1],n_chunks=nch,mask_arm=ma,lr_mult=mult,K0=(a.k0 if arm in ('NATMASK','NATMASKC') else None),nll_pre=nll,ce=[e['ce'] for e in r['log']],update_norm=[e['update_norm'] for e in r['log']],gnorm=[e['gnorm'] for e in r['log']],coef=[e['coef'] for e in r['log']],sec=time.monotonic()-t0,ids_sha=hashlib.sha256(ids.tobytes()).hexdigest())
            if not np.all(np.isfinite(nll)):rec['diverged']=True
            json.dump(rec,open(fn,'w'),default=float);lg('wrote',fn,f'{rec["sec"]:.0f}s mean nll chunks>=8: {np.mean(nll[K0:]):.4f}');del L,r;gc.collect();LD.jax.clear_caches()
        del rc,Kh0,Vh0;gc.collect()
def cluster_ci(d,B=4000,seed=0):
    """d [streams, chunks] differences -> (mean over all pairs, 95% CI by resampling STREAMS (clusters), 95% CI by resampling chunks over the pooled pairs)"""
    d=np.asarray(d,float);rng=np.random.default_rng(seed);m=d.mean();s,c=d.shape
    cl=np.array([d[rng.integers(0,s,s)].mean() for _ in range(B)]);flat=d.reshape(-1);ch=np.array([flat[rng.integers(0,len(flat),len(flat))].mean() for _ in range(B)])
    return float(m),[float(np.quantile(cl,.025)),float(np.quantile(cl,.975))],[float(np.quantile(ch,.025)),float(np.quantile(ch,.975))]
def analyze(a):
    out=pathlib.Path(a.out);ns=NSTREAM;N={}
    for arm in ARMS_ALL:
        fs=[out/f'E57_dclmttt_{arm}_st{s}.json' for s in range(ns)]
        if not all(f.exists() for f in fs) and getattr(a,'ref_dir',None):fs=[pathlib.Path(a.ref_dir)/f'E57_dclmttt_{arm}_st{s}.json' for s in range(ns)]      # W0 / NATS of the K0 = 8 run (K0 does not change them)
        if all(f.exists() for f in fs):
            recs=[json.load(open(f)) for f in fs];assert not any(r.get('diverged') for r in recs),('diverged',arm);N[arm]=np.array([r['nll_pre'] for r in recs])
    k0s={}
    for arm in ARMS_ALL:
        for s_ in range(ns):
            f=out/f'E57_dclmttt_{arm}_st{s_}.json'
            if not f.exists() and getattr(a,'ref_dir',None):f=pathlib.Path(a.ref_dir)/f'E57_dclmttt_{arm}_st{s_}.json'
            if f.exists() and arm in ('NATMASK','NATMASKC'):k0s.setdefault(arm,set()).add(json.load(open(f)).get('K0',8))      # records without the tag are the K0 = 8 job 1274
    allk=set().union(*k0s.values()) if k0s else set();assert len(allk)<=1,('mask arms with different K0: never pool across K0',k0s)
    res={'K0_mask_arms':(sorted(allk)[0] if allk else None),'run_type':getattr(a,'run_type',None),'n_pairs':int(ns*(NCH-K0)),'per_arm_mean_nll_chunks_8_127':{k:float(v[:,K0:].mean()) for k,v in N.items()}}
    def diff(x,y):return N[x][:,K0:]-N[y][:,K0:]
    for x,y in (('NATMASK','NATS'),('NATMASKC','NATS'),('NATS','W0'),('NATMASK','W0'),('NATMASKC','W0')):
        if x in N and y in N:
            m,ccl,cch=cluster_ci(diff(x,y));res[f'{x}-{y}']=dict(mean=m,ci95_stream_cluster=ccl,ci95_chunk=cch,per_stream=[float(v) for v in diff(x,y).mean(1)])
    if 'NATMASK-NATS' in res:res['gate']=dict(rule='NATMASK-NATS point estimate <= +0.01 nat/token (CI upper bound shown; 4 clusters make the CI rough)',point=res['NATMASK-NATS']['mean'],ci_upper_stream_cluster=res['NATMASK-NATS']['ci95_stream_cluster'][1],passed_point=bool(res['NATMASK-NATS']['mean']<=0.01))
    if 'NATS' in N:                 # cross check: for NATS the pre-write natural NLL equals the update-path ce
        r=[json.load(open(out/f'E57_dclmttt_NATS_st{s}.json')) for s in range(ns)];res['natS_nll_vs_ce_max_abs']=float(max(np.max(np.abs(np.array(x['nll_pre'])-np.array(x['ce']))) for x in r))
    print(json.dumps(res,indent=1))
    if a.json:json.dump(res,open(a.json,'w'),indent=1)
def selftest():
    import tempfile
    NM.K0=8;NM.selftest()
    with tempfile.TemporaryDirectory() as td:
        f=pathlib.Path(td)/'x.bin';rng=np.random.default_rng(0);arr=rng.integers(0,1000,size=(70,8193),dtype=np.uint32);arr.tofile(f);S=streams(f)
        assert len(S)==4 and all(len(s)==16*8192 for s in S) and np.array_equal(S[1][:8192],arr[16,:8192].astype(np.int64)) and np.array_equal(S[3][-8192:],arr[63,:8192].astype(np.int64))
    assert arm_spec('W0')==('NATQ',0.0) and arm_spec('NATMASKC')[0].startswith('KEY')
    ma,_=arm_spec('NATMASKC');assert ma.startswith('KEYC:')
    d=np.full((4,120),0.004)+np.random.default_rng(1).normal(0,0.001,(4,120));m,ccl,cch=cluster_ci(d);assert abs(m-0.004)<5e-4 and ccl[0]<=m<=ccl[1] and cch[0]<=m<=cch[1]
    print('e57_dclm_ttt_run selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='run');ap.add_argument('--out');ap.add_argument('--arms',default=','.join(ARMS_ALL));ap.add_argument('--streams',default='0,1,2,3');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--rerun',action='store_true');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');ap.add_argument('--k0',type=int,default=8);ap.add_argument('--run-type',default=None,choices=[None,'confirmation','k0_sensitivity','pilot_exploration','legacy_k08']);ap.add_argument('--ref-dir');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.stage=='run':run(a)
    else:analyze(a)
