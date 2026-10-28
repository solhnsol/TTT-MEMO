"""E57 closure masks (CPU only; job 289 input; post-hoc, exploratory). Builds the mask file e57_closure_masks.npz that e57_closure_run.py (GPU, W0, forward only) applies to every chunk k = 8..279 of the one-cycle news stream (full window).
Every mask is a PER-QUERY window keep, expressed by two bit-packed arrays per mask m and chunk k:
  kw_<m>_<k>  uint8 [1024]  = packbits of bool [WIN=8192]: window KEYS hidden from EVERY query of the chunk (key-side mask)
  qb_<m>_<k>  uint8 [128]   = packbits of bool [CH=1024]: query ROWS that see NO window key at all (query-side mask: window-less attention for those rows, in the prefix AND suffix layers); chunk keys always stay visible
The keep of the runner is keep[i, :WIN] = ~kw & ~qb[i], keep[i, WIN:] = natural (causal). One masked forward per mask and chunk serves key-side, query-side and combined masks (build_keep).
Row i of a chunk is the attention row whose output predicts token i+1 (tgt[i] == q_ids[i+1]): query-side masks are defined on ROWS i, labels on the predicted TOKEN i+1.
Masks (names): ORCA_K (oracle sentences of window, rank 5 of e57_pod_masks.npz), C1_K (candidate 1, rank 1), RC1_K (same-amount random of C1, rank 3), RORCA_K (rank 6); ORCA_Q (rows predicting ORCA tokens of the chunk), GAIN_Q_M (top rows by window gain of job 287a, count = round(hidden window fraction of ORCA_K x 1024) i.e. the same number of hidden (query x key) pairs as ORCA_K), GAIN_Q_p5/10/15/25 (top p % rows), RAND_Q_M (random rows, same count as GAIN_Q_M), RAND_Q_p10, RAND_Q_p25, RAND_Q_ORCA (same count as ORCA_Q). Extra masks (e.g. concentration based after job 288) are added with --extra a file of the same key format. Chunk-part masks of the pod run (hidden CHUNK sentences) are NOT used: the closure is about window access.
amount[m, k] = hidden (query x key) pairs / (1024 x 8192), saved as 'amt' [n_masks, 280]; the closure per token = (nll_mask - nll_win) / (nll_chunk - nll_win) is computed by the analysis after the run (not here).
Determinism: the builder is deterministic (RNG seed 289, fixed order) for the key-side and the gain / ORCA masks; only RAND_Q_* depend on numpy's Generator stream. content_hashes(file) = (det, all): sha256 over the sorted keys, dtypes, shapes and bytes of the arrays (the .npz container itself is NOT hashed: zip time stamps differ), det excludes the RAND_Q_* masks. The builder writes <out>.contenthash.json; --verify FILE [--expect-det H] prints the hashes and exits 9 on a mismatch of det.
  python3 e57_closure_masks.py --bundle B --pod-masks e57_pod_masks.npz --gain E57_windowless_W0.npz --out e57_closure_masks.npz [--extra X.npz] | --verify FILE [--expect-det H] | --selftest"""
import sys,json,argparse,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8;WIN=NW*CH;K0=8;K1=280
KEY_MASKS={'ORCA_K':5,'C1_K':1,'RC1_K':3,'RORCA_K':6}
def content_hashes(path):
    Z=np.load(path);d=hashlib.sha256();a=hashlib.sha256()
    for k in sorted(Z.files):
        v=np.ascontiguousarray(Z[k]);h=f'{k}|{v.dtype}|{v.shape}|'.encode()+v.tobytes();a.update(h)
        if '_RAND_Q_' not in k and not (k=='amt' or k=='names'):d.update(h)
    return d.hexdigest(),a.hexdigest()
def pk(b):return np.packbits(np.asarray(b,bool))
def up(a,n):return np.unpackbits(a)[:n].astype(bool)
def build_keep(keep1d,kw,qb):
    """-> bool [CH, WIN+CH]: natural 1-D keep of the chunk, window keys kw hidden for all queries, rows qb without any window key"""
    kp=np.broadcast_to(np.asarray(keep1d,bool),(CH,WIN+CH)).copy();kp[:,:WIN]&=~np.asarray(kw,bool)[None,:];kp[np.asarray(qb,bool),:WIN]=False;return kp
def band_keep(keep1d,k):
    """-> bool [CH, WIN+CH]: each query row i sees only the previous k tokens (itself and the k-1 before it, window AND chunk keys): key column c is at distance i + WIN - c; allowed 0 <= distance < k, intersected with the natural 1-D keep"""
    d=np.arange(CH)[:,None]+WIN-np.arange(WIN+CH)[None,:];return np.broadcast_to(np.asarray(keep1d,bool),(CH,WIN+CH))&(d>=0)&(d<int(k))
def rows_eligible(lm,sid):
    """rows i (0..1022) with a loss and whose predicted token i+1 lies inside a sentence"""
    ok=np.zeros(CH,bool);ok[:CH-1]=np.asarray(lm,bool)[:CH-1]&(np.asarray(sid)[1:]>=0);return ok
def top_rows(gain,elig,n):
    m=np.zeros(CH,bool);ix=np.nonzero(elig)[0];n=int(min(n,len(ix)))
    if n>0:m[ix[np.argsort(-gain[ix],kind='stable')[:n]]]=True
    return m
def rand_rows(elig,n,rng):
    m=np.zeros(CH,bool);ix=np.nonzero(elig)[0];n=int(min(n,len(ix)))
    if n>0:m[rng.choice(ix,n,replace=False)]=True
    return m
def chunk_masks(k,hk_by_name,orca_tok,gain,lm,sid,rng):
    """hk_by_name {name: bool [WIN]} key-side masks of the chunk; orca_tok bool [CH] (token level); gain [CH] float (row i); -> {name: (kw bool [WIN], qb bool [CH])}"""
    out={n:(h,np.zeros(CH,bool)) for n,h in hk_by_name.items()};elig=rows_eligible(lm,sid)
    nM=int(round(hk_by_name['ORCA_K'].sum()/WIN*CH));orcq=np.zeros(CH,bool);orcq[:CH-1]=np.asarray(orca_tok,bool)[1:]&elig[:CH-1]
    z=np.zeros(WIN,bool)
    out['ORCA_Q']=(z,orcq);out['GAIN_Q_M']=(z,top_rows(gain,elig,nM));out['RAND_Q_M']=(z,rand_rows(elig,nM,rng));out['RAND_Q_ORCA']=(z,rand_rows(elig,int(orcq.sum()),rng))
    for p in (5,10,15,25):out[f'GAIN_Q_p{p}']=(z,top_rows(gain,elig,int(round(elig.sum()*p/100))))
    for p in (10,25):out[f'RAND_Q_p{p}']=(z,rand_rows(elig,int(round(elig.sum()*p/100)),rng))
    return out
def amount(kw,qb):return (int(kw.sum())*CH+int(qb.sum())*WIN-int(kw.sum())*int(qb.sum()))/(CH*WIN)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--pod-masks');ap.add_argument('--gain');ap.add_argument('--out');ap.add_argument('--extra');ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');ap.add_argument('--verify');ap.add_argument('--expect-det');a=ap.parse_args()
    if a.verify:
        det,al=content_hashes(a.verify);print(f'content sha256 det={det} all={al}',flush=True)
        if a.expect_det and a.expect_det!=det:print(f'MISMATCH: expected det {a.expect_det}',flush=True);sys.exit(9)
        return
    if a.selftest:selftest();return
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_mask_stage as MS
    from tokenizers import Tokenizer
    ROOT=HERE.parents[1];tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);tk=Tokenizer.from_file(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids)
    P=np.load(a.pod_masks);Gz=np.load(a.gain);nw,nc,lm=Gz['nll_win'],Gz['nll_chunk'],Gz['lm'];memo={};rng=np.random.default_rng(289);buf={};names=None;amt={}
    X=np.load(a.extra) if a.extra else None
    for k in range(K0,K1):
        wsid,wst,wlen=UVm.window_sents(ends,k,CH,NW);q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        hk={}
        for nm,r in KEY_MASKS.items():
            assert np.array_equal(P[f'wlen{k}'],wlen),('pod masks do not match this stream at chunk',k)
            hw=np.asarray(P[f'h{r}_{k}'],bool);hk[nm]=(wsid>=0)&hw[np.clip(wsid,0,max(len(wlen)-1,0))]
        _,hc=MS.oracle_hidden(k,ends,ids,tk,items,UVm,memo);orc=np.zeros(CH,bool)
        for i,(s_,e_) in enumerate(q):orc[s_:e_]=hc[i]
        M=chunk_masks(k,hk,orc,nc[k]-nw[k],lm[k],sid,rng)
        if X is not None:
            for key in X.files:
                if key.startswith('kw_') and key.endswith(f'_{k}'):nm=key[3:-len(str(k))-1];M[nm]=(up(X[key],WIN),up(X[f'qb_{nm}_{k}'],CH))
        if names is None:names=sorted(M)
        assert sorted(M)==names,(k,sorted(M),names)
        for nm,(kw,qb) in M.items():buf[f'kw_{nm}_{k}']=pk(kw);buf[f'qb_{nm}_{k}']=pk(qb);amt.setdefault(nm,np.zeros(K1))[k]=amount(kw,qb)
    np.savez_compressed(a.out,names=np.asarray(names),amt=np.stack([amt[n] for n in names]),chunks=np.asarray([K0,K1]),**buf)
    det,al=content_hashes(a.out);pathlib.Path(str(a.out)+'.contenthash.json').write_text(json.dumps(dict(det=det,all=al,names=names)));print(f'content sha256 det={det} all={al}')
    print('masks',names);print('mean hidden-pair amount per mask:',{n:round(float(amt[n][K0:K1].mean()),4) for n in names})
def selftest():
    rng=np.random.default_rng(0);lm=np.ones(CH,bool);sid=np.zeros(CH,int);sid[:5]=-1
    hk={n:np.zeros(WIN,bool) for n in KEY_MASKS};hk['ORCA_K'][:2048]=True;hk['C1_K'][100:900]=True
    orc=np.zeros(CH,bool);orc[200:260]=True;gain=rng.standard_normal(CH)
    M=chunk_masks(9,hk,orc,gain,lm,sid,rng);e=rows_eligible(lm,sid)
    assert amount(*M['ORCA_K'])==0.25 and M['GAIN_Q_M'][1].sum()==256 and abs(amount(*M['GAIN_Q_M'])-0.25)<1e-9 and M['RAND_Q_M'][1].sum()==256
    assert M['ORCA_Q'][1][199:259].all() and M['ORCA_Q'][1].sum()==60 and M['RAND_Q_ORCA'][1].sum()==60 and not M['GAIN_Q_M'][1][~e].any()
    g=M['GAIN_Q_p10'][1];assert g.sum()==int(round(e.sum()*.1)) and gain[g].min()>=np.sort(gain[e])[-g.sum()]-1e-12
    kp=build_keep(np.ones(WIN+CH,bool),M['ORCA_K'][0],M['GAIN_Q_p5'][1]);qb=M['GAIN_Q_p5'][1]
    assert kp.shape==(CH,WIN+CH) and not kp[~qb][:,:2048].any() and kp[~qb][:,2048:WIN].all() and not kp[qb][:,:WIN].any() and kp[:,WIN:].all()
    assert np.array_equal(up(pk(M['ORCA_K'][0]),WIN),M['ORCA_K'][0]) and np.array_equal(up(pk(qb),CH),qb)
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        np.savez_compressed(pathlib.Path(t)/'a.npz',names=np.asarray(['X']),amt=np.zeros((1,3)),kw_X_9=pk(M['ORCA_K'][0]),kw_RAND_Q_M_9=pk(M['ORCA_K'][0]));np.savez(pathlib.Path(t)/'b.npz',names=np.asarray(['X']),amt=np.zeros((1,3)),kw_X_9=pk(M['ORCA_K'][0]),kw_RAND_Q_M_9=pk(M['RAND_Q_M'][0]))
        da,aa=content_hashes(pathlib.Path(t)/'a.npz');db,ab=content_hashes(pathlib.Path(t)/'b.npz');assert da==db and aa!=ab,'det hash ignores RAND_Q_* and the container'
    nat=np.ones(WIN+CH,bool);b=band_keep(nat,128)
    assert b.sum(1)[0]==128 and b.sum(1)[500]==128 and b.sum(1)[1023]==128 and b[0,WIN] and b[0,WIN-127] and not b[0,WIN-128] and not b[5,WIN+6] and b[700,WIN+700-127] and not b[700,WIN+700-128]
    b1=band_keep(nat,1024);assert b1[0,WIN-1023] and not b1[0,WIN-1024] and b1[1023,WIN] and not b1[1023,WIN-1] and b1.sum(1)[0]==1024
    print('e57_closure_masks selftest ok')
if __name__=='__main__':main()
