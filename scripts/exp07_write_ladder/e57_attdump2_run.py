"""E57 token-level attention dump v2 runner (GPU forward only, W0, no write; post-hoc, exploratory). For every write chunk k = 1 .. n-1 of the one-cycle news stream (280 chunks, the stream and layout of job 278) the arrays of e57_attdump2 (T_max, T_key, T_sum, T_ent, T_sink, T_rec, T_cur, POS, VN) plus q_start, q_len, q_ids (to align with the job-278 dump). Shards of 40 chunks <out>/e57_attdump2_s<ss>.npz (uncompressed, keys 'W0/c<k>/<name>'), manifest <out>/e57_attdump2_manifest.json, shards read-only.
  python3 e57_attdump2_run.py --bundle B --out O [--check-only] | --selftest | --check-278 --out O [--chunks 1,17,33]  (CPU: cross-check against the job-278 dump)
Expected size: per chunk about 24*32*1024*(7 + 9) float16 = 25 MB + VN 1.5 MB = about 27 MB, about 7.4 GB in total (printed by --check-only)."""
import sys,json,time,hashlib,pathlib,argparse,bisect,os
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8;SHARD=40;STATE='W0'
def shard_file(out,s):return pathlib.Path(out)/f'e57_attdump2_s{s:02d}.npz'
def load_chunk(out,k,state=STATE):
    Z=np.load(shard_file(out,k//SHARD));pre=f'{state}/c{k}/';return {n[len(pre):]:Z[n] for n in Z.files if n.startswith(pre)}
def run(out,ids,nch,passfn,qlayout,log=print,k0=1):
    """passfn(k, ids_k) -> dict of arrays; qlayout(k) -> (q_start, q_len)"""
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);t0=time.monotonic();chunks={};shards={}
    for s in range(k0//SHARD,(nch-1)//SHARD+1):
        ks=[k for k in range(max(k0,s*SHARD),min(nch,(s+1)*SHARD))];f=shard_file(out,s)
        if f.exists() and f.stat().st_size>0 and (out/f'.done2_s{s:02d}').exists():
            for k in ks:chunks[k]=s
            shards[s]=dict(file=f.name,bytes=f.stat().st_size,sha256=(out/f'.done2_s{s:02d}').read_text().strip());continue          # resume: a finished shard is kept
        buf={}
        for k in ks:
            ids_k=np.asarray(ids[k*CH:(k+1)*CH]);d=passfn(k,ids_k);qs,ql=qlayout(k)
            for n_,v in d.items():buf[f'{STATE}/c{k}/{n_}']=v
            buf[f'{STATE}/c{k}/q_start']=np.asarray(qs,np.int32);buf[f'{STATE}/c{k}/q_len']=np.asarray(ql,np.int32);buf[f'{STATE}/c{k}/q_ids']=ids_k.astype(np.int32);chunks[k]=s
            if k%16==0:log(f'  token dump chunk {k}: {time.monotonic()-t0:.0f}s')
        np.savez(f,**buf);h=hashlib.sha256(open(f,'rb').read()).hexdigest();(out/f'.done2_s{s:02d}').write_text(h);shards[s]=dict(file=f.name,bytes=f.stat().st_size,sha256=h);del buf
    man=dict(state=STATE,chunk_to_shard={str(k):v for k,v in chunks.items()},shards={str(s):v for s,v in shards.items()},shard_size=SHARD,first_chunk=min(chunks),last_chunk=max(chunks),total_bytes=sum(v['bytes'] for v in shards.values()),sec_total=time.monotonic()-t0,layout='see module docstring of e57_attdump2.py')
    (out/'e57_attdump2_manifest.json').write_text(json.dumps(man,indent=1))
    for s in shards:os.chmod(out/shards[s]['file'],0o444)
    os.chmod(out/'e57_attdump2_manifest.json',0o444);return man
def check_278(out,old,ks):
    """CPU cross-check: mean over the tokens of a query sentence of T_sum[h] must equal sum_w AL[h,s,w] of the job-278 dump (float16 tolerance); T_key / T_max consistent with POS bins"""
    import e57_attdump_run as AR
    for k in ks:
        a=load_chunk(out,k);b=AR.load_chunk(old,k);qs=b['q_start'];ql=b['q_len']
        assert np.array_equal(a['q_start'],qs) and np.array_equal(a['q_len'],ql) and np.array_equal(a['q_ids'],b['q_ids']),f'chunk {k}: sentence layout / token ids differ from the job-278 dump'
        al=b['AL'].astype(np.float32).sum(-1);ts=a['T_sum'].astype(np.float32)
        ref=np.stack([ts[:,:,s_:s_+n_].mean(-1) for s_,n_ in zip(qs,ql)],axis=-1)
        err=np.abs(ref-al).max();print(f'chunk {k}: max |mean T_sum - sum AL| = {err:.4f}  (T_sum mean {ts.mean():.3f})',flush=True);assert err<0.02,('T_sum does not match AL of job 278',k,err)
        pos=a['POS'].astype(np.float32).sum(-1);print(f'   POS bins sum vs T_sum max err {np.abs(pos-ts).max():.4f}; T_rec vs last 5 bins {np.abs(a["T_rec"].astype(np.float32)-a["POS"][...,4:].astype(np.float32).sum(-1)).max():.4f}')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--tokenizer');ap.add_argument('--check-only',action='store_true');ap.add_argument('--selftest',action='store_true');ap.add_argument('--check-278',action='store_true');ap.add_argument('--chunks',default='1,17,33,65,129,257');ap.add_argument('--old');a=ap.parse_args()
    if a.selftest:selftest();return
    ROOT=HERE.parents[1]
    if a.check_278:check_278(a.out,a.old or str(ROOT/'experiments/exp07_write_ladder/e57_attdump'),[int(x) for x in a.chunks.split(',')]);return
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH
    ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids);out=pathlib.Path(a.out)
    def qlayout(k):q=UVm.chunk_sents(ends,k,CH);return [a_ for a_,_ in q],[b_-a_ for a_,b_ in q]
    est=(n-1)*24*32*CH*(7+9)*2+(n-1)*24*32*CH*2
    print(f'stream {n} chunks; token dump per chunk about {24*32*CH*16*2/2**20:.1f} MB + VN {24*32*CH*2/2**20:.1f} MB, total about {est/2**30:.2f} GiB; shards of {SHARD} chunks; output {out}',flush=True)
    if a.check_only:print('token dump check-only OK');return
    import e55_run as R,e57_attdump2 as AD;LD,s3,C=R.lab();rt=LD.Runtime();L=C.Ladder55(rt,save_dir=out/'states'/'E57_attdump2',canary=None);lg=lambda *x:print(*x,flush=True)
    with rt.mesh:Kh,Vh,_=L.record(np.asarray(ids)[:n*CH])
    AD.install(s3)
    def passfn(k,ids_k):
        with rt.mesh:Kc,Vc=L.ctx_kv(Kh,Vh,k)
        return AD.att5_pass(L,rt,s3,rt.base,Kc,Vc,ids_k,k)
    man=run(out,ids,n,passfn,qlayout,log=lg);lg(f'wrote {len(man["shards"])} shards, {man["total_bytes"]/2**30:.2f} GiB, read-only')
def selftest():
    import tempfile,e57_attdump2 as AD
    H=3;n=40;rng=np.random.default_rng(0)
    # reduce5 vs a brute-force loop on a tiny softmax with the real WIN (window of 8192 keys)
    sc=rng.standard_normal((H,n,AD.WIN+n)).astype(np.float32)*3;sc[:,:,:AD.WIN//2]-=4
    ph=np.exp(sc-sc.max(-1,keepdims=True));ph/=ph.sum(-1,keepdims=True);vn=rng.random((H,1024)).astype(np.float32)
    r=AD.reduce5(ph,vn);pw=ph[:,:,:AD.WIN]
    for h in range(H):
        for q in range(0,n,7):
            row=pw[h,q];assert np.isclose(r['T_max'][h,q],row.max()) and r['T_key'][h,q]==row.argmax() and np.isclose(r['T_sum'][h,q],row.sum(),rtol=1e-5)
            assert np.isclose(r['T_ent'][h,q],-(row*np.log(row+1e-30)).sum(),rtol=1e-5) and np.isclose(r['T_sink'][h,q],row[:4].sum()) and np.isclose(r['T_rec'][h,q],row[-1024:].sum(),rtol=1e-5)
            assert np.isclose(r['T_cur'][h,q],ph[h,q,AD.WIN:].sum(),rtol=1e-5)
            e=AD.POS_EDGES
            for i in range(9):assert np.isclose(r['POS'][h,q,i],row[e[i]:e[i+1]].sum(),rtol=1e-5,atol=1e-9)
    assert np.allclose(r['POS'].sum(-1),r['T_sum'],rtol=1e-5) and np.allclose(r['POS'][...,4:].sum(-1),r['T_rec'],rtol=1e-4) and np.allclose(r['POS'][...,0],r['T_sink'])
    # run(): shards, resume, load_chunk, read-only
    N=5;ids=np.arange(N*CH,dtype=np.int64)
    def passfn(k,ids_k):
        d=AD.reduce5(ph,vn);return {k_:np.asarray(v_,np.float16 if k_ in AD.F16 else np.int16) for k_,v_ in d.items()}
    with tempfile.TemporaryDirectory() as t:
        man=run(t,ids,N,passfn,lambda k:([0,100],[100,924]),log=lambda *x:None);assert man['first_chunk']==1 and man['last_chunk']==N-1
        z=load_chunk(t,3);assert z['POS'].shape==(H,n,9) and z['q_ids'].shape==(CH,) and z['T_key'].dtype==np.int16 and z['VN'].shape==(H,1024)
        assert not os.access(pathlib.Path(t)/man['shards']['0']['file'],os.W_OK),'shards are read-only'
        vs={j:np.ones((H,1024),np.float16)*j for j in range(1,N)};w=AD.vn_window(vs,3);assert w.shape==(H,8192) and w[0,-1]==3.0 and w[0,0]==0.0
    print('e57_attdump2_run selftest ok')
if __name__=='__main__':main()
