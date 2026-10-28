"""E57 key-mass dump v3 runner (GPU forward only, W0, no write; post-hoc, exploratory). For every write chunk k = 8 .. 279 of the one-cycle news stream (280 chunks, full window; stream and sentence layout of job 278 / 288) the arrays of e57_attdump3 plus q_start, q_len, q_ids (to align with the dumps of job 278 / 288). Shards of 20 chunks <out>/e57_attdump3_s<ss>.npz (uncompressed, keys 'W0/c<k>/<name>'), manifest <out>/e57_attdump3_manifest.json, shards read-only.
  python3 e57_attdump3_run.py --make-heads --dump2 D2 --out sharp_heads.npz        (CPU: sharp head set of every chunk from the job-288 dump, [280,24,32] bool)
  python3 e57_attdump3_run.py --bundle B --out O --sharp-heads sharp_heads.npz [--check-only]     (GPU)
  python3 e57_attdump3_run.py --check-288 --out O --dump2 D2 [--chunks 8,24,..]   (CPU: cross-check against the 288 dump)  |  --selftest
Expected size about 3.8 GiB without the dense validation chunks, 4.4 GiB with them."""
import sys,json,time,hashlib,pathlib,argparse,os
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;SHARD=20;STATE='W0';K0=8
def shard_file(out,s):return pathlib.Path(out)/f'e57_attdump3_s{s:02d}.npz'
def load_chunk(out,k,state=STATE):
    Z=np.load(shard_file(out,k//SHARD));pre=f'{state}/c{k}/';return {n[len(pre):]:Z[n] for n in Z.files if n.startswith(pre)}
def make_heads(dump2,out):
    import e57_attdump2_run as A2
    sh=np.zeros((280,24,32),bool)
    for k in range(1,280):
        d=A2.load_chunk(dump2,k);Tm=d['T_max'].astype(np.float32);ok=np.zeros(CH,bool)
        for a,n in zip(d['q_start'],d['q_len']):ok[a:a+n]=True
        m=Tm[:,:,ok].mean(-1);g=np.zeros(m.shape,bool);g[16:]=True;ii=np.argwhere(g);top=ii[np.argsort(-m[g],kind='stable')[:len(ii)//2]];sh[k][top[:,0],top[:,1]]=True
    np.savez_compressed(out,sharp=sh);return hashlib.sha256(sh.tobytes()).hexdigest()
def run(out,ids,nch,passfn,qlayout,log=print,k0=K0):
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);t0=time.monotonic();chunks={};shards={}
    for s in range(k0//SHARD,(nch-1)//SHARD+1):
        ks=[k for k in range(max(k0,s*SHARD),min(nch,(s+1)*SHARD))];f=shard_file(out,s)
        if f.exists() and f.stat().st_size>0 and (out/f'.done3_s{s:02d}').exists():
            for k in ks:chunks[k]=s
            shards[s]=dict(file=f.name,bytes=f.stat().st_size,sha256=(out/f'.done3_s{s:02d}').read_text().strip());continue
        buf={}
        for k in ks:
            ids_k=np.asarray(ids[k*CH:(k+1)*CH]);d=passfn(k,ids_k);qs,ql=qlayout(k)
            for n_,v in d.items():buf[f'{STATE}/c{k}/{n_}']=v
            buf[f'{STATE}/c{k}/q_start']=np.asarray(qs,np.int32);buf[f'{STATE}/c{k}/q_len']=np.asarray(ql,np.int32);buf[f'{STATE}/c{k}/q_ids']=ids_k.astype(np.int32);chunks[k]=s
            if k%8==0:log(f'  key-mass dump chunk {k}: {time.monotonic()-t0:.0f}s')
        np.savez(f,**buf);h=hashlib.sha256(open(f,'rb').read()).hexdigest();(out/f'.done3_s{s:02d}').write_text(h);shards[s]=dict(file=f.name,bytes=f.stat().st_size,sha256=h);del buf
    man=dict(state=STATE,chunk_to_shard={str(k):v for k,v in chunks.items()},shards={str(s):v for s,v in shards.items()},shard_size=SHARD,first_chunk=min(chunks),last_chunk=max(chunks),total_bytes=sum(v['bytes'] for v in shards.values()),sec_total=time.monotonic()-t0,layout='see module docstring of e57_attdump3.py')
    (out/'e57_attdump3_manifest.json').write_text(json.dumps(man,indent=1))
    for s in shards:os.chmod(out/shards[s]['file'],0o444)
    os.chmod(out/'e57_attdump3_manifest.json',0o444);return man
def check_288(out,dump2,ks):
    """CPU cross-check against the job-288 dump: the sum over keys of the SH / ALL mass equals the sum over sharp / all heads of T_sum (mean over the 288 rows); the top-1 vote counts of the SH far keys are consistent with T_key"""
    import e57_attdump2_run as A2
    sharp=np.load(os.environ.get('SHARP_HEADS',''))['sharp'] if os.environ.get('SHARP_HEADS') else None
    for k in ks:
        a=load_chunk(out,k);b=A2.load_chunk(dump2,k);assert np.array_equal(a['q_ids'],b['q_ids']),f'chunk {k}: token ids differ'
        ts=b['T_sum'].astype(np.float64)       # [24,32,CH]
        ref_all=ts.sum((0,1));err=np.abs(a['win_ALL']-ref_all).max()/max(ref_all.max(),1e-9)
        msg=f'chunk {k}: max rel |win_ALL - sum T_sum| = {err:.4f}'
        if sharp is not None:
            ref_sh=(ts*sharp[k][:,:,None]).sum((0,1));e2=np.abs(a['win_SH']-ref_sh).max()/max(ref_sh.max(),1e-9);msg+=f', win_SH {e2:.4f}';assert e2<5e-3
        print(msg,flush=True);assert err<5e-3,(k,err)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--tokenizer');ap.add_argument('--sharp-heads');ap.add_argument('--dump2');ap.add_argument('--check-only',action='store_true');ap.add_argument('--selftest',action='store_true');ap.add_argument('--check-288',action='store_true');ap.add_argument('--make-heads',action='store_true');ap.add_argument('--chunks',default='8,24,56,120,248,279');a=ap.parse_args()
    if a.selftest:selftest();return
    ROOT=HERE.parents[1]
    if a.make_heads:print('sharp heads sha256 (bool array bytes):',make_heads(a.dump2,a.out));return
    if a.check_288:check_288(a.out,a.dump2 or str(ROOT/'experiments/exp07_write_ladder/e57_attdump2'),[int(x) for x in a.chunks.split(',')]);return
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH
    ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids);out=pathlib.Path(a.out);SH=np.load(a.sharp_heads)['sharp'];assert SH.shape==(280,24,32)
    def qlayout(k):q=UVm.chunk_sents(ends,k,CH);return [a_ for a_,_ in q],[b_-a_ for a_,b_ in q]
    est=(n-K0)*(14.0*2**20)+((n-K0)//16+1)*2*16*2**20
    print(f'stream {n} chunks; chunks {K0}..{n-1}; about 14 MB per chunk + 32 MB dense every 16th chunk, total about {est/2**30:.2f} GiB; shards of {SHARD} chunks; output {out}',flush=True)
    if a.check_only:print('key-mass dump check-only OK');return
    import e55_run as R,e57_attdump3 as AD;LD,s3,C=R.lab();rt=LD.Runtime();L=C.Ladder55(rt,save_dir=out/'states'/'E57_attdump3',canary=None);lg=lambda *x:print(*x,flush=True)
    with rt.mesh:Kh,Vh,_=L.record(np.asarray(ids)[:n*CH])
    AD.install(s3)
    def passfn(k,ids_k):
        with rt.mesh:Kc,Vc=L.ctx_kv(Kh,Vh,k)
        return AD.att6_pass(L,rt,s3,rt.base,Kc,Vc,ids_k,k,SH[k],dense=(k%16==0))
    man=run(out,ids,n,passfn,qlayout,log=lg);lg(f'wrote {len(man["shards"])} shards, {man["total_bytes"]/2**30:.2f} GiB, read-only')
def selftest():
    import jax,jax.numpy as jnp,e57_attdump3 as AD,tempfile
    cfg=dict(WIN=128,N=16,M=40,MV=4,MG=4,KS=(8,24,40),RK_ALL=(1,2,4),RK_SH=(1,2,4,8),L=3,H=4);c=dict(AD.CFG,**cfg);W,N,H,L=c['WIN'],c['N'],c['H'],c['L']
    rng=np.random.default_rng(0);D=AD.Dump3(cfg);PHs=[];VNs=[];SHs=[]
    for l in range(L):
        sc=rng.standard_normal((H,N,W+N)).astype(np.float32)*2.5;causal=np.arange(W+N)[None,:]<=(np.arange(N)[:,None]+W);sc=np.where(causal[None],sc,-1e30)
        ph=np.exp(sc-sc.max(-1,keepdims=True));ph/=ph.sum(-1,keepdims=True);vn=(rng.random((H,W))+.5).astype(np.float32);sh=rng.random(H)<.5;sh[0]=True
        PHs.append(ph);VNs.append(vn);SHs.append(sh);D.layer(jnp.asarray(ph),jnp.asarray(vn),sh,l)
    out=D.finalize(dense=True)
    PH=np.stack(PHs);PW=PH[:,:,:,:W];SHm=np.stack(SHs)           # [L,H,N,W], [L,H]
    shf=SHm[:,:,None,None].astype(np.float32);massSH=(PW*shf).sum((0,1));massALL=PW.sum((0,1));VN=np.stack(VNs)
    massSHV=(PW*shf*VN[:,:,None,:]).sum((0,1));massALLV=(PW*VN[:,:,None,:]).sum((0,1));massG3=PW[2].sum(0)
    assert np.allclose(out['win_SH'],massSH.sum(-1),rtol=1e-4) and np.allclose(out['win_ALL'],massALL.sum(-1),rtol=1e-4)
    assert np.allclose(out['dense_SH'].astype(np.float32),massSH,rtol=2e-3,atol=1e-4) and np.allclose(out['dense_ALL'].astype(np.float32),massALL,rtol=2e-3,atol=1e-4)
    i_=np.arange(N)[:,None];c_=np.arange(W)[None,:]
    for k in c['KS']:
        far=((i_+W-c_)>=k)&(c_>=4)
        for nm,mass,rks in (('SH',massSH,c['RK_SH']),('ALL',massALL,c['RK_ALL'])):
            mf=np.where(far,mass,0);tot=mf.sum(-1);o=np.argsort(-mf,axis=1)[:,:c['M']];v=np.take_along_axis(mf,o,1)
            assert np.allclose(out[f'{nm}{k}_far'],tot,rtol=1e-4) and np.array_equal(out[f'{nm}{k}_idx'].astype(int),o) and np.allclose(out[f'{nm}{k}_mass'].astype(np.float32),v,rtol=2e-3)
            p=mf/np.maximum(tot[:,None],1e-30);assert np.allclose(out[f'{nm}{k}_ent'],-(p*np.log(p+1e-30)).sum(-1),rtol=1e-3,atol=1e-4)
            cum=np.cumsum(v,1)/tot[:,None];assert np.allclose(out[f'{nm}{k}_cum'].astype(np.float32),cum[:,[0,3,7,15,31]],rtol=2e-3)
            # counts: rank cutoffs among the far keys of each head, then edge
            cnt=np.zeros((N,c['M'],len(rks)+len(c['EDGE'])),np.int64)
            for l in range(L):
                for h in range(H):
                    if nm=='SH' and not SHm[l,h]:continue
                    pf=np.where(far,PW[l,h],-1.0);rk=np.argsort(-pf,axis=1)
                    for j,r in enumerate(rks):
                        top=rk[:,:r];valid=np.take_along_axis(pf,top,1)>0
                        for row in range(N):
                            for key,vd in zip(top[row],valid[row]):
                                if vd:
                                    m=np.nonzero(o[row]==key)[0]
                                    if len(m):cnt[row,m[0],j]+=1
                    for j,t in enumerate(c['EDGE']):
                        cnt[:,:,len(rks)+j]+=(np.take_along_axis(PW[l,h],o,1)>=t)
            assert np.array_equal(out[f'{nm}{k}_cnt'].astype(np.int64),np.minimum(cnt,255)),(nm,k)
            if nm=='ALL':
                for g in range(3):assert np.allclose(out[f'ALL{k}_grp'][:,:,g].astype(np.float32),np.take_along_axis(PW[g].sum(0)*1.0,o[:,:c['MG']],1),rtol=2e-3)
        for nm,mass in (('SHV',massSHV),('ALLV',massALLV),('G3',massG3)):
            mf=np.where(far,mass,0);o=np.argsort(-mf,axis=1)[:,:c['MV']];assert np.array_equal(out[f'{nm}{k}_idx'].astype(int),o) and np.allclose(out[f'{nm}{k}_mass'].astype(np.float32),np.take_along_axis(mf,o,1),rtol=2e-3)
    PCm=PH[:,:,:,W:];cm=(PCm*SHm[:,:,None,None]).sum((0,1));dd=np.arange(N)[:,None]-np.arange(N)[None,:];edges=AD.BINS
    for b in range(5):
        msk=(dd>=0)&(dd<=edges[b+1])&((dd>edges[b]) if b>0 else True);assert np.allclose(out['chunk_SH'][:,b].astype(np.float32),(cm*msk).sum(-1),rtol=2e-3,atol=1e-4)
    assert out['headscore'].shape==(L,H,5)
    # run(): shards / resume / read-only on this tiny output
    with tempfile.TemporaryDirectory() as t:
        ids=np.arange(40*CH,dtype=np.int64);man=run(t,ids,40,lambda k,ids_k:{'x':np.ones(3,np.float16)*k},lambda k:([0],[CH]),log=lambda *x:None,k0=8)
        import stat;z=load_chunk(t,21);assert z['x'][0]==21 and man['first_chunk']==8 and man['last_chunk']==39
        assert (stat.S_IMODE(os.stat(pathlib.Path(t)/man['shards']['0']['file']).st_mode)&0o222)==0
    print('e57_attdump3_run selftest ok')
if __name__=='__main__':main()
