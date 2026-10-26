"""E57 news attention dump (GPU forward only, no write step; post-hoc, exploratory): for every write chunk k = 1 .. n-1 of the one-cycle news stream (280 chunks) the attention of the frozen prefix layers (W0 weights, natural unmasked window, every token of the chunk a query), same layout and key names as the seed-61 synthetic dump (e55_sentdump_s61.npz: keys '<state>/c<chunk>/<name>', state = 'W0'):
  AL [24, 32, nQ, nW] float16   mass of query sentence s (token mean) on window sentence w (sum over its key tokens), per layer and head  (window columns only, as in the synthetic dump)
  AL_chunk [24, 32, nQ, nQ] float16   NEW (no synthetic counterpart): mass of query sentence s on EARLIER sentences c of the current chunk (zero for c >= s)
  q_start, q_len [nQ] int32 (chunk sentences: offset in the chunk, tokens), w_start, w_len, w_chunk [nW] int32 (window sentences: offset in the 8192-token window, tokens, absolute chunk number), win_ids [8192] int32, q_ids [1024] int32 (token ids of window and chunk)
  q_event [nQ], w_event [nW] int16   NEW: index into manifest['events'] of the article the sentence lies in (event axis); -1 none
Shards <out>/e57_attdump_s<ss>.npz hold 40 chunks each (np.savez, uncompressed; about 0.7 GB), <out>/e57_attdump_manifest.json lists events, shards, the chunk -> shard map, shapes and sha256 of every shard; the shards are made read-only after the last chunk. Reading: load_chunk(out, k) -> dict of the names above. CPU analysis of the learned head weighting: features(d) with d['AL'] and d['w_len'] exactly as in e55_sentgrid_ext.
  python3 e57_attdump_run.py --bundle B --out O [--check-only] | --selftest"""
import sys,json,time,hashlib,pathlib,argparse,bisect,os
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8;SHARD=40;STATE='W0'
def shard_file(out,s):return pathlib.Path(out)/f'e57_attdump_s{s:02d}.npz'
def to_layout(d,k,ev_of,T):
    """d of e57_attdump.assemble (AL with window + chunk columns) -> arrays of the synthetic-dump layout"""
    nW=len(d['w_len']);nQ=len(d['q_len']);AL=np.asarray(d['AL'],np.float16)
    w_abs=(k-NW)*CH+np.asarray(d['w_start'],np.int64);q_abs=k*CH+np.asarray(d['q_start'],np.int64)
    return {'AL':AL[...,:nW],'AL_chunk':AL[...,nW:],'q_start':np.asarray(d['q_start'],np.int32),'q_len':np.asarray(d['q_len'],np.int32),'w_start':np.asarray(d['w_start'],np.int32),'w_len':np.asarray(d['w_len'],np.int32),'w_chunk':np.asarray(d['w_chunk'],np.int32),
        'win_ids':np.asarray(d['win_ids'],np.int32),'q_ids':np.asarray(d['q_ids'],np.int32),'q_event':np.asarray([ev_of(int(p)%T) for p in q_abs],np.int16),'w_event':np.asarray([ev_of(int(p)%T) if p>=0 else -1 for p in w_abs],np.int16)}
def load_chunk(out,k,state=STATE):
    out=pathlib.Path(out);Z=np.load(shard_file(out,k//SHARD));pre=f'{state}/c{k}/';return {n[len(pre):]:Z[n] for n in Z.files if n.startswith(pre)}
def run(out,ids,nch,passfn,ev_of,events,log=print,k0=1):
    """passfn(k, ids_k) -> d of e57_attdump.assemble; writes the shards and the manifest"""
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);t0=time.monotonic();T=len(ids);chunks={};shards={}
    for s in range(k0//SHARD,(nch-1)//SHARD+1):
        ks=[k for k in range(max(k0,s*SHARD),min(nch,(s+1)*SHARD))]
        f=shard_file(out,s)
        if f.exists() and f.stat().st_size>0 and (out/f'.done_s{s:02d}').exists():
            for k in ks:chunks[k]=s
            shards[s]=dict(file=f.name,bytes=f.stat().st_size,sha256=(out/f'.done_s{s:02d}').read_text().strip());continue          # resume: a finished shard is kept
        buf={}
        for k in ks:
            d=passfn(k,np.asarray(ids[k*CH:(k+1)*CH]));L=to_layout(d,k,ev_of,T)
            for n,v in L.items():buf[f'{STATE}/c{k}/{n}']=v
            chunks[k]=s
            if k%16==0:log(f'  attention dump chunk {k}: {len(L["q_len"])} chunk sentences x {len(L["w_len"])} window sentences, AL {tuple(L["AL"].shape)}, {time.monotonic()-t0:.0f}s')
        np.savez(f,**buf);h=hashlib.sha256(open(f,'rb').read()).hexdigest();(out/f'.done_s{s:02d}').write_text(h);shards[s]=dict(file=f.name,bytes=f.stat().st_size,sha256=h);del buf
    man=dict(state=STATE,events=events,chunk_to_shard={str(k):v for k,v in chunks.items()},shards={str(s):v for s,v in shards.items()},shard_size=SHARD,first_chunk=min(chunks),last_chunk=max(chunks),total_bytes=sum(v['bytes'] for v in shards.values()),sec_total=time.monotonic()-t0,layout='see module docstring of e57_attdump_run.py')
    (out/'e57_attdump_manifest.json').write_text(json.dumps(man,indent=1))
    for s in shards:os.chmod(out/shards[s]['file'],0o444)
    os.chmod(out/'e57_attdump_manifest.json',0o444);return man
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--tokenizer');ap.add_argument('--check-only',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    ROOT=HERE.parents[1];import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH
    events=sorted({x[2] for x in iv});starts=[x[0] for x in iv];ev_idx={e:i for i,e in enumerate(events)};ev_of=lambda p:ev_idx[iv[max(0,bisect.bisect_right(starts,p)-1)][2]]
    ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids);out=pathlib.Path(a.out);nq=[len(UVm.chunk_sents(ends,k,CH)) for k in range(1,n)];nw=[len(UVm.window_sents(ends,k,CH,NW)[2]) for k in range(1,n)]
    est=sum(24*32*q*(w+q)*2 for q,w in zip(nq,nw))
    print(f'stream {n} chunks, {len(events)} events; chunk sentences per chunk {min(nq)}..{max(nq)}, window sentences {min(nw)}..{max(nw)}; AL float16 bytes in total about {est/2**30:.2f} GiB ({est/len(nq)/2**20:.1f} MB per chunk); shards of {SHARD} chunks; output {out}',flush=True)
    if a.check_only:print('attention dump check-only OK');return
    import e55_run as R,e57_attdump as AD;LD,s3,C=R.lab();rt=LD.Runtime();L=C.Ladder55(rt,save_dir=out/'states'/'E57_attdump',canary=None);L.rec_cache={};lg=lambda *x:print(*x,flush=True)
    with rt.mesh:Kh,Vh,_=L.record(np.asarray(ids)[:n*CH])
    AD.install(s3)
    def passfn(k,ids_k):
        with rt.mesh:Kc,Vc=L.ctx_kv(Kh,Vh,k)
        return AD.att4_pass(L,rt,s3,UVm,rt.base,Kc,Vc,ids_k,k,ends,ids)
    man=run(out,ids,n,passfn,ev_of,events,log=lg);lg(f'wrote {len(man["shards"])} shards, {man["total_bytes"]/2**30:.2f} GiB, read-only')
def selftest():
    import tempfile,e57_mask_stage as MS,e55_uv as UVm
    N=6;ids=np.arange(N*CH,dtype=np.int64);ends=np.zeros(N*CH,bool);ends[np.arange(N*CH)%200==199]=True;ends[np.arange(1,N+1)*CH-1]=True
    def passfn(k,ids_k):
        q=UVm.chunk_sents(ends,k,CH);wsid,wst,wl=UVm.window_sents(ends,k,CH,NW);d=MS.synth_dump(k,nQ=len(q),nW=len(wl),L=3,H=2);d['w_len']=np.asarray(wl,np.int32);d['w_start']=np.asarray(wst,np.int32);d['q_len']=d['c_len']=np.asarray([b-a for a,b in q],np.int32);d['q_start']=np.asarray([a for a,_ in q],np.int32);d['w_chunk']=(k-NW+d['w_start']//CH).astype(np.int32)
        d['AL']=np.random.default_rng(k).random((3,2,len(q),len(wl)+len(q))).astype(np.float16);return d
    ev_of=lambda p:0 if p<3*CH else 1
    with tempfile.TemporaryDirectory() as t:
        man=run(t,ids,N,passfn,ev_of,['e0','e1'],log=lambda *x:None);assert man['first_chunk']==1 and man['last_chunk']==N-1
        z=load_chunk(t,3);nW=len(z['w_len']);nQ=len(z['q_len']);assert z['AL'].dtype==np.float16 and z['AL'].shape[2:]==(nQ,nW) and z['AL_chunk'].shape[2:]==(nQ,nQ),(z['AL'].shape,z['AL_chunk'].shape)
        assert z['q_event'].shape==(nQ,) and z['w_event'].shape==(nW,) and set(z['q_event'].tolist())<={0,1} and z['win_ids'].shape==(8192,)
        full=passfn(3,None)['AL'];assert np.array_equal(np.concatenate([z['AL'],z['AL_chunk']],axis=3),full),'window + chunk columns reassemble the dump'
        assert not os.access(pathlib.Path(t)/man['shards']['0']['file'],os.W_OK),'shards are read-only'
    print('e57_attdump_run selftest ok')
if __name__=='__main__':main()
