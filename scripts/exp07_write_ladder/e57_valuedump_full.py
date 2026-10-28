"""E57 value-corrected key-mass dump over ALL chunks 8..279 for a set of query rows per chunk (GPU forward only, W0, no write; post-hoc, exploratory; job 313). The pass of e57_valuedump.py (job 300: SHV / ALLV = sum over heads and layers of p x ||v||; SHVn / ALLVn = per-head value-normalised share over the window keys, summed over heads; SH = sharp heads, ALL = all 768 heads) for the rows rows_<k> of a rows file, but the full [R, 8192] vectors are reduced on the host to the TOP-128 far keys per band b in (256, 512): far keys of row i = window keys 4 <= c <= lim_i = min(8191, i + 8192 - b). Per chunk k, per name n in (SHV, ALLV, SHVn, ALLVn), per band b:  c<k>/<n>_b<b>_idx int16 [R,128], _score float32 [R,128] (descending), _far float32 [R] (sum of the score over ALL far keys of the row);  rows_<k>.  Shards of 8 chunks: <out>/E57_valuedump_full_s<j>.npz (j = chunk index // 8 - 1; existing shards are skipped: resumable). About 0.4 MB per chunk and shard row set.
  python3 e57_valuedump_full.py --bundle B --out O --rows ROWS.npz --sharp-heads S.npz   (GPU) | --selftest (CPU)"""
import sys,time,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_valuedump as VD
CH=1024;WIN=8192;TOPM=128;BANDS=(256,512);NAMES=('SHV','ALLV','SHVn','ALLVn');SHARD=8
def reduce_rows(vec,rows,band,m=TOPM):
    """vec [R, WIN] float32, rows [R] query row indices -> (idx int16 [R,m] (-1 padded), score float32 [R,m] descending, far float32 [R])"""
    R=vec.shape[0];lim=np.minimum(WIN-1,np.asarray(rows)+WIN-band);c=np.arange(WIN)[None,:];ok=(c>=4)&(c<=lim[:,None])
    sc=np.where(ok,vec,-np.inf);far=np.where(ok,vec,0.0).sum(1).astype(np.float32)
    part=np.argpartition(-sc,m-1,axis=1)[:,:m];ps=np.take_along_axis(sc,part,1);o=np.argsort(-ps,axis=1,kind='stable');ix=np.take_along_axis(part,o,1);s=np.take_along_axis(ps,o,1)
    bad=~np.isfinite(s);ix=np.where(bad,-1,ix).astype(np.int16);s=np.where(bad,0.0,s).astype(np.float32);return ix,s,far
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--rows');ap.add_argument('--sharp-heads');ap.add_argument('--bank');ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    ROOT=HERE.parents[1]
    import gen_world_ctl as G,e57_reads as RD
    tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH
    M=np.load(a.rows);SH=np.load(a.sharp_heads)['sharp'];assert SH.shape==(280,24,32);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    chunks=list(range(8,280));shards=sorted({k//SHARD for k in chunks});todo=[s for s in shards if not (out/f'E57_valuedump_full_s{s-1:02d}.npz').exists()]
    print(f'value dump full: chunks 8..279, shards {len(shards)}, to do {len(todo)}',flush=True)
    if not todo:print('all shards exist');return
    import e55_run as R;LD,s3,C=R.lab();rt=LD.Runtime();L=C.Ladder55(rt,save_dir=out/'states'/'E57_valuedump_full',canary=None);lg=lambda *x:print(*x,flush=True)
    with rt.mesh:Kh,Vh,_=L.record(np.asarray(ids)[:n*CH])
    VD.AD.install(s3);t0=time.monotonic()
    for s in todo:
        res={};ks=[k for k in chunks if k//SHARD==s]
        for k in ks:
            rows=M[f'rows_{k}'].astype(int);ids_k=np.asarray(ids[k*CH:(k+1)*CH])
            with rt.mesh:Kc,Vc=L.ctx_kv(Kh,Vh,k)
            d=VD.value_pass(L,rt,s3,rt.base,Kc,Vc,ids_k,k,SH[k],rows)
            for nm in NAMES:
                for b in BANDS:
                    ix,sc,far=reduce_rows(d[nm],rows,b);res[f'c{k}/{nm}_b{b}_idx']=ix;res[f'c{k}/{nm}_b{b}_score']=sc;res[f'c{k}/{nm}_b{b}_far']=far
            res[f'rows_{k}']=rows.astype(np.int16)
            if k%8==0:lg(f'  value dump chunk {k}: {time.monotonic()-t0:.0f}s, rows {len(rows)}')
        tmp=out/f'.tmp_s{s-1:02d}.npz';np.savez(tmp,chunks=np.asarray(ks),**res);tmp.rename(out/f'E57_valuedump_full_s{s-1:02d}.npz');lg('wrote shard',s-1)
def selftest():
    rng=np.random.default_rng(0);rows=np.array([0,5,300,1023]);vec=rng.random((4,WIN)).astype(np.float32);vec[1,::7]=0
    for b in BANDS:
        ix,sc,far=reduce_rows(vec,rows,b)
        for r,row in enumerate(rows):
            lim=min(WIN-1,row+WIN-b);cand=np.arange(4,lim+1);v=vec[r,cand];o=np.argsort(-v,kind='stable')[:TOPM]
            assert np.allclose(sc[r,:len(o)],v[o]) and np.all(ix[r,:len(o)]>=4) and np.all(ix[r,:len(o)]<=lim) and abs(far[r]-v.sum())<1e-2*max(1,v.sum())*1e-3+1e-3
            assert len(set(ix[r,:len(o)].tolist()))==len(o) and (ix[r,len(o):]==-1).all()
    ix,sc,far=reduce_rows(vec[:1,:],np.array([0]),512)
    print('e57_valuedump_full selftest ok')
if __name__=='__main__':main()
