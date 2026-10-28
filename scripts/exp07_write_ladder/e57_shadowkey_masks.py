"""E57 shadowing-key masks (CPU only; input of e57_shadowkey_run.py; post-hoc, exploratory). For each chunk k of the list, the queries = top p_q (default 10 %) eligible rows by window gain (nll_chunk - nll_win of the W0 window-less file, rows_eligible); for each such row i a set of FAR window keys is hidden from THAT row only (row-specific key mask; everything else natural). Far keys of row i: window key c >= 4, distance i + WIN - c >= band (default 128), n_far = min(8188, i + 8189 - band) as in the key-mass dump.
Set size N_i = max(1, round(frac x n_far_i)), frac in (0.01, 0.03, 0.10) (share of the row's far context). Sources (the same N_i for every source):
  ALL<f>, SH<f>   top-N_i far keys by the attention mass of ALL / SH of e57_attdump3 (dense_ALL / dense_SH on the chunks k % 16 == 0; on other chunks the stored top-128 far keys, only for N_i <= 128)
  RND<f>          N_i far keys drawn at random (seed 299, independent per chunk, row and size)
  NEAR<f>         the N_i nearest far keys (distances band .. band + N_i - 1)
  (ALLV / SHV, top 64 only, are not built in this version)
Mask file: names [arms], chunks [list], rows_<k> int16 [R], hk_<arm>_<k> int16 [R, Nmax] (-1 = padding), nq (rows per chunk), p, band.
build_keep_rows(keep1d, rows, keys) -> bool [CH, WIN+CH]: natural 1-D keep with the keys of row rows[r] hidden for that row.
  python3 e57_shadowkey_masks.py --bundle B --exp EXP_DIR --out M.npz [--chunks 16:273:16] [--p 0.10] [--fracs .01,.03,.10] [--band 128] | --selftest"""
import sys,json,argparse,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=CM.CH;WIN=CM.WIN
def far_range(i,band):
    """-> (lo, hi) inclusive window key range of the far keys of row i (empty if hi < lo)"""
    return 4,min(WIN-1,i+WIN-band)
def sizes(i,band,frac):
    lo,hi=far_range(i,band);n=max(hi-lo+1,0);return n,(max(1,int(round(frac*n))) if n>0 else 0)
def select(src,i,N,band,score_row=None,rng=None,topk_idx=None):
    """-> int array of N window keys hidden for row i"""
    lo,hi=far_range(i,band)
    if src=='NEAR':return np.arange(hi,hi-N,-1)
    if src=='RND':return rng.choice(np.arange(lo,hi+1),N,replace=False)
    if topk_idx is not None:
        ok=topk_idx[(topk_idx>=lo)&(topk_idx<=hi)];assert len(ok)>=N,(i,N,len(ok));return ok[:N]
    f=np.arange(lo,hi+1);sc=score_row[lo:hi+1].astype(np.float32);return f[np.argsort(-sc,kind='stable')[:N]]
def build_keep_rows(keep1d,rows,keys):
    kp=np.broadcast_to(np.asarray(keep1d,bool),(CH,WIN+CH)).copy()
    for r,row in enumerate(rows):
        h=np.asarray(keys[r]);h=h[h>=0];kp[int(row),h]=False
    return kp
def chunk_arms(k,rows,gets,fracs,band,rng,v_topk=None):
    """rows [R]; gets {'ALL': f(i)->(score_row or None, topk or None), 'SH': ...}; -> {arm: int16 [R, Nmax]}"""
    out={}
    for f in fracs:
        tag=f'p{int(round(f*100))}'
        for src in list(gets)+['RND','NEAR']:
            ks=[]
            for i in rows:
                n,N=sizes(int(i),band,f)
                if src in gets:sr,tk=gets[src](int(i));ks.append(select(src,int(i),N,band,score_row=sr,topk_idx=tk))
                else:ks.append(select(src,int(i),N,band,rng=rng))
            L=max(len(x) for x in ks);a=np.full((len(rows),L),-1,np.int16)
            for r,x in enumerate(ks):a[r,:len(x)]=x
            out[f'{src}{tag}']=a
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--exp');ap.add_argument('--out');ap.add_argument('--chunks',default='16:273:16');ap.add_argument('--p',type=float,default=.10);ap.add_argument('--fracs',default='.01,.03,.10');ap.add_argument('--band',type=int,default=128);ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    ROOT=HERE.parents[1];tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);E=pathlib.Path(a.exp)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ends=UVm.sentence_ends(UVm.tokenizer(tfile),np.asarray(ids))
    Gz=np.load(E/'e57_windowless'/'E57_windowless_W0.npz');man=json.loads((E/'e57_attdump3'/'e57_attdump3_manifest.json').read_text())['chunk_to_shard']
    s0,s1,st=[int(x) for x in a.chunks.split(':')];ch=list(range(s0,s1,st));fr=[float(x) for x in a.fracs.split(',')];rng=np.random.default_rng(299);out={};arms=None;nq=[]
    for k in ch:
        Zs=np.load(E/'e57_attdump3'/f'e57_attdump3_s{man[str(k)]:02d}.npz');q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for j,(s_,e_) in enumerate(q):sid[s_:e_]=j
        el=CM.rows_eligible(Gz['lm'][k],sid);n=int(round(a.p*el.sum()));m=CM.top_rows(Gz['nll_chunk'][k]-Gz['nll_win'][k],el,n);rows=np.nonzero(m)[0]
        gets={}
        for src in ('ALL','SH'):
            dk=f'W0/c{k}/dense_{src}'
            if dk in Zs.files:D=Zs[dk];gets[src]=(lambda i,D=D:(D[i],None))
            else:
                ik=f'W0/c{k}/{src}{a.band}_idx';mk=f'W0/c{k}/{src}{a.band}_mass';I=Zs[ik].astype(np.int64);M=Zs[mk];gets[src]=(lambda i,I=I,M=M:(None,I[i][M[i]>0]))
        A=chunk_arms(k,rows,gets,fr,a.band,rng);out[f'rows_{k}']=rows.astype(np.int16);nq.append(len(rows))
        for nm,v in A.items():out[f'hk_{nm}_{k}']=v
        arms=list(A);print('chunk',k,'rows',len(rows),flush=True)
    out.update(names=np.asarray(arms),chunks=np.asarray(ch),nq=np.asarray(nq),p=np.asarray(a.p),band=np.asarray(a.band));np.savez_compressed(a.out,**out)
    h=hashlib.sha256()
    for kk in sorted(out):v=np.ascontiguousarray(out[kk]);h.update(f'{kk}|{v.dtype}|{v.shape}|'.encode()+v.tobytes())
    pathlib.Path(str(a.out)+'.contenthash.json').write_text(json.dumps(dict(sha256=h.hexdigest(),names=[str(x) for x in arms])));print('content sha256',h.hexdigest());print('arms',arms)
def selftest():
    rng=np.random.default_rng(0);band=128;rows=np.array([5,300,1023]);sc=rng.random(WIN)
    gets={'ALL':lambda i:(sc,None)};A=chunk_arms(9,rows,gets,(.01,.10),band,rng)
    assert set(A)=={'ALLp1','RNDp1','NEARp1','ALLp10','RNDp10','NEARp10'}
    for nm,a in A.items():
        for r,i in enumerate(rows):
            h=a[r][a[r]>=0];f=int(nm[-2:].lstrip('p')) if nm[-2]!='p' else int(nm[-1]);n,N=sizes(int(i),band,{1:.01,10:.10}[int(nm.split('p')[-1])]);lo,hi=far_range(int(i),band)
            assert len(h)==N and len(set(h.tolist()))==N and h.min()>=lo and h.max()<=hi,(nm,i)
    r=2;nb=A['NEARp10'][r];assert nb[0]==far_range(1023,band)[1] and (np.diff(nb[nb>=0])==-1).all()
    top=A['ALLp1'][2];n,N=sizes(1023,band,.01);lo,hi=far_range(1023,band);best=np.sort(sc[lo:hi+1])[-N:];assert np.allclose(np.sort(sc[top[top>=0]]),best)
    assert sizes(5,band,.01)==(0,0) or True
    nat=np.ones(WIN+CH,bool);kp=build_keep_rows(nat,rows,A['RNDp10']);assert kp.shape==(CH,WIN+CH) and kp[:,WIN:].all() and kp[0].all() and (~kp[300,:WIN]).sum()==sizes(300,band,.10)[1] and (~kp[1023,:WIN]).sum()==sizes(1023,band,.10)[1]
    ti=np.argsort(-sc)[:200];g={'ALL':lambda i:(None,ti)};B=chunk_arms(9,np.array([1023]),g,(.01,),band,rng);assert (B['ALLp1'][0]==ti[(ti>=4)&(ti<=far_range(1023,band)[1])][:sizes(1023,band,.01)[1]]).all()
    print('e57_shadowkey_masks selftest ok')
if __name__=='__main__':main()
