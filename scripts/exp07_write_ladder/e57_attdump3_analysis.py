"""E57 key-selection analysis on the v3 key-mass dump (CPU only; reads e57_attdump3/, e57_pod_masks.npz, E57_windowless_W0.npz; post-hoc, exploratory; reports numbers, sets no criterion).
Question: how well does a per-query-row, token-level key choice (top-m far keys of a score) hit the keys of ORCA window sentences (a window sentence holding the gold of any bank question = h5 of e57_pod_masks.npz) compared with a random choice of the same size among the row's far keys?
Rows: eligible rows (loss and predicted token inside a sentence, e57_closure_masks.rows_eligible), all of them (ALL) and the rows predicting ORCA tokens of the chunk (ORCQ). Far keys of band k: window key c >= 4 with distance i + WIN - c >= k. Only rows with at least one ORCA far key count (the others have no positive; their number is reported).
Scores (top-128 stored per row and band; the first m by the stored order): SH / ALL / G3 = mass; SHV / ALLV = mass x ||v||; SHcnt16 / ALLcnt16 = heads voting rank<=16 (ties by mass); SHthr.05 = heads with p >= .05 (ties by mass).
Per row: prec(m) = ORCA keys in the first m selected / m; base = ORCA far keys / far keys (expected precision of random); cover(m) = 1 if any selected key is ORCA; rcover(m) = 1 - C(n_far - n_orca, m) / C(n_far, m). Pooled per chunk (sum over rows) then mean over chunks with a bootstrap over chunks (5000 draws, seed 0).
  .venv/bin/python scripts/exp07_write_ladder/e57_attdump3_analysis.py --bundle B --exp EXP_DIR --out OUT.json [--chunks 8:280] [--selftest]   (run with CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu)"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;WIN=8192;MS=(1,4,8,16,32,64,128);BANDS=(128,512,1024)
SCORES=('SH','ALL','G3','SHV','ALLV','SHcnt16','ALLcnt16','SHthr.05')
def order(Z,k,band,name):
    """-> (idx [n,M] int, valid bool [n,M]) of the first M selected far keys in the order of the score"""
    pre={'SHcnt16':'SH','ALLcnt16':'ALL','SHthr.05':'SH'}.get(name,name);idx=Z[f'{pre}{band}_idx'].astype(np.int64);mass=Z[f'{pre}{band}_mass'].astype(np.float32)
    if name in('SHcnt16','ALLcnt16','SHthr.05'):
        cnt=Z[f'{pre}{band}_cnt'];col=2 if name.endswith('16') else (4+3 if pre=='SH' else 3+3)  # SH cnt: 4 ranks then edges .005,.01,.02,.05 -> col 7; ALL cnt: 3 ranks then edges -> col 6
        o=np.lexsort((-mass,-cnt[:,:,col].astype(np.int32)),axis=1);idx=np.take_along_axis(idx,o,1);mass=np.take_along_axis(mass,o,1)
    return idx,mass>0
def row_stats(idx,valid,orca,i,band):
    """per row: nfar, norca, hits at MS"""
    cs=np.concatenate([[0],np.cumsum(orca)]);hi=np.minimum(WIN-1,i+WIN-band);nfar=np.maximum(hi-4+1,0);norca=np.where(hi>=4,cs[hi+1]-cs[4],0)
    h=(orca[idx]&valid).astype(np.int32);ch=np.cumsum(h,1);return nfar,norca,{m:(ch[:,m-1] if m<=ch.shape[1] else np.full(len(ch),np.nan)) for m in MS},np.cumsum(valid,1)
def rcover(nf,no,m):
    """1 - C(nf-no, m)/C(nf, m) = 1 - prod_{j<m} (nf-no-j)/(nf-j), vectorised"""
    p=np.ones(len(nf))
    for j in range(m):p*=np.clip((nf-no-j)/np.maximum(nf-j,1),0,1)
    return 1-p
def chunk_acc(Z,k,orca,rows_sel,scores=SCORES,bands=BANDS):
    """-> {(band,score,rows): {'prec':[m..], 'cover':[..], 'rcover':[..], 'base':x, 'n':n}} sums for one chunk"""
    out={};i_all=np.arange(CH)
    for band in bands:
        for sc in scores:
            idx,valid=order(Z,k,band,sc);nf,no,hits,nval=row_stats(idx,valid,orca,i_all,band)
            for rn,rm in rows_sel.items():
                r=rm&(no>0)
                if not r.any():continue
                o={'n':int(r.sum()),'base':float((no[r]/nf[r]).sum()),'prec':[],'cover':[],'rcover':[]}
                for m in MS:
                    o['prec'].append(float((hits[m][r]/m).sum()));o['cover'].append(float((hits[m][r]>0).sum()));o['rcover'].append(float(rcover(nf[r],no[r],m).sum()))
                out[(band,sc,rn)]=o
    return out
def boot(vals,w,rng,nb=5000):
    """ratio of sums over chunks with bootstrap over chunks: vals,w [nchunk]"""
    vals=np.asarray(vals,float);w=np.asarray(w,float);pt=vals.sum()/w.sum();ix=rng.integers(0,len(w),(nb,len(w)));bs=vals[ix].sum(1)/w[ix].sum(1);return [float(pt),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
def summarize(acc):
    """acc {(band,score,rows): list of per-chunk dicts} -> nested json"""
    rng=np.random.default_rng(0);res={}
    for (band,sc,rn),L in acc.items():
        n=[d['n'] for d in L];e={'chunks':len(L),'rows':int(sum(n)),'base':boot([d['base'] for d in L],n,rng)}
        for key in('prec','cover','rcover'):e[key]={str(m):boot([d[key][j] for d in L],n,rng) for j,m in enumerate(MS)}
        e['lift_prec']={str(m):e['prec'][str(m)][0]/e['base'][0] for m in MS};res[f'{band}|{sc}|{rn}']=e
    return res
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--exp');ap.add_argument('--out');ap.add_argument('--chunks',default='8:280');ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_mask_stage as MS_,e57_closure_masks as CM
    from tokenizers import Tokenizer
    ROOT=HERE.parents[1];tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);tk=Tokenizer.from_file(tfile);E=pathlib.Path(a.exp)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids)
    P=np.load(E/'e57_pod'/'e57_pod_masks.npz');lmz=np.load(E/'e57_windowless'/'E57_windowless_W0.npz')['lm'];man=json.loads((E/'e57_attdump3'/'e57_attdump3_manifest.json').read_text());c2s=man['chunk_to_shard'];memo={}
    k0,k1=[int(x) for x in a.chunks.split(':')];acc={};shard=None;nskip=0;info=[]
    for k in range(k0,k1):
        s=c2s[str(k)]
        if shard!=s:shard=s;Zs=np.load(E/'e57_attdump3'/f'e57_attdump3_s{s:02d}.npz')
        wsid,wst,wlen=UVm.window_sents(ends,k,CH,8);assert np.array_equal(P[f'wlen{k}'],wlen)
        hw=np.asarray(P[f'h5_{k}'],bool);orca=(wsid>=0)&hw[np.clip(wsid,0,max(len(wlen)-1,0))]
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for j,(s_,e_) in enumerate(q):sid[s_:e_]=j
        _,hc=MS_.oracle_hidden(k,ends,ids,tk,items,UVm,memo);ot=np.zeros(CH,bool)
        for j,(s_,e_) in enumerate(q):ot[s_:e_]=hc[j]
        el=CM.rows_eligible(lmz[k],sid);orcq=np.zeros(CH,bool);orcq[:CH-1]=ot[1:]&el[:CH-1]
        info.append((k,int(orca.sum()),int(el.sum()),int(orcq.sum())))
        Z={n_.split('/')[-1]:Zs[n_] for n_ in Zs.files if n_.startswith(f'W0/c{k}/')} if any(n_.startswith('W0/c') for n_ in Zs.files) else None
        if Z is None:raise SystemExit('unexpected shard keys '+str(Zs.files[:5]))
        Z={kk[len(''):]:v for kk,v in Z.items()}
        # names are SH<k>_idx with k = band; strip the chunk prefix already done
        r=chunk_acc(Z,k,orca,{'ALL':el,'ORCQ':orcq})
        for key,v in r.items():acc.setdefault(key,[]).append(v)
        if k%20==0:print('chunk',k,flush=True)
    res=summarize(acc);meta={'chunks':[k0,k1],'orca_keys_per_chunk_mean':float(np.mean([x[1] for x in info])),'chunks_without_orca_key':int(sum(1 for x in info if x[1]==0)),'eligible_rows_mean':float(np.mean([x[2] for x in info])),'orcq_rows_mean':float(np.mean([x[3] for x in info])),'MS':list(MS)}
    pathlib.Path(a.out).write_text(json.dumps({'meta':meta,'res':res}))
    for key in sorted(res):
        e=res[key];print(key,'rows',e['rows'],'base %.4f'%e['base'][0],'prec@8 %.4f lift %.2f'%(e['prec']['8'][0],e['lift_prec']['8']),'cover@32 %.3f (rand %.3f)'%(e['cover']['32'][0],e['rcover']['32'][0]))
def selftest():
    rng=np.random.default_rng(0);n=CH;band=128;k=9
    orca=np.zeros(WIN,bool);orca[1000:1100]=True;Z={}
    for sc in('SH','ALL','G3','SHV','ALLV'):
        idx=np.tile(np.arange(1000,1128),(n,1)).astype(np.int16);Z[f'{sc}{band}_idx']=idx[:,:(128 if sc in('SH','ALL') else 64)];Z[f'{sc}{band}_mass']=np.ones_like(Z[f'{sc}{band}_idx'],np.float16)
    for sc,nc in(('SH',10),('ALL',9)):Z[f'{sc}{band}_cnt']=np.zeros((n,128,nc),np.uint8)
    nf,no,h,_=row_stats(*order(Z,k,band,'SH'),orca,np.arange(n),band)
    assert (nf[:5]==np.maximum(np.minimum(WIN-1,np.arange(5)+WIN-band)-3,0)).all() and int(no[1023])==100 and int(h[8][0])==8 and int(h[128][0])==100
    rc=rcover(np.array([100.]),np.array([10.]),1);assert abs(rc[0]-.1)<1e-9
    Zb=dict(Z);r=chunk_acc({kk:v for kk,v in Z.items()},k,orca,{'ALL':np.ones(n,bool)},scores=('SH','ALL','G3','SHV','ALLV'),bands=(128,));print(r[(128,'SH','ALL')]['n'])
    print('e57_attdump3_analysis selftest ok')
if __name__=='__main__':main()
