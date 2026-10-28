"""E57 query-band write arms (CPU only; job 291 / 292 input; post-hoc, exploratory). Arms of the write experiment in which SELECTED QUERY ROWS see only the previous k tokens (band k: the row itself and the k-1 before it, window AND chunk keys, prefix AND suffix layers, e57_closure_masks.band_keep) and all other rows see the natural window; no key-side mask, so the effect of the QUERY SELECTION alone is measured (Hans 05:42Z). Row i is the attention row whose output predicts token i+1.
Arms (k = 128, 512, 1024):
  ALLQ<k>   every row is banded (selection-free reference: the whole window beyond k is hidden for every query)
  RANDQ<k>  a random p fraction of the eligible rows is banded (same count as GAINQ<k> / ORCAQ<k> candidates; analytic comparator)
  ORCAQ<k>  rows predicting ORCA tokens (chunk sentences with the gold of any bank question) are banded (reference)
  GAINQ<k>  the top p fraction of eligible rows by window gain for band k (nll_band_k - nll_win of job 290 at W0) is banded   [needs --band-npz from job 290; job 292]
  NATQ      no mask (the NAT arm of the SELC1 engine = NATADAMH)
p is a candidate (default 0.26, = ORCA token share), NOT a decision. Chunks k = 8..279 only (full window); chunks 1..7 are written unmasked in every arm.
Mask file: names [arms], band_<ARM> int, all_<ARM> bool-as-int (ALLQ), qb_<ARM>_<k> uint8 [128] = packbits of bool [1024], p, chunks [8,280].
content_det(file) = sha256 over sorted keys / dtypes / shapes / bytes excluding the RANDQ keys (random generator stream).
  python3 e57_qband_masks.py --bundle B --gain E57_windowless_W0.npz --out M.npz [--p 0.26] [--bands 128,512,1024] [--band-npz E57_closure_W0.npz (job 290)] [--only-gain] | --allq-only --bands 128,512,1024 --out M.npz | --verify FILE [--expect-det H] | --selftest"""
import sys,json,argparse,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=CM.CH;WIN=CM.WIN;K0=8;K1=280
def content_det(path):
    Z=np.load(path);d=hashlib.sha256();a=hashlib.sha256()
    for k in sorted(Z.files):
        v=np.ascontiguousarray(Z[k]);h=f'{k}|{v.dtype}|{v.shape}|'.encode()+v.tobytes();a.update(h)
        if 'RANDQ' not in k:d.update(h)
    return d.hexdigest(),a.hexdigest()
def arm_rows(arm,elig,orcq,gain_k,p,rng):
    """-> bool [CH] banded rows of one chunk"""
    n=int(round(p*elig.sum()))
    if arm.startswith('ALLQ'):return np.ones(CH,bool)
    if arm.startswith('RANDQ'):return CM.rand_rows(elig,n,rng)
    if arm.startswith('ORCAQ'):return orcq
    if arm.startswith('GAINQ'):return CM.top_rows(gain_k,elig,n)
    raise ValueError(arm)
def build(chunk_iter,bands,p,band_gain=None,only_gain=False,seed=291,kinds=None):
    """chunk_iter yields (k, lm bool [CH], sid int [CH] (-1 outside sentences), orca_tok bool [CH], nll_win [CH]); band_gain {band: nll_band [280,CH]} -> npz dict"""
    rng=np.random.default_rng(seed);arms=[];out={}
    for k_ in bands:
        for pre in (kinds or (('GAINQ',) if only_gain else ('ALLQ','RANDQ','ORCAQ')+(('GAINQ',) if band_gain else ()))):arms.append(f'{pre}{k_}')
    for k,lm,sid,orc,nw in chunk_iter:
        elig=CM.rows_eligible(lm,sid);orcq=np.zeros(CH,bool);orcq[:CH-1]=np.asarray(orc,bool)[1:]&elig[:CH-1]
        for a in arms:
            b=int(a.lstrip('ALQNDRCOGIMP')) if False else int(''.join(ch for ch in a if ch.isdigit()))
            gk=(band_gain[b][k]-nw) if (band_gain and a.startswith('GAINQ')) else None
            if a.startswith('ALLQ'):continue
            out[f'qb_{a}_{k}']=CM.pk(arm_rows(a,elig,orcq,gk,p,rng))
    for a in arms:
        out[f'band_{a}']=np.asarray(int(''.join(ch for ch in a if ch.isdigit())));out[f'all_{a}']=np.asarray(int(a.startswith('ALLQ')))
    out['names']=np.asarray(arms);out['p']=np.asarray(p);out['chunks']=np.asarray([K0,K1]);return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--gain');ap.add_argument('--out');ap.add_argument('--p',type=float,default=0.26);ap.add_argument('--bands',default='128,512,1024');ap.add_argument('--band-npz');ap.add_argument('--only-gain',action='store_true');ap.add_argument('--allq-only',action='store_true');ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');ap.add_argument('--verify');ap.add_argument('--expect-det');a=ap.parse_args()
    if a.selftest:selftest();return
    if a.verify:
        det,al=content_det(a.verify);print(f'content sha256 det={det} all={al}',flush=True)
        if a.expect_det and a.expect_det!=det:print(f'MISMATCH: expected det {a.expect_det}',flush=True);sys.exit(9)
        return
    if a.allq_only:      # ALLQ arms need no per-chunk rows: no bundle, no 287a file
        out=build([],[int(x) for x in a.bands.split(',')],a.p,kinds=('ALLQ',));np.savez_compressed(a.out,**out);det,al=content_det(a.out);pathlib.Path(str(a.out)+'.contenthash.json').write_text(json.dumps(dict(det=det,all=al,names=[str(x) for x in out['names']])));print(f'content sha256 det={det} all={al}');print('arms',[str(x) for x in out['names']]);return
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_mask_stage as MS
    from tokenizers import Tokenizer
    ROOT=HERE.parents[1];tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);tk=Tokenizer.from_file(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids)
    Gz=np.load(a.gain);memo={};bands=[int(x) for x in a.bands.split(',')]
    bg=None
    if a.band_npz:
        B=np.load(a.band_npz);bg={int(b):B['nll_band'][i] for i,b in enumerate(B['bands'])};assert set(bands)<=set(bg),(bands,list(bg))
    def it():
        for k in range(K0,K1):
            q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
            for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
            _,hc=MS.oracle_hidden(k,ends,ids,tk,items,UVm,memo);orc=np.zeros(CH,bool)
            for i,(s_,e_) in enumerate(q):orc[s_:e_]=hc[i]
            yield k,Gz['lm'][k],sid,orc,Gz['nll_win'][k]
    out=build(it(),bands,a.p,bg,a.only_gain);np.savez_compressed(a.out,**out)
    det,al=content_det(a.out);pathlib.Path(str(a.out)+'.contenthash.json').write_text(json.dumps(dict(det=det,all=al,names=[str(x) for x in out['names']],p=a.p)))
    print(f'content sha256 det={det} all={al}');print('arms',[str(x) for x in out['names']])
    for nm in out['names']:
        nm=str(nm);rows=[int(CM.up(out[f"qb_{nm}_{k}"],CH).sum()) for k in range(K0,K1)] if f'qb_{nm}_{K0}' in out else [CH]*(K1-K0);print(f'  {nm}: mean banded rows per chunk {np.mean(rows):.1f}')
def selftest():
    rng=np.random.default_rng(0);lm=np.ones(CH,bool);sid=np.zeros(CH,int);sid[:3]=-1;orc=np.zeros(CH,bool);orc[100:300]=True;nw=rng.random(CH)
    bg={128:{9:nw+rng.random(CH)}}
    o=build([(9,lm,sid,orc,nw)],[128],0.26,bg)
    assert [str(x) for x in o['names']]==['ALLQ128','RANDQ128','ORCAQ128','GAINQ128'] and int(o['all_ALLQ128'])==1 and 'qb_ALLQ128_9' not in o
    e=CM.rows_eligible(lm,sid);n=int(round(.26*e.sum()));qr=CM.up(o['qb_RANDQ128_9'],CH);qg=CM.up(o['qb_GAINQ128_9'],CH);qo=CM.up(o['qb_ORCAQ128_9'],CH)
    assert qr.sum()==n and qg.sum()==n and not qr[~e].any() and not qg[~e].any() and qo.sum()==200-0 and qo[99:299].all()
    g=(bg[128][9]-nw);assert g[qg].min()>=np.sort(g[e])[-n]-1e-12
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        np.savez_compressed(pathlib.Path(t)/'a.npz',**o);o2=dict(o);o2['qb_RANDQ128_9']=CM.pk(np.zeros(CH,bool));np.savez(pathlib.Path(t)/'b.npz',**o2)
        assert content_det(pathlib.Path(t)/'a.npz')[0]==content_det(pathlib.Path(t)/'b.npz')[0] and content_det(pathlib.Path(t)/'a.npz')[1]!=content_det(pathlib.Path(t)/'b.npz')[1]
    print('e57_qband_masks selftest ok')
if __name__=='__main__':main()
