"""E57 synthetic world: query-band masks for the write arms (CPU only; new file). From the pass file syn_s<seed>_nll.npz of e57_syn_pass.py (nll_win, nll_band [nb,128,1024], lm) it builds a mask file in the format of e57_qband_masks.py (names, band_<arm>, all_<arm>, qb_<arm>_<k> = packbits of the banded rows of chunk k, k = 8..127, p) for e57_qband_run.SelMaskQ:
  GAINQ<b><P>  the top round(p x eligible) rows of each chunk by band gain nll_band_b - nll_win   (eligible = loss rows whose next token is inside a sentence)
  RANDQ<b><P>  the same number of random eligible rows (seed 291, independent draws per arm and chunk)
  ALLQ<b>      every row of the chunk (all_ = 1; p ignored)
<P> = '' for p = .26, 'P10' for .10, 'P50' for .50 (arm names GAINQ256, GAINQ256P10, ...).
  .venv/bin/python scripts/exp07_write_ladder/e57_syn_qmasks.py --seed 64 --dir D --out O.npz [--bands 256,512] [--ps .26,.10,.50] [--kinds GAINQ,RANDQ,ALLQ] | --selftest"""
import sys,argparse,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024;K0=8;K1=128
def psuffix(p):return '' if abs(p-.26)<1e-9 else f'P{int(round(p*100)):02d}'
def arm_rows(kind,elig,gain,p,rng):
    n=int(round(p*elig.sum()))
    if kind=='GAINQ':return CM.top_rows(gain,elig,n)
    if kind=='RANDQ':return CM.rand_rows(elig,n,rng)
    raise ValueError(kind)
def selftest():
    rng=np.random.default_rng(0);elig=rng.random(CH)<.8;g=rng.standard_normal(CH);m=arm_rows('GAINQ',elig,g,.26,rng);n=int(round(.26*elig.sum()));assert int(m.sum())==n and not (m&~elig).any()
    top=np.sort(g[elig])[::-1][:n];assert np.allclose(np.sort(g[m])[::-1],top)
    r=arm_rows('RANDQ',elig,g,.26,np.random.default_rng(1));assert int(r.sum())==n and not (r&~elig).any()
    assert psuffix(.26)=='' and psuffix(.1)=='P10' and psuffix(.5)=='P50'
    print('e57_syn_qmasks selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--dir');ap.add_argument('--out');ap.add_argument('--bands',default='256,512');ap.add_argument('--ps',default='.26,.10,.50');ap.add_argument('--kinds',default='GAINQ,RANDQ,ALLQ');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e55_uv as UVm,e55_sentdump as SD
    import e57_syn_pass as SP
    bands=[int(x) for x in a.bands.split(',')];ps=[float(x) for x in a.ps.split(',')];kinds=a.kinds.split(',')
    enc,w,tf=R55.build(SP.synth_args(a.seed),a.seed,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids)
    Z=np.load(pathlib.Path(a.dir)/f'syn_s{a.seed}_nll.npz');bl=list(Z['bands']);out={};names=[];rngs={}
    def nm(kind,b,p):return f'{kind}{b}{psuffix(p)}'
    for kind in kinds:
        for b in bands:
            for p in (ps if kind!='ALLQ' else [.26]):
                n_=nm(kind,b,p);names.append(n_);out[f'band_{n_}']=np.asarray(b);out[f'all_{n_}']=np.asarray(int(kind=='ALLQ'));rngs[n_]=np.random.default_rng(291)
    for k in range(K0,K1):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm
        for kind in kinds:
            if kind=='ALLQ':continue
            for b in bands:
                g=(Z['nll_band'][bl.index(b)][k]-Z['nll_win'][k]).astype(np.float64)
                for p in ps:
                    n_=nm(kind,b,p);out[f'qb_{n_}_{k}']=CM.pk(arm_rows(kind,el,g,p,rngs[n_]))
    out['names']=np.asarray(names);out['p']=np.asarray(.26);out['chunks']=np.asarray([K0,K1]);out['seed']=np.asarray(a.seed);out['stream_sha']=np.asarray(sha[:16])
    np.savez_compressed(a.out,**out);d=hashlib.sha256()
    for kk in sorted(out):
        v=np.ascontiguousarray(out[kk]);d.update(f'{kk}|{v.dtype}|{v.shape}|'.encode()+v.tobytes())
    print('arms',names);print('mask content sha256',d.hexdigest())
if __name__=='__main__':main()
