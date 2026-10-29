"""E57 post-write BAND gain (band-k NLL - window NLL) by selected / not selected query rows (CPU only; new file; numbers only). For the saved band files of job 345 / 348 (E57_band_<LABEL>.npz: nll_win [280,1024], nll_band [nb,280,1024], bands, lm) and the exact p=.26 GAINQ row masks of the band (qb_GAINQ<b>_<chunk>), over the chunks 8..279 and the groups SEL (banded rows), NOSEL (other eligible rows), ALL (eligible rows): mean window NLL, mean band NLL, band gain = band - window, and the ratio of the band gain to the band gain of W0 on the same rows (W0 band gain from e57_band_W0.npz for b = 128 / 512 / 1024 and from E57_closure_W0.npz of job 301 for b = 256). 95 % interval of the ratio: bootstrap over the chunks.
  .venv/bin/python scripts/exp07_write_ladder/e57_band_gain_split.py --bundle B --states STATE1=band,STATE2=band,... --band-dir D1,D2 --w0-band e57_band_W0.npz --w0-256 E57_closure_W0.npz --masks M128_512_1024.npz,M256.npz --out O.json"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--states');ap.add_argument('--band-dir');ap.add_argument('--w0-band');ap.add_argument('--w0-256');ap.add_argument('--masks');ap.add_argument('--out');ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids)
    M={}
    for f in a.masks.split(','):M.update({k:v for k,v in np.load(f,allow_pickle=True).items() if k.startswith('qb_')})
    W=np.load(a.w0_band);W2=np.load(a.w0_256);w0={int(b):(W['nll_band'][i],W['nll_win']) for i,b in enumerate(W['bands'])};w0[256]=(W2['nll_band'][list(W2['bands']).index(256)],W2['nll_win'])
    sid={}
    for k in range(8,280):
        q=UVm.chunk_sents(ends,k,CH);s=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):s[s_:e_]=i
        sid[k]=s
    out={}
    for spec in a.states.split(','):
        lab,_,bd=spec.partition('=');b=int(bd);z=None
        for d in a.band_dir.split(','):
            f=pathlib.Path(d)/f'E57_band_{lab}.npz'
            if f.exists():z=np.load(f);break
        assert z is not None,lab
        nb=z['nll_band'][list(z['bands']).index(b)];nw=z['nll_win'];wb,ww=w0[b];per={g:[] for g in ('SEL','NOSEL','ALL')}
        for k in range(8,280):
            el=CM.rows_eligible(z['lm'][k],sid[k])&z['lm'][k];qb=CM.up(M[f'qb_GAINQ{b}_{k}'],CH)
            for g,m in (('SEL',el&qb),('NOSEL',el&~qb),('ALL',el)):per[g].append((float(nw[k][m].sum()),float(nb[k][m].sum()),float((wb[k]-ww[k])[m].sum()),int(m.sum())))
        rng=np.random.default_rng(0);r={}
        for g in per:
            p=np.array(per[g]);n=p[:,3].sum();gain=(p[:,1].sum()-p[:,0].sum())/n;g0=p[:,2].sum()/n;bs=[]
            for _ in range(500):
                i=rng.integers(0,len(p),len(p));bs.append(((p[i,1].sum()-p[i,0].sum())/p[i,3].sum())/(p[i,2].sum()/p[i,3].sum()))
            r[g]=dict(rows=int(n),window=float(p[:,0].sum()/n),band=float(p[:,1].sum()/n),band_gain=float(gain),band_gain_W0=float(g0),ratio=float(gain/g0),ratio_ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))])
        out[f'{lab}@{b}']=r
    json.dump(dict(chunks='8..279',definition='band gain = band-b NLL - window NLL; ratio = state band gain / W0 band gain on the same rows; SEL = GAINQ<b> p=.26 rows',rows=out),open(a.out,'w'),indent=1)
    for k,r in out.items():print(f"{k:22s} "+' | '.join(f"{g}: win {v['window']:.3f} band {v['band']:.3f} gain {v['band_gain']:+.3f} (W0 {v['band_gain_W0']:+.3f}) ratio {v['ratio']:.3f} [{v['ratio_ci95'][0]:.3f},{v['ratio_ci95'][1]:.3f}]" for g,v in r.items()))
if __name__=='__main__':main()
