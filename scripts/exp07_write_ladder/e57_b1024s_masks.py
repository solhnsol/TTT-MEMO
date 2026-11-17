"""E57 synthetic world: deciding rows for the band-1024 loss-weight arms (CPU only; new file; job 1308 proposal v2 (clipped SGD, K0 = 0), Hans 11/17; e57_syn_mmasks.py is imported, not edited). From the pass file syn_s<seed>_nll.npz of e57_syn_w0pass.py (run with --bands 256,1024: nll_win, nll_band [nb,n,1024], lm) and the same eligibility as e57_syn_mmasks.py (loss rows whose next token is inside a sentence) it writes for every chunk k = 8..n-1:
  qb_TOP30G1024_<k>  the top round(0.30 x eligible) rows by the window gain g1024 = nll_band1024 - nll_win (band 1024 = every row sees only the previous 1024 tokens, window AND chunk keys)  [the arm M3G1024]
  qb_RAND30_<k>      the SAME number of rows drawn uniformly at random among the eligible rows (rng seeded by (seed, chunk, 30))            [the arm M3R1024]
plus, for the clipped-SGD wrapper e57_b1024s_run.py, the mask-arm keys band_<code> = 1024 and all_<code> = 1 for the codes B1024G, B1024R, B1024GN, B1024RN, B1024W1T (every row banded, as ALLQ1024 of e57_natmask_run.py), and records how many of the g1024 rows lie in the first 1023 positions of the chunk (rows whose 1024-token band still reaches into the window).
  .venv/bin/python scripts/exp07_write_ladder/e57_b1024s_masks.py --seed 71 --dir experiments/exp07_write_ladder/e57_b1024s_s71 --out OUT.npz | --selftest"""
import sys,argparse,json,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
import e57_syn_mmasks as MM
CH=MM.CH;K0=MM.K0;P=MM.P
CODES=('B1024G','B1024R','B1024GN','B1024RN','B1024W1T','B1024G15','B1024R15')
def rand_rows(elig,n,seed,k,tag=30):
    rng=np.random.default_rng((seed,k,tag));idx=np.nonzero(elig)[0];pick=rng.choice(idx,n,replace=False);m=np.zeros(len(elig),bool);m[pick]=True;return m
def rows_for_chunk(gain,elig,seed,k,p=P):
    """-> (top-p rows by gain, random rows of the same count) among the eligible rows"""
    t=MM.deciding(gain,elig,p);n=int(t.sum());r=rand_rows(elig,n,seed,k,int(round(100*p)));return t,r
def selftest():
    rng=np.random.default_rng(0);elig=rng.random(CH)<.8;g=rng.standard_normal(CH);t,r=rows_for_chunk(g,elig,71,9);n=int(round(.30*elig.sum()))
    assert int(t.sum())==n==int(r.sum()) and not (t&~elig).any() and not (r&~elig).any() and not np.array_equal(t,r)
    assert np.allclose(np.sort(g[t])[::-1],np.sort(g[elig])[::-1][:n])
    t2,r2=rows_for_chunk(g,elig,71,9);assert np.array_equal(r,r2) and np.array_equal(t,t2)               # deterministic
    assert not np.array_equal(r,rows_for_chunk(g,elig,72,9)[1]) and not np.array_equal(r,rows_for_chunk(g,elig,71,10)[1])
    z=np.random.default_rng(1).standard_normal((200,CH));hits=[(rows_for_chunk(g,elig,71,k)[1]&(g>np.percentile(g[elig],70))).sum() for k in range(200)];assert abs(np.mean(hits)/(.3*elig.sum()*.3)-1)<0.15   # random rows pick the high-gain rows at the base rate (30%)
    t15,r15=rows_for_chunk(g,elig,71,9,0.15);n15=int(round(.15*elig.sum()));assert int(t15.sum())==n15==int(r15.sum()) and (t15&~t).sum()==0 and not np.array_equal(r15,rand_rows(elig,n15,71,9,30))      # top 15% is inside top 30%; the 15% random draw uses its own seed tag
    pk=CM.pk(t);assert np.array_equal(CM.up(pk,CH),t)
    print('e57_b1024s_masks selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--cell',default='v3_16_1');ap.add_argument('--dir');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e55_uv as UVm
    import e57_syn_pass as SP
    enc,w,tf=R55.build(SP.synth_args(a.seed,a.cell),a.seed,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids)
    Z=np.load(pathlib.Path(a.dir)/f'syn_s{a.seed}_nll.npz');bl=list(Z['bands']);assert 1024 in bl,'the W0 pass must be run with --bands 256,1024';b=bl.index(1024);out={};st=[];K1=int(Z['lm'].shape[0])
    for k in range(K0,K1):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;g=(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64);t,r=rows_for_chunk(g,el,a.seed,k);out[f'qb_TOP30G1024_{k}']=CM.pk(t);out[f'qb_RAND30_{k}']=CM.pk(r);t15,r15=rows_for_chunk(g,el,a.seed,k,0.15);out[f'qb_TOP15G1024_{k}']=CM.pk(t15);out[f'qb_RAND15_{k}']=CM.pk(r15)
        st.append(dict(chunk=k,n_eligible=int(el.sum()),n_deciding=int(t.sum()),n_random=int(r.sum()),n_g_in_first_1023=int(t[:1023].sum()),mean_gain_top=float(g[t].mean()),mean_gain_random=float(g[r].mean()),share_of_positive_gain_top=float(g[t].clip(0).sum()/max(g[el].clip(0).sum(),1e-9))))
    for code in CODES:out[f'band_{code}']=np.asarray(1024);out[f'all_{code}']=np.asarray(1)
    out['chunks']=np.asarray([K0,K1]);out['p']=np.asarray(P);out['seed']=np.asarray(a.seed);out['stream_sha']=np.asarray(sha[:16]);np.savez_compressed(a.out,**out)
    s={k:float(np.mean([x[k] for x in st])) for k in st[0] if k!='chunk'};pathlib.Path(a.out).with_suffix('.json').write_text(json.dumps(dict(summary=s,chunks=st,out_sha=hashlib.sha256(open(a.out,'rb').read()).hexdigest()[:16])));print(json.dumps(s))
if __name__=='__main__':main()
