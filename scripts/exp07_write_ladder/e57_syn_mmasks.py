"""E57 synthetic world: deciding-row mask for the M2h / M3 arms (CPU only; new file; thread 'ALLQ보다 나은 가림 방법 찾기', design confirmed by Hans 11/03 09:39 KST). From the pass file syn_s<seed>_nll.npz of e57_syn_pass.py (nll_win, nll_band [nb,128,1024], lm) and the same eligibility as e57_syn_qmasks.py (loss rows whose next token is inside a sentence) it writes for every chunk k = 8..127 the DECIDING ROWS = the top round(0.30 x eligible) rows of the chunk by window gain g = nll_band256 - nll_win (qb_TOP30_<k> = packbits of the rows) and records how many of them lie in the first 255 positions of the chunk (rows whose 256-token band still reaches into the window).
  .venv/bin/python scripts/exp07_write_ladder/e57_syn_mmasks.py --seed 64 --dir experiments/exp07_write_ladder/e57_syn_s64 --out OUT.npz | --selftest"""
import sys,argparse,json,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024;K0=8;K1=128;P=0.30
def deciding(gain,elig,p=P):
    n=int(round(p*elig.sum()));return CM.top_rows(gain,elig,n)
def selftest():
    rng=np.random.default_rng(0);elig=rng.random(CH)<.8;g=rng.standard_normal(CH);m=deciding(g,elig);n=int(round(.30*elig.sum()));assert int(m.sum())==n and not (m&~elig).any()
    assert np.allclose(np.sort(g[m])[::-1],np.sort(g[elig])[::-1][:n]);print('e57_syn_mmasks selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--dir');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e55_uv as UVm
    import e57_syn_pass as SP
    enc,w,tf=R55.build(SP.synth_args(a.seed),a.seed,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids)
    Z=np.load(pathlib.Path(a.dir)/f'syn_s{a.seed}_nll.npz');bl=list(Z['bands']);b=bl.index(256);out={};st=[]
    for k in range(K0,K1):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;g=(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64);m=deciding(g,el);out[f'qb_TOP30_{k}']=CM.pk(m)
        st.append(dict(chunk=k,n_eligible=int(el.sum()),n_deciding=int(m.sum()),n_deciding_in_first_255=int(m[:255].sum()),mean_gain_deciding=float(g[m].mean()),share_of_positive_gain=float(g[m].clip(0).sum()/max(g[el].clip(0).sum(),1e-9))))
    out['chunks']=np.asarray([K0,K1]);out['p']=np.asarray(P);out['seed']=np.asarray(a.seed);out['stream_sha']=np.asarray(sha[:16]);np.savez_compressed(a.out,**out)
    s={k:float(np.mean([x[k] for x in st])) for k in st[0] if k!='chunk'};s['frac_deciding_in_first_255']=float(np.sum([x['n_deciding_in_first_255'] for x in st])/np.sum([x['n_deciding'] for x in st]));pathlib.Path(a.out).with_suffix('.json').write_text(json.dumps(dict(summary=s,chunks=st,out_sha=hashlib.sha256(open(a.out,'rb').read()).hexdigest()[:16])));print(json.dumps(s))
if __name__=='__main__':main()
