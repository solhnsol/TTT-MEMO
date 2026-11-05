"""E57 synthetic s64: partial-mask row sets for the arms M3-D30 / M3-D50 / M3-R30 (CPU only; new file; method thread, Hans card 11/05): from the W0 pass file syn_s64_nll.npz and the eligibility of e57_syn_mmasks.py (loss rows whose next token is inside a sentence) per chunk k = 8..127: qp_D30_k = top 30% of the eligible rows by the window gain g = nll_band256 - nll_win (the deciding rows of jobs 710 / 730), qp_D50_k = top 50%, qp_R30_k = a uniformly random subset of the eligible rows with the SAME size as D30 (np.random.default_rng(6400 + k)). The arms mask (band 256) and weight (w = 8) exactly these rows, all other rows keep the natural window. Output: <out> npz + json (rows per chunk, share of the eligible rows).
  .venv/bin/python scripts/exp07_write_ladder/e57_syn_pmasks.py --out F.npz | --selftest"""
import sys,json,argparse,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_syn_mmasks as MM
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;K0=8;K1=128
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        rng=np.random.default_rng(0);el=rng.random(CH)<.8;g=rng.standard_normal(CH);d30=MM.deciding(g,el,0.30);d50=MM.deciding(g,el,0.50);assert d30.sum()==int(round(.3*el.sum())) and d50.sum()==int(round(.5*el.sum())) and (d30&~d50).sum()==0;print('e57_syn_pmasks selftest ok');return
    import e55_run as R55,e55_uv as UVm,e57_syn_pass as SP
    enc,w,tf=R55.build(SP.synth_args(64),64,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);Z=np.load(EXP/'e57_syn_s64/syn_s64_nll.npz');b=list(Z['bands']).index(256);out={};st=[]
    for k in range(K0,K1):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;g=(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64);d30=MM.deciding(g,el,0.30);d50=MM.deciding(g,el,0.50)
        idx=np.where(el)[0];r=np.random.default_rng(6400+k).choice(idx,size=int(d30.sum()),replace=False);r30=np.zeros(CH,bool);r30[r]=True
        out[f'qp_D30_{k}']=CM.pk(d30);out[f'qp_D50_{k}']=CM.pk(d50);out[f'qp_R30_{k}']=CM.pk(r30)
        st.append(dict(chunk=k,n_eligible=int(el.sum()),n_D30=int(d30.sum()),n_D50=int(d50.sum()),n_R30=int(r30.sum()),mean_g_D30=float(g[d30].mean()),mean_g_D50=float(g[d50].mean()),mean_g_R30=float(g[r30].mean()),mean_g_eligible=float(g[el].mean()),share_pos_gain_D30=float(g[d30].clip(0).sum()/max(g[el].clip(0).sum(),1e-9)),share_pos_gain_D50=float(g[d50].clip(0).sum()/max(g[el].clip(0).sum(),1e-9)),share_pos_gain_R30=float(g[r30].clip(0).sum()/max(g[el].clip(0).sum(),1e-9))))
    out['chunks']=np.asarray([K0,K1]);np.savez_compressed(a.out,**out)
    s={k:float(np.mean([x[k] for x in st])) for k in st[0] if k!='chunk'};s['share_of_eligible_D30']=s['n_D30']/s['n_eligible'];s['share_of_eligible_D50']=s['n_D50']/s['n_eligible'];s['share_of_eligible_R30']=s['n_R30']/s['n_eligible']
    pathlib.Path(a.out).with_suffix('.json').write_text(json.dumps(dict(summary=s,chunks=st,out_sha=hashlib.sha256(open(a.out,'rb').read()).hexdigest()[:16])));print(json.dumps(s))
if __name__=='__main__':main()
