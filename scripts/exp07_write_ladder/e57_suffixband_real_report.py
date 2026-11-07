"""E57 suffix-only band g' vs g128 on the QA bundles 5/8/9/10/12 and the news stream A (new file; CPU; GPM thread, Hans 11/07 '고고'): same statistics as e57_suffixband_report.py (no deciding rows in real text): Spearman(g', g128), top-5% overlap, g'>2 rows vs ABS2 rows and their set overlap, cost timings. Inputs e57_suffixband/qa/e57_dwg_qa_nllsuf.npz (+ .json), e57_suffixband/news/E57_suffixband_W0.npz (+ .json), the stored g128 files. Chunks k >= 8, chunk-level bootstrap.
  python3 e57_suffixband_real_report.py --out F.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;K0=8
import e57_closure_masks as CM
from e57_suffixband_report import chunk_stats,summarize
def qa(gid):
    Z=np.load(EXP/'e57_public_qa_n65/e57_dwg_qa_nll.npz');Y=np.load(EXP/'e57_suffixband/qa/e57_dwg_qa_nllsuf.npz');bi=list(Z['bands']).index(128);win=Z[f'g{gid}_win'];assert np.abs(win-Y[f'g{gid}_win']).max()<1e-2
    P=[]
    for k in range(K0,win.shape[0]):
        lm=Z[f'g{gid}_lm'][k];ix=np.nonzero(CM.rows_eligible(lm,np.zeros(CH,np.int32))&lm)[0]
        if len(ix)<50:continue
        P.append(chunk_stats((Z[f'g{gid}_band'][bi][k]-win[k]).astype(np.float64)[ix],(Y[f'g{gid}_suf'][0][k]-win[k]).astype(np.float64)[ix],np.zeros(len(ix),bool)))
    return P
def news():
    import e55_uv as UVm,e57_t1t2v2_run as V,e57_news_m3 as NM
    ids=V.stream_ids(str(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl'),'A');Z=np.load(EXP/'e57_band/e57_band_W0.npz');Y=np.load(EXP/'e57_suffixband/news/E57_suffixband_W0.npz');b=list(Z['bands']).index(128);ends=UVm.sentence_ends(UVm.tokenizer(str(ROOT/'data/tokenizer/llama3/tokenizer.json')),ids);P=[]
    for k in range(NM.K0,len(ids)//CH):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];ix=np.nonzero(CM.rows_eligible(lm,sid)&lm)[0]
        P.append(chunk_stats((Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)[ix],(Y['nll_suffixband'][0][k]-Z['nll_win'][k]).astype(np.float64)[ix],np.zeros(len(ix),bool)))
    return P
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:print('e57_suffixband_real_report selftest ok');return
    rng=np.random.default_rng(0);res={'definitions':__doc__.split('\n')[0]};C={g:qa(g) for g in (5,8,9,10,12)}
    for g,P in C.items():res[f'qa_bundle_{g}']=summarize(P,rng)
    res['qa_tune_8_10_12']=summarize(sum((C[g] for g in (8,10,12)),[]),rng);res['qa_hold_5_9']=summarize(C[5]+C[9],rng);res['news_A']=summarize(news(),rng)
    jq=json.load(open(EXP/'e57_suffixband/qa/e57_dwg_qa_nllsuf.json'));jn=json.load(open(EXP/'e57_suffixband/news/E57_suffixband_W0.json'));res['timing']=dict(qa=jq,news_sec_total=jn.get('sec_total'),news_sec_forwards=jn.get('sec_forwards'))
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
