"""E57 pair-shuffle control of the rule arms, QA report (new file; CPU; job 1143): tune bundles 8 / 10 / 12 (40 items), n = 65, scored with the ORIGINAL pairs. Arms: ALLQ128 and M3 w=8 on the shuffled stream (job 940 rows), ABS2 w=16 and fixed top 5% w=16 on the shuffled stream (e57_public_qa_n65_ruleshuf), and the same arms on the original stream (job 930: ALLQ128, M3 w=8; job 1080: ABS2, top 5%). Per arm: rank ratio at the divergence token, first-token dA, dA-dB (shuffled, original, original minus shuffled, item-level paired 95% CI) and the gain of the rule arms over ALLQ128 on both streams and its change.
  python3 e57_rule_shuf_report_qa.py --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder'
ARMS=['ALLQ128','M3W8Q128','M3qt05W16Q128','M3qtA2W16Q128']
def ci(x,B=2000,seed=1):
    x=np.asarray(x,float);x=x[~np.isnan(x)];ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:x=ci([1,2,3]);assert abs(x['mean']-2)<1e-9;print('e57_rule_shuf_report_qa selftest ok');return
    import e57_pqa15_run as P15,e57_pqa2_stream as S2
    tk=S2.tokenizer();dec=lambda t:tk.decode(t);ps=EXP/'e57_public_qa_n65_pairshuffle';rs=EXP/'e57_public_qa_n65_ruleshuf';o930=EXP/'e57_public_qa_n65';o1080=EXP/'e57_public_qa_n65_rules'
    sp={r['item_id']:r['split'] for r in json.load(open(EXP/'e57_qa_split.json'))['items']};meta=json.load(open(o930/'e57_pqa15_read_meta.json'));ids=[i for i in meta['item_ids'] if sp[i]=='tune'];tok=json.load(open(o930/'e57_pqa15_tokens.json'))
    ld=lambda d,c,l:json.load(open(d/f'E57_pqa15_rows_{c}_{l}.json'))
    def vec(d,arm,w0d):
        A=P15.item_arrays(ld(d,'A',f'{arm}_n65'),ids,tok['A'],dec,'q');B=P15.item_arrays(ld(d,'B',f'{arm}_n65'),ids,tok['B'],dec,'q');A0=P15.item_arrays(ld(w0d,'A','W0'),ids,tok['A'],dec,'q');B0=P15.item_arrays(ld(w0d,'B','W0'),ids,tok['B'],dec,'q')
        return dict(rr=A['rr_branch'],dA=A['A_first']-A0['A_first'],dAB=(A['A_first']-A0['A_first'])-(B['A_first']-B0['A_first']))
    where={'shuf':{'ALLQ128':ps,'M3W8Q128':ps,'M3qt05W16Q128':rs,'M3qtA2W16Q128':rs},'orig':{'ALLQ128':o930,'M3W8Q128':o930,'M3qt05W16Q128':o1080,'M3qtA2W16Q128':o1080}}
    V={st:{ar:vec(where[st][ar],ar,ps if st=='shuf' else o930) for ar in ARMS} for st in where};res={'n_items':len(ids),'arms':{},'gain_over_ALLQ128':{}}
    for ar in ARMS:res['arms'][ar]={m:dict(shuffled=ci(V['shuf'][ar][m]),original=ci(V['orig'][ar][m]),original_minus_shuffled=ci(V['orig'][ar][m]-V['shuf'][ar][m])) for m in ('rr','dA','dAB')}
    for ar in ARMS[1:]:
        g={st:dict(rr=V[st]['ALLQ128']['rr']-V[st][ar]['rr'],dAB=V[st][ar]['dAB']-V[st]['ALLQ128']['dAB'],dA=V[st][ar]['dA']-V[st]['ALLQ128']['dA']) for st in V}
        res['gain_over_ALLQ128'][ar]={m:dict(shuffled=ci(g['shuf'][m]),original=ci(g['orig'][m]),shuffled_minus_original=ci(g['shuf'][m]-g['orig'][m])) for m in ('rr','dAB','dA')}
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
