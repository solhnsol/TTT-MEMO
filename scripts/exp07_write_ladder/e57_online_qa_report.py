"""E57 online ABS2 rule on the QA 65-version stream, report (new file; CPU; job 1160): n = 65, five bundles, tune bundles 8 / 10 / 12 (40 items) and hold-out bundles 5 / 9 (33 items). Arms: ALLQ128 and M3 w=8 (job 930, e57_public_qa_n65), top 5% w=16 and ABS2 w=16 with the W0 g (job 1080, e57_public_qa_n65_rules), the online ABS2 (job 1160, e57_public_qa_n65_online). Per arm and part: rank ratio at the divergence token and its improvement over W0, gold log-probability gain over W0 (sum of the content-token log-probs of the gold answer), first-token dA, dA-dB, item-level 95% CIs; paired differences online - W0-ABS2 and online - ALLQ128; canary rise (mean over the bundles of the part); mean share of rows selected by the online rule.
  python3 e57_online_qa_report.py --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder'
def ci(x,B=2000,seed=1):
    x=np.asarray(x,float);x=x[~np.isnan(x)];ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:x=ci([1,2,3]);assert abs(x['mean']-2)<1e-9;print('e57_online_qa_report selftest ok');return
    import e57_pqa15_run as P15,e57_pqa2_stream as S2
    tk=S2.tokenizer();dec=lambda t:tk.decode(t);o930=EXP/'e57_public_qa_n65';o1080=EXP/'e57_public_qa_n65_rules';o1160=EXP/'e57_public_qa_n65_online'
    sp={r['item_id']:r['split'] for r in json.load(open(EXP/'e57_qa_split.json'))['items']};meta=json.load(open(o930/'e57_pqa15_read_meta.json'));ids=meta['item_ids'];tok=json.load(open(o930/'e57_pqa15_tokens.json'))
    ld=lambda d,c,l:json.load(open(d/f'E57_pqa15_rows_{c}_{l}.json'))
    def gold_lp(rows):
        byid={r['id']:r for r in rows};return np.array([P15.content_lps(byid[i]['cand_tok_lp'],tok['A'][i],dec)[0][0].sum() for i in ids])
    W0A=ld(o930,'A','W0');W0B=ld(o930,'B','W0');A0=P15.item_arrays(W0A,ids,tok['A'],dec,'q');B0=P15.item_arrays(W0B,ids,tok['B'],dec,'q');g0=gold_lp(W0A)
    where={'ALLQ128':o930,'M3W8Q128':o930,'M3qt05W16Q128':o1080,'M3qtA2W16Q128':o1080,'M3OA2W16Q128':o1160};V={}
    for ar,d in where.items():
        rA=ld(d,'A',f'{ar}_n65');A=P15.item_arrays(rA,ids,tok['A'],dec,'q');B=P15.item_arrays(ld(d,'B',f'{ar}_n65'),ids,tok['B'],dec,'q')
        V[ar]=dict(rr=A['rr_branch'],rr_impr=A0['rr_branch']-A['rr_branch'],gold_lp=gold_lp(rA)-g0,dA=A['A_first']-A0['A_first'],dAB=(A['A_first']-A0['A_first'])-(B['A_first']-B0['A_first']))
    can={}
    for d,f,names in ((o1080,'e57_qa_rules_canary.json',['M3qt05W16Q128','M3qtA2W16Q128']),(o930,'e57_qa_ext_canary.json',['ALLQ128','M3W8Q128']),(o1160,'e57_qa_online_canary.json',['M3OA2W16Q128'])):
        c=json.load(open(d/f))
        for ar in names:
            for part,bs in (('tune',('8','10','12')),('hold',('5','9'))):
                last=[c[ar][b][max(c[ar][b],key=int)] for b in bs];can.setdefault(ar,{})[part]=dict(gen_nll_rise=float(np.mean([x['gen_nll_rise'] for x in last])),tail_nll_rise=float(np.mean([x['tail_nll_rise'] for x in last])))
    W=json.load(open(o1160/'e57_pqa6_writes.json'));sh={g:W[g]['M3OA2W16Q128'].get('online_share_mean') for g in W}
    res={'n_items':len(ids),'canary_n65':can,'online_rule_share_by_bundle':sh,'parts':{}}
    for part in ('tune','hold'):
        m=np.array([sp[i]==part for i in ids]);res['parts'][part]={'arms':{ar:{k:ci(v[m]) for k,v in V[ar].items()} for ar in V},'paired':{}}
        for ref in ('M3qtA2W16Q128','ALLQ128'):
            res['parts'][part]['paired'][f'online_minus_{ref}']={k:ci((V['M3OA2W16Q128'][k]-V[ref][k])[m]) for k in ('rr_impr','gold_lp','dA','dAB')}
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
