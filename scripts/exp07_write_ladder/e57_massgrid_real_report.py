"""E57 mass-matched top-share grid on the news stream A and the QA 65-version stream (new file; CPU; jobs 1240 / 1241, GPM thread, Hans 11/08 '점유율 다시 찾기'): arms with rows fixed by the W0 g128 rank, ALLQ128 on every other row: news NEWSqt10W8 (top 10%, w=8, extra mass .70), NEWSqt15W6 (15%, w=6, .75), NEWSqb_TOP30W3p5 (30%, w=3.5, .75); QA M3qt10W8Q128, M3qb_TOP30W3p5Q128 (bundles 5/8/9/10/12; tune 8/10/12, hold-out 5/9). Reused: ALLQ128, M3 w=8 (30%, mass 2.1), TOP5 w=16 (.75), W0-ABS2 w=16, QA gain-mass (job 1110, n = 65). Per arm and part: branch-token rank ratio and its improvement over W0 (QA: gold log-prob gain, first-token dA, dA - dB), canary rise, selected-row share; paired differences of every new arm minus ALLQ128 / M3 w=8 / TOP5 (improvement difference, positive = the new arm better). QA / news have no separate out-of-window reading.
  python3 e57_massgrid_real_report.py --source news|qa --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder'
def ci(x,B=2000,seed=1):
    x=np.asarray(x,float);x=x[~np.isnan(x)];ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
NEWS_NEW=('NEWSqt10W8','NEWSqt15W6','NEWSqb_TOP30W3p5');QA_NEW=('M3qt10W8Q128','M3qb_TOP30W3p5Q128')
def news():
    import e57_news_tokread as NT
    T1=EXP/'e57_news_massgrid';new=json.load(open(T1/'read/e57_news_tokread.json'));rul=json.load(open(EXP/'e57_news_rules/read/e57_news_tokread.json'));old=json.load(open(EXP/'e57_news_tokread/e57_news_tokread.json'));sp=json.load(open(EXP/'e57_news_split.json'));s='v2A';meta=new['meta'][s]
    F=lambda tl:np.array([NT.frame_metrics(t,mm,'all')['rr_branch'] for t,mm in zip(tl,meta)])
    tok={'ALLQ128':old['tokens'][s]['ALLQ128'],'M3W8Q128':rul['tokens'][s]['M3W8Q128@c279'],'NEWSqt05W16':rul['tokens'][s]['NEWSqt05W16@c279'],'NEWSqtA2W16':rul['tokens'][s]['NEWSqtA2W16@c279']};tok.update({a:new['tokens'][s][f'{a}@c279'] for a in NEWS_NEW})
    R={a:F(t) for a,t in tok.items()};R0=F(new['tokens'][s]['W0']);res={'canary_rise_last':{},'mean_selected_share':{},'parts':{}}
    msh=json.load(open(EXP/'e57_news_m3/news_rules_masks_share.json'));res['mean_selected_share']={'NEWSqt10W8':msh['qt10'],'NEWSqt15W6':msh['qt15'],'NEWSqb_TOP30W3p5':msh['qb_TOP30'],'NEWSqt05W16':msh['qt05'],'NEWSqtA2W16':msh['qtA2'],'M3W8Q128':msh['qb_TOP30']}
    for a in NEWS_NEW:
        d=json.load(open(T1/f'E57_news_{a}C.arms.json'));tr=d['trajectory'][-1]['canary'];cw=d['canary_w0'];res['canary_rise_last'][a]=dict(gen_nll_rise=tr['_gen_nll']-cw['_gen_nll'],tail_nll_rise=tr['_tail_nll']-cw['_tail_nll'])
    for part in ('tune','hold'):
        ev=set(sp['events_'+part]);m=np.array([it['event'] in ev for it in meta]);res['parts'][part]={'n_frames':int(m.sum()),'arms':{a:dict(rr_branch=ci(R[a][m]),rr_impr_vs_W0=ci((R0-R[a])[m])) for a in R},'paired':{f'{a}_minus_{b}':ci((R[b]-R[a])[m]) for a in NEWS_NEW for b in ('ALLQ128','M3W8Q128','NEWSqt05W16','NEWSqtA2W16')}}
    return res
def qa():
    import e57_pqa15_run as P15,e57_pqa2_stream as S2
    tk=S2.tokenizer();dec=lambda t:tk.decode(t);o930=EXP/'e57_public_qa_n65';o1080=EXP/'e57_public_qa_n65_rules';o1110=EXP/'e57_public_qa_n65_gm';o1241=EXP/'e57_public_qa_n65_massgrid'
    sp={r['item_id']:r['split'] for r in json.load(open(EXP/'e57_qa_split.json'))['items']};meta=json.load(open(o930/'e57_pqa15_read_meta.json'));ids=meta['item_ids'];tok=json.load(open(o930/'e57_pqa15_tokens.json'))
    ld=lambda d,c,l:json.load(open(d/f'E57_pqa15_rows_{c}_{l}.json'))
    def gold_lp(rows):
        byid={r['id']:r for r in rows};return np.array([P15.content_lps(byid[i]['cand_tok_lp'],tok['A'][i],dec)[0][0].sum() for i in ids])
    W0A=ld(o930,'A','W0');W0B=ld(o930,'B','W0');A0=P15.item_arrays(W0A,ids,tok['A'],dec,'q');B0=P15.item_arrays(W0B,ids,tok['B'],dec,'q');g0=gold_lp(W0A)
    where={'ALLQ128':o930,'M3W8Q128':o930,'M3qt05W16Q128':o1080,'M3qtA2W16Q128':o1080,'M3qg50W16Q128':o1110,'M3qg70W16Q128':o1110,'M3qg85W16Q128':o1110,'M3qt10W8Q128':o1241,'M3qb_TOP30W3p5Q128':o1241};V={}
    for ar,d in where.items():
        try:rA=ld(d,'A',f'{ar}_n65');A=P15.item_arrays(rA,ids,tok['A'],dec,'q');B=P15.item_arrays(ld(d,'B',f'{ar}_n65'),ids,tok['B'],dec,'q')
        except FileNotFoundError:continue
        V[ar]=dict(rr=A['rr_branch'],rr_impr=A0['rr_branch']-A['rr_branch'],gold_lp=gold_lp(rA)-g0,dA=A['A_first']-A0['A_first'],dAB=(A['A_first']-A0['A_first'])-(B['A_first']-B0['A_first']))
    can={}
    for d,f,names in ((o1080,'e57_qa_rules_canary.json',['M3qt05W16Q128','M3qtA2W16Q128']),(o930,'e57_qa_ext_canary.json',['ALLQ128','M3W8Q128']),(o1241,'e57_qa_massgrid_canary.json',list(QA_NEW))):
        c=json.load(open(d/f))
        for ar in names:
            for part,bs in (('tune',('8','10','12')),('hold',('5','9'))):
                last=[c[ar][b][max(c[ar][b],key=int)] for b in bs];can.setdefault(ar,{})[part]=dict(gen_nll_rise=float(np.mean([x['gen_nll_rise'] for x in last])),tail_nll_rise=float(np.mean([x['tail_nll_rise'] for x in last])))
    msh=json.load(open(o930/'qa_rules_masks_share.json'));res={'canary_n65':can,'mean_selected_share_tune':msh['mean_share_tune'],'mean_selected_share_hold':msh['mean_share_hold'],'parts':{}}
    for part in ('tune','hold'):
        m=np.array([sp[i]==part for i in ids]);res['parts'][part]={'arms':{ar:{k:ci(v[m]) for k,v in V[ar].items()} for ar in V},'paired':{}}
        for a in QA_NEW:
            for b in ('ALLQ128','M3W8Q128','M3qt05W16Q128','M3qtA2W16Q128'):res['parts'][part]['paired'][f'{a}_minus_{b}']={k:ci((V[a][k]-V[b][k])[m]) for k in ('rr_impr','gold_lp','dA','dAB')}
    return res
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:x=ci([1,2,3]);assert abs(x['mean']-2)<1e-9;print('e57_massgrid_real_report selftest ok');return
    res={'definitions':__doc__.split('\n')[0],a.source:{'news':news,'qa':qa}[a.source]()};pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out)
if __name__=='__main__':main()
