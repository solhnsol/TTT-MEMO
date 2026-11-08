"""E57 online g' ABS2 write arms on the QA 65-version stream (bundles 5/8/9/10/12; tune 8/10/12 = 40 items, hold-out 5/9 = 33 items) and the news stream A (v2A frames, tune / hold-out by the frozen event split) (new file; CPU; job 1230, GPM thread, Hans 11/08): QA arms ALLQ128 + M3 w=8 (job 930), TOP5 + W0-ABS2 (job 1080), online g ABS2 (job 1160), online g' ABS2 (job 1230); news arms ALLQ128, M3 w=8, TOP5 (NEWSqt05W16), W0-ABS2 (NEWSqtA2W16) of the earlier reads and the online g ABS2 (NEWSOA2W16) / online g' ABS2 (NEWSOS2W16) of job 1230. Per arm and part: branch-token rank ratio, its improvement over W0 (QA: also gold log-prob gain, first-token dA, dA - dB), canary rise, online selected-row share; paired differences online g' minus every other arm (item / frame level bootstrap, 2000 draws, 95% CI).
  python3 e57_online_gprime_real_report.py --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder'
def ci(x,B=2000,seed=1):
    x=np.asarray(x,float);x=x[~np.isnan(x)];ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def qa():
    import e57_pqa15_run as P15,e57_pqa2_stream as S2
    tk=S2.tokenizer();dec=lambda t:tk.decode(t);o930=EXP/'e57_public_qa_n65';o1080=EXP/'e57_public_qa_n65_rules';o1160=EXP/'e57_public_qa_n65_online';o1230=EXP/'e57_public_qa_n65_gprime'
    sp={r['item_id']:r['split'] for r in json.load(open(EXP/'e57_qa_split.json'))['items']};meta=json.load(open(o930/'e57_pqa15_read_meta.json'));ids=meta['item_ids'];tok=json.load(open(o930/'e57_pqa15_tokens.json'))
    ld=lambda d,c,l:json.load(open(d/f'E57_pqa15_rows_{c}_{l}.json'))
    def gold_lp(rows):
        byid={r['id']:r for r in rows};return np.array([P15.content_lps(byid[i]['cand_tok_lp'],tok['A'][i],dec)[0][0].sum() for i in ids])
    W0A=ld(o930,'A','W0');W0B=ld(o930,'B','W0');A0=P15.item_arrays(W0A,ids,tok['A'],dec,'q');B0=P15.item_arrays(W0B,ids,tok['B'],dec,'q');g0=gold_lp(W0A)
    where={'ALLQ128':o930,'M3W8Q128':o930,'M3qt05W16Q128':o1080,'M3qtA2W16Q128':o1080,'M3OA2W16Q128':o1160,'M3OS2W16Q128':o1230};V={}
    for ar,d in where.items():
        rA=ld(d,'A',f'{ar}_n65');A=P15.item_arrays(rA,ids,tok['A'],dec,'q');B=P15.item_arrays(ld(d,'B',f'{ar}_n65'),ids,tok['B'],dec,'q')
        V[ar]=dict(rr=A['rr_branch'],rr_impr=A0['rr_branch']-A['rr_branch'],gold_lp=gold_lp(rA)-g0,dA=A['A_first']-A0['A_first'],dAB=(A['A_first']-A0['A_first'])-(B['A_first']-B0['A_first']))
    can={}
    for d,f,names in ((o1080,'e57_qa_rules_canary.json',['M3qt05W16Q128','M3qtA2W16Q128']),(o930,'e57_qa_ext_canary.json',['ALLQ128','M3W8Q128']),(o1160,'e57_qa_online_canary.json',['M3OA2W16Q128']),(o1230,'e57_qa_online_canary.json',['M3OS2W16Q128'])):
        c=json.load(open(d/f))
        for ar in names:
            for part,bs in (('tune',('8','10','12')),('hold',('5','9'))):
                last=[c[ar][b][max(c[ar][b],key=int)] for b in bs];can.setdefault(ar,{})[part]=dict(gen_nll_rise=float(np.mean([x['gen_nll_rise'] for x in last])),tail_nll_rise=float(np.mean([x['tail_nll_rise'] for x in last])))
    sh={}
    for nm,d in (('M3OA2W16Q128',o1160),('M3OS2W16Q128',o1230)):
        W=json.load(open(d/'e57_pqa6_writes.json'));sh[nm]={g:W[g][nm].get('online_share_mean') for g in W}
    res={'n_items':len(ids),'canary_n65':can,'online_selected_share_by_bundle':sh,'parts':{}}
    for part in ('tune','hold'):
        m=np.array([sp[i]==part for i in ids]);res['parts'][part]={'arms':{ar:{k:ci(v[m]) for k,v in V[ar].items()} for ar in V},'paired':{}}
        for ref in V:
            if ref!='M3OS2W16Q128':res['parts'][part]['paired'][f'onlineGp_minus_{ref}']={k:ci((V['M3OS2W16Q128'][k]-V[ref][k])[m]) for k in ('rr_impr','gold_lp','dA','dAB')}
    return res
def news():
    import e57_news_tokread as NT
    T1=EXP/'e57_news_online_gprime';new=json.load(open(T1/'read/e57_news_tokread.json'));rul=json.load(open(EXP/'e57_news_rules/read/e57_news_tokread.json'));old=json.load(open(EXP/'e57_news_tokread/e57_news_tokread.json'));sp=json.load(open(EXP/'e57_news_split.json'));s='v2A'
    meta=new['meta'][s];assert rul['meta'][s]==meta or [m['id'] for m in rul['meta'][s]]==[m['id'] for m in meta]
    F=lambda tl:np.array([NT.frame_metrics(t,mm,'all')['rr_branch'] for t,mm in zip(tl,meta)])
    arms={'ALLQ128':old['tokens'][s]['ALLQ128'],'M3W8Q128':old['tokens'][s]['M3W8Q128'] if 'M3W8Q128' in old['tokens'][s] else rul['tokens'][s]['M3W8Q128@c279'],'NEWSqt05W16':rul['tokens'][s]['NEWSqt05W16@c279'],'NEWSqtA2W16':rul['tokens'][s]['NEWSqtA2W16@c279'],'NEWSOA2W16':new['tokens'][s]['NEWSOA2W16@c279'],'NEWSOS2W16':new['tokens'][s]['NEWSOS2W16@c279']}
    R={a:F(t) for a,t in arms.items()};R0=F(new['tokens'][s]['W0']);res={'arms_labels':list(arms),'parts':{},'canary_rise_last':{},'online_selected_share':{}}
    for a in ('NEWSOS2W16','NEWSOA2W16'):
        d=json.load(open(T1/f'E57_news_{a}C.arms.json'));tr=d['trajectory'][-1]['canary'];cw=d['canary_w0'];res['canary_rise_last'][a]=dict(gen_nll_rise=tr['_gen_nll']-cw['_gen_nll'],tail_nll_rise=tr['_tail_nll']-cw['_tail_nll']);res['online_selected_share'][a]=d.get('online_share_mean')
    for part in ('tune','hold'):
        ev=set(sp['events_'+part]);m=np.array([it['event'] in ev for it in meta]);res['parts'][part]={'n_frames':int(m.sum()),'arms':{a:dict(rr_branch=ci(R[a][m]),rr_impr_vs_W0=ci((R0-R[a])[m])) for a in R},'paired':{f'onlineGp_minus_{b}':dict(rr_impr=ci((R[b]-R['NEWSOS2W16'])[m])) for b in R if b!='NEWSOS2W16'}}
    return res
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:x=ci([1,2,3]);assert abs(x['mean']-2)<1e-9;print('e57_online_gprime_real_report selftest ok');return
    res={'definitions':__doc__.split('\n')[0],'qa':qa(),'news_v2A':news()};pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out)
if __name__=='__main__':main()
