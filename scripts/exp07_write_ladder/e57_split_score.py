"""E57 split scoring of the rule arms (new file; CPU; GPM thread, Hans 11/05): the branch-token rank ratio (rr = rank of the gold at the first differing token among its candidates / number of candidates; lower is better) and its improvement over W0 (W0 - arm), for the TUNE and HOLD-OUT parts, per source:
  QA   : the 5 bundles of the 65-version stream, tune = bundles 8 / 10 / 12 (40 items), hold-out = 5 / 9 (33 items), rows A of the read points n (default 9 / 33 / 65), item-level bootstrap; arms of job 1080 (e57_public_qa_n65_rules) + ALLQ128 / M3 w=8 of job 930 (e57_public_qa_n65); canary rise (reference only) per part = mean over its bundles
  news : stream A end state (chunk 279), question sets v1 (150) and v2A (74), tune / hold-out EVENTS of e57_news_split.json; arms of job 1090 (e57_news_rules/read) + M3 w=8 + ALLQ128 / NATQ of the stored tokread
  syn  : worlds of job 1070 (e57_syn_rules_s<seed>/e57_syn_tokread_s<seed>.json), end state a = 128, fact-level bootstrap; the out-of-window read (inline read.json) as second measure; tune worlds s62 s71 s72
Families: ABS tau 2 / 4 / 6 (w = 16), fixed top 5% (w = 16), M3 top 30% (w = 8), ALLQ (w = 1). Output: per source and part, per family: mean rr, mean improvement, CI; then the equal-weight mean over the sources (QA, news, synthetic) of the improvement. Reference values only; no verdicts.
  python3 e57_split_score.py --qa --news --syn 62:v3_16_1,71:v3_4_1,72:v3_32_1 --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder'
FAM={'ABS2':dict(qa='M3qtA2W16Q128',news='NEWSqtA2W16',syn='M3TA2W16N05'),'ABS4':dict(qa='M3qtA4W16Q128',news='NEWSqtA4W16',syn='M3TA4W16N05'),'ABS6':dict(qa='M3qtA6W16Q128',news='NEWSqtA6W16',syn='M3TA6W16N05'),'TOP5':dict(qa='M3qt05W16Q128',news='NEWSqt05W16',syn='M3T05W16N05'),'M3_TOP30_W8':dict(qa='M3W8Q128',news='M3W8Q128',syn='M3W8N05'),'ALLQ':dict(qa='ALLQ128',news='ALLQ128',syn='M3W1N05')}
def boot(x,B=2000,seed=1):
    x=np.asarray(x,np.float64);x=x[~np.isnan(x)]
    if not len(x):return dict(mean=None,ci95=None,n=0)
    ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def qa(ns=(9,33,65),newdir='e57_public_qa_n65_rules',olddir='e57_public_qa_n65'):
    import e57_pqa15_run as P15,e57_pqa2_stream as S2
    tk=S2.tokenizer();dec=lambda t:tk.decode(t);new=EXP/newdir;old=EXP/olddir;meta=json.load(open(old/'e57_pqa15_read_meta.json'));ids=meta['item_ids'];tok=json.load(open(old/'e57_pqa15_tokens.json'))
    sp={r['item_id']:r['split'] for r in json.load(open(EXP/'e57_qa_split.json'))['items']};part=np.array([sp[i] for i in ids])
    rr=lambda d,l:P15.item_arrays(json.load(open(d/f'E57_pqa15_rows_A_{l}.json')),ids,tok['A'],dec,'q')['rr_branch']
    w0=rr(old,'W0');res={};
    for fam,names in FAM.items():
        arm=names['qa'];d=new if (new/f'E57_pqa15_rows_A_{arm}_n{ns[0]}.json').exists() else old
        if not (d/f'E57_pqa15_rows_A_{arm}_n{ns[0]}.json').exists():continue
        for n in ns:
            x=rr(d,f'{arm}_n{n}')
            for p in ('tune','hold'):
                m=part==p;res.setdefault(fam,{}).setdefault(p,{})[f'n{n}']=dict(rr=boot(x[m]),improvement=boot((w0-x)[m]))
    can={}
    for d_ in (new,old):
        f=d_/('e57_qa_rules_canary.json' if d_==new else 'e57_qa_ext_canary.json')
        if f.exists():
            for arm,v in json.load(open(f)).items():
                for p,bs in (('tune',('8','10','12')),('hold',('5','9'))):
                    last=[v[b][max(v[b],key=int)] for b in bs if b in v]
                    if last:can.setdefault(arm,{})[p]=dict(gen_nll_rise=float(np.mean([x['gen_nll_rise'] for x in last])),tail_nll_rise=float(np.mean([x['tail_nll_rise'] for x in last])))
    return dict(rows=res,canary_reference=can)
def news(newf='e57_news_rules/read/e57_news_tokread.json',oldf='e57_news_tokread/e57_news_tokread.json',sets=('v2A','v1')):
    import e57_news_tokread as NT
    new=json.load(open(EXP/newf));old=json.load(open(EXP/oldf));sp=json.load(open(EXP/'e57_news_split.json'));ev={'tune':set(sp['events_tune']),'hold':set(sp['events_hold'])};res={}
    for s in sets:
        meta=new['meta'][s];T=new['tokens'][s];To=old['tokens'][s];F=lambda tl,m:NT.frame_metrics(tl,m,'all')['rr_branch']
        w0=np.array([F(t,m) for t,m in zip(T['W0'],meta)])
        for fam,names in FAM.items():
            arm=names['news'];lab=f'{arm}@c279' if f'{arm}@c279' in T else (arm if arm in T else None)
            tl=T.get(lab) if lab else To.get(arm)
            if tl is None:continue
            x=np.array([F(t,m) for t,m in zip(tl,meta)])
            for p in ('tune','hold'):
                m=np.array([it['event'] in ev[p] for it in meta]);res.setdefault(s,{}).setdefault(fam,{})[p]=dict(rr=boot(x[m]),improvement=boot((w0-x)[m]))
    return res
def syn(spec):
    import e57_syn_tokread as TR
    res={}
    for x in [x for x in spec.split(',') if x]:
        sd,cell=x.split(':');d=json.load(open(EXP/f'e57_syn_rules_s{sd}/e57_syn_tokread_s{sd}.json'));rj=json.load(open(next(EXP.glob(f'e57_synq/E55_synqrl_*_s{sd}.read.json'))))['meta'];W0=TR.metrics(d['W0'],d['tokids'],rj);facts=sorted(W0);w0=np.array([W0[f]['rank_branch']/16 for f in facts])
        for fam,names in FAM.items():
            lab=f"{names['syn']}@c127"
            if lab not in d['arms']:continue
            M=TR.metrics(d['arms'][lab],d['tokids'],rj);v=np.array([M[f]['rank_branch']/16 for f in facts]);res.setdefault(f's{sd}_{cell}',{})[fam]=dict(rr=boot(v),improvement=boot(w0-v))
    return res
def equal_weight(out):
    """mean over the sources QA (n = last), news (v2A) and synthetic (worlds averaged) of the improvement, per family and part"""
    ew={}
    for fam in FAM:
        for p in ('tune','hold'):
            vals=[]
            q=out.get('qa',{}).get('rows',{}).get(fam,{}).get(p)
            if q:vals.append(q[max(q,key=lambda k:int(k[1:]))]['improvement']['mean'])
            nw=out.get('news',{}).get('v2A',{}).get(fam,{}).get(p)
            if nw:vals.append(nw['improvement']['mean'])
            if p=='tune':
                sv=[w[fam]['improvement']['mean'] for w in out.get('syn',{}).values() if fam in w]
                if sv:vals.append(float(np.mean(sv)))
            if vals:ew.setdefault(fam,{})[p]=dict(n_sources=len(vals),mean_improvement=float(np.mean(vals)))
    return ew
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qa',action='store_true');ap.add_argument('--news',action='store_true');ap.add_argument('--syn',default='');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        b=boot([1,2,3,4]);assert abs(b['mean']-2.5)<1e-9;out={'qa':{'rows':{'ABS2':{'tune':{'n9':{'improvement':{'mean':.1}}}}}},'news':{'v2A':{'ABS2':{'tune':{'improvement':{'mean':.3}}}}},'syn':{'s62':{'ABS2':{'improvement':{'mean':.2}}}}};e=equal_weight(out);assert abs(e['ABS2']['tune']['mean_improvement']-.2)<1e-9 and e['ABS2']['tune']['n_sources']==3;print('e57_split_score selftest ok');return
    out={}
    if a.qa:out['qa']=qa()
    if a.news:out['news']=news()
    if a.syn:out['syn']=syn(a.syn)
    out['equal_weight_over_sources']=equal_weight(out);pathlib.Path(a.out).write_text(json.dumps(out,indent=1,default=float));print('wrote',a.out)
if __name__=='__main__':main()
