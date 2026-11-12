"""E57 section 10 (6.1) CPU recount for the lead30 design (lead variant = first ~350 tokens of every article, 30 events, fixed-order cyclic SPREAD without filler, CLUSTER as contrast). CPU only, no model, no GPU, no download; reuses titles.pkl / clusters.pkl and the shards of the job-202 run.
Report <out>/e57_news10_report.md (+ .json):
  (a) can 30 events be drawn under the rules (independent articles >= 8, domains >= 8, exclusions, near-duplicate removal, one article per domain, >= 1 / >= 3 / >= 5 lead targets with c >= 3, f >= 2), extrapolated from the seeded sample to all title-stage clusters; target count n for random 30-event draws (20 draws: mean, min, max) and for the 30 richest events (selection-biased, shown as an upper bound); n for targets with c in {3,4}.
  (b) number of facts that occur in 3-4 different articles: by c (3, 4, 5, 6+), by type, by number of distinct sentence forms f (f = 1 means the same form in every article; the bank rule needs f >= 2; counted here too), per event.
  (c) M4' for lead30: CLUSTER and fixed-order SPREAD without filler, (i)/(ii)/(iii) decomposition, gaps, round length, updates at 128 exposures, 20 seeded event draws; once with all targets and once with c in {3,4} only.
  plus M5' on lead targets with c in {3,4}: >= 10 / >= 15 middle candidates (c' >= 2, |c-c'| <= 2).
PROXIES as e57_news.py (regex facts, title-token clusters, whole-article candidate exposures). The text stage is redone once with the f >= 1 bank rule (cache analyze_cache_f1.pkl; the f >= 2 targets are a subset, so the rule f >= 2 is applied by filtering)."""
import argparse,json,sys,math,random,pickle,pathlib,time,collections
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
import e57_news96 as Q
def with_f(res,fmin):
    out=[]
    for r in res:
        if not r['ok']:out.append(r);continue
        r2=dict(r);r2['targets']=[t for t in r['targets'] if t['f']>=fmin];out.append(r2)
    return out
def quant(v,q):
    v=sorted(v);return v[min(len(v)-1,int(q*(len(v)-1)))] if v else None
def m5_lead(res,cset,ks=(10,15)):
    n=0;cnt={k:0 for k in ks};strict=0;rx=0;dist=collections.Counter()
    for r in res:
        if not r['ok']:continue
        V=N.variant(r,'lead')
        for t in V['targets']:
            if t['c_v'] not in cset:continue
            cn=r['pool_counts'].get(f"{t['ty']}|{t['key']}",[]);c=t['c_v'];m=sum(1 for x in cn if x>=2 and abs(x-c)<=2);n+=1
            for k in ks:cnt[k]+=m>=k
            strict+=sum(1 for x in cn if abs(x-c)<=1)>=15;rx+=len(cn)>=15;dist[min(m,40)//5*5]+=1
    return dict(targets=n,**{f'middle_ge{k}':cnt[k]/max(1,n) for k in ks},strict_pm1_ge15=strict/max(1,n),any_ge15=rx/max(1,n),middle_count_hist={str(k):v for k,v in sorted(dist.items())},
        note="candidate exposures c' are counted over whole articles while the answer's c is the lead-variant count; the 10.3 rule asks middle >= 10 for c in {3,4} with fulfilment >= 0.9")
def run(a):
    out=pathlib.Path(a.out);tp=out/'titles.pkl';cp=out/'clusters.pkl';assert tp.exists() and cp.exists(),('job-202 stage files missing in',str(out))
    a.min_f=1;res1,ncl=Q.get_results(a,tp,cp);res=with_f(res1,2);ok=[r for r in res if r['ok']];ok1=[r for r in res1 if r['ok']]
    M=dict(n_title_clusters=ncl,n_sampled=len(res1),n_pass_independence=len(ok),extrapolated_pass=int(round(len(ok)/max(1,len(res1))*ncl)))
    lead=[(r,N.variant(r,'lead')) for r in ok];full=[(r,N.variant(r,'full')) for r in ok]
    pool=[(r,f,l) for (r,f),(r2,l) in zip(full,lead) if l['targets']]              # events with >= 1 lead target (c >= 3 in the lead variant, f >= 2)
    per=[len(l['targets']) for _,_,l in pool];ex=lambda k:int(round(sum(1 for x in per if x>=k)/max(1,len(res1))*ncl))
    M['a']=dict(pool_ge1=len(per),pool_ge3=sum(x>=3 for x in per),pool_ge5=sum(x>=5 for x in per),extrapolated_ge1=ex(1),extrapolated_ge3=ex(3),extrapolated_ge5=ex(5),targets_total=sum(per),targets_per_event=dict(mean=sum(per)/max(1,len(per)),median=quant(per,0.5),p10=quant(per,0.1),p90=quant(per,0.9),max=max(per) if per else None),
        pool_can_give_30=bool(len(per)>=30))
    draws=[]
    for d in range(20):
        if len(pool)>=30:
            s=random.Random(a.seed+500+d).sample(range(len(pool)),30);draws.append(sum(per[i] for i in s))
    M['a']['n_random30']=dict(mean=sum(draws)/len(draws),min=min(draws),max=max(draws),draws=len(draws)) if draws else None
    M['a']['n_richest30_upper_bound']=sum(sorted(per,reverse=True)[:30]) if len(per)>=30 else None
    cs=collections.Counter();bytype=collections.defaultdict(collections.Counter);bf=collections.Counter()
    for _,_,l in pool:
        for t in l['targets']:cs[min(t['c_v'],6)]+=1;bytype[min(t['c_v'],6)][t['ty']]+=1
    M['a']['c_hist_all_pool']={str(k):v for k,v in sorted(cs.items())}
    n34=sum(cs[3]+cs[4] for _ in [0]);M['a']['targets_c34']=n34
    if len(pool)>=30:
        d34=[]
        for d in range(20):
            s=random.Random(a.seed+500+d).sample(range(len(pool)),30);d34.append(sum(1 for i in s for t in pool[i][2]['targets'] if t['c_v'] in(3,4)))
        M['a']['n_c34_random30']=dict(mean=sum(d34)/len(d34),min=min(d34),max=max(d34))
    # (b) facts in 3-4 articles, by f (all f>=1 targets of the f1 run, lead variant)
    lead1=[N.variant(r,'lead') for r in ok1];tab=collections.defaultdict(lambda:collections.Counter())
    # f of a lead target: from the f1 run's target list (same keys)
    for r,V in zip(ok1,lead1):
        fof={(t['ty'],t['key']):t['f'] for t in r['targets']}
        for t in V['targets']:
            c=min(t['c_v'],6);f=fof[(t['ty'],t['key'])];tab[c]['f1' if f==1 else ('f2' if f==2 else 'f3+')]+=1;tab[c][t['ty']]+=1;tab[c]['all']+=1
    M['b']=dict(by_c={str(c):dict(v) for c,v in sorted(tab.items())},note='counts over the sampled clusters that pass the independence rules (not extrapolated); f1 = the same slot-masked form in every article (not allowed by the bank rule), f2 / f3+ = 2 / >= 3 distinct forms',
        c34_f1=sum(tab[c]['f1'] for c in(3,4)),c34_f2plus=sum(tab[c]['f2']+tab[c]['f3+'] for c in(3,4)),c34_all=sum(tab[c]['all'] for c in(3,4)))
    # (c) M4' lead30
    n30=min(30,len(pool));M['c_events']=n30      # fewer than 30 events available: the layout is measured with all of them and the report says so
    a.cset=None;M['c_all']=Q.config(pool,2,'lead',n30,False,a,a.draws) if n30>=2 else dict(skipped='no events');a.cset={3,4};M['c_c34']=Q.config(pool,2,'lead',n30,False,a,a.draws) if n30>=2 else dict(skipped='no events');a.cset=None
    M['m5_c34']=m5_lead(res,{3,4})
    tpe=M['a']['targets_per_event']['mean'];M['power']={nev:dict(n_targets=int(round(tpe*nev)),table=C.power_table(int(round(tpe*nev)))) for nev in(24,30)}
    json.dump(M,open(out/'e57_news10_report.json','w'),indent=1,default=str);open(out/'e57_news10_report.md','w').write(report(M,a));print(open(out/'e57_news10_report.md').read())
def fm(x,d=3):return Q.fm(x,d)
def report(M,a):
    A=M['a'];f=lambda x:'PASS' if x else 'FAIL'
    L=['# E57 section 10 CPU recount: lead30 design (regex proxy)','',f"Same cluster sample and seeds as job 202 (seed {a.seed}); lead variant = first ~350 tokens per article; bank rule c >= 3, f >= 2, no conflict (the f >= 1 run is used only for the f = 1 counts).",'',
       '## (a) can 30 events be drawn under the rules, and how many targets n','',f"- title-stage clusters {M['n_title_clusters']}; sampled for the text stage {M['n_sampled']}; pass independence (>= 8 articles and >= 8 domains after near-duplicate removal, one article per domain): {M['n_pass_independence']} (extrapolated to all title-stage clusters: {M['extrapolated_pass']})",
       f"- events with >= 1 lead target (c >= 3 in the lead variant): {A['pool_ge1']} in the sample (extrapolated {A['extrapolated_ge1']}); >= 3 targets: {A['pool_ge3']} ({A['extrapolated_ge3']}); >= 5 targets: {A['pool_ge5']} ({A['extrapolated_ge5']}) -> **30 events available in the sample: {f(A['pool_can_give_30'])}**; the extrapolation counts clusters of the whole title stage, not only the sample",
       f"- targets in the pool {A['targets_total']}; per event mean {A['targets_per_event']['mean']:.2f}, median {A['targets_per_event']['median']}, p10 {A['targets_per_event']['p10']}, p90 {A['targets_per_event']['p90']}, max {A['targets_per_event']['max']}"]
    if A['n_random30']:L.append(f"- n for 30 random events (20 draws): mean {A['n_random30']['mean']:.1f}, min {A['n_random30']['min']}, max {A['n_random30']['max']}; the 30 richest events (upper bound, selection-biased): {A['n_richest30_upper_bound']}")
    if A.get('n_c34_random30'):L.append(f"- targets with c in {{3,4}} in 30 random events: mean {A['n_c34_random30']['mean']:.1f} (min {A['n_c34_random30']['min']}, max {A['n_c34_random30']['max']}); in the whole pool {A['targets_c34']}")
    L+=['',f"- c histogram of the pool targets (6 = 6 or more): {A['c_hist_all_pool']}",'',
        '## (b) facts that occur in different articles (3-4 and more), lead variant','','| c | all | f = 1 (same form) | f = 2 | f >= 3 | NUM | DATE | NAME |','|---|---|---|---|---|---|---|---|']
    for c,v in M['b']['by_c'].items():L.append(f"| {c} | {v.get('all',0)} | {v.get('f1',0)} | {v.get('f2',0)} | {v.get('f3+',0)} | {v.get('NUM',0)} | {v.get('DATE',0)} | {v.get('NAME',0)} |")
    L+=['',f"- c in {{3,4}}: {M['b']['c34_all']} targets, of them f = 1 {M['b']['c34_f1']}, f >= 2 {M['b']['c34_f2plus']}. {M['b']['note']}",'',
        f"## (c) M4' lead30: CLUSTER and fixed-order SPREAD, no filler (criterion CLUSTER (i)+(ii) >= 0.5 and SPREAD (i) <= 0.2); events used: {M['c_events']}"+(" (FEWER THAN 30: all available events, the 30-event result is not measured)" if M['c_events']<30 else ""),'','| targets | draws | CLUSTER (i) | CLUSTER (i)+(ii) | SPREAD (i) | SPREAD (ii) | SPREAD (iii) | SPREAD any same-event article in window | gap min (worst) | gap 5th pct | round chunks | updates @128 exposures | n targets | mean c | M4\' |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for nm,k in(('all targets','c_all'),('c in {3,4}','c_c34')):
        r=M[k]
        if 'skipped' in r:L.append(f"| {nm} | skipped: {r['skipped']} |");continue
        L.append(f"| {nm} | {r['draws']} | {fm(r['CLUSTER']['i'])} | {fm(r['CLUSTER']['i_or_ii'])} | {fm(r['SPREAD']['i'])} | {fm(r['SPREAD']['ii'])} | {fm(r['SPREAD']['iii'])} | {fm(r['SPREAD']['event_any'])} | {fm(r['gap_min_worst'],0)} | {fm(r['gap_p5'],0)} | {fm(r['round_chunks'],1)} | {fm(r['updates128'],0)} | {fm(r['n_targets'],1)} | {fm(r['mean_c'],2)} | **{f(r['passed'])}** |")
    m=M['m5_c34'];L+=['',"## M5' on lead targets with c in {3,4}",'',f"- targets {m['targets']}: middle candidates (c' >= 2, |c-c'| <= 2) >= 10: {m['middle_ge10']:.3f}; >= 15: {m['middle_ge15']:.3f}; strict (+-1) >= 15: {m['strict_pm1_ge15']:.3f}; any >= 15: {m['any_ge15']:.3f}; histogram of the number of middle candidates (bin start): {m['middle_count_hist']}",'- '+m['note'],'',
        '## power with the measured targets per event (E56 G1 formula)','','| events | n targets | DEFF | n_eff | theta | power @0.03 | @0.10 | @0.15 |','|---|---|---|---|---|---|---|---|']
    for nev,x in M['power'].items():
        for p in x['table']:L.append(f"| {nev} | {x['n_targets']} | {p['deff']} | {p['n_eff']:.1f} | {p['theta']:.3f} | {p['power_0.03']:.2f} | {p['power_0.1']:.2f} | {p['power_0.15']:.2f} |")
    return '\n'.join(L)+'\n'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw')
    ap.add_argument('--min-articles',type=int,default=8);ap.add_argument('--min-domains',type=int,default=8);ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-c',type=int,default=3)
    ap.add_argument('--max-clusters',type=int,default=3000);ap.add_argument('--max-per-event',type=int,default=12);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--draws',type=int,default=20);ap.add_argument('--rerun',action='store_true');a=ap.parse_args();run(a)
if __name__=='__main__':main()
