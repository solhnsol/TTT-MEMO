"""E57 named-events survey (CPU only, no model, no GPU, NO download, no new clustering): the survey of job 214 (independent articles, targets with c or more independent articles, candidate-pool satisfaction, publication-order stream, A over time) for a FIXED list of clearly identifiable news events, chosen by Hans from the job-219 list. The event set is fixed in this file BEFORE any result (EVENTS below: for each event the title must match ALL of its regular-expression groups; case-insensitive; matching runs over every English title of the job-202 title table, so an event is not limited to one title-stage cluster).
Events: Rashmika-Vijay wedding, Spider-Man Brand New Day, Venezuela-Machado (title mentions Machado; the Venezuela tanker stories are NOT merged into it, they are counted as a side line with their overlap), Mamdani inauguration, Nipah virus, El Paso airspace, HDFC Bank chairman resignation, IDFC First Bank fraud, Hannah Montana 20th anniversary, Mayweather-Pacquiao, Ranji Trophy J&K, NCERT textbooks. Salim Khan health is excluded. No change of the ethics filter: ETH-v1 (e57_freeze.ethics) is shown as a report column only; the event selection rule change (named events instead of the seeded draw) is recorded in E57 section 6 by the design thread.
Per event: titles matched, texts read, after one article per hostname (the longest) and near-duplicate removal (e57_news.analyze, dup Jaccard 0.5): independent articles, independent domains, first / last date, densest 7-day share; targets (S, A) with f >= 2 forms appearing in >= c independent articles for c >= 3, 5, 10, 20, 50 (lead variant: first sentences up to 350 tokens; full-article variant: up to 1500 tokens); candidate-pool satisfaction of those targets (strict: >= 15 same-type candidates with exposure within +-1 of the answer; middle: >= 15 candidates with exposure >= 2 and within +-2; relaxed: >= 15 candidates of any exposure); share of targets with a competing value A' for the same S and whether it sits at another time (median dates >= 2 days apart).
Together: all events with targets once in publication order (TIME, ties seeded shuffle) with CLUSTER / SPREAD references as in job 214: total tokens, chunks (= updates), token distance to the previous occurrence of the same target, share within 8192 tokens before the chunk (rho_win), distinct events per 8192-token block, neighbour switch rate. Report <out>/e57_namedevents/e57_namedevents_report.md (+ .json). PROXIES as in all E57 CPU jobs (regex (S, A), one article per hostname)."""
import argparse,json,sys,math,random,pickle,pathlib,time,collections,re
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
import e57_freeze as FZ
import e57_news10 as Q10
import e57_events as EV
EVENTS=collections.OrderedDict([
 ('Rashmika-Vijay wedding',[r'rashmika',r'vijay',r'wedding|weds|wed\b|marri|engage|knot|nuptial|bride|groom|ceremony']),
 ('Spider-Man Brand New Day',[r'spider-?man',r'brand new day']),
 ('Venezuela-Machado',[r'machado',r'venezuela|maduro|nobel|opposition|trump']),
 ('Mamdani inauguration',[r'mamdani',r'sworn|swear|oath|inaugurat|takes office|take office|sworn-in|first day|bible|quran|koran']),
 ('Nipah virus',[r'nipah']),
 ('El Paso airspace',[r'el paso',r'airspace|flight|faa|drone|laser|closure|closed|no-fly|no fly']),
 ('HDFC Bank chairman resignation',[r'hdfc',r'resign|quits?|steps? down|chairman|chair\b']),
 ('IDFC First Bank fraud',[r'idfc',r'fraud|scam|embezzl|misappropr|cheat|crore|siphon']),
 ('Hannah Montana 20th anniversary',[r'hannah montana',r'20th|twenty|anniversary|reunion|20 years|two decades']),
 ('Mayweather-Pacquiao',[r'mayweather',r'pacquiao']),
 ('Ranji Trophy J&K',[r'ranji',r'j&k|j & k|jammu|kashmir']),
 ('NCERT textbooks',[r'ncert',r'textbook|text book|book|chapter|syllabus|class \d|curriculum'])])
TANKER=[r'venezuela',r'tanker']
CSET=EV.CS
def compile_ev():return collections.OrderedDict((k,[re.compile(x,re.I) for x in v]) for k,v in EVENTS.items())
def match_titles(titles,rxs):
    return [i for i,t in enumerate(titles) if all(rx.search(t) for rx in rxs)]
def pool_sat(r,variant):
    """candidate-pool tiers over the targets of the variant: strict / middle / relaxed shares and the number of targets"""
    V=N.variant(r,variant);st=mi=rl=0;n=0
    for t in V['targets']:
        cnts=r['pool_counts'].get(f"{t['ty']}|{t['key']}",[]);c=t['c_v'];n+=1
        st+= sum(1 for x in cnts if abs(x-c)<=1)>=15;mi+= sum(1 for x in cnts if x>=2 and abs(x-c)<=2)>=15;rl+= len(cnts)>=15
    return dict(n=n,strict=st,middle=mi,relaxed=rl)
def run(a):
    out=pathlib.Path(a.out);tp=out/'titles.pkl';assert tp.exists(),('titles.pkl missing in',str(out))
    fd=out/'e57_namedevents';fd.mkdir(exist_ok=True);rep=fd/'e57_namedevents_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo):',rep);sys.exit(8)
    t0=time.time();D=pickle.load(open(tp,'rb'));pub_date={r:EV.to_ord(d) for r,d in zip(D['rid'],D['date'])};rx=compile_ev()
    matched={k:match_titles(D['title'],v) for k,v in rx.items()};print({k:len(v) for k,v in matched.items()},flush=True)
    tank=set(match_titles(D['title'],[re.compile(x,re.I) for x in TANKER]));mach=set(matched['Venezuela-Machado'])
    # one article per hostname before the near-duplicate removal: the longest text of every hostname (needs the texts)
    allm=sorted({i for v in matched.values() for i in v});got=N.read_texts(D,allm);print(f'  texts read: {len(got)} of {len(allm)}, {time.time()-t0:.0f}s',flush=True)
    jobs=[];names=[];meta=[]
    for k,v in matched.items():
        byh={}
        for i in v:
            r_=D['rid'][i]
            if r_ not in got:continue
            h=D['host'][i]
            if h not in byh or len(got[r_])>len(got[D['rid'][byh[h]]]):byh[h]=i
        ids=sorted(byh.values());names.append(k);meta.append([D['rid'][i] for i in ids])
        jobs.append(dict(title=k,texts=[got[D['rid'][i]] for i in ids],hosts=[D['host'][i] for i in ids],args=dict(dup_jac=a.dup_jac,min_articles=a.min_articles,min_domains=a.min_domains,min_c=a.min_c,min_f=1,detail=True,conflicts=True)))
    if a.workers>1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as p:res=p.map(N.analyze,jobs,chunksize=1)
    else:res=[N.analyze(j) for j in jobs]
    for r,rids in zip(res,meta):
        if r['ok']:r['rids']=[rids[k_] for k_ in r['keep']]
    print(f'  events analysed, {time.time()-t0:.0f}s',flush=True)
    res=Q10.with_f(res,2);rows=[]
    for k,r,v,rids in zip(names,res,matched.values(),meta):
        dates=[pub_date[D['rid'][i]] for i in v]
        row=dict(event=k,titles=len(v),hosts_one_per=len(rids),first=min(D['date'][i] for i in v) if v else None,last=max(D['date'][i] for i in v) if v else None,densest7=BC_dense(dates),ok=r['ok'],n_dedup=r['n_dedup'],n_before=r['n_before'])
        if r['ok']:
            row['independent_domains']=len({x['host'] for x in r['arts']});row['titles_examples']=[D['title'][v[0]][:100]]
            for vn in('lead','full'):
                tc,V=EV.target_counts(r,vn);row[vn]=dict(counts=tc,targets=len(V['targets']),pool=pool_sat(r,vn))
            lt=EV.lead_texts(r);why=FZ.ethics(lt);row['eth_pass']=not why;row['eth_why']=why
        rows.append(row)
    okidx=[i for i,r in enumerate(res) if r['ok'] and EV.target_counts(r,'lead')[1]['targets']]
    comb={}
    if len(okidx)>=2:
        comb['events']=[names[i] for i in okidx]
        for vn in('lead','full'):
            events,tokof,occ,tord,dates=EV.build_set(okidx,res,pub_date,vn,a.seed);occ={kk:vv for kk,vv in occ.items() if len(vv)>=3}
            L=C.layouts(events,random.Random(a.seed),rounds=1)
            comb[vn]=dict(articles=len(tord),n_targets=len(occ),days=(max(dates.values())-min(dates.values()))+1,TIME=dict(EV.occ_stats(tord,tokof,occ),**EV.mixing(tord,tokof)),CLUSTER=dict(EV.occ_stats(L['CLUSTER'],tokof,occ),**EV.mixing(L['CLUSTER'],tokof)),SPREAD=dict(EV.occ_stats(L['SPREAD'],tokof,occ),**EV.mixing(L['SPREAD'],tokof)))
    A_all=EV.a_change(res,okidx,pub_date) if okidx else None
    A_ev={names[i]:EV.a_change(res,[i],pub_date) for i in okidx}
    M=dict(rows=rows,combined=comb,A_all=A_all,A_ev=A_ev,tanker=dict(titles=len(tank),also_machado=len(tank&mach),machado=len(mach)),sec=time.time()-t0)
    json.dump(M,open(fd/'e57_namedevents.json','w'),indent=1,default=str);txt=report(M,a);rep.write_text(txt);print(txt[:7000])
def BC_dense(dates):
    import e57_bigclusters as BC
    return BC.date_stats(dates)['densest7_share'] if dates else None
def report(M,a):
    f=EV.fmt
    L=['# E57 named-events survey (CPU, regex proxy)','',f"PROXIES: (S, A) regex candidates, event = all English titles in the window that match the fixed regular-expression groups of the event (list in e57_namedevents.py, fixed before the data), one article per hostname (longest) then near-duplicate removal, independent articles = what remains. No ethics-filter change (ETH-v1 shown as a column only). Run {M['sec']:.0f} s.",'',
       '## independent articles per event','','| event | matched titles | one per hostname | independent articles | independent domains | first | last | densest 7-day share | ETH-v1 | lead targets (c >= 3) | full targets (c >= 3) |','|---|---|---|---|---|---|---|---|---|---|---|']
    for r in M['rows']:
        L.append(f"| {r['event']} | {r['titles']} | {r['hosts_one_per']} | {r['n_dedup']}{'' if r['ok'] else ' (below the analysis minimum)'} | {r.get('independent_domains','n/a')} | {r['first']} | {r['last']} | {f(r['densest7'])} | {('pass' if r['eth_pass'] else 'REJECT '+'; '.join(r['eth_why'])[:60]) if r['ok'] else 'n/a'} | {r['lead']['targets'] if r['ok'] else 'n/a'} | {r['full']['targets'] if r['ok'] else 'n/a'} |")
    t=M['tanker'];L+=['',f"- side line (not merged): Venezuela + tanker titles {t['titles']}, of which also matching Machado {t['also_machado']}; Machado titles {t['machado']}.",'','## targets with c or more independent articles (f >= 2 forms)','','| event | lead >=3 | >=5 | >=10 | >=20 | >=50 | full >=3 | >=5 | >=10 | >=20 | >=50 |','|---|---|---|---|---|---|---|---|---|---|---|']
    tot={v:{c:0 for c in CSET} for v in('lead','full')}
    for r in M['rows']:
        if not r['ok']:L.append(f"| {r['event']} | "+' | '.join(['n/a']*10)+' |');continue
        L.append(f"| {r['event']} | "+' | '.join(str(r['lead']['counts'][c]) for c in CSET)+' | '+' | '.join(str(r['full']['counts'][c]) for c in CSET)+' |')
        for v in tot:
            for c in CSET:tot[v][c]+=r[v]['counts'][c]
    L.append('| sum | '+' | '.join(str(tot['lead'][c]) for c in CSET)+' | '+' | '.join(str(tot['full'][c]) for c in CSET)+' |')
    L+=['','## candidate-pool satisfaction (share of the targets with c >= 3)','','strict = >= 15 same-type candidates with exposure within +-1 of the answer; middle = >= 15 candidates with exposure >= 2 and within +-2; relaxed = >= 15 candidates of any exposure.','','| event | variant | targets | strict | middle | relaxed |','|---|---|---|---|---|---|']
    pt={v:dict(n=0,strict=0,middle=0,relaxed=0) for v in('lead','full')}
    for r in M['rows']:
        if not r['ok']:continue
        for v in('lead','full'):
            p=r[v]['pool'];L.append(f"| {r['event']} | {v} | {p['n']} | {f(p['strict']/p['n'] if p['n'] else None)} | {f(p['middle']/p['n'] if p['n'] else None)} | {f(p['relaxed']/p['n'] if p['n'] else None)} |")
            for k_ in pt[v]:pt[v][k_]+=p[k_]
    for v in('lead','full'):p=pt[v];L.append(f"| all events | {v} | {p['n']} | {f(p['strict']/p['n'] if p['n'] else None)} | {f(p['middle']/p['n'] if p['n'] else None)} | {f(p['relaxed']/p['n'] if p['n'] else None)} |")
    if M['combined']:
        c=M['combined'];L+=['','## all events with targets once in publication order','',f"Events: {', '.join(c['events'])}. Layouts: TIME = publication date order (ties seeded shuffle); CLUSTER / SPREAD = the same articles with events one after another / round robin; one pass; rho_win = share of repeated occurrences whose previous occurrence lies in the 8192 tokens before their chunk start.",'']
        for vn in('lead','full'):
            s=c[vn];L+=[f"{vn} variant: {s['articles']} articles over {s['days']} days, {s['n_targets']} targets with c >= 3.",'','| layout | total tokens | chunks (= updates) | repeated occurrences | rho_win (prev chunks) | distance to previous occurrence 25 / 50 / 75 / 90 % | share within 8192 | distinct events per 8192 block | effective events | neighbour switch |','|---|---|---|---|---|---|---|---|---|---|']
            for lay in('TIME','CLUSTER','SPREAD'):
                y=s[lay];L.append(f"| {lay} | {y['total_tokens']} | {y['chunks']} | {y['n_repeat_occ']} | {f(y['rho_prev_chunks'])} | "+' / '.join(f(y['dist_q'][q],0) for q in(0.25,0.5,0.75,0.9))+f" | {f(y['dist_within_win'])} | {f(y['distinct_events_mean'],2)} | {f(y['effective_events_mean'],2)} | {f(y['neighbour_switch'])} |")
            L.append('')
    A=M['A_all']
    if A:
        L+=['## does A change over time for the same S?','',f"- all events: targets that exist without the conflict rule (f >= 2): {A['targets_f2_without_conflict_rule']}; with a competing value of the same slot: {A['conflicting']} ({f(A['conflict_share'])}); competitor at a different time (median dates >= 2 days apart): {A['time_shifted']} ({f(A['time_shifted_share_of_conflicting'])} of the conflicting, {f(A['time_shifted_share_of_all'])} of all).",'','| event | targets | conflicting | share | time-shifted | share of conflicting |','|---|---|---|---|---|---|']
        for k,x in M['A_ev'].items():L.append(f"| {k} | {x['targets_f2_without_conflict_rule']} | {x['conflicting']} | {f(x['conflict_share'])} | {x['time_shifted']} | {f(x['time_shifted_share_of_conflicting'])} |")
        L+=['','examples (event title, S, value, competing value, median date numbers, time-shifted):']+[f"- {e[0]} | {e[1]} | {e[2]} | {e[3]} | {e[4]} / {e[5]} | {e[6]}" for e in A['examples']]
    L+=['','Titles and names are shown only to identify events; no article text is written to this report.']
    return '\n'.join(L)+'\n'
def _selftest():
    import tempfile,datetime,pyarrow as pa,pyarrow.parquet as pq
    td=pathlib.Path(tempfile.mkdtemp());rid=[];host=[];title=[];date=[];text=[]
    dd=lambda n:int((datetime.date(2026,1,1)+datetime.timedelta(days=n)).strftime('%Y%m%d'))
    def add(t,d,h,tx):rid.append(f'r{len(rid)}');host.append(h);title.append(t);date.append(d);text.append(tx)
    names=['Maria Gonzalez','Pavel Novak','Helena Duarte','Tomas Albrecht']
    def art(ev,k):
        s=[f'On January {1+k%3}, {names[ev]} said the fund raised $%d million after the {names[(ev+1)%4]} agreement was signed in Springfield.'%(10+ev) if k%4 else f'{names[ev]} announced that the fund raised $%d million in total today.'%(10+ev)]
        return ' '.join(s+[f'Reporter {k} filler sentence number {j} with several varied words about topic {ev} host {k} item {j*k}.' for j in range(25)])
    for k in range(14):add(f'Nipah virus case confirmed update {k}',dd(5+k//7),f'ha{k}',art(0,k))
    for k in range(14):add(f'Mayweather faces Pacquiao rematch talks {k}',dd(20+k//7),f'hb{k}',art(1,k))
    for k in range(6):add(f'Mayweather alone news {k}',dd(25),f'hc{k}',art(1,k))      # not matching (needs both names)
    for k in range(14):add(f'Machado speaks to Venezuela press {k}',dd(30+k//7),f'hd{k}',art(2,k))
    for k in range(5):add(f'Venezuela tanker seized {k}',dd(31),f'he{k}',art(3,k))
    pq.write_table(pa.table(dict(warc_record_id=rid,text=text)),td/'s0.parquet')
    D=dict(rid=rid,host=host,title=title,date=date,fidx=[0]*len(rid),files=[str(td/'s0.parquet')],n_rows=len(rid));pickle.dump(D,open(td/'titles.pkl','wb'))
    import types
    class E:
        def encode_batch(self,ss,add_special_tokens=False):return [types.SimpleNamespace(ids=list(range(len(s.split())))) for s in ss]
    N._ENC=E()
    a=argparse.Namespace(out=str(td),dup_jac=0.5,min_articles=5,min_domains=5,min_c=3,workers=1,seed=5,rerun=True)
    run(a);M=json.load(open(td/'e57_namedevents'/'e57_namedevents.json'))
    byn={r['event']:r for r in M['rows']}
    assert byn['Nipah virus']['titles']==14 and byn['Mayweather-Pacquiao']['titles']==14 and byn['Venezuela-Machado']['titles']==14 and byn['Spider-Man Brand New Day']['titles']==0
    assert byn['Nipah virus']['ok'] and byn['Nipah virus']['lead']['targets']>=1 and M['tanker']['titles']==5 and M['tanker']['also_machado']==0
    assert M['combined'] and M['combined']['lead']['TIME']['chunks']>=1 and 'rho_win' in (td/'e57_namedevents'/'e57_namedevents_report.md').read_text()
    assert not byn['Spider-Man Brand New Day']['ok']
    print('e57_namedevents selftest ok:',{k:(v['titles'],v['n_dedup']) for k,v in byn.items() if v['titles']})
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw');ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-articles',type=int,default=5);ap.add_argument('--min-domains',type=int,default=5);ap.add_argument('--min-c',type=int,default=3)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
