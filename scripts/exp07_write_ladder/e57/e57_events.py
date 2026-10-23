"""E57 news events survey (CPU only, no model, no GPU, NO download): how many independent articles do the big news events have, how many (S, A) targets reach c articles, what a publication-time stream of the top events looks like, and how often the value A of the same S changes over time. Reuses titles.pkl, clusters.pkl and the monthly shards of the job-202 run (--out / --raw); the text stage is redone for the scanned clusters only. PROXIES as everywhere in E57: regex (S, A), title-token clusters, one article per hostname after near-duplicate removal.
Parts:
  (1) distribution of the title-stage cluster sizes (members, hostnames) over ALL title-stage clusters; independent article counts (near-duplicate removal, one article per hostname) for the SCAN top clusters by hostname count and for a seeded sample of other clusters (ratio independent / hostnames); the top-20 list by independent count with the independent domains, first / last date and the ethics filter ETH-v1 (e57_freeze.ethics) on the lead texts.
  (2) per top-20 cluster: number of targets (S, A) with f >= 2 forms that appear in >= c distinct independent articles, c >= 3, 5, 10, 20, 50, for the lead variant (first sentences up to 350 tokens) and the full-article variant (up to 1500 tokens).
  (3) the top N = 5 / 10 / 15 ETH-passing clusters of the top-20 list put once into publication order (date, ties by a seeded shuffle): total tokens, 1024-token chunks (= updates), per target the token distance to the previous occurrence of the same target, share of repeated occurrences whose previous occurrence lies in the 8192 tokens before their chunk (rho_win), the mixing of events (distinct events per 8192-token block, share of neighbouring articles from different events); references: events one after another (CLUSTER) and round robin (SPREAD), one pass each, same metric.
  (4) A over time: of the targets that would be kept without the conflict rule (same S, same type, a competing value A' of the same slot form in >= max(2, c/2) articles), the share that has such a competitor; of those, the share whose competing value sits at a different time (median publication dates of the two article sets >= 2 days apart); and the same restricted to the top-N sets.
Report: <out>/e57_events/e57_events_report.md (+ .json)."""
import argparse,json,sys,math,random,pickle,pathlib,time,collections,os
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
import e57_freeze as FZ
import e57_news10 as Q10
CS=(3,5,10,20,50);NS=(5,10,15);BUCKETS=(8,12,20,30,50,100,200,500,1000)
def pct(v,q):
    v=sorted(v);return v[min(len(v)-1,int(q*(len(v)-1)))] if v else float('nan')
def analyze_set(D,CL,idxs,a,detail):
    """text stage for the clusters CL['clusters'][idxs] -> list of results (same order); detail adds sentences / kept indices / rids"""
    cl=CL['clusters'];members=sorted({i for ci in idxs for i in cl[ci]['members']});got=N.read_texts(D,members);jobs=[];meta=[]
    for ci in idxs:
        c=cl[ci];ids=[(i,D['rid'][i]) for i in c['members'] if D['rid'][i] in got]
        jobs.append(dict(title=c['title'],texts=[got[r] for _,r in ids],hosts=[D['host'][i] for i,_ in ids],args=dict(dup_jac=a.dup_jac,min_articles=a.min_articles,min_domains=a.min_domains,min_c=a.min_c,min_f=1,detail=detail,conflicts=detail)));meta.append([r for _,r in ids])
    if a.workers>1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as p:res=p.map(N.analyze,jobs,chunksize=1)
    else:res=[N.analyze(j) for j in jobs]
    for r,rids in zip(res,meta):
        if r['ok'] and detail:r['rids']=[rids[k] for k in r['keep']]
    return res
def title_stage(CL,D):
    cl=CL['clusters'];nm=[len(c['members']) for c in cl];nh=[c['n_hosts'] for c in cl]
    return dict(n=len(cl),rejected=CL['rejected'],members_buckets={b:sum(x>=b for x in nm) for b in BUCKETS},hosts_buckets={b:sum(x>=b for x in nh) for b in BUCKETS},members_q={q:pct(nm,q) for q in(0.5,0.9,0.99,1.0)},hosts_q={q:pct(nh,q) for q in(0.5,0.9,0.99,1.0)})
def lead_texts(r):
    V=N.variant(r,'lead');return [' '.join(r['sents'][ai][:V['kept'][ai]]) for ai in range(len(r['arts']))]
def target_counts(r,variant):
    V=N.variant(r,variant);cs=[t['c_v'] for t in V['targets']];return {c:sum(x>=c for x in cs) for c in CS},V
def occ_stats(order,tokof,occ,CH=1024,WIN=8192):
    """one pass over `order` (list of article keys): token positions; for every target the occurrences sorted by position; non-first occurrences: distance to the previous occurrence, share with a previous occurrence inside the 8192 tokens before the chunk start (previous chunks only) and before the occurrence itself"""
    pos=0;start={}
    for a in order:start[a]=pos;pos+=tokof[a]
    dist=[];h1=h2=n=0;first=0
    for t,per in occ.items():
        ev=sorted(start[a]+o for a,offs in per.items() if a in start for o in offs)
        first+=1 if ev else 0
        for i,x in enumerate(ev[1:],1):
            cs=(x//CH)*CH;prev=ev[:i];n+=1;dist.append(x-prev[-1]);h1+=any(cs-WIN<=y<cs for y in prev);h2+=any(cs-WIN<=y<x for y in prev)
    return dict(total_tokens=pos,chunks=math.ceil(pos/CH),n_repeat_occ=n,n_targets_seen=first,rho_prev_chunks=h1/n if n else float('nan'),rho_incl_chunk=h2/n if n else float('nan'),dist_q={q:pct(dist,q) for q in(0.25,0.5,0.75,0.9)},dist_within_win=sum(d<=WIN for d in dist)/len(dist) if dist else float('nan'))
def mixing(order,tokof,WIN=8192):
    """distinct events per non-overlapping 8192-token block of the stream, effective number of events (exp entropy of the token shares), share of neighbouring articles from different events"""
    pos=0;blocks=collections.defaultdict(collections.Counter)
    for a in order:
        blocks[pos//WIN][a[0]]+=tokof[a];pos+=tokof[a]
    nd=[len(b) for b in blocks.values()];ne=[]
    for b in blocks.values():
        t=sum(b.values());ne.append(math.exp(-sum(v/t*math.log(v/t) for v in b.values())))
    sw=sum(1 for x,y in zip(order,order[1:]) if x[0]!=y[0])/max(1,len(order)-1)
    return dict(blocks=len(blocks),distinct_events_mean=sum(nd)/len(nd) if nd else float('nan'),effective_events_mean=sum(ne)/len(ne) if ne else float('nan'),neighbour_switch=sw)
def build_set(chosen,results,pub_date,variant,seed):
    """chosen: indices into results; -> events, tokof, occ, time-ordered article list"""
    events=[];tokof={};occ={};dates={}
    for ei,ri in enumerate(chosen):
        r=results[ri];V=N.variant(r,variant);arts=[(ei,ai) for ai in range(len(r['arts']))];events.append(arts)
        for (e,ai) in arts:tokof[(e,ai)]=V['tok'][ai];dates[(e,ai)]=pub_date[r['rids'][ai]]
        for ti,t in enumerate(V['targets']):occ[(ei,ti)]={(ei,ai):v for ai,v in t['occ_tok'].items()}
    rng=random.Random(seed);allk=[k for ev in events for k in ev];rng.shuffle(allk);time_order=sorted(allk,key=lambda k:dates[k])
    return events,tokof,occ,time_order,dates
def a_change(results,idxs,pub_date):
    """(4): targets without the conflict rule (f >= 2): kept + conflicting; conflicting with a time-shifted competitor"""
    kept=conf=shift=0;rows=[]
    for ri in idxs:
        r=results[ri];kept+=len([t for t in r['targets'] if t['f']>=2])
        for cr in r.get('conf_rec',[]):
            conf+=1
            da=[pub_date[r['rids'][i]] for i in cr['arts_a']];db=[pub_date[r['rids'][i]] for i in cr['arts_b']]
            ma,mb=pct(da,0.5),pct(db,0.5);sh=abs(ma-mb)>=2;shift+=sh;rows.append((r['title'][:60],cr['S'],cr['key'],cr['other'],ma,mb,sh))
    allt=kept+conf
    return dict(targets_f2_without_conflict_rule=allt,kept=kept,conflicting=conf,conflict_share=conf/allt if allt else float('nan'),time_shifted=shift,time_shifted_share_of_conflicting=shift/conf if conf else float('nan'),time_shifted_share_of_all=shift/allt if allt else float('nan'),examples=rows[:12])
def to_ord(d):
    import datetime;return datetime.date(d//10000,(d//100)%100,d%100).toordinal()
def run(a):
    out=pathlib.Path(a.out);tp=out/'titles.pkl';cp=out/'clusters.pkl';assert tp.exists() and cp.exists(),('job-202 stage files missing in',str(out))
    fd=out/'e57_events';fd.mkdir(exist_ok=True);rep=fd/'e57_events_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo):',rep);sys.exit(8)
    D=pickle.load(open(tp,'rb'));CL=pickle.load(open(cp,'rb'));cl=CL['clusters'];t0=time.time()
    pub_date={r:to_ord(d) for r,d in zip(D['rid'],D['date'])}
    TS=title_stage(CL,D);order=sorted(range(len(cl)),key=lambda i:(-cl[i]['n_hosts'],-len(cl[i]['members']),i))
    scan=order[:a.scan];rest=order[a.scan:];rng=random.Random(a.seed);samp=rng.sample(rest,min(a.sample,len(rest)))
    print(f'title stage: {len(cl)} clusters; text stage on {len(scan)} top-hostname clusters (detail) + {len(samp)} sampled others',flush=True)
    res=analyze_set(D,CL,scan,a,True);print(f'  scanned clusters analysed, {time.time()-t0:.0f}s',flush=True)
    rs=analyze_set(D,CL,samp,a,False);print(f'  sample analysed, {time.time()-t0:.0f}s',flush=True)
    res=Q10.with_f(res,2)
    ok=[i for i,r in enumerate(res) if r['ok']];ok.sort(key=lambda i:-res[i]['n_dedup'])
    top=ok[:a.top];rows=[]
    for i in top:
        r=res[i];c=cl[scan[i]];eth=FZ.ethics(lead_texts(r));tl,V=target_counts(r,'lead');tf,_=target_counts(r,'full')
        rows.append(dict(rank=len(rows)+1,idx=i,title=c['title'][:90],title_members=len(c['members']),title_hosts=c['n_hosts'],independent=r['n_dedup'],independent_domains=len({x['host'] for x in r['arts']}),first=min(D['date'][j] for j in c['members']),last=max(D['date'][j] for j in c['members']),eth_pass=not eth,eth_why=eth,lead=tl,full=tf,n_targets_lead=len(V['targets'])))
    ratio=[r['n_dedup']/max(1,cl[j]['n_hosts']) for r,j in zip(rs,samp) if r['ok']];frac_ok=sum(r['ok'] for r in rs)/max(1,len(rs))
    ind_all=[x['independent'] for x in rows];ind_scan=[res[i]['n_dedup'] for i in ok]
    sets={}
    elig=[x['idx'] for x in rows if x['eth_pass']]
    for n in NS:
        if len(elig)<n:sets[n]=None;continue
        ch=elig[:n];S=dict(clusters=[res[i]['title'][:70] for i in ch],A=a_change(res,ch,pub_date))
        for vn in('lead','full'):
            events,tokof,occ,tord,dates=build_set(ch,res,pub_date,vn,a.seed+n)
            occ={k:v for k,v in occ.items() if len(v)>=3}
            L=C.layouts(events,random.Random(a.seed+n),rounds=1);days=(max(dates.values())-min(dates.values()))+1
            S[vn]=dict(articles=len(tord),n_targets=len(occ),days=days,TIME=dict(occ_stats(tord,tokof,occ),**mixing(tord,tokof)),CLUSTER=dict(occ_stats(L['CLUSTER'],tokof,occ),**mixing(L['CLUSTER'],tokof)),SPREAD=dict(occ_stats(L['SPREAD'],tokof,occ),**mixing(L['SPREAD'],tokof)))
        sets[n]=S
    A_top=a_change(res,[x['idx'] for x in rows],pub_date)
    schema=None
    try:
        import pyarrow.parquet as pq;schema=[f.name for f in pq.read_schema(D['files'][0])]
    except Exception as e:schema=[f'(schema not readable: {e})']
    M=dict(title_stage=TS,scan=len(scan),sample=len(samp),sample_ok_frac=frac_ok,sample_ratio_q={q:pct(ratio,q) for q in(0.1,0.5,0.9)},scan_ok=len(ok),scan_independent_buckets={b:sum(x>=b for x in ind_scan) for b in BUCKETS},top=rows,sets=sets,A_top=A_top,schema=schema,sec=time.time()-t0)
    json.dump(M,open(fd/'e57_events.json','w'),indent=1,default=str);txt=report(M,a);rep.write_text(txt);print(txt)
def fmt(x,d=3):
    return 'n/a' if x is None or (isinstance(x,float) and math.isnan(x)) else (f'{x:.{d}f}' if isinstance(x,float) else str(x))
def report(M,a):
    TS=M['title_stage']
    L=['# E57 news events survey (CPU, regex proxy)','',f"PROXIES: (S, A) regex candidates, title-token clusters, one article per hostname after near-duplicate removal; counts are indicative. Months {a.date_from} .. {a.date_to}. Text stage on the {M['scan']} title-stage clusters with the most hostnames ({M['scan_ok']} survive the independence filter) and on a seeded sample of {M['sample']} other clusters. Run {M['sec']:.0f} s.",'',
       '## (1) independent articles per event','',f"- title-stage clusters: {TS['n']} (rejected at the title stage: {TS['rejected']}). Members per cluster: median {fmt(TS['members_q'][0.5],0)}, 90 % {fmt(TS['members_q'][0.9],0)}, 99 % {fmt(TS['members_q'][0.99],0)}, max {fmt(TS['members_q'][1.0],0)}.",
       '- clusters with at least n hostnames / members (title stage, all clusters): '+'; '.join(f"n >= {b}: {TS['hosts_buckets'][b]} / {TS['members_buckets'][b]}" for b in BUCKETS),
       f"- independent articles (after near-duplicate removal, one per hostname) of the scanned clusters: clusters with at least n: "+'; '.join(f"n >= {b}: {M['scan_independent_buckets'][b]}" for b in BUCKETS),
       f"- sample of other clusters: {M['sample_ok_frac']:.2f} pass the independence filter; independent / hostnames ratio 10 %, median, 90 %: {fmt(M['sample_ratio_q'][0.1])}, {fmt(M['sample_ratio_q'][0.5])}, {fmt(M['sample_ratio_q'][0.9])} (use it to read the title-stage hostname counts above as independent counts).",'',
       '### top clusters by independent articles','','| # | event (title of one article) | title members | title hostnames | independent articles | independent domains | first | last | ETH-v1 | targets lead (c >= 3) |','|---|---|---|---|---|---|---|---|---|---|']
    for x in M['top']:L.append(f"| {x['rank']} | {x['title']} | {x['title_members']} | {x['title_hosts']} | {x['independent']} | {x['independent_domains']} | {x['first']} | {x['last']} | {'pass' if x['eth_pass'] else 'REJECT: '+'; '.join(x['eth_why'])[:80]} | {x['n_targets_lead']} |")
    L+=['','## (2) targets with c or more independent articles (f >= 2 forms)','','| # | event | lead >=3 | >=5 | >=10 | >=20 | >=50 | full >=3 | >=5 | >=10 | >=20 | >=50 |','|---|---|---|---|---|---|---|---|---|---|---|---|']
    tot={v:{c:0 for c in CS} for v in('lead','full')}
    for x in M['top']:
        L.append(f"| {x['rank']} | {x['title'][:50]} | "+' | '.join(str(x['lead'][c]) for c in CS)+' | '+' | '.join(str(x['full'][c]) for c in CS)+' |')
        for v in tot:
            for c in CS:tot[v][c]+=x[v][c]
    L.append('| sum | all listed | '+' | '.join(str(tot['lead'][c]) for c in CS)+' | '+' | '.join(str(tot['full'][c]) for c in CS)+' |')
    L+=['','## (3) the top N ETH-passing events once in publication order','',f"N = {', '.join(map(str,NS))}; one pass; rho_win = share of repeated occurrences (second and later occurrence of a target) with an earlier occurrence of the same target inside the 8192 tokens before their chunk start; TIME = publication date order (ties seeded shuffle); CLUSTER / SPREAD = reference layouts with the same articles (events one after another / round robin).",'']
    for n in NS:
        S=M['sets'][n]
        if S is None:L.append(f'- N = {n}: fewer than {n} ETH-passing events in the top list.');continue
        L+=[f"### N = {n}: {'; '.join(S['clusters'])}",'']
        for vn in('lead','full'):
            s=S[vn];L+=[f"{vn} variant: {s['articles']} articles over {s['days']} days, {s['n_targets']} targets with c >= 3.",'','| layout | total tokens | chunks (= updates) | repeated occurrences | rho_win (prev chunks) | rho incl. chunk | distance to previous occurrence 25 / 50 / 75 / 90 % | share within 8192 | distinct events per 8192 block | effective events | neighbour switch |','|---|---|---|---|---|---|---|---|---|---|---|']
            for lay in('TIME','CLUSTER','SPREAD'):
                y=s[lay];L.append(f"| {lay} | {y['total_tokens']} | {y['chunks']} | {y['n_repeat_occ']} | {fmt(y['rho_prev_chunks'])} | {fmt(y['rho_incl_chunk'])} | "+' / '.join(fmt(y['dist_q'][q],0) for q in(0.25,0.5,0.75,0.9))+f" | {fmt(y['dist_within_win'])} | {fmt(y['distinct_events_mean'],2)} | {fmt(y['effective_events_mean'],2)} | {fmt(y['neighbour_switch'])} |")
            L.append('')
    L+=['## (4) does A change over time for the same S?','',f"- all listed top events: targets that exist without the conflict rule (f >= 2): {M['A_top']['targets_f2_without_conflict_rule']}; with a competing value of the same slot: {M['A_top']['conflicting']} ({fmt(M['A_top']['conflict_share'])}); of those, the competitor sits at a different time (median dates >= 2 days apart): {M['A_top']['time_shifted']} ({fmt(M['A_top']['time_shifted_share_of_conflicting'])} of the conflicting, {fmt(M['A_top']['time_shifted_share_of_all'])} of all)."]
    for n in NS:
        S=M['sets'][n]
        if S:A=S['A'];L.append(f"- top {n} ETH-passing: targets {A['targets_f2_without_conflict_rule']}, conflicting {A['conflicting']} ({fmt(A['conflict_share'])}), time-shifted {A['time_shifted']} ({fmt(A['time_shifted_share_of_conflicting'])} of the conflicting, {fmt(A['time_shifted_share_of_all'])} of all).")
    L+=['','examples (event, S, value, competing value, median date numbers, time-shifted):']+[f"- {e[0]} | {e[1]} | {e[2]} | {e[3]} | {e[4]} / {e[5]} | {e[6]}" for e in M['A_top']['examples']]
    L+=['','## columns of the shards (for an IPTC split later)','',', '.join(map(str,M['schema'])),'','Note: the article text itself is not written to the report; titles are shown only to name the event.']
    return '\n'.join(L)+'\n'
def _selftest():
    import tempfile,pyarrow as pa,pyarrow.parquet as pq,types
    class E:                                   # whitespace tokenizer stand-in (the lab uses the llama3 tokenizer)
        def encode_batch(self,ss,add_special_tokens=False):return [types.SimpleNamespace(ids=list(range(len(s.split())))) for s in ss]
    N._ENC=E();td=pathlib.Path(tempfile.mkdtemp());rng=random.Random(0)
    names=['Maria Gonzalez','Pavel Novak','Helena Duarte','Tomas Albrecht'];rid=[];host=[];title=[];date=[];texts=[]
    for ev in range(3):
        for k in range(14):
            day=20260101+ev*5+k//7
            sents=[f'On January {1+k%3}, {names[ev]} said the fund raised $%d million after the {names[(ev+1)%4]} agreement was signed in Springfield.'%(10+ev) if k%4 else f'{names[ev]} announced that the fund raised $%d million in total today.'%(10+ev)]
            sents+=[f'Reporter {k} filler sentence number {j} with several varied words about topic {ev} host {k} item {j*k}.' for j in range(25)]
            rid.append(f'e{ev}a{k}');host.append(f'h{ev}-{k}.example');title.append(f'Fund{ev} raised agreement signed {names[ev].split()[0]}{ev} news{k%2}');date.append(day);texts.append(' '.join(sents))
    pq.write_table(pa.table(dict(warc_record_id=rid,text=texts)),td/'s0.parquet')
    D=dict(rid=rid,host=host,title=title,date=date,fidx=[0]*len(rid),files=[str(td/'s0.parquet')],n_rows=len(rid))
    cl=[dict(members=list(range(ev*14,ev*14+14)),n_hosts=14,dates=[date[ev*14],date[ev*14+13]],title=title[ev*14]) for ev in range(3)]
    pickle.dump(D,open(td/'titles.pkl','wb'));pickle.dump(dict(clusters=cl,rejected={},n_titles=len(rid),n_buckets=3,n_skipped_titles=0),open(td/'clusters.pkl','wb'))
    a=argparse.Namespace(out=str(td),dup_jac=0.5,min_articles=8,min_domains=8,min_c=3,workers=1,scan=3,sample=0,top=3,seed=5,rerun=True,date_from='2026-01-01',date_to='2026-04-30')
    global NS;NS0=NS;NS=(1,2,3)
    try:
        run(a);M=json.load(open(td/'e57_events'/'e57_events.json'))
    finally:NS=NS0
    assert len(M['top'])==3 and all(x['independent']>=8 for x in M['top']),M['top']
    assert M['sets']['3'] is not None and M['sets']['3']['lead']['TIME']['chunks']>=1
    for n in('1','2','3'):
        s=M['sets'][n]['lead'];assert s['CLUSTER']['neighbour_switch']<=s['SPREAD']['neighbour_switch'] or n=='1'
    assert 'rho_win' in (td/'e57_events'/'e57_events_report.md').read_text()
    # unit checks of the stream metrics
    tok={(0,0):2000,(0,1):2000,(0,2):2000};occ={('t',):{(0,0):[10],(0,1):[10],(0,2):[10]}}
    st=occ_stats([(0,0),(0,1),(0,2)],tok,occ);assert st['n_repeat_occ']==2 and st['rho_prev_chunks']==1.0 and st['dist_q'][0.5]==2000,st
    mx=mixing([(0,0),(1,0),(0,1),(1,1)],{(0,0):4096,(1,0):4096,(0,1):4096,(1,1):4096});assert mx['neighbour_switch']==1.0 and abs(mx['distinct_events_mean']-2)<1e-9
    print('e57_events selftest ok:',{n:(M['sets'][n]['lead']['TIME']['chunks'],M['sets'][n]['lead']['TIME']['rho_prev_chunks']) for n in('1','2','3')},'A',M['A_top']['conflicting'],'/',M['A_top']['targets_f2_without_conflict_rule'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw');ap.add_argument('--scan',type=int,default=60);ap.add_argument('--sample',type=int,default=200);ap.add_argument('--top',type=int,default=20)
    ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-articles',type=int,default=8);ap.add_argument('--min-domains',type=int,default=8);ap.add_argument('--min-c',type=int,default=3);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seed',type=int,default=57)
    ap.add_argument('--date-from',default='2026-01-01');ap.add_argument('--date-to',default='2026-04-30');ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
