"""E57 named-events survey v3 (CPU only, no model, no GPU, NO download, no new clustering; job 222). Job 221 (e57_namedevents2) with the changes Hans approved / asked for on 2026-10-23 18:07Z, all fixed here before the result:
 (1) the family-word rule only applies when S is a private individual: a target whose windows (sentence of S and A plus one before and one after) have family words (wife, daughter, parents ...) in >= 20 % is rejected only if the private-individual proxy of S also holds (age marker or role word such as student / villager / resident / passenger in >= 20 % of the windows); every other rule of job 221 is unchanged (b e-mail / phone / street address, c age + violence keyword, d minor + violence keyword in any window; address words, health words, victim / suspect words in >= 20 % of the windows; person-S proxy). NOTE: with this change the family words never reject a target on their own, they only add a label to targets that the person-S proxy rejects anyway.
 (2) 16 events instead of 12: the 12 fixed events of job 220 / 221 plus the next 4 clusters of the job-219 list (<out>/e57_bigclusters/e57_biglist.json, rows by members) that are: not recurring-type and not product-type suspects (keyword share < 0.3), not overlapping with an event already taken (>= 20 % of the members are in the matched titles of an event of the 12 or of an earlier added cluster, or >= 2 of the 3 top title tokens are equal), taken in the order of members. ETH-v1 of the list is ignored (the ethics filter is applied per target only). The members of those clusters are the titles of the event (no regular expression).
 (3) per event and per answer type (NAME / NUM / DATE ...): number of targets and the c distribution, and the size of the candidate pool of the same type inside the event (strict = exposure within +-1 of the target's c, middle = exposure >= 2 and within +-2, relaxed = any exposure): median and share of targets with >= 15 candidates.
 (4) cap on the number of independent articles per event of 20 / 30 / 40 (seeded random subset, mean of 5 draws): number of targets with c >= 3 left (lead and full variant), against no cap.
Everything else (strict title matching, target-level ethics filter, outside-window repeats, saved rejects, the stream tables) as in job 221. Outputs <out>/e57_namedevents3/e57_namedevents3_report.md, .json, _rejects.json. No article text is written."""
import argparse,json,sys,random,pickle,pathlib,time,collections,re,statistics
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
import e57_freeze as FZ
import e57_news10 as Q10
import e57_events as EV
import e57_namedevents as ND
import e57_namedevents2 as N2
DROP=('Nipah virus','IDFC First Bank fraud');CAPS=(20,30,40);NDRAW=5;NEXTRA=6;OVERLAP=0.2;SUSPECT=0.3
def target_reasons(W):
    """job-221 rules, except that the family words reject only together with the private-individual proxy of S"""
    if not W:return []
    n=len(W);why=[]
    sh=lambda rx:sum(bool(rx.search(w)) for w in W)/n
    if any(FZ.PII.search(w) for w in W):why.append('b: e-mail / phone / street address')
    if any(FZ.AGE.search(w) and FZ.VIOL.search(w) for w in W):why.append('c: age with a violence keyword')
    if any(FZ.MINOR.search(w) and FZ.VIOL.search(w) for w in W):why.append('d: minor with a violence keyword')
    for nm,rx in(('addr: address words',N2.ADDR),('health: health words',N2.HEALTH),('victim: victim / suspect words',N2.VICT)):
        if sh(rx)>=N2.SHARE:why.append(nm)
    if sh(FZ.AGE)>=N2.SHARE or sh(N2.ROLE)>=N2.SHARE:
        why.append('person-S: private-individual proxy (age or role word)')
        if sh(N2.FAM)>=N2.SHARE:why.append('family (with person-S): family words')
    return why
def filter_targets(r,D,nex=5):
    keep=[];cnt=collections.Counter();ex=collections.defaultdict(list);nrej=0
    for t in r['targets']:
        why=target_reasons(N2.target_windows(r,t))
        if not why:keep.append(t);continue
        nrej+=1
        for w in why:
            cnt[w]+=1
            if len(ex[w])<nex:ex[w].append(dict(S=t['S'],ty=t['ty'],key=t['key'],c=t['c'],titles=[D['title'][D['_ridx'][r['rids'][ai]]][:100] for ai in list(t['occ'])[:2]]))
    return keep,dict(cnt),dict(ex),nrej
def extra_events(D,out,matched_sets,n=NEXTRA):
    """next n clusters of the job-219 list under the fixed rule; -> [(name, member ids, row)]"""
    jp=out/'e57_bigclusters'/'e57_biglist.json';cp=out/'e57_bigclusters'/'clusters_big.pkl'
    assert jp.exists() and cp.exists(),('job-215 / job-219 outputs missing',str(jp))
    M=json.load(open(jp));cl=pickle.load(open(cp,'rb'))['clusters'];taken=[set(s) for s in matched_sets];toks=[None]*len(taken);res=[]
    for r in sorted(M['rows'],key=lambda r:(-r['members'],-r['core'])):
        if len(res)>=n:break
        if r['recur']>=SUSPECT or r['product']>=SUSPECT:continue
        mem=set(cl[r['ci']]['members']);tt={x if isinstance(x,str) else x[0] for x in r['top_tokens']}   # real rows: [token, share] pairs (json lists)
        bad=False
        for s,t_ in zip(taken,toks):
            if len(mem&s)>=OVERLAP*len(mem) or (t_ is not None and len(tt&t_)>=2):bad=True;break
        if bad:continue
        res.append(('Cluster: '+r['examples'][0][:60],sorted(mem),r));taken.append(mem);toks.append(tt)
    return res
def by_type(r,vn):
    """per answer type: targets, c distribution, pool tiers (median size of the candidate pool and share with >= 15) for the targets of the variant"""
    V=N.variant(r,vn);out={}
    for t in V['targets']:
        o=out.setdefault(t['ty'],dict(n=0,cs=[],strict=[],middle=[],relaxed=[]));cnts=r['pool_counts'].get(f"{t['ty']}|{t['key']}",[]);c=t['c_v'];o['n']+=1;o['cs'].append(c)
        o['strict'].append(sum(1 for x in cnts if abs(x-c)<=1));o['middle'].append(sum(1 for x in cnts if x>=2 and abs(x-c)<=2));o['relaxed'].append(len(cnts))
    res={}
    for ty,o in out.items():
        res[ty]=dict(n=o['n'],c=dict(ge3=o['n'],ge5=sum(x>=5 for x in o['cs']),ge10=sum(x>=10 for x in o['cs']),ge20=sum(x>=20 for x in o['cs']),ge50=sum(x>=50 for x in o['cs'])))
        for tier in('strict','middle','relaxed'):res[ty][tier]=dict(median=statistics.median(o[tier]),share15=sum(x>=15 for x in o[tier])/o['n'])
    return res
def cap_stats(r,vn,seed):
    """mean number of targets with c >= 3 when only a seeded random subset of CAP independent articles is kept (mean of NDRAW draws); None entry = no cap"""
    V=N.variant(r,vn);n=len(r['arts']);res={'all':len(V['targets'])}
    for cap in CAPS:
        if n<=cap:res[cap]=len(V['targets']);continue
        tot=0
        for d in range(NDRAW):
            sub=set(random.Random(seed*1000+cap*10+d).sample(range(n),cap));tot+=sum(1 for t in V['targets'] if sum(1 for ai in t['occ_tok'] if ai in sub)>=3)
        res[cap]=tot/NDRAW
    res['articles']=n;return res
def analyze_all(a):
    """events (regex events minus a.drop + added job-219 clusters), texts, analysis, target-level ethics filter -> dict used by run() and by e57_lite"""
    out=pathlib.Path(a.out);tp=out/'titles.pkl';assert tp.exists(),('titles.pkl missing in',str(out))
    drop=getattr(a,'drop',DROP)
    t0=time.time();D=pickle.load(open(tp,'rb'));D['_ridx']={r:i for i,r in enumerate(D['rid'])};pub_date={r:EV.to_ord(d) for r,d in zip(D['rid'],D['date'])}
    rx=collections.OrderedDict((k,[re.compile(x,re.I) for x in v]) for k,v in N2.EVENTS.items() if k not in drop)
    matched={k:ND.match_titles(D['title'],v) for k,v in rx.items()};print({k:len(v) for k,v in matched.items()},flush=True)
    extra=extra_events(D,out,list(matched.values()),a.n_extra)
    for nm,mem,row in extra:matched[nm]=mem
    print('extra events from the job-219 list:',[(nm,len(mem)) for nm,mem,_ in extra],flush=True)
    tank=set(ND.match_titles(D['title'],[re.compile(x,re.I) for x in N2.TANKER]));mach=set(matched['Venezuela-Machado'])
    prng=random.Random(a.seed);tinfo={}
    for k,v in matched.items():
        cn=collections.Counter(D['title'][i].strip()[:100] for i in v);smp=prng.sample(v,min(20,len(v)))
        tinfo[k]=dict(top5=cn.most_common(5),sample20=[D['title'][i][:100] for i in smp])
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
    res=Q10.with_f(res,2);rows=[];rejects={}
    for k,r,v,rids in zip(names,res,matched.values(),meta):
        dates=[pub_date[D['rid'][i]] for i in v]
        row=dict(event=k,source='regex' if k in N2.EVENTS else 'job-219 list',titles=len(v),hosts_one_per=len(rids),first=min(D['date'][i] for i in v) if v else None,last=max(D['date'][i] for i in v) if v else None,densest7=ND.BC_dense(dates),ok=r['ok'],n_dedup=r['n_dedup'],n_before=r['n_before'],tinfo=tinfo[k])
        if r['ok']:
            row['independent_domains']=len({x['host'] for x in r['arts']})
            lt=EV.lead_texts(r);why=FZ.ethics(lt);nv=sum(bool(FZ.VIOL.search(t)) for t in lt)
            row['eth_pass']=True;row['eth_why']=[];row['viol_share']=nv/len(lt) if lt else None;row['eth_v1_event_level']=why
            n0=len(r['targets']);keep,cnt,ex,nrej=filter_targets(r,D);r['targets']=keep
            row['targets_f2_before']=n0;row['rejected_targets']=nrej;row['reject_counts']=cnt;rejects[k]=dict(counts=cnt,examples=ex,before=n0,rejected=nrej)
            for vn in('lead','full'):
                tc,V=EV.target_counts(r,vn);row[vn]=dict(counts=tc,targets=len(V['targets']),pool=ND.pool_sat(r,vn),by_type=by_type(r,vn),caps=cap_stats(r,vn,a.seed))
        rows.append(row)
    return dict(D=D,names=names,matched=matched,res=res,meta=meta,rows=rows,rejects=rejects,extra=extra,tank=tank,mach=mach,pub_date=pub_date,t0=t0,tinfo=tinfo)
def run(a):
    out=pathlib.Path(a.out);fd=out/'e57_namedevents3';fd.mkdir(exist_ok=True);rep=fd/'e57_namedevents3_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo):',rep);sys.exit(8)
    X=analyze_all(a);D,names,matched,res,meta,rows,rejects,extra,tank,mach,pub_date,t0=[X[k] for k in('D','names','matched','res','meta','rows','rejects','extra','tank','mach','pub_date','t0')]
    okidx=[i for i,r in enumerate(res) if r['ok'] and EV.target_counts(r,'lead')[1]['targets']]
    for i in okidx:
        for vn in('lead','full'):rows[i][vn]['outside']=N2.out_of_window(res,i,pub_date,vn,a.seed)
    comb={}
    if len(okidx)>=2:
        comb['events']=[names[i] for i in okidx]
        for vn in('lead','full'):
            events,tokof,occ,tord,dates=EV.build_set(okidx,res,pub_date,vn,a.seed);occ={kk:vv for kk,vv in occ.items() if len(vv)>=3}
            L=C.layouts(events,random.Random(a.seed),rounds=1)
            comb[vn]=dict(articles=len(tord),n_targets=len(occ),days=(max(dates.values())-min(dates.values()))+1,TIME=dict(EV.occ_stats(tord,tokof,occ),**EV.mixing(tord,tokof)),CLUSTER=dict(EV.occ_stats(L['CLUSTER'],tokof,occ),**EV.mixing(L['CLUSTER'],tokof)),SPREAD=dict(EV.occ_stats(L['SPREAD'],tokof,occ),**EV.mixing(L['SPREAD'],tokof)))
    A_all=EV.a_change(res,okidx,pub_date) if okidx else None
    A_ev={names[i]:EV.a_change(res,[i],pub_date) for i in okidx}
    M=dict(rows=rows,combined=comb,A_all=A_all,A_ev=A_ev,extra=[dict(name=nm,members=len(mem),first=row['first'],last=row['last'],core=row['core'],largest=row['split']['largest_share'],recur=row['recur'],product=row['product']) for nm,mem,row in extra],tanker=dict(titles=len(tank),also_machado=len(tank&mach),machado=len(mach)),sec=time.time()-t0)
    json.dump(M,open(fd/'e57_namedevents3.json','w'),indent=1,default=str);json.dump(rejects,open(fd/'e57_namedevents3_rejects.json','w'),indent=1,default=str)
    txt=report(M,a);rep.write_text(txt);print(txt[:9000])
def report(M,a):
    f=EV.fmt;base=N2.report(M,a).split('\n')
    L=['# E57 named-events survey v3 (CPU, regex proxy)','',f"Changes against job 221: family words reject only together with the private-individual proxy of S (so they never reject on their own); {len(M['rows'])} events = the 12 fixed regex events + the next {len(M['extra'])} clusters of the job-219 list (rule in the docstring of e57_namedevents3.py); per-type target / pool tables; article caps 20 / 30 / 40. Sections below, then the job-221 report (whose title says v2).",'','## added events from the job-219 list','','| event | members | first | last | core | largest component | recurring share | product share |','|---|---|---|---|---|---|---|---|']
    for e in M['extra']:L.append(f"| {e['name']} | {e['members']} | {e['first']} | {e['last']} | {f(e['core'])} | {f(e['largest'])} | {f(e['recur'])} | {f(e['product'])} |")
    L+=['','## targets by answer type and candidate pools inside the event (targets with c >= 3 of the variant)','','strict = candidates of the same type with exposure within +-1 of the target c, middle = exposure >= 2 and within +-2, relaxed = any exposure. median = median pool size over the targets; share = share of targets with >= 15 candidates.','','| event | variant | type | targets | c>=5 | c>=10 | c>=20 | c>=50 | strict median / share | middle median / share | relaxed median / share |','|---|---|---|---|---|---|---|---|---|---|---|']
    for r in M['rows']:
        if not r['ok']:continue
        for vn in('lead','full'):
            for ty,o in sorted(r[vn]['by_type'].items()):L.append(f"| {r['event']} | {vn} | {ty} | {o['n']} | {o['c']['ge5']} | {o['c']['ge10']} | {o['c']['ge20']} | {o['c']['ge50']} | "+' | '.join(f"{o[t]['median']:.0f} / {f(o[t]['share15'])}" for t in('strict','middle','relaxed'))+' |')
    L+=['',f"## article cap: targets with c >= 3 when only {', '.join(map(str,CAPS))} independent articles per event are kept (seeded random subsets, mean of {NDRAW} draws)",'','| event | variant | independent articles | no cap | cap 20 | cap 30 | cap 40 |','|---|---|---|---|---|---|---|']
    tot={vn:collections.Counter() for vn in('lead','full')}
    for r in M['rows']:
        if not r['ok']:continue
        for vn in('lead','full'):
            c=r[vn]['caps'];L.append(f"| {r['event']} | {vn} | {c['articles']} | {c['all']} | {c[20]:.1f} | {c[30]:.1f} | {c[40]:.1f} |")
            for k in('all',20,30,40):tot[vn][k]+=c[k]
    for vn in('lead','full'):L.append(f"| all events | {vn} | | {tot[vn]['all']} | {tot[vn][20]:.1f} | {tot[vn][30]:.1f} | {tot[vn][40]:.1f} |")
    return '\n'.join(L+['','---','']+base)
def _selftest(after=None):
    import tempfile,datetime,types,pyarrow as pa,pyarrow.parquet as pq
    W=['His wife and daughter were there.']
    assert target_reasons(W)==[] and N2.target_reasons(W)
    assert any('person-S' in x for x in target_reasons(['A student said his wife and daughter left.']))
    td=pathlib.Path(tempfile.mkdtemp());rid=[];host=[];title=[];date=[];text=[]
    dd=lambda n:int((datetime.date(2026,1,1)+datetime.timedelta(days=n)).strftime('%Y%m%d'))
    def add(t,d,h,tx):rid.append(f'r{len(rid)}');host.append(h);title.append(t);date.append(d);text.append(tx)
    names=['Maria Gonzalez','Pavel Novak','Helena Duarte','Tomas Albrecht','Ivan Petrov','Sara Kim']
    def art(ev,k):
        s=[f'On January {1+k%3}, {names[ev]} said the fund raised $%d million after the {names[(ev+1)%6]} agreement was signed in Springfield.'%(10+ev) if k%4 else f'{names[ev]} announced that the fund raised $%d million in total today.'%(10+ev)]
        return ' '.join(s+[f'Reporter {k} filler sentence number {j} with several varied words about topic {ev} host {k} item {j*k}.' for j in range(25)])
    for k in range(14):add(f'Nipah virus case confirmed update {k}',dd(5+k//7),f'ha{k}',art(0,k))
    for k in range(14):add(f'Mayweather faces Pacquiao rematch talks {k}',dd(20+k//7),f'hb{k}',art(1,k))
    for k in range(14):add(f'Machado speaks to Venezuela press {k}',dd(30+k//7),f'hd{k}',art(2,k))
    for k in range(10):add(f'Zebra festival draws crowds in Nairobi {k}',dd(40+k//5),f'hz{k}',art(4,k))
    for k in range(10):add(f'Quarterly results earnings report {k}',dd(50+k),f'hq{k}',art(5,k))
    pq.write_table(pa.table(dict(warc_record_id=rid,text=text)),td/'s0.parquet')
    D=dict(rid=rid,host=host,title=title,date=date,fidx=[0]*len(rid),files=[str(td/'s0.parquet')],n_rows=len(rid));pickle.dump(D,open(td/'titles.pkl','wb'))
    fd=td/'e57_bigclusters';fd.mkdir()
    idx=lambda p:[i for i,t in enumerate(title) if t.startswith(p)]
    cl=[dict(members=idx('Nipah'),n_hosts=14),dict(members=idx('Zebra'),n_hosts=10),dict(members=idx('Quarterly'),n_hosts=10)]
    pickle.dump(dict(clusters=cl),open(fd/'clusters_big.pkl','wb'))
    rows=[dict(ci=0,members=14,core=.9,first=20260101,last=20260110,top_tokens=[['nipah',.9],['virus',.8],['case',.7]],recur=0.,product=0.,examples=['Nipah virus case confirmed update 0'],split=dict(largest_share=.9)),dict(ci=2,members=10,core=.9,first=20260101,last=20260110,top_tokens=[['quarterly',.9],['results',.8],['earnings',.7]],recur=1.,product=0.,examples=['Quarterly results earnings report 0'],split=dict(largest_share=.9)),dict(ci=1,members=10,core=.9,first=20260101,last=20260110,top_tokens=[['zebra',.9],['festival',.8],['crowds',.7]],recur=0.,product=0.,examples=['Zebra festival draws crowds in Nairobi 0'],split=dict(largest_share=.9))]
    json.dump(dict(rows=rows),open(fd/'e57_biglist.json','w'))
    class E:
        def encode_batch(self,ss,add_special_tokens=False):return [types.SimpleNamespace(ids=list(range(len(s.split())))) for s in ss]
    N._ENC=E()
    a=argparse.Namespace(out=str(td),dup_jac=0.5,min_articles=5,min_domains=5,min_c=3,workers=1,seed=5,rerun=True,n_extra=4,drop=())
    run(a);M=json.load(open(td/'e57_namedevents3'/'e57_namedevents3.json'));byn={r['event']:r for r in M['rows']}
    assert [e['name'] for e in M['extra']]==['Cluster: Zebra festival draws crowds in Nairobi 0'],M['extra']     # Nipah overlaps an earlier event, Quarterly is a recurring suspect
    z=byn['Cluster: Zebra festival draws crowds in Nairobi 0'];assert z['ok'] and z['lead']['targets']>=1 and z['lead']['caps']['20']>=0 and z['lead']['by_type']
    n=byn['Nipah virus'];assert n['lead']['caps']['all']==n['lead']['targets'] and n['lead']['caps']['40']==n['lead']['targets']
    t=(td/'e57_namedevents3'/'e57_namedevents3_report.md').read_text();assert 'article cap' in t and 'candidate pools' in t
    print('e57_namedevents3 selftest ok:',{k:(v['titles'],v['n_dedup'],v.get('lead',{}).get('targets')) for k,v in byn.items() if v['titles']})
    if after:after(td)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw');ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-articles',type=int,default=5);ap.add_argument('--min-domains',type=int,default=5);ap.add_argument('--min-c',type=int,default=3)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--n-extra',type=int,default=NEXTRA);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
