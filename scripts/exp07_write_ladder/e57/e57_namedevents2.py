"""E57 named-events survey v2 (CPU only, no model, no GPU, NO download, no new clustering; job 221). Same 12 events and the same measurements as e57_namedevents.py (job 220); four changes, fixed here BEFORE the result (Hans 2026-10-23 17:51Z in the news thread):
 (a) the ethics filter works on the (S, A) TARGET, not on the event: only the sentence where S and A occur together and one sentence before and after it (per occurrence, at most 3 occurrences per article) is looked at. A target is rejected for: b e-mail / phone / street address in a window; c a person named by age together with a violence keyword in a window; d a minor together with a violence keyword in a window (b, c, d: ANY window of the target, as in ETH-v1); addr address words (lives at / resident of / home address ...), family (wife, daughter, parents ...), health (cancer, diagnosed, hospitalised, patient ...), victim (victim, survivor, deceased, accused, suspect ...) in >= 20 % of the windows; person-S: S looks like a private individual (an age marker or a role word such as student / villager / resident / passenger in >= 20 % of the windows). Rule a of ETH-v1 (event-level violence share) is only a report column. All of this is a regex proxy, no list of public figures exists.
 (b) stricter event matching: word boundaries (the old 'ncert' matched inside 'concert'), generic words dropped; the report lists the 5 most frequent matched titles per event and a seeded sample of 20 matched titles per event for a manual precision check (the precision itself is NOT computed, it has to be read).
 (c) per event, in the date-ordered stream of that event alone, the number of repeated occurrences whose previous occurrence lies OUTSIDE the 8192 tokens before their chunk.
 (d) the rejected targets are saved: counts per reason per event and up to 5 examples per reason (S, A, c, two article titles) in e57_namedevents2_rejects.json; the report has counts and article titles only.
Outputs <out>/e57_namedevents2/e57_namedevents2_report.md, e57_namedevents2.json, e57_namedevents2_rejects.json. No article text is written."""
import argparse,json,sys,random,pickle,pathlib,time,collections,re
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
import e57_freeze as FZ
import e57_news10 as Q10
import e57_events as EV
import e57_namedevents as ND
EVENTS=collections.OrderedDict([
 ('Rashmika-Vijay wedding',[r'\brashmika\b',r'\bvijay\b',r'\b(wedding|weds|wed|married|marriage|engagement|nuptials|bride|groom)\b']),
 ('Spider-Man Brand New Day',[r'\bspider-?man\b',r'\bbrand new day\b']),
 ('Venezuela-Machado',[r'\bmachado\b',r'\b(venezuela\w*|maduro|nobel|opposition|trump)\b']),
 ('Mamdani inauguration',[r'\bmamdani\b',r'\b(sworn|swear\w*|oath|inaugurat\w*|takes office|took office|sworn-in)\b']),
 ('Nipah virus',[r'\bnipah\b']),
 ('El Paso airspace',[r'\bel paso\b',r'\b(airspace|no[- ]fly|faa|ground stop|flight restrictions?|laser|drones?)\b']),
 ('HDFC Bank chairman resignation',[r'\bhdfc\b',r'\bchair(man|person)\b',r'\b(resign\w*|quits?|steps? down|stepped down)\b']),
 ('IDFC First Bank fraud',[r'\bidfc\b',r'\b(fraud|scam|embezzl\w*|misappropriat\w*|siphon\w*)\b']),
 ('Hannah Montana 20th anniversary',[r'\bhannah montana\b',r'\b(20th|twenty|anniversary|reunion|20 years|two decades)\b']),
 ('Mayweather-Pacquiao',[r'\bmayweather\b',r'\bpacquiao\b']),
 ('Ranji Trophy J&K',[r'\branji\b',r'(?<!\w)j\s?&\s?k(?!\w)|\bjammu\b|\bkashmir\b']),
 ('NCERT textbooks',[r'\bncert\b',r'\b(textbooks?|text books?|syllabus|curriculum|chapters?)\b'])])
TANKER=ND.TANKER
FAM=re.compile(r"\b(wife|husband|daughter|son|mother|father|brother|sister|parents?|grandmother|grandfather|family|relatives?|girlfriend|boyfriend|fianc[eé]e?)\b",re.I)
HEALTH=re.compile(r"\b(cancer|diagnos\w*|surgery|hospitali[sz]ed|illness|ailing|chronic|disease|infected|infection|patients?|medical condition|mental health|pregnan\w*)\b",re.I)
VICT=re.compile(r"\b(victims?|survivors?|deceased|complainant|accused|suspects?)\b",re.I)
ADDR=re.compile(r"\b(lives? (?:at|in)|resid(?:es|ing|ence)|home address|house no\.?|flat no\.?|resident of)\b",re.I)
ROLE=re.compile(r"\b(student|villager|resident|passenger|neighbou?r|employee|worker|patient|driver|teacher)\b",re.I)
SHARE=0.2;WIN_PER_ART=3
def target_windows(r,t):
    """sentence of S and A together plus one before and one after, per occurrence (<= 3 per article)"""
    W=[]
    for ai,v in t['occ'].items():
        ss=r['sents'][ai]
        for s in v[:WIN_PER_ART]:W.append(' '.join(ss[max(0,s-1):s+2]))
    return W
def target_reasons(W):
    """-> list of reason names (empty = target accepted)"""
    if not W:return []
    n=len(W);why=[]
    sh=lambda rx:sum(bool(rx.search(w)) for w in W)/n
    if any(FZ.PII.search(w) for w in W):why.append('b: e-mail / phone / street address')
    if any(FZ.AGE.search(w) and FZ.VIOL.search(w) for w in W):why.append('c: age with a violence keyword')
    if any(FZ.MINOR.search(w) and FZ.VIOL.search(w) for w in W):why.append('d: minor with a violence keyword')
    for nm,rx in(('addr: address words',ADDR),('family: family words',FAM),('health: health words',HEALTH),('victim: victim / suspect words',VICT)):
        if sh(rx)>=SHARE:why.append(nm)
    if sh(FZ.AGE)>=SHARE or sh(ROLE)>=SHARE:why.append('person-S: private-individual proxy (age or role word)')
    return why
def filter_targets(r,D,nex=5):
    """-> (kept targets, dict reason -> count, dict reason -> examples, n rejected)"""
    keep=[];cnt=collections.Counter();ex=collections.defaultdict(list);nrej=0
    for t in r['targets']:
        why=target_reasons(target_windows(r,t))
        if not why:keep.append(t);continue
        nrej+=1
        for w in why:
            cnt[w]+=1
            if len(ex[w])<nex:ex[w].append(dict(S=t['S'],ty=t['ty'],key=t['key'],c=t['c'],titles=[D['title'][D['_ridx'][r['rids'][ai]]][:100] for ai in list(t['occ'])[:2]]))
    return keep,dict(cnt),dict(ex),nrej
def out_of_window(order_res,i,pub_date,vn,seed):
    """repeated occurrences of the event alone in date order: total and those whose previous occurrence is outside the 8192 tokens before their chunk"""
    events,tokof,occ,tord,dates=EV.build_set([i],order_res,pub_date,vn,seed);occ={k:v for k,v in occ.items() if len(v)>=3}
    s=EV.occ_stats(tord,tokof,occ);n=s['n_repeat_occ'];rho=s['rho_prev_chunks']
    return dict(repeats=n,outside=(n-int(round(rho*n))) if n else 0,rho=rho if n else None,chunks=s['chunks'],targets=len(occ))
def run(a):
    out=pathlib.Path(a.out);tp=out/'titles.pkl';assert tp.exists(),('titles.pkl missing in',str(out))
    fd=out/'e57_namedevents2';fd.mkdir(exist_ok=True);rep=fd/'e57_namedevents2_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo):',rep);sys.exit(8)
    t0=time.time();D=pickle.load(open(tp,'rb'));D['_ridx']={r:i for i,r in enumerate(D['rid'])};pub_date={r:EV.to_ord(d) for r,d in zip(D['rid'],D['date'])}
    rx=collections.OrderedDict((k,[re.compile(x,re.I) for x in v]) for k,v in EVENTS.items())
    matched={k:ND.match_titles(D['title'],v) for k,v in rx.items()};print({k:len(v) for k,v in matched.items()},flush=True)
    tank=set(ND.match_titles(D['title'],[re.compile(x,re.I) for x in TANKER]));mach=set(matched['Venezuela-Machado'])
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
        row=dict(event=k,titles=len(v),hosts_one_per=len(rids),first=min(D['date'][i] for i in v) if v else None,last=max(D['date'][i] for i in v) if v else None,densest7=ND.BC_dense(dates),ok=r['ok'],n_dedup=r['n_dedup'],n_before=r['n_before'],tinfo=tinfo[k])
        if r['ok']:
            row['independent_domains']=len({x['host'] for x in r['arts']})
            lt=EV.lead_texts(r);why=FZ.ethics(lt);nv=sum(bool(FZ.VIOL.search(t)) for t in lt)
            row['eth_pass']=True;row['eth_why']=[];row['viol_share']=nv/len(lt) if lt else None;row['eth_v1_event_level']=why
            n0=len(r['targets']);keep,cnt,ex,nrej=filter_targets(r,D);r['targets']=keep
            row['targets_f2_before']=n0;row['rejected_targets']=nrej;row['reject_counts']=cnt;rejects[k]=dict(counts=cnt,examples=ex,before=n0,rejected=nrej)
            for vn in('lead','full'):
                tc,V=EV.target_counts(r,vn);row[vn]=dict(counts=tc,targets=len(V['targets']),pool=ND.pool_sat(r,vn))
        rows.append(row)
    okidx=[i for i,r in enumerate(res) if r['ok'] and EV.target_counts(r,'lead')[1]['targets']]
    for i in okidx:
        for vn in('lead','full'):rows[i][vn]['outside']=out_of_window(res,i,pub_date,vn,a.seed)
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
    json.dump(M,open(fd/'e57_namedevents2.json','w'),indent=1,default=str);json.dump(rejects,open(fd/'e57_namedevents2_rejects.json','w'),indent=1,default=str)
    txt=report(M,a);rep.write_text(txt);print(txt[:9000])
def report(M,a):
    f=EV.fmt;base=ND.report(M,a).split('\n')
    L=['# E57 named-events survey v2 (CPU, regex proxy)','','CHANGES against job 220: (a) ethics filter on the (S, A) target (sentence of S and A plus one before and after; rules in the docstring of e57_namedevents2.py), ETH-v1 rule a only as a report column (column ETH-v1 below is therefore always pass; the event-level violence-keyword share is in the table of section M); (b) strict title matching; (c) repeats outside the window per event; (d) rejected targets saved in e57_namedevents2_rejects.json. All counts below are AFTER the target filter.','']
    L+=['## M. matching: top 5 titles and a sample of 20 per event (read these to judge the precision; it is not computed)','']
    for r in M['rows']:
        L+=[f"### {r['event']} ({r['titles']} matched titles; event-level violence-keyword share of lead texts {f(r.get('viol_share'))}; ETH-v1 as it would have judged the whole event: {'; '.join(r.get('eth_v1_event_level') or []) or ('pass' if r['ok'] else 'n/a')})",'','top 5 (title, count):']
        L+=[f"- {t} ({n})" for t,n in r['tinfo']['top5']]+['','sample of 20 (seeded):']+[f"- {t}" for t in r['tinfo']['sample20']]+['']
    L+=['## target filter (ethics, per target)','','| event | targets f >= 2 before | rejected | kept | reasons (count of targets; a target can have several) |','|---|---|---|---|---|']
    tb=tr=0
    for r in M['rows']:
        if not r['ok']:L.append(f"| {r['event']} | n/a | n/a | n/a | below the analysis minimum |");continue
        L.append(f"| {r['event']} | {r['targets_f2_before']} | {r['rejected_targets']} | {r['targets_f2_before']-r['rejected_targets']} | "+'; '.join(f"{k} {v}" for k,v in sorted(r['reject_counts'].items()))+' |');tb+=r['targets_f2_before'];tr+=r['rejected_targets']
    L+=[f"| all events | {tb} | {tr} | {tb-tr} | |",'','Examples (S, A, article titles) per reason are in e57_namedevents2_rejects.json on the server.','']
    L+=['## repeated occurrences whose previous occurrence is outside the window (event alone, date order, targets c >= 3)','','| event | variant | targets | repeated occurrences | previous occurrence outside 8192 tokens | share outside | chunks |','|---|---|---|---|---|---|---|']
    for r in M['rows']:
        for vn in('lead','full'):
            o=r.get(vn,{}).get('outside') if r['ok'] else None
            if o:L.append(f"| {r['event']} | {vn} | {o['targets']} | {o['repeats']} | {o['outside']} | {f(o['outside']/o['repeats'] if o['repeats'] else None)} | {o['chunks']} |")
    L+=['']
    return '\n'.join(L+['---','','# survey tables (same as job 220)','']+base)
def _selftest():
    import tempfile,datetime,types,pyarrow as pa,pyarrow.parquet as pq
    # unit: matching and target reasons
    rx={k:[re.compile(x,re.I) for x in v] for k,v in EVENTS.items()}
    m=lambda ev,t:all(x.search(t) for x in rx[ev])
    assert not m('NCERT textbooks','Cultural concert and textbook fair') and m('NCERT textbooks','NCERT revises class 8 textbooks')
    assert not m('El Paso airspace','ARO Med Spa opens in El Paso with a flight of services') and m('El Paso airspace','FAA closes El Paso airspace for laser test')
    assert m('Nipah virus','Nipah virus case confirmed') and m('Ranji Trophy J&K','Ranji Trophy: J&K beat Delhi') and not m('Mamdani inauguration','Mamdani speaks')
    r=dict(sents=[['Anna Lee, 34-year-old, was shot dead on Monday.','Her husband said so.','Other.'],['Tom Bell won the cup in Paris.','Bell said he was happy.','Fine.']],arts=[{},{}],title='x')
    assert target_reasons(target_windows(r,dict(occ={0:[0]})))     # age + violence
    assert target_reasons(target_windows(r,dict(occ={1:[0]})))==[]
    assert any('family' in x for x in target_reasons(['His wife and daughter were there.']))
    td=pathlib.Path(tempfile.mkdtemp());rid=[];host=[];title=[];date=[];text=[]
    dd=lambda n:int((datetime.date(2026,1,1)+datetime.timedelta(days=n)).strftime('%Y%m%d'))
    def add(t,d,h,tx):rid.append(f'r{len(rid)}');host.append(h);title.append(t);date.append(d);text.append(tx)
    names=['Maria Gonzalez','Pavel Novak','Helena Duarte','Tomas Albrecht']
    def art(ev,k):
        s=[f'On January {1+k%3}, {names[ev]} said the fund raised $%d million after the {names[(ev+1)%4]} agreement was signed in Springfield.'%(10+ev) if k%4 else f'{names[ev]} announced that the fund raised $%d million in total today.'%(10+ev)]
        return ' '.join(s+[f'Reporter {k} filler sentence number {j} with several varied words about topic {ev} host {k} item {j*k}.' for j in range(25)])
    def art_bad(ev,k):return f'{names[ev]}, 34-year-old, was shot dead after the fund raised $%d million.'%(10+ev)+' '+' '.join(f'Reporter {k} filler sentence number {j} with several varied words about topic {ev} host {k} item {j*k}.' for j in range(25))
    for k in range(14):add(f'Nipah virus case confirmed update {k}',dd(5+k//7),f'ha{k}',art(0,k))
    for k in range(14):add(f'Mayweather faces Pacquiao rematch talks {k}',dd(20+k//7),f'hb{k}',art_bad(1,k) if k%2 else art(1,k))
    for k in range(14):add(f'Machado speaks to Venezuela press {k}',dd(30+k//7),f'hd{k}',art(2,k))
    for k in range(5):add(f'Venezuela tanker seized {k}',dd(31),f'he{k}',art(3,k))
    for k in range(6):add(f'Cultural concert textbook fair {k}',dd(3),f'hf{k}',art(3,k))
    pq.write_table(pa.table(dict(warc_record_id=rid,text=text)),td/'s0.parquet')
    D=dict(rid=rid,host=host,title=title,date=date,fidx=[0]*len(rid),files=[str(td/'s0.parquet')],n_rows=len(rid));pickle.dump(D,open(td/'titles.pkl','wb'))
    class E:
        def encode_batch(self,ss,add_special_tokens=False):return [types.SimpleNamespace(ids=list(range(len(s.split())))) for s in ss]
    N._ENC=E()
    a=argparse.Namespace(out=str(td),dup_jac=0.5,min_articles=5,min_domains=5,min_c=3,workers=1,seed=5,rerun=True)
    run(a);M=json.load(open(td/'e57_namedevents2'/'e57_namedevents2.json'));RJ=json.load(open(td/'e57_namedevents2'/'e57_namedevents2_rejects.json'))
    byn={r['event']:r for r in M['rows']}
    assert byn['NCERT textbooks']['titles']==0 and byn['Nipah virus']['titles']==14 and byn['Nipah virus']['ok']
    assert byn['Mayweather-Pacquiao']['rejected_targets']>=1 or byn['Mayweather-Pacquiao']['lead']['targets']<=byn['Mayweather-Pacquiao']['targets_f2_before'],'filter ran'
    assert 'Mayweather-Pacquiao' in RJ and (td/'e57_namedevents2'/'e57_namedevents2_report.md').exists()
    assert 'previous occurrence outside' in (td/'e57_namedevents2'/'e57_namedevents2_report.md').read_text()
    print('e57_namedevents2 selftest ok:',{k:(v['titles'],v['n_dedup'],v.get('targets_f2_before'),v.get('rejected_targets')) for k,v in byn.items() if v['titles']})
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw');ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-articles',type=int,default=5);ap.add_argument('--min-domains',type=int,default=5);ap.add_argument('--min-c',type=int,default=3)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
