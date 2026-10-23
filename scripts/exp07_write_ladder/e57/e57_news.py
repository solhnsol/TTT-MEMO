"""E57 news CPU measurements M1-M7 (design E57 section 3). CPU only, no model, no GPU; needs pyarrow, numpy, tokenizers (llama3 tokenizer.json of the repo).
Stages (each writes a file in --out and is skipped when it exists, --rerun to redo):
  fetch    download the monthly parquet shards of ruggsea/infini-news-corpus (huggingface_hub, allow-pattern per month) into --raw   [only if --raw lacks them]
  titles   title stage: read only [warc_record_id, url_hostname, title, publish_date, language_iso639_3, text_length]; English, 1200..30000 characters, date in the window -> titles.pkl
  cluster  event clusters from titles (rare title tokens + date proximity, union-find) -> clusters.pkl, M1 title-stage counts
  measure  text stage on a seeded sample of clusters: near-duplicate removal, independence criteria (M1), regex fact proxy and (S,A) mining (M2), lead share (M3), layouts CLUSTER / SPREAD and rho_win (M4), candidate pools (M5), power (M6), license text (M7) -> e57_news_report.md + .json
PROXIES (stated in the report): (S, A) are mined by regular expressions (numbers, money, percent, dates, capitalised name sequences) instead of the LLM extraction of the design; clusters come from title tokens instead of embeddings. The counts are therefore indicative, not the final bank."""
import argparse,json,os,re,sys,time,math,random,pickle,collections,pathlib,itertools
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
T=C.T
STOP=set('the a an and or of in on at to for with by from as is are was were be been it its this that these those will would could should can may might has have had not no but about after before over under into than then up out more most new says say said us we you he she they his her their our your who what when where why how amid via per vs'.split())
EXCL={'recurring':r'\b(earnings|quarterly|quarter results|stocks?|shares|futures|markets? (?:open|close|update)|forecast|weather|horoscope|lottery|powerball|mega millions|live updates?|top stories|headlines|what to know|odds|betting|prediction)\b',
 'sports':r'\b(vs\.?|score|scores|match|league|season|game|playoffs?|championship|nba|nfl|nhl|mlb|premier league|champions league|fixtures?|transfer|goal|beat|defeat)\b',
 'victim':r'\b(killed|dies|dead|death|shot|murder|stabbed|arrested|missing|body found|suspect|charged|sentenced|victim|rape|abuse)\b'}
EXRE={k:re.compile(v,re.I) for k,v in EXCL.items()}
def ttoks(t):return [w for w in re.findall(r"[a-z0-9][a-z0-9'-]{2,}",t.lower()) if w not in STOP]
def stage_titles(a,out):
    import pyarrow.parquet as pq,pyarrow.compute as pc
    files=sorted(pathlib.Path(a.raw).rglob('*.parquet'));files=[f for f in files if 'annotations' not in str(f)]
    if a.max_files:files=files[:a.max_files]
    assert files,('no parquet files under',a.raw)
    cols=['warc_record_id','url_hostname','title','publish_date','language_iso639_3','text_length']
    rid=[];host=[];title=[];date=[];fidx=[];lo=int(a.date_from.replace('-',''));hi=int(a.date_to.replace('-',''));n_all=0;t0=time.time()
    for fi,f in enumerate(files):
        t=pq.read_table(f,columns=cols);n_all+=t.num_rows
        m=pc.and_(pc.equal(t['language_iso639_3'],'eng'),pc.and_(pc.greater_equal(t['text_length'],1200),pc.less_equal(t['text_length'],30000)))
        t=t.filter(m)
        for r,h,ti,d in zip(t['warc_record_id'].to_pylist(),t['url_hostname'].to_pylist(),t['title'].to_pylist(),t['publish_date'].to_pylist()):
            if not ti or not d or len(d)<10 or not re.fullmatch(r'\d{4}-\d\d-\d\d',d[:10]):continue
            di=int(d[:4]+d[5:7]+d[8:10])
            if not lo<=di<=hi:continue
            rid.append(r);host.append(h or '');title.append(ti.strip());date.append(di);fidx.append(fi)
        if fi%20==0:print(f'  titles: file {fi+1}/{len(files)} kept {len(rid)} of {n_all} rows, {time.time()-t0:.0f}s',flush=True)
    pickle.dump(dict(rid=rid,host=host,title=title,date=date,fidx=fidx,files=[str(f) for f in files],n_rows=n_all),open(out,'wb'));print('titles',len(rid),'of',n_all,'rows')
def ordinal(d):
    import datetime;return datetime.date(d//10000,(d//100)%100,d%100).toordinal()
def stage_cluster(a,tp,out):
    D=pickle.load(open(tp,'rb'));N=len(D['rid']);toks=[ttoks(t) for t in D['title']];df=collections.Counter(w for ts in toks for w in set(ts))
    par=list(range(N))
    def find(x):
        while par[x]!=x:par[x]=par[par[x]];x=par[x]
        return x
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y:par[y]=x
    dord=[ordinal(d) for d in D['date']];buckets=collections.defaultdict(list);skipped=0
    for i,ts in enumerate(toks):
        rare=sorted({w for w in ts if 3<=df[w]<=a.df_max},key=lambda w:(df[w],w))[:4]
        if len(rare)<2:skipped+=1;continue
        for x,y in itertools.combinations(sorted(rare),2):buckets[(x,y)].append(i)
    nb=0
    for k,v in buckets.items():
        if len(v)<2 or len(v)>5000:continue
        nb+=1;v.sort(key=lambda i:dord[i])
        for p,q in zip(v,v[1:]):
            if dord[q]-dord[p]<=a.date_gap:
                sp,sq=set(toks[p]),set(toks[q])
                if len(sp&sq)/max(1,len(sp|sq))>=a.title_jac:union(p,q)
    groups=collections.defaultdict(list)
    for i in range(N):groups[find(i)].append(i)
    cl=[];rej=collections.Counter()
    for g in groups.values():
        if len(g)<a.min_articles:rej['size<min']+=1;continue
        if len(g)>a.max_cluster:rej['size>max (chained)']+=1;continue
        hosts={D['host'][i] for i in g}
        if len(hosts)<a.min_domains:rej['domains<min']+=1;continue
        tit=[D['title'][i] for i in g];why=[k for k,rx in EXRE.items() if sum(bool(rx.search(t)) for t in tit)/len(tit)>=0.3]
        if why:rej['excluded:'+why[0]]+=1;continue
        cl.append(dict(members=g,n_hosts=len(hosts),dates=[min(D['date'][i] for i in g),max(D['date'][i] for i in g)],title=tit[0]))
    pickle.dump(dict(clusters=cl,rejected=dict(rej),n_titles=N,n_buckets=nb,n_skipped_titles=skipped),open(out,'wb'))
    print(f'clusters (title stage): {len(cl)} pass size/domains/exclusions; rejected {dict(rej)}; titles {N}')
def read_texts(D,members,files_filter=None):
    import pyarrow.parquet as pq
    by=collections.defaultdict(set)
    for i in members:by[D['fidx'][i]].add(D['rid'][i])
    got={}
    for fi,ids in by.items():
        t=pq.read_table(D['files'][fi],columns=['warc_record_id','text'],filters=[('warc_record_id','in',sorted(ids))])
        for r,x in zip(t['warc_record_id'].to_pylist(),t['text'].to_pylist()):got[r]=x
    return got
_ENC=None
def enc():
    global _ENC
    if _ENC is None:_ENC=T.load_enc(os.environ.get('E57_TOKENIZER'))
    return _ENC
def split_article(text):
    out=[]
    for para in re.split(r'\n+',text):
        para=para.strip()
        if len(para)<20:continue
        out+=T.split_sents(para)
    return out
def analyze(cl):
    """cl: dict(title, hosts [..], texts [..], args) -> dict or None. Dedup, independence, facts, targets, per-article token geometry."""
    A=cl['args'];texts=cl['texts'];keep=C.minhash_dup(texts,A['dup_jac']);hosts=[cl['hosts'][i] for i in keep]
    # one article per hostname (the longest) so that 'independent articles' are also independent domains
    byh={}
    for i in keep:
        h=cl['hosts'][i]
        if h not in byh or len(texts[i])>len(texts[byh[h]]):byh[h]=i
    keep=sorted(byh.values())
    if len(keep)<A['min_articles'] or len(byh)<A['min_domains']:return dict(ok=False,n_dedup=len(keep),n_before=len(texts))
    e=enc();arts=[]
    for i in keep:
        ss=split_article(texts[i])[:200];tl=[len(x.ids) for x in e.encode_batch(ss,add_special_tokens=False)] if ss else []
        cum=[0];[cum.append(cum[-1]+x) for x in tl];arts.append(dict(host=cl['hosts'][i],sents=ss,tl=tl,cum=cum))
    # facts and (S,A) mining
    pair_art=collections.defaultdict(dict);fact_art=collections.defaultdict(set);ftype={};surf={}
    for ai,ar in enumerate(arts):
        ar['facts']=[]
        for si,s in enumerate(ar['sents']):
            fs=C.facts_in_sentence(s);ar['facts'].append(fs)
            for ty,key,sf,_,_ in fs:fact_art[ty,key].add(ai);ftype[ty,key]=ty;surf.setdefault((ty,key),sf)
            names=[f for f in fs if f[0]=='NAME']
            for S in names:
                for Af in fs:
                    if (Af[0],Af[1])==(S[0],S[1]) or (Af[0]=='NAME' and (Af[1] in S[1] or S[1] in Af[1])):continue
                    pair_art[(S[1],Af[0],Af[1])].setdefault(ai,si)
    targets=[];used=set()
    best={}
    for (sk,ty,ak),d in pair_art.items():
        c=len(d)
        if c<A['min_c']:continue
        cur=best.get((ty,ak))
        if cur is None or c>cur[0]:best[(ty,ak)]=(c,sk)
    for (ty,ak),(c,sk) in sorted(best.items(),key=lambda x:-x[1][0]):
        if ty=='NUM' and not (ak.startswith('$') or ak.endswith('%') or float(ak)>=100):continue     # a bare small number is not a fact key
        d=pair_art[(sk,ty,ak)];items=[];occ={}
        for ai in d:
            ss=[si for si,fs in enumerate(arts[ai]['facts']) if any((f[0],f[1])==(ty,ak) for f in fs) and any(f[0]=='NAME' and f[1]==sk for f in fs)]
            occ[ai]=ss
            for si in ss[:1]:items.append((arts[ai]['sents'][si],[surf[(ty,ak)],surf.get(('NAME',sk),sk)]+[f[2] for f in arts[ai]['facts'][si] if (f[0],f[1]) in ((ty,ak),('NAME',sk))]))
        f=C.n_forms(items)
        if f<A.get('min_f',2):continue
        # conflict proxy: another value of the same type that co-occurs with S, in >= max(2, c/2) articles, in a sentence of the SAME slot-masked form as a sentence of this target (a competing value for the same slot)
        if ty in('NUM','DATE'):
            tg=[C.ngr(C.words(C.masked(x,[y for y in sfs])),5) for x,sfs in items]
            conf=False
            for (s2,t2,k2),dd in pair_art.items():
                if s2!=sk or t2!=ty or k2==ak or len(dd)<max(2,c/2):continue
                for ai2,si2 in dd.items():
                    sent2=arts[ai2]['sents'][si2];g2=C.ngr(C.words(C.masked(sent2,[surf[(ty,k2)],surf.get(('NAME',sk),sk)])),5)
                    if any(C.jac(g2,g1)>=0.6 for g1 in tg):conf=True;break
                if conf:break
            if conf:continue
        targets.append(dict(ty=ty,key=ak,S=sk,c=c,f=f,occ={int(k):v for k,v in occ.items()},surface=surf[(ty,ak)]))
    pools={}
    for t in targets:pools[(t['ty'],t['key'])]=[(k,len(v)) for k,v in fact_art.items() if k[0]==t['ty'] and k[1]!=t['key']]
    r=dict(ok=True,n_dedup=len(keep),n_before=len(texts),arts=[dict(host=x['host'],tl=x['tl'],cum=x['cum'],n_sents=len(x['sents'])) for x in arts],targets=targets,
        pool_counts={f'{k[0]}|{k[1]}':[c for _,c in v] for k,v in pools.items()},title=cl['title'])
    if A.get('detail'):         # E57 freeze: sentences, the indices of the kept articles inside cl['texts'], candidate strings
        r['keep']=list(keep);r['sents']=[x['sents'] for x in arts];r['facts']=[[[(f[0],f[1],f[2]) for f in fs] for fs in x['facts']] for x in arts]
        r['pool_detail']={f'{k[0]}|{k[1]}':[(k2[0],k2[1],surf.get(k2,k2[1]),cn) for k2,cn in v] for k,v in pools.items()}
    return r
def variant(res,mode,lead_tok=350,cap=1500):
    """geometry of a variant: per article the kept sentences (prefix) and token length; targets restricted to occurrences inside"""
    lim=lead_tok if mode=='lead' else cap;V=dict(tok={},kept={})
    for ai,ar in enumerate(res['arts']):
        k=0
        while k<ar['n_sents'] and (ar['cum'][k+1]<=lim or k==0):k+=1
        V['kept'][ai]=k;V['tok'][ai]=ar['cum'][k]+2
    tg=[]
    for t in res['targets']:
        occ={ai:[ar_cum for ar_cum in [res['arts'][ai]['cum'][si] for si in v if si<V['kept'][ai]]] for ai,v in t['occ'].items()}
        occ={ai:v for ai,v in occ.items() if v}
        if len(occ)>=3:tg.append(dict(t,occ_tok=occ,c_v=len(occ)))
    V['targets']=tg;return V
def stage_measure(a,tp,cp,out):
    D=pickle.load(open(tp,'rb'));CL=pickle.load(open(cp,'rb'));cl=CL['clusters'];rng=random.Random(a.seed);samp=cl if len(cl)<=a.max_clusters else rng.sample(cl,a.max_clusters)
    print(f'measure: {len(cl)} title-stage clusters, text stage on {len(samp)} (seed {a.seed})',flush=True)
    t0=time.time();allm=sorted({i for c in samp for i in c['members']});got=read_texts(D,allm);print(f'  text read: {len(got)} of {len(allm)} articles from {len({D["fidx"][i] for i in allm})} files, {time.time()-t0:.0f}s',flush=True)
    jobs=[]
    for c in samp:
        ids=[(i,D['rid'][i]) for i in c['members'] if D['rid'][i] in got]
        jobs.append(dict(title=c['title'],texts=[got[r] for _,r in ids],hosts=[D['host'][i] for i,_ in ids],args=dict(dup_jac=a.dup_jac,min_articles=a.min_articles,min_domains=a.min_domains,min_c=a.min_c)))
    if a.workers>1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as p:res=p.map(analyze,jobs,chunksize=4)
    else:res=[analyze(j) for j in jobs]
    ok=[r for r in res if r['ok']];n_ok=len(ok);frac=n_ok/max(1,len(samp))
    M={}
    M['M1']=dict(title_stage_clusters=len(cl),sampled=len(samp),pass_after_dedup=n_ok,frac=frac,extrapolated_pass=int(round(frac*len(cl))),crit='>= 12 clusters with >= 8 independent articles and >= 8 independent domains after exclusions',passed=bool(round(frac*len(cl))>=12),rejected_title_stage=CL['rejected'])
    withT=[r for r in ok if any(True for _ in r['targets'])]
    full=[(r,variant(r,'full')) for r in ok];lead=[(r,variant(r,'lead')) for r in ok]
    nt_full=[len(v['targets']) for _,v in full];nt_lead=[len(v['targets']) for _,v in lead]
    M['M2']=dict(targets_per_cluster_full=float(sum(nt_full)/max(1,n_ok)),targets_per_cluster_lead=float(sum(nt_lead)/max(1,n_ok)),clusters_with_target=sum(1 for x in nt_full if x),mean_c_full=float(sum(t['c_v'] for _,v in full for t in v['targets'])/max(1,sum(nt_full))),extrapolated_n_12_clusters_full=float(12*sum(nt_full)/max(1,n_ok)),extrapolated_n_12_clusters_lead=float(12*sum(nt_lead)/max(1,n_ok)),crit='mean targets per cluster >= 4 and total n >= 60 (c >= 3)',type_counts=dict(collections.Counter(t['ty'] for _,v in full for t in v['targets'])))
    M['M2']['passed']=bool(M['M2']['targets_per_cluster_full']>=4 and M['M2']['extrapolated_n_12_clusters_full']>=60)
    # M3 lead share: first occurrence per (article, target) ends within 400 tokens
    inl=tot=0
    for r,v in full:
        for t in v['targets']:
            for ai,offs in t['occ_tok'].items():
                tot+=1;inl+= offs[0]<=400
    M['M3']=dict(lead_share=inl/max(1,tot),pairs=tot,crit='lead share (first (S,A) sentence inside the first 400 tokens) >= 0.6 -> lead variant, else full articles',passed=bool(inl/max(1,tot)>=0.6))
    # M4 layouts on 12 events (seeded) per variant
    pool=[(r,f,l) for (r,f),(r2,l) in zip(full,lead) if f['targets']]
    ev_sel=pool if len(pool)<=12 else random.Random(a.seed+1).sample(pool,12)
    M4={}
    for vn,idx in (('full',1),('lead',2)):
        events=[];tokof={};occ={};tinfo=[];cs=[]
        for ei,t3 in enumerate(ev_sel):
            r=t3[0];V=t3[idx];arts=[(ei,ai) for ai in range(len(r['arts']))]
            if len(arts)>a.max_per_event:arts=random.Random(a.seed+2+ei).sample(arts,a.max_per_event)
            events.append(arts)
            for key in arts:tokof[key]=V['tok'][key[1]]
            for ti,t in enumerate(V['targets']):
                o={(ei,ai):v for ai,v in t['occ_tok'].items() if (ei,ai) in tokof}
                if len(o)>=3:occ[(ei,ti)]=o;cs.append(len(o))
        L=C.layouts(events,random.Random(a.seed+3),rounds=2);row=dict(n_events=len(events),n_targets=len(occ),mean_c=float(sum(cs)/max(1,len(cs))))
        for lay in('CLUSTER','SPREAD'):
            s=L[lay];rp,ri,n=C.rho_win(tokof,s,occ);toks=sum(tokof[x] for x in s[:len(s)//2]);row[lay]=dict(rho_prev_chunks=rp,rho_incl_chunk=ri,n_occ=n,round_tokens=toks,round_chunks=toks/1024,occ_per_chunk=(n/(toks/1024)) if toks else None)
        mc=row['mean_c'];row['R_for_128_exposures']=math.ceil(128/mc) if mc else None;row['updates_at_128_exposures']=int(row['R_for_128_exposures']*row['CLUSTER']['round_chunks']) if mc else None
        M4[vn]=row
    M['M4']=dict(by_variant=M4,crit='CLUSTER rho_win(prev chunks) >= 0.5 and SPREAD <= 0.2',passed={vn:bool(not math.isnan(x['CLUSTER']['rho_prev_chunks']) and x['CLUSTER']['rho_prev_chunks']>=0.5 and x['SPREAD']['rho_prev_chunks']<=0.2) for vn,x in M4.items()})
    # M5 candidate pools
    st=rx=n5=0
    for r,v in full:
        for t in v['targets']:
            cnts=r['pool_counts'].get(f"{t['ty']}|{t['key']}",[]);n5+=1;c=t['c_v']
            st+= sum(1 for x in cnts if abs(x-c)<=1)>=15;rx+= len(cnts)>=15
    M['M5']=dict(strict_pm1_ge15=st/max(1,n5),any_c_ge15=rx/max(1,n5),targets=n5,crit='>= 0.8 of targets have 15 same-type candidates inside the same event with exposure within +-1 article of the answer',passed=bool(st/max(1,n5)>=0.8))
    nsel=sum(len(t3[1]['targets']) for t3 in ev_sel)
    M['M6']=dict(n_selected_events=len(ev_sel),n_targets_selected_full=nsel,power_selected=C.power_table(nsel) if nsel else [],power_if_12_clusters=C.power_table(int(M['M2']['extrapolated_n_12_clusters_full'])),power_if_24_clusters=C.power_table(int(2*M['M2']['extrapolated_n_12_clusters_full'])),crit='power >= 0.5 at effect 0.10 (DEFF 1.5, W0 intraclass correlation is not available on the CPU)')
    p12=[x for x in M['M6']['power_if_12_clusters'] if x['deff']==1.5];M['M6']['passed']=bool(p12 and p12[0]['power_0.1']>=0.5)
    M['M7']=dict(license_data=a.license_text or 'LICENSE-DATA not provided (run fetch)',note='CC-BY 4.0 covers the derived columns; the article text stays under the publishers\' copyright and downstream users bear the compliance responsibility. Research use is neither granted nor denied by this text; redistribution of article text is not possible.')
    M['events_selected']=[t3[0]['title'] for t3 in ev_sel]
    json.dump(M,open(out+'.json','w'),indent=1,default=str)
    open(out+'.md','w').write(report(M,a))
    print(open(out+'.md').read())
def report(M,a):
    f=lambda x:'PASS' if x else 'FAIL'
    L=['# E57 news CPU measurement M1-M7 (regex proxy), months '+a.date_from+' .. '+a.date_to,'','PROXY NOTE: (S, A) are regex candidates (numbers, money, percent, dates, capitalised names), clusters come from title tokens, no LLM, no embeddings; the counts below are indicative of the final bank, not equal to it. M1 pass counts and n are extrapolated from a seeded sample of the title-stage clusters. Lead variant = first sentences up to 350 tokens; full variant = article up to 1500 tokens.','',
       '## M1 independent clusters','',f"- title-stage clusters (>= {a.min_articles} articles, >= {a.min_domains} hostnames, exclusions applied): {M['M1']['title_stage_clusters']}; sampled for the text stage {M['M1']['sampled']}; pass after near-duplicate removal and one-article-per-domain independence: {M['M1']['pass_after_dedup']} ({M['M1']['frac']:.2f}); extrapolated {M['M1']['extrapolated_pass']} -> criterion {M['M1']['crit']}: **{f(M['M1']['passed'])}**",f"- title-stage rejections: {M['M1']['rejected_title_stage']}",'',
       '## M2 targets (c >= 3, f >= 2, no conflict)','',f"- targets per passing cluster: full {M['M2']['targets_per_cluster_full']:.2f}, lead {M['M2']['targets_per_cluster_lead']:.2f}; mean c {M['M2']['mean_c_full']:.2f}; clusters with >= 1 target {M['M2']['clusters_with_target']}; extrapolated n for 12 clusters: full {M['M2']['extrapolated_n_12_clusters_full']:.0f}, lead {M['M2']['extrapolated_n_12_clusters_lead']:.0f}; types {M['M2']['type_counts']} -> {M['M2']['crit']}: **{f(M['M2']['passed'])}**",'',
       '## M3 lead share','',f"- {M['M3']['lead_share']:.3f} of (article, target) pairs have the first (S,A) sentence inside the first 400 tokens ({M['M3']['pairs']} pairs) -> {M['M3']['crit']}: **{f(M['M3']['passed'])}**",'',
       '## M4 layouts (12 seeded events, 2 rounds, rho over the last round)','','| variant | layout | rho_win prev chunks | rho_win incl. own chunk | occurrences | round tokens | round chunks | (S,A) sentences per chunk |','|---|---|---|---|---|---|---|---|']
    for vn,x in M['M4']['by_variant'].items():
        for lay in('CLUSTER','SPREAD'):y=x[lay];L.append(f"| {vn} | {lay} | {y['rho_prev_chunks']:.3f} | {y['rho_incl_chunk']:.3f} | {y['n_occ']} | {y['round_tokens']} | {y['round_chunks']:.1f} | {y['occ_per_chunk'] if y['occ_per_chunk'] is None else round(y['occ_per_chunk'],2)} |")
    L+=['',f"- criteria {M['M4']['crit']}: {M['M4']['passed']}",'- exposure 128 -> R and update count: '+'; '.join(f"{vn}: mean c {x['mean_c']:.2f}, targets {x['n_targets']}, R {x['R_for_128_exposures']}, updates {x['updates_at_128_exposures']}" for vn,x in M['M4']['by_variant'].items()),'',
       '## M5 candidate pools','',f"- strict (same type, same event, exposure within +-1 article, >= 15 candidates): {M['M5']['strict_pm1_ge15']:.3f}; relaxed (any exposure, >= 15): {M['M5']['any_c_ge15']:.3f}; targets {M['M5']['targets']} -> {M['M5']['crit']}: **{f(M['M5']['passed'])}**",'',
       '## M6 power (E56 6.4 G1 formula: SE = 0.379/sqrt(n_eff), theta = max(0.10, 1.06/sqrt(n_eff)))','','| scenario | DEFF | n_eff | theta | power @0.03 | @0.10 | @0.15 |','|---|---|---|---|---|---|---|']
    for sc in('power_selected','power_if_12_clusters','power_if_24_clusters'):
        for x in M['M6'][sc]:L.append(f"| {sc} | {x['deff']} | {x['n_eff']:.1f} | {x['theta']:.3f} | {x['power_0.03']:.2f} | {x['power_0.1']:.2f} | {x['power_0.15']:.2f} |")
    L+=['',f"- criterion {M['M6']['crit']} (12 clusters, DEFF 1.5): **{f(M['M6']['passed'])}**",'','## M7 license','','```',str(M['M7']['license_data']),'```',M['M7']['note'],'','## selected events (seeded; titles of the first article)','']+[f'- {t}' for t in M['events_selected']]
    return '\n'.join(L)+'\n'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['fetch','titles','cluster','measure','all']);ap.add_argument('--raw',default='news_raw');ap.add_argument('--out',default='e57_news');ap.add_argument('--months',default='01,02,03,04')
    ap.add_argument('--date-from',default='2026-01-01');ap.add_argument('--date-to',default='2026-04-30');ap.add_argument('--max-files',type=int,default=0);ap.add_argument('--df-max',type=int,default=2000);ap.add_argument('--date-gap',type=int,default=3);ap.add_argument('--title-jac',type=float,default=0.2)
    ap.add_argument('--min-articles',type=int,default=8);ap.add_argument('--min-domains',type=int,default=8);ap.add_argument('--max-cluster',type=int,default=1500);ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-c',type=int,default=3)
    ap.add_argument('--max-clusters',type=int,default=3000);ap.add_argument('--max-per-event',type=int,default=12);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--rerun',action='store_true');ap.add_argument('--license-text');a=ap.parse_args()
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);tp=out/'titles.pkl';cp=out/'clusters.pkl'
    a.date_from=a.date_from;a.license_text=open(a.license_text).read() if a.license_text and os.path.exists(a.license_text) else None
    if a.cmd=='fetch':
        from huggingface_hub import snapshot_download
        pats=[f'data/year=2026/month={m}/*' for m in a.months.split(',')]+['LICENSE-DATA']
        if a.max_files:
            import fnmatch;from huggingface_hub import list_repo_files
            fl=sorted(f for f in list_repo_files('ruggsea/infini-news-corpus',repo_type='dataset') if any(fnmatch.fnmatch(f,q) for q in pats[:-1]))
            pats=fl[:a.max_files]+['LICENSE-DATA']
        snapshot_download('ruggsea/infini-news-corpus',repo_type='dataset',allow_patterns=pats,local_dir=a.raw);return
    if a.cmd in('titles','all') and (a.rerun or not tp.exists()):stage_titles(a,tp)
    if a.cmd in('cluster','all') and (a.rerun or not cp.exists()):stage_cluster(a,tp,cp)
    if a.cmd in('measure','all'):
        lf=pathlib.Path(a.raw)/'LICENSE-DATA'
        if a.license_text is None and lf.exists():a.license_text=lf.read_text()
        stage_measure(a,tp,cp,str(out/'e57_news_report'))
if __name__=='__main__':main()
