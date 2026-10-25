"""E57 section 9.6 news re-measurement (CPU only, no model, no GPU, no download): reuses titles.pkl / clusters.pkl and the parquet shards of the job-202 run (--out, --raw of that run); the text stage is redone (same seed -> same sampled clusters) and its per-cluster results are cached in <out>/analyze_cache.pkl.
Requests of section 9.6, answered in <out>/e57_news96_report.md (+ .json):
  1  rho_win decomposed: (i) another article of the same stream, same (S,A) target, inside the 8192 tokens before the chunk start; (ii) the same article, earlier sentence of the same round; (iii) the copy of the same article of the previous round; plus 'any other article of the same event overlaps the window'. Reproduction check of the relayed M4 numbers with the ORIGINAL shuffled layouts and seeds. Round order random (yes: event order and article order are re-shuffled every round in the original layouts), mean article length per variant.
  2  M4' for CLUSTER and the deterministic SPREAD (fixed event order, one article per event per pass, same order every round, uniform filler tokens per pass chosen as the smallest amount giving every pair of different articles of one event a gap >= 9000 tokens): full articles with 12 events, lead variant with 12 / 20 / 26 / 30 events, each without and with filler. Pass criterion: CLUSTER (i)+(ii) >= 0.5 and SPREAD (i) <= 0.2 (SPREAD (i)+(ii)+(iii) shown). Gap distribution (min, 5th percentile), filler share, round length, updates at 128 exposures.
  3  M5': middle candidates (c' >= 2 and |c - c'| <= 2): share of targets with >= 15 of them; distribution of candidate exposures c' against the answer's c.
  4  power with 24-30 events: n from the targets per event, DEFF table (E56 G1 formula).
PROXIES as in e57_news.py (regex facts, title-token clusters). Filler is only COUNTED (tokens), no filler text is generated."""
import argparse,json,os,sys,math,random,pickle,pathlib,time,collections
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
CH=1024;WIN=8192;GAP=9000
RELAYED={'full':{'CLUSTER':0.767,'SPREAD':0.454},'lead':{'CLUSTER':0.766,'SPREAD':0.643}}   # relayed M4 numbers (design 9.1), for the reproduction check only
def cluster_stream(events,rng,rounds=2):
    """events contiguous, event order and article order reshuffled every round (the layout of e57_common.layouts). -> [(key, round)]"""
    out=[]
    for r in range(rounds):
        ev=[list(e) for e in events];rng.shuffle(ev)
        for e in ev:rng.shuffle(e)
        out+=[(a,r) for e in ev for a in e]
    return out
def shuffled_spread(events,rng,rounds=2):
    out=[]
    for r in range(rounds):
        ev=[list(e) for e in events];rng.shuffle(ev)
        for e in ev:rng.shuffle(e)
        mx=max(len(e) for e in ev);out+=[(e[i],r) for i in range(mx) for e in ev if i<len(e)]
    return out
def fixed_spread(events,rng,rounds,filler):
    """deterministic cycle: event order and article order inside an event drawn ONCE, then one article per event per pass, every round the same; `filler` tokens after every pass. -> [(key, round)] with ('F',n) items for filler"""
    ev=[list(e) for e in events];rng.shuffle(ev)
    for e in ev:rng.shuffle(e)
    mx=max(len(e) for e in ev);out=[]
    for r in range(rounds):
        for i in range(mx):
            out+=[(e[i],r) for e in ev if i<len(e)]
            if filler:out.append((('F',filler),r))
    return out
def positions(stream,tokof):
    pos=0;inst=[]
    for k,r in stream:
        if k[0]=='F':pos+=k[1];continue
        inst.append((pos,pos+tokof[k],k,r));pos+=tokof[k]
    return inst,pos
def event_gaps(inst):
    """gaps (tokens between the end of one article and the start of the NEXT article of the same event, different article)"""
    last={};g=[]
    for s,e,k,r in inst:
        ev=k[0]
        if ev in last:
            ps,pe,pk=last[ev]
            if pk!=k:g.append(s-pe)
        last[ev]=(s,e,k)
    return g
def min_filler(events,rng_seed,rounds,tokof):
    """smallest uniform filler per pass (multiple of 50) with min gap >= GAP"""
    lo,hi=0,40000
    def ok(F):
        inst,_=positions(fixed_spread(events,random.Random(rng_seed),rounds,F),tokof);g=event_gaps(inst);return (min(g) if g else 10**9)>=GAP
    if ok(0):return 0
    if not ok(hi):return None
    while hi-lo>50:
        m=(lo+hi)//2
        if ok(m):hi=m
        else:lo=m
    return hi
def rho_decomp(stream,tokof,occ,rounds):
    """over the occurrences of the LAST round only: shares with a prior occurrence of the same target inside [chunk start - 8192, chunk start): (i) another article, (ii) the same article same round, (iii) the same article, an earlier round; also 'any other article of the same event overlaps the window'. -> dict"""
    inst,total=positions(stream,tokof);start={};
    byk=collections.defaultdict(list)
    for s,e,k,r in inst:byk[k].append((s,e,r))
    byev=collections.defaultdict(list)
    for s,e,k,r in inst:byev[k[0]].append((s,e,k))
    n=0;c=collections.Counter()
    for t,per in occ.items():
        pts=[]
        for k,offs in per.items():
            for s,e,r in byk.get(k,[]):
                for o in offs:pts.append((s+o,k,r))
        pts.sort()
        for x,k,r in pts:
            if r!=rounds-1:continue
            cs=(x//CH)*CH;lo=cs-WIN;n+=1;fi=fii=fiii=False
            for y,k2,r2 in pts:
                if y>=x:break
                if not lo<=y<cs:continue
                if k2!=k:fi=True
                elif r2==r:fii=True
                else:fiii=True
            c['i']+=fi;c['ii']+=fii;c['iii']+=fiii;c['i_or_ii']+=(fi or fii);c['all']+=(fi or fii or fiii)
            c['event_any']+=any(k3!=k and s3<cs and e3>lo for s3,e3,k3 in byev[k[0]])
    d={k:(v/n if n else float('nan')) for k,v in c.items()};d['n_occ']=n
    for k in('i','ii','iii','i_or_ii','all','event_any'):d.setdefault(k,float('nan'))
    d['round_tokens']=total/rounds
    return d
def build_event_inputs(ev_sel,idx,a,seed_art):
    """-> events (lists of keys), tokof, occ (targets with >= 3 articles among the kept ones), n_targets_per_event"""
    events=[];tokof={};occ={};per_ev=[]
    for ei,t3 in enumerate(ev_sel):
        r=t3[0];V=t3[idx];arts=[(ei,ai) for ai in range(len(r['arts']))]
        if len(arts)>a.max_per_event:arts=random.Random(seed_art+ei).sample(arts,a.max_per_event)
        events.append(arts)
        for key in arts:tokof[key]=V['tok'][key[1]]
        n=0
        for ti,t in enumerate(V['targets']):
            o={(ei,ai):v for ai,v in t['occ_tok'].items() if (ei,ai) in tokof}
            if len(o)>=3 and (getattr(a,'cset',None) is None or len(o) in a.cset):occ[(ei,ti)]=o;n+=1
        per_ev.append(n)
    return events,tokof,occ,per_ev
def get_results(a,tp,cp):
    mf=getattr(a,'min_f',2);cache=pathlib.Path(a.out)/('analyze_cache.pkl' if mf==2 else f'analyze_cache_f{mf}.pkl')
    if cache.exists() and not a.rerun:
        d=pickle.load(open(cache,'rb'));print(f'cached text-stage results: {len(d["res"])} clusters ({cache})',flush=True);return d['res'],d['n_title_clusters']
    D=pickle.load(open(tp,'rb'));CL=pickle.load(open(cp,'rb'));cl=CL['clusters'];rng=random.Random(a.seed);samp=cl if len(cl)<=a.max_clusters else rng.sample(cl,a.max_clusters)
    print(f'text stage on {len(samp)} of {len(cl)} title-stage clusters (seed {a.seed}, same sample as job 202)',flush=True);t0=time.time()
    allm=sorted({i for c in samp for i in c['members']});got=N.read_texts(D,allm);print(f'  text read {len(got)} of {len(allm)} articles, {time.time()-t0:.0f}s',flush=True)
    jobs=[]
    for c in samp:
        ids=[(i,D['rid'][i]) for i in c['members'] if D['rid'][i] in got]
        jobs.append(dict(title=c['title'],texts=[got[r] for _,r in ids],hosts=[D['host'][i] for i,_ in ids],args=dict(dup_jac=a.dup_jac,min_articles=a.min_articles,min_domains=a.min_domains,min_c=a.min_c,min_f=getattr(a,'min_f',2))))
    if a.workers>1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as p:res=p.map(N.analyze,jobs,chunksize=4)
    else:res=[N.analyze(j) for j in jobs]
    pickle.dump(dict(res=res,n_title_clusters=len(cl)),open(cache,'wb'));print(f'  text stage done {time.time()-t0:.0f}s, cached',flush=True)
    return res,len(cl)
def avg(v):
    v=[x for x in v if x is not None and not (isinstance(x,float) and math.isnan(x))];return sum(v)/len(v) if v else float('nan')
def config(pool,idx,vn,nev,fill,a,draws):
    """one M4' row: variant index idx (1 full, 2 lead), nev events, filler yes/no, averaged over `draws` seeded event draws"""
    if len(pool)<nev:return dict(skipped=f'pool of events with targets has only {len(pool)} < {nev}')
    R=[]
    for d in range(draws):
        ev_sel=pool if len(pool)==nev else random.Random(a.seed+100*d+nev).sample(pool,nev)
        events,tokof,occ,per_ev=build_event_inputs(ev_sel,idx,a,a.seed+7+d)
        if len(occ)<3:continue
        rs=a.seed+3+d;rounds=2
        cl_rho=rho_decomp(cluster_stream(events,random.Random(rs),rounds),tokof,occ,rounds)
        F=0
        if fill:
            F=min_filler(events,rs,rounds,tokof)
            if F is None:R.append(dict(fail_filler=True));continue
        st=fixed_spread(events,random.Random(rs),rounds,F);sp_rho=rho_decomp(st,tokof,occ,rounds);inst,total=positions(st,tokof);g=sorted(event_gaps(inst))
        art_tok=sum(tokof[k] for k in tokof)/len(tokof);mc=avg([len(o) for o in occ.values()])
        round_tok=total/rounds;Rn=math.ceil(128/mc) if mc else None
        R.append(dict(cluster=cl_rho,spread=sp_rho,filler_per_pass=F,filler_share=(total-sum(tokof[k] for s,e,k,r in inst))/total,gap_min=g[0] if g else None,gap_p5=g[int(0.05*(len(g)-1))] if g else None,round_chunks=round_tok/CH,
            n_targets=len(occ),mean_c=mc,R128=Rn,updates128=int(Rn*round_tok/CH) if Rn else None,mean_article_tokens=art_tok,targets_per_event=avg(per_ev),n_passes=max(len(e) for e in events)))
    ok=[r for r in R if 'cluster' in r]
    if not ok:return dict(skipped='no draw had >= 3 targets' if not any(r.get('fail_filler') for r in R) else 'filler search failed (> 40000 tokens per pass)')
    row=dict(variant=vn,n_events=nev,fill=bool(fill),draws=len(ok),pool=len(pool))
    for lay,key in(('CLUSTER','cluster'),('SPREAD','spread')):
        row[lay]={k:avg([r[key][k] for r in ok]) for k in('i','ii','iii','i_or_ii','all','event_any','n_occ')}
    for k in('filler_per_pass','filler_share','gap_min','gap_p5','round_chunks','n_targets','mean_c','R128','updates128','mean_article_tokens','targets_per_event','n_passes'):row[k]=avg([r[k] for r in ok])
    row['gap_min_worst']=min(r['gap_min'] for r in ok if r['gap_min'] is not None) if any(r['gap_min'] is not None for r in ok) else None
    row['passed']=bool(row['CLUSTER']['i_or_ii']>=0.5 and row['SPREAD']['i']<=0.2 and (not fill or row['filler_share']<=0.30))     # a filler row only counts with filler share <= 0.30 (design 9.4)
    row['spread_all_sources']=row['SPREAD']['all']
    return row
def reproduce(pool,a):
    """the original M4 of job 202: shuffled layouts, 12 events, seeds as in e57_news.stage_measure"""
    out={}
    ev_sel=pool if len(pool)<=12 else random.Random(a.seed+1).sample(pool,12)
    for vn,idx in(('full',1),('lead',2)):
        events=[];tokof={};occ={}
        for ei,t3 in enumerate(ev_sel):
            r=t3[0];V=t3[idx];arts=[(ei,ai) for ai in range(len(r['arts']))]
            if len(arts)>a.max_per_event:arts=random.Random(a.seed+2+ei).sample(arts,a.max_per_event)
            events.append(arts)
            for key in arts:tokof[key]=V['tok'][key[1]]
            for ti,t in enumerate(V['targets']):
                o={(ei,ai):v for ai,v in t['occ_tok'].items() if (ei,ai) in tokof}
                if len(o)>=3:occ[(ei,ti)]=o
        rng=random.Random(a.seed+3);ev=events                      # one layouts() call gave CLUSTER then SPREAD from the same rng
        L={'CLUSTER':[],'SPREAD':[]}
        for r in range(2):
            e2=[list(e) for e in ev];rng.shuffle(e2)
            for e in e2:rng.shuffle(e)
            L['CLUSTER']+=[(x,r) for e in e2 for x in e]
            mx=max(len(e) for e in e2);L['SPREAD']+=[(e[i],r) for i in range(mx) for e in e2 if i<len(e)]
        row={'n_events':len(events),'n_targets':len(occ),'mean_article_tokens':avg([tokof[k] for k in tokof]),'mean_event_tokens':avg([sum(tokof[k] for k in e) for e in events]),'mean_articles_per_event':avg([len(e) for e in events])}
        for lay in('CLUSTER','SPREAD'):row[lay]=rho_decomp(L[lay],tokof,occ,2);row[lay]['relayed_union_rho']=RELAYED[vn][lay]
        out[vn]=row
    return out
def m5prime(res,a):
    """middle candidates (c' >= 2, |c - c'| <= 2): share of targets with >= 15; distribution of candidate exposures c' by answer exposure c (full variant, same targets as M5)"""
    by=collections.defaultdict(list);n=st=mid=rx=0;mid_cnt=[];strict_cnt=[]
    for r in res:
        if not r['ok']:continue
        V=N.variant(r,'full')
        for t in V['targets']:
            cnts=r['pool_counts'].get(f"{t['ty']}|{t['key']}",[]);c=t['c_v'];n+=1
            s1=sum(1 for x in cnts if abs(x-c)<=1);m2=sum(1 for x in cnts if x>=2 and abs(x-c)<=2)
            st+=s1>=15;mid+=m2>=15;rx+=len(cnts)>=15;mid_cnt.append(m2);strict_cnt.append(s1)
            by[min(c,8)].append((len(cnts),sum(1 for x in cnts if x>=2),s1,m2,sum(cnts)/len(cnts) if cnts else 0.0))
    tab=[]
    for c in sorted(by):
        v=by[c];tab.append(dict(c=('>=8' if c==8 else c),targets=len(v),mean_n_candidates=avg([x[0] for x in v]),mean_n_cand_c2=avg([x[1] for x in v]),mean_n_within1=avg([x[2] for x in v]),mean_n_middle=avg([x[3] for x in v]),share_middle_ge15=sum(1 for x in v if x[3]>=15)/len(v),share_strict_ge15=sum(1 for x in v if x[2]>=15)/len(v),mean_cand_exposure=avg([x[4] for x in v])))
    mid_cnt.sort()
    return dict(targets=n,strict_pm1_ge15=st/max(1,n),middle_ge15=mid/max(1,n),any_ge15=rx/max(1,n),median_middle_candidates=mid_cnt[len(mid_cnt)//2] if mid_cnt else None,by_c=tab,crit="M5' >= 0.8 of targets have >= 15 middle candidates (c' >= 2, |c-c'| <= 2); strict and relaxed shown for comparison",passed=bool(mid/max(1,n)>=0.8),
        note='candidate exposure c is counted over whole articles (full variant); the lead variant restricts the answer to the first ~350 tokens but candidates are not recounted for it')
def power_events(rows_by_variant):
    out={}
    for vn,(tpe,sd) in rows_by_variant.items():
        out[vn]={}
        for nev in(24,26,28,30):
            n=int(round(tpe*nev));out[vn][nev]=dict(n_targets=n,table=C.power_table(n))
        out[vn]['targets_per_event_mean']=tpe;out[vn]['targets_per_event_sd']=sd
        out[vn]['implied_deff']={f'icc_{r}':1+(tpe-1)*r for r in(0.0,0.1,0.2,0.3)}
    return out
def run(a):
    out=pathlib.Path(a.out);tp=out/'titles.pkl';cp=out/'clusters.pkl'
    assert tp.exists() and cp.exists(),('run the job-202 stages first (titles.pkl / clusters.pkl missing in)',str(out))
    res,ncl=get_results(a,tp,cp);ok=[r for r in res if r['ok']]
    full=[(r,N.variant(r,'full')) for r in ok];lead=[(r,N.variant(r,'lead')) for r in ok]
    pool=[(r,f,l) for (r,f),(r2,l) in zip(full,lead) if f['targets']]
    print(f'{len(ok)} clusters pass; pool of events with >= 1 target: {len(pool)}',flush=True)
    M=dict(n_title_clusters=ncl,n_sampled=len(res),n_pass=len(ok),pool=len(pool))
    M['reproduction']=reproduce(pool,a)
    cfgs=[('full',1,12,False),('full',1,12,True),('lead',2,12,False),('lead',2,12,True),('lead',2,20,False),('lead',2,20,True),('lead',2,26,False),('lead',2,26,True),('lead',2,30,False),('lead',2,30,True)]
    M['m4p']=[]
    for vn,idx,nev,fill in cfgs:
        t0=time.time();row=config(pool,idx,vn,nev,fill,a,a.draws);row.setdefault('variant',vn);row.setdefault('n_events',nev);row.setdefault('fill',fill);M['m4p'].append(row);print(f'  M4p {vn} {nev} events fill={fill}: {"skipped: "+row["skipped"] if "skipped" in row else "pass="+str(row["passed"])} ({time.time()-t0:.0f}s)',flush=True)
    M['m5p']=m5prime(res,a)
    tpe={}
    for vn,idx in(('full',1),('lead',2)):
        v=[len([t for t in t3[idx]['targets']]) for t3 in pool];m=avg(v);tpe[vn]=(m,math.sqrt(avg([(x-m)**2 for x in v])) if v else float('nan'))
    M['power']=power_events(tpe)
    json.dump(M,open(out/'e57_news96_report.json','w'),indent=1,default=str)
    open(out/'e57_news96_report.md','w').write(report(M,a));print(open(out/'e57_news96_report.md').read())
def fm(x,d=3):return 'n/a' if x is None or (isinstance(x,float) and math.isnan(x)) else (f'{x:.{d}f}' if isinstance(x,float) else str(x))
def report(M,a):
    f=lambda x:'PASS' if x else 'FAIL'
    L=['# E57 section 9.6 news re-measurement (CPU, regex proxy)','',f"clusters: title stage {M['n_title_clusters']}, text-stage sample {M['n_sampled']}, passing dedup/independence {M['n_pass']}, events with >= 1 target (pool) {M['pool']}. Same sample and seeds as job 202 (seed {a.seed}); rho = share of occurrences in the LAST of 2 rounds whose target has an earlier occurrence in the 8192 tokens before the chunk start.",'',
       '## 1 rho_win decomposition and reproduction of the relayed M4 (ORIGINAL layouts: event order and article order re-shuffled every round, 12 events)','',
       '| variant | layout | (i) other article | (ii) same article, same round | (iii) same article, previous round | (i)+(ii) | union (i,ii,iii) = original rho | relayed original | any other same-event article in window | occurrences |','|---|---|---|---|---|---|---|---|---|---|']
    for vn,x in M['reproduction'].items():
        for lay in('CLUSTER','SPREAD'):y=x[lay];L.append(f"| {vn} | {lay} | {fm(y['i'])} | {fm(y['ii'])} | {fm(y['iii'])} | {fm(y['i_or_ii'])} | {fm(y['all'])} | {y['relayed_union_rho']} | {fm(y['event_any'])} | {y['n_occ']} |")
    L+=['','Round order is random in the original layouts (yes). Article geometry (12 events):']+[f"- {vn}: mean article {x['mean_article_tokens']:.0f} tokens, mean articles per event {x['mean_articles_per_event']:.1f}, mean event total {x['mean_event_tokens']:.0f} tokens, targets {x['n_targets']}" for vn,x in M['reproduction'].items()]
    L+=['','Reproduction check: the union column should equal the relayed original rho (same sample, same seeds) up to differences of the pool; a mismatch is reported, not hidden.','',
       "## 2 M4' (CLUSTER = events contiguous, reshuffled every round; SPREAD = deterministic cycle, fixed order, same every round; filler = uniform tokens after every pass so that every pair of different articles of one event is >= 9000 tokens apart)",'',
       "| variant | events | filler | draws | CLUSTER (i)+(ii) | SPREAD (i) | SPREAD (i)+(ii) | SPREAD all sources | gap min (worst draw) | gap 5th pct | filler per pass | filler share | round chunks | updates @128 exposures | targets | mean c | M4' |",'|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in M['m4p']:
        if 'skipped' in r:L.append(f"| {r['variant']} | {r['n_events']} | {'yes' if r['fill'] else 'no'} | - | skipped: {r['skipped']} | | | | | | | | | | | | |");continue
        L.append(f"| {r['variant']} | {r['n_events']} | {'yes' if r['fill'] else 'no'} | {r['draws']} | {fm(r['CLUSTER']['i_or_ii'])} | {fm(r['SPREAD']['i'])} | {fm(r['SPREAD']['i_or_ii'])} | {fm(r['SPREAD']['all'])} | {fm(r['gap_min_worst'],0)} | {fm(r['gap_p5'],0)} | {fm(r['filler_per_pass'],0)} | {fm(r['filler_share'])} | {fm(r['round_chunks'],1)} | {fm(r['updates128'],0)} | {fm(r['n_targets'],1)} | {fm(r['mean_c'],2)} | **{f(r['passed'])}** |")
    L+=['',"- criteria: CLUSTER (i)+(ii) >= 0.5 and SPREAD (i) <= 0.2; filler share must be <= 0.30 for a filler row to count (design 9.4). 'SPREAD all sources' adds (iii), the copy of the same article in the previous round.",'- the loss of CLUSTER rho under 20-30 events is checked in the table, not assumed.','',
       "## 3 M5' middle candidates",'',f"- targets {M['m5p']['targets']}: strict (+-1) {M['m5p']['strict_pm1_ge15']:.3f}, middle (c' >= 2, |c-c'| <= 2) {M['m5p']['middle_ge15']:.3f}, relaxed (any) {M['m5p']['any_ge15']:.3f}; median number of middle candidates per target {M['m5p']['median_middle_candidates']}; criterion {M['m5p']['crit']}: **{f(M['m5p']['passed'])}**",'- '+M['m5p']['note'],'',
       "| answer exposure c | targets | candidates (any) | with c' >= 2 | within +-1 | middle | share middle >= 15 | share strict >= 15 | mean candidate exposure |",'|---|---|---|---|---|---|---|---|---|']
    for x in M['m5p']['by_c']:L.append(f"| {x['c']} | {x['targets']} | {x['mean_n_candidates']:.1f} | {x['mean_n_cand_c2']:.1f} | {x['mean_n_within1']:.1f} | {x['mean_n_middle']:.1f} | {x['share_middle_ge15']:.2f} | {x['share_strict_ge15']:.2f} | {x['mean_cand_exposure']:.2f} |")
    L+=['','## 4 power with 24-30 events (E56 G1 formula SE = 0.379/sqrt(n_eff), theta = max(0.10, 1.06/sqrt(n_eff)))','','The intra-event correlation of the outcome g cannot be measured on CPU; the DEFF table below follows from DEFF = 1 + (targets per event - 1) x ICC for ICC 0 / 0.1 / 0.2 / 0.3 and the E56 DEFF list 1.0 / 1.5 / 2.0.','']
    for vn,x in M['power'].items():
        L.append(f"- {vn}: targets per event mean {x['targets_per_event_mean']:.2f} (sd {x['targets_per_event_sd']:.2f}); implied DEFF "+', '.join(f"{k} -> {v:.2f}" for k,v in x['implied_deff'].items()))
    L+=['','| variant | events | n targets | DEFF | n_eff | theta | power @0.03 | @0.10 | @0.15 |','|---|---|---|---|---|---|---|---|---|']
    for vn,x in M['power'].items():
        for nev in(24,26,28,30):
            for p in x[nev]['table']:L.append(f"| {vn} | {nev} | {x[nev]['n_targets']} | {p['deff']} | {p['n_eff']:.1f} | {p['theta']:.3f} | {p['power_0.03']:.2f} | {p['power_0.1']:.2f} | {p['power_0.15']:.2f} |")
    return '\n'.join(L)+'\n'
def _selftest():
    # 12 events x 4 articles of 2500 tokens, one target per event seen in all 4 articles at offset 10
    ev=[[(e,i) for i in range(4)] for e in range(12)];tok={k:2500 for e in ev for k in e};occ={(e,0):{(e,i):[10] for i in range(4)} for e in range(12)}
    cl=rho_decomp(cluster_stream(ev,random.Random(0)),tok,occ,2);assert cl['i_or_ii']>=0.5,cl
    sp=rho_decomp(fixed_spread(ev,random.Random(0),2,0),tok,occ,2);assert sp['i']==0 and sp['iii']==0,sp         # cycle 30000 tokens > window, round 120000
    # short articles: 12 events x 350 tokens: cycle 4200 -> inside the window, filler lifts it
    tok2={k:350 for k in tok};sp2=rho_decomp(fixed_spread(ev,random.Random(0),2,0),tok2,occ,2);assert sp2['i']>0.9,sp2
    F=min_filler(ev,0,2,tok2);assert F is not None and 4000<F<6000,F
    inst,_=positions(fixed_spread(ev,random.Random(0),2,F),tok2);assert min(event_gaps(inst))>=GAP
    sp3=rho_decomp(fixed_spread(ev,random.Random(0),2,F),tok2,occ,2);assert sp3['i']==0,sp3
    # (ii)/(iii): one event, two articles, target twice in article 0; second round copies
    ev1=[[(0,0),(0,1)]];tk={(0,0):3000,(0,1):3000};oc={(0,0):{(0,0):[10,2500],(0,1):[10]}}
    d=rho_decomp(fixed_spread(ev1,random.Random(0),2,0),tk,oc,2);assert d['ii']>0 and d['iii']>=0
    assert abs(avg([1.0,float('nan'),3.0])-2.0)<1e-9
    print('e57_news96 selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw')
    ap.add_argument('--min-articles',type=int,default=8);ap.add_argument('--min-domains',type=int,default=8);ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-c',type=int,default=3)
    ap.add_argument('--max-clusters',type=int,default=3000);ap.add_argument('--max-per-event',type=int,default=12);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--draws',type=int,default=5);ap.add_argument('--rerun',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
