"""E57 big-cluster check (CPU only, no model, no GPU, NO download). The title-stage clustering of job 202 refused clusters above --max-cluster 1500 members ('size>max (chained)'); this job redoes ONLY the title-stage clustering with a much larger limit (default 20000) into a SEPARATE folder (<out>/e57_bigclusters/clusters_big.pkl; the job-202 clusters.pkl is not touched) and reports, for the largest clusters:
  - size distribution (members, hostnames), how many clusters are above the old limit and how many articles they hold, rejection counts of the new run;
  - for the top --top clusters by members: title examples (the first title and seeded random members), hostnames, first / last date;
  - over-merge check: (a) the most frequent title tokens and their member share, core coverage (share of members that contain >= 2 of the 3 most frequent tokens); (b) a stricter re-split of the cluster (title Jaccard >= 0.35, date gap <= 3 days, on a seeded sample of at most --check-sample members): share of the largest component, number of components with >= 8 members, singleton share; (c) date concentration: span, share of members in the densest 7-day window, number of days holding >= 5 % of the members;
  - ETH-v1 (e57_freeze.ethics) on the first 1500 characters of up to --eth-sample seeded members of each listed cluster (a sample proxy of the lead-text filter; the full filter runs on the lead texts of the final bank).
PROXIES: title-token clusters (single-link union-find, the chaining risk is exactly what the check looks at). The report names events by article titles only and writes no article text. Report: <out>/e57_bigclusters/e57_bigclusters_report.md (+ .json)."""
import argparse,json,sys,math,random,pickle,pathlib,time,collections,itertools
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_news as N
import e57_freeze as FZ
def pct(v,q):
    v=sorted(v);return v[min(len(v)-1,int(q*(len(v)-1)))] if v else float('nan')
def date_stats(dates):
    """dates: ordinal list -> span, share in the densest 7-day window, days with >= 5 % of the members"""
    c=collections.Counter(dates);n=len(dates);lo,hi=min(c),max(c);best=0
    for d in range(lo,hi+1):best=max(best,sum(c.get(d+k,0) for k in range(7)))
    return dict(span_days=hi-lo+1,densest7_share=best/n,days_ge5pct=sum(1 for v in c.values() if v>=0.05*n))
def core_stats(toks):
    df=collections.Counter(w for ts in toks for w in set(ts));n=len(toks);top=[w for w,_ in df.most_common(10)]
    core=sum(1 for ts in toks if len(set(ts)&set(top[:3]))>=2)/n
    return dict(top_tokens=[(w,df[w]/n) for w in top],core_coverage=core)
def resplit(toks,dord,jac=0.35,gap=3,df_cap=60):
    """stricter single-link components inside one cluster: pairs sharing a token (token df <= df_cap inside the cluster), Jaccard of the title token sets >= jac, date gap <= gap"""
    n=len(toks);par=list(range(n))
    def find(x):
        while par[x]!=x:par[x]=par[par[x]];x=par[x]
        return x
    post=collections.defaultdict(list)
    for i,ts in enumerate(toks):
        for w in set(ts):post[w].append(i)
    sets=[set(t) for t in toks];seen=set()
    for w,v in post.items():
        if len(v)<2 or len(v)>df_cap:continue
        for p,q in itertools.combinations(v,2):
            if (p,q) in seen or abs(dord[p]-dord[q])>gap:continue
            seen.add((p,q))
            if len(sets[p]&sets[q])/max(1,len(sets[p]|sets[q]))>=jac:
                a,b=find(p),find(q)
                if a!=b:par[b]=a
    comp=collections.Counter(find(i) for i in range(n));sizes=sorted(comp.values(),reverse=True)
    return dict(n=n,largest_share=sizes[0]/n,components_ge8=sum(s>=8 for s in sizes),singleton_share=sum(s==1 for s in sizes)/n,sizes_top=sizes[:6])
def run(a):
    out=pathlib.Path(a.out);tp=out/'titles.pkl';assert tp.exists(),('titles.pkl missing in',str(out))
    fd=out/'e57_bigclusters';fd.mkdir(exist_ok=True);rep=fd/'e57_bigclusters_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo):',rep);sys.exit(8)
    t0=time.time();D=pickle.load(open(tp,'rb'));cpb=fd/'clusters_big.pkl'
    if a.rerun or not cpb.exists():
        a2=argparse.Namespace(df_max=a.df_max,date_gap=a.date_gap,title_jac=a.title_jac,min_articles=a.min_articles,min_domains=a.min_domains,max_cluster=a.max_cluster);N.stage_cluster(a2,tp,cpb)
    CL=pickle.load(open(cpb,'rb'));cl=CL['clusters'];print(f'  clustering {time.time()-t0:.0f}s: {len(cl)} clusters',flush=True)
    old=None;cp_old=out/'clusters.pkl'
    if cp_old.exists():old=pickle.load(open(cp_old,'rb'))
    sizes=sorted((len(c['members']) for c in cl),reverse=True);big=[c for c in cl if len(c['members'])>a.old_limit]
    order=sorted(range(len(cl)),key=lambda i:-len(cl[i]['members']))[:a.top];rng=random.Random(a.seed);rows=[];import e57_news as N2
    # ETH text sample: one shard pass for all listed clusters
    ethm={ci:rng.sample(cl[ci]['members'],min(a.eth_sample,len(cl[ci]['members']))) for ci in order}
    got=N.read_texts(D,sorted({i for v in ethm.values() for i in v}));print(f'  text sample read: {len(got)} articles, {time.time()-t0:.0f}s',flush=True)
    for rk,ci in enumerate(order,1):
        c=cl[ci];mem=c['members'];toks=[N.ttoks(D['title'][i]) for i in mem];dord=[N.ordinal(D['date'][i]) for i in mem]
        chk=rng.sample(range(len(mem)),min(a.check_sample,len(mem)))
        eth=FZ.ethics([got[D['rid'][i]][:1500] for i in ethm[ci] if D['rid'][i] in got])
        ex=[D['title'][mem[0]]]+[D['title'][mem[j]] for j in rng.sample(range(len(mem)),min(3,len(mem)))]
        rows.append(dict(rank=rk,members=len(mem),hosts=c['n_hosts'],first=min(D['date'][i] for i in mem),last=max(D['date'][i] for i in mem),above_old_limit=len(mem)>a.old_limit,titles=[t[:100] for t in ex],core=core_stats(toks),split=resplit([toks[j] for j in chk],[dord[j] for j in chk],a.strict_jac),dates=date_stats(dord),eth_pass=not eth,eth_why=eth,eth_n=len([i for i in ethm[ci] if D['rid'][i] in got])))
    M=dict(n_clusters=len(cl),rejected=CL['rejected'],old_rejected=old['rejected'] if old else None,old_n_clusters=len(old['clusters']) if old else None,above_old=len(big),articles_above_old=sum(len(c['members']) for c in big),size_q={q:pct(sizes,q) for q in(0.5,0.9,0.99,1.0)},size_buckets={b:sum(s>=b for s in sizes) for b in(8,50,200,500,1500,3000,6000,12000)},top=rows,sec=time.time()-t0,max_cluster=a.max_cluster)
    json.dump(M,open(fd/'e57_bigclusters.json','w'),indent=1,default=str);txt=report(M,a);rep.write_text(txt);print(txt)
def f3(x):return f'{x:.3f}'
def report(M,a):
    L=['# E57 big-cluster check (title stage, CPU)','',f"PROXY: title-token single-link clusters. This run uses max cluster size {M['max_cluster']} (job 202: 1500) in a separate folder; the job-202 clusters.pkl is untouched. Run {M['sec']:.0f} s.",'',
       f"- clusters now: {M['n_clusters']} (job 202: {M['old_n_clusters']}); rejected now: {M['rejected']}; rejected in job 202: {M['old_rejected']}.",
       f"- clusters above the old limit ({a.old_limit} members): {M['above_old']} holding {M['articles_above_old']} article records. Size quantiles 50 / 90 / 99 / max: {M['size_q'][0.5]} / {M['size_q'][0.9]} / {M['size_q'][0.99]} / {M['size_q'][1.0]}. Clusters with at least n members: "+'; '.join(f'n >= {b}: {v}' for b,v in M['size_buckets'].items()),'',
       f"## top {len(M['top'])} clusters by members","",'Over-merge check columns: core = share of members with >= 2 of the 3 most frequent title tokens; split = after a stricter re-split (title Jaccard >= '+str(a.strict_jac)+', date gap <= 3 days, sample of at most '+str(a.check_sample)+' members): share in the largest component / components with >= 8 members / singleton share; dates = span in days / share of members in the densest 7-day window / days holding >= 5 % of the members. ETH-v1 on the first 1500 characters of a seeded sample of members.','',
       '| # | members | hostnames | first | last | above old limit | core | split: largest / comps >= 8 / singletons | dates: span / densest 7 d / days >= 5 % | ETH-v1 (sample) | most frequent title tokens (share) | title examples |','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for x in M['top']:
        tt=', '.join(f"{w} {s:.2f}" for w,s in x['core']['top_tokens'][:5]);sp=x['split'];d=x['dates']
        L.append(f"| {x['rank']} | {x['members']} | {x['hosts']} | {x['first']} | {x['last']} | {'yes' if x['above_old_limit'] else 'no'} | {f3(x['core']['core_coverage'])} | {f3(sp['largest_share'])} / {sp['components_ge8']} / {f3(sp['singleton_share'])} | {d['span_days']} / {f3(d['densest7_share'])} / {d['days_ge5pct']} | {'pass' if x['eth_pass'] else 'REJECT '+'; '.join(x['eth_why'])[:70]} (n={x['eth_n']}) | {tt} | "+' / '.join(t.replace('|',' ') for t in x['titles'])+' |')
    L+=['','Reading guide (fixed before the data): a cluster looks like ONE event if core coverage >= 0.5, the strict re-split keeps the largest component >= 0.5 of the members and the densest 7-day window holds >= 0.4 of the members; otherwise it is probably a chain of several events or a recurring topic and would have to be split (stricter threshold or embeddings) before use.','','Titles are shown only to name the clusters; no article text is written to this report.']
    return '\n'.join(L)+'\n'
def _selftest():
    import tempfile,pyarrow as pa,pyarrow.parquet as pq
    td=pathlib.Path(tempfile.mkdtemp());rid=[];host=[];title=[];date=[];text=[]
    def add(t,d,h):rid.append(f'r{len(rid)}');host.append(h);title.append(t);date.append(d);text.append('Plain sentence about the event. '*60)
    for k in range(40):add('Zorvex pipeline blaze',20260110+k%6,f'ha{k}')        # event A
    for k in range(40):add('Quillon summit treaty',20260119+k%4,f'hb{k}')      # event B
    for k in range(6):add('Zorvex pipeline summit treaty',20260117,f'hc{k}')                                    # bridge titles sharing one rare token with each event
    for k in range(12):add(f'Unrelated weather piece number{k}',20260201+k,f'hd{k}')
    pq.write_table(pa.table(dict(warc_record_id=rid,text=text)),td/'s0.parquet')
    D=dict(rid=rid,host=host,title=title,date=date,fidx=[0]*len(rid),files=[str(td/'s0.parquet')],n_rows=len(rid));pickle.dump(D,open(td/'titles.pkl','wb'))
    a=argparse.Namespace(out=str(td),rerun=True,df_max=2000,date_gap=3,title_jac=0.2,min_articles=8,min_domains=8,max_cluster=20000,old_limit=60,top=3,seed=3,eth_sample=10,check_sample=500,strict_jac=0.5)
    run(a);M=json.load(open(td/'e57_bigclusters'/'e57_bigclusters.json'))
    assert M['n_clusters']>=1 and M['top'][0]['members']>=60,M['top'][0]['members']
    assert M['top'][0]['above_old_limit'] and M['top'][0]['split']['largest_share']<1.0,M['top'][0]['split']      # the bridge-merged cluster falls apart under the stricter split
    assert pct([1,2,3,4],0.5)==2 and date_stats([1,1,2,9])['span_days']==9
    print('e57_bigclusters selftest ok: top cluster',M['top'][0]['members'],'members, largest strict component share',round(M['top'][0]['split']['largest_share'],2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw');ap.add_argument('--max-cluster',type=int,default=20000);ap.add_argument('--old-limit',type=int,default=1500);ap.add_argument('--top',type=int,default=50)
    ap.add_argument('--df-max',type=int,default=2000);ap.add_argument('--date-gap',type=int,default=3);ap.add_argument('--title-jac',type=float,default=0.2);ap.add_argument('--min-articles',type=int,default=8);ap.add_argument('--min-domains',type=int,default=8)
    ap.add_argument('--strict-jac',type=float,default=0.35);ap.add_argument('--check-sample',type=int,default=3000);ap.add_argument('--eth-sample',type=int,default=40);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
