"""E57 big-cluster list (CPU only, no model, no GPU, NO new clustering, no download): re-processes the output of job 215 (<out>/e57_bigclusters/clusters_big.pkl, all clusters of the title-stage clustering without the 1500 limit) and lists the clusters that are coherent events by BOTH over-merge checks of job 215.
Selection (fixed before the data, Hans 2026-10-23): clusters with >= --min-members members (default 50) whose core coverage >= --core-min (0.5; share of ALL members whose title contains >= 2 of the 3 most frequent title tokens) AND whose strict re-split largest-component share >= --split-min (0.5; title Jaccard >= 0.35, date gap <= 3 days, hub tokens above 60 titles ignored, seeded sample of at most 3000 members; same functions as e57_bigclusters). The re-split is computed only for clusters that pass the core test (the funnel counts are reported).
Columns: rank (by members, then core), members, hostnames, first / last date, core, largest-component share, densest 7-day share, ETH-v1 on the first 1500 characters of up to --eth-sample seeded members (report column only, the rule is not changed), 2-3 title examples, RECURRING suspect (>= 30 % of the titles match quarterly / annual / results / schedule / exam / ranking / weekly type words) and PRODUCT suspect (>= 30 % match review / deal / price / buy / launch / product type words), both with the matching share.
Top of the report: funnel and counts per size band (50-100, 100-300, 300+). Writes <out>/e57_bigclusters/e57_biglist_report.md and e57_biglist.csv (all selected clusters). The report names clusters by article titles only and contains no article text."""
import argparse,json,sys,math,random,pickle,pathlib,time,collections,re,csv
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_news as N
import e57_freeze as FZ
import e57_bigclusters as BC
RECUR=re.compile(r'\b(quarter(?:ly)?|q[1-4]|earnings|results?|annual|yearly|weekly|daily|monthly|exams?|schedule|fixtures?|standings|rankings?|horoscope|lottery|powerball|forecast|top \d+|live updates?|live blog|what to know|headlines|stocks?|shares)\b',re.I)
PRODUCT=re.compile(r'\b(reviews?|best|deals?|sale|discount|coupon|prices?|buy(?:ing)?|vs\.?|specs?|unboxing|launch(?:es|ed)?|pre-?order|release date|iphone|galaxy|pixel|gift|amazon|black friday|\$\d+)\b',re.I)
def share(rx,titles):return sum(bool(rx.search(t)) for t in titles)/max(1,len(titles))
def band(n):return '50-100' if n<100 else '100-300' if n<300 else '300+'
def run(a):
    out=pathlib.Path(a.out);fd=out/'e57_bigclusters';cpb=fd/'clusters_big.pkl';tp=out/'titles.pkl'
    assert cpb.exists() and tp.exists(),('job-215 outputs missing: run job 215 first',str(cpb))
    rep=fd/'e57_biglist_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo):',rep);sys.exit(8)
    t0=time.time();D=pickle.load(open(tp,'rb'));CL=pickle.load(open(cpb,'rb'));cl=CL['clusters'];rng=random.Random(a.seed)
    big=[i for i,c in enumerate(cl) if len(c['members'])>=a.min_members];rows=[];n_core=0
    for n_,ci in enumerate(big):
        c=cl[ci];mem=c['members'];toks=[N.ttoks(D['title'][i]) for i in mem];core=BC.core_stats(toks)
        if core['core_coverage']<a.core_min:continue
        n_core+=1;chk=rng.sample(range(len(mem)),min(a.check_sample,len(mem)));dord=[N.ordinal(D['date'][i]) for i in mem]
        sp=BC.resplit([toks[j] for j in chk],[dord[j] for j in chk],a.strict_jac)
        if sp['largest_share']<a.split_min:continue
        titles=[D['title'][i] for i in mem];rows.append(dict(ci=ci,members=len(mem),hosts=c['n_hosts'],first=min(D['date'][i] for i in mem),last=max(D['date'][i] for i in mem),core=core['core_coverage'],top_tokens=core['top_tokens'][:3],split=sp,dates=BC.date_stats(dord),recur=share(RECUR,titles),product=share(PRODUCT,titles),
            examples=[titles[0]]+[titles[j] for j in rng.sample(range(len(mem)),min(2,len(mem)))]))
        if n_%500==0:print(f'  {n_}/{len(big)} clusters checked, {len(rows)} selected, {time.time()-t0:.0f}s',flush=True)
    rows.sort(key=lambda r:(-r['members'],-r['core']))
    for k,r in enumerate(rows,1):r['rank']=k
    # ETH-v1 on a text sample of the selected clusters: one pass over the shards
    ethm={r['ci']:rng.sample(cl[r['ci']]['members'],min(a.eth_sample,len(cl[r['ci']]['members']))) for r in rows}
    got=N.read_texts(D,sorted({i for v in ethm.values() for i in v})) if rows else {}
    for r in rows:
        why=FZ.ethics([got[D['rid'][i]][:1500] for i in ethm[r['ci']] if D['rid'][i] in got]);r['eth_pass']=not why;r['eth_why']=why;r['eth_n']=len([i for i in ethm[r['ci']] if D['rid'][i] in got])
    M=dict(n_clusters=len(cl),n_ge_min=len(big),n_core=n_core,n_selected=len(rows),bands={b:sum(band(r['members'])==b for r in rows) for b in('50-100','100-300','300+')},bands_all={b:sum(band(len(cl[i]['members']))==b for i in big) for b in('50-100','100-300','300+')},
        flags=dict(recur=sum(r['recur']>=0.3 for r in rows),product=sum(r['product']>=0.3 for r in rows),eth_reject=sum(not r['eth_pass'] for r in rows),clean=sum(r['recur']<0.3 and r['product']<0.3 and r['eth_pass'] for r in rows)),rows=rows,sec=time.time()-t0)
    json.dump(M,open(fd/'e57_biglist.json','w'),indent=1,default=str)
    with open(fd/'e57_biglist.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['rank','members','hosts','first','last','core','largest_component','densest7','eth_pass','recurring_share','product_share','title1','title2','title3'])
        for r in rows:w.writerow([r['rank'],r['members'],r['hosts'],r['first'],r['last'],f"{r['core']:.3f}",f"{r['split']['largest_share']:.3f}",f"{r['dates']['densest7_share']:.3f}",int(r['eth_pass']),f"{r['recur']:.2f}",f"{r['product']:.2f}"]+[t[:120] for t in r['examples']])
    txt=report(M,a);rep.write_text(txt);print(txt[:6000])
def report(M,a):
    L=['# E57 big-cluster list: coherent events by both over-merge checks (CPU)','',f"Source: job-215 clusters (no 1500 limit), {M['n_clusters']} clusters. Selection fixed before the data: members >= {a.min_members}; core coverage >= {a.core_min}; strict re-split largest-component share >= {a.split_min} (title Jaccard >= {a.strict_jac}, date gap <= 3 days, sample of at most {a.check_sample} members). ETH-v1 is a report column on the first 1500 characters of up to {a.eth_sample} members per cluster (rule unchanged). Run {M['sec']:.0f} s.",'',
       '## funnel and size bands','',f"- clusters in total {M['n_clusters']} -> with >= {a.min_members} members {M['n_ge_min']} -> core >= {a.core_min} {M['n_core']} -> also largest component >= {a.split_min}: **{M['n_selected']}**",
       f"- selected clusters by members: 50-100: {M['bands']['50-100']}; 100-300: {M['bands']['100-300']}; 300 or more: {M['bands']['300+']} (all clusters with >= {a.min_members} members for comparison: {M['bands_all']['50-100']} / {M['bands_all']['100-300']} / {M['bands_all']['300+']}).",
       f"- among the selected: recurring-type suspects (>= 30 % of the titles) {M['flags']['recur']}; product-type suspects {M['flags']['product']}; ETH-v1 rejected {M['flags']['eth_reject']}; neither suspect and ETH-v1 passed: {M['flags']['clean']}.",'',
       f"## list (by members, first {min(len(M['rows']),a.list_top)} of {M['n_selected']}; the csv has all)",'','| # | members | hostnames | first | last | core | largest component | densest 7 d | ETH-v1 | recurring share | product share | title examples |','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in M['rows'][:a.list_top]:
        L.append(f"| {r['rank']} | {r['members']} | {r['hosts']} | {r['first']} | {r['last']} | {r['core']:.2f} | {r['split']['largest_share']:.2f} | {r['dates']['densest7_share']:.2f} | {'pass' if r['eth_pass'] else 'REJECT '+'; '.join(r['eth_why'])[:60]} | {r['recur']:.2f}{' SUSPECT' if r['recur']>=0.3 else ''} | {r['product']:.2f}{' SUSPECT' if r['product']>=0.3 else ''} | "+' / '.join(t[:90].replace('|',' ') for t in r['examples'])+' |')
    L+=['','Suspect flags are keyword shares over the member titles (a screening aid, not a decision): recurring-type = quarterly / annual / results / schedule / exam / ranking / weekly words; product-type = review / deal / price / buy / launch / product words.','Titles are shown only to name the clusters; no article text is written to this report.']
    return '\n'.join(L)+'\n'
def _selftest():
    import tempfile,pyarrow as pa,pyarrow.parquet as pq
    td=pathlib.Path(tempfile.mkdtemp());fd=td/'e57_bigclusters';fd.mkdir();rid=[];host=[];title=[];date=[];text=[]
    import datetime
    dd=lambda n:int((datetime.date(2026,1,1)+datetime.timedelta(days=n)).strftime('%Y%m%d'))
    def add(t,d,h):rid.append(f'r{len(rid)}');host.append(h);title.append(t);date.append(d);text.append('Plain sentence about the event. '*60)
    for k in range(60):add('Zorvex pipeline blaze refinery',dd(9+k%5),f'ha{k}')                      # coherent event, 60 articles
    for k in range(60):add(f'Best deals price review gadget{k}',dd(k*2),f'hb{k}')                  # product-type, dispersed titles and dates
    for k in range(60):add(f'Quarterly earnings results company{k%3} report',dd(31+k%4),f'hc{k}')    # recurring-type
    pq.write_table(pa.table(dict(warc_record_id=rid,text=text)),td/'s0.parquet')
    D=dict(rid=rid,host=host,title=title,date=date,fidx=[0]*len(rid),files=[str(td/'s0.parquet')],n_rows=len(rid));pickle.dump(D,open(td/'titles.pkl','wb'))
    cl=[dict(members=list(range(g*60,g*60+60)),n_hosts=60,dates=[0,0],title=title[g*60]) for g in range(3)]
    pickle.dump(dict(clusters=cl,rejected={},n_titles=len(rid),n_buckets=3,n_skipped_titles=0),open(fd/'clusters_big.pkl','wb'))
    a=argparse.Namespace(out=str(td),min_members=50,core_min=0.5,split_min=0.5,strict_jac=0.35,check_sample=3000,eth_sample=10,seed=3,rerun=True,list_top=50)
    run(a);M=json.load(open(fd/'e57_biglist.json'))
    names=[r['examples'][0] for r in M['rows']];assert any('Zorvex' in n for n in names),names
    ev=[r for r in M['rows'] if 'Zorvex' in r['examples'][0]][0];assert ev['core']==1.0 and ev['split']['largest_share']==1.0 and ev['recur']<0.3 and ev['product']<0.3 and ev['eth_pass']
    rc=[r for r in M['rows'] if 'Quarterly' in r['examples'][0]];assert rc and rc[0]['recur']>=0.3,M['rows']
    pr=[r for r in M['rows'] if 'Best deals' in r['examples'][0]];assert pr and pr[0]['product']>=0.3,'a coherent product-title cluster passes both tests but is flagged'
    assert M['n_ge_min']==3 and M['n_selected']==3 and (fd/'e57_biglist.csv').exists()
    print('e57_biglist selftest ok: funnel',M['n_ge_min'],'->',M['n_core'],'->',M['n_selected'],'flags',M['flags'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw');ap.add_argument('--min-members',type=int,default=50);ap.add_argument('--core-min',type=float,default=0.5);ap.add_argument('--split-min',type=float,default=0.5)
    ap.add_argument('--strict-jac',type=float,default=0.35);ap.add_argument('--check-sample',type=int,default=3000);ap.add_argument('--eth-sample',type=int,default=20);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--list-top',type=int,default=300);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
