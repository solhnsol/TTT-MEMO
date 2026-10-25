"""E57 v4 freeze (design section 11.3; CPU only, no model, no GPU, no download). From the job-202/206 files (titles.pkl, clusters.pkl, analyze_cache_f1.pkl or a redone text stage) it
  1. draws the 30 events: pool = events with >= 5 lead targets (c >= 3 in the lead variant, f >= 2); a seeded permutation of the pool (FREEZE_SEED); an automatic ethics filter (ETH-v1, below) is applied after the draw and the same permutation continues until 30 events are accepted (rule fixed before the result);
  2. freezes the lead texts (first sentences up to ~350 tokens), the fixed-order cyclic round (event order and article order shuffled ONCE with ORDER_SEED, one article per event per pass, articles end with a blank-line separator, the round is padded with blank-line tokens to a multiple of 1024 tokens) as token ids, and the target / question bank with candidate tiers (strict +-1, middle c' >= 2 and |c-c'| <= 2, relaxed) and the confirmation subset (c in {3,4} with >= 10 middle candidates);
  3. records G1 (n, n_eff, theta, power for the main bank and the confirmation subset), G2 (rho_win of the frozen layout: CLUSTER vs the fixed-order SPREAD, decomposed) and G3 (RoPE position check, e57_rope).
Outputs in <out>/e57_freeze/: round_ids.npy, articles.jsonl (lead texts: publisher copyright, stays on the server), events.json, bank.json, freeze_manifest.json (sha256 of every file), e57_freeze_report.md.
ETH-v1 (automatic, conservative): an event is rejected when (a) >= 20 % of its lead texts contain a violence / death / abuse / arrest keyword, (b) any lead text contains an e-mail address, a phone number or a street address, (c) a lead text names a person by age ('NN-year-old') in the same article as such a keyword, (d) it mentions a minor with such a keyword. The rejected events and reasons are listed in the report. PROXIES: regex facts, title-token clusters, whole-article candidate exposures c' (c is the lead count)."""
import argparse,json,sys,math,random,pickle,pathlib,time,collections,hashlib,re
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
import e57_news96 as Q
import e57_news10 as Q10
import e57_rope as RP
FREEZE_SEED=57101;ORDER_SEED=57102;CH=1024;SEP='\n\n'
VIOL=re.compile(r'\b(kill(?:ed|ing|s)?|murder(?:ed|s)?|stabb(?:ed|ing)|shot dead|shooting|rape[ds]?|sexual(?:ly)?|abus(?:e|ed|ing)|suicid\w*|died|dies|death|deaths|dead|body found|bodies|victims?|missing|arrest(?:ed|s)?|charged|sentenced|assault(?:ed)?|hostages?|massacre|funeral|corpse|injur(?:ed|ies)|crash(?:ed)?)\b',re.I)
PII=re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+|\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b|\b\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Lane|Drive|Dr\.)\b')
AGE=re.compile(r'\b\d{1,2}-year-old\b|\baged \d{1,2}\b',re.I)
MINOR=re.compile(r'\b(child|children|boy|girl|teen(?:ager)?s?|minor|toddler|infant|baby|pupil|schoolchild(?:ren)?)\b',re.I)
def ethics(lead_texts):
    """-> list of reasons (empty = accepted)"""
    why=[];n=len(lead_texts)
    nv=sum(bool(VIOL.search(t)) for t in lead_texts)
    if n and nv/n>=0.20:why.append(f'a: {nv}/{n} lead texts with a violence / death / arrest keyword')
    if any(PII.search(t) for t in lead_texts):why.append('b: e-mail / phone / street address in a lead text')
    if any(AGE.search(t) and VIOL.search(t) for t in lead_texts):why.append('c: a person named by age in an article with a violence keyword')
    if any(MINOR.search(t) and VIOL.search(t) for t in lead_texts):why.append('d: a minor in an article with a violence keyword')
    return why
def sample_clusters(a,cp):
    CL=pickle.load(open(cp,'rb'));cl=CL['clusters'];rng=random.Random(a.seed);return (cl if len(cl)<=a.max_clusters else rng.sample(cl,a.max_clusters)),len(cl)
def lead_text(r,ai):
    k=r['kept'][ai];return ' '.join(r['sents'][ai][:k])
def analyze_detail(D,samp,idxs,a):
    """text stage with detail for the clusters samp[idxs] (one pass over the shards); -> {idx: result}"""
    members=sorted({i for ci in idxs for i in samp[ci]['members']});got=N.read_texts(D,members);jobs=[];meta=[]
    for ci in idxs:
        c=samp[ci];ids=[(i,D['rid'][i]) for i in c['members'] if D['rid'][i] in got]
        jobs.append(dict(title=c['title'],texts=[got[r] for _,r in ids],hosts=[D['host'][i] for i,_ in ids],args=dict(dup_jac=a.dup_jac,min_articles=a.min_articles,min_domains=a.min_domains,min_c=a.min_c,min_f=1,detail=True)));meta.append([r for _,r in ids])
    if a.workers>1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as p:res=p.map(N.analyze,jobs,chunksize=2)
    else:res=[N.analyze(j) for j in jobs]
    out={}
    for ci,r,rids in zip(idxs,res,meta):
        if r['ok']:r['rids']=[rids[k] for k in r['keep']]
        out[ci]=r
    return out
def run(a):
    out=pathlib.Path(a.out);tp=out/'titles.pkl';cp=out/'clusters.pkl';assert tp.exists() and cp.exists(),('job-202 stage files missing in',str(out))
    fd=out/'e57_freeze';fd.mkdir(exist_ok=True);rep=fd/'e57_freeze_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo):',rep);sys.exit(8)
    a.min_f=1;res1,ncl=Q.get_results(a,tp,cp);D=pickle.load(open(tp,'rb'));samp,_=sample_clusters(a,cp);assert len(samp)==len(res1),(len(samp),len(res1))
    res=Q10.with_f(res1,2)
    pool_idx=[i for i,r in enumerate(res) if r['ok'] and len(N.variant(r,'lead')['targets'])>=5]
    print(f'pool of events with >= 5 lead targets: {len(pool_idx)} (sample {len(samp)} clusters of {ncl})',flush=True)
    perm=random.Random(FREEZE_SEED).sample(pool_idx,len(pool_idx));pos=0;chosen=[];rejected=[];seen=0;t0=time.time()
    while len(chosen)<a.n_events and pos<len(perm):
        batch=perm[pos:pos+max(12,(a.n_events-len(chosen))+10)];pos+=len(batch)
        det=analyze_detail(D,samp,batch,a);print(f'  detail text stage for {len(batch)} drawn events, {time.time()-t0:.0f}s',flush=True)
        for ci in batch:
            if len(chosen)>=a.n_events:break
            r=det[ci];seen+=1
            if not r['ok']:rejected.append(dict(title=samp[ci]['title'],why=['analysis failed on the second pass']));continue
            r2=dict(r);r2['targets']=[t for t in r['targets'] if t['f']>=2];V=N.variant(r2,'lead');r2['kept']=V['kept']
            assert len(V['targets'])==len(N.variant(res[ci],'lead')['targets']),('second pass differs from the cached first pass',ci)
            leads=[lead_text(r2,ai) for ai in range(len(r2['arts']))];why=ethics(leads)
            if why:rejected.append(dict(title=samp[ci]['title'],why=why));continue
            chosen.append(dict(ci=ci,r=r2,V=V,leads=leads))
    assert len(chosen)==a.n_events,(f'only {len(chosen)} events accepted after {pos} draws')
    enc=N.enc();sep_ids=enc.encode(SEP,add_special_tokens=False).ids;pad_id=sep_ids[0]
    # events: cap articles, fixed order
    orng=random.Random(ORDER_SEED);evorder=list(range(len(chosen)));orng.shuffle(evorder);events=[];artinfo={};tokof={};occ={};targets=[]
    for ei,e_i in enumerate(evorder):
        ch=chosen[e_i];r=ch['r'];V=ch['V'];n_art=len(r['arts']);arts=list(range(n_art))
        if n_art>a.max_per_event:arts=sorted(random.Random(ORDER_SEED+1000+ei).sample(arts,a.max_per_event))
        random.Random(ORDER_SEED+2000+ei).shuffle(arts)
        keys=[]
        for ai in arts:
            ids=enc.encode(ch['leads'][ai],add_special_tokens=False).ids+sep_ids;key=(ei,ai);keys.append(key);tokof[key]=len(ids)
            artinfo[key]=dict(event=ei,art=ai,rid=r['rids'][ai],host=r['arts'][ai]['host'],n_sents_kept=V['kept'][ai],n_tokens=len(ids),text=ch['leads'][ai],ids=ids)
        events.append(keys)
        for ti,t in enumerate(V['targets']):
            o={(ei,ai):v for ai,v in t['occ_tok'].items() if (ei,ai) in tokof}
            if len(o)>=3:occ[(ei,len(targets))]=o;targets.append(dict(event=ei,e_i=e_i,t=t,c=len(o),keys=sorted(o)))
    # the round: one article per event per pass, fixed order; padded to whole chunks
    mx=max(len(e) for e in events);order=[(e[i]) for i in range(mx) for e in events if i<len(e)]
    ids_round=[x for key in order for x in artinfo[key]['ids']];real_tokens=len(ids_round);pad=(-real_tokens)%CH;ids_round+= [pad_id]*pad
    arr=np.asarray(ids_round,np.int32);rc=len(arr)//CH
    np.save(fd/'round_ids.npy',arr)
    # bank
    bank=[]
    for ti,tg in enumerate(targets):
        t=tg['t'];ch=chosen[tg['e_i']];r=ch['r'];det=r['pool_detail'].get(f"{t['ty']}|{t['key']}",[]);c=tg['c']
        cands=[dict(surface=s,key=k2,c_prime=cn) for ty2,k2,s,cn in det if k2!=t['key']]
        tie=lambda x:hashlib.md5((str(FREEZE_SEED)+x['key']).encode()).hexdigest()
        order_c=sorted(cands,key=lambda x:(abs(x['c_prime']-c),tie(x)))
        mid=[x for x in order_c if x['c_prime']>=2 and abs(x['c_prime']-c)<=2];strict=[x for x in order_c if abs(x['c_prime']-c)<=1]
        occs=[]
        for ai,sidx in t['occ'].items():
            if (tg['event'],ai) not in tokof:continue
            si=[s for s in sidx if s<ch['V']['kept'][ai]]
            if not si:continue
            facts=r['facts'][ai][si[0]];sf=[f[2] for f in facts if (f[0],f[1]) in((t['ty'],t['key']),('NAME',t['S']))]
            occs.append(dict(rid=r['rids'][ai],sent_idx=si[0],sentence=r['sents'][ai][si[0]],masked=C.masked(r['sents'][ai][si[0]],sf)))
        bank.append(dict(id=ti,event=tg['event'],type=t['ty'],subject_key=t['S'],answer_key=t['key'],answer_surface=t['surface'],c_lead=c,f_forms=t['f'],occurrences=occs,
            cand_strict=[dict(x) for x in strict[:16]],cand_middle=[dict(x) for x in mid[:16]],n_strict=len(strict),n_middle=len(mid),n_relaxed=len(cands),cand_relaxed=[dict(x) for x in order_c[:16]],
            main=True,confirm=bool(c in(3,4) and len(mid)>=10)))
    json.dump(bank,open(fd/'bank.json','w'),indent=1)
    with open(fd/'articles.jsonl','w') as f:
        for key in order:
            x=dict(artinfo[key]);x.pop('ids');f.write(json.dumps(x)+'\n')
    json.dump(dict(freeze_seed=FREEZE_SEED,order_seed=ORDER_SEED,events=[dict(event=ei,title=chosen[evorder[ei]]['r']['title'],n_articles=len(e),articles=[artinfo[k]['rid'] for k in e]) for ei,e in enumerate(events)],
        order=[artinfo[k]['rid'] for k in order],rejected=rejected,drawn=seen),open(fd/'events.json','w'),indent=1)
    # G2 on the frozen layout (real token counts), 20 seeded CLUSTER layouts vs the fixed SPREAD, all targets and c in {3,4}
    def rho(cset):
        oc={k:v for k,v in occ.items() if len(v) in cset} if cset else occ;rows=[]
        for d in range(a.draws):
            cl=Q.rho_decomp(Q.cluster_stream(events,random.Random(a.seed+d),2),tokof,oc,2);rows.append(cl)
        spread_stream=[(k,r_) for r_ in range(2) for k in order]                                    # the frozen fixed round, twice
        sp=Q.rho_decomp(spread_stream,tokof,oc,2);inst,total=Q.positions(spread_stream,tokof);g=sorted(Q.event_gaps(inst))
        return dict(CLUSTER={k:Q.avg([x[k] for x in rows]) for k in('i','ii','iii','i_or_ii','all','event_any')},SPREAD=sp,n_targets=len(oc),gap_min=g[0] if g else None,gap_p5=g[int(0.05*(len(g)-1))] if g else None)
    G2=dict(all=rho(None),c34=rho({3,4}))
    # G1
    nmain=len(bank);nconf=sum(b['confirm'] for b in bank);m=nmain/len(events);G1=dict(n_main=nmain,n_confirm=nconf,targets_per_event=m,implied_deff={f'icc_{x}':1+(m-1)*x for x in(0.0,0.1,0.2,0.3)},main=C.power_table(nmain),confirm=C.power_table(nconf) if nconf else [],
        c_hist=dict(collections.Counter(min(b['c_lead'],6) for b in bank)),n_c34=sum(b['c_lead'] in(3,4) for b in bank),middle_ge10_among_c34=sum(1 for b in bank if b['c_lead'] in(3,4) and b['n_middle']>=10),
        by_type=dict(collections.Counter(b['type'] for b in bank)))
    G3=RP.run(len(arr),rec_rounds=2,seq_len=a.seq_len,rounds_max=a.rounds_max)
    stats=dict(events=len(events),articles=len(order),real_tokens=real_tokens,pad_tokens=pad,round_tokens=len(arr),round_chunks=rc,mean_article_tokens=real_tokens/len(order),est_updates={f'R={R}':R*rc for R in(4,8,12,16,20,24,28,32)},
        sha256_round_ids=hashlib.sha256(arr.tobytes()).hexdigest(),pad_id=pad_id,sep_ids=sep_ids)
    man=dict(stats=stats,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(fd.glob('*')) if p.is_file() and p.name!='freeze_manifest.json'},G1=G1,G2=G2,G3={k:v for k,v in G3.items() if k!='rows'})
    json.dump(man,open(fd/'freeze_manifest.json','w'),indent=1,default=str);open(rep,'w').write(report(man,rejected,G3,a,len(pool_idx),ncl,seen));print(open(rep).read())
def fm(x,d=3):return Q.fm(x,d)
def report(man,rejected,G3,a,npool,ncl,seen):
    S=man['stats'];G1=man['G1'];G2=man['G2']
    L=['# E57 v4 freeze report (CPU, regex proxy)','',f"- pool of events with >= 5 lead targets: {npool} (text-stage sample of {ncl} title-stage clusters); seeded permutation (FREEZE_SEED {FREEZE_SEED}); {seen} events examined, {len(rejected)} rejected by ETH-v1, {S['events']} accepted",
       f"- fixed round: {S['articles']} articles, {S['real_tokens']} tokens (mean {S['mean_article_tokens']:.0f} per article) + {S['pad_tokens']} padding tokens = {S['round_tokens']} = **{S['round_chunks']} chunks**; updates for R = 4 / 8 / 12 / 16 / 32 rounds: {S['est_updates']['R=4']} / {S['est_updates']['R=8']} / {S['est_updates']['R=12']} / {S['est_updates']['R=16']} / {S['est_updates']['R=32']}",
       f"- round_ids.npy sha256 {S['sha256_round_ids']}; files frozen in freeze_manifest.json",'',
       '## ETH-v1 rejections','']+([f"- {x['title'][:90]}: {'; '.join(x['why'])}" for x in rejected] or ['- none'])
    L+=['','## G1 bank and power (E56 G1 formula SE = 0.379/sqrt(n_eff), theta = max(0.10, 1.06/sqrt(n_eff)))','',f"- main bank (c >= 3 in the lead variant, f >= 2): **n = {G1['n_main']}** ({G1['targets_per_event']:.2f} per event); by type {G1['by_type']}; c histogram (6 = 6 or more) {G1['c_hist']}",
        f"- confirmation subset (c in {{3,4}} with >= 10 middle candidates): **n = {G1['n_confirm']}** of {G1['n_c34']} targets with c in {{3,4}} ({G1['middle_ge10_among_c34']} have >= 10 middle candidates, fulfilment {G1['middle_ge10_among_c34']/max(1,G1['n_c34']):.3f}); the confirmation subset has LOW power, see the table",
        f"- implied DEFF = 1 + (targets per event - 1) x ICC: {', '.join(f'{k} -> {v:.2f}' for k,v in G1['implied_deff'].items())} (the ICC of the outcome g is not measurable on CPU)",'','| bank | DEFF | n_eff | theta | power @0.03 | @0.10 | @0.15 |','|---|---|---|---|---|---|---|']
    for nm in('main','confirm'):
        for p in G1[nm]:L.append(f"| {nm} | {p['deff']} | {p['n_eff']:.1f} | {p['theta']:.3f} | {p['power_0.03']:.2f} | {p['power_0.1']:.2f} | {p['power_0.15']:.2f} |")
    L+=['',"## G2 rho_win of the frozen layout (2 rounds, last round; CLUSTER = 20 seeded layouts with events contiguous, SPREAD = the frozen fixed round)",'','| targets | layout | (i) other article | (ii) same article same round | (iii) previous round copy | (i)+(ii) | any same-event article in window | gap min | gap 5th pct |','|---|---|---|---|---|---|---|---|---|']
    for nm,k in(('all (c >= 3)','all'),('c in {3,4}','c34')):
        g=G2[k]
        for lay in('CLUSTER','SPREAD'):y=g[lay];L.append(f"| {nm} ({g['n_targets']}) | {lay} | {fm(y['i'])} | {fm(y['ii'])} | {fm(y['iii'])} | {fm(y['i_or_ii'])} | {fm(y['event_any'])} | {fm(g['gap_min'],0) if lay=='SPREAD' else ''} | {fm(g['gap_p5'],0) if lay=='SPREAD' else ''} |")
    L+=['','- criterion M4\': CLUSTER (i)+(ii) >= 0.5 and SPREAD (i) <= 0.2.','',RP.report(G3)]
    return '\n'.join(L)+'\n'
def _selftest():
    assert ethics(['Officials announced the budget on Tuesday.'])==[]
    assert 'a' == ethics(['A man was killed in the crash.','Two people died.'])[0][0]
    assert any(w.startswith('b') for w in ethics(['Contact us at press@example.com for details.']))
    assert any(w.startswith('c') for w in ethics(['The 34-year-old was arrested on Monday.','Fine.','Fine.','Fine.','Fine.','Fine.','Fine.','Fine.','Fine.','Fine.']))
    assert ethics(['Local children enjoyed the new park.']*5)==[]
    print('e57_freeze selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw')
    ap.add_argument('--min-articles',type=int,default=8);ap.add_argument('--min-domains',type=int,default=8);ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-c',type=int,default=3)
    ap.add_argument('--max-clusters',type=int,default=3000);ap.add_argument('--max-per-event',type=int,default=12);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--draws',type=int,default=20)
    ap.add_argument('--n-events',type=int,default=30);ap.add_argument('--seq-len',type=int,default=131072);ap.add_argument('--rounds-max',type=int,default=32);ap.add_argument('--rerun',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
