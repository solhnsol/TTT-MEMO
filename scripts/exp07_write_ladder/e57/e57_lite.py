"""E57-lite freeze (CPU only, no model, no GPU, NO download, no new clustering; job 223; E57 section 16.1 / 16.10 requests 1-2). Fixes, BEFORE any result of this job:
 1) the event list: the 12 fixed regex events of job 220 / 221 without Nipah virus and IDFC First Bank fraud (10 events) + the next --n-extra (default 6) clusters of the job-219 list (e57_biglist.json: members >= 50, core >= .5, largest component >= .5 already; here: not recurring-type / product-type suspects (keyword share >= 0.3), not overlapping with an event already taken (>= 20 % of the members shared or >= 2 of the 3 top title tokens equal), in the order of members) -> about 16 events. The ethics filter is applied per target only (rules of job 222: b / c / d in any window; address, health, victim words, private-individual proxy of S in >= 20 % of the windows; family words reject only together with the private-individual proxy).
 2) the frozen bank of the LEAD variant (first sentences up to 350 tokens, c >= 3 independent articles): per target the question for S, the candidate strings and the S' for the cross null.
   - question (fixed frame per answer type, S inserted): 'Question: In the news about this event, which <name | number or amount | date> is mentioned together with <S>?\\nAnswer:'. The frame without S is checked for 5-gram overlap with the lead sentences of the event (target counted as overlapping if any; reported, the frame is NOT changed here).
   - candidates: strings of the same type with a lead exposure c' >= 1 in the same event (c' = independent articles whose lead contains the string), excluding the answer, names containing / contained in S or in the answer, and strings that appear together with S in a lead sentence of >= 2 independent articles (a competing true pair; with one article the candidate stays and carries cooc_S = 1). Fill order: strict (|c' - c| <= 1), then middle (c' >= 2 and |c' - c| <= 2), then relaxed (any), by |c' - c| then a seeded hash, at most 16 per target; each candidate carries its tier. G = number of candidates; main = G >= 8 (otherwise supplementary only). Counts of the full tiers are stored (n_strict / n_middle / n_relaxed) and whether each tier alone has >= 8.
   - cross null: up to 4 seeded random S' per target = subjects of OTHER targets of the same event, unrelated to S (no name containment) and never in a lead sentence together with the answer A.
 3) the strata table: targets per event and per c stratum 3-4 / 5-9 / 10-19 / 20+ (all targets, main targets with G >= 8, targets with 4 S', targets per answer type, tier sizes).
Outputs <out>/e57_lite/: lite_events.json, lite_bank.json, lite_manifest.json (sha256 of both + rules), e57_lite_report.md. The bank holds no article text (article ids, sentence indices, strings). The step-by-step article lead texts for the stream are not frozen here (section 16.10 request 3)."""
import argparse,json,sys,random,pickle,pathlib,time,collections,re,hashlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
import e57_events as EV
import e57_namedevents3 as N3
CAPN=(10,15,20);FREEZE_SEED=57231;G_MIN=8;NCAND=16;NSP=4;STRATA=(('3-4',3,4),('5-9',5,9),('10-19',10,19),('20+',20,10**9))
NOUN={'NAME':'name','NUM':'number or amount','DATE':'date'}
FRAME='Question: In the news about this event, which {noun} is mentioned together with {S}?\nAnswer:'
def hsh(*x):return hashlib.md5(('|'.join(map(str,(FREEZE_SEED,)+x))).encode()).hexdigest()
def stratum(c):
    for nm,lo,hi in STRATA:
        if lo<=c<=hi:return nm
def related(x,y):return x==y or x in y or y in x
def event_bank(ei,name,r,D,id0):
    """bank entries of one event (lead variant); r = filtered analysis result with detail"""
    V=N.variant(r,'lead');kept=V['kept'];arts_of=collections.defaultdict(set);surf={};P=collections.defaultdict(set);lead_sents=[]
    for ai in range(len(r['arts'])):
        for si in range(kept[ai]):
            fs=r['facts'][ai][si];lead_sents.append(r['sents'][ai][si])
            for ty,key,sf in fs:arts_of[(ty,key)].add(ai);surf.setdefault((ty,key),sf)
            for n_ in(f for f in fs if f[0]=='NAME'):
                for f in fs:
                    if (f[0],f[1])!=(n_[0],n_[1]):P[(n_[1],f[0],f[1])].add(ai)
    ng=set()
    for s_ in lead_sents:ng|=C.ngr(C.words(s_),5)
    tg=V['targets'];subj_surf={}
    for t in tg:
        for ai,sidx in t['occ'].items():
            for si in sidx:
                if si<kept[ai]:
                    for f in r['facts'][ai][si]:
                        if f[0]=='NAME' and f[1]==t['S']:subj_surf.setdefault(t['S'],f[2])
    subjects=sorted({t['S'] for t in tg});bank=[]
    for k,t in enumerate(tg):
        S=t['S'];ty=t['ty'];A=t['key'];c=t['c_v'];tid=id0+k
        Ss=subj_surf.get(S,S);q=FRAME.format(noun=NOUN.get(ty,'string'),S=Ss);fw=C.ngr(C.words(FRAME.format(noun=NOUN.get(ty,'string'),S='')),5)
        cands=[]
        for (ty2,key2),aset in arts_of.items():
            if ty2!=ty or key2==A or len(P.get((S,ty,key2),()))>=2:continue
            if ty=='NAME' and (related(key2,S) or related(key2,A)):continue
            cands.append(dict(key=key2,surface=surf[(ty2,key2)],c_prime=len(aset),cooc_S=len(P.get((S,ty,key2),()))))
        cands.sort(key=lambda x:(abs(x['c_prime']-c),hsh(tid,x['key'])))
        strict=[x for x in cands if abs(x['c_prime']-c)<=1];mid=[x for x in cands if x['c_prime']>=2 and abs(x['c_prime']-c)<=2]
        filled=[];seen=set()
        for tier,lst in(('strict',strict),('middle',mid),('relaxed',cands)):
            for x in lst:
                if x['key'] in seen or len(filled)>=NCAND:continue
                seen.add(x['key']);filled.append(dict(x,tier=tier))
        elig=[u for u in subjects if not related(u,S) and (u,ty,A) not in P]
        rng=random.Random(int(hsh(tid,'sp')[:12],16));sp=rng.sample(elig,min(NSP,len(elig)))
        occ=[dict(rid=r['rids'][ai],sent_idx=[si for si in sidx if si<kept[ai]]) for ai,sidx in sorted(t['occ'].items())]
        bank.append(dict(id=tid,event=ei,event_name=name,type=ty,subject_key=S,subject_surface=Ss,answer_key=A,answer_surface=t['surface'],c_lead=c,f_forms=t['f'],stratum=stratum(c),question=q,frame_overlap=bool(fw&ng),occurrences=occ,
            n_strict=len(strict),n_middle=len(mid),n_relaxed=len(cands),strict_ge8=len(strict)>=G_MIN,middle_ge8=len(mid)>=G_MIN,relaxed_ge8=len(cands)>=G_MIN,candidates=filled,G=len(filled),main=len(filled)>=G_MIN,
            cross_null=[dict(S=u,surface=subj_surf.get(u,u),question=FRAME.format(noun=NOUN.get(ty,'string'),S=subj_surf.get(u,u))) for u in sp]))
    return bank
def subset_view(r,keep):
    """analysis result restricted to the articles `keep` (old indices), re-indexed; targets keep the mined set, their occurrences are restricted"""
    new={o:n for n,o in enumerate(keep)};r2=dict(r)
    for k in('arts','sents','facts','rids'):r2[k]=[r[k][o] for o in keep]
    r2['targets']=[dict(t,occ={new[ai]:v for ai,v in t['occ'].items() if ai in new}) for t in r['targets']]
    return r2
def cap_select(r,pub_date,N_):
    """-> (main rule: articles with the most lead targets first, ties by publication date; date rule: the first N_ by date); old indices"""
    V=N.variant(r,'lead');score=collections.Counter()
    for t in V['targets']:
        for ai in t['occ_tok']:score[ai]+=1
    n=len(r['arts']);dt=lambda ai:(pub_date[r['rids'][ai]],r['rids'][ai])
    main=sorted(sorted(range(n),key=dt),key=lambda ai:-score[ai])[:N_];date=sorted(range(n),key=dt)[:N_]
    return sorted(main),sorted(date)
def cap_tables(names,res,D,pub_date):
    """for every N in CAPN and both rules: per event frozen selection and the bank statistics"""
    out={};sel={}
    for N_ in CAPN:
        for rule in('main','date'):
            per=[];allb=[];tok_tot=0;nid=0
            for ei,(nm,r) in enumerate(zip(names,res)):
                if not r['ok']:continue
                m,d=cap_select(r,pub_date,N_);keep=m if rule=='main' else d;r2=subset_view(r,keep);b=event_bank(ei,nm,r2,D,nid);nid+=len(b);allb+=b
                V=N.variant(r2,'lead');tok=sum(V['tok'].values());tok_tot+=tok
                per.append(dict(event=ei,name=nm,articles=len(keep),tokens=tok,targets=len(b),main=sum(x['main'] for x in b),sp4=sum(len(x['cross_null'])==NSP for x in b)))
                sel[f'{N_}|{rule}|{nm}']=[r['rids'][ai] for ai in keep]
            out[f'{N_}|{rule}']=dict(per=per,targets=len(allb),strata={s[0]:sum(x['stratum']==s[0] for x in allb) for s in STRATA},main=sum(x['main'] for x in allb),sp4=sum(len(x['cross_null'])==NSP for x in allb),tokens=tok_tot,
                tok_med=sorted(p['tokens'] for p in per)[len(per)//2] if per else 0,tok_max=max((p['tokens'] for p in per),default=0),over8k=sum(p['tokens']>8192 for p in per),events=len(per))
    return out,sel
def run(a):
    out=pathlib.Path(a.out);fd=out/'e57_lite';fd.mkdir(exist_ok=True);rep=fd/'e57_lite_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo):',rep);sys.exit(8)
    a.drop=N3.DROP;X=N3.analyze_all(a);D,names,res,meta,rows,pub_date=[X[k] for k in('D','names','res','meta','rows','pub_date')];t0=X['t0']
    bank=[];events=[];nid=0
    for ei,(nm,r,row) in enumerate(zip(names,res,rows)):
        b=event_bank(ei,nm,r,D,nid) if r['ok'] else [];nid+=len(b);bank+=b
        arts=[] if not r['ok'] else sorted(((pub_date[x],x) for x in r['rids']))
        events.append(dict(event=ei,name=nm,source=row['source'],matched_titles=row['titles'],independent_articles=row['n_dedup'] if r['ok'] else 0,ok=r['ok'],first=row['first'],last=row['last'],
            targets_before_ethics=row.get('targets_f2_before'),rejected_by_ethics=row.get('rejected_targets'),bank_targets=len(b),articles=[dict(rid=x,date=d) for d,x in arts]))
    caps,sel=cap_tables(names,res,D,pub_date);(fd/'lite_caps.json').write_text(json.dumps(dict(rule='main = articles with the most lead targets first, ties by publication date then id; date = first N by publication date',caps=caps,selection=sel),indent=1))
    ej=json.dumps(dict(seed=FREEZE_SEED,rules=dict(g_min=G_MIN,ncand=NCAND,nsp=NSP,frame=FRAME,dropped=list(N3.DROP),n_extra=a.n_extra),events=events),indent=1,default=str)
    bj=json.dumps(bank,indent=1,default=str);(fd/'lite_events.json').write_text(ej);(fd/'lite_bank.json').write_text(bj)
    man=dict(files={'lite_caps.json':hashlib.sha256((fd/'lite_caps.json').read_bytes()).hexdigest(),'lite_events.json':hashlib.sha256(ej.encode()).hexdigest(),'lite_bank.json':hashlib.sha256(bj.encode()).hexdigest()},n_events=len(events),n_events_with_bank=sum(bool(e['bank_targets']) for e in events),n_targets=len(bank),seed=FREEZE_SEED)
    (fd/'lite_manifest.json').write_text(json.dumps(man,indent=1));txt=report(events,bank,X,a,time.time()-t0,caps);rep.write_text(txt);print(txt[:9000])
def report(events,bank,X,a,sec,caps=None):
    f=EV.fmt;L=['# E57-lite freeze (CPU, regex proxy)','',f"{len(events)} events ({sum(e['source']=='regex' for e in events)} fixed regex events + {sum(e['source']!='regex' for e in events)} clusters of the job-219 list), {len(bank)} frozen targets (lead variant, c >= 3, after the target-level ethics filter). Candidates / questions / cross null as defined in the docstring of e57_lite.py; the pool uses LEAD exposures (job 220 / 221 used whole-article exposures for the pool tables, so the numbers differ). Run {sec:.0f} s.",'',
       '## events','','| # | event | source | matched titles | independent articles | first | last | targets f>=2 before ethics | rejected | frozen targets | main (G >= 8) |','|---|---|---|---|---|---|---|---|---|---|---|']
    for e in events:
        bb=[b for b in bank if b['event']==e['event']];L.append(f"| {e['event']} | {e['name']} | {e['source']} | {e['matched_titles']} | {e['independent_articles']}{'' if e['ok'] else ' (below the analysis minimum)'} | {e['first']} | {e['last']} | {e['targets_before_ethics']} | {e['rejected_by_ethics']} | {len(bb)} | {sum(b['main'] for b in bb)} |")
    st=[s[0] for s in STRATA]
    def tab(title,key,rows_):
        o=['',f'## {title}','','| '+key+' | '+' | '.join(st)+' | all |','|---|'+'---|'*(len(st)+1)]
        for nm,sel in rows_:
            cs=[sum(1 for b in sel if b['stratum']==s) for s in st];o.append(f"| {nm} | "+' | '.join(map(str,cs))+f' | {len(sel)} |')
        return o
    L+=tab('targets per c stratum (all frozen targets)','event',[(e['name'],[b for b in bank if b['event']==e['event']]) for e in events]+[('all',bank)])
    L+=tab('main targets (G >= 8 candidates after the fill order strict > middle > relaxed)','event',[(e['name'],[b for b in bank if b['event']==e['event'] and b['main']]) for e in events]+[('all',[b for b in bank if b['main']])])
    L+=tab('per answer type (all frozen targets)','type',[(ty,[b for b in bank if b['type']==ty]) for ty in sorted({b['type'] for b in bank})])
    L+=tab('per answer type (main targets)','type',[(ty,[b for b in bank if b['type']==ty and b['main']]) for ty in sorted({b['type'] for b in bank})])
    L+=['','## tier sizes (a tier alone has >= 8 candidates)','','| stratum | targets | strict >= 8 | middle >= 8 | relaxed >= 8 | median G | cross null with 4 S\' | frame 5-gram overlap |','|---|---|---|---|---|---|---|---|']
    import statistics
    for s in st+['all']:
        sel=[b for b in bank if s=='all' or b['stratum']==s]
        if sel:L.append(f"| {s} | {len(sel)} | {sum(b['strict_ge8'] for b in sel)} | {sum(b['middle_ge8'] for b in sel)} | {sum(b['relaxed_ge8'] for b in sel)} | {statistics.median(b['G'] for b in sel):.0f} | {sum(len(b['cross_null'])==NSP for b in sel)} | {sum(b['frame_overlap'] for b in sel)} |")
        else:L.append(f'| {s} | 0 | | | | | | |')
    cn=collections.Counter(len(b['cross_null']) for b in bank);L+=['',f"- cross-null S' available per target (count of S' -> targets): {dict(sorted(cn.items()))}",f"- ethics: {sum(e['rejected_by_ethics'] or 0 for e in events)} targets rejected of {sum(e['targets_before_ethics'] or 0 for e in events)}; reasons per event in e57_namedevents3-style counts are in the rejects json of this run (kept by job 222, not written here)."]
    if caps:
        L+=['',f"## article cap per event N = {', '.join(map(str,CAPN))} (selection frozen in lite_caps.json)",'','Rule main: articles ordered by the number of lead targets they contain (most first), ties by publication date; rule date: the first N articles by publication date. Targets are re-counted on the selected articles only (c >= 3 inside the selection; the mined target set is the one of all articles). Round tokens = lead tokens (+2 per article) of the selected articles of all events; per event tokens are compared with 8192.','',
           '| N | rule | events | targets c >= 3 | '+' | '.join('c '+x for x in st)+' | main (G >= 8) | with 4 S\' | round tokens (all events) | tokens per event median / max | events over 8192 |','|---|---|---|---|'+'---|'*len(st)+'---|---|---|---|---|']
        for N_ in CAPN:
            for rule in('main','date'):
                c=caps[f'{N_}|{rule}'];L.append(f"| {N_} | {rule} | {c['events']} | {c['targets']} | "+' | '.join(str(c['strata'][x]) for x in st)+f" | {c['main']} ({c['main']/max(1,c['targets']):.2f}) | {c['sp4']} ({c['sp4']/max(1,c['targets']):.2f}) | {c['tokens']} | {c['tok_med']} / {c['tok_max']} | {c['over8k']} |")
        for N_ in CAPN:
            L+=['',f'### N = {N_}, rule main: per event','','| event | articles | tokens | targets | main | with 4 S\' |','|---|---|---|---|---|---|']
            L+=[f"| {p['name']} | {p['articles']} | {p['tokens']} | {p['targets']} | {p['main']} | {p['sp4']} |" for p in caps[f'{N_}|main']['per']]
    L+=['','Titles of the added clusters are shown only to identify them; no article text is written.']
    return '\n'.join(L)+'\n'
def _selftest():
    def after(td):
        a=argparse.Namespace(out=str(td),dup_jac=0.5,min_articles=5,min_domains=5,min_c=3,workers=1,seed=5,rerun=True,n_extra=4,drop=())
        N3.DROP=();run(a);fd=td/'e57_lite';B=json.load(open(fd/'lite_bank.json'));E=json.load(open(fd/'lite_events.json'));M=json.load(open(fd/'lite_manifest.json'))
        assert B and M['n_targets']==len(B) and (fd/'e57_lite_report.md').exists() and (fd/'lite_caps.json').exists() and 'article cap per event' in (fd/'e57_lite_report.md').read_text() and 'lite_caps.json' in M['files']
        for b in B:
            assert b['question'].startswith('Question:') and b['subject_surface'] in b['question'] and all(c['key']!=b['answer_key'] for c in b['candidates']) and b['stratum'] in('3-4','5-9','10-19','20+')
            assert all(u['S']!=b['subject_key'] for u in b['cross_null']) and b['G']==len(b['candidates']) and b['main']==(b['G']>=G_MIN)
        assert hashlib.sha256((fd/'lite_bank.json').read_bytes()).hexdigest()==M['files']['lite_bank.json']
        print('e57_lite selftest ok:',len(E['events']),'events,',len(B),'targets, main',sum(b['main'] for b in B))
    N3._selftest(after)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw');ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-articles',type=int,default=5);ap.add_argument('--min-domains',type=int,default=5);ap.add_argument('--min-c',type=int,default=3)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--n-extra',type=int,default=6);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
