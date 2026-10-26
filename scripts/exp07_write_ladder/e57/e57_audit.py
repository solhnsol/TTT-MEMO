"""E57 pre-freeze audit (CPU only, READ-ONLY: writes only audit_pre_freeze.{json,md}; no model, no GPU; private internal research material). Checks, for the questions that would be frozen: (1) number of questions (by event / type / stratum, unique ids, ids of validation = distractors = nominations); (2) candidates per question (easy count, hard-list count, nominated count 2-4 with the accepted extras, gold / alias / duplicate / unparsable checks); (3) every question re-checked from scratch with the validator's own check_question (frame {S}, evidence ids and evidence sentences holding the answer, c / c_SA / competitor), plus a proof-reading table (frame, S, gold, variants, evidence sentences); (4) article exposure per question: c = lead articles holding the answer, c_window = articles holding it within +-64 tokens of an S word, and the number of reading rounds R needed for 128 exposures (ceil(128 / c)); exposures at R = 8, 16, 22, 32.
  python3 e57_audit.py --out OUT --nominations nominations_v4.json"""
import sys,json,math,argparse,pathlib,collections,statistics
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_bank as B
TARGET=128;RS=(8,16,22,32)
def run(out,nomfile):
    out=pathlib.Path(out);fd=out/'e57_bank';st,idx=B.load(out);V=[json.loads(l) for l in open(fd/'validated.jsonl') if l.strip()];D={d['id']:d for d in json.load(open(fd/'distractors.json'))}
    N={n['id']:n for n in json.load(open(nomfile))['nominations']};XC={}
    if (fd/'extras_check.json').exists():
        for r in json.load(open(fd/'extras_check.json'))['questions']:
            for e in r['extras']:XC[(r['id'],e['extra'])]=e
    P={};issues=[];rows=[]
    ids=[q['id'] for q in V];c1={'questions':len(V),'unique_ids':len(set(ids))==len(ids),'ids_equal_distractors':set(ids)==set(D),'ids_equal_nominations':set(ids)==set(N),
        'by_event':dict(collections.Counter(q['event'] for q in V)),'by_type':dict(collections.Counter(q['type'] for q in V)),'by_stratum':dict(collections.Counter(q.get('stratum','main') for q in V))}
    if not (c1['unique_ids'] and c1['ids_equal_distractors'] and c1['ids_equal_nominations']):issues.append('ids differ between validated / distractors / nominations: '+str((set(ids)^set(D))|(set(ids)^set(N))))
    golds=collections.defaultdict(list)
    for q in V:golds[(q['event'],B.norm(q['canonical']))].append(q['id'])
    for k,v in golds.items():
        if len(v)>1:issues.append(f'same gold answer {k[1]!r} in the same event for {v}')
    for q in V:
        qid=q['id'];E=st[q['event']];pf=out/'e57_bank'/'packets'/(B.slug(q['event'])+'.packet.json');Pk=P.setdefault(q['event'],json.load(open(pf)));d=D[qid];n=N.get(qid,{})
        why,stt=B.check_question(q,E,Pk);hard={B.norm(h['surface']) for h in d['hard']}
        picks=[x['surface'] for x in n.get('nominated_from_list',[])];extras=n.get('proposed_extra_needs_code_check',[]);acc=[x for x in extras if XC.get((qid,x),{}).get('accepted')];rej=[x for x in extras if not XC.get((qid,x),{}).get('accepted')]
        nom=picks+acc;vs=[q['canonical']]+list(q['variants']);prob=[]
        if why:prob.append('re-check fails: '+'; '.join(why))
        if not 2<=len(nom)<=4:prob.append(f'{len(nom)} nominated distractors (need 2-4)')
        if d['n_easy']<B.EASY_MIN:prob.append(f"{d['n_easy']} easy candidates (< {B.EASY_MIN})")
        if len({B.norm(x) for x in nom})<len(nom):prob.append('duplicate nominated distractors')
        for x in nom:
            if any(B.related(x,v) for v in vs):prob.append(f'nominated {x!r} is related to the gold')
        for x in picks:
            if B.norm(x) not in hard:prob.append(f'list pick {x!r} not in the hard list')
        for x in extras:
            if x not in acc and x not in rej:prob.append(f'extra {x!r} unchecked')
        sm={s['s']:(a['a'],s['text']) for a in Pk['articles'] for s in a['sentences']};holds,keys=B.ans_matcher(q);ev=[]
        for e in q['evidence']:
            aid,sid=e.split('-')[0],e.split('-')[-1];t=sm.get(sid,('',''))[1];ev.append(dict(id=e,holds=bool(holds(t)),text=t[:200]))
        if not all(x['holds'] for x in ev):prob.append('an evidence sentence does not hold the answer')
        if q['frame'].count('{S}')!=1:prob.append('frame has not exactly one {S}')
        c=stt['c'];cw=stt['c_window'];cs=stt['c_SA']
        rows.append(dict(id=qid,event=q['event'],type=q['type'],stratum=q.get('stratum','main'),frame=q['frame'],S=q['S'],gold=q['canonical'],variants=q['variants'],evidence=ev,n_evidence_articles=len({e.split('-')[0] for e in q['evidence']}),
            c=c,c_SA=round(cs,3),c_window=cw,comp_top=stt['comp_top'][:1],n_easy=d['n_easy'],n_hard_list=len(d['hard']),n_nominated=len(nom),picks=picks,extras_accepted=acc,extras_rejected=rej,
            R_for_128=math.ceil(TARGET/max(c,1)),R_for_128_window=math.ceil(TARGET/max(cw,1)),**{f'exp_R{r}':c*r for r in RS},problems=prob))
        for p in prob:issues.append(f'{qid}: {p}')
    cs_=[r['c'] for r in rows];cw_=[r['c_window'] for r in rows]
    c4={'c':dict(min=min(cs_),median=statistics.median(cs_),max=max(cs_)),'c_window':dict(min=min(cw_),median=statistics.median(cw_),max=max(cw_)),'R_for_128_each':dict(min=min(r['R_for_128'] for r in rows),median=statistics.median(r['R_for_128'] for r in rows),max=max(r['R_for_128'] for r in rows)),
        'R_for_128_all_questions':max(r['R_for_128'] for r in rows),'R_for_128_all_questions_window':max(r['R_for_128_window'] for r in rows),'questions_below_128_at':{f'R{r}':sum(1 for x in rows if x['c']*r<TARGET) for r in RS},'exposure_spread_at_R22':dict(min=min(x['c'] for x in rows)*22,median=statistics.median(cs_)*22,max=max(cs_)*22)}
    byev=collections.defaultdict(list)
    for r in rows:byev[r['event']].append(r['c'])
    c4['by_event_c_median']={k:statistics.median(v) for k,v in byev.items()}
    rep=dict(counts=c1,exposure=c4,issues=issues,questions=rows);(fd/'audit_pre_freeze.json').write_text(json.dumps(rep,indent=1,ensure_ascii=False,default=str))
    L=['# E57 pre-freeze audit (counts and question text; private internal research material)','',f"## 1. Questions: {c1['questions']}",'',f"unique ids {c1['unique_ids']}; ids equal in validated / distractors / nominations: {c1['ids_equal_distractors']} / {c1['ids_equal_nominations']}",'',f"by event: {c1['by_event']}",'',f"by type: {c1['by_type']}",'',f"by stratum: {c1['by_stratum']}",'','## 2. Candidates per question','','| id | easy | hard list | nominated (list + accepted extras) | rejected extras | problems |','|---|---|---|---|---|---|']
    for r in rows:L.append(f"| {r['id']} | {r['n_easy']} | {r['n_hard_list']} | {r['n_nominated']}: {', '.join(r['picks']+r['extras_accepted'])} | {', '.join(r['extras_rejected']) or '-'} | {'; '.join(r['problems']) or '-'} |")
    L+=['','## 3. Question and answer proof-reading table (re-checked with the validator code)','','| id | stratum | frame | S | gold (variants) | evidence ids / holds | evidence article count |','|---|---|---|---|---|---|---|']
    for r in rows:L.append(f"| {r['id']} | {r['stratum']} | {r['frame']} | {r['S']} | {r['gold']} ({', '.join(r['variants'])}) | {', '.join(e['id']+(' ok' if e['holds'] else ' NO') for e in r['evidence'])} | {r['n_evidence_articles']} |")
    L+=['','Evidence sentences (first 200 characters):','']
    for r in rows:L+=[f"- **{r['id']}** gold {r['gold']}: "+' || '.join(f"{e['id']}: {e['text']}" for e in r['evidence'])]
    L+=['',f'## 4. Article exposure (target {TARGET} exposures per fact; exposure = c x R, c = lead articles holding the answer, c_window = within +-64 tokens of an S word)','',f"c: {c4['c']}; c_window: {c4['c_window']}; R needed for 128 per question (c): {c4['R_for_128_each']}; R needed so that EVERY question reaches 128: {c4['R_for_128_all_questions']} (c) / {c4['R_for_128_all_questions_window']} (c_window); questions below 128 exposures: {c4['questions_below_128_at']}; exposure spread at R=22: {c4['exposure_spread_at_R22']}",'',f"median c per event: {c4['by_event_c_median']}",'','| id | event | c | c_SA | c_window | R for 128 (c) | R for 128 (c_window) | exp R=8 | R=16 | R=22 | R=32 |','|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows,key=lambda x:x['c']):L.append(f"| {r['id']} | {r['event']} | {r['c']} | {r['c_SA']} | {r['c_window']} | {r['R_for_128']} | {r['R_for_128_window']} | {r['exp_R8']} | {r['exp_R16']} | {r['exp_R22']} | {r['exp_R32']} |")
    L+=['','## Issues found','']+([f'- {i}' for i in issues] or ['none'])
    (fd/'audit_pre_freeze.md').write_text('\n'.join(L)+'\n');print(f"audit: {len(rows)} questions, {len(issues)} issues");return rep
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--nominations');a=ap.parse_args();run(a.out,a.nominations)
