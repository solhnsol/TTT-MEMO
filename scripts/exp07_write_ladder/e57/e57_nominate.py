"""E57 distractor nominations check (CPU only, no model, no GPU, no download; private internal research). Input: the nominations of the news thread (nominations_v3.json: per question picks from the code's hard list + 'proposed extras' that are NOT in the list), distractors.json (job 248), validated.jsonl, the bundle store. For every extra the stream decides: it is ACCEPTED if it is not the gold answer or a variant / alias / container of it (related()), and it occurs in >= 1 lead sentence of the event (c_prime = articles, c_prime_win = articles with the string within +-64 tokens of an S core word); the supporting sentences (<= 3, packet_ref a##-s### when in the packet) are written next to it. c_prime < 3 is flagged weak (no exclusion). Accepted extras are appended to the hard list of their question (source 'nominated extra (news thread), verified in the stream') in distractors_ext.json; nominations_freeze.jsonl is the freeze input (nominated = list picks + accepted extras, <= 4). Nothing is dropped silently: extras_check.json lists every extra with its verdict and reason.
  python3 e57_nominate.py --out OUT --nominations nominations_v3.json     |   --selftest"""
import sys,json,re,argparse,pathlib,collections
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_bank as B
MAXN=4;WEAK=3
def extra_regex(q,s):
    s=s.strip()
    if q['type']=='NUMBER' and re.fullmatch(r'\d{1,4}',s):                                    # bare integer: same pattern as the answer matcher
        return re.compile(r'(?<![\w$.,])'+re.escape(s)+r'(?![\w%]|[.,]\d|\s?(?:million|billion|trillion|thousand|percent|per cent)\b)')
    return re.compile(r'(?<![\w])'+re.escape(s)+r'(?![\w])',re.I)
def check_extra(q,s,E,pk):
    vs=[q['canonical']]+list(q.get('variants',[]))
    if any(B.related(s,v) for v in vs):return dict(extra=s,accepted=False,reason='alias / container / variant of the gold answer',c_prime=None,c_prime_win=None,support=[])
    rx=extra_regex(q,s);cs=B.core(q['S']);arts=set();win=set();ex_w=[];ex_n=[]
    for ai,sents,cum in B.lead_stream(E):
        n=len(sents);ww=[B.words_of(t) for t in sents];sw={j for i in range(n) if cs&ww[i] for j in B.near_idx(cum,i,n)}
        for j,t in enumerate(sents):
            if rx.search(t):
                arts.add(ai);(ex_w if j in sw else ex_n).append((ai,j,t))
                if j in sw:win.add(ai)
    if not arts:return dict(extra=s,accepted=False,reason='does not occur in the lead sentences of the event',c_prime=0,c_prime_win=0,support=[])
    return dict(extra=s,accepted=True,reason='occurs in the stream'+(' (weak: c_prime < %d)'%WEAK if len(arts)<WEAK else ''),c_prime=len(arts),c_prime_win=len(win),support=B.support_for(E,ex_w+ex_n,pk))
def run(out,nomfile,log=print):
    out=pathlib.Path(out);fd=out/'e57_bank';st,idx=B.load(out);Q={json.loads(l)['id']:json.loads(l) for l in open(fd/'validated.jsonl') if l.strip()};D={d['id']:d for d in json.load(open(fd/'distractors.json'))}
    N=json.load(open(nomfile))['nominations'];rep=[];lines=[];prob=[]
    for n in N:
        qid=n['id']
        if qid not in Q:prob.append(f'{qid}: not a validated question');continue
        q=Q[qid];d=D[qid];E=st[q['event']];pk=B.packet_refs(out,q['event']);have={B.norm(h['surface']) for h in d['hard']}
        picks=[x['surface'] for x in n.get('nominated_from_list',[])];bad=[x for x in picks if B.norm(x) not in have]
        if bad:prob.append(f'{qid}: list picks not in the hard list {bad}');continue
        res=[check_extra(q,s,E,pk) for s in n.get('proposed_extra_needs_code_check',[])]
        acc=[r for r in res if r['accepted']]
        for r in acc:
            if B.norm(r['extra']) not in have:
                d['hard'].append(dict(surface=r['extra'],key=B.norm(r['extra']),c_prime=r['c_prime'],c_prime_win=r['c_prime_win'],source='nominated extra (news thread), verified in the stream',support=r['support']));have.add(B.norm(r['extra']))
        nominated=(picks+[r['extra'] for r in acc])
        if len(nominated)>MAXN:prob.append(f'{qid}: {len(nominated)} nominated (> {MAXN}): the news thread must reduce (freeze refuses more than {MAXN})')
        lines.append(dict(id=qid,nominated=nominated,evidence=[]))
        rep.append(dict(id=qid,event=q['event'],type=q['type'],list_picks=picks,extras=[{k:v for k,v in r.items() if k!='support'} for r in res],n_nominated=len(nominated)))
    (fd/'distractors_ext.json').write_text(json.dumps([D[i] for i in D],indent=1,ensure_ascii=False))
    (fd/'nominations_freeze.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in lines)+'\n');(fd/'extras_check.json').write_text(json.dumps(dict(problems=prob,questions=rep),indent=1,ensure_ascii=False))
    ne=sum(len(r['extras']) for r in rep);na=sum(1 for r in rep for e in r['extras'] if e['accepted']);log(f'extras: {ne} checked, {na} accepted, {ne-na} rejected; problems: {prob or "none"}')
    L=['# E57 distractor nominations: extras check (counts only)','',f'{ne} extras checked, {na} accepted. Rule: not an alias / container of the gold answer and occurring in >= 1 lead sentence of the event (weak if c_prime < {WEAK}).','','| question | type | extra | accepted | c_prime | c_prime_win | reason |','|---|---|---|---|---|---|---|']
    for r in rep:
        for e in r['extras']:L.append(f"| {r['id']} | {r['type']} | {e['extra']} | {'yes' if e['accepted'] else 'NO'} | {e['c_prime']} | {e['c_prime_win']} | {e['reason']} |")
    L+=['','Questions with fewer than 2 nominated distractors after the check: '+(', '.join(f"{r['id']} ({r['n_nominated']})" for r in rep if r['n_nominated']<2) or 'none')]
    (fd/'extras_check.md').write_text('\n'.join(L)+'\n');return rep,prob
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--nominations');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        q=dict(type='NUMBER',canonical='34',variants=['34'],S='x');rx=extra_regex(q,'28');assert rx.search('aged 28 years') and not rx.search('$28 million') and not rx.search('4.28');rx=extra_regex(dict(type='PERSON'),'Claudia Sheinbaum');assert rx.search('said claudia sheinbaum today') and not rx.search('Claudiaa Sheinbaum')
        print('e57_nominate selftest ok');sys.exit(0)
    run(a.out,a.nominations)
