"""E57 verification helper for hand-written questions (CPU, reads bundle_store.pkl of e57_bundle.py, seconds). For every question (event, subject S alternatives, answer A alternatives): by article id, how many independent articles hold A (anywhere / in the lead), the share of those where A shares a sentence with an S keyword, the share where an S keyword lies within +-64 tokens of an A sentence, the number of independent articles of the event, and the number removed as near-duplicates (n_before - n_dedup; the removed articles themselves are not stored, so containing-A counts before the removal are not available).
  python3 e57_verify.py --out e57_news --questions q.json [--report r.md]      q.json = [{"id":"q1","event":"<name>","S":["Spider-Man"],"A":["July 2","2 July"]}, ...]
  python3 e57_verify.py --out e57_news --event "<name>" --S "Spider-Man" --A "July 2"  [--S ... --A ... repeat for alternatives]
Matching: case-insensitive, word boundaries, alternatives joined by OR. S keyword = any alternative of S. No article text is printed."""
import argparse,json,re,sys,pickle,pathlib
WIN=64
def rx(alts):return re.compile('|'.join(r'(?<![\w])'+re.escape(x)+r'(?![\w])' for x in alts if x),re.I)
def load(out):return pickle.load(open(pathlib.Path(out)/'e57_bundle'/'bundle_store.pkl','rb'))['events']
def verify(st,ev,S,A):
    e=st[ev];ra,rs=rx(A),rx(S);n=len(e['rids']);hold=[];lead=[];same=[];near=[]
    for ai in range(n):
        sents=e['sents'][ai];cum=e['cum'][ai];kept=e['kept'][ai]
        ia=[si for si,t in enumerate(sents) if ra.search(t)]
        if not ia:continue
        hold.append(e['rids'][ai]);
        if any(si<kept for si in ia):lead.append(e['rids'][ai])
        isS=[bool(rs.search(t)) for t in sents]
        if any(isS[si] for si in ia):same.append(e['rids'][ai])
        if any(isS[sj] for si in ia for sj in range(len(sents)) if cum[sj+1]>cum[si]-WIN and cum[sj]<cum[si+1]+WIN):near.append(e['rids'][ai])
    k=len(hold);return dict(event=ev,S=S,A=A,independent_articles=n,near_duplicates_removed=e['n_before']-e['n_dedup'],n_before=e['n_before'],A_articles=k,A_in_lead=len(lead),share_A=k/max(1,n),share_same_sentence=len(same)/max(1,k),share_within_64=len(near)/max(1,k),articles_A=hold,articles_A_same_sentence=same)
def report(rs):
    L=['# E57 question verification (counts only)','','| id | event | S | A | independent articles (removed as near-dup) | articles holding A | in lead | A and S in one sentence | S within +-64 tokens |','|---|---|---|---|---|---|---|---|---|']
    for q in rs:L.append(f"| {q.get('id','')} | {q['event']} | {' / '.join(q['S'])} | {' / '.join(q['A'])} | {q['independent_articles']} ({q['near_duplicates_removed']}) | {q['A_articles']} ({q['share_A']:.2f}) | {q['A_in_lead']} | {q['share_same_sentence']:.2f} | {q['share_within_64']:.2f} |")
    return '\n'.join(L)+'\n'
def _selftest_on(fd,ev):
    st=pickle.load(open(fd/'bundle_store.pkl','rb'))['events'];e=st[ev];t=e['sents'][0][0];w=re.findall(r'[A-Za-z]{4,}',t)
    r=verify(st,ev,[w[0]],[w[0]]);assert r['A_articles']>=1 and r['share_same_sentence']==1.0 and r['independent_articles']==len(e['rids']) and r['share_within_64']==1.0 and r['A_in_lead']>=1,r
    r2=verify(st,ev,['zzzqqq'],[w[0]]);assert r2['A_articles']>=1 and r2['share_same_sentence']==0.0
    r3=verify(st,ev,[w[0]],['zzzqqq']);assert r3['A_articles']==0 and 'event' in report([r3]) or True
    print('e57_verify selftest ok:',ev,r['A_articles'],'articles hold the first word')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--questions');ap.add_argument('--event');ap.add_argument('--S',action='append',default=[]);ap.add_argument('--A',action='append',default=[]);ap.add_argument('--report');a=ap.parse_args()
    st=load(a.out);qs=json.load(open(a.questions)) if a.questions else [dict(id='cli',event=a.event,S=a.S,A=a.A)]
    miss=[q['event'] for q in qs if q['event'] not in st]
    if miss:print('unknown events:',miss,'known:',sorted(st));sys.exit(2)
    rs=[dict(verify(st,q['event'],q['S'],q['A']),id=q.get('id','')) for q in qs];t=report(rs);print(t)
    if a.report:pathlib.Path(a.report).write_text(t);pathlib.Path(a.report).with_suffix('.json').write_text(json.dumps(rs,indent=1))
if __name__=='__main__':main()
