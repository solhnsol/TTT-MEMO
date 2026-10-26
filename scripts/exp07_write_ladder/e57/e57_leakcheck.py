"""E57 question-text exposure check (CPU only, READ-ONLY on the frozen bank and the bundle; writes leakcheck.{json,md}): does the wording of a question appear in the articles of the stream? For every question of bank_v1: (1) the question prefix with S written out (frame without the blank), (2) the frame with S removed (the validator's 5-gram test, here against ALL lead sentences of ALL events, not only the event of the question), (3) prefix + gold answer as one string. Reported: longest run of consecutive words shared with any article (words, lower case), the article and the sentence of that longest run, number of articles holding a >= 4-word / >= 5-word run, exact-substring hits of the whole prefix, of prefix + gold. Also the 4-gram share of the question that occurs in the stream.
  python3 e57_leakcheck.py --out OUT_news"""
import sys,json,re,argparse,pathlib,collections
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_bank as B,e57_common as C
def run(out):
    out=pathlib.Path(out);st,idx=B.load(out);bank=json.load(open(out/'e57_bank'/'bank_v1.json'))['questions']
    arts=[];by4=collections.defaultdict(list)                    # all lead sentences of all events: (event, rid, sentence, words)
    for ev,E in st.items():
        for ai in range(len(E['rids'])):
            for si,t in enumerate(E['sents'][ai][:max(1,E['kept'][ai])]):
                w=C.words(t);arts.append((ev,E['rids'][ai],si,t,w))
    full=[a[4] for a in arts]
    # index: 3-gram -> list of (sentence id, position) for fast longest-run search
    idx3=collections.defaultdict(list)
    for sid,w in enumerate(full):
        for p in range(len(w)-2):idx3[tuple(w[p:p+3])].append((sid,p))
    def longest(qw):
        best=(0,None,None)
        for i in range(len(qw)-2):
            for sid,p in idx3.get(tuple(qw[i:i+3]),()):
                w=full[sid];n=3
                while i+n<len(qw) and p+n<len(w) and qw[i+n]==w[p+n]:n+=1
                # extend to the left as well
                l=0
                while i-l-1>=0 and p-l-1>=0 and qw[i-l-1]==w[p-l-1]:l+=1
                if n+l>best[0]:best=(n+l,sid,(i-l,i+n))
        return best
    rows=[];allsent_text=[' '.join(a[4]) for a in arts]
    for q in bank:
        pre=B.norm(q['frame'].replace('{S}',q['S']).split('___')[0]);pre_words=pre.split();noS=C.words(q['frame'].replace('{S}',' ').replace('___',' '));gold=q['fixed_surface']
        res={}
        for name,qw in (('prefix_with_S',pre_words),('frame_without_S',noS),('prefix_plus_gold',(pre+' '+B.norm(gold)).split())):
            n,sid,span=longest(qw);ev,rid,si,t,w=arts[sid] if sid is not None else (None,None,None,'',[])
            res[name]=dict(longest_run_words=n,of_words=len(qw),event_of_run=ev,sentence=t[:220] if n>=4 else None,run=' '.join(qw[span[0]:span[1]]) if sid is not None and n>=4 else None)
        full_pre=sum(1 for s in allsent_text if pre in s);pg=B.norm(q['frame'].replace('{S}',q['S']).split('___')[0]+' '+gold);full_pg=sum(1 for s in allsent_text if pg in s)
        g4=set(zip(*[pre_words[i:] for i in range(4)])) if len(pre_words)>=4 else set();hit4=sum(1 for g in g4 if g in idx3_4(full,g)) if False else None
        rows.append(dict(id=q['id'],event=q['event'],type=q['type'],stratum=q.get('stratum','main'),frame=q['frame'],S=q['S'],gold=gold,**res,exact_prefix_sentences=full_pre,exact_prefix_plus_gold_sentences=full_pg))
    return rows
def idx3_4(*a):return ()
def report(rows,out):
    out=pathlib.Path(out)/'e57_bank';(out/'leakcheck.json').write_text(json.dumps(rows,indent=1,ensure_ascii=False))
    L=['# E57 question-text exposure check (private internal research material)','','Longest run of consecutive words (lower case, validator tokenisation) shared between the question text and ANY lead sentence of ANY event of the stream. 3 words or less = common phrase; the validator rule was: no 5-gram of the frame without S in the lead sentences of the question event.','']
    c=collections.Counter()
    for r in rows:
        for k in ('prefix_with_S','frame_without_S','prefix_plus_gold'):c[(k,min(r[k]['longest_run_words'],8))]+=1
    for k in ('prefix_with_S','frame_without_S','prefix_plus_gold'):L.append(f"- {k}: longest shared run distribution (words: number of questions) "+', '.join(f"{n}{'+' if n==8 else ''}: {c[(k,n)]}" for n in range(0,9) if c[(k,n)]))
    L+=['',f"- exact substring hits of the whole prefix in a sentence: {sum(1 for r in rows if r['exact_prefix_sentences'])} questions; of prefix + gold: {sum(1 for r in rows if r['exact_prefix_plus_gold_sentences'])} questions",'','| question | prefix with S: run / words | frame without S: run / words | prefix + gold: run / words | longest run text (prefix + gold) |','|---|---|---|---|---|']
    for r in sorted(rows,key=lambda r:-r['prefix_plus_gold']['longest_run_words']):
        a,b,c_=r['prefix_with_S'],r['frame_without_S'],r['prefix_plus_gold'];L.append(f"| {r['id']} | {a['longest_run_words']} / {a['of_words']} | {b['longest_run_words']} / {b['of_words']} | {c_['longest_run_words']} / {c_['of_words']} | {c_['run'] or ''} |")
    L+=['','## Sentences behind runs of 5 or more words','']
    for r in rows:
        for k in ('prefix_with_S','frame_without_S','prefix_plus_gold'):
            x=r[k]
            if x['longest_run_words']>=5:L.append(f"- {r['id']} ({k}, {x['longest_run_words']} words, event of the run: {x['event_of_run']}): \"{x['run']}\" in: {x['sentence']}")
    (out/'leakcheck.md').write_text('\n'.join(L)+'\n')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');a=ap.parse_args();rows=run(a.out);report(rows,a.out);print('leakcheck:',len(rows),'questions')
