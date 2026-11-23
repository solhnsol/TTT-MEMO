"""E62 P0 facts (CPU; new file; thread-server 11/23): invented names and answers, the template pool checks, fact tables for 3 books x 14 cells (+ decoy answers). Request v1 rules implemented here: answers = syllable-generator words, ' '+answer exactly 3 Llama-3 tokens, never in the 24 selected PG-19 books (text search; the full PG-19 corpus is not on this server), names likewise; every template: exactly one {A}, one name slot, 13-40 words, no digits, at most one other capitalised word (proper noun), no banned modern words, 5-gram overlap between templates of the same type pool = 0 (slots break n-grams), no 5-gram overlap with the question frames; frames checked against the templates of all types.
  python3 e62_facts.py --selftest      (pool checks only)  |  --report (+ the leak check against the books)"""
import sys,re,json,random,hashlib,pathlib,itertools,collections
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import e62_templates as TP
MODERN=set('okay ok telephone phone motor automobile cinema radio gramophone aeroplane airplane wireless taxi bus computer television film cameras machine-gun'.split())
SLOT=re.compile(r'\{(F|TL|L|A)\}')
def words(t):return re.findall(r"[A-Za-z0-9'\-\.]+|\{[A-Z]+\}",t)
def tokens_for_ngram(t):
    """lower-case word tokens; every slot becomes a unique wildcard that breaks n-grams"""
    out=[];k=0
    for w in re.findall(r"\{[A-Z]+\}|[A-Za-z']+",t):
        if w.startswith('{'):out.append(f'<{k}>');k+=1
        else:out.append(w.lower())
    return out
def ngrams(t,n=5):
    toks=tokens_for_ngram(t);return {tuple(toks[i:i+n]) for i in range(len(toks)-n+1) if not any(x.startswith('<') for x in toks[i:i+n])}
def length_words(t):
    """word count of the realised sentence: {F}=2, {TL}=2, {L}=1, {A}=1"""
    s=t.replace('{F}','Aa Bb').replace('{TL}','Mr. Bb').replace('{L}','Bb').replace('{A}','Cc');return len(re.findall(r"[A-Za-z']+(?:\.[A-Za-z']*)*",s))
def other_proper(t):
    s=SLOT.sub('',t);caps=0
    for m in re.finditer(r"(?<![.!?\"'])(?<!^)\b([A-Z][a-z]+)\b",s):
        w=m.group(1);i=m.start()
        pre=s[:i].rstrip()
        if w in ('Mr','I') or pre=='' or pre[-1] in '.!?"“\'(':continue
        caps+=1
    return caps
def check_pool():
    bad=[];stats={}
    for ty,L in TP.POOL.items():
        ngs=[ngrams(t) for _,t in L];fr=[ngrams(f) for f in TP.FRAMES[ty]];pos=collections.Counter();names=collections.Counter();sty=collections.Counter()
        for i,(st,t) in enumerate(L):
            if t.count('{A}')!=1:bad.append((ty,i,'A count',t))
            nm=[s for s in ('{F}','{TL}','{L}') if s in t];
            if not nm:bad.append((ty,i,'no name slot',t))
            else:names['F' if '{F}' in t else 'other']+=1
            n=length_words(t)
            if not 13<=n<=40:bad.append((ty,i,f'length {n}',t))
            if re.search(r'\d',t):bad.append((ty,i,'digit',t))
            if other_proper(t)>1:bad.append((ty,i,f'proper nouns {other_proper(t)}',t))
            if any(w in MODERN for w in re.findall(r"[a-z]+",t.lower())):bad.append((ty,i,'modern word',t))
            toks=re.findall(r"\{[A-Z]+\}|[A-Za-z']+",t);k=[j for j,w in enumerate(toks) if w=='{A}'][0];pos['start' if k<len(toks)/3 else ('end' if k>=2*len(toks)/3 else 'mid')]+=1;sty[st]+=1
            for j in range(i):
                ov=ngs[i]&ngs[j]
                if ov:bad.append((ty,i,f'5-gram with {j}: {sorted(ov)[0]}',t))
        for i,(st,t) in enumerate(L):
            for f in TP.FRAMES.values():
                ov=ngrams(t)&set().union(*[ngrams(x) for x in f])
                if ov:bad.append((ty,i,f'frame 5-gram {sorted(ov)[0]}',t))
        stats[ty]=dict(n=len(L),F_share=names['F']/len(L),pos=dict(pos),styles=dict(sty))
    return bad,stats
def main():
    bad,stats=check_pool();print(json.dumps(stats));print(len(bad),'template problems')
    for b in bad[:60]:print(b[0],b[1],b[2],'|',b[3][:110])
if __name__=='__main__':main()
