"""Pass 1 over the Wikipedia dump: target articles, place-article leads, backlink paragraphs, person-name statistics."""
import pyarrow.parquet as pq, json, re, sys, pickle, os
from collections import Counter, defaultdict
from multiprocessing import Pool
import ahocorasick
NT=os.environ.get('NT_DIR','/tmp/claude-0/nt');DL=NT+'/dl'; OUT=NT+'/work'
def base(t): return re.sub(r'\s*\(.*\)$','',t).strip()
def load_cands():
    pop=pq.read_table(f'{DL}/popqa.parquet').to_pylist()
    C={}  # title -> dict
    for r in pop:
        if r['prop']=='place of birth' and r['s_pop']<=2000:
            c=C.setdefault(r['s_wiki_title'],{'title':r['s_wiki_title'],'src':set(),'facts':{}})
            c['src'].add('popqa');c['facts'].setdefault('birth',[]).append({'obj':r['obj'],'aliases':json.loads(r['o_aliases']),'o_title':r['o_wiki_title'],'o_pop':r['o_pop'],'s_pop':r['s_pop'],'s_aliases':json.loads(r['s_aliases'])})
    tr=[]
    for i in range(3): tr+=pq.read_table(f'{DL}/trex_tr{i}.parquet').to_pylist()
    for r in tr:
        rel={'[Person] is born in [Location]':'birth','[Person] studies at [School]':'school'}.get(r['relation'])
        if rel and base(r['title'])==r['head']:
            c=C.setdefault(r['title'],{'title':r['title'],'src':set(),'facts':{}})
            c['src'].add('trex');c['facts'].setdefault(rel,[]).append({'obj':r['tail'],'aliases':[],'o_title':r['tail'],'trex_text':r['text']})
    return C
NAT=re.compile(r"\)\s*(?:is|was) (?:an?|the) ((?:[A-Z][a-z]+)(?:-[A-Z][a-z]+)?)\b")
HE=re.compile(r'\b(?:he|him|his)\b',re.I); SHE=re.compile(r'\b(?:she|her|hers)\b',re.I)
def work(i):
    C=pickle.load(open(f'{OUT}/cands.pkl','rb'))
    names={}; 
    for t in C: names.setdefault(base(t),[]).append(t)
    places=set()
    for c in C.values():
        for fs in c['facts'].values():
            for f in fs: places.add(f['o_title'])
    A=ahocorasick.Automaton()
    for n in names: A.add_word(n,n)
    A.make_automaton()
    w=pq.read_table(f'{DL}/w{i}.parquet',columns=['title','text'])
    titles=w.column('title').to_pylist(); texts=w.column('text').to_pylist(); del w
    targets={}; pleads={}; back=defaultdict(list); bcount=Counter()
    given=Counter(); surn=Counter(); 
    for t,tx in zip(titles,texts):
        if t in C: targets[t]=tx
        if t in places: pleads[t]=tx[:600]
        bt=base(t)
        hit=set()
        for end,n in A.iter(tx):
            s=end-len(n)+1
            if (s>0 and tx[s-1].isalnum()) or (end+1<len(tx) and tx[end+1].isalnum()): continue
            if n==bt: continue
            hit.add(n)
        if hit:
            paras=tx.split('\n\n')
            for n in hit:
                bcount[n]+=1
                if len(back[n])<60:
                    for p in paras:
                        if n in p and 150<len(p)<2500: back[n].append((t,p));break
        lead=tx[:400]; m=NAT.search(lead)
        if m and re.search(r'\d{4}',lead[:m.start()+1]):
            toks=bt.split()
            if 2<=len(toks)<=3 and all(re.fullmatch(r"[A-Z][a-z]+",x) for x in toks):
                h=len(HE.findall(tx[:5000])); s_=len(SHE.findall(tx[:5000]))
                g='m' if h>=4*max(s_,1) else ('f' if s_>=4*max(h,1) else None)
                if g: given[(m.group(1),g,toks[0])]+=1
                surn[(m.group(1),toks[-1])]+=1
    pickle.dump({'targets':targets,'pleads':pleads,'back':dict(back),'bcount':bcount,'given':given,'surn':surn},open(f'{OUT}/scan1_{i}.pkl','wb'))
    return i,len(targets),len(back)
if __name__=='__main__':
    if not os.path.exists(f'{OUT}/cands.pkl'):
        C=load_cands();pickle.dump(C,open(f'{OUT}/cands.pkl','wb'))
        print('cands',len(C),Counter(tuple(sorted(c['src'])) for c in C.values()),flush=True)
    todo=[i for i in range(41) if not os.path.exists(f'{OUT}/scan1_{i}.pkl')]
    with Pool(3) as p:
        for r in p.imap_unordered(work,todo): print(r,flush=True)
