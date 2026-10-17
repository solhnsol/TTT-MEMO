import pyarrow.parquet as pq, json, re, sys
from collections import Counter, defaultdict
from tokenizers import Tokenizer
D=sys.argv[1]
pop=pq.read_table(f'{D}/popqa.parquet').to_pylist()
REL={'place of birth','occupation','sport','father','mother','religion'}
by=defaultdict(list)
for r in pop:
    if r['prop'] in REL: by[r['s_wiki_title']].append(r)
w=pq.read_table(f'{D}/w0.parquet',columns=['title','text'])
titles=w.column('title').to_pylist(); texts=w.column('text').to_pylist()
print('articles in shard',len(titles),'popqa person subjects',len(by))
tok=Tokenizer.from_file(f'{D}/llama3_tok.json')
hit=[(i,t) for i,t in enumerate(titles) if t in by]
print('popqa subjects in shard0',len(hit))
rows=[]
for i,t in hit:
    tx=texts[i]; rs=by[t]; ntok=len(tok.encode(tx).ids)
    he=len(re.findall(r'\b(he|him|his)\b',tx,re.I)); she=len(re.findall(r'\b(she|her|hers)\b',tx,re.I))
    sur=t.split(' (')[0].split()[-1]
    for r in rs:
        alts=[r['obj']]+json.loads(r['o_aliases'])
        m=sum(len(re.findall(r'\b'+re.escape(a)+r'\b',tx)) for a in alts if len(a)>2)
        rows.append(dict(t=t,rel=r['prop'],obj=r['obj'],pop=r['s_pop'],ntok=ntok,gold_mentions=m,he=he,she=she,name_mentions=len(re.findall(r'\b'+re.escape(sur)+r'\b',tx))))
import statistics as st
for rel in REL:
    R=[x for x in rows if x['rel']==rel]
    if not R: continue
    g=[x['gold_mentions'] for x in R]
    print(rel,len(R),'pop med',st.median(x['pop'] for x in R),'ntok med',st.median(x['ntok'] for x in R),'gold>=1',sum(v>=1 for v in g),'gold>=2',sum(v>=2 for v in g),'gold>=3',sum(v>=3 for v in g))
json.dump(rows,open(f'{D}/pilot_rows.json','w'))
# backlink estimate for 40 low-pop subjects: count other articles containing full title name
cand=[x for x in rows if x['gold_mentions']>=1 and x['pop']<2000][:60]
names=list({x['t'].split(' (')[0] for x in cand})
pat=re.compile('|'.join(re.escape(n) for n in names))
cnt=Counter()
for j,tx in enumerate(texts):
    for m in set(pat.findall(tx)):
        if titles[j].split(' (')[0]!=m: cnt[m]+=1
print('backlink-ish mentions in shard0 (x41 for full):',sorted(cnt.values(),reverse=True)[:30], 'zero:',sum(1 for n in names if cnt[n]==0),'of',len(names))
