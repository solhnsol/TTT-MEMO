import json,re,hashlib,collections,sys
sys.path.insert(0,'/tmp/claude-0/work')
tok=lambda s:re.findall(r"\w+",s.lower())
def longest_run(q,c):
    qs,cs=tok(q),tok(c);pos=collections.defaultdict(list)
    for j,t in enumerate(cs):pos[t].append(j)
    best=0
    for i,t in enumerate(qs):
        for j in pos.get(t,[]):
            k=0
            while i+k<len(qs) and j+k<len(cs) and qs[i+k]==cs[j+k]:k+=1
            best=max(best,k)
    return best
S='/tmp/claude-0/sq/'
old=json.load(open('/mnt/project-files/data/public-qa/rewrite/paragraphs.json'))
oldctx={p['context'].strip() for p in old}; olditems={i for p in old for i in p['item_ids']}
pool=[]
for split in('train','dev'):
    for a in json.load(open(S+split+'.json'))['data']:
        for pi,p in enumerate(a['paragraphs']):
            if len(p['qas'])!=1: continue
            qa=p['qas'][0]; golds=[]
            for x in qa['answers']:
                if x['text'] not in golds: golds.append(x['text'])
            pool.append(dict(split=split,title=a['title'],ctx=p['context'],qid=qa['id'],q=qa['question'].strip(),golds=golds))
print('one-question paragraphs',len(pool))
ex=[r for r in pool if r['title'] in('Super_Bowl_50','Warsaw')]; print('excluded titles',len(ex))
ov=[r for r in pool if r['ctx'].strip() in oldctx or r['qid'] in olditems]; print('overlap with public QA 1000',len(ov))
pool=[r for r in pool if r['title'] not in('Super_Bowl_50','Warsaw') and r['ctx'].strip() not in oldctx and r['qid'] not in olditems]
print('after',len(pool))
elig=[]
for r in pool:
    g=r['golds'][0]; n=len(r['ctx'].split())
    if len(g)<2 or len(g.split())>6 or g.lower() not in r['ctx'].lower(): continue
    if longest_run(r['q'],r['ctx'])>3: continue
    if not 40<=n<=160: continue
    elig.append(r)
print('eligible',len(elig))
H=lambda s:hashlib.md5(('e57-qa128-v1'+s).encode()).hexdigest()
by=collections.defaultdict(list)
for r in sorted(elig,key=lambda r:H(r['qid'])):by[r['title']].append(r)
titles=sorted(by,key=H);ch=[]
while len(ch)<128 and any(by.values()):
    for t in titles:
        if by[t] and len(ch)<128:ch.append(by[t].pop(0))
for r in ch:
    r['pid']=hashlib.md5(r['qid'].encode()).hexdigest()[:10]
print(len(ch),len({r['title'] for r in ch}),sum(len(r['ctx'].split()) for r in ch)/128)
json.dump(ch,open('/tmp/claude-0/work/paragraphs128.json','w'),ensure_ascii=False,indent=0)
