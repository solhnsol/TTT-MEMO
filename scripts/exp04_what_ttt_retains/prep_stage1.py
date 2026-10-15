"""Stage 1 data: held-out narrative streams with controlled histories. CPU only.
Per doc: target F (4 chunks) preceded by history H (24 chunks) under 5 conditions with identical token counts:
REL (true preceding text), SHUF (same history, sentences shuffled), UNREL (other document), CF (proper names
permuted within the doc), NONE (F alone). F is identical across conditions."""
import re,json,random,sys,collections
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[2]
TOK=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
HN,FN=24*1024,4*1024
def clean(t):
    a=re.search(r'\*\*\* ?START OF[^\n]*\n',t);b=re.search(r'\*\*\* ?END OF',t)
    if a:t=t[a.end():b.start() if b else None]
    return re.sub(r'\r','',t).strip()
def words(t):return [(m.group(0),m.start(),m.end()) for m in re.finditer(r"[A-Za-z][A-Za-z'’-]*",t)]
def name_set(t):
    cap=collections.Counter();low=collections.Counter()
    for m in re.finditer(r"(?<![.!?\"“”‘’'\n] )(?<!^)\b([A-Za-z][a-z]{2,})\b",t):
        w=m.group(1)
        (cap if w[0].isupper() else low)[w]+=1
    # capitalised mid-sentence words rarely seen lowercase
    return {w for w,c in cap.items() if c>=8 and low.get(w.lower(),0)<0.1*c and w.lower() not in STOP}
STOP=set('the and but for with that this then there when what where while which who whom whose how why not you your his her their our its mr mrs miss sir lord lady dr chapter yes oh well now here said'.split())
def permute_names(text,names,rng):
    top=[w for w,_ in collections.Counter(re.findall(r'\b[A-Z][a-z]{2,}\b',text)).most_common(400) if w in names][:40]
    perm=top[:];
    for _ in range(100):
        rng.shuffle(perm)
        if all(a!=b for a,b in zip(top,perm)):break
    mp=dict(zip(top,perm))
    return re.sub(r'\b[A-Z][a-z]{2,}\b',lambda m:mp.get(m.group(0),m.group(0)),text),mp
def sent_shuffle(text,rng):
    paras=re.split(r'(\n\s*\n)',text);sents=[]
    for p in paras:sents+=re.split(r'(?<=[.!?])\s+',p)
    rng.shuffle(sents);return ' '.join(sents)
def take_last(text,n):
    ids=TOK.encode(text,add_special_tokens=False).ids;return ids[-n:]
def build(path,seed,other_text):
    rng=random.Random(seed);text=clean(Path(path).read_text(errors='ignore'))
    enc=TOK.encode(text,add_special_tokens=False);ids=enc.ids
    if len(ids)<HN+FN+12000:return None
    # F starts at a random point in the last 60% leaving >= 8K tokens before it beyond H
    lo=int(len(ids)*0.35);hi=len(ids)-FN-1
    f0=rng.randint(max(lo,HN+4000),hi);f0=min(f0,hi)
    # align to sentence start: first token after '.\n' like boundary
    for j in range(f0,f0+400):
        s=TOK.decode(ids[j-1:j+1])
        if re.search(r'[.!?”"]\s*\n?\s*$',TOK.decode(ids[j-1:j])) :f0=j;break
    F=ids[f0:f0+FN];Hrel=ids[f0-HN:f0]
    pre_text=TOK.decode(ids[:f0]);names=name_set(text)
    cf_text,mp=permute_names(TOK.decode(ids[f0-HN-4000:f0]),names,rng);Hcf=take_last(cf_text,HN)
    Hshuf=take_last(sent_shuffle(TOK.decode(ids[f0-HN-4000:f0]),rng),HN)
    Hun=take_last(other_text[:int(len(other_text)*0.8)],HN)
    # pad (rare) short histories by repeating from front to keep exact token count
    bos=[128000]
    conds={'REL':bos+Hrel+F,'SHUF':bos+Hshuf+F,'UNREL':bos+Hun+F,'CF':bos+Hcf+F,'NONE':bos+F}
    for k,v in conds.items():assert (k=='NONE' and len(v)==FN+1) or len(v)==HN+FN+1,(k,len(v))
    return {'doc':Path(path).name,'f0':f0,'F':F,'names':sorted(names),'perm':mp,'conds':conds,'doc_text_ids_len':len(ids),
            'pre_ids':ids[:f0]}
def main(out,ndocs=40):
    files=sorted(Path(ROOT/'data/narrative_gutenberg').glob('pg*.txt'));rng=random.Random(20261012)
    texts={f.name:clean(f.read_text(errors='ignore')) for f in files}
    NON={'pg17.txt','pg1727.txt','pg6130.txt','pg205.txt','pg3600.txt','pg2147.txt','pg3176.txt','pg2500.txt','pg4300.txt','pg2600.txt'} # scripture/epic/essays/nonfiction/stream-of-consciousness excluded
    good=[f for f in files if len(texts[f.name])>230000 and f.name not in NON]
    rng.shuffle(good);recs=[]
    for i,f in enumerate(good):
        other=texts[good[(i+7)%len(good)].name]
        r=build(f,20261012+i,other)
        if r is None:continue
        recs.append(r)
        if len(recs)==ndocs:break
    Path(out).mkdir(parents=True,exist_ok=True)
    with open(Path(out)/'stage1_docs.jsonl','w') as fh:
        for r in recs:fh.write(json.dumps(r)+'\n')
    print(len(recs),[r['doc'] for r in recs][:60])
if __name__=='__main__':main(sys.argv[1])
