"""E62 P0a planting (CPU; new file; thread-server 11/23; request v1): invented facts, cell assignment, mention schedule, insertion into the book text at sentence boundaries, manifest. 3 books (first NCH chunks of 1024 llama3 tokens), 14 cells per book (n, gap kind): n = 2: close, far, edge; n = 4: close, mixed, far, edge; n = 8: close, mixed, edge; n = 16: close, mixed; n = 32: close, mixed. Gap = chunk distance between consecutive mentions (same chunk allowed): close all <= 5, mixed >= 1 gap <= 5 and >= 1 gap >= 11 (n = 32: >= 2 gaps >= 11), far all >= 11, edge all in 6..10. First mention in the first 60% of the book, mentions at sentence boundaries away from headings, <= 2 planted sentences of different facts inside 256 tokens, different template for every mention (n > 24: the 24 distinct templates are cycled, exact-repeat share recorded), consecutive mentions of a fact differ in style and first word.
  python3 e62_plant.py --selftest [--seed 0]"""
import sys,re,json,random,bisect,hashlib,pathlib,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import e62_templates as TP,e62_facts as FA
CH=1024;NCH=180;TYPES=('birth','school','ship','house','patron','mare')
BOOKS=('pg19test_92_Coningsby','pg19test_91_Old Mortality','pg19test_37_The Ragged')
CELLS=[(2,'close'),(2,'far'),(2,'edge'),(4,'close'),(4,'mixed'),(4,'far'),(4,'edge'),(8,'close'),(8,'mixed'),(8,'edge'),(16,'close'),(16,'mixed'),(32,'close'),(32,'mixed')]
FIRST=['Silas','Ezra','Alden','Orrin','Jabez','Lemuel','Hosea','Tobias','Ambrose','Cyrus','Edmund','Gilbert','Ivor','Josiah','Leander','Matthias','Nathaniel','Osmond','Percival','Quentin','Roland','Septimus','Thaddeus','Ulick','Valentine','Wilfred','Zachary','Barnaby','Crispin','Dudley','Eustace','Fenwick','Godfrey','Hilary','Isidore']
def syl(rng,kind):
    C='b d f g h k l m n p r s t v w z br dr fr gr pr tr st sh th'.split();V='a e i o u ae ea ia'.split()
    stem=''.join(rng.choice(C)+rng.choice(V) for _ in range(rng.choice([2,3])))
    end={'birth':['ford','ley','wick','by','ton','stead','mere','worth'],'school':['field','den','hurst','gate','ham','minster'],'ship':['a','ine','ora','ella','ia'],'house':['hall','lea','combe','dene','thorpe','holme'],'patron':['son','ham','worth','ley','ridge','well'],'mare':['ette','ina','elle','ora','ka']}[kind]
    return (stem+rng.choice(end)).capitalize()
def three_tokens(tk,w):return len(tk.encode(' '+w,add_special_tokens=False).ids)==3
def load_books(rec=ROOT.parent/'TTT-reg-stage0/data/natural-text/recur/sel24_books.jsonl'):
    allb={}
    for line in open(rec):
        r=json.loads(line);allb[r['id']]=r['text']
    return allb
def gen_facts(tk,allb,seed=0,pool_per_type=60):
    """-> answers[type] (pool_per_type valid answers), names (list of (first, last)); all absent from the 24 books, answers 3 tokens"""
    rng=random.Random(f'E62|{seed}|facts');low='\n'.join(allb.values()).lower();ans={};names=[]
    for ty in TYPES:
        out=[];seen=set()
        while len(out)<pool_per_type:
            w=syl(rng,ty)
            if w in seen or w.lower() in low or not three_tokens(tk,w):continue
            seen.add(w);out.append(w)
        ans[ty]=out
    seen=set()
    while len(names)<42+12:
        f=rng.choice(FIRST);l=syl(rng,'patron')
        if l in seen or l.lower() in low or f+' '+l in low:continue
        if l in sum(ans.values(),[]):continue
        seen.add(l);names.append((f,l))
    return ans,names
def assign_types(seed,book):
    """types of the 14 cells of one book: balanced (each type 2-3 cells), the big cells (n = 16, 32) on different types"""
    rng=random.Random(f'E62|{seed}|types|{book}')
    for _ in range(2000):
        t=[TYPES[i%6] for i in range(14)];rng.shuffle(t)
        big=[t[i] for i,(n,_) in enumerate(CELLS) if n>=16]
        dem=collections.Counter()
        for i,(n,_) in enumerate(CELLS):dem[t[i]]+=min(n,24)
        if len(set(big))==len(big) and max(dem.values())<=60:return t
    raise RuntimeError('no type assignment')
def gaps_for(rng,n,kind):
    g=n-1
    if kind=='close':return [rng.randint(0,5) for _ in range(g)]
    if kind=='far':return [rng.randint(11,20) for _ in range(g)]
    if kind=='edge':return [rng.randint(6,10) for _ in range(g)]
    nfar=2 if n==32 else 1;pf=0.5 if n<=8 else 0.12;out=[(rng.randint(11,20) if rng.random()<pf else rng.randint(0,5)) for _ in range(g)];idx=rng.sample(range(g),g)
    for i in idx[:nfar]:out[i]=rng.randint(11,20)
    if not any(x<=5 for x in out):out[idx[nfar]]=rng.randint(0,5)
    # the free gaps must not fall in 6..10 only; no further constraint
    return out
def schedule(seed,book,nch=NCH):
    """-> list of cells: dict(n, kind, type, chunks=[final chunk of every mention])"""
    rng=random.Random(f'E62|{seed}|sched|{book}');types=assign_types(seed,book);cells=[]
    for i,(n,kind) in enumerate(CELLS):
        for _ in range(500):
            g=gaps_for(rng,n,kind);span=sum(g);lo=8;hi=min(int(0.6*nch),nch-3-span)
            if hi<lo:continue
            f=rng.randint(lo,hi);ch=[f];[ch.append(ch[-1]+x) for x in g]
            if ch[-1]<=nch-3:break
        else:raise RuntimeError(('no schedule',n,kind))
        cells.append(dict(n=n,kind=kind,type=types[i],chunks=ch,gaps=g))
    return cells
def pick_templates(rng,cell,used):
    """templates for the mentions of a fact: min(n,24) distinct (not used by another fact of the book), >= 50% {F}, {A} position balanced, consecutive style / first word differ"""
    ty=cell['type'];pool=[(i,st,t) for i,(st,t) in enumerate(TP.POOL[ty]) if (ty,i) not in used];n=cell['n'];m=min(n,24)
    pos=lambda t:(lambda toks,k:('start' if k<len(toks)/3 else ('end' if k>=2*len(toks)/3 else 'mid')))(re.findall(r"\{[A-Z]+\}|[A-Za-z']+",t),[j for j,w in enumerate(re.findall(r"\{[A-Z]+\}|[A-Za-z']+",t)) if w=='{A}'][0])
    for _ in range(4000):
        sel=rng.sample(pool,m)
        if sum('{F}' in t for _,_,t in sel)<0.5*m:continue
        c=collections.Counter(pos(t) for _,_,t in sel)
        if m>=5 and min(c['start'],c['mid'],c['end'])<0.2*m:continue
        # order so that consecutive mentions differ in style and first word
        seq=sel[:];rng.shuffle(seq)
        for _ in range(200):
            bad=[i for i in range(1,len(seq)) if seq[i][1]==seq[i-1][1] or seq[i][2].split()[0]==seq[i-1][2].split()[0]]
            if not bad:break
            i=bad[0];j=rng.randrange(len(seq));seq[i],seq[j]=seq[j],seq[i]
        else:continue
        full=[seq[i%m] for i in range(n)]
        if n>m:                                           # cycled: re-check consecutive differences at the seams
            if any(full[i][1]==full[i-1][1] or full[i][2].split()[0]==full[i-1][2].split()[0] for i in range(1,n)):continue
        for i,_,_ in sel:used.add((ty,i))
        return full
    raise RuntimeError(('no template set',cell['type'],n))
def render(t,name,ans):
    f,l=name;return t.replace('{F}',f+' '+l).replace('{TL}','Mr. '+l).replace('{L}',l).replace('{A}',ans)
def sentence_bounds(text):
    out=[];i=0
    for m in re.finditer(r'(?<=[.!?])["”\']?\s+',text):
        out.append((m.end()));i=m.end()
    return out
def heading_near(text,b):
    seg=text[max(0,b-120):b+120]
    return bool(re.search(r'CHAPTER|BOOK [IVXL]+|[A-Z][A-Z \.]{12,}',seg))
def plant_book(tk,text,cells,facts,seed,book,nch=NCH):
    """-> (planted text, manifest rows, stats). facts[i] = (name, answer) per cell. Positions are planned in FINAL chunk coordinates and verified by re-tokenising."""
    rng=random.Random(f'E62|{seed}|plant|{book}');used=set();temps=[pick_templates(rng,c,used) for c in cells]
    bounds=sentence_bounds(text);enc=tk.encode(text,add_special_tokens=False);offs=enc.offsets;starts=[o[0] for o in offs]
    tokb=[bisect.bisect_left(starts,b) for b in bounds]                                   # token index of every sentence boundary (original text)
    jobs=[]                                                                               # (cell index, mention index, desired final chunk, sentence)
    for ci,c in enumerate(cells):
        for k,ch in enumerate(c['chunks']):jobs.append([ci,k,ch,render(temps[ci][k][2],facts[ci][0],facts[ci][1]),temps[ci][k]])
    shift=[0]*len(jobs);chosen=None;Dtok=[j[2]*CH+rng.randint(150,CH-150) for j in jobs]
    for it in range(30):
        pick=[];lastb=-10**9;taken=[]
        order=sorted(range(len(jobs)),key=lambda j:(jobs[j][2],j))
        for j in order:
            ci,k,ch,s,tmpl=jobs[j];tgt=Dtok[j]-shift[j]
            i0=bisect.bisect_left(tokb,tgt);cand=[x for x in range(max(0,i0-40),min(len(bounds),i0+40)) if x not in [p for _,p in pick] and not heading_near(text,bounds[x]) and all(abs(tokb[x]-tokb[p])>=260 or jobs[q][0]==ci for q,p in pick)]
            best=min(cand,key=lambda x:abs(tokb[x]-tgt)) if cand else None
            if best is None or abs(tokb[best]-tgt)>120:
                cand=[x for x in range(max(0,i0-200),min(len(bounds),i0+200)) if x not in [p for _,p in pick] and not heading_near(text,bounds[x]) and all(abs(tokb[x]-tokb[p])>=100 or jobs[q][0]==ci for q,p in pick)]
                best=min(cand,key=lambda x:abs(tokb[x]-tgt))
            x=best;pick.append((j,x))
        # build text with insertions (sorted by boundary; stable for equal boundaries)
        ins=sorted(pick,key=lambda jx:(bounds[jx[1]],jx[0]));parts=[];prev=0;spans=[]
        for j,x in ins:
            b=bounds[x];parts.append(text[prev:b]);cur=sum(len(p) for p in parts);parts.append(jobs[j][3]+' ');spans.append((j,cur,cur+len(jobs[j][3])));prev=b
        parts.append(text[prev:]);newt=''.join(parts);e2=tk.encode(newt,add_special_tokens=False);st2=[o[0] for o in e2.offsets];ok=True;newshift=list(shift);fin={}
        for j,a,b in spans:
            ts=bisect.bisect_left(st2,a);ch=ts//CH;fin[j]=(a,b,ts,ch)
            if ch!=jobs[j][2]:ok=False
            newshift[j]=shift[j]+(ts-Dtok[j])
        if ok:chosen=(newt,fin,e2);break
        shift=newshift
    if chosen is None:raise RuntimeError(('planting did not converge',book,[(jobs[j][0],jobs[j][1],jobs[j][2],fin[j][3],fin[j][2]-Dtok[j]) for j in range(len(jobs)) if fin[j][3]!=jobs[j][2]]))
    newt,fin,e2=chosen;rows=[]
    for j,(ci,k,ch,s,tmpl) in enumerate(jobs):
        a,b,ts,chk=fin[j];te=bisect.bisect_left([o[0] for o in e2.offsets],b);ans=facts[ci][1];a0=a+s.index(ans);a1=a0+len(ans);ans_tok=[i for i in range(ts,te) if e2.offsets[i][1]>a0 and e2.offsets[i][0]<a1];rows.append(dict(book=book,cell=ci,n=cells[ci]['n'],kind=cells[ci]['kind'],type=cells[ci]['type'],mention=k,chunk=chk,char=(a,b),tok=(ts,te),ans_tok=ans_tok,template=tmpl[0],style=tmpl[1],text=s))
    ntok=sum(r['tok'][1]-r['tok'][0] for r in rows);stats=dict(planted_tokens=ntok,total_tokens=len(e2.ids),planted_share=ntok/len(e2.ids),exact_repeat_share={ci:(1-len({t[0] for t in temps[ci]})/cells[ci]['n']) for ci in range(len(cells))})
    return newt,rows,stats
def selftest(seed=0):
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));allb=load_books();ans,names=gen_facts(tk,allb,seed)
    print({k:v[:3] for k,v in ans.items()},names[:3])
    assert all(three_tokens(tk,w) for v in ans.values() for w in v)
    bad,st=FA.check_pool();assert not bad,bad[:3]
    ids=[k for k in allb if any(k.startswith(b) for b in BOOKS)];assert len(ids)==3,ids
    nm=list(names);tot=0
    for bi,bk in enumerate(ids):
        cells=schedule(seed,bk);facts=[];used_a=collections.Counter()
        for ci,c in enumerate(cells):
            a=ans[c['type']][used_a[c['type']]];used_a[c['type']]+=1;facts.append((nm.pop(0),a))
        text=allb[bk];e0=tk.encode(text,add_special_tokens=False);cut=e0.offsets[min(len(e0.ids),(NCH+40)*CH)-1][1];text=text[:cut]
        newt,rows,stt=plant_book(tk,text,cells,facts,seed,bk)
        by=collections.defaultdict(list)
        for r in rows:by[r['cell']].append(r)
        for ci,c in enumerate(cells):
            ch=[r['chunk'] for r in sorted(by[ci],key=lambda r:r['mention'])];g=[b-a for a,b in zip(ch,ch[1:])];assert ch==c['chunks'],(bk,ci,ch,c['chunks']);n,kind=c['n'],c['kind']
            if kind=='close':assert all(x<=5 for x in g)
            if kind=='far':assert all(x>=11 for x in g)
            if kind=='edge':assert all(6<=x<=10 for x in g)
            if kind=='mixed':assert any(x<=5 for x in g) and sum(x>=11 for x in g)>=(2 if n==32 else 1)
        assert min(r['chunk'] for r in rows)>=0 and max(r['chunk'] for r in rows)<NCH
        first=[min(r['chunk'] for r in by[ci]) for ci in by];assert all(f<=0.6*NCH for f in first),first
        assert stt['planted_share']<=0.06,stt['planted_share']
        print(f'{bk[:40]}: {len(rows)} mentions, planted tokens {stt["planted_tokens"]} = {100*stt["planted_share"]:.2f}% of {stt["total_tokens"]}, types {[c["type"] for c in cells]}');tot+=len(rows)
    print(f'e62_plant selftest ok: {tot} mentions in 3 books (expected 426)')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest(int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 0)
