"""E55 pairless fact world (CPU only; needs a tokenizer only through the `enc` callable). Design: /mnt/project-files/experiments/E55-controlled-world-binding-design.md section 10 (v2, 2026-10-22 14:50 KST; Hans 14:46 KST: no pairs, no worlds A/B).
A world has k facts (name X, place P, occupation o), split into p groups (fact f -> group f % p; names / places are drawn anew for every seed, so the group assignment is random with respect to spelling). Chunk c (exactly 1024 tokens) carries the n sentences of every fact of group c % p
(k n / p sentences per chunk, each sentence an independent block at a random position between background units, random order, no fixed neighbours, none touches a chunk edge), a fact appears n_app = 16 times (chunks g, g+p, ...), the stream has 16 p chunks and is read ONCE. The n sentences of an
appearance use different frames of the bank; one fact never repeats a frame while the bank is large enough (50 frames: 16 n <= 50).
Background = units of at most ~40 tokens cut from the narrative paragraphs (sentence boundaries), because real paragraphs are long and a chunk with 16 sentences leaves room for only a few of them (the pair world asserted 'too few background paragraphs per chunk').
Decoy world = the same generator, different seed, names / places disjoint from the real world, same k, n, p, bank; its background units come from the second half of the paragraph pool (real world: first half).
Items (design 10.2): one per fact, question frame = fact id mod 3, 16 candidate places = gold + 15 places of other facts (p >= 2: 7 of the same group + 8 of other groups, fewer same-group candidates if the group is smaller; p = 1: 15 random), order random and fixed by the seed; plus 4 control items (a name that never appears, fake gold = a random exposed place).
Everything is deterministic in (seed, k, n, p, n_app, bank). enc(text) -> token ids is passed in (server: tokenizers.Tokenizer.from_file(data/tokenizer/llama3/tokenizer.json), load_enc)."""
import random,re,json,sys,hashlib
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import gen_world_ctl as G0
from gen_world_ctl import FRAMES50,QFRAMES,OCC,SYL1X,SYL2X,PL_PREX,PL_SUFX,A,sha12,load_enc,_synth_bg,default_bg,question_overlap,_h
class StubTok:
    """CPU tests without the tokenizers package: words and punctuation are tokens, one injective id per distinct string (assigned in order of first use, so streams are reproducible for a fixed call order; the injectivity makes the fact-free checks meaningful)"""
    def __init__(self):self.v={}
    def __call__(self,text):
        out=[]
        for w in (re.findall(r"\w+|[^\w\s]|\n\n",text.replace('\n\n',' \n\n ')) if text else []):out.append(self.v.setdefault(w,len(self.v)+1))
        return out
CH=1024;WIN=8192;NQ=3;NCAND=16;NCTRL=4;READ_APPS=(2,4,8,16);UNIT_TOK=40
def _contexts(bank,slot):
    """distinct local contexts in which `slot` ('{X}' name or '{P}' place) stands in the bank frames (<=4 words each side; a slot at the very start of a sentence has an empty left side and no leading space, so it can tokenise differently)"""
    out=set()
    for f in bank:
        if slot not in f:continue
        pre,post=f.split(slot,1);pre=' '.join(pre.split(' ')[-4:]) if pre and pre.strip() else pre;post=' '.join(post.split(' ')[:4])
        out.add(re.sub(r'\{\w+\}','Aldmere',pre)+'{N}'+re.sub(r'\{\w+\}','Aldmere',post))
    return sorted(out)
def _grams(ids,n=13):
    ids=list(ids);return {tuple(ids[i:i+n]) for i in range(len(ids)-n+1)}
def reference_grams(enc,root=None):
    """13-gram sets (token ids) of the texts the background must never share a phrase with: the neutral SQuAD readout context (dev paragraphs [:400]), the canary books pg11 / pg1342, the frame bank sentences (slots removed) and the question frames. A missing file = None (the callers treat that as failure)."""
    import gen_world as gw
    root=root or gw.ROOT;out={};sq=root/'data/squad/dev-v1.1.json'
    out['SQuAD readout context']=_grams(enc(' '.join(' '.join(p['context'].split()) for p in [p for x in json.load(open(sq))['data'] for p in x['paragraphs']][:400]))) if sq.exists() else None
    for b in ('pg11','pg1342'):
        f=root/f'data/narrative_gutenberg/{b}.txt';out['canary '+b]=_grams(enc(f.read_text(errors='ignore'))) if f.exists() else None
    bank=G0.frame_bank('300',0,0,0)[0];g=set()
    for t in bank:
        for x in re.split(r'\{\w+\}',t):
            if len(x.split())>=3:g|=_grams(enc(x))
    out['frame bank (slots removed)']=g;out['question frames']=set().union(*[_grams(enc(q.replace('{X}','X'))) for q in QFRAMES]);return out
def _clean_paragraphs(books):
    import gen_world as gw
    out=[];seen=set()
    for f in sorted((gw.ROOT/'data/narrative_gutenberg').glob('pg*.txt')):
        if f.name not in books:continue
        t=f.read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t;t=t[len(t)//5:];t=re.sub(r'\r','',t)
        for p in [p for p in re.split(r'\n\s*\n',t) if 200<len(p)<900 and not re.search(r'CHAPTER|\bVol|\[|\]|\*',p)]:
            p=re.sub(r'(?<=[a-z,;] )(?:[A-Z][a-z]+)(?: [A-Z][a-z]+)*','someone',p);p=re.sub(r"\b(?:Mr|Mrs|Miss|Dr|Sir|Lady|Lord)\.? \w+",'the stranger',p);p=' '.join(p.split())
            if p not in seen:seen.add(p);out.append(p)           # exact duplicates (after the name stripping) removed
    return out
_POOL_CACHE={}
def default_bg_pool(seed,enc=None,refs=None,books=('pg98.txt','pg1260.txt','pg768.txt','pg766.txt')):
    """E55 v3 (design 13.3): paragraph pool of the four narrative books, cleaned like default_bg_all (names stripped, 200-900 characters), exact duplicates removed, and (when enc is given) FILTERED at the level of the background units the streams really use (make_units): in a fixed canonical order a paragraph is dropped if any of its units is a unit already kept, if any of its 13-grams was already kept in another paragraph, or if it shares a 13-gram with a reference text (refs = reference_grams(enc): SQuAD readout context, canary books, frame bank, question frames; a None entry raises). The kept paragraphs are then shuffled with a SEED-SPECIFIC order; consumers take paragraphs from the front of their own slice (no cycling)."""
    key=(tuple(books),id(enc) if enc is not None else None)
    if enc is None:paras=_clean_paragraphs(books)
    else:
        assert refs is not None and all(v is not None for v in refs.values()),'reference texts missing: '+str([k for k,v in (refs or {}).items() if v is None])
        if key not in _POOL_CACHE:
            sep=enc('\n\n');ref=set().union(*refs.values());seen_u=set();seen_g=set();kept=[];drop={'unit':0,'13gram':0,'ref':0}
            for p in sorted(_clean_paragraphs(books)):
                us=[tuple(u[0][:-len(sep)]) for u in make_units([p],enc,sep)];per=[_grams(u) for u in us];gs=set().union(*per) if us else set()
                if sum(len(x) for x in per)!=len(gs):drop['13gram']+=1;continue            # a phrase repeated inside the paragraph (two units share a 13-gram)
                if len(us)!=len(set(us)) or any(u in seen_u for u in us):drop['unit']+=1;continue
                if gs&ref:drop['ref']+=1;continue
                if gs&seen_g:drop['13gram']+=1;continue
                seen_u.update(us);seen_g|=gs;kept.append(p)
            _POOL_CACHE[key]=(kept,drop)
        paras=_POOL_CACHE[key][0]
    out=paras[:];random.Random(f'E55bgpool|{seed}').shuffle(out);return out
def default_bg_all(n=40000):
    """background paragraphs of ALL four narrative books of the E35 source (same cleaning as gen_world.background, which uses one random book): about 4x more units, needed for the largest worlds (p = 9: 144 chunks). NOT the default of the 180a / 180b runs (they use gen_world_ctl.default_bg, whose streams are pinned by their stream hashes)."""
    import gen_world as gw
    out=[];
    for f in sorted((gw.ROOT/'data/narrative_gutenberg').glob('pg*.txt')):
        if f.name not in ('pg98.txt','pg1260.txt','pg768.txt','pg766.txt'):continue
        t=f.read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t;t=t[len(t)//5:];t=re.sub(r'\r','',t)
        for p in [p for p in re.split(r'\n\s*\n',t) if 200<len(p)<900 and not re.search(r'CHAPTER|\bVol|\[|\]|\*',p)]:
            p=re.sub(r'(?<=[a-z,;] )(?:[A-Z][a-z]+)(?: [A-Z][a-z]+)*','someone',p);p=re.sub(r"\b(?:Mr|Mrs|Miss|Dr|Sir|Lady|Lord)\.? \w+",'the stranger',p);out.append(' '.join(p.split()))
    random.Random(5500).shuffle(out);return out[:n]
def _alloc(seed,k,enc,bank,decoy_world=True,nu=0):
    """names / places in the modal token-length class (leading space). The REAL names (places) are drawn in PAIRS (real_i, fake_i) with the same token length in every local context of the frame bank, so that the fake window ('names' decoy: same chunks, fake names / places) keeps every sentence length. Returns {'persons': (real k, fake k, ctrl NCTRL, decoy-world k), 'places': (real k, fake k, unexposed nu, decoy-world k)}"""
    nt=lambda s:len(enc(' '+s));res={}
    for what,pool,slot,rest in (('persons',[a+b for a in SYL1X for b in SYL2X],'{X}',NCTRL),('places',[a+b for a in PL_PREX for b in PL_SUFX],'{P}',nu)):
        byl={}
        for n in sorted(set(pool)):byl.setdefault(nt(n),[]).append(n)
        L=max(byl,key=lambda x:len(byl[x]));cls=byl[L][:];need=2*k+rest+(k if decoy_world else 0);assert len(cls)>=need,(what,'token length',L,'class size',len(cls),'need',need,{x:len(v) for x,v in sorted(byl.items())},'extend the E55 pools')
        cx=_contexts(bank,slot);sig={n:tuple(len(enc(m.replace('{N}',n))) for m in cx) for n in cls};g={}
        for n in cls:g.setdefault(sig[n],[]).append(n)
        r=random.Random(f'E55f|{seed}|{what}|pairs');pairs=[]
        for key in sorted(g):
            v=g[key][:];r.shuffle(v);pairs+=[(v[i],v[i+1]) for i in range(0,len(v)-1,2)]
        assert len(pairs)>=k,(what,'not enough sentence-length-compatible (real, fake) pairs',len(pairs),'need',k,'signature classes',sorted(len(v) for v in g.values()),'contexts',len(cx))
        r.shuffle(pairs);use=pairs[:k];used={x for pr in use for x in pr};left=[x for x in cls if x not in used];random.Random(f'E55f|{seed}|{what}|rest').shuffle(left)
        assert len(left)>=rest+(k if decoy_world else 0),(what,'pool too small for control / decoy-world names',len(left))
        res[what]=([a for a,_ in use],[b for _,b in use],left[:rest],left[rest:rest+(k if decoy_world else 0)],L)
    return res
def fresh_places(world,enc,n=16):
    """E55 design 17: n never-exposed place names with the SAME token length as the world's places (modal class, leading space), disjoint from every place of the real world, the fake window ('names'), the decoy world and the existence-check unexposed list; deterministic in (seed, world). They never occur in any stream."""
    used={f['P'] for f in world['facts']}|{f['P'] for f in world.get('names',{}).get('facts',[])}|{f['P'] for f in world.get('decoy',{}).get('facts',[])}|set(world.get('unexposed',[]))
    L=world['place_len'];cls=sorted({a+b for a in PL_PREX for b in PL_SUFX if len(enc(' '+a+b))==L}-used);assert len(cls)>=n,('too few fresh places',len(cls));r=random.Random(f"E55fresh|{world['seed']}|{world['k']}")
    out=r.sample(cls,n);assert len(set(out))==n and not set(out)&used and all(len(enc(' '+x))==L for x in out);return out
def make_units(bg,enc,sep,maxtok=UNIT_TOK):
    """background units: [(ids + sep, text)], each <= maxtok tokens (+ sep), cut at sentence ends (a sentence longer than maxtok is cut by tokens)"""
    out=[]
    for para in bg:
        cur=[];txt=[]
        def flush():
            if cur:out.append((cur[:]+sep,' '.join(txt)))
            cur.clear();txt.clear()
        for s in [x for x in re.split(r'(?<=[.!?])\s+',' '.join(para.split())) if x]:
            first=enc(s)
            if len(first)>maxtok:
                flush()
                for i in range(0,len(first),maxtok):out.append((first[i:i+maxtok]+sep,s))
                continue
            ids=enc(' '+s) if cur else first
            if cur and len(cur)+len(ids)>maxtok:flush();ids=first
            cur+=ids;txt.append(s)
        flush()
    return out
def _make_facts(names,places,seed,k,tag,bank_kind,bank,owner,n,n_app):
    r=random.Random(f'E55f|{seed}|occ|{tag}');return [{'id':f,'group':None,'X':names[f],'P':places[f],'o':r.choice(OCC),'frames':G0.person_frames(bank_kind,bank,owner,f'{seed}|{tag}',f,f,n_app*n)} for f in range(k)]
def _stream(facts,k,n,p,n_app,bank,enc,units,seed,tag,sep):
    """-> chunks [{'k','group','app','ids' (exactly 1024),'sents':[{fact,frame,app,start,end}]}], units used"""
    nch=n_app*p;ui=0;chunks=[];stats=[]
    for c in range(nch):
        g=c%p;app=c//p;rng=random.Random(f'E55f|{seed}|chunk|{tag}|{c}');mine=[f for f in facts if f['id']%p==g];assert len(mine)==k//p
        sents=[]
        for f in mine:
            for j in range(n):
                fi=f['frames'][app*n+j];t=bank[fi].format(X=f['X'],P=f['P'],o=f['o'],a_o=A(f['o']));ids=enc(t);sents.append((f['id'],fi,ids+sep,t))
        rng.shuffle(sents);cost=sum(len(s[2]) for s in sents);assert cost<CH-100,('sentences fill the chunk',cost,len(sents))
        rem=CH-cost;whole=[];tail=None
        while True:
            assert ui<len(units),('background units exhausted',tag,c)
            ids=units[ui][0]
            if len(ids)<=rem:whole.append(ids);rem-=len(ids);ui+=1
            else:
                if rem>0:tail=ids[:rem]
                ui+=1;break
        nb=len(whole);assert nb>=2,('too few background units per chunk',nb);stats.append(nb)
        slots=[rng.randint(1,nb-1) for _ in sents];order=sorted(range(len(sents)),key=lambda i:(slots[i],i))        # sentences were shuffled: ties keep a random order
        ids_=[];sp=[];oi=0
        for pos in range(nb+1):
            while oi<len(order) and slots[order[oi]]==pos:
                f_,fi,ids,t_=sents[order[oi]];sp.append({'fact':f_,'frame':fi,'app':app,'start':len(ids_),'end':len(ids_)+len(ids)-len(sep),'text':t_});ids_+=ids;oi+=1
            if pos<nb:ids_+=whole[pos]
        if tail:ids_+=tail
        assert len(ids_)==CH,(c,len(ids_));chunks.append({'k':c,'group':g,'app':app,'ids':ids_,'sents':sp})
    return chunks,ui,min(stats)
def bg_stream(nch,enc,units,seed,sep):
    """background-only stream (no sentence at all) of nch chunks of exactly 1024 tokens: the fact-free 'fake window' of arm SWB"""
    ui=0;chunks=[]
    for c in range(nch):
        rem=CH;ids_=[]
        while True:
            assert ui<len(units),('background units exhausted','bgonly',c);ids=units[ui][0];ui+=1
            if len(ids)<=rem:ids_+=ids;rem-=len(ids)
            else:ids_+=ids[:rem];break
        assert len(ids_)==CH;chunks.append({'k':c,'group':c%1,'app':c,'ids':ids_,'sents':[]})
    return chunks
def build_fact_world(seed,k,n,p,n_app=16,bank='50',enc=None,bg=None,decoys=('names','world','nofact'),read_apps=READ_APPS):
    """bg: list of background paragraphs (str); enc: text -> ids. decoys: which fake streams to build (each is optional, so the definition of the fake window can change without touching the real stream):
      'names'  = the REAL chunks with the names and places of every fact replaced by token-length-compatible fakes (same frames, positions, background; fact-free in the sense that no real fact is in it)
      'world'  = a second fact world (other seed, other names / places / background half)
      'nofact' = the REAL chunks with every fact sentence removed and its span filled with background text of exactly the same token length (all other tokens identical; no name / place of the world in it)
    The real stream does not depend on which decoys are requested except that the background pool is cut in halves ('world' without 'bgonly') or thirds (both)."""
    assert enc is not None and bg is not None and p>=1 and k%p==0 and n>=1 and (k>=NCAND or (p==1 and k>=2)),'k < 16 only as the p = 1 existence check'
    nu=max(0,NCAND-k)           # existence check (k < 16): the 16 candidate slots = k exposed places + 16 - k never-exposed places (familiarity control)
    bk,owner=G0.frame_bank(bank,seed,k,n_app*n);dw='world' in decoys;al=_alloc(seed,k,enc,bk,dw,nu);sep=enc('\n\n');Lp=al['persons'][4];Ll=al['places'][4]
    facts=_make_facts(al['persons'][0],al['places'][0],seed,k,'real',bk and bank,bk,owner,n,n_app)
    for f in facts:f['group']=f['id']%p
    ctrl=[{'id':k+i,'X':x} for i,x in enumerate(al['persons'][2])]
    nth=3 if (dw and 'nofact' in decoys) else 2;third=len(bg)//nth
    w={'seed':seed,'k':k,'n':n,'p':p,'n_app':n_app,'bank_kind':bank,'n_chunks':n_app*p,'name_len':Lp,'place_len':Ll,'bank':bk,'facts':facts,'ctrl':ctrl,'unexposed':al['places'][2],'read_apps':[a for a in read_apps if a<=n_app],'read_chunks':[a*p-1 for a in read_apps if a<=n_app],'sep':sep,'decoys':list(decoys)}
    u=make_units(bg[:third],enc,sep);w['chunks'],used,w['min_units_per_chunk']=_stream(facts,k,n,p,n_app,bk,enc,u,seed,'real',sep);w['units_used']=used
    if 'names' in decoys:
        fake=[{**f,'X':al['persons'][1][f['id']],'P':al['places'][1][f['id']]} for f in facts]
        u2=make_units(bg[:third],enc,sep);ch,_,_=_stream(fake,k,n,p,n_app,bk,enc,u2,seed,'real',sep);w['names']={'facts':fake,'chunks':ch}
        for a_,b_ in zip(w['chunks'],ch):assert [(x['start'],x['end']) for x in a_['sents']]==[(x['start'],x['end']) for x in b_['sents']] and len(a_['ids'])==len(b_['ids'])==CH,('fake chunk layout differs from the real chunk',a_['k'])
    if dw:
        dseed=seed+100000;dper=_make_facts(al['persons'][3],al['places'][3],dseed,k,'decoy',bank,bk,owner,n,n_app)
        for f in dper:f['group']=f['id']%p
        ud=make_units(bg[third:2*third],enc,sep);w['decoy']={'seed':dseed,'facts':dper};w['decoy']['chunks'],_,_=_stream(dper,k,n,p,n_app,bk,enc,ud,dseed,'decoy',sep)
    if 'nofact' in decoys:
        fill=[t for u in make_units(bg[(nth-1)*third:],enc,sep) for t in u[0][:-len(sep)]];fi_=0;ch=[]
        for c in w['chunks']:
            ids=c['ids'][:]
            for x in c['sents']:
                L_=x['end']-x['start'];assert fi_+L_<=len(fill),'filler background exhausted';ids[x['start']:x['end']]=fill[fi_:fi_+L_];fi_+=L_
            assert len(ids)==CH;ch.append({'k':c['k'],'group':c['group'],'app':c['app'],'ids':ids,'sents':[]})
        w['nofact']={'chunks':ch}
    return w
def realize(world,kind='real'):
    """-> list of chunks {'k','ids' (exactly 1024),'sents'} (the chunks are stored in the world; kept as a function for the pair-world API)"""
    if kind is True:kind='world'
    return {'real':lambda:world['chunks'],'names':lambda:world['names']['chunks'],'world':lambda:world['decoy']['chunks'],'nofact':lambda:world['nofact']['chunks']}[kind]()
def stream_ids(chunks):return [t for c in chunks for t in c['ids']]
def stream_sha(chunks):return _h(*[_h(*c['ids']) for c in chunks])[:16]
def build_items(world):
    """items keyed by fact (kind 'fact') + NCTRL control items (kind 'ctrl'); candidates ' '+place"""
    facts=world['facts'];seed=world['seed'];p=world['p'];k=world['k'];items=[]
    def cands(r,f,pool):
        others=[g for g in pool if g['id']!=f['id']]
        if p>=2 and k>=NCAND:
            same=[g for g in others if g['group']==f['group']];oth=[g for g in others if g['group']!=f['group']];r.shuffle(same);r.shuffle(oth);ns=min(7,len(same));cs=same[:ns]+oth[:NCAND-1-ns]
        else:r.shuffle(others);cs=others[:NCAND-1]
        assert len(cs)==min(NCAND,k)-1;return cs
    ux=world.get('unexposed',[])
    def tail(d):
        d['cands']+=[' '+u for u in ux];d['cand_fact']+=[-1]*len(ux);d['cand_group']+=[-1]*len(ux);d['cand_same']+=[False]*len(ux);d['cand_exposed']=[True]*(NCAND-len(ux))+[False]*len(ux);d['n_rank']=NCAND-len(ux);return d
    for f in facts:
        r=random.Random(f"E55f|{seed}|q|{f['id']}");cs=cands(r,f,facts);cl=[f]+cs;r.shuffle(cl);gi=[x['id'] for x in cl].index(f['id'])
        items.append(tail({'kind':'fact','fact':f['id'],'group':f['group'],'name_id':f['id'],'place_id':f['id'],'name':f['X'],'qframe':f['id']%NQ,'prefix':QFRAMES[f['id']%NQ].format(X=f['X']),'cands':[' '+x['P'] for x in cl],'gold_idx':gi,'cand_fact':[x['id'] for x in cl],'cand_group':[x['group'] for x in cl],'cand_same':[x['group']==f['group'] for x in cl]}))
    for i,c in enumerate(world['ctrl']):
        r=random.Random(f"E55f|{seed}|qc|{i}");fg=facts[r.randrange(k)];cs=cands(r,fg,facts);cl=[fg]+cs;r.shuffle(cl);gi=[x['id'] for x in cl].index(fg['id'])
        items.append(tail({'kind':'ctrl','fact':-1,'group':-1,'name_id':c['id'],'place_id':fg['id'],'name':c['X'],'qframe':c['id']%NQ,'prefix':QFRAMES[c['id']%NQ].format(X=c['X']),'cands':[' '+x['P'] for x in cl],'gold_idx':gi,'cand_fact':[x['id'] for x in cl],'cand_group':[x['group'] for x in cl],'cand_same':[False]*len(cl)}))
    return items
def rows(items):
    """item-major rows [(prefix, candidate text)] (row i x NCAND + c)"""
    return [(it['prefix'],c) for it in items for c in it['cands']]
def item_meta(items):
    m={x:[it[x] for it in items] for x in ('kind','fact','group','qframe','name_id','place_id','gold_idx','cand_fact','cand_group','cand_same')};m['gi']=list(m['gold_idx']);m['cand_exposed']=[it['cand_exposed'] for it in items];m['n_rank']=[it['n_rank'] for it in items];return m
def verify(world,enc):
    k,n,p,n_app=world['k'],world['n'],world['p'],world['n_app'];res={};A_=realize(world);sep=world['sep']
    assert len(A_)==n_app*p and all(len(c['ids'])==CH for c in A_);res['chunk_len_ok']=True
    for c in A_:
        assert len(c['sents'])==k*n//p and {x['fact'] for x in c['sents']}=={f['id'] for f in world['facts'] if f['id']%p==c['k']%p}
        for x in c['sents']:assert x['app']==c['k']//p and x['start']>0 and x['end']<CH-len(sep)
    res['sentences_per_chunk']=k*n//p;res['min_units_per_chunk']=world['min_units_per_chunk']
    cnt={}
    for c in A_:
        for x in c['sents']:cnt.setdefault(x['fact'],[]).append(c['k'])
    assert len(cnt)==k and all(len(v)==n*n_app and len(set(v))==n_app for v in cnt.values())
    mn=None
    for f in range(k):
        sp=[(c['k']*CH+x['start'],c['k']*CH+x['end'],c['k']) for c in A_ for x in c['sents'] if x['fact']==f]
        for a in range(n_app-1):
            e=[x for x in sp if x[2]==a*p+f%p];nx=[x for x in sp if x[2]==(a+1)*p+f%p];d=min(y[0]-x[1] for x in e for y in nx);mn=d if mn is None else min(mn,d)
    res['min_gap_between_appearances']=mn
    if p>=9:assert mn>=WIN+1,('appearance distance',mn)
    res['frame_repeats_per_fact']=max(len(f['frames'])-len(set(f['frames'])) for f in world['facts'])
    if world['bank_kind'] in ('50','disjoint','300'):assert res['frame_repeats_per_fact']==0
    res['question_ngram_overlap']=question_overlap(world['bank']);assert res['question_ngram_overlap']==0
    names={x['X'] for x in world['facts']}|{x['X'] for x in world['ctrl']};pl={x['P'] for x in world['facts']};assert len(names)==k+NCTRL and len(pl)==k
    if 'decoy' in world:
        dn={x['X'] for x in world['decoy']['facts']};dp={x['P'] for x in world['decoy']['facts']};assert not names&dn and not pl&dp;res['decoy_disjoint']=True
        D_=realize(world,'world');assert len(D_)==len(A_) and all(len(c['ids'])==CH for c in D_);res['sha_stream_decoy']=stream_sha(D_)
    # names / places (with and without a leading space) must not occur as token subsequences in the decoy and background-only streams
    strs=[(kd,stream_ids(realize(world,kd))) for kd in ('names','world','nofact') if {'names':'names','world':'decoy','nofact':'nofact'}[kd] in world]
    if 'names' in world:
        fn={x['X'] for x in world['names']['facts']};fp={x['P'] for x in world['names']['facts']};assert len(fn)==len(fp)==k and not fn&names and not fp&pl and not fp&set(world['unexposed']) and not fn&{x['X'] for x in world['ctrl']};res['names_decoy_disjoint']=True
        for a_,b_ in zip(A_,realize(world,'names')):assert len(a_['ids'])==len(b_['ids'])==CH and [(x['start'],x['end']) for x in a_['sents']]==[(x['start'],x['end']) for x in b_['sents']] and a_['ids']!=b_['ids']
        res['sha_stream_names']=stream_sha(realize(world,'names'))
    if 'nofact' in world:
        for a_,b_ in zip(A_,realize(world,'nofact')):
            keep=np.ones(CH,bool)
            for x in a_['sents']:keep[x['start']:x['end']]=False
            assert np.array_equal(np.asarray(a_['ids'])[keep],np.asarray(b_['ids'])[keep]) and (np.asarray(a_['ids'])[~keep]!=np.asarray(b_['ids'])[~keep]).any(),'nofact chunk differs outside the sentence spans'
        res['sha_stream_nofact']=stream_sha(realize(world,'nofact'))
    for nm,ids in strs:
        ids=np.asarray(ids);hits=0
        for f in world['facts']+world['ctrl']:
            for w_ in [f['X']]+([f['P']] if 'P' in f else [])+list(world.get('unexposed',[]) if f['id']==0 else []):
                for t in (enc(w_),enc(' '+w_)):
                    t=np.asarray(t);pos=np.where(ids[:len(ids)-len(t)+1]==t[0])[0]
                    hits+=sum(1 for q in pos if np.array_equal(ids[q:q+len(t)],t))
        if nm=='names':hits+=0
        assert hits==0,(nm,'stream contains a name / place of the real world',hits)
        res[{'names':'names_decoy','world':'decoy','nofact':'nofact'}[nm]+'_fact_free']=True
    items=build_items(world);assert len(items)==k+NCTRL
    for it in items:
        assert len(it['cands'])==NCAND==len(set(it['cands'])) and it['cands'][it['gold_idx']]==' '+world['facts'][it['place_id']]['P']
        if it['kind']=='fact':
            assert it['cand_fact'][it['gold_idx']]==it['fact'] and sum(1 for c in it['cand_fact'] if c==it['fact'])==1
            if p>=2 and k>=NCAND:assert sum(it['cand_same'])-1==min(7,k//p-1)
        else:assert it['name'] not in {x['X'] for x in world['facts']}
    res['n_items']=len(items);res['sha_stream']=stream_sha(A_);res['n_sentences_total']=sum(len(c['sents']) for c in A_);return res,A_
def stats(world,A_):
    k,n,p=world['k'],world['n'],world['p'];tok=[sum(x['end']-x['start'] for x in c['sents']) for c in A_];same=[];prior=[]
    for c in A_:
        kk=c['k']
        if kk<8:continue
        win=[x for c2 in A_[kk-8:kk] for x in c2['sents']];prior.append(sum(1 for j in range(kk-8,kk) if j%p==kk%p))
        for x in c['sents']:
            o=[y for y in win if y['fact']!=x['fact']]
            if o:same.append(sum(1 for y in o if y['frame']==x['frame'])/len(o))
    return {'k':k,'n':n,'p':p,'bank':world['bank_kind'],'n_chunks':world['n_chunks'],'sent_per_chunk':k*n//p,'sent_tokens_per_chunk':sum(tok)/len(tok),'sent_share':sum(tok)/len(tok)/CH,'min_units_per_chunk':world['min_units_per_chunk'],'earlier_appearances_in_window':(sum(prior)/len(prior) if prior else 0.0),'same_frame_share_window':(sum(same)/len(same) if same else 0.0),'read_chunks':world['read_chunks']}
def fmt_stats(rows_):
    L=['| k facts | n | p | bank | chunks | sentences / chunk | sentence tokens / chunk (share) | min background units / chunk | earlier appearances of a fact inside the 8-chunk window | share of window sentences of OTHER facts with the same frame | read after chunk index (appearances 2,4,8,16) |','|'+'---|'*11]
    for r in rows_:L.append(f"| {r['k']} | {r['n']} | {r['p']} | {r['bank']} | {r['n_chunks']} | {r['sent_per_chunk']} | {r['sent_tokens_per_chunk']:.0f} ({100*r['sent_share']:.1f}%) | {r['min_units_per_chunk']} | {r['earlier_appearances_in_window']:.2f} | {r['same_frame_share_window']:.3f} | {r['read_chunks']} |")
    return '\n'.join(L)
CELLS={'base':dict(k=32,n=1,p=2,bank='50'),'p1':dict(k=16,n=1,p=1,bank='50'),'p4':dict(k=64,n=1,p=4,bank='50'),'p9':dict(k=144,n=1,p=9,bank='50'),'k16':dict(k=16,n=1,p=2,bank='50'),'k64':dict(k=64,n=1,p=2,bank='50'),'n2':dict(k=32,n=2,p=2,bank='50'),'n2k16':dict(k=16,n=2,p=2,bank='50'),'bank12':dict(k=32,n=1,p=2,bank='12'),'disjoint':dict(k=32,n=1,p=2,bank='disjoint'),'exist8':dict(k=8,n=1,p=1,bank='50'),
       # design v3 (E55 section 13): 128 exposures per fact, bank of 300 frames, p = 1; read points in APPEARANCES = exposures / n (exposures 16, 32, 64, 128)
       'v3_16_1':dict(k=16,n=1,p=1,bank='300',n_app=128,read_apps=(16,32,64,128)),'v3_8_1':dict(k=8,n=1,p=1,bank='300',n_app=128,read_apps=(16,32,64,128)),'v3_8_4':dict(k=8,n=4,p=1,bank='300',n_app=32,read_apps=(4,8,16,32)),
       'v3_4_8':dict(k=4,n=8,p=1,bank='300',n_app=16,read_apps=(2,4,8,16)),'v3_4_1':dict(k=4,n=1,p=1,bank='300',n_app=128,read_apps=(16,32,64,128))}
def cell_world(seed,name,enc,bg,**kw):
    """build_fact_world for a named cell (a cell may carry its own n_app / read_apps; the older cells use n_app = 16)"""
    return build_fact_world(seed,enc=enc,bg=bg,**{'n_app':16,**CELLS[name],**kw})
CELLS_BGONLY=('p1',)
def selftest():
    enc=StubTok();bg=_synth_bg(9000);rows_=[]
    for name in ('base','p1','p4','p9','k64','bank12','disjoint','exist8'):
        cfg=CELLS[name];w=build_fact_world(55,n_app=16,enc=enc,bg=bg,**cfg);v,A_=verify(w,enc);rows_.append(stats(w,A_))
        w2=build_fact_world(55,n_app=16,enc=enc,bg=bg,**cfg);v2,_=verify(w2,enc);assert v['sha_stream']==v2['sha_stream'],'not reproducible'
        w3=build_fact_world(56,n_app=16,enc=enc,bg=bg,**cfg);assert verify(w3,enc)[0]['sha_stream']!=v['sha_stream']
        if name=='p9':assert v['min_gap_between_appearances']>=WIN+1 and w['n_chunks']==144
    w=build_fact_world(55,n_app=16,enc=enc,bg=bg,**CELLS['base']);items=build_items(w);assert items==build_items(w) and len(rows(items))==NCAND*len(items) and len({i['qframe'] for i in items if i['kind']=='fact'})==3
    m=item_meta(items);assert m['gi']==m['gold_idx'] and len(m['cand_group'])==36 and all(len(x)==16 for x in m['cand_same'])
    # control items: never-seen name, fake gold is an exposed place
    ct=[i for i in items if i['kind']=='ctrl'];assert len(ct)==4 and all(i['group']==-1 for i in ct)
    # seeds redraw the name / place assignment
    w2=build_fact_world(56,n_app=16,enc=enc,bg=bg,**CELLS['base']);assert [f['P'] for f in w['facts']]!=[f['P'] for f in w2['facts']]
    print(fmt_stats(rows_));print('gen_facts SELFTEST OK (stub tokenizer, synthetic background)')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--stats',default=None,help='comma list of cells (%s) or all'%','.join(CELLS));ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--pools',action='store_true');ap.add_argument('--seed',type=int,default=55);ap.add_argument('--stub',action='store_true');ap.add_argument('--out',default=None);ap.add_argument('--check-repro',action='store_true');ap.add_argument('--sample',type=int,default=0,help='print this many generated sentences of the base world');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    if a.stub:enc=StubTok();bg=_synth_bg(9000)
    else:
        tf=a.tokenizer_file or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=load_enc(tf);bg=None
    if a.pools:print(json.dumps(G0.pool_report(enc)));sys.exit(0)
    bg=bg or default_bg_all()
    if a.sample:
        w=build_fact_world(a.seed,n_app=16,enc=enc,bg=bg,**CELLS['base']);out=[]
        for ch in w['chunks'][:3]:
            for x in ch['sents']:out.append(x['text'])
        print('\n'.join(f'{i+1} {t}' for i,t in enumerate(out[:a.sample])));sys.exit(0)
    names=list(CELLS) if a.stats in (None,'all') else a.stats.split(',');rows_=[];reps={}
    for nm in names:
        w=cell_world(a.seed,nm,enc,bg);v,A_=verify(w,enc);rows_.append(stats(w,A_));reps[nm]=v
        if a.check_repro:
            w2=cell_world(a.seed,nm,enc,bg);v2,_=verify(w2,enc);reps[nm]['repro_same_sha']=(v['sha_stream']==v2['sha_stream'])
    md=fmt_stats(rows_)+'\n\n'+'\n'.join(f'- {k}: {json.dumps(v)}' for k,v in reps.items());print(md)
    if a.out:Path(a.out).write_text(md+'\n')
