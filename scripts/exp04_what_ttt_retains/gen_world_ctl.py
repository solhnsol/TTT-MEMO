"""E55 controlled synthetic world (CPU only, no GPU; needs a tokenizer only through the `enc` callable). Design: /mnt/project-files/experiments/E55-controlled-world-binding-design.md (sections 2.1, 2.2, 2.4).
A world has N = M x p swap pairs (two persons X1, X2 with places P1, P2, a never-repeated random occupation). Pairs are split into p groups (pair i -> group i % p); chunk c (1024 tokens exactly) carries the sentences of group c % p only, so EVERY chunk has exactly M pairs
and 2 M s sentences (s sentences per person, all with different frames while the frame bank is large enough). A pair appears n_app times (chunks g, g+p, ...), the stream has n_app x p chunks and is read ONCE (no passes, no background reuse).
The two persons of a pair stand in one contiguous block (2 s sentences in random order); blocks sit at random positions between background paragraphs; no sentence touches a chunk edge (asserted: for p = 9 every earlier sentence of the same pair is >= 8193 tokens away from the next appearance).
Frame banks: '50' = the 50 frames of E55-frame-samples.md (pinned below; the draft could not be regenerated from seed 55 with frame_bank.make, see FRAMES50_NOTE), '12' = 12 of them (seeded), 'disjoint' = pair-owned frames (no frame shared between pairs, frame_bank 'sentence' generator).
Decoy world = the same generator, different seed, persons / places disjoint from the real world, same M, s, p, bank, same chunk length; the background paragraphs are the second half of the pool (real world: first half).
World B = world A with the two names of every pair swapped everywhere (same token lengths, so the layout is identical). Questions: 3 frames x names, 16 candidate places (gold, partner, 14 random other world places), 4 unexposed control pairs.
Everything is deterministic in (seed, M, s, p, n_app, bank). The module imports no tokenizer: `enc(text) -> list of token ids` is passed in (server: tokenizers.Tokenizer.from_file(data/tokenizer/llama3/tokenizer.json), see load_enc)."""
import random,re,json,sys,hashlib,itertools,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
CH=1024;WIN=8192;NQ=3;NCAND=16;NCTRL_PAIRS=4;READ_APPS=(2,4,8,16)
FRAMES50_NOTE='FRAMES50 = the 50 frames of E55-frame-samples.md (design thread draft, seed 55 of frame_bank); the exact draw could not be reproduced with frame_bank.make("sentence", ...) for any (seed in 55/E55/E48, npairs, R) tried, so the list is pinned verbatim and checked by sha256 (first 12 hex) = FRAMES50_SHA'
FRAMES50=[
 "In {P}, {X} the {o} wrapped the patched hook in the loft.",
 "Soon after, folk in {P} called the {o} {X}, and he wrapped the cracked rope near the mill.",
 "Beside the {o}'s bench in {P}, {X} dried the copper funnel.",
 "Some evenings, {X}, {P}'s {o}, sorted the worn chisel late at night.",
 "The letter said that down in {P}, the {o} called {X} fetched the dented basket.",
 "{P} had one {o}, {X}, who filled the rough plank.",
 "Records show that from {P} came {X}, {a_o}, who guarded the mossy basket.",
 "In {P}, {X} the {o} bundled the hollow chest.",
 "Long ago, {X}, {P}'s {o}, rolled the worn trowel at harvest.",
 "After the harvest, {P} had one {o}, {X}, who wrapped the rusty cask on Sundays.",
 "{X}, once {a_o} in {P}, sorted the polished satchel before the frost.",
 "{X}, {a_o} from {P}, tested the cracked wedge after the flood.",
 "Everyone in {P} knew {X} the {o}, who tested the dented barrel before the wedding.",
 "The {o} of {P} named {X} bundled the creaking lantern.",
 "{X} the {o}, who lived in {P}, rolled the silvered broom late at night.",
 "{X}, who was {a_o} in {P}, tied the copper barrel.",
 "{X}, the {o}, a native of {P}, emptied the narrow funnel over the hill.",
 "Visitors to {P} met {X}, the {o}, who buried the rough anvil with patient hands.",
 "Under the old king, {X}, {a_o} from {P}, loaded the dusty bucket.",
 "The {o} of {P}, {X}, lent the tattered crate.",
 "{X}, {P}'s {o}, hauled the ragged trunk behind the church.",
 "Everyone agreed that {X} the {o}, who lived in {P}, lent the worn handle.",
 "{X} kept the {o}'s trade in {P} and tied the stained chisel.",
 "Some evenings, {X} was the {o} at {P}, and stored the battered spool in the long dusk.",
 "Beside the {o}'s bench in {P}, {X} tied the wooden spool.",
 "It was noted that {X} the {o}, who lived in {P}, sorted the old bellows.",
 "As a boy, in {P}, {X} the {o} hammered the rusty chisel.",
 "{X}, the {o} of {P}, scrubbed the stained rake all afternoon.",
 "{X}, the only {o} in {P}, dragged the hollow pail with patient hands.",
 "That winter, the {o}, {X}, of {P} mended the striped hook.",
 "The {o} {X}, late of {P}, hauled the old hook.",
 "The {o} {X}, late of {P}, dried the ragged bench.",
 "Everyone agreed that in {P}, {X} the {o} ordered the leather kettle by the river.",
 "In the wet year, {P} had one {o}, {X}, who repaired the patched shelf near the mill.",
 "At {P}, the {o}, {X}, tied the rough cask at dusk.",
 "The ledger showed that {X}, once {a_o} in {P}, guarded the rough jug.",
 "In those years, near {P}, {X} the {o} measured the striped hammer past the bridge.",
 "{X}, the only {o} in {P}, traded the bent handle.",
 "As rumour had it, when in {P}, {X} the {o} measured the tattered kettle with great care.",
 "Of an evening, {X} of {P}, the village {o}, filled the tattered harness on Sundays.",
 "Soon after, {X}, who was {a_o} in {P}, emptied the battered bucket.",
 "The {o} of {P}, {X}, packed the ragged bellows past the bridge.",
 "From {P} came {X}, {a_o}, who stacked the crooked pail.",
 "Through hard times, {P}'s {o}, {X}, hammered the twisted trunk.",
 "Some say that {X}, the {o}, a native of {P}, sealed the sturdy hook on Sundays.",
 "That year, {X} was the {o} at {P}, and buried the dented saw before the fair.",
 "Last autumn, {X} kept the {o}'s trade in {P} and polished the dented funnel.",
 "Now and then, folk in {P} called the {o} {X}, and he buried the lopsided shelf after the flood.",
 "For decades, down in {P}, the {o} called {X} trimmed the damp jug.",
 "All that summer, from {P} came {X}, {a_o}, who dried the brass ladder in silence.",
]
FRAMES50_SHA='832418efc944'
QFRAMES=['Nobody could say exactly where {X} had grown up, though most guessed the town of','If you asked anyone which place {X} called home, they would name the village of','The registry listed the birthplace of {X} as the settlement of']
OCC=['ferryman','apothecary','cartographer','blacksmith','weaver','beekeeper','lamplighter','tanner','cooper','falconer','chandler','glassblower','shepherd','mason','cobbler','miller','potter','saddler','brewer','thatcher','fishwife','locksmith','clockmaker','dyer','sexton','tinker','carter','fuller','draper','turner']
SYL1=['Mar','Tol','Vel','Cas','Bren','Dor','Eld','Fen','Gal','Hol','Ister','Jor','Kel','Lun','Mor','Nev','Orl','Pen','Quin','Ras','Sil','Tav','Ulm','Var','Wen','Yar','Zel','Ash','Bel','Cor']
SYL1X=SYL1+['Bar','Dar','Esk','Fyn','Gor','Hal','Ing','Jas','Kor','Lor','Mel','Nor','Osk','Per','Rin','Sev','Tor','Urs','Vor','Wyl','Xan','Yor','Zan','Aln','Brun','Cer','Dev','Eri','Fol','Gwen']
SYL2=['ek','ane','imor','wick','dara','ouin','eth','alo','ric','ussa','ine','orn','ade','ish','olm','avel','ena','ard','ova','ume']
SYL2X=SYL2+['an','el','or','ik','um','ar','on','ise','ald','ewn']
PL_SUF=['moor','haven','ford','stead','wick','mere','dale','holm','crest','ness','by','combe','thorpe','vale','garth']
PL_SUFX=PL_SUF+['field','wood','bridge','mouth','hurst','ton','worth','sted','well','burn']
PL_PRE=['Ash','Bram','Cal','Dun','Elm','Fair','Glen','Harn','Ive','Kirk','Lang','Mal','Nor','Oak','Pell','Rook','Stan','Thorn','Ulf','Wex']
PL_PREX=PL_PRE+['Alder','Birch','Cliff','Dove','Eden','Fox','Grey','Hay','Iron','Jay','Kings','Lime','Marsh','North','Over','Pine','Quarry','Red','Salt','Tarn']
A=lambda o:('an ' if o[0] in 'aeiou' else 'a ')+o
def _h(*a):return hashlib.sha256('|'.join(map(str,a)).encode()).hexdigest()
def sha12(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest()[:12]
def load_enc(tokenizer_file):
    """server: enc(text) -> ids with the llama3 tokenizer (no special tokens)"""
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(str(tokenizer_file));return lambda s:tk.encode(s,add_special_tokens=False).ids
class StubTok:
    """CPU tests without the tokenizers package: words and punctuation are tokens (a long word is one token, names are one token)"""
    def __call__(s,text):return [zlib.crc32(w.encode())%50000+1 for w in re.findall(r"\w+|[^\w\s]|\n\n",text.replace('\n\n',' \n\n '))] if text else []
def pool_report(enc):
    """token-length classes of the person / place pools (server check that the modal class is large enough: needs 4N + 8 persons and places for the largest N = 72)"""
    nt=lambda s:len(enc(' '+s));out={}
    for nm,pool in (('persons',[a+b for a in SYL1X for b in SYL2X]),('places',[a+b for a in PL_PREX for b in PL_SUFX])):
        c={}
        for n in pool:c[nt(n)]=c.get(nt(n),0)+1
        out[nm]=dict(sorted(c.items()))
    return out
def _contexts(bank):
    """distinct local contexts in which a person's name appears in the bank frames (<=3 words before, <=3 after; a frame that starts with the name gives a context with an empty left side)"""
    import re as _re
    out=set()
    for f in bank:
        if '{X}' not in f:continue
        pre,post=f.split('{X}',1)[0],f.split('{X}',1)[1]
        pre=' '.join(pre.split(' ')[-4:]) if pre and pre.strip() else pre;post=' '.join(post.split(' ')[:4])
        out.add(_re.sub(r'\{\w+\}','Aldmere',pre)+'{N}'+_re.sub(r'\{\w+\}','Aldmere',post))
    return sorted(out)
def _alloc(seed,N,enc,decoy=True,bank=None):
    """names / places with one token length in the modal class (counted with a leading space as mid-sentence). Persons are additionally PAIRED (names[2j], names[2j+1]) so that both names of a pair give the same token length in every local context of the frame bank (a name at the very start of a sentence has no leading space and can tokenise differently): world B swaps the two names of a pair and must keep every sentence length."""
    nt=lambda s:len(enc(' '+s));need=2*N+2*NCTRL_PAIRS+(2*N if decoy else 0);res={}
    for what,pool in (('persons',[a+b for a in SYL1X for b in SYL2X]),('places',[a+b for a in PL_PREX for b in PL_SUFX])):
        byl={}
        for n in sorted(set(pool)):byl.setdefault(nt(n),[]).append(n)
        L=max(byl,key=lambda k:len(byl[k]));cls=byl[L][:];assert len(cls)>=need,(what,'token length',L,'class size',len(cls),'need',need,{k:len(v) for k,v in sorted(byl.items())},'extend the E55 pools')
        if what=='persons' and bank is not None:
            cx=_contexts(bank);sig={n:tuple(len(enc(m.replace('{N}',n))) for m in cx) for n in cls};g={}
            for n in cls:g.setdefault(sig[n],[]).append(n)
            r=random.Random(f'E55|{seed}|pairs');pairs=[]
            for k in sorted(g):
                v=g[k][:];r.shuffle(v);pairs+=[(v[i],v[i+1]) for i in range(0,len(v)-1,2)]
            assert len(pairs)>=need//2,('persons: not enough sentence-length-compatible name pairs',len(pairs),'need',need//2,'signature classes',sorted(len(v) for v in g.values()),'contexts',len(cx))
            r.shuffle(pairs);cls=[x for pr in pairs[:need//2] for x in pr]
        else:random.Random(f'E55|{seed}|{what}').shuffle(cls)
        res[what]=(cls[:need],L)
    return res
FRAMES300_SHA='7178214a6087'      # experiments/E55-frame-bank300.md (design thread draft, 300 frames, <= 2 of 6 shared components); the file frames300.txt is that list, one frame per line
def frame_bank(kind,seed,N,need):
    """-> (bank list, owner function pair -> list of frame indices or None for shared banks). need = n_app x s frames per person."""
    if kind=='50':assert sha12('\n'.join(FRAMES50))==FRAMES50_SHA;return FRAMES50[:],None
    if kind=='300':
        fr=[l for l in (Path(__file__).resolve().parent/'frames300.txt').read_text().split('\n') if l];assert len(fr)==300 and sha12('\n'.join(fr))==FRAMES300_SHA,'frames300.txt differs from E55-frame-bank300.md';return fr,None
    if kind=='shared64':          # E55 design 19: the first 64 frames of the 300-bank, shared by every fact (the assignment is a Latin rectangle, see gen_facts.latin_frames)
        return frame_bank('300',seed,N,need)[0][:64],None
    if kind=='shared16rep':        # E55 design 21.2: the first 16 frames, shared by every fact, each fact cycles through them 4 times (64 exposures)
        return frame_bank('300',seed,N,need)[0][:16],None
    if kind=='12':return random.Random(f'E55|bank12|{seed}').sample(FRAMES50,12),None
    if kind=='disjoint':
        import frame_bank as FB
        assert N%2==0;bank,info=FB.make('sentence',need,N//2,FRAMES50,f'E55|disjoint|{seed}');assert len(bank)==N*need and len(set(bank))==len(bank);return bank,lambda i:list(range(i*need,(i+1)*need))
    raise ValueError(kind)
def person_frames(kind,bank,owner,seed,pid,pair,need):
    r=random.Random(f'E55|{seed}|fr|{pid}')
    if owner is not None:idx=owner(pair);seq=idx[:];r.shuffle(seq);return seq
    seq=[]
    while len(seq)<need:
        perm=list(range(len(bank)));r.shuffle(perm);seq+=perm
    return seq[:need]
def _make_persons(names,places,seed,N,tag,bank_kind,bank,owner,s,n_app):
    r=random.Random(f'E55|{seed}|occ|{tag}');ps=[]
    for i in range(2*N):
        pr=i//2;ps.append({'id':i,'pair':pr,'X':names[i],'P':places[i],'o':r.choice(OCC),'group':None,'frames':person_frames(bank_kind,bank,owner,f'{seed}|{tag}',i,pr,n_app*s)})
    return ps
def _stream(persons,N,M,s,p,n_app,bank,enc,bg,seed,tag,sep):
    """-> list of chunk dicts {'k','group','pieces':[('bg',ids,text) | ('sent',text,pid,frame_idx,app,n_tokens) | ('tail',ids,text)]}. Uses the A names (token counts are identical in B)."""
    nch=n_app*p;bgi=0;chunks=[]
    for c in range(nch):
        g=c%p;app=c//p;rng=random.Random(f'E55|{seed}|chunk|{tag}|{c}')
        pairs=[i for i in range(N) if i%p==g];assert len(pairs)==M
        blocks=[];cost=0
        for pr in pairs:
            sents=[]
            for pid in (2*pr,2*pr+1):
                for j in range(s):
                    fi=persons[pid]['frames'][app*s+j];t=bank[fi].format(X=f"@X{pid}@",P=persons[pid]['P'],o=persons[pid]['o'],a_o=A(persons[pid]['o']))
                    n=len(enc(t.replace(f'@X{pid}@',persons[pid]['X'])));sents.append(('sent',t,pid,fi,app,n));cost+=n+len(sep)
            rng.shuffle(sents);blocks.append(sents)
        assert cost<CH-200,('sentences fill the chunk',cost)
        rem=CH-cost;whole=[];tail=None
        while True:
            assert bgi<len(bg),('background pool exhausted',tag,c)
            ids=enc(bg[bgi]);L=len(ids)+len(sep)
            if L<=rem:whole.append(('bg',ids+sep,bg[bgi]));rem-=L;bgi+=1
            else:
                if rem>0:tail=('tail',(ids+sep)[:rem],bg[bgi]);bgi+=1
                break
        # blocks go between whole paragraphs, never before the first one and never after the last one (so no sentence touches a chunk edge)
        nb=len(whole);assert nb>=2,'too few background paragraphs per chunk'
        slots=sorted(rng.randint(1,nb-1) for _ in blocks) if nb>=2 else [1]*len(blocks)
        seq=[];bi=0;order=list(range(len(blocks)));rng.shuffle(order)
        for pos in range(nb+1):
            while bi<len(slots) and slots[bi]==pos:seq+=blocks[order[bi]];bi+=1
            if pos<nb:seq.append(whole[pos])
        if tail:seq.append(tail)
        chunks.append({'k':c,'group':g,'app':app,'pieces':seq})
    return chunks,bgi
def build_ctl_world(seed,M,s,p,n_app=16,bank='50',enc=None,bg=None,decoy=True):
    """bg: list of background paragraphs (str, >= 2 x the paragraphs needed); enc: text -> ids. Returns the world dict (JSON-able except nothing is tokenized lazily: pieces carry token ids of background and token counts of sentences)."""
    assert enc is not None and bg is not None and M%2==0 and p>=1 and s>=1;N=M*p;need=n_app*s
    bk,owner=frame_bank(bank,seed,N,need)
    al=_alloc(seed,N,enc,decoy,bk);P_,Lp=al['persons'];L_,Ll=al['places'];sep=enc('\n\n')
    persons=_make_persons(P_[:2*N],L_[:2*N],seed,N,'real',bank,bk,owner,s,n_app)
    ctrl=[{'id':2*N+i,'pair':f'c{i//2}','X':P_[2*N+i],'P':L_[2*N+i],'o':random.Random(f'E55|{seed}|cocc|{i}').choice(OCC)} for i in range(2*NCTRL_PAIRS)]
    half=len(bg)//2;w={'seed':seed,'M':M,'s':s,'p':p,'n_app':n_app,'bank_kind':bank,'N':N,'n_chunks':n_app*p,'name_len':Lp,'place_len':Ll,'bank':bk,'persons':persons,'ctrl':ctrl,'read_chunks':[a*p-1 for a in READ_APPS if a<=n_app],'sep':sep}
    w['chunks'],used=_stream(persons,N,M,s,p,n_app,bk,enc,bg[:half],seed,'real',sep)
    if decoy:
        dseed=seed+100000;dper=_make_persons(P_[2*N+2*NCTRL_PAIRS:],L_[2*N+2*NCTRL_PAIRS:],dseed,N,'decoy',bank,bk,owner,s,n_app)
        w['decoy']={'seed':dseed,'persons':dper};w['decoy']['chunks'],_=_stream(dper,N,M,s,p,n_app,bk,enc,bg[half:],dseed,'decoy',sep)
    return w
def names_map(world,swap,decoy=False):
    ps=(world['decoy']['persons'] if decoy else world['persons']);nm={x['id']:x['X'] for x in ps}
    if swap:
        for i in range(0,len(ps),2):nm[i],nm[i+1]=nm[i+1],nm[i]
    return nm
def realize(world,swap,enc,decoy=False):
    """-> list of chunks {'ids' (exactly 1024), 'sents':[{pid,frame,app,start,end}]}; world B (swap) re-encodes only the sentences and asserts equal token counts."""
    nm=names_map(world,swap,decoy);sep=world['sep'];out=[]
    for ch in (world['decoy']['chunks'] if decoy else world['chunks']):
        ids=[];sents=[]
        for pc in ch['pieces']:
            if pc[0]=='sent':
                _,t,pid,fi,app,n=pc;x=enc(re.sub(r'@X(\d+)@',lambda m:nm[int(m.group(1))],t));assert len(x)==n,('token count differs between worlds A and B',t)
                sents.append({'pid':pid,'frame':fi,'app':app,'start':len(ids),'end':len(ids)+n});ids+=x+sep
            else:ids+=pc[1]
        assert len(ids)==CH,(ch['k'],len(ids));out.append({'k':ch['k'],'ids':ids,'sents':sents})
    return out
def stream_sha(real):return _h(*[ _h(*c['ids']) for c in real])[:16]
# ---------------------------------------------------------------- questions
def _slotless_ngrams(t,n=5):
    ws=re.findall(r"[a-z']+|\{\w+\}",t.lower());out=set()
    for i in range(len(ws)-n+1):
        g=ws[i:i+n]
        if not any(x.startswith('{') for x in g):out.add(tuple(g))
    return out
def question_overlap(bank_frames):
    qs=set().union(*[_slotless_ngrams(q) for q in QFRAMES]);fs=set().union(*[_slotless_ngrams(f) for f in bank_frames]);return len(qs&fs)
def build_items(world):
    """items keyed by NAME (one item per person of the real world + the 2 x 4 unexposed control persons): prefix, 16 candidates ' '+place (shuffled), gi / fi = candidate index of the place of the person who owns that name in world A and of the partner (in world B the bound target is fi, same convention as e49_stats). Control items: gold / foil = their own never-shown places, the other 14 are exposed world places."""
    ps=world['persons'];ct=world['ctrl'];allp=[x['P'] for x in ps];items=[]
    for grp,kind in ((ps,'exposed'),(ct,'ctrl')):
        base=grp[0]['id']
        for x in grp:
            partner=grp[(x['id']-base)^1];r=random.Random(f"E55|{world['seed']}|q|{x['id']}")
            others=[q for q in allp if q not in (x['P'],partner['P'])];r.shuffle(others)
            cand=[x['P'],partner['P']]+others[:NCAND-2];r.shuffle(cand);assert len(cand)==NCAND and len(set(cand))==NCAND,(len(cand),x)
            items.append({'name_id':x['id'],'name':x['X'],'kind':kind,'pair':f"p{x['pair']}" if kind=='exposed' else x['pair'],'qframe':x['id']%NQ,'prefix':QFRAMES[x['id']%NQ].format(X=x['X']),'cands':[' '+c for c in cand],'gi':cand.index(x['P']),'fi':cand.index(partner['P'])})
    return items
def rows(items):
    """item-major rows [(prefix, candidate text)] (row i x NCAND + c)"""
    return [(it['prefix'],c) for it in items for c in it['cands']]
# ---------------------------------------------------------------- verification and statistics
def verify(world,enc):
    M,s,p,n_app,N=world['M'],world['s'],world['p'],world['n_app'],world['N'];res={}
    A_=realize(world,False,enc);B_=realize(world,True,enc)
    for a,b in zip(A_,B_):assert [(x['start'],x['end']) for x in a['sents']]==[(x['start'],x['end']) for x in b['sents']] and len(a['ids'])==len(b['ids'])==CH
    res['chunk_len_ok']=True
    for ch in A_:
        c=ch['k'];g=c%p;assert len(ch['sents'])==2*M*s,(c,len(ch['sents']));prs={x['pid']//2 for x in ch['sents']};assert len(prs)==M and all(q%p==g for q in prs)
        for x in ch['sents']:assert x['app']==c//p and x['start']>0 and x['end']<CH-len(world['sep'])
    res['per_chunk_pairs']=M;res['per_chunk_sentences']=2*M*s
    cnt={}
    for ch in A_:
        for x in ch['sents']:cnt.setdefault(x['pid'],set()).add(ch['k'])
    assert all(len(v)==n_app for v in cnt.values()) and len(cnt)==2*N
    # distance between consecutive appearances of a pair (end of an earlier sentence to start of a later one)
    mn=None
    for pr in range(N):
        sp=[(ch['k']*CH+x['start'],ch['k']*CH+x['end'],ch['k']) for ch in A_ for x in ch['sents'] if x['pid']//2==pr]
        for a in range(n_app-1):
            e=[x for x in sp if x[2]==a*p+pr%p];nx=[x for x in sp if x[2]==(a+1)*p+pr%p]
            d=min(y[0]-x[1] for x in e for y in nx);mn=d if mn is None else min(mn,d)
    res['min_gap_between_appearances']=mn
    if p>=9:assert mn>=WIN+1,('appearance distance',mn)
    pf=[x['frames'] for x in world['persons']];res['frame_repeats_per_person']=max(len(f)-len(set(f)) for f in pf)
    if world['bank_kind']=='50':assert res['frame_repeats_per_person']==0
    res['question_ngram_overlap']=question_overlap(world['bank']);assert res['question_ngram_overlap']==0
    names={x['X'] for x in world['persons']}|{x['X'] for x in world['ctrl']};dn={x['X'] for x in world['decoy']['persons']} if 'decoy' in world else set()
    pl={x['P'] for x in world['persons']}|{x['P'] for x in world['ctrl']};dp={x['P'] for x in world['decoy']['persons']} if 'decoy' in world else set()
    assert not names&dn and not pl&dp;res['decoy_disjoint']=True
    if 'decoy' in world:
        D_=realize(world,False,enc,decoy=True);assert len(D_)==len(A_) and all(len(c['ids'])==CH for c in D_)
        bgreal=[pc[2] for ch in world['chunks'] for pc in ch['pieces'] if pc[0]!='sent'];bgdec=[pc[2] for ch in world['decoy']['chunks'] for pc in ch['pieces'] if pc[0]!='sent']
        res['bg_reuse_real']=len(bgreal)-len(set(bgreal));res['bg_reuse_decoy']=len(bgdec)-len(set(bgdec));res['bg_shared_real_decoy']=len(set(bgreal)&set(bgdec));assert res['bg_shared_real_decoy']==0 and res['bg_reuse_real']==0
    items=build_items(world);assert len(items)==2*N+2*NCTRL_PAIRS
    for it in items:assert it['cands'][it['gi']]==' '+(world['persons'][it['name_id']]['P'] if it['kind']=='exposed' else world['ctrl'][it['name_id']-2*N]['P'])
    res['n_items']=len(items);res['sha_stream_A']=stream_sha(A_);res['sha_stream_B']=stream_sha(B_);return res,A_
def stats(world,A_,enc):
    M,s,p,n_app,N=world['M'],world['s'],world['p'],world['n_app'],world['N']
    sent_tok=[sum(x['end']-x['start'] for x in ch['sents']) for ch in A_];same=[];prior=[]
    for ch in A_:
        k=ch['k']
        if k<8:continue
        win=[x for c2 in A_[k-8:k] for x in c2['sents']];prior.append(sum(1 for j in range(k-8,k) if j%p==k%p))
        for x in ch['sents']:
            o=[y for y in win if y['pid']//2!=x['pid']//2]
            if o:same.append(sum(1 for y in o if y['frame']==x['frame'])/len(o))
    return {'p':p,'M':M,'s':s,'N':N,'bank':world['bank_kind'],'n_chunks':world['n_chunks'],'sent_per_chunk':2*M*s,'sent_tokens_per_chunk':sum(sent_tok)/len(sent_tok),'sent_share':sum(sent_tok)/len(sent_tok)/CH,'prior_appearances_in_window':(sum(prior)/len(prior) if prior else 0.0),'same_frame_share_window':(sum(same)/len(same) if same else 0.0),'read_chunks':world['read_chunks']}
def fmt_stats(rows_):
    L=['| p | M | s | N pairs | bank | chunks | sentences / chunk | sentence tokens / chunk (share) | earlier appearances of the pair inside the 8-chunk window | share of window sentences of OTHER pairs with the same frame | read after chunk index (appearances 2,4,8,16) |','|'+'---|'*11]
    for r in rows_:L.append(f"| {r['p']} | {r['M']} | {r['s']} | {r['N']} | {r['bank']} | {r['n_chunks']} | {r['sent_per_chunk']} | {r['sent_tokens_per_chunk']:.0f} ({100*r['sent_share']:.1f}%) | {r['prior_appearances_in_window']:.2f} | {r['same_frame_share_window']:.3f} | {r['read_chunks']} |")
    return '\n'.join(L)
def default_bg(enc_unused=None,n=4000):
    """server: background paragraphs of the E35 sources (gen_world.background; real narrative text, capitalised names removed)"""
    sys.path.insert(0,str(HERE));import gen_world as gw
    return gw.background(random.Random(5500))[:n]
CELLS={'base':dict(M=8,s=1,p=2,bank='50'),'p1':dict(M=8,s=1,p=1,bank='50'),'p4':dict(M=8,s=1,p=4,bank='50'),'p9':dict(M=8,s=1,p=9,bank='50'),'M4':dict(M=4,s=1,p=2,bank='50'),'M16':dict(M=16,s=1,p=2,bank='50'),'s2M4':dict(M=4,s=2,p=2,bank='50'),'bank12':dict(M=8,s=1,p=2,bank='12'),'disjoint':dict(M=8,s=1,p=2,bank='disjoint')}
def _synth_bg(n,seed=0):
    r=random.Random(seed);vocab=[''.join(r.choice('abcdefghijklmnop') for _ in range(r.randint(3,8))) for _ in range(900)];out=[]
    for i in range(n):out.append(' '.join(r.choice(vocab) for _ in range(r.randint(40,120)))+f' item{i}.')
    return out
def selftest():
    enc=StubTok();bg=_synth_bg(6000);rows_=[]
    for name in ('base','p1','p9','M4','s2M4','bank12','disjoint'):
        cfg=CELLS[name];w=build_ctl_world(55,n_app=16,enc=enc,bg=bg,**cfg);v,A_=verify(w,enc);rows_.append(stats(w,A_,enc))
        w2=build_ctl_world(55,n_app=16,enc=enc,bg=bg,**cfg);v2,_=verify(w2,enc);assert v['sha_stream_A']==v2['sha_stream_A'] and v['sha_stream_B']==v2['sha_stream_B'],'not reproducible'
        assert v['sha_stream_A']!=v['sha_stream_B']
        if name=='p9':assert v['min_gap_between_appearances']>=WIN+1 and w['n_chunks']==144
        if name=='bank12':assert v['frame_repeats_per_person']>0
        else:assert v['frame_repeats_per_person']==0,(name,v)
    items=build_items(w);assert len(items)==2*w['N']+8 and all(it['cands'][it['gi']]!=it['cands'][it['fi']] for it in items)
    w=build_ctl_world(55,n_app=16,enc=enc,bg=bg,**CELLS['base']);items=build_items(w);assert items==build_items(w);ex=[i for i in items if i['kind']=='exposed'];assert len({i['qframe'] for i in ex})==3 and len(rows(items))==NCAND*len(items)
    print(fmt_stats(rows_));print('gen_world_ctl SELFTEST OK (stub tokenizer, synthetic background)')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--stats',default=None,help='comma list of cells (%s) or all'%','.join(CELLS));ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--pools',action='store_true');ap.add_argument('--seed',type=int,default=55);ap.add_argument('--stub',action='store_true');ap.add_argument('--out',default=None);ap.add_argument('--check-repro',action='store_true');ap.add_argument('--sample',type=int,default=0,help='print this many generated sentences of the base world (world A)');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    if a.stub:enc=StubTok();bg=_synth_bg(6000)
    else:
        tf=a.tokenizer_file or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=load_enc(tf);bg=None
    if a.pools:print(json.dumps(pool_report(enc)));sys.exit(0)
    bg=bg or default_bg()
    if a.sample:
        w=build_ctl_world(a.seed,n_app=16,enc=enc,bg=bg,**CELLS['base']);nm=names_map(w,False);out=[]
        for ch in w['chunks'][:4]:
            for pc in ch['pieces']:
                if pc[0]=='sent':out.append(re.sub(r'@X(\d+)@',lambda m:nm[int(m.group(1))],pc[1]))
        print('\n'.join(f'{i+1} {t}' for i,t in enumerate(out[:a.sample])));sys.exit(0)
    names=list(CELLS) if a.stats in (None,'all') else a.stats.split(',');rows_=[];reps={}
    for nm in names:
        w=build_ctl_world(a.seed,n_app=16,enc=enc,bg=bg,**CELLS[nm]);v,A_=verify(w,enc);st=stats(w,A_,enc);rows_.append(st);reps[nm]=v
        if a.check_repro:
            w2=build_ctl_world(a.seed,n_app=16,enc=enc,bg=bg,**CELLS[nm]);v2,_=verify(w2,enc);reps[nm]['repro_same_sha']=(v['sha_stream_A']==v2['sha_stream_A'] and v['sha_stream_B']==v2['sha_stream_B'])
    md=fmt_stats(rows_)+'\n\n'+'\n'.join(f'- {k}: {json.dumps(v)}' for k,v in reps.items());print(md)
    if a.out:Path(a.out).write_text(md+'\n')
