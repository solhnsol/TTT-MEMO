"""Stage 2 v2 (dose-response design): synthetic narrative worlds, CPU only.
A 'triple' = person X (occupation o) from place P. 15 swap-pairs (X1,X2) -> 30 triples, unique occupations, all names the same
token length. EVERY triple receives one exposure per ROUND (frames never repeat per triple); a pair's two exposures are
adjacent in the text every round, so frequency, tokenization and recency are matched inside each swap pair.
Snapshots (probe points) follow rounds 1, 4 and 16, each separated from the last exposure by >8192 tokens of filler.
World B = world A with the two person names of every pair swapped everywhere (bindings flip, everything else identical).
Background prose is real narrative text with capitalised names removed."""
import random,re,json,sys,os
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[2]
TOK=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
OCC=['ferryman','apothecary','cartographer','blacksmith','weaver','beekeeper','lamplighter','tanner','cooper','falconer','chandler','glassblower','shepherd','mason','cobbler','miller','potter','saddler','brewer','thatcher','fishwife','locksmith','clockmaker','dyer','sexton','tinker','carter','fuller','draper','turner']
SYL1=['Mar','Tol','Vel','Cas','Bren','Dor','Eld','Fen','Gal','Hol','Ister','Jor','Kel','Lun','Mor','Nev','Orl','Pen','Quin','Ras','Sil','Tav','Ulm','Var','Wen','Yar','Zel','Ash','Bel','Cor']
SYL2=['ek','ane','imor','wick','dara','ouin','eth','alo','ric','ussa','ine','orn','ade','ish','olm','avel','ena','ard','ova','ume']
PL_SUF=['moor','haven','ford','stead','wick','mere','dale','holm','crest','ness','by','combe','thorpe','vale','garth']
PL_PRE=['Ash','Bram','Cal','Dun','Elm','Fair','Glen','Harn','Ive','Kirk','Lang','Mal','Nor','Oak','Pell','Rook','Stan','Thorn','Ulf','Wex']
def person_names(rng,n):
    names=set()
    while len(names)<n:names.add(rng.choice(SYL1)+rng.choice(SYL2))
    return sorted(names)
def place_names(rng,n):
    names=set()
    while len(names)<n:names.add(rng.choice(PL_PRE)+rng.choice(PL_SUF))
    return sorted(names)
def ntok(s):return len(TOK.encode(' '+s,add_special_tokens=False).ids)
# exposure frames: {X} person, {P} place, {o} occupation (article handled)
A=lambda o:('an ' if o[0] in 'aeiou' else 'a ')+o
FR=[
 "{X}, the {o} of {P}, had never once left the valley.","Everyone in {P} knew {X} as the village {o}.","It was {X} the {o} who first noticed the smoke above {P}.",
 "In {P} the {o} was a quiet fellow named {X}.","When the rain came to {P}, {X} the {o} closed the shutters early.","{X} worked as {a_o} in {P}, as his father had before him.",
 "No one in {P} could match {X}, the local {o}, for patience.","The people of {P} still spoke of {X}, their {o}, with a certain fondness.","{P} had only one {o}, and his name was {X}.",
 "Years later, {X} would remember the {o}'s shop in {P} more clearly than his own face.","The traveller asked after the {o} of {P} and was told to look for {X}.",
 "{X} left {P} at dawn, leaving the {o}'s tools in the cold.","She had been told that {X}, {a_o} from {P}, could mend almost anything.","A letter addressed to {X}, {o}, of {P}, lay unopened on the sill.",
 "Between the market and the mill in {P}, {X} kept his {o}'s bench.","They say that {X} of {P}, the {o}, once refused a bishop's coin.","The {o} {X} came down from {P} on the morning cart.",
 "{X} was {a_o}, and {P} was the only home he had ever known.","At the inn the talk turned to {X}, the {o} from {P}, and his odd habits.","Neither the rector nor the {o}, {X}, would speak of what happened at {P}.",
 "Of all the households in {P}, the {o}'s, {X}'s, was the last to light a fire.","Little {X} grew up in {P} to become its {o}.","{P} remembered {X} chiefly as {a_o} with a long memory for debts.",
 "On market day {X}, {o} of {P}, set out his wares before anyone else.","'Ask {X},' they said, 'the {o} in {P}; he will know.'","The {o} of {P}, {X}, wiped his hands and said nothing.",
 "A cold wind followed {X} the {o} all the way from {P}.","As {a_o} from {P}, {X} was used to being sent for at odd hours.","It was said in {P} that the {o}, {X}, slept little.","{X}'s cousin lived far from {P}, but {X} the {o} stayed."]
# held-out probe frames (never used in the text): prefix + gold continuation
PR_OCC="Asked who the {o} of the valley was, the old woman answered that it was"  # occ -> person name
PR_HOME="Nobody could say exactly where {X} had grown up, though most guessed the town of"  # name -> place
PR_REV="The elders of {P} always mentioned one man above the rest; his name was"  # place -> person name
PR_FREQ_P="Late that evening a stranger arrived and gave his name as"
PR_FREQ_L="After three days on the road the party finally reached the town of"
BACK_DROP=re.compile(r"\b[A-Z][a-z]{2,}\b")
def background(rng):
    files=[f for f in sorted((ROOT/'data/narrative_gutenberg').glob('pg*.txt')) if f.name in ('pg98.txt','pg1260.txt','pg768.txt','pg766.txt')]
    f=rng.choice(files);t=f.read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t
    t=t[len(t)//5:];t=re.sub(r'\r','',t)
    paras=[p for p in re.split(r'\n\s*\n',t) if 200<len(p)<900 and not re.search(r'CHAPTER|\bVol|\[|\]|\*',p)]
    # drop real character names: replace capitalised non-initial words by lower-case-free generic nouns
    out=[]
    for p in paras:
        p=re.sub(r'(?<=[a-z,;] )(?:[A-Z][a-z]+)(?: [A-Z][a-z]+)*','someone',p)
        p=re.sub(r"\b(?:Mr|Mrs|Miss|Dr|Sir|Lady|Lord)\.? \w+",'the stranger',p)
        out.append(' '.join(p.split()))
    rng.shuffle(out);return out
ROUNDS={14:1,33:4,62:16}   # probe chunk index -> cumulative exposure rounds seen by every entity
NCH=63
BLOCKS_V1=[(1,1,0,3),(2,4,15,22),(5,16,34,52)]  # v1 (seeds 1-3): later-round exposures could land inside the probe window (found by the integrity check)
BLOCKS=[(1,1,0,3),(2,4,17,22),(5,16,37,52)]     # v2 (clean seeds): >2K tokens of margin between each probe chunk start and the next round
def build_world(seed,pairs=15,v=2,frames=None):
    """frames: 'shared' (default, the 30 shared frames FR, bit-identical to the earlier code), 'pair' (E48 W1: every pair owns R disjoint frames) or 'sentence' (E48 W2: every exposure sentence has its own frame); default from the environment variable E48_FRAMES. Only the sentence templates change: the rng stream (names, places, occupations, schedule, background order) is the same in all modes."""
    frames=frames or os.environ.get('E48_FRAMES','shared')
    rng=random.Random(seed);npairs=pairs;nt=npairs*2
    pn=person_names(rng,nt*4);pl=place_names(rng,nt*4)
    byl={}
    for n in pn:byl.setdefault(ntok(n),[]).append(n)
    L=max(byl,key=lambda k:len(byl[k]));persons=byl[L]
    byp={}
    for n in pl:byp.setdefault(ntok(n),[]).append(n)
    Lp=max(byp,key=lambda k:len(byp[k]));places=byp[Lp]
    assert len(persons)>=nt+8 and len(places)>=nt+8 and nt<=len(OCC),(len(persons),len(places))
    rng.shuffle(persons);rng.shuffle(places);occs=OCC[:];rng.shuffle(occs)
    triples=[{'id':i,'pair':f'p{i//2}','X':persons[i],'P':places[i],'o':occs[i]} for i in range(nt)]
    ctrl_persons=persons[nt:nt+8];ctrl_places=places[nt:nt+8]
    fr={t['id']:rng.sample(FR,len(FR)) for t in triples}   # always drawn (keeps the rng stream identical in every frames mode)
    bank=None
    if frames!='shared':
        import frame_bank as FB
        R=max(b[1] for b in (BLOCKS if v>=2 else BLOCKS_V1));bank,fr,_=FB.assign(frames,seed,triples,npairs,R,FR)
    # events: (round, pair) -> two adjacent sentences, order random per event
    bg=background(rng);texts=[];events=[]  # texts: list of dict(kind,text,...)
    chunk_tokens=1024
    # token budget schedule: spread each round's events through its window
    sched=[]  # (target_token_position, round, pair)
    for (r0,r1,c0,c1) in (BLOCKS if v>=2 else BLOCKS_V1):
        nr=r1-r0+1;span=(c1-c0+1)*chunk_tokens;w=span/nr
        for r in range(r0,r1+1):
            lo=c0*chunk_tokens+(r-r0)*w
            for p in range(npairs):sched.append((lo+rng.random()*w*0.9,r,p))
    sched.sort();si=0;ntoks=0;bi=0
    while ntoks<NCH*chunk_tokens+3000:
        if si<len(sched) and ntoks>=sched[si][0]:
            _,r,p=sched[si];si+=1;ts=[triples[2*p],triples[2*p+1]]
            if rng.random()<0.5:ts=ts[::-1]
            for t in ts:
                f=fr[t['id']][(r-1)%len(fr[t['id']])]
                texts.append({'kind':'expo','t':t['id'],'round':r,'text':f.format(X='@X%d@'%t['id'],P=t['P'],o=t['o'],a_o=A(t['o']))});ntoks+=len(TOK.encode(texts[-1]['text'].replace('@X','Xx').replace('@',''),add_special_tokens=False).ids)+1
        else:
            txt=bg[bi%len(bg)];bi+=1;texts.append({'kind':'bg','text':txt});ntoks+=len(TOK.encode(txt,add_special_tokens=False).ids)+1
    w={'seed':seed,'v':v,'triples':triples,'ctrl_persons':ctrl_persons,'ctrl_places':ctrl_places,'texts':texts,'name_len':L,'place_len':Lp,'pairs':npairs}
    if bank is not None:w['frames_mode']=frames;w['frames']=bank   # E48: the bank of the frames of this world (s3_sets.sentence_table identifies the frame of a sentence in it)
    return w
def realize(world,swap):
    """full text with names substituted + char spans of exposure sentences; swap=True is the counterfactual world B."""
    pairs={}
    for t in world['triples']:pairs.setdefault(t['pair'],[]).append(t['id'])
    name={t['id']:t['X'] for t in world['triples']}
    if swap:
        for ids in pairs.values():a,b=ids;name[a],name[b]=name[b],name[a]
    out=[];pos=0;spans=[]
    for t in world['texts']:
        txt=re.sub(r'@X(\d+)@',lambda m:name[int(m.group(1))],t['text'])
        if t['kind']=='expo':spans.append({'t':t['t'],'round':t['round'],'start':pos,'end':pos+len(txt),'text':txt})
        out.append(txt);pos+=len(txt)+2
    return '\n\n'.join(out),name,spans
def probes(world,name):
    """probe items for a realized world. gold/foil depend on the world (bindings flip in B)."""
    P=[];byp={}
    for t in world['triples']:byp.setdefault(t['pair'],[]).append(t)
    for pair,(t1,t2) in byp.items():
        for a,b in ((t1,t2),(t2,t1)):
            P.append({'level':'L2','pair':pair,'tid':a['id'],'prefix':PR_OCC.format(o=a['o']),'gold':' '+name[a['id']],'foil':' '+name[b['id']]})
            P.append({'level':'L3','pair':pair,'tid':a['id'],'prefix':PR_HOME.format(X=name[a['id']]),'gold':' '+a['P'],'foil':' '+b['P']})
            P.append({'level':'L3r','pair':pair,'tid':a['id'],'prefix':PR_REV.format(P=a['P']),'gold':' '+name[a['id']],'foil':' '+name[b['id']]})
    for t in world['triples']:
        k=t['id']%len(world['ctrl_persons'])
        P.append({'level':'L1p','pair':t['pair'],'tid':t['id'],'prefix':PR_FREQ_P,'gold':' '+name[t['id']],'foil':' '+world['ctrl_persons'][k]})
        P.append({'level':'L1l','pair':t['pair'],'tid':t['id'],'prefix':PR_FREQ_L,'gold':' '+t['P'],'foil':' '+world['ctrl_places'][k]})
    return P
if __name__=='__main__':
    w=build_world(int(sys.argv[1]) if len(sys.argv)>1 else 1)
    ta,na,sa=realize(w,False);tb,nb,sb=realize(w,True)
    ia=TOK.encode(ta,add_special_tokens=False);ib=TOK.encode(tb,add_special_tokens=False)
    print(len(ia.ids),len(ib.ids),len(sa),len(probes(w,na)),w['name_len'],w['place_len'])
    print(sa[0]['text'],'|',sb[0]['text']);print(ta[:300])
