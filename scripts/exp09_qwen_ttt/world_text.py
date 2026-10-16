"""Exp09: model-agnostic texts of the Exp07 write ladder (worlds, exposure sentences, probe items, readout contexts,
canary texts), re-implemented WITHOUT JAX so they can be tokenized with any `tokenizers.Tokenizer` (Qwen here).

Every string is produced by the same code path as the JAX pipeline:
  - worlds / realisation / probes / background: scripts/exp04_what_ttt_retains/gen_world.py, imported directly
    (pure Python + the llama3 tokenizer it uses for its own token budgets and name-length matching; that tokenizer is
    deliberately kept, so the WORLD STRINGS are identical to the JAX run's);
  - enc / item_ids / make_slot: copied from exp04 stage2.py (stage2 imports JAX);
  - neutral tail / pseudo names / item families: copied from exp05 readouts.py (Readouts.__init__, FAMILIARITY=True);
  - squad tail, pg11 canary text, sentences, l1_rows, fs_streams: copied from exp07 ladder.py.
test_texts_vs_jax.py checks, with the llama3 tokenizer, that every token list built here equals the JAX pipeline's
(and that the items equal the meta_L1_main_s{seed}.json files the JAX runs wrote), for seeds 101-103.
`bos`: token id prepended to every document-like sequence (stream, tails, canary), or None for no BOS."""
import sys,re,json,random,hashlib,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
import gen_world as gw          # pure python (tokenizers only)

R=48
FILLER_SENT="The road wound on through the fields and the long afternoon passed without any news of consequence to anyone."
TAIL_BOOK='pg1342.txt'
N_PSEUDO=16
JAX_CH=1024;JAX_NCH=gw.NCH;JAX_TAIL_LEN=8192-1024

def _pre(bos):return [] if bos is None else [bos]
def enc(tok,s):return tok.encode(s,add_special_tokens=False)
def item_ids(tok,slot_ids,prefix,cand):
    text="\n\n"+prefix+cand;e=enc(tok,text);a=len("\n\n"+prefix)
    pos=[i for i,(s,t) in enumerate(e.offsets) if t>a and s<len(text)]
    assert pos and tok.decode([e.ids[i] for i in pos]).strip()==cand.strip(),(prefix,cand,pos)
    ids=list(slot_ids)+e.ids;off=len(slot_ids)
    return ids,(off+pos[0],off+pos[-1]+1)
def make_slot(tok,keep_text):
    fill=enc(tok,(FILLER_SENT+' ')*6).ids
    if keep_text is None:return fill[:R]
    k=enc(tok,' '+keep_text).ids[:R];return fill[:R-len(k)]+k

def neutral_tail_text():
    f=ROOT/'data/narrative_gutenberg'/TAIL_BOOK;raw=f.read_bytes();t=raw.decode(errors='ignore')
    t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t;t=t[len(t)//3:].replace('\r','')
    paras=[p for p in re.split(r'\n\s*\n',t) if 200<len(p)<900 and not re.search(r'CHAPTER|\bVol|\[|\]|\*',p)]
    out=[]
    for p in paras:
        p=re.sub(r'(?<=[a-z,;] )(?:[A-Z][a-z]+)(?: [A-Z][a-z]+)*','someone',p)
        p=re.sub(r"\b(?:Mr|Mrs|Miss|Dr|Sir|Lady|Lord)\.? \w+",'the stranger',p)
        out.append(' '.join(p.split()))
    return '\n\n'.join(out),hashlib.sha256(raw).hexdigest()
def neutral_tail(tok,bos,length=JAX_TAIL_LEN):
    text,sha=neutral_tail_text();ids=_pre(bos)+enc(tok,text).ids
    assert len(ids)>=length,(len(ids),length)
    return ids[:length],text,sha

def squad_text(forbid):
    d=json.load(open(ROOT/'data/squad/dev-v1.1.json'));paras=[p['context'] for a in d['data'] for p in a['paragraphs']]
    paras=[' '.join(x.split()) for x in paras[:400]];paras=[x for x in paras if not any(nm in x for nm in forbid)]
    text='\n\n'.join(paras)
    for nm in forbid:assert nm not in text,nm
    return text
def squad_tail(tok,bos,forbid,length=JAX_TAIL_LEN):
    """ladder.squad_tail: SQuAD dev Wikipedia paragraphs, BOS + tokens, cut to `length`."""
    ids=_pre(bos)+enc(tok,squad_text(forbid)).ids;assert len(ids)>=length;return ids[:length]

def gen_text():
    t=(ROOT/'data/narrative_gutenberg/pg11.txt').read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t
    return ' '.join(t.replace('\r','').split())
def gen_ids(tok,bos,length=8*1024):
    ids=(_pre(bos)+enc(tok,gen_text()).ids)[:length];assert len(ids)==length;return ids

def pseudo_names(world,n,forbid_text):
    """readouts.pseudo_names (its tok argument is unused there; gw.ntok = llama3 token length, kept on purpose)."""
    rng=random.Random(world['seed']*1000+7)
    used={t['X'] for t in world['triples']}|{t['P'] for t in world['triples']}|set(world['ctrl_persons'])|set(world['ctrl_places'])
    def draw(gen,L):
        out=[]
        while len(out)<n:
            s=gen()
            if s in used or s in out or gw.ntok(s)!=L or re.search(r'\b'+s+r'\b',forbid_text):continue
            out.append(s)
        return out
    pp=draw(lambda:rng.choice(gw.SYL1)+rng.choice(gw.SYL2),world['name_len'])
    pl=draw(lambda:rng.choice(gw.PL_PRE)+rng.choice(gw.PL_SUF),world['place_len'])
    return pp,pl

def build_items(w,tail_text):
    """item families in the exact order of ladder.Ladder.fam: L3, XP, PS, FN, FP, FN0, FP0, L3r, L2."""
    _,name,_=gw.realize(w,False)   # canonical (world-A) names
    allp=gw.probes(w,name);l3=[it for it in allp if it['level']=='L3']
    streams=gw.realize(w,False)[0]+gw.realize(w,True)[0]
    for t in w['triples']:assert not re.search(r'\b'+t['X']+r'\b|\b'+t['P']+r'\b',tail_text),'world name inside neutral tail'
    tr={t['id']:t for t in w['triples']};npairs=w['pairs'];xp=[]
    for p in range(npairs):
        q=(p+1)%npairs
        for j in (0,1):
            a=2*p+j
            xp.append({'level':'XP','pair':f'x{p}','tid':a,'prefix':gw.PR_HOME.format(X=name[a]),'gold':' '+tr[2*q+j]['P'],'foil':' '+tr[2*q+1-j]['P']})
    pp,pl=pseudo_names(w,2*N_PSEUDO,streams+'\n'+tail_text)
    ps=[]
    for p in range(N_PSEUDO):
        for x,y in ((2*p,2*p+1),(2*p+1,2*p)):
            ps.append({'level':'PS','pair':f's{p}','tid':200+x,'prefix':gw.PR_HOME.format(X=pp[x]),'gold':' '+pl[x],'foil':' '+pl[y]})
    fam={'L3':l3,'XP':xp,'PS':ps}
    fam['FN']=[it for it in allp if it['level']=='L1p'];fam['FP']=[it for it in allp if it['level']=='L1l']
    cp,cl=w['ctrl_persons'],w['ctrl_places']
    fam['FN0']=[{'level':'FN0','pair':f's{x//2}','tid':300+x,'prefix':gw.PR_FREQ_P,'gold':' '+pp[x],'foil':' '+cp[x%len(cp)]} for x in range(2*N_PSEUDO)]
    fam['FP0']=[{'level':'FP0','pair':f's{x//2}','tid':400+x,'prefix':gw.PR_FREQ_L,'gold':' '+pl[x],'foil':' '+cl[x%len(cl)]} for x in range(2*N_PSEUDO)]
    fam['L3r']=[it for it in allp if it['level']=='L3r'];fam['L2']=[it for it in allp if it['level']=='L2']
    return fam,{'persons':pp,'places':pl}

class World:
    """all texts / token lists of one world seed for one tokenizer. tok: tokenizers.Tokenizer; bos: id or None."""
    def __init__(self,seed,tok,bos,tail_len=JAX_TAIL_LEN,gen_len=8*1024):
        self.seed=seed;self.tok=tok;self.bos=bos;self.w=w=gw.build_world(seed,v=2)
        self.tail_ids,self.tail_text,self.tail_sha=neutral_tail(tok,bos,tail_len)
        self.fam,self.pseudo=build_items(w,self.tail_text)
        self.items=[it for v in self.fam.values() for it in v]
        self.forbid=[t['X'] for t in w['triples']]+[t['P'] for t in w['triples']]+list(w['ctrl_persons'])+list(w['ctrl_places'])
        self.alt_ids=squad_tail(tok,bos,self.forbid,tail_len)
        self.gen_ids=gen_ids(tok,bos,gen_len)
    def probe_rows(self):
        """(ids, (a,b)) per (item, candidate) in item order, gold then foil; candidate tokens are ids[a:b]."""
        slot=make_slot(self.tok,None);out=[]
        for it in self.items:
            for c in (it['gold'],it['foil']):out.append(item_ids(self.tok,slot,it['prefix'],c))
        return out
    def text(self,swap):return gw.realize(self.w,swap)[0]
    def stream_ids(self,swap):
        """the full token stream of world A/B (BOS + the whole realised text; NOT truncated)."""
        return _pre(self.bos)+enc(self.tok,self.text(swap)).ids
    def sentences(self,swap):
        _,_,spans=gw.realize(self.w,swap);return [enc(self.tok,'\n\n'+s['text']).ids for s in spans]
    def l1_rows(self,swap,variant,fill_len=1024):
        """ladder.Ladder.l1_rows: rows (ids, (a,b)), loss positions a..b-1 (position i predicts token i+1)."""
        _,_,spans=gw.realize(self.w,swap);e=lambda t:enc(self.tok,t).ids;tri={t['id']:t for t in self.w['triples']}
        rows=[]
        if variant in ('plain','fill'):rows=[(x,(0,len(x)-1)) for x in (e('\n\n'+s_['text']) for s_ in spans)]
        if variant in ('ctx','irr'):
            by={}
            for s_ in spans:by.setdefault((tri[s_['t']]['pair'],s_['round']),[]).append(s_['text'])
            npairs=self.w['pairs']
            for s_ in spans:
                cur=e('\n\n'+s_['text']);r=s_['round'];pr=tri[s_['t']]['pair']
                ctx=e('\n\n'+'\n\n'.join(by[(pr,r-1)])) if r>=2 else []
                if variant=='irr' and ctx:
                    k=int(pr[1:]);irr=[];j=1
                    while len(irr)<len(ctx):
                        irr+=e('\n\n'+'\n\n'.join(by[(f'p{(k+j)%npairs}',r-1)]));j+=1
                    ctx=irr[:len(ctx)]
                rows.append((ctx+cur,(max(len(ctx)-1,0),len(ctx)+len(cur)-1)))
        if variant=='fill':
            for t in self.w['texts']:
                if t['kind']=='bg':x=e('\n\n'+t['text']);rows.append((x[:fill_len],(0,min(len(x),fill_len)-1)))
        return rows
    def fs_streams(self,swap,passes,length):
        """ladder.Ladder.fs_streams with the stream length as a parameter: pass 1 = the original stream cut to `length`,
        pass p >= 2 = same texts with background draw 7000+p, extended with further background until >= length."""
        streams=[self.stream_ids(swap)[:length]];w0=self.w;tok=self.tok
        for ps in range(2,passes+1):
            w=copy.deepcopy(w0);pool=gw.background(random.Random(7000+ps));bi=0;new=[]
            for t in w['texts']:
                if t['kind']=='bg':new.append({'kind':'bg','text':pool[bi%len(pool)]});bi+=1
                else:new.append(t)
            w['texts']=new;ids=_pre(self.bos)+enc(tok,gw.realize(w,swap)[0]).ids
            while len(ids)<length:
                w['texts'].append({'kind':'bg','text':pool[bi%len(pool)]});bi+=1;ids=_pre(self.bos)+enc(tok,gw.realize(w,swap)[0]).ids
            streams.append(ids[:length])
        _,_,spans=gw.realize(w0,swap)
        need=[s_['text'] for s_ in spans]
        if length<len(self.stream_ids(swap)):   # truncated stream (smoke runs): require what pass 1 contains
            t1=tok.decode(list(streams[0]));need=[x for x in need if x in t1]
        for ps,ids in enumerate(streams,1):
            assert len(ids)>=length,(ps,len(ids));txt=tok.decode(list(ids[:length]))
            miss=[x for x in need if x not in txt];assert not miss,(ps,len(miss),miss[:2])
            if ps>1:assert list(ids[:length])!=list(streams[0][:length]),ps
        return streams

def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))
def qwen_tokenizer_path(model_dir):return Path(model_dir)/'tokenizer.json'
def llama_tokenizer():return load_tokenizer(ROOT/'data/tokenizer/llama3/tokenizer.json')
def h(x):return hashlib.sha256(json.dumps(x).encode()).hexdigest()
