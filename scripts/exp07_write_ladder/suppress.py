"""Token side of the suppression-signal screen (PROTOCOL_stage01.md, one-step replication + suppression screen,
2026-10-16). Pure Python (gen_world + tokenizers); no JAX. Builds, per world, the inputs whose place-token NLL gradients
form the oracle-free contrast signals:
  c1 (name ablation)  g(X-P) - g(null-P): the person name X replaced by the generic noun "the person"
                      ("The person" at the start of a sentence), loss on the place-name tokens only;
  c2 (entity swap)    g(X-P) - g(Y-P):    X replaced by Y, a random person of the same world that is neither X nor
                      X's swap partner (fixed per sentence: random.Random(seed*7919 + sentence index)).
Two conditions: 'iso'   every exposure sentence alone ('\\n\\n' + sentence), as in the stage-2 / ONESTEP g;
                'chunk' the stream chunks of the masked write (chunk alone, positions from 0, filler included), loss on
                        the place tokens of the exposure sentences lying fully inside the chunk.
Replacement in 'chunk' is done on tokens: the tokens covering an occurrence of X are re-tokenized from their own
characters with X replaced (the unreplaced re-tokenization must reproduce the original tokens, otherwise the occurrence
is left unchanged and counted)."""
import random,re
import numpy as np
import gen_world as gw
CH=1024;NCH=gw.NCH
def null_for(sent,x0):return 'The person' if x0==0 else 'the person'
def _sent_info(w,swap):
    _,name,spans=gw.realize(w,swap);tri={t['id']:t for t in w['triples']};pairs={}
    for t in w['triples']:pairs.setdefault(t['pair'],[]).append(t['id'])
    out=[]
    for i,s in enumerate(spans):
        t=tri[s['t']];X=name[t['id']];P=t['P'];partner=[u for u in pairs[t['pair']] if u!=t['id']][0]
        cand=sorted(name[u] for u in name if u not in (t['id'],partner));Y=random.Random(w['seed']*7919+i).choice(cand)
        xs=[m.start() for m in re.finditer(r'\b'+re.escape(X)+r'\b',s['text'])];ps=[m.start() for m in re.finditer(r'\b'+re.escape(P)+r'\b',s['text'])]
        assert xs and len(ps)==1,(s['text'],X,P)
        out.append(dict(s,idx=i,X=X,P=P,Y=Y,xs=xs,p0=ps[0]))
    return out
def _replace(text,X,xs,kind,Y):
    """replace every occurrence (char starts xs) of X; returns new text and the char shift function."""
    out=[];pos=0
    for x0 in xs:
        r=null_for(text,x0) if kind=='null' else Y
        out.append(text[pos:x0]);out.append(r);pos=x0+len(X)
    out.append(text[pos:]);return ''.join(out)
def _ptoks(offsets,c0,c1,base=0):
    """token indices (in the encoding) overlapping chars [c0,c1)."""
    return [j for j,(a,b) in enumerate(offsets) if b>c0 and a<c1]
def iso_rows(w,swap,tok,kind):
    """kind in X/null/Y -> list of (ids, (a,b)) rows with loss positions a..b-1 predicting the place tokens."""
    rows=[]
    for s in _sent_info(w,swap):
        txt=s['text'] if kind=='X' else _replace(s['text'],s['X'],s['xs'],kind,s['Y'])
        full='\n\n'+txt;e=tok.encode(full,add_special_tokens=False)
        assert txt.count(s['P'])==1;p0=2+txt.index(s['P'])
        pt=_ptoks(e.offsets,p0,p0+len(s['P']));assert pt and pt==list(range(pt[0],pt[-1]+1))
        assert tok.decode(e.ids[pt[0]:pt[-1]+1]).strip()==s['P'],(tok.decode(e.ids[pt[0]:pt[-1]+1]),s['P'])
        rows.append((e.ids,(pt[0]-1,pt[-1])))
    return rows
def chunk_rows(w,swap,tok,bos,eos,kind):
    """per stream chunk with >= 1 exposure sentence fully inside: (ids[CH] int32, lossmask[CH] bool, n_loss, info).
    kind 'X' = original chunk, place-token loss; 'null'/'Y' = chunk with X replaced inside those sentences;
    'ntp' = original chunk, loss on every position whose target is in the chunk (the masked write's own NTP loss)."""
    text,_,_=gw.realize(w,swap);e=tok.encode(text,add_special_tokens=False);ids=[bos]+list(e.ids);offs=[(0,0)]+list(e.offsets)
    assert len(ids)>=NCH*CH+1;info=_sent_info(w,swap);out=[];skipped=0
    def ctok(c):return next(j for j,(a,b) in enumerate(offs) if j>0 and b>c)
    for s in info:s['t0']=ctok(s['start']);s['t1']=ctok(s['end']-1)+1
    for k in range(NCH):
        lo,hi=k*CH,(k+1)*CH;inside=[s for s in info if s['t0']>=lo and s['t1']<=hi]
        if kind=='ntp':   # every chunk (the masked write updates on all of them), all in-chunk targets
            c=np.asarray(ids[lo:hi],np.int32);mk=np.ones(CH,bool);mk[-1]=False;out.append((c,mk,int(mk.sum()),{'k':k,'len':CH}));continue
        if not inside:continue
        # token edits: list of (t_start, t_end, new_tokens) in stream token indices
        edits=[]
        if kind!='X':
            for s in inside:
                for x0 in s['xs']:
                    c0=s['start']+x0;c1=c0+len(s['X']);tt=_ptoks(offs,c0,c1);tt=[j for j in tt if j>0]
                    a,b=offs[tt[0]][0],offs[tt[-1]][1];sub=text[a:b]
                    if tok.encode(sub,add_special_tokens=False).ids!=ids[tt[0]:tt[-1]+1]:skipped+=1;continue
                    r=null_for(s['text'],x0) if kind=='null' else s['Y']
                    new=sub[:c0-a]+r+sub[c1-a:];edits.append((tt[0],tt[-1]+1,tok.encode(new,add_special_tokens=False).ids))
        edits.sort();newids=[];pmap={};j=lo;ei=0
        while j<hi:
            if ei<len(edits) and edits[ei][0]==j:
                newids+=edits[ei][2];j=edits[ei][1];ei+=1;continue
            pmap[j]=len(newids);newids.append(ids[j]);j+=1
        mk=np.zeros(CH,bool);n=0
        for s in inside:
            c0=s['start']+s['p0'];pt=[j for j in _ptoks(offs,c0,c0+len(s['P'])) if j>0]
            assert pt and tok.decode([ids[j] for j in pt]).strip()==s['P']
            for j in pt:
                q=pmap[j]
                if 1<=q<CH:mk[q-1]=True;n+=1
        L0=len(newids);newids=(newids+[eos]*CH)[:CH]   # net-longer chunks lose their last (filler) tokens; shorter ones are padded
        out.append((np.asarray(newids,np.int32),mk,n,{'k':k,'n_edits':len(edits),'len':L0}))
    return out,skipped

# ---- label-free suppression screen (PROTOCOL_stage01.md "Label-free suppression screen", 2026-10-16 22:15 KST) ----
# cells: 'lfswap' (NER PER spans -> external name), 'lfablate' (NER PER spans -> "the person"), 'bridge' (gold name
# occurrences -> external name); all-token loss on the target set shared by the original and the edited chunk.
SYLX1=['Bar','Cel','Dav','Ern','Fal','Gor','Hen','Ilv','Jas','Kor','Lam','Mes','Nor','Oth','Par','Rin','Sev','Tir','Ur','Vas','Wil','Xan','Yel','Zor',
       'Ald','Bry','Cyr','Dun','Ev','Fyn','Gil','Hal','Ivo','Kas','Lor','Mil','Ned','Ost','Pry','Rol','Sab','Tam','Ulf','Vin','Wes','Aur','Bas','Cad',
       'Dem','Eir','Fil','Gwen','Hug','Ivan','Kip','Lys','Mad','Nil','Odo','Rud']
SYLX2=['on','ia','us','el','ar','in','ey','os','an','ira','ert','ald','ent','ola','ian','ott','ina','ius','ery','ond','ica','as','ew','olt','ys',
       'ek2','ul','em','ino','ach','ett','eda','ion','ith','ulf','ida','ost','ana','err','yn']
SYLX2=[s for s in SYLX2 if s.isalpha()]
SENT_RE=re.compile(r'(?<=[.!?])\s+|\n\n')
def external_names(w,ntok_fn,L):
    """synthetic single-word names from syllables disjoint from gen_world's (SYL1/SYL2), filtered to llama3 token length L
    (the world's name length rule); asserted disjoint from every gen_world person-name combination and the world's places."""
    assert not set(SYLX1)&set(gw.SYL1) and not set(SYLX2)&set(gw.SYL2)
    world_like={a+b for a in gw.SYL1 for b in gw.SYL2};places={t['P'] for t in w['triples']}|set(w['ctrl_places'])
    out=sorted({a+b for a in SYLX1 for b in SYLX2 if ntok_fn(a+b)==L})
    assert not set(out)&world_like and not set(out)&places and not set(out)&({t['X'] for t in w['triples']}|set(w['ctrl_persons']))
    return out
def sentence_starts(text):
    st=[0]+[m.end() for m in SENT_RE.finditer(text)];return st
def lf_chunk_pairs(w,swap,tok,bos,eos,cell,spans_char,ext,seed):
    """per chunk with >= 1 replaceable span fully inside: ((orig_ids, mk, n, info), (edit_ids, mk2, n, info)) with the SAME
    target set (original token indices) in both; plus counters. spans_char: list of (c0, c1) in the realized text."""
    import bisect
    text,_,_=gw.realize(w,swap);e=tok.encode(text,add_special_tokens=False);ids=[bos]+list(e.ids);offs=[(0,0)]+list(e.offsets)
    ss=sentence_starts(text);cnt={'spans':len(spans_char),'cross_chunk':0,'retok_skip':0,'used':0,'chunks':0,'cut_targets':0}
    def sidx(c):return bisect.bisect_right(ss,c)-1
    out=[];ends=[b_ for _,b_ in offs];starts=[a_ for a_,_ in offs];sp_tt=[]
    for c0,c1 in spans_char:   # tokens overlapping [c0,c1) (offsets are sorted), computed once
        j0=max(1,bisect.bisect_right(ends,c0));j1=bisect.bisect_left(starts,c1,lo=j0);sp_tt.append((c0,c1,[j for j in range(j0,j1) if ends[j]>c0 and starts[j]<c1]))
    for k in range(NCH):
        lo,hi=k*CH,(k+1)*CH;edits=[]
        for c0,c1,tt in sp_tt:
            if not tt or tt[-1]<lo-CH or tt[0]>=hi+CH:continue
            if tt[0]<lo or tt[-1]>=hi:
                if tt[0]<hi and tt[-1]>=lo:cnt['cross_chunk']+=1
                continue
            a,b=offs[tt[0]][0],offs[tt[-1]][1];sub=text[a:b]
            if tok.encode(sub,add_special_tokens=False).ids!=ids[tt[0]:tt[-1]+1]:cnt['retok_skip']+=1;continue
            si=sidx(c0)
            if cell=='lfablate':r='The person' if c0==ss[si] else 'the person'
            else:r=random.Random(seed*7919+si).choice(ext)
            new=sub[:c0-a]+r+sub[c1-a:];edits.append((tt[0],tt[-1]+1,tok.encode(new,add_special_tokens=False).ids))
        if not edits:continue
        edits.sort()
        for x,y in zip(edits,edits[1:]):assert x[1]<=y[0],'overlapping edits'
        newids=[];pmap={};j=lo;ei=0;inedit=set()
        while j<hi:
            if ei<len(edits) and edits[ei][0]==j:
                inedit|=set(range(edits[ei][0],edits[ei][1]));newids+=edits[ei][2];j=edits[ei][1];ei+=1;continue
            pmap[j]=len(newids);newids.append(ids[j]);j+=1
        T=[]
        for j in range(lo+1,hi):
            if j in inedit or j not in pmap:continue
            q=pmap[j]
            if not (1<=q<CH):cnt['cut_targets']+=1;continue
            assert newids[q]==ids[j];T.append(j)
        L0=len(newids);newids=(newids+[eos]*CH)[:CH]
        mk=np.zeros(CH,bool);mk2=np.zeros(CH,bool)
        for j in T:mk[j-lo-1]=True;mk2[pmap[j]-1]=True
        assert mk.sum()==mk2.sum()==len(T)
        # same target set: the target token at every masked position is identical in both inputs
        to=[ids[lo+p+1] for p in np.flatnonzero(mk)];tn=[newids[p+1] for p in np.flatnonzero(mk2)];assert to==tn
        cnt['used']+=len(edits);cnt['chunks']+=1
        info={'k':k,'n_edits':len(edits),'len':L0,'targets':T,'edit_starts':[x[0] for x in edits],'lo':lo,'pmap':pmap}
        out.append(((np.asarray(ids[lo:hi],np.int32),mk,len(T),info),(np.asarray(newids,np.int32),mk2,len(T),info)))
    return out,cnt
def gold_spans(w,swap):
    """gold person-name occurrences (every name of the world) in the realized text."""
    text,name,_=gw.realize(w,swap);g=[]
    for n in set(name.values()):g+=[(m.start(),m.end()) for m in re.finditer(r'\b'+re.escape(n)+r'\b',text)]
    return sorted(g)
