"""Label-free v2 token side (draft spec paper/2026-10-17-label-free-v2-spec.md as relayed 2026-10-17 00:24 KST; CPU code only,
not run on GPU until the registered version arrives). Cells (all with the c2-type swap rule of suppress.lf_chunk_pairs):
  KL-sel    NER PERSON spans -> external name; candidate targets = tokens of the swapped sentence after the span, without
            the first one; the loss uses the top-3 candidates per swapped sentence by W0 next-token KL(orig || swap) (kl_select;
            ties -> earlier position). tmode 'after_span_sentence'.
  NER-ent   same swap; targets = tokens of NER LOC/ORG/MISC spans in the same sentence, outside the PERSON span. tmode 'ents'.
  bridge-P  gold name spans -> external name; targets = gold place tokens of the sentence. tmode 'gold_place'.
Both inputs keep exactly the same target set (asserted, as in v1)."""
import random,re,bisect
import numpy as np
from suppress import *
from suppress import _ptoks
def lf_chunk_pairs_v2(w,swap,tok,bos,eos,cell,spans_char,ext,seed,tmode,aux_spans=None):
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
            new=sub[:c0-a]+r+sub[c1-a:];edits.append((tt[0],tt[-1]+1,tok.encode(new,add_special_tokens=False).ids,si,c0,c1))
        if not edits:continue
        edits.sort(key=lambda e_:e_[0])
        for x,y in zip(edits,edits[1:]):assert x[1]<=y[0],'overlapping edits'
        newids=[];pmap={};j=lo;ei=0;inedit=set()
        while j<hi:
            if ei<len(edits) and edits[ei][0]==j:
                inedit|=set(range(edits[ei][0],edits[ei][1]));newids+=edits[ei][2];j=edits[ei][1];ei+=1;continue
            pmap[j]=len(newids);newids.append(ids[j]);j+=1
        allow=set();groups=[]
        for e_ in edits:
            si_=e_[3];sc0=ss[si_];sc1=ss[si_+1] if si_+1<len(ss) else len(text)
            if tmode=='after_span_sentence':   # KL-sel candidates: targets inside the swapped sentence after the span, minus the first one
                grp=[j for j in range(e_[1]+1,hi) if ends[j]<=sc1]
            elif tmode=='ents':                # NER-ent: tokens of LOC/ORG/MISC spans in the same sentence, outside the PERSON span
                grp=[j for (a_,b_) in aux_spans if a_>=sc0 and b_<=sc1 and not (a_<e_[5] and b_>e_[4]) for j in range(lo+1,hi) if ends[j]>a_ and starts[j]<b_]
            elif tmode=='gold_place':          # bridge-P: the gold place tokens of the sentence
                grp=[j for (a_,b_) in aux_spans if a_>=sc0 and b_<=sc1 for j in range(lo+1,hi) if ends[j]>a_ and starts[j]<b_]
            else:raise ValueError(tmode)
            grp=sorted({j for j in grp if j not in inedit and j in pmap and 1<=pmap[j]<CH});groups.append(grp);allow|=set(grp)
        T=[]
        for j in range(lo+1,hi):
            if j in inedit or j not in pmap or j not in allow:continue
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
        info={'k':k,'n_edits':len(edits),'len':L0,'targets':T,'edit_starts':[x[0] for x in edits],'edit_ends':[x[1] for x in edits],'lo':lo,'pmap':pmap,'groups':groups}
        out.append(((np.asarray(ids[lo:hi],np.int32),mk,len(T),info),(np.asarray(newids,np.int32),mk2,len(T),info)))
    return out,cnt

def place_spans(w,swap):
    """gold place occurrences (every world place name) in the realized text."""
    text,_,_=gw.realize(w,swap);places={t['P'] for t in w['triples']}|set(w['ctrl_places'])
    return sorted((m.start(),m.end()) for P in places for m in re.finditer(r'\b'+re.escape(P)+r'\b',text))
def kl_select(groups,kl_of,k=3):
    """per group (one swapped sentence) the k targets with the largest KL; ties -> the earlier position."""
    sel=set()
    for g in groups:sel|=set(sorted(sorted(g),key=lambda j:-kl_of[j])[:k])
    return sorted(sel)
def restrict(pair,keep):
    """the pair (orig, edit) with both masks restricted to the original target indices in `keep` (same set in both)."""
    (o,e)=pair;inf=o[3];keep=[j for j in inf['targets'] if j in set(keep)]
    mo=np.zeros(CH,bool);me=np.zeros(CH,bool)
    for j in keep:mo[j-inf['lo']-1]=True;me[inf['pmap'][j]-1]=True
    return ((o[0],mo,len(keep),dict(inf,targets=keep)),(e[0],me,len(keep),dict(inf,targets=keep)))

# ---- NER pipeline v2 (PROTOCOL "Label-free v2", 2026-10-17 00:33 KST): document-level majority consistency ----
def ner_v2(text,per_spans,other_spans):
    """per_spans / other_spans: char spans tagged PER / LOC-ORG-MISC (v1 tagger, word-snapped). A string is replaced only if most
    of its detections are PER (tie -> not replaced); then EVERY whole-word occurrence of it in the document is replaced.
    Returns (spans, table) with spans sorted and non-overlapping (longer strings first on overlap)."""
    from collections import Counter
    cp=Counter(text[a:b] for a,b in per_spans);co=Counter(text[a:b] for a,b in other_spans)
    nocc={s:len(re.findall(r'(?<!\w)'+re.escape(s)+r'(?!\w)',text)) for s in cp}
    # amendment 00:45 KST (+ tie rule dropped 01:00): (a) uppercase first letter, (c) PER detections cover >= 50% of the occurrences (exactly 50% replaces)
    keep=sorted([s for s in cp if s.strip() and s[0].isupper() and cp[s]>=0.5*max(nocc[s],1)],key=lambda s:-len(s))
    occ=[]
    for s in keep:occ+=[(m.start(),m.end()) for m in re.finditer(r'(?<!\w)'+re.escape(s)+r'(?!\w)',text)]
    out=[]
    for a,b in sorted(occ,key=lambda x:(x[0],-(x[1]-x[0]))):
        if out and a<out[-1][1]:continue
        out.append((a,b))
    return out,{'strings_per':len(cp),'strings_kept':len(keep),'strings_dropped_minority':len(cp)-len(keep)}
def ner_report(text,spans,gold_names,gold_places,CHc=None,tok_offs=None):
    gs=set(gold_names);ds=set(spans);tp=len(gs&ds)
    hit=sum(1 for a,b in gold_names if any(s<b and e>a for s,e in spans))
    plc=sum(1 for a,b in gold_places if any(s<b and e>a for s,e in spans))
    nontarget=[(s,e) for s,e in spans if not any(s<b and e>a for a,b in gold_names)]
    r={'n_gold':len(gold_names),'n_spans':len(spans),'precision_exact':tp/max(len(ds),1),'recall_exact':tp/max(len(gs),1),
       'target_detection_rate':hit/max(len(gold_names),1),'gold_place_swapped_rate':plc/max(len(gold_places),1),'n_nontarget_swaps':len(nontarget)}
    if tok_offs is not None:
        import bisect;ends=[b for _,b in tok_offs];r['chunks_with_nontarget']=len({bisect.bisect_right(ends,s)//CHc for s,_ in nontarget})
    return r
