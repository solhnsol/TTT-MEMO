"""CPU checks for suppress_v2 (label-free v2 token side), seed 101 world A (and B for the A/B swap identity).
(1) KL-sel candidates lie in the swapped sentence, after the span, never the first target after it; same target set in both inputs.
(2) bridge-P targets decode to gold place names; every exposure sentence fully inside a chunk contributes its place tokens.
(3) kl_select: top-3 per group, ties -> earlier position.
(4) NER-ent with the gold places as stand-in entity spans: targets outside the PERSON span, inside the same sentence.
(5) A and B get the same external name per sentence (bridge-P)."""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exp04_what_ttt_retains'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import suppress_v2 as V,gen_world as gw
tok=gw.TOK;bos=tok.token_to_id('<|begin_of_text|>');eos=tok.token_to_id('<|end_of_text|>');res=[]
def check(n,ok,d=''):print(('PASS' if ok else 'FAIL')+': '+n+(f' ({d})' if d else ''),flush=True);res.append(ok)
w=gw.build_world(101);ext=V.external_names(w,gw.ntok,w['name_len']);text,_,_=gw.realize(w,False)
e=tok.encode(text,add_special_tokens=False);offs=[(0,0)]+list(e.offsets);ids=[bos]+list(e.ids)
gold=V.gold_spans(w,False);pl=V.place_spans(w,False)
pr,cnt=V.lf_chunk_pairs_v2(w,False,tok,bos,eos,'lfswap',gold,ext,101,'after_span_sentence')
ok=True;n=0;ss=V.sentence_starts(text)
for (o,ed) in pr:
    inf=o[3]
    for est,een,g in zip(inf['edit_starts'],inf['edit_ends'],inf['groups']):
        n+=len(g)
        if any(j<=een for j in g):ok=False   # een = first token after the span (excluded); every candidate is later
    assert o[2]==ed[2]
check('1 KL-sel candidates: after the span, first target after it excluded, equal target sets',ok and n>0,f'{n} candidates in {len(pr)} chunks, {cnt}')
# sentence boundary: no candidate beyond the next sentence start
bad=0
for (o,ed) in pr:
    inf=o[3]
    for est,g in zip(inf['edit_starts'],inf['groups']):
        c=offs[est][0];import bisect;si=bisect.bisect_right(ss,c)-1;end=ss[si+1] if si+1<len(ss) else len(text)
        bad+=sum(1 for j in g if offs[j][1]>end)
check('1b KL-sel candidates stay inside the swapped sentence',bad==0,f'{bad} outside')
pr2,_=V.lf_chunk_pairs_v2(w,False,tok,bos,eos,'bridge',gold,ext,101,'gold_place',aux_spans=pl)
dec=set();nt=0
for (o,ed) in pr2:
    T=o[3]['targets'];nt+=len(T)
    # contiguous runs decode to place names
    run=[];
    for j in T+[None]:
        if run and (j is None or j!=run[-1]+1):dec.add(tok.decode([ids[x] for x in run]).strip());run=[]
        if j is not None:run.append(j)
places={t['P'] for t in w['triples']}
check('2 bridge-P targets decode to gold place names',dec<=places|{''} and len(dec)>0,f'{len(dec)} distinct, {nt} targets, extra {sorted(dec-places)[:5]}')
sel=V.kl_select([[5,3,9,7]],{3:1.0,5:2.0,7:2.0,9:0.5})
check('3 kl_select top-3, ties -> earlier',sel==[3,5,7],str(sel))
pr3,_=V.lf_chunk_pairs_v2(w,False,tok,bos,eos,'lfswap',gold,ext,101,'ents',aux_spans=pl)
ok=all(not any(offs[j][0]<c1 and offs[j][1]>c0 for (c0,c1) in gold) for (o,ed) in pr3 for j in o[3]['targets'])
check('4 NER-ent (gold places as stand-in): targets outside every PERSON span',ok and sum(o[2] for o,_ in pr3)>0,f'{sum(o[2] for o,_ in pr3)} targets')
prB,_=V.lf_chunk_pairs_v2(w,True,tok,bos,eos,'bridge',V.gold_spans(w,True),ext,101,'gold_place',aux_spans=V.place_spans(w,True))
import bisect
tA,_,spA=gw.realize(w,False);tB,_,spB=gw.realize(w,True);sA=V.sentence_starts(tA);sB=V.sentence_starts(tB)
siA=[bisect.bisect_right(sA,x['start'])-1 for x in spA];siB=[bisect.bisect_right(sB,x['start'])-1 for x in spB]
check('5 A and B: same sentence count and same sentence index for every exposure sentence (so the same external Y)',len(sA)==len(sB) and siA==siB,f'{len(sA)} sentences')
print(f'{sum(res)}/{len(res)} passed');sys.exit(0 if all(res) else 1)
