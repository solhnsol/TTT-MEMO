"""E57 ORC v2 mask draft (CPU only; E57 section 79; no GPU, no model). Variants of the answer-sentence mask over the one-cycle news stream (sentence fragments inside chunks as e55_uv.chunk_sents):
  orc    : 261 / ORCA rule (sentence_ends of e55_uv, fragment hidden when the strict bank matcher of ANY question holds)
  v1     : R1 = abbreviation-aware sentence splitting (no sentence end after Jan. Feb. Mar. Apr. Jun. Jul. Aug. Sep. Sept. Oct. Nov. Dec., Mr. Mrs. Ms. Dr. Jr. Sr. St. No., or a single-letter initial), strict matcher only
  v2     : R1 + R2 = strict matcher OR a PARTIAL form of a question in a fragment of an article of the SAME event as the question: DATE month alone (full / abbreviation, with or without period) and day alone (with ordinals); PERSON surname alone and given name alone (parts of <= 2 characters skipped); PLACE the main words (content words >= 4 letters, generic place words skipped); NUMBER the digits alone (with ordinal suffix)   [default of the draft]
  v2g    : R1 + R2 with the partial forms applied to the WHOLE stream (all events), for the over-masking comparison only
  v2p    : R2' = R1 + strict OR (partial form AND the key entity of the question (core words of S or the string S) stands in the same fragment or in the fragment before / after it of the same article)   [comparison only]
R3 (masking the next sentence) is not implemented."""
import sys,re,bisect,collections
import numpy as np
CH=1024;NW=8
ABBR={'jan.','feb.','mar.','apr.','jun.','jul.','aug.','sep.','sept.','oct.','nov.','dec.','mr.','mrs.','ms.','dr.','jr.','sr.','st.','no.'}
MONFULL=['january','february','march','april','may','june','july','august','september','october','november','december']
MONABB={1:['jan'],2:['feb'],3:['mar'],4:['apr'],5:['may'],6:['jun'],7:['jul'],8:['aug'],9:['sep','sept'],10:['oct'],11:['nov'],12:['dec']}
GENERIC_PLACE={'city','county','jail','airport','park','palace','fort','hotel','stadium','island','valley','beach','river','lake','state','province','district','region','airspace'}
STOPW={'the','and','of','for','from','with'}
def sentence_ends_v2(tk,ids):
    """as e55_uv.sentence_ends but a '.'-token that closes an abbreviation (ABBR) or a single-letter initial does not end the sentence; newline tokens still do"""
    ids=np.asarray(ids);toks=[tk.id_to_token(int(i)) or '' for i in ids];n=len(ids);e=np.zeros(n,bool)
    for i,t in enumerate(toks):
        nl='Ċ' in t;core=t.rstrip('"\')]}”’');nxt=toks[i+1] if i+1<n else ' '
        sp=(nxt[:1] in ('Ġ','Ċ',' ','\n')) or i+1==n
        if nl:e[i]=True;continue
        if core[-1:] in ('.','?','!') and sp:
            if core[-1:]=='.':
                j=i;parts=[core]
                while j>0 and not (toks[j][:1] in ('Ġ','Ċ')):j-=1;parts.append(toks[j])
                word=''.join(reversed(parts)).replace('Ġ','').replace('Ċ','')
                if word.lower() in ABBR or re.fullmatch(r'[A-Z]\.',word):continue
            e[i]=True
    if n:e[-1]=True
    return e
def partial_rxs(q):
    """-> list of compiled regexes of the partial forms of the gold of q (R2)"""
    ty=q['type'];c=q['canonical'].strip();out=[]
    if ty=='DATE':
        m=re.fullmatch(r'\s*([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*',c)
        if m and m.group(1).lower() in MONFULL:
            mi=MONFULL.index(m.group(1).lower())+1;d=int(m.group(2));names=[MONFULL[mi-1]]+MONABB[mi]
            out.append(re.compile(r'(?<![\w])(?:'+'|'.join(sorted(set(names),key=len,reverse=True))+r')\.?(?![\w])',re.I))
            out.append(re.compile(r'(?<![\w$.,:/-])0?%d(?:st|nd|rd|th)?(?![\w%%:]|[.,]\d)'%d))
    elif ty=='NUMBER':
        dg=re.findall(r'\d+',c)
        if dg:out.append(re.compile(r'(?<![\w$.,])'+re.escape(dg[0])+r'(?:st|nd|rd|th)?(?![\w%]|[.,]\d)'))
    elif ty=='PLACE':
        w=[x for x in re.findall(r"[A-Za-z][A-Za-z'’\-]+",c) if len(x)>=4 and x.lower() not in STOPW and x.lower() not in GENERIC_PLACE]
        for x in w:out.append(re.compile(r'(?<![\w])'+re.escape(x)+r'(?![\w])',re.I))
    else:
        w=[x for x in re.findall(r"[A-Za-z][A-Za-z'’\-]+",c)]
        pick=[w[0],w[-1]] if len(w)>=2 else w
        for x in dict.fromkeys(pick):
            if len(x)>2:out.append(re.compile(r'(?<![\w])'+re.escape(x)+r'(?![\w])',re.I))
    return out
def build(variant,ids,tk,tfile,items,iv,UVm,B,ends_old=None):
    """-> dict(fr [(chunk,start,len)], txt [str], hidden bool[], strict bool[], partial_only bool[], cross_event_only bool[], event [str], ends)"""
    n=len(ids)//CH
    if variant=='orc':ends=ends_old if ends_old is not None else UVm.sentence_ends(UVm.tokenizer(tfile),ids[:n*CH])
    else:ends=sentence_ends_v2(UVm.tokenizer(tfile),ids[:n*CH])
    fr=[]
    for c in range(n):
        for s_,e_ in UVm.chunk_sents(ends,c,CH):fr.append((c,c*CH+s_,e_-s_))
    txt=[tk.decode([int(x) for x in ids[s:s+l]]) for _,s,l in fr];starts=[x[0] for x in iv];art=[max(0,bisect.bisect_right(starts,s)-1) for _,s,_l in fr];ev=[iv[a][2] for a in art]
    ms=[B.ans_matcher(it['q'])[0] for it in items];nF=len(fr)
    strict_q=[[qi for qi,m in enumerate(ms) if m(t)] for t in txt];strict=np.array([bool(x) for x in strict_q])
    strict_same=np.array([any(items[qi]['event']==ev[j] for qi in strict_q[j]) for j in range(nF)]);cross_only=strict&~strict_same
    hidden=strict.copy();part=np.zeros(nF,bool)
    if variant in('v2','v2g','v2p'):
        for qi,it in enumerate(items):
            q=it['q'];prx=partial_rxs(q)
            if not prx:continue
            cs=B.core(q['S']);Sl=q['S'].lower()
            def has_ent(j):
                t=txt[j];return bool(cs&B.words_of(t)) or Sl in t.lower()
            for j in range(nF):
                if hidden[j] and variant!='v2p':continue
                if variant in('v2','v2p') and ev[j]!=q['event']:continue
                if not any(r.search(txt[j]) for r in prx):continue
                if variant=='v2p':
                    ok=has_ent(j) or (j>0 and art[j-1]==art[j] and has_ent(j-1)) or (j+1<nF and art[j+1]==art[j] and has_ent(j+1))
                    if not ok:continue
                part[j]=True
        hidden=hidden|part
    return dict(fr=fr,txt=txt,hidden=hidden,strict=strict,partial_only=part&~strict,cross_event_only=cross_only,event=ev,ends=ends,art=art)
def fractions(bd,n):
    """per write chunk k = 8 .. n-1: hidden share of the window tokens (chunks k-8..k-1) and of the current chunk; -> arrays"""
    fr=bd['fr'];hid=bd['hidden'];tok=np.zeros(n);hid_tok=np.zeros(n)
    for (c,s,l),h in zip(fr,hid):tok[c]+=l;hid_tok[c]+=l if h else 0
    win=[];chunk=[]
    for k in range(NW,n):
        t=tok[k-NW:k].sum();win.append(hid_tok[k-NW:k].sum()/t if t else np.nan);chunk.append(hid_tok[k]/tok[k] if tok[k] else np.nan)
    return np.array(win),np.array(chunk)
def summarize_variant(bd,n):
    win,chunk=fractions(bd,n);fr=bd['fr'];L=np.array([f[2] for f in fr]);h=bd['hidden']
    tot=int(L[h].sum());po=int(L[bd['partial_only']&h].sum());co=int(L[bd['cross_event_only']&h].sum());st=int(L[bd['strict']&h].sum())
    f=lambda a:dict(mean=float(np.nanmean(a)),std=float(np.nanstd(a)),min=float(np.nanmin(a)),max=float(np.nanmax(a)))
    return dict(hidden_fragments=int(h.sum()),fragments=len(fr),hidden_tokens=tot,hidden_token_share_of_stream=tot/int(L.sum()),window_hidden_share=f(win),chunk_hidden_share=f(chunk),
        masked_tokens_by_strict_match=st,masked_tokens_partial_only_no_strict_match=po,share_partial_only=po/tot if tot else None,masked_tokens_strict_but_only_other_event_questions=co,share_strict_cross_event_only=co/tot if tot else None,
        masked_tokens_strict_with_a_same_event_question=st-co,share_related_to_own_event_answer=(st-co)/tot if tot else None)
def selftest():
    import pathlib
    ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'scripts/exp07_write_ladder'),str(ROOT/'scripts/exp04_what_ttt_retains'),str(ROOT/'scripts/exp05_write_dynamics')]
    import e55_uv as UVm
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
    text="The film premieres on Mar. 24. Dr. Smith said so. J. Smith left on Sept. 19 and Jan. 5, 2026. It ended. Mr. Lee agreed."
    ids=tk.encode(text,add_special_tokens=False).ids;e=sentence_ends_v2(tk,ids);parts=[];cur=[]
    for i,t in enumerate(ids):
        cur.append(t)
        if e[i]:parts.append(tk.decode(cur));cur=[]
    assert parts==['The film premieres on Mar. 24.',' Dr. Smith said so.',' J. Smith left on Sept. 19 and Jan. 5, 2026.',' It ended.',' Mr. Lee agreed.'],parts
    q=dict(type='DATE',canonical='March 24',variants=['March 24'],S='x');rx=partial_rxs(q);assert len(rx)==2 and rx[0].search('on Mar. 24') and rx[0].search('MARCH') and rx[1].search('the 24th') and not rx[1].search('2024') and not rx[1].search('$24')
    q=dict(type='PERSON',canonical='Rashmika Mandanna',variants=[],S='x');rx=partial_rxs(q);assert len(rx)==2 and rx[0].search('Rashmika said') and rx[1].search('mandanna')
    q=dict(type='PERSON',canonical='Li Wu',variants=[],S='x');rx=partial_rxs(q);assert len(rx)==0,'parts of 2 characters or fewer are skipped'
    q=dict(type='NUMBER',canonical='112th',variants=[],S='x');rx=partial_rxs(q);assert rx[0].search('the 112th mayor') and not rx[0].search('1120')
    q=dict(type='PLACE',canonical='Jammu and Kashmir',variants=[],S='x');rx=partial_rxs(q);assert len(rx)==2
    print('e57_orc2 selftest ok')
if __name__=='__main__' and 'selftest' in sys.argv:selftest()
