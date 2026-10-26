"""E57 post-hoc exploratory (CPU only): how much answer information is left in the 264 IN_DELORC windows (W0 masked-window read, answer sentences deleted by the ORC rule, window refilled to 8192 tokens), per question at its primary position. Private research material (the example file holds stream sentences).
Window re-built exactly as e57_main_run.w0mask_plan (same fragments, same hidden rule, same refill). Per question q (bank_v1 gold forms, type, key entity S):
 (a) strict: regex occurrences / fragments of the gold forms (canonical + variants + fixed_surface, word boundary, case-insensitive) inside the kept window fragments, and of the bank matcher (ans_matcher incl. normalized DATE / NUMBER key).
 (b) partial: DATE month name alone / day number alone / month+day in any other spelling; NUMBER digits alone (any context, also '112th'); PERSON surname alone and first name alone; PLACE first / last word alone (>= 3 letters, not a stop word); occurrences and fragments.
 (c) sentences (fragments) in which a value of the answer's TYPE (type_strings: DATE / NUMBER / PERSON / PLACE) stands in the same sentence as the key entity (core words of S); the number of such sentences, the number of distinct values, and how many of the values are the gold.
 (d) DATE questions: timestamp-like fragments (<= 14 tokens, a date and a word of Published / Updated / IST / GMT / UTC / ET / AM / PM) that match the answer date; and the number of window articles whose publish date (metadata, NOT in the stream text) equals the answer month-day (informational).
 (e) why the ORC rule did not remove a leftover STRICT occurrence: split by the chunk boundary (string spans two consecutive fragments of different chunks), else other; leftover PARTIAL occurrences without any strict one are counted as alias / partial.
Gain of a question = rank ratio of OUT minus rank ratio of IN_DELORC (positive: the deleted window still helps); top-10 gain questions vs the rest.
  python3 e57_delorc_leak.py --bundle B --w0mask E57_w0mask.json --out PREFIX    (writes PREFIX.json and PREFIX_examples_private.json)"""
import sys,json,re,bisect,argparse,pathlib,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
for p in (HERE,HERE.parent,ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8
MON=['january','february','march','april','may','june','july','august','september','october','november','december']
STOPW={'the','and','for','from','with','said','news','city','state','court','police','government','minister','president','bank','new','group','company','national','international','university','united','day','man','of'}
TS_WORDS=re.compile(r'\b(published|updated|ist|gmt|utc|et|am|pm|posted)\b',re.I)
def forms(q):
    vs=[]
    for v in [q['canonical']]+list(q.get('variants',[]))+[q.get('fixed_surface','')]:
        if v and v.lower() not in {x.lower() for x in vs}:vs.append(v)
    return vs
def rx_of(vs):return re.compile('|'.join(r'(?<![\w])'+re.escape(v)+r'(?![\w])' for v in vs),re.I) if vs else None
def partial_rx(q):
    """-> {name: compiled regex} of the partial forms of the gold"""
    ty=q['type'];c=q['canonical'].strip();out={}
    if ty=='DATE':
        m=re.fullmatch(r'\s*([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*',c)
        if m and m.group(1).lower() in MON:
            mo=m.group(1).lower();d=int(m.group(2));mi=MON.index(mo)+1
            out['month_alone']=re.compile(r'(?<![\w])(?:%s|%s)\.?(?![\w])'%(mo,mo[:3]),re.I);out['day_alone']=re.compile(r'(?<![\w$.,])0?%d(?:st|nd|rd|th)?(?![\w%%]|[.,]\d)'%d)
            out['other_spelling']=re.compile(r'(?<![\w])(?:(?:%s|%s)\.?\s+0?%d(?:st|nd|rd|th)?|0?%d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:%s|%s)\.?|0?%d/0?%d|0?%d/0?%d)(?![\w%%])'%(mo,mo[:3],d,d,mo,mo[:3],mi,d,d,mi),re.I)
    elif ty=='NUMBER':
        dg=re.findall(r'\d+',c)
        if dg:out['digits_alone']=re.compile(r'(?<![\w$.,])'+re.escape(dg[0])+r'(?:st|nd|rd|th)?(?![\w%]|[.,]\d)')
    else:
        w=[x for x in re.findall(r"[A-Za-z][A-Za-z'’\-]+",c) if len(x)>=3 and x.lower() not in STOPW]
        if len(w)>=2:out['last_word_alone']=re.compile(r'(?<![\w])'+re.escape(w[-1])+r'(?![\w])',re.I);out['first_word_alone']=re.compile(r'(?<![\w])'+re.escape(w[0])+r'(?![\w])',re.I)
        elif len(w)==1:out['single_word']=re.compile(r'(?<![\w])'+re.escape(w[0])+r'(?![\w])',re.I)
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--w0mask');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_bank as B
    from tokenizers import Tokenizer
    tf=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);tk=Tokenizer.from_file(tf)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH
    W=json.load(open(a.w0mask));prim=W['primary'];rows=W['rows'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids[:n*CH]);T=n*CH
    fr=[]
    for c in range(n):
        for s_,e_ in UVm.chunk_sents(ends,c,CH):fr.append((c,c*CH+s_,e_-s_))
    fst=np.array([f[1] for f in fr]);fln=np.array([f[2] for f in fr]);fch=np.array([f[0] for f in fr]);ms=[B.ans_matcher(it['q'])[0] for it in items];txt=[tk.decode([int(x) for x in ids[s:s+l]]) for _,s,l in fr]
    hidden=np.array([any(m(t) for m in ms) for t in txt])
    art_starts=[x[0] for x in iv];art_event=[x[2] for x in iv]
    E,idx=B.load(a.bundle);ev_dates={}
    for ev,Ev in E.items():
        for ai in range(len(Ev['rids'])):ev_dates[(ev,Ev['rids'][ai])]=Ev['dates'][ai]
    def window_pieces(kk):
        e=(kk+1)*CH;hi=int(np.searchsorted(fst+fln,e,side='right'));tot=0;j=hi-1;pieces=[]
        while j>=0 and tot<8192:
            if not hidden[j]:pieces.append(j);tot+=int(fln[j])
            j-=1
        return list(reversed(pieces))
    byrow=collections.defaultdict(dict)
    for r in rows:
        if r['condition']=='OUT':byrow['OUT'][r['question']]=r
        elif r['primary_position'] and r['condition']=='IN_DELORC':byrow['DELORC'][r['question']]=r
    res={};examples={};cache={}
    for qi,it in enumerate(items):
        q=it['q'];kk=prim[it['id']]
        if kk not in cache:cache[kk]=window_pieces(kk)
        pieces=cache[kk];vs=forms(q);srx=rx_of(vs);prx=partial_rx(q);M=ms[qi];cs=B.core(q['S']);ty=q['type']
        strict_occ=0;strict_frag=0;matcher_frag=0;part=collections.Counter();part_frag=collections.Counter();ent_type=0;ent_vals=collections.Counter();ent_gold=0;ts=0;split=0;abbr=0;exs=[]
        # chunk-boundary split: string spans the end of fragment j and the start of the next window piece j+1 when the two are adjacent in the stream (next chunk)
        pset=set(pieces)
        for jj,j in enumerate(pieces):
            t=txt[j];ns=len(srx.findall(t)) if srx else 0;strict_occ+=ns;strict_frag+=1 if ns else 0;matcher_frag+=1 if M(t) else 0
            for nm,rx in prx.items():
                k_=len(rx.findall(t));part[nm]+=k_;part_frag[nm]+=1 if k_ else 0
            if j+1 in pset and fst[j]+fln[j]==fst[j+1] and (srx or 'other_spelling' in prx):
                joined=t+txt[j+1];sp=[m for rx_ in ([srx] if srx else [])+([prx['other_spelling']] if 'other_spelling' in prx else []) for m in rx_.finditer(joined) if m.start()<len(t)<m.end()]
                if sp:
                    if fch[j]!=fch[j+1]:split+=1;exs.append(('chunk_boundary_split',t[-120:]+' ||| '+txt[j+1][:120]))
                    else:abbr+=1;exs.append(('sentence_split_inside_chunk (abbreviation such as Mar.)',t[-120:]+' ||| '+txt[j+1][:120]))
            low=t.lower();has_ent=bool(cs&B.words_of(t)) or (q['S'].lower() in low)
            if has_ent:
                vals=[(tp,key,sf) for tp,key,sf,a_,b_ in B.type_strings(t) if tp==ty]
                if vals:
                    ent_type+=1
                    for tp,key,sf in vals:
                        ent_vals[B.norm(sf) if ty in('PERSON','PLACE') else key]+=1
                        if (ty in('DATE','NUMBER') and key in B.ans_matcher(q)[1]) or (ty in('PERSON','PLACE') and any(B.related(sf,v) for v in vs)):ent_gold+=1
                    if len(exs)<8:exs.append(('entity_and_type_value',t[:240]))
            if ty=='DATE' and len(t.split())<=14 and TS_WORDS.search(t) and 'other_spelling' in prx and prx['other_spelling'].search(t):ts+=1
        pub=0
        if ty=='DATE' and 'other_spelling' in prx:
            mo,d=sorted(B.ans_matcher(q)[1])[0].split('-')[0:2] if B.ans_matcher(q)[1] else (None,None)
            arts=sorted({max(0,bisect.bisect_right(art_starts,int(fst[j]))-1) for j in pieces})
            for ai in arts:
                dd=ev_dates.get((art_event[ai],iv[ai][3]))
                if dd and mo and str(dd)[-4:]==f'{int(mo):02d}{int(d):02d}':pub+=1
        o=byrow['OUT'].get(it['id']);dl=byrow['DELORC'].get(it['id']);gain=(o['rank_ratio_all']-dl['rank_ratio_all']) if o and dl else None;gacc=(dl['acc_all']-o['acc_all']) if o and dl else None
        for nm,v in part_frag.items():
            if v and not strict_frag and len(exs)<8:exs.append((f'partial:{nm}',next((txt[j][:240] for j in pieces if prx[nm].search(txt[j])),'')))
        alias_only=sum(1 for nm in part if part[nm]>0 and not strict_occ)
        res[it['id']]=dict(event=it['event'],type=ty,primary_position=kk,window_fragments=len(pieces),rank_ratio_out=o['rank_ratio_all'] if o else None,rank_ratio_delorc=dl['rank_ratio_all'] if dl else None,acc_out=o['acc_all'] if o else None,acc_delorc=dl['acc_all'] if dl else None,gain_rank_ratio=gain,gain_acc=gacc,
            strict_occurrences=strict_occ,strict_fragments=strict_frag,matcher_fragments=matcher_frag,partial_occurrences=dict(part),partial_fragments=dict(part_frag),entity_and_type_value_sentences=ent_type,distinct_type_values_with_entity=len(ent_vals),gold_values_among_them=ent_gold,timestamp_like_date_fragments=ts,window_articles_with_publish_date_equal_answer=pub if ty=='DATE' else None,
            leftover_strict_boundary_split=split,leftover_answer_spelling_split_by_sentence_boundary_inside_chunk=abbr,leftover_strict_other=max(0,strict_occ-split),partial_without_strict=bool(part and not strict_occ and any(v>0 for v in part.values())))
        examples[it['id']]=[dict(kind=k,text=t) for k,t in exs[:3]]
    ok=[(k,v) for k,v in res.items() if v['gain_rank_ratio'] is not None];ok.sort(key=lambda kv:-kv[1]['gain_rank_ratio']);top=[k for k,_ in ok[:10]];rest=[k for k,_ in ok[10:]];flat=[k for k,v in ok if v['gain_rank_ratio']<=0]
    def mean(g,key):return float(np.mean([ (res[k][key] if not isinstance(res[k][key],dict) else sum(res[k][key].values())) or 0 for k in g])) if g else None
    keys=['strict_occurrences','strict_fragments','entity_and_type_value_sentences','distinct_type_values_with_entity','gold_values_among_them','timestamp_like_date_fragments','leftover_strict_boundary_split','leftover_answer_spelling_split_by_sentence_boundary_inside_chunk','window_fragments','partial_occurrences','partial_fragments']
    grp={nm:{k:mean(g,k) for k in keys} for nm,g in (('top10_gain',top),('rest',rest),('gain_le_0',flat))}
    out=dict(note='W0 IN_DELORC windows of 264 at each question primary position; counts of what is left of the answer information; gain = rank ratio OUT - rank ratio IN_DELORC',top10_gain_questions=top,group_means=grp,per_question=res,n_questions=len(res))
    pathlib.Path(str(a.out)+'.json').write_text(json.dumps(out,indent=1,default=float));pathlib.Path(str(a.out)+'_examples_private.json').write_text(json.dumps(dict(note='private, internal research only; stream sentences',examples=examples),indent=1,ensure_ascii=False))
    print('top10',top);print(json.dumps(grp,indent=0,default=float)[:1500])
if __name__=='__main__':main()
