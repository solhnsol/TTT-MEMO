"""E57 post-hoc exploratory (CPU only; reads the frozen bank and the news bundle, writes json + tsv): what the ORACLE answer mask of job 261 (ORC) hides, per write chunk, and how much of the answer occurrences in its window it covers.
The mask rule is re-derived from the code of e57_main_run.oracle_masks (not from the saved masks): for the write chunk k the window is the chunks k-8 .. k-1 (the chunk k being written is NOT in the window), cut into sentence fragments by e55_uv.chunk_sents (a sentence never crosses a chunk boundary: a sentence that spans two chunks is two fragments); a fragment is hidden when it holds the gold of ANY question of the bank (bank matcher e57_bank.ans_matcher: canonical form + listed variants, word-boundary, case-insensitive, plus the normalized DATE / NUMBER key found by facts_in_sentence). Windows k = 8 .. n+7 of a two-cycle copy of the stream cover the steady state of every later cycle (cycle 1 has no window before chunk 8).
Per window: hidden tokens (total), split by the reason: (a) fragment holds a regex form of a question of the SAME event as the article the fragment lies in; (b) only the normalized key (no regex form) of a same-event question; (c) only questions of OTHER events (the bank matcher is applied to every sentence of the stream). Occurrences: regex occurrences of the forms of every question in the contiguous decoded window (all articles) versus inside hidden fragments (the difference = occurrences cut by a chunk boundary), the same-event ones separately. Tokens of all fragments holding a same-event regex form, tokens of all fragments of the same-event articles. Alternate-spelling probe for DATE questions: month-name + day forms (Jul 31, July 31st, 31st of July, 31/07 ...) not covered by the question's own forms and not hidden."""
import sys,json,re,pathlib,argparse,bisect,collections
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
for p in (HERE,HERE.parent,ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import numpy as np
import e57_reads as RD
import e57_bank as B
import e55_uv as UVm
CH=1024;NW=8
MON=['january','february','march','april','may','june','july','august','september','october','november','december']
def _forms(q):
    vs=[]
    for v in [q['canonical']]+list(q.get('variants',[]))+[q.get('fixed_surface','')]:
        if v and v.lower() not in {x.lower() for x in vs}:vs.append(v)
    return vs
def _rx(vs):return re.compile('|'.join(r'(?<![\w])'+re.escape(v)+r'(?![\w])' for v in vs),re.I) if vs else None
def date_alt(q):
    """regexes of other spellings of a DATE answer 'Month D' (None when the canonical form is not of that shape)"""
    m=re.fullmatch(r'\s*([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*',q['canonical'])
    if not m or m.group(1).lower() not in MON:return None
    mi=MON.index(m.group(1).lower())+1;d=int(m.group(2));mn=MON[mi-1];ab=mn[:3]
    return re.compile(r'(?<![\w])(?:(?:%s|%s)\.?\s+0?%d(?:st|nd|rd|th)?|0?%d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:%s|%s)\.?|0?%d/0?%d|0?%d/0?%d)(?![\w%%])'%(mn,ab,d,d,mn,ab,mi,d,d,mi),re.I)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True);ap.add_argument('--out',required=True,help='prefix of the output files (.json, _windows.tsv, _questions.tsv)');ap.add_argument('--bank',default=None);ap.add_argument('--tokenizer',default=None);a=ap.parse_args()
    import gen_world_ctl as G
    tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);tk=UVm.tokenizer(tfile)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json'
    items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH;T=n*CH
    ids2=np.tile(ids,2);ends=UVm.sentence_ends(tk,ids2)
    starts=[x[0] for x in iv];ev_of=lambda pos:iv[max(0,bisect.bisect_right(starts,pos%T)-1)][2]
    Q=[]
    for it in items:
        q=it['q'];M,keys=B.ans_matcher(q);vs=_forms(q);Q.append(dict(id=it['id'],event=it['event'],type=it['type'],q=q,M=M,keys=keys,forms=vs,rx=_rx(vs),alt=date_alt(q) if it['type']=='DATE' else None))
    memo={}
    def frag(c,s,e):
        """analysis of the fragment of chunk c, tokens s..e of the chunk (cached per absolute start)"""
        a0=c*CH+s
        if a0 in memo:return memo[a0]
        txt=tk.decode([int(x) for x in ids2[a0:a0+(e-s)]]);ev=ev_of(a0);r=dict(tok=e-s,hit=False,same_rx=False,same_key=False,other=False,occ={},alt={})
        for j,qq in enumerate(Q):
            hold=qq['M'](txt)
            if not hold:continue
            r['hit']=True
            nrx=len(qq['rx'].findall(txt)) if qq['rx'] else 0
            if qq['event']==ev:
                if nrx:r['same_rx']=True
                else:r['same_key']=True
            else:r['other']=True
        for j,qq in enumerate(Q):                                   # occurrences of the forms of every question (also of questions that did not hold: none)
            nrx=len(qq['rx'].findall(txt)) if qq['rx'] else 0
            if nrx:r['occ'][j]=nrx
            if qq['alt'] is not None and qq['event']==ev:
                na=len(qq['alt'].findall(txt))
                if na:r['alt'][j]=na
        r['event']=ev;memo[a0]=r;return r
    W=[];per_q=collections.defaultdict(lambda:dict(win=0,occ=0,cov=0,cut=0,same_occ=0,same_cov=0))
    for k in range(NW,n+NW):
        hid=dict(total=0,same_rx=0,same_key_only=0,other_only=0,frags=0,hid_frags=0);tok_same_rx=0;tok_same_event_articles=0;tok_win=0
        occ_all=collections.Counter();occ_hidden=collections.Counter();alt_n=collections.Counter();alt_unhidden=collections.Counter()
        for c in range(k-NW,k):
            for s,e in UVm.chunk_sents(ends,c,CH):
                f=frag(c,s,e);hid['frags']+=1;tok_win+=e-s
                if f['hit']:
                    hid['hid_frags']+=1;hid['total']+=e-s
                    if f['same_rx']:hid['same_rx']+=e-s
                    elif f['same_key']:hid['same_key_only']+=e-s
                    else:hid['other_only']+=e-s
                if f['same_rx']:tok_same_rx+=e-s
                for j,v in f['occ'].items():
                    occ_all[j]+=v
                    if f['hit']:occ_hidden[j]+=v
                for j,v in f['alt'].items():
                    alt_n[j]+=v
                    if not f['hit']:alt_unhidden[j]+=v
        wtxt=tk.decode([int(x) for x in ids2[(k-NW)*CH:k*CH]]);occ_cont={}
        for j,qq in enumerate(Q):
            m=len(qq['rx'].findall(wtxt)) if qq['rx'] else 0
            if m:occ_cont[j]=m
        rec=dict(k=k,window_tokens=tok_win,hidden_tokens=hid['total'],hidden_share=hid['total']/max(1,tok_win),hidden_same_event_regex=hid['same_rx'],hidden_same_event_key_only=hid['same_key_only'],hidden_other_event_only=hid['other_only'],hidden_fragments=hid['hid_frags'],fragments=hid['frags'],
            tokens_in_fragments_with_same_event_regex_form=tok_same_rx,occ_contiguous=sum(occ_cont.values()),occ_in_fragments=sum(occ_all.values()),occ_in_hidden_fragments=sum(occ_hidden.values()),
            alt_spelling_occ_same_event=sum(alt_n.values()),alt_spelling_occ_unhidden=sum(alt_unhidden.values()))
        for j in set(occ_cont)|set(occ_all):
            d=per_q[Q[j]['id']];d['win']+=1;d['occ']+=occ_cont.get(j,0);d['cov']+=occ_hidden.get(j,0);d['cut']+=occ_cont.get(j,0)-occ_all.get(j,0)
            ev=Q[j]['event']
        W.append(rec)
    mean=lambda key:float(np.mean([w[key] for w in W]))
    summ={k:mean(k) for k in ('window_tokens','hidden_tokens','hidden_share','hidden_same_event_regex','hidden_same_event_key_only','hidden_other_event_only','tokens_in_fragments_with_same_event_regex_form','occ_contiguous','occ_in_fragments','occ_in_hidden_fragments','alt_spelling_occ_same_event','alt_spelling_occ_unhidden')}
    summ['n_windows']=len(W);summ['coverage_of_contiguous_occurrences']=sum(w['occ_in_hidden_fragments'] for w in W)/max(1,sum(w['occ_contiguous'] for w in W))
    pq={i:dict(windows_with_occurrence=d['win'],occ_sum=d['occ'],covered=d['cov'],cut_by_chunk_boundary=d['cut'],coverage=(d['cov']/d['occ'] if d['occ'] else None)) for i,d in per_q.items()}
    out=pathlib.Path(a.out);json.dump(dict(rule='see module docstring',summary=summ,per_question=pq,windows=W),open(str(out)+'.json','w'),indent=1)
    with open(str(out)+'_windows.tsv','w') as fh:
        cols=list(W[0]);fh.write('\t'.join(cols)+'\n')
        for w in W:fh.write('\t'.join(str(w[c]) for c in cols)+'\n')
    with open(str(out)+'_questions.tsv','w') as fh:
        fh.write('question\twindows_with_occurrence\toccurrences_in_windows(sum)\tcovered_by_mask\tcut_by_chunk_boundary\tcoverage\n')
        for i,d in sorted(pq.items()):fh.write(f"{i}\t{d['windows_with_occurrence']}\t{d['occ_sum']}\t{d['covered']}\t{d['cut_by_chunk_boundary']}\t{'' if d['coverage'] is None else round(d['coverage'],4)}\n")
    print('mask coverage:',json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in summ.items()}),flush=True)
if __name__=='__main__':main()
