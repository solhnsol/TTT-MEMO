"""E57 post-hoc exploratory (CPU only, reads the frozen bank and the news bundle, writes one json): occurrences of every question's answer in the cycle-1 stream.
Per question and per article of its event (stream order): token end, number of sentences the bank matcher accepts (what n_k counts, one article = one exposure if >=1), regex occurrences of every surface form (canonical + variants + fixed_surface, word-boundary, case-insensitive, repeats inside an article counted), and the normalized-key matches (DATE / NUMBER forms found by facts_in_sentence).
Summaries: cumulative occurrence count at the end of every article and at the end of the cycle, next to n_k (articles holding the gold). Exposures of one cycle = the same stream again, so cycle c totals = c x cycle-1 totals."""
import sys,json,re,pathlib,argparse
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
for p in (HERE,HERE.parent,ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_reads as RD
import e57_bank as B
def forms(q):
    vs=[]
    for v in [q['canonical']]+list(q.get('variants',[]))+[q.get('fixed_surface','')]:
        if v and v.lower() not in {x.lower() for x in vs}:vs.append(v)
    return vs
def count_forms(vs,text):
    return {v:len(re.findall(r'(?<![\w])'+re.escape(v)+r'(?![\w])',text,re.I)) for v in vs}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True);ap.add_argument('--out',required=True);ap.add_argument('--bank',default=None);ap.add_argument('--tokenizer',default=None);a=ap.parse_args()
    import gen_world_ctl as G
    enc=G.load_enc(a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json'
    items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items)
    st,idx=B.load(a.bundle);arts=[]
    for ev,E in st.items():
        for ai in range(len(E['rids'])):arts.append((int(E['dates'][ai]),ev,E['rids'][ai],E['sents'][ai][:max(1,E['kept'][ai])]))
    arts.sort(key=lambda x:(x[0],x[1],x[2]));assert len(arts)==len(iv),(len(arts),len(iv))
    out={'bank':str(bank),'stream_tokens':int(len(ids)),'n_articles':len(iv),'questions':{}}
    for it in items:
        q=it['q'];M,keys=B.ans_matcher(q);vs=forms(q);rows=[];tot={v:0 for v in vs};tot_sent=0;tot_key=0;n_art=0;cum=[]
        for k,(d,ev,rid,sents) in enumerate(arts):
            if ev!=it['event']:continue
            txt=' '.join(sents);s_hold=sum(1 for s in sents if M(s));fc=count_forms(vs,txt)
            key_only=0
            if keys:
                for s in sents:
                    rxhit=any(count_forms(vs,s).values())
                    if M(s) and not rxhit:key_only+=1
            nocc=sum(fc.values());n_art+=1 if M(txt) else 0
            for v in vs:tot[v]+=fc[v]
            tot_sent+=s_hold;tot_key+=key_only
            rows.append(dict(article=k,rid=rid,token_end=iv[k][1],matcher_article=bool(M(txt)),sentences_matched=s_hold,occurrences=nocc,by_form=fc,key_only_sentences=key_only,max_in_one_sentence=max([sum(count_forms(vs,s).values()) for s in sents]+[0])))
            cum.append(dict(token_end=iv[k][1],n_k_articles=n_art,cum_occurrences=sum(tot.values()),cum_sentences_matched=tot_sent))
        out['questions'][it['id']]=dict(event=it['event'],type=it['type'],stratum=it['stratum'],forms=vs,gold=it['cands'][0],n_k_cycle=n_art,occurrences_cycle=sum(tot.values()),sentences_matched_cycle=tot_sent,key_only_sentences_cycle=tot_key,occurrences_by_form_cycle=tot,articles=rows,cumulative=cum)
    summ=[(i,v['n_k_cycle'],v['occurrences_cycle'],v['sentences_matched_cycle'],v['key_only_sentences_cycle']) for i,v in out['questions'].items()]
    out['summary_columns']=['question','n_k_articles_per_cycle','regex_occurrences_per_cycle','matched_sentences_per_cycle','sentences_matched_only_by_normalized_key'];out['summary']=summ
    json.dump(out,open(a.out,'w'),ensure_ascii=False,indent=1)
    print(f"wrote {a.out}; questions {len(summ)}; sum n_k {sum(s[1] for s in summ)}; sum occurrences {sum(s[2] for s in summ)}; sum sentences {sum(s[3] for s in summ)}",flush=True)
if __name__=='__main__':main()
