"""E57 answer-enriched stream B: size calculation (CPU only, READ-ONLY except streamb_size.{json,md}; no model, no GPU; private internal research material). Stream B = the unique answer-bearing sentences (sentences of the lead stream that hold the gold of a validated question, +-1 neighbour sentence) of the articles that hold a gold, in article date order, each article's selected sentences once even if several questions use them (duplicates across questions are merged). Reports: articles, sentences, words, tokens (real tokenizer when --tokenizer is given, else words x 1.3) and 1024-token chunks per cycle; per fact: c (articles), distinct answer sentences, and exposure for R = 1..8 cycles counted as (a) articles that hold the fact, (b) answer sentences, after merging.
  python3 e57_streamb.py --out OUT [--tokenizer tokenizer.json]"""
import sys,json,math,argparse,pathlib,collections,statistics
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_bank as B
CH=1024;RS=range(1,9)
def run(out,tokenizer=None):
    out=pathlib.Path(out);fd=out/'e57_bank';st,idx=B.load(out);V=[json.loads(l) for l in open(fd/'validated.jsonl') if l.strip()]
    enc=None
    if tokenizer:
        from tokenizers import Tokenizer;tk=Tokenizer.from_file(str(tokenizer));enc=lambda s:tk.encode(s,add_special_tokens=False).ids
    sel=collections.defaultdict(set);per_q={}                                  # (event, article idx) -> selected sentence indices
    for q in V:
        E=st[q['event']];holds,keys=B.ans_matcher(q);arts=0;sents=0;rid2ai={r:i for i,r in enumerate(E['rids'])};used=set()
        for ai in range(len(E['rids'])):
            lead=E['sents'][ai][:max(1,E['kept'][ai])];hit=[j for j,t in enumerate(lead) if holds(t)]
            if hit:
                arts+=1;sents+=len(hit)
                for j in hit:
                    for k in (j-1,j,j+1):
                        if 0<=k<len(lead):sel[(q['event'],ai)].add(k)
        per_q[q['id']]=dict(event=q['event'],c=arts,answer_sentences=sents)
    rows=[];tot_w=tot_t=0;n_s=0
    for (ev,ai) in sorted(sel,key=lambda k:(int(st[k[0]]['dates'][k[1]]),k[0],st[k[0]]['rids'][k[1]])):
        E=st[ev];lead=E['sents'][ai];txt=' '.join(lead[j] for j in sorted(sel[(ev,ai)]));w=len(txt.split());t=len(enc(txt+'\n\n')) if enc else round(w*1.3)
        tot_w+=w;tot_t+=t;n_s+=len(sel[(ev,ai)]);rows.append((ev,ai,len(sel[(ev,ai)]),w,t))
    chunks=tot_t/CH;byev=collections.defaultdict(lambda:[0,0,0,0])
    for ev,ai,ns,w,t in rows:
        b=byev[ev];b[0]+=1;b[1]+=ns;b[2]+=w;b[3]+=t
    exp={qid:{f'R{r}_articles':v['c']*r for r in RS}|{f'R{r}_sentences':v['answer_sentences']*r for r in RS} for qid,v in per_q.items()}
    ns_=[v['answer_sentences'] for v in per_q.values()];cs_=[v['c'] for v in per_q.values()]
    rep=dict(articles=len(rows),sentences=n_s,words=tot_w,tokens=tot_t,token_method='real tokenizer' if enc else 'words x 1.3',chunks_per_cycle=chunks,updates_per_cycle=math.ceil(chunks),by_event={k:dict(articles=v[0],sentences=v[1],words=v[2],tokens=v[3],chunks=round(v[3]/CH,1)) for k,v in byev.items()},
        per_question=per_q,exposure=exp,summary=dict(c=dict(min=min(cs_),median=statistics.median(cs_),max=max(cs_)),answer_sentences=dict(min=min(ns_),median=statistics.median(ns_),max=max(ns_)),
        questions_ge_128_articles={f'R{r}':sum(1 for v in per_q.values() if v['c']*r>=128) for r in RS},questions_ge_128_sentences={f'R{r}':sum(1 for v in per_q.values() if v['answer_sentences']*r>=128) for r in RS},
        updates={f'R{r}':math.ceil(chunks)*r for r in RS}))
    (fd/'streamb_size.json').write_text(json.dumps(rep,indent=1,ensure_ascii=False))
    L=['# E57 answer-enriched stream B: size per cycle (counts only)','',f"{rep['articles']} articles, {rep['sentences']} sentences, {rep['words']} words, {rep['tokens']} tokens ({rep['token_method']}), **{chunks:.1f} chunks per cycle** (one update per 1024-token chunk -> {rep['updates_per_cycle']} updates per cycle).",'','| event | articles | sentences | words | tokens | chunks |','|---|---|---|---|---|---|']
    for k,v in rep['by_event'].items():L.append(f"| {k} | {v['articles']} | {v['sentences']} | {v['words']} | {v['tokens']} | {v['chunks']} |")
    L+=['','Updates for R cycles: '+', '.join(f"R={r}: {rep['summary']['updates'][f'R{r}']}" for r in RS),'',f"Questions with >= 128 exposures, counted by articles: {rep['summary']['questions_ge_128_articles']}; counted by answer sentences: {rep['summary']['questions_ge_128_sentences']}",'','| question | event | articles c | answer sentences | exp R=1 | R=2 | R=4 | R=6 | R=8 (articles) | R=8 (sentences) |','|---|---|---|---|---|---|---|---|---|---|']
    for qid,v in sorted(per_q.items(),key=lambda kv:kv[1]['c']):L.append(f"| {qid} | {v['event']} | {v['c']} | {v['answer_sentences']} | {v['c']} | {v['c']*2} | {v['c']*4} | {v['c']*6} | {v['c']*8} | {v['answer_sentences']*8} |")
    (fd/'streamb_size.md').write_text('\n'.join(L)+'\n');print(f"stream B: {rep['articles']} articles, {n_s} sentences, {tot_t} tokens ({rep['token_method']}), {chunks:.1f} chunks per cycle");return rep
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--tokenizer');a=ap.parse_args();run(a.out,a.tokenizer)
