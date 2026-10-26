"""E57 lead-length table (CPU only, READ-ONLY on the bank files; writes leadlen_table.{json,md}; no model, no GPU; private internal research material). Every independent article of the bundle (all events) is truncated by ONE fixed rule: the first 1 / 2 / 3 sentences, or the first 64 / 128 / 192 tokens (real llama3 tokenizer; the truncated ids are decoded back to text), plus the current lead (reference). For each rule: (1) tokens and 1024-token chunks per cycle = updates per cycle (stream = all truncated articles of all events, one cycle); (2) c per question = articles whose truncated text holds the gold (the validator's answer matcher), and c_S = articles that also hold a core word of S; (3) number of questions with c < 10 (and c_S < 10); (4) min and median exposure c x R and total updates for R = 1..8.
  python3 e57_leadlen.py --out OUT --tokenizer tokenizer.json"""
import sys,json,math,argparse,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_bank as B
CH=1024;RS=range(1,9)
RULES=[('sent',1),('sent',2),('sent',3),('tok',64),('tok',128),('tok',192),('lead',0)]
def run(out,tokenizer):
    out=pathlib.Path(out);fd=out/'e57_bank';st,idx=B.load(out);V=[json.loads(l) for l in open(fd/'validated.jsonl') if l.strip()]
    from tokenizers import Tokenizer;tk=Tokenizer.from_file(str(tokenizer));enc=lambda s:tk.encode(s,add_special_tokens=False).ids
    M={q['id']:(B.ans_matcher(q)[0],B.core(q['S'])) for q in V};res=[]
    for kind,n in RULES:
        name={'sent':f'first {n} sentence(s)','tok':f'first {n} tokens','lead':'current lead (reference)'}[kind];ntok=0;narts=0;c={q['id']:0 for q in V};cs={q['id']:0 for q in V};texts=0
        for ev,E in st.items():
            ev_q=[q for q in V if q['event']==ev]
            for ai in range(len(E['rids'])):
                sents=E['sents'][ai];lead=sents[:max(1,E['kept'][ai])]
                if kind=='sent':txt=' '.join(sents[:n]);tk_n=len(enc(txt+'\n\n'))
                elif kind=='tok':ids=enc(' '.join(lead))[:n];txt=tk.decode(ids);tk_n=len(ids)+len(enc('\n\n'))
                else:txt=' '.join(lead);tk_n=len(enc(txt+'\n\n'))
                ntok+=tk_n;narts+=1
                if not ev_q:continue
                w=B.words_of(txt)
                for q in ev_q:
                    h,core=M[q['id']]
                    if h(txt):
                        c[q['id']]+=1
                        if core&w:cs[q['id']]+=1
        chunks=ntok/CH;upd=math.ceil(chunks);cv=sorted(c.values());csv=sorted(cs.values())
        row=dict(rule=name,articles=narts,tokens=ntok,chunks=round(chunks,1),updates_per_cycle=upd,c=c,c_S=cs,below10_c=sum(1 for x in cv if x<10),below10_cS=sum(1 for x in csv if x<10),c_min=cv[0],c_median=statistics.median(cv),c_S_min=csv[0],c_S_median=statistics.median(csv),
            exposure={f'R{r}':dict(min_c=cv[0]*r,median_c=statistics.median(cv)*r,min_cS=csv[0]*r,median_cS=statistics.median(csv)*r,updates=upd*r,q_ge_128_c=sum(1 for x in cv if x*r>=128),q_ge_128_cS=sum(1 for x in csv if x*r>=128)) for r in RS})
        res.append(row);print(f"{name}: {narts} articles, {ntok} tokens, {upd} updates/cycle, c min/median {cv[0]}/{statistics.median(cv)}, below 10: {row['below10_c']}",flush=True)
    (fd/'leadlen_table.json').write_text(json.dumps(res,indent=1,ensure_ascii=False))
    L=['# E57 lead-length table (counts only)','','Stream = all independent articles of the 10 events, one cycle; tokens with the llama3 tokenizer (+ blank-line separator); updates = ceil(tokens / 1024), one update per chunk. c = articles whose truncated text holds the gold; c_S = also holds a core word of S.','','| rule | tokens | updates per cycle | c min / median | questions c < 10 | c_S min / median | questions c_S < 10 |','|---|---|---|---|---|---|---|']
    for r in res:L.append(f"| {r['rule']} | {r['tokens']} | {r['updates_per_cycle']} | {r['c_min']} / {r['c_median']} | {r['below10_c']} of {len(V)} | {r['c_S_min']} / {r['c_S_median']} | {r['below10_cS']} of {len(V)} |")
    L+=['','Exposure c x R (c, not c_S) and total updates:','','| rule | R | min exposure | median exposure | questions >= 128 | total updates |','|---|---|---|---|---|---|']
    for r in res:
        for k in (1,2,4,6,8):e=r['exposure'][f'R{k}'];L.append(f"| {r['rule']} | {k} | {e['min_c']} | {e['median_c']} | {e['q_ge_128_c']} | {e['updates']} |")
    L+=['','Per question c (and c_S) for each rule:','','| question | '+' | '.join(r['rule'] for r in res)+' |','|---|'+'---|'*len(res)]
    for q in V:L.append(f"| {q['id']} | "+' | '.join(f"{r['c'][q['id']]} ({r['c_S'][q['id']]})" for r in res)+' |')
    (fd/'leadlen_table.md').write_text('\n'.join(L)+'\n');return res
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--tokenizer',required=True);a=ap.parse_args();run(a.out,a.tokenizer)
