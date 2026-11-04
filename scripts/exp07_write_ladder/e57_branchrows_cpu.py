"""E57 deciding rows: share of branch-token rows and token types (CPU only; new file; method thread, Hans 11/04 09:20 KST 'cpu 점검 먼저 해봐 분기토큰 비율'): the deciding rows (top 30% of the eligible rows of every chunk by window gain g, eligible = loss rows whose next token is inside a sentence) of the news M3 w=8 write (band 128, e57_news_m3/news_m30_masks.npz, chunks 8..279) and of the synthetic s64 M3 arms (band 256, e57_synq/syn_s64_m30_masks.npz, chunks 8..127) compared for (1) the share of deciding rows that predict a BRANCH token (the first content token of a fact string), (2) the recall: the share of all branch-token rows that are deciding rows, with the base rate and the lift, (3) the token types of the predicted token of the deciding rows against all eligible rows.
Branch token definitions. Synthetic: fact strings = the place P (the read answer) [and the name X] of every fact sentence (world table); branch token = first content token of each occurrence. News (stream A, defined on the article text, not on T1/T2): fact string = (a) a capitalised word run of an article sentence (proper-noun phrase; a single capitalised word at the start of a sentence is not counted) or (b) a number / date expression (digits, month + day (+ year)); branch token = its first content token; second definition: the strings (gold + variants) of the 42 news bank questions found in the stream (word boundary).
Token types of the predicted token: number (has a digit), proper (capitalised word not at a sentence start), sentence-initial capital, common (lower-case word / word piece), punctuation, whitespace / format (space, newline only).
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python3 e57_branchrows_cpu.py --out O | --selftest"""
import sys,re,json,types,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024
MONTH=r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?'
NUM_RE=re.compile(r'\b'+MONTH+r'\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?|\b\d{1,2}(?:st|nd|rd|th)?\s+'+MONTH+r'(?:\s+\d{4})?|\d[\d,\.]*\d|\d')
CAP_RE=re.compile(r"[A-Z][\w'’\-]*(?:\.[A-Z]\.)?(?:\s+(?:of|the|de|la|von|van|and|&)\s+[A-Z][\w'’\-]*|\s+[A-Z][\w'’\-]*)*")
def fact_spans(text):
    """-> list of (start, end) char spans of the fact strings of a text"""
    out=[(m.start(),m.end()) for m in NUM_RE.finditer(text)]
    for m in CAP_RE.finditer(text):
        s,e=m.span();single='  ' not in m.group() and len(m.group().split())==1;pre=text[:s].rstrip();sent_start=(pre=='' or pre[-1:] in '.!?"\'“‘(' or text[max(0,s-1):s]=='\n')
        if single and sent_start:continue
        out.append((s,e))
    return sorted(set(out))
def first_content(tk,offs,ids,a,b):
    for j,((s,e),i_) in enumerate(zip(offs,ids)):
        if e>a and s<b and tk.decode([i_]).strip()!='':return j
    return None
def token_type(prev,cur):
    t=cur.strip()
    if t=='':return 'whitespace/format'
    if any(c.isdigit() for c in t):return 'number'
    if not any(c.isalpha() for c in t):return 'punctuation'
    p=prev.strip()
    if t[0].isupper():return 'sentence-initial capital' if (p=='' or p[-1] in '.!?' or '\n' in prev) else 'proper'
    return 'common'
CATS=['number','proper','sentence-initial capital','common','punctuation','whitespace/format']
def summarize(chunks,dec,elig,g,branch,cat,name):
    """dec / elig / branch / cat: dict chunk -> arrays [CH]"""
    nd=ne=nb=nbd=0;gb=[];gn=[];dc=np.zeros(len(CATS));ec=np.zeros(len(CATS))
    for k in chunks:
        e=elig[k];d=dec[k]&e;b=branch[k]&e;nd+=int(d.sum());ne+=int(e.sum());nb+=int(b.sum());nbd+=int((b&d).sum());gb.append(g[k][b]);gn.append(g[k][e&~b])
        for ci,c in enumerate(CATS):dc[ci]+=int(((cat[k]==ci)&d).sum());ec[ci]+=int(((cat[k]==ci)&e).sum())
    gb=np.concatenate(gb);gn=np.concatenate(gn);prec=nbd/nd;rec=nbd/nb if nb else None;base=nb/ne
    return dict(name=name,n_chunks=len(chunks),eligible_rows=ne,deciding_rows=nd,branch_rows=nb,branch_rows_in_deciding=nbd,base_rate_of_branch_among_eligible=base,share_of_deciding_that_is_branch=prec,recall_of_branch_rows_in_deciding=rec,lift=prec/base if base else None,mean_g_branch_rows=float(gb.mean()) if len(gb) else None,mean_g_other_eligible_rows=float(gn.mean()),
        token_types_of_deciding_rows={c:float(dc[i]/dc.sum()) for i,c in enumerate(CATS)},token_types_of_eligible_rows={c:float(ec[i]/ec.sum()) for i,c in enumerate(CATS)})
def selftest():
    sp=fact_spans('Mamdani won on July 31, 2026 in New York City with 3 votes. He left.');txt='Mamdani won on July 31, 2026 in New York City with 3 votes. He left.'
    got=[txt[a:b] for a,b in sp];assert 'July 31, 2026' in got and 'New York City' in got and '3' in got and 'He' not in got and 'Mamdani' not in got,got
    assert token_type('.',' The')=='sentence-initial capital' and token_type(' in',' York')=='proper' and token_type(' in',' 31')=='number' and token_type(' a',' the')=='common' and token_type('x',',')=='punctuation' and token_type('x','\n')=='whitespace/format'
    print('e57_branchrows_cpu selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_uv as UVm,e57_pqa2_stream as S,e57_dwg_analyze as DW,e57_t1t2v2_run as V
    tk=S.tokenizer();vocab_dec={};dec_tok=lambda i:vocab_dec.setdefault(int(i),tk.decode([int(i)]))
    res={}
    def common(Z,band_i,M,ids,key_prefix,chunks,tg):
        ends=UVm.sentence_ends(UVm.tokenizer(str(ROOT/'data/tokenizer/llama3/tokenizer.json')),ids);elig={};dec={};g={};cat={}
        for k in chunks:
            q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
            for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
            lm=Z['lm'][k];elig[k]=CM.rows_eligible(lm,sid)&lm;dec[k]=CM.up(M[f'{key_prefix}{k}'],CH);g[k]=(Z['nll_band'][band_i][k]-Z['nll_win'][k]).astype(np.float64)
            cat[k]=np.asarray([CATS.index(token_type(dec_tok(tg[k][i-1]) if i>0 else '',dec_tok(tg[k][i]))) for i in range(CH)])
        return elig,dec,g,cat
    # ---- synthetic s64
    import e55_run as R55,e57_syn_pass as SP
    enc,w,tf=R55.build(SP.synth_args(64),64,arms=['NAT']);Sx=R55.streams(w,enc);ids,sha=Sx['A'];ids=np.asarray(ids);Z=np.load(EXP/'e57_syn_s64/syn_s64_nll.npz');M=np.load(EXP/'e57_synq/syn_s64_m30_masks.npz',allow_pickle=True);chunks=list(range(8,128));tg=Z['tgt']
    elig,dec,g,cat=common(Z,list(Z['bands']).index(256),M,ids,'qb_TOP30_',chunks,tg)
    place=np.zeros(128*CH+1,bool);nameb=np.zeros(128*CH+1,bool)
    for r in DW.locate_syn(tk,w,None):place[r['pos'][0]]=True
    for c in w['chunks']:
        for s in c['sents']:
            X=w['facts'][s['fact']]['X'];seg=list(c['ids'][s['start']:s['end']])
            for pre in (' ',''):
                p=tk.encode(pre+X,add_special_tokens=False).ids;hit=[i for i in range(len(seg)-len(p)+1) if seg[i:i+len(p)]==p]
                if hit:nameb[c['k']*CH+s['start']+hit[0]]=True;break
    row_of=lambda arr:{k:np.asarray([arr[k*CH+i+1] for i in range(CH)]) for k in chunks}
    res['synthetic_s64']={'place_first_token':summarize(chunks,dec,elig,g,row_of(place),cat,'synthetic: first content token of the place (the read answer)'),'name_or_place_first_token':summarize(chunks,dec,elig,g,row_of(place|nameb),cat,'synthetic: first content token of the name or the place of a fact sentence')}
    # ---- news stream A
    frags=[json.loads(x) for x in open(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl')];frags=[f for f in frags if f.get('type')=='frag'];nid=V.stream_ids(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl','A');nchn=len(nid)//CH;heur=np.zeros(nchn*CH+1,bool);bank=np.zeros(nchn*CH+1,bool)
    bk=json.load(open(EXP/'e57_news/e57_bank/bank_v1.json'))['questions'];strs=sorted({s for q in bk for s in list(q.get('variants',[]))+[q.get('fixed_surface','')] if len(s)>=3},key=len,reverse=True);bre=re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(s) for s in strs)+r')(?![A-Za-z0-9])')
    for f in frags:
        e=tk.encode(f['text'],add_special_tokens=False);offs=e.offsets;ii=e.ids
        for (a_,b_) in fact_spans(f['text']):
            j=first_content(tk,offs,ii,a_,b_)
            if j is not None:heur[f['tok0']+j]=True
        for m in bre.finditer(f['text']):
            j=first_content(tk,offs,ii,*m.span())
            if j is not None:bank[f['tok0']+j]=True
    Zn=np.load(EXP/'e57_band/e57_band_W0.npz');Mn=np.load(EXP/'e57_news_m3/news_m30_masks.npz',allow_pickle=True);chn=list(range(8,nchn));tgn=Zn['tgt']
    elig,dec,g,cat=common(Zn,list(Zn['bands']).index(128),Mn,nid,'qb_TOP30_',chn,tgn)
    rown=lambda arr:{k:np.asarray([arr[k*CH+i+1] for i in range(CH)]) for k in chn}
    res['news_streamA']={'article_fact_strings':summarize(chn,dec,elig,g,rown(heur),cat,'news: first content token of a proper-noun phrase or number / date expression in the article text'),'bank_strings_in_stream':summarize(chn,dec,elig,g,rown(bank),cat,'news: first content token of a gold / variant string of the 42 bank questions found in the stream')}
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'e57_branchrows_cpu.json').write_text(json.dumps(res,indent=1))
    for st,d in res.items():
        for nm,s in d.items():print(st,nm,'eligible',s['eligible_rows'],'deciding',s['deciding_rows'],'branch rows',s['branch_rows'],'| base',round(s['base_rate_of_branch_among_eligible'],4),'share of deciding that is branch',round(s['share_of_deciding_that_is_branch'],4),'recall',round(s['recall_of_branch_rows_in_deciding'],3),'lift',round(s['lift'],2),'| g branch',round(s['mean_g_branch_rows'],2),'other',round(s['mean_g_other_eligible_rows'],2))
if __name__=='__main__':main()
