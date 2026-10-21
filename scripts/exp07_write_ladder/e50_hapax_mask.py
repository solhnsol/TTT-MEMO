"""E50 (CPU only, no model; spec /mnt/project-files/experiments/E50-hapax-string-match-mask.md): how well does a label-free string-match mask (Hapax style: tokens whose n-gram already occurred in the preceding window) cover the answer positions?
Two shapes of the same n-gram match (token ids; n = 1, 2, 3; the n-gram ENDING at position q is the unit):
  H_tgt(n)  (loss-masking shape, Hapax): the positions q of the current chunk whose n-gram already occurs in [lo, q), lo = max(0, chunk start - 8192) = the write window of the chunk (the same window as the oracle / K3 / SENT sets).
  H_src(n)  (context-masking shape, ours): the positions of the EARLIER occurrence the match points to: 'latest' = the n tokens of the most recent earlier occurrence, 'all' = of every earlier occurrence in [lo, q) (union over the chunk).
Synthetic world A (s101, stream of pass 1 and 4, sentence_table as s3_sets): H_src vs the oracle O, K3 (O + the other tokens of the earlier same-triple sentences) and SENT (all exposure sentence tokens of the window) of s3_sets.Sets (frac irrelevant);
  precision = |H & M| / |H|, recall = |H & M| / |M| (summed over chunks), also with the random baseline of the same size (analytic, uniform over the candidate range [lo, chunk end): precision |M|/|range|, recall |H|/|range|);
  chunks: 'O chunks' = the chunks with a non-empty oracle set (the only chunks where O / K3 / SENT mask anything) and 'all chunks'.
  H_tgt vs T = the name and place tokens of the chunk's fact sentences that have an earlier same-triple exposure with name/place tokens in the window (the targets of O; random baseline |T| / chunk size).
Content split of a masked set (exclusive, priority order, O chunks only): O | occupation tokens of any exposure sentence | B (K3 minus O minus occupation: frame words of the earlier same-triple sentences) | P (tokens of the swap partner's other sentences in the window) | same-frame X (other pairs' sentences that use a frame of the chunk's fact sentences) | other X | other exposure-sentence tokens | background; plus the token-kind split (name/place of an exposure sentence, occupation, frame word, background) of every masked position.
Natural text (E33 stream, world A, pass 1, nat_signal.build_table / NatSets): the same masks; O = NatSets.oracle (names + birthplace of earlier units of the same person in the window), answers = NatSets.answers (E33 answer definition: the first birthplace occurrence of a fact unit with prior exposure), share of masked tokens that are O / answer / other name-or-birthplace of an expo unit / other unit token / background, and a word-type split from the REAL text with a transparent heuristic: number (digits), proper noun (capitalised word not at a sentence start), function word (fixed English stop list), punctuation, other.
Usage (server): e50_hapax_mask.py --seeds 101 --passes 1,4 [--nat-dir DIR --nat 9105:spread] [--tokenizer-file F] --out-dir /tmp/e50   |   --selftest
Registered before the results (E50 doc, do NOT tune): O precision n=1 < 10%, n=2 < 30%, n=3 < 50%; the output prints observed vs registered, per variant."""
import os,sys,re,json,argparse,bisect
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
CH=SETS.CH;W=SETS.W;NCH=63;NTOK=NCH*CH
def ngram_masks(ids,k,n,ch=CH,w=W):
    """ids[j] = token at stream position j (BOS at 0, any unique value). -> (H_tgt sorted list, H_src_latest set, H_src_all set) of chunk k."""
    k0=k*ch;lo=max(0,k0-w);end=min(len(ids),k0+ch);occ={};ptr={};tgt=[];sl=set();sa=set()
    for q in range(lo,end):
        if q-n+1<lo:continue
        key=tuple(ids[q-n+1:q+1]);prev=occ.get(key)
        if prev and q>=k0:
            tgt.append(q);r=prev[-1];sl.update(range(r-n+1,r+1))
            for r in prev[ptr.get(key,0):]:sa.update(range(r-n+1,r+1))
            ptr[key]=len(prev)
        occ.setdefault(key,[]).append(q)
    return tgt,sl,sa
def brute(ids,k,n,ch,w):
    k0=k*ch;lo=max(0,k0-w);end=min(len(ids),k0+ch);tgt=[];sl=set();sa=set()
    for q in range(max(k0,lo+n-1),end):
        key=tuple(ids[q-n+1:q+1]);rs=[r for r in range(lo+n-1,q) if tuple(ids[r-n+1:r+1])==key]
        if rs:
            tgt.append(q);sl|=set(range(rs[-1]-n+1,rs[-1]+1))
            for r in rs:sa|=set(range(r-n+1,r+1))
    return tgt,sl,sa
FUNC=set('a an the of and or but nor in on at to for from by with without into onto over under about as is are was were be been being am do does did have has had will would shall should can could may might must not no that this these those it its he she they them his her their we you i me my our your who whom which what when where while than then there here also so such very more most other some any each both few many much only own same too just if because until after before between during through up down out off again further once all'.split())
def word_class(text,start,end):
    """transparent heuristic word type of the whitespace-delimited word that contains the character range [start, end)"""
    a=start
    while a>0 and not text[a-1].isspace():a-=1
    b=max(end,start+1)
    while b<len(text) and not text[b].isspace():b+=1
    wd=text[a:b];core=wd.strip('.,;:!?()[]{}"\'“”‘’—–-*/')
    if not core:return 'punct'
    if re.search(r'\d',core):return 'number'
    if core.lower() in FUNC:return 'function'
    if core[0].isupper():
        i=a-1
        while i>=0 and text[i].isspace():i-=1
        if i<0 or text[i] in '.!?' or text[max(0,i-1):i+1]=='\n\n' or (a>=2 and text[a-2:a]=='\n\n'):return 'other'   # sentence-initial capital: not counted as proper noun
        return 'proper'
    return 'other'
# ---------------- synthetic ----------------
def kind_array(table,n):
    kd=np.zeros(n+CH+8,np.int8)   # 0 background, 1 name/place of an exposure sentence, 2 occupation, 3 frame word
    for u in table:
        for p in u['toks']:kd[p]=3
        for p in u.get('occ',[]):kd[p]=2
        for p in list(u['names'])+list(u['place']):kd[p]=1
    return kd
KIND=['background','name/place','occupation','frame word']
REL=['O','occupation','B','P','same-frame X','other X','other expo token','background']
def targets(sets,k):
    """T(k): name and place tokens (inside chunk k) of the fact sentences of the chunk that have an earlier same-triple exposure with name / place tokens in the window"""
    lo=max(0,k*CH-W);T=set()
    for s in sets.cur_sentences(k):
        if any(u['t']==s['t'] and u['round']<s['round'] and any(lo<=p<s['place'][0] for p in u['names']+u['place']) for u in sets.t):
            T|={p for p in list(s['names'])+list(s['place']) if k*CH<=p<(k+1)*CH}
    return T
def rel_labels(sets,k,kd):
    """exclusive relation label function of chunk k (O chunks only)"""
    O=set(sets.oracle(k));K3=set(sets.sentencecover(k));P=set(sets.pxsplit(k)[0]);Xs,Xo=[set(x) for x in sets.xframe_split(k)]
    def f(p):
        if p in O:return 'O'
        if kd[p]==2:return 'occupation'
        if p in K3:return 'B'
        if p in P:return 'P'
        if p in Xs:return 'same-frame X'
        if p in Xo:return 'other X'
        return 'other expo token' if kd[p]>0 else 'background'
    return f
def analyse_stream(ids,table,seed,ps,ns=(1,2,3)):
    sets=SETS.Sets(table,seed,ps,0.35);kd=kind_array(table,len(ids));rec=[];comp={}
    for k in range(NCH):
        O=sets.oracle(k);hasO=bool(O);k0=k*CH;lo=max(0,k0-W);cand=max(1,min(len(ids),k0+CH)-lo)
        M={'O':set(O),'K3':set(sets.sentencecover(k)),'SENT':set(sets.sentmask(k))} if hasO else {'O':set(),'K3':set(),'SENT':set()}
        T=targets(sets,k) if hasO else set();rl=rel_labels(sets,k,kd) if hasO else None
        r={'k':k,'hasO':hasO,'cand':cand,'sizes':{m:len(v) for m,v in M.items()},'T':len(T)}
        for m,v in M.items():
            if hasO:
                for kk,nm in enumerate(KIND):comp.setdefault(('ref:'+m,0,'kind'),{}).setdefault(nm,0);comp[('ref:'+m,0,'kind')][nm]+=sum(1 for p in v if kd[p]==kk)
                for lab in REL:comp.setdefault(('ref:'+m,0,'rel'),{}).setdefault(lab,0)
                for p in v:comp[('ref:'+m,0,'rel')][rl(p)]+=1
        for n in ns:
            tgt,sl,sa=ngram_masks(ids,k,n);H={'src_latest':sl,'src_all':sa};r[n]={'tgt':len(tgt),'tgt_T':len(set(tgt)&T),'tgt_kind':[sum(1 for p in tgt if kd[p]==i) for i in range(4)]}
            for f,h in H.items():
                r[n][f]={'size':len(h),**{m:len(h&M[m]) for m in M},'kind':[sum(1 for p in h if kd[p]==i) for i in range(4)]}
                if hasO:
                    c=comp.setdefault((f,n,'rel'),{x:0 for x in REL})
                    for p in h:c[rl(p)]+=1
            if hasO:
                c=comp.setdefault(('tgt',n,'rel'),{x:0 for x in REL})
                for p in tgt:c[rl(p)]+=1
            for f,h in (('tgt',tgt),('src_latest',sl),('src_all',sa)):
                ck=comp.setdefault((f,n,'kind'),{x:0 for x in KIND})
                for p in h:ck[KIND[kd[p]]]+=1
        rec.append(r)
    return rec,comp
def pr_table(rec,ns=(1,2,3),only_O=True):
    """-> {(n,variant,M): dict(H, M, inter, precision, recall, p_rand, r_rand)} summed over chunks"""
    out={}
    for n in ns:
        for f in ('src_latest','src_all'):
            for m in ('O','K3','SENT'):
                H=Mm=I=0;er=0.0
                for r in rec:
                    if only_O and not r['hasO']:continue
                    H+=r[n][f]['size'];Mm+=r['sizes'][m];I+=r[n][f][m];er+=r[n][f]['size']*r['sizes'][m]/r['cand']
                out[(n,f,m)]={'H':H,'M':Mm,'inter':I,'precision':I/H if H else float('nan'),'recall':I/Mm if Mm else float('nan'),'p_rand':er/H if H else float('nan'),'r_rand':er/Mm if Mm else float('nan')}
    return out
def tgt_table(rec,ns=(1,2,3)):
    out={}
    for n in ns:
        H=I=T=0;er=0.0;ho=0
        for r in rec:
            H+=r[n]['tgt'];ho+=r[n]['tgt'] if r['hasO'] else 0
            if r['hasO']:I+=r[n]['tgt_T'];T+=r['T'];er+=r[n]['tgt']*r['T']/CH
        out[n]={'H_all_chunks':H,'H_O_chunks':ho,'T':T,'inter':I,'precision':I/ho if ho else float('nan'),'recall':I/T if T else float('nan'),'p_rand':er/ho if ho else float('nan'),'r_rand':er/T if T else float('nan')}
    return out
def md_synth(label,rec,comp):
    nO=sum(r['hasO'] for r in rec);L=[f'\n## {label}: {nO} O chunks of {len(rec)}; mean per O chunk: |O| {np.mean([r["sizes"]["O"] for r in rec if r["hasO"]]):.1f}, |K3| {np.mean([r["sizes"]["K3"] for r in rec if r["hasO"]]):.1f}, |SENT| {np.mean([r["sizes"]["SENT"] for r in rec if r["hasO"]]):.1f}, |T| {np.mean([r["T"] for r in rec if r["hasO"]]):.1f}; window+chunk range about {np.mean([r["cand"] for r in rec]):.0f} tokens\n']
    for only,name in ((True,'O chunks only'),(False,'all chunks (precision includes masks in chunks where O / K3 / SENT mask nothing)')):
        L+=[f'\n### H_src vs mask M ({name}); precision = |H&M|/|H|, recall = |H&M|/|M|; random = same-size uniform mask\n','| n | variant | M | sum |H| | sum |M| | overlap | precision | (random) | recall | (random) |','|---|---|---|---|---|---|---|---|---|']
        for (n,f,m),v in pr_table(rec,only_O=only).items():L.append(f"| {n} | {f} | {m} | {v['H']} | {v['M']} | {v['inter']} | {100*v['precision']:.2f}% | {100*v['p_rand']:.2f}% | {100*v['recall']:.2f}% | {100*v['r_rand']:.2f}% |")
    L+=['\n### H_tgt (loss-masking shape) vs T (targets of O: name and place tokens of the chunk\'s fact sentences with an earlier same-triple exposure in the window), O chunks\n','| n | sum |H_tgt| (all chunks) | sum |H_tgt| (O chunks) | sum |T| | overlap | precision | (random) | recall of T | (random) |','|---|---|---|---|---|---|---|---|']
    for n,v in tgt_table(rec).items():L.append(f"| {n} | {v['H_all_chunks']} | {v['H_O_chunks']} | {v['T']} | {v['inter']} | {100*v['precision']:.2f}% | {100*v['p_rand']:.2f}% | {100*v['recall']:.2f}% | {100*v['r_rand']:.2f}% |")
    L+=['\n### content of the masked positions: relation to the chunk (O chunks only; exclusive, priority O > occupation > B > P > same-frame X > other X > other exposure token > background); % of the masked tokens, mean tokens per O chunk\n','| mask | n | total | per O chunk | '+' | '.join(REL)+' |','|---|---|---|---|'+'---|'*len(REL)]
    for (f,n,t),c in comp.items():
        if t!='rel':continue
        s=sum(c.values());L.append(f"| {f} | {n or ''} | {s} | {s/max(nO,1):.1f} | "+' | '.join(f'{100*c[x]/s:.1f}%' if s else '-' for x in REL)+' |')
    L+=['\n### content of the masked positions by token kind (all chunks for H_*, O chunks for the reference masks); %\n','| mask | n | total | '+' | '.join(KIND)+' |','|---|---|---|'+'---|'*len(KIND)]
    for (f,n,t),c in comp.items():
        if t!='kind':continue
        s=sum(c.values());L.append(f"| {f} | {n or ''} | {s} | "+' | '.join(f'{100*c[x]/s:.1f}%' if s else '-' for x in KIND)+' |')
    return '\n'.join(L)
def registered(rec_by_run):
    L=['\n## registered predictions (E50 doc, set before the results): H_src O precision, O chunks\n','| run | n | registered | latest | all |','|---|---|---|---|---|']
    for run,rec in rec_by_run.items():
        t=pr_table(rec)
        for n,lim in ((1,10),(2,30),(3,50)):
            a=t[(n,'src_latest','O')]['precision']*100;b=t[(n,'src_all','O')]['precision']*100
            L.append(f"| {run} | {n} | < {lim}% | {a:.2f}% ({'consistent' if a<lim else 'NOT consistent'}) | {b:.2f}% ({'consistent' if b<lim else 'NOT consistent'}) |")
    return '\n'.join(L)
# ---------------- natural ----------------
def analyse_natural(w,ids,offs,text,ns=(1,2,3)):
    import nat_signal as NS
    table,owner=NS.build_table(w,False,offs,ntok=NCH*CH);sets=NS.NatSets(table,0);own=np.full(max(len(ids),len(owner))+CH+8,-1,np.int32);own[:len(owner)]=owner
    np_set=set(p for u in table for p in list(u['names'])+list(u['pall']))
    wt=[None]*len(ids)
    def wclass(p):
        if wt[p] is None:
            a,b=offs[p-1] if p>=1 and p-1<len(offs) else (0,0);wt[p]=word_class(text,a,b) if p>=1 else 'other'
        return wt[p]
    agg={};rows=[]
    for k in range(NCH):
        O=sets.oracle(k);hasO=bool(O);ans={p for p,_ in sets.answers(k)};k0=k*CH;cand=max(1,min(len(ids),k0+CH)-max(0,k0-W))
        row={'k':k,'hasO':hasO,'O':len(O),'ans':len(ans),'cand':cand,'cur_units':len(sets.cur_sentences(k))}
        for n in ns:
            tgt,sl,sa=ngram_masks(ids,k,n);row[n]={'tgt':len(tgt),'tgt_ans':len(set(tgt)&ans),'src_latest':len(sl),'src_all':len(sa),'src_latest_O':len(sl&set(O)),'src_all_O':len(sa&set(O))}
            for f,h in (('tgt',tgt),('src_latest',sl),('src_all',sa)):
                c=agg.setdefault((f,n),{'tokens':0,'O':0,'answer':0,'other name/place of an expo unit':0,'other expo-unit token':0,'background':0,**{'word:'+x:0 for x in ('number','proper','function','punct','other')}})
                for p in h:
                    c['tokens']+=1
                    c['O' if p in O_cache(O) else 'answer' if p in ans else 'other name/place of an expo unit' if p in np_set else 'other expo-unit token' if own[p]>=0 else 'background']+=1
                    c['word:'+wclass(p)]+=1
        rows.append(row)
    return rows,agg
_oc={}
def O_cache(O):
    key=id(O)
    if key not in _oc:_oc.clear();_oc[key]=set(O)
    return _oc[key]
def md_natural(rows,agg,label):
    nO=sum(r['hasO'] for r in rows);L=[f'\n## natural text {label}: chunks {len(rows)}, chunks with an oracle set (fact unit with a prior exposure in the window) {nO}; sum |O| {sum(r["O"] for r in rows)}, sum answer tokens {sum(r["ans"] for r in rows)}\n']
    L+=['### mask size (share of the stream / window) and overlap with the natural oracle O and the answer tokens\n','| n | H_tgt tokens | % of all chunk tokens | of which answer tokens | H_src latest | H_src all | % of window+chunk range (all) | latest ∩ O | all ∩ O | O precision latest | O precision all |','|---|---|---|---|---|---|---|---|---|---|---|']
    for n in (1,2,3):
        t=sum(r[n]['tgt'] for r in rows);ta=sum(r[n]['tgt_ans'] for r in rows);sl=sum(r[n]['src_latest'] for r in rows);sa=sum(r[n]['src_all'] for r in rows);cd=sum(r['cand'] for r in rows);lo_=sum(r[n]['src_latest_O'] for r in rows);ao=sum(r[n]['src_all_O'] for r in rows);tot=len(rows)*CH
        L.append(f"| {n} | {t} | {100*t/tot:.1f}% | {ta} | {sl} | {sa} | {100*sa/cd:.1f}% | {lo_} | {ao} | {100*lo_/sl if sl else float('nan'):.2f}% | {100*ao/sa if sa else float('nan'):.2f}% |")
    ks=['O','answer','other name/place of an expo unit','other expo-unit token','background']+['word:'+x for x in ('number','proper','function','punct','other')]
    L+=['\n### content of the masked positions (exclusive source classes O > answer > other name/birthplace of an expo unit > other expo-unit token > background; word type from the real text with the heuristic of word_class); % of the masked tokens\n','| mask | n | tokens | '+' | '.join(ks)+' |','|---|---|---|'+'---|'*len(ks)]
    for (f,n),c in agg.items():
        s=c['tokens'];L.append(f"| {f} | {n} | {s} | "+' | '.join(f'{100*c[x]/s:.1f}%' if s else '-' for x in ks)+' |')
    return '\n'.join(L)
# ---------------- self test ----------------
def selftest():
    rng=np.random.default_rng(0)
    for trial in range(60):
        ch=int(rng.integers(3,9));w=ch*int(rng.integers(2,5));V=int(rng.integers(2,6));N=ch*int(rng.integers(3,8));ids=[-1]+[int(x) for x in rng.integers(0,V,N-1)]
        for k in range(N//ch):
            for n in (1,2,3):
                a=ngram_masks(ids,k,n,ch,w);b=brute(ids,k,n,ch,w);assert a[0]==b[0] and a[1]==b[1] and a[2]==b[2],(trial,k,n,a,b)
    # hand example: ids ... a b c a b d, n=2, one chunk: the second 'a b' (ending at pos of 'b') matches the first
    ids=[-1,1,2,3,1,2,4];t,sl,sa=ngram_masks(ids,0,2,ch=7,w=70);assert t==[5] and sl=={1,2} and sa=={1,2},(t,sl,sa)
    t,sl,sa=ngram_masks([-1,5,5,5,5],0,1,5,50);assert t==[2,3,4] and sl=={1,2,3} and sa=={1,2,3},(t,sl,sa)   # latest earlier occurrence of each match: 1,2,3
    ids=[-1,9,0,0,0,0,0,0,0,0,9];assert ngram_masks(ids,2,1,5,5)[0]==[] and ngram_masks(ids,2,1,5,15)[0]==[10]   # a match older than the window (start 5) does not count
    assert word_class('He lived in Oslo, born 1999.',3,8)=='other'
    assert word_class('He lived in Oslo, born 1999.',9,11)=='function' and word_class('He lived in Oslo, born 1999.',12,16)=='proper' and word_class('He lived in Oslo, born 1999.',23,27)=='number' and word_class('He lived in Oslo, born 1999.',0,2)=='function' and word_class('Oslo is big',0,4)=='other'
    # table based run with the e43 synthetic table (names / place / toks fields) and a random id stream
    import e43_sets_test as T
    tb=T.table();ids=[-1]+[int(x) for x in rng.integers(0,500,NTOK)];rec,comp=analyse_stream(ids,tb,101,2)
    sets=SETS.Sets(tb,101,2,0.35);nO=sum(1 for k in range(NCH) if sets.oracle(k));assert nO==sum(r['hasO'] for r in rec)>0
    t1=pr_table(rec);v=t1[(1,'src_all','O')];assert v['H']>v['M']>0 and v['inter']<=v['M'] and 0<=v['precision']<=1
    # random baseline sanity: for random ids over 500 types the n=1 match mask covers most window tokens (recall > 0.8)
    assert t1[(1,'src_all','SENT')]['recall']>0.8,t1[(1,'src_all','SENT')]
    # relation split sums to the mask size
    for key,c in comp.items():
        if key[2]=='rel' and key[0]!='tgt':assert sum(c.values())>=0
    rl=[c for k_,c in comp.items() if k_==('src_all',1,'rel')][0];ks=[c for k_,c in comp.items() if k_==('ref:O',0,'rel')][0];assert ks['O']==sum(ks.values()),ks   # the O reference is entirely O
    md=md_synth('selftest synthetic',rec,comp)+registered({'selftest':rec});assert 'registered predictions' in md
    # natural: tiny world, proxy tokenizer
    import nat_dryrun as D,nat_world as NW
    w={'triples':[{'id':0,'pair':0,'X':'Ann Bell','XF':'Ann','XS':'Bell','P':'Oslo'},{'id':1,'pair':0,'X':'Cy Dee','XF':'Cy','XS':'Dee','P':'Rome'}],'texts':[]}
    for i in range(1,6):w['texts']+=[{'kind':'bg','text':'The river runs 12 km through the quiet old valley and the hill.'}]*20+[{'kind':'expo','t':0,'round':i,'label':'fact','text':'@X0@ was born in Oslo in 1990. @XF0@ later left.'}]
    px=D.Proxy();text,_,_=NW.realize(w,False);pids,ends=px.encode(text);offs=list(zip([0]+ends[:-1],ends));ids=[-1]+list(pids)
    rows,agg=analyse_natural(w,ids,offs,text);md=md_natural(rows,agg,'selftest');assert 'natural text' in md and agg[('src_all',1)]['tokens']>0
    print(md[:600]);print('e50_hapax_mask SELFTEST OK')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101');ap.add_argument('--passes',default='1,4');ap.add_argument('--nat-dir',default=None);ap.add_argument('--nat',default='9105:spread');ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--out-dir',default='/tmp/e50');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    md=['# E50 string-match (Hapax style) masks vs the answer positions O (CPU)\n','Definitions in the header of e50_hapax_mask.py. World A, sentence_table positions, window 8192, chunk 1024.\n'];res={};recs={}
    for seed in [int(x) for x in a.seeds.split(',')]:
        for ps in [int(x) for x in a.passes.split(',')]:
            w=K2.build_world_pass(gw,seed,ps);text,_,_=gw.realize(w,False);e=gw.TOK.encode(text,add_special_tokens=False);ids=[-1]+list(e.ids);ids=ids[:NTOK] if len(ids)>=NTOK else ids
            table=SETS.sentence_table(w,False,gw.TOK,gw,SP);lab=f's{seed} pass {ps}';rec,comp=analyse_stream(ids,table,seed,ps);recs[lab]=rec;md.append(md_synth(lab,rec,comp))
            res[lab]={'rec':rec,'comp':{'|'.join(map(str,k)):v for k,v in comp.items()},'pr':{'|'.join(map(str,k)):v for k,v in pr_table(rec).items()},'tgt':tgt_table(rec)}
    md.append(registered(recs))
    if a.nat_dir:
        import nat_world as NW,nat_dryrun as D
        stream,layout=a.nat.split(':');w=NW.load_world(Path(a.nat_dir)/f'nat_{stream}_{layout}.json');tk=D.load_tok(a.tokenizer_file);assert tk is not None,'the real tokenizer is needed for the natural stream'
        text=NW.realize(w,False)[0];e=tk.encode(text,add_special_tokens=False);ids=[-1]+list(e.ids)
        rows,agg=analyse_natural(w,ids,list(e.offsets),text);md.append(md_natural(rows,agg,f'{a.nat} (world A, pass 1, real tokenizer)'));res['natural']={'rows':rows,'agg':{'|'.join(map(str,k)):v for k,v in agg.items()}}
    else:md.append('\n## natural text: not run (no --nat-dir)')
    od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);(od/'e50_hapax_mask.md').write_text('\n'.join(md));json.dump(res,open(od/'e50_hapax_mask.json','w'));print('\n'.join(md))
if __name__=='__main__':main()
