"""E57 write-time NLL analysis (CPU only): per-token NLL of the NAT write (E57_wnll_NAT.npz of e57_wnll_run.py, one cycle) split by answer sentences. Post-hoc, exploratory; no inference, counts and means.
Units: sentence fragments (e55_uv.chunk_sents of the stream; a sentence never crosses a chunk boundary). A fragment is an ANSWER FRAGMENT of question q when the bank matcher of q (e57_bank.ans_matcher: canonical + variants + normalized DATE / NUMBER key) holds on its decoded text. For every (fragment, q):
  copy class (before the write of its chunk k): B = a fragment holding the answer of the SAME question q lies in the window (chunks k-8 .. k-1); C = none in the window but an earlier fragment of the same chunk holds it (visible to the later tokens of the chunk); A = none of these (first appearance within reach).
  exposure rank = position of the fragment's article among the gold-bearing articles of q in the stream (1 = first exposure).
Per record: token count, mean NLL over the loss-mask tokens of the fragment, and the mean NLL over the GOLD-SPAN tokens (tokens whose characters overlap a regex occurrence of a gold form of q, found by decoding token prefixes). Reference: mean NLL of the tokens of fragments that hold no answer of any bank question.
  python3 e57_wnll_analysis.py --bundle B --npz O/E57_wnll_NAT.npz --out O/E57_wnll_analysis.json | --selftest"""
import sys,json,re,bisect,argparse,pathlib,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8
RANK_BINS=[(1,1),(2,3),(4,7),(8,15),(16,10**9)]
def rank_bin(r):
    for a,b in RANK_BINS:
        if a<=r<=b:return f'{a}' if a==b else (f'{a}-{b}' if b<10**9 else f'{a}+')
def find_shift(nll_tgt,ids,n):
    """target token t of chunk k sits at stream position k*CH+t+s: s in (0,1) with the best agreement"""
    best=None
    for s in (0,1):
        m=[];
        for k in range(min(n,12)):
            T=len(ids);pos=k*CH+np.arange(CH)+s;ok=pos<T;m.append(float(np.mean(nll_tgt[k][ok]==ids[pos[ok]])))
        r=float(np.mean(m))
        if best is None or r>best[1]:best=(s,r)
    assert best[1]>0.999,('target tokens do not match the stream (shift search)',best);return best[0]
def analyze(ids,ends,nll,lm,tgt,decode,matchers,forms_rx,events,art_starts,art_hits,UVm,n=None):
    """ids [N]; nll,lm,tgt [nch, CH]; decode(list of ids)->text; matchers [q] -> f(text)->bool; forms_rx [q] regex of the gold forms; events [q]; art_starts: start token of every article (sorted); art_hits[q]: article indices holding the gold; -> dict(records, reference, shift)"""
    n=n or nll.shape[0];s=find_shift(tgt,ids,n);nq=len(matchers)
    flat=np.full(len(ids),np.nan);fm=np.zeros(len(ids),bool)
    for k in range(n):
        pos=k*CH+np.arange(CH)+s;ok=pos<len(ids);flat[pos[ok]]=nll[k][ok];fm[pos[ok]]=lm[k][ok]
    frags=[]
    for c in range(n):
        for a,b in UVm.chunk_sents(ends,c,CH):frags.append((c,c*CH+a,b-a))
    hit=[[] for _ in range(nq)];anyhit=np.zeros(len(frags),bool);texts={}
    for i,(c,st,ln) in enumerate(frags):
        txt=decode([int(x) for x in ids[st:st+ln]]);texts[i]=txt
        for q in range(nq):
            if matchers[q](txt):hit[q].append(i);anyhit[i]=True
    ref_t=[];ref_f=0
    for i,(c,st,ln) in enumerate(frags):
        if not anyhit[i]:
            v=flat[st:st+ln][fm[st:st+ln]];
            if len(v):ref_t.append(v);ref_f+=1
    ref=np.concatenate(ref_t) if ref_t else np.zeros(0)
    recs=[]
    for q in range(nq):
        hc=[(frags[i][0],frags[i][1],i) for i in hit[q]]
        for i in hit[q]:
            c,st,ln=frags[i];prior_w=[1 for (c2,s2,j) in hc if c-NW<=c2<c];prior_c=[1 for (c2,s2,j) in hc if c2==c and s2<st]
            cls='B' if prior_w else ('C' if prior_c else 'A')
            art=bisect.bisect_right(art_starts,st)-1;rank=(art_hits[q].index(art)+1) if art in art_hits[q] else None
            sel=fm[st:st+ln];v=flat[st:st+ln][sel];txt=texts[i];gm=[m for m in forms_rx[q].finditer(txt)] if forms_rx[q] is not None else []
            gv=[]
            if gm:
                cum=[len(decode([int(x) for x in ids[st:st+j+1]])) for j in range(ln)];prev=0
                for j in range(ln):
                    lo,hi=prev,cum[j];prev=cum[j]
                    if fm[st+j] and any(m.start()<hi and m.end()>lo for m in gm):gv.append(flat[st+j])
            recs.append(dict(q=q,event=events[q],chunk=c,start=st,n_tok=int(sel.sum()),mean_nll=float(v.mean()) if len(v) else None,cls=cls,rank=rank,rank_bin=rank_bin(rank) if rank else None,gold_tok=len(gv),gold_mean_nll=float(np.mean(gv)) if gv else None))
    return dict(shift=s,records=recs,reference=dict(fragments=ref_f,tokens=int(len(ref)),mean_nll=float(ref.mean()) if len(ref) else None),n_fragments=len(frags),n_answer_fragments=int(anyhit.sum()))
def summarize(res):
    R=[r for r in res['records'] if r['mean_nll'] is not None];out={'reference_non_answer_fragments':res['reference']}
    def cell(rows,key='mean_nll',w='n_tok'):
        rows=[r for r in rows if r[key] is not None and r[w]>0]
        if not rows:return dict(n_records=0)
        v=np.asarray([r[key] for r in rows]);wt=np.asarray([r[w] for r in rows],float);return dict(n_records=len(rows),events=len({r['event'] for r in rows}),tokens=int(wt.sum()),mean_of_records=float(v.mean()),token_weighted_mean=float((v*wt).sum()/wt.sum()),se_records=float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else None)
    out['answer_fragments_all']=cell(R)
    for cls in 'ABC':
        out[f'class_{cls}']=cell([r for r in R if r['cls']==cls]);out[f'class_{cls}_gold_span']=cell([r for r in R if r['cls']==cls],'gold_mean_nll','gold_tok')
    for a,b in RANK_BINS:
        nm=rank_bin(a)
        for cls in 'ABC':
            rr=[r for r in R if r['cls']==cls and r['rank_bin']==nm]
            if rr:out[f'rank{nm}_class_{cls}']=cell(rr);out[f'rank{nm}_class_{cls}_gold_span']=cell(rr,'gold_mean_nll','gold_tok')
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--npz');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    ROOT=HERE.parents[1];import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_bank as B
    from tokenizers import Tokenizer
    tf=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);tk=Tokenizer.from_file(tf)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items)
    Z=np.load(a.npz);nll,lm,tgt=Z['nll'],Z['lm'],Z['tgt'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids)
    def forms(q):return [v for v in [q['canonical']]+list(q.get('variants',[]))+[q.get('fixed_surface','')] if v]
    rxs=[re.compile('|'.join(r'(?<![\w])'+re.escape(v)+r'(?![\w])' for v in forms(it['q'])),re.I) for it in items]
    res=analyze(ids,ends,nll,lm,tgt,lambda x:tk.decode(x),[B.ans_matcher(it['q'])[0] for it in items],rxs,[it['event'] for it in items],[x[0] for x in iv],[hits[it['id']] for it in items],UVm,n=len(ids)//CH)
    for r in res['records']:r['question']=items[r['q']]['id']
    S=summarize(res);pathlib.Path(a.out).write_text(json.dumps(dict(summary=S,shift=res['shift'],n_fragments=res['n_fragments'],n_answer_fragments=res['n_answer_fragments'],records=res['records']),default=float,indent=0))
    for k,v in S.items():print(k,json.dumps({x:(round(y,4) if isinstance(y,float) else y) for x,y in v.items()}) if isinstance(v,dict) else v)
def selftest():
    import e55_uv as UVm
    rng=np.random.default_rng(3);n=14;N=n*CH;ids=np.arange(N)%997+5;ends=np.zeros(N,bool);ends[np.arange(N)%100==99]=True;ends[np.arange(1,n+1)*CH-1]=True
    # question 0: its answer token 777 appears in sentences at chunks 2, 4, 5 (window copies), question 1 in chunk 3 only
    ids=ids.copy();ids[ids==777]=778;ids[2*CH+10]=777;ids[4*CH+210]=777;ids[5*CH+510]=777;ids[3*CH+310]=555;ids[ids==555]=556;ids[3*CH+310]=555
    nll=rng.random((n,CH)).astype(np.float32)+1.0;tgt=np.zeros((n,CH),np.int32);lm=np.ones((n,CH),bool)
    for k in range(n):
        pos=k*CH+np.arange(CH)+1;ok=pos<N;tgt[k][ok]=ids[pos[ok]]
    tgt[n-1][CH-1]=0;lm[n-1][CH-1]=False
    # the target of position p is the token p+1: answer tokens at positions 2*CH+10 -> chunk 2, t=9
    for (p) in (2*CH+10,4*CH+210,5*CH+510):nll[p//CH][p%CH-1]=0.1
    dec=lambda x:' '.join(f'w{i}' for i in x)
    ms=[lambda t:' w777 ' in ' '+t+' ',lambda t:' w555 ' in ' '+t+' ']
    rx=[re.compile(r'(?<![\w])w777(?![\w])'),re.compile(r'(?<![\w])w555(?![\w])')]
    art_starts=[0,3*CH,6*CH];art_hits=[[0,1],[1]]
    res=analyze(ids,ends,nll,lm,tgt,dec,ms,rx,['e0','e1'],art_starts,art_hits,UVm,n=n);assert res['shift']==1
    R={(r['q'],r['chunk']):r for r in res['records']}
    assert R[(0,2)]['cls']=='A' and R[(0,4)]['cls']=='B' and R[(0,5)]['cls']=='B' and R[(1,3)]['cls']=='A',{k:v['cls'] for k,v in R.items()}
    assert R[(0,2)]['gold_tok']==1 and abs(R[(0,2)]['gold_mean_nll']-0.1)<1e-6,R[(0,2)]
    assert R[(0,2)]['rank']==1 and R[(0,4)]['rank']==2 and R[(0,5)]['rank']==2
    S=summarize(res);assert S['class_B']['n_records']==2 and S['class_A']['n_records']==2 and S['class_B_gold_span']['mean_of_records']<0.2
    assert res['reference']['mean_nll']>1.0
    print('e57_wnll_analysis selftest ok')
if __name__=='__main__':main()
