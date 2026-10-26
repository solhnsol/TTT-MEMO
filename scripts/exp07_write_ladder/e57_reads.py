"""E57 trial readout (CPU parts; the GPU orchestration is e57_trial_run.py). Candidate rows of the frozen bank for the reader of E55 (e55_gpu.Reader55 / packing.Packed: summed span log-prob of the candidate text after the question prefix, E49 convention, fp32), window bookkeeping, rank / accuracy metrics and the log rows of E57 section 39:
  one row per {cycle, read position, window start / end offsets, arm, condition (OUT | IN), IN window info (present | absent by gold surface-form match in the decoded window), question id, event, type, stratum, cumulative exposure n_k (gold-bearing articles fully read, counted from the article ids actually in the stream), c_SA, gold logprob, gold rank, n candidates, hard-only rank / n, accuracy all / hard} + the log-prob of every candidate (cand_lp, in the order of cand_surface) and the token length of every candidate (cand_ntok: the summed log-prob favours short candidates; per-token means can be computed from these two lists).
Candidates of a question: gold (the most frequent surface form in the lead = bank 'fixed_surface') + the easy candidates + the frozen hard (nominated) candidates; hard-only set = gold + hard_nominated."""
import sys,json,pathlib,re
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'e57'))
CH=1024
def prefix_of(q):
    t=q['frame'].replace('{S}',q['S']);return t.split('___')[0].rstrip()      # 3 frames end with '?' and no blank: the candidate follows the question mark
def load_items(bankfile):
    """-> items [{id,event,type,stratum,prefix,cands [str],gold_idx 0,hard_idx [..],n_easy,c_SA..}] in the order of the frozen bank"""
    b=json.load(open(bankfile));items=[]
    for q in b['questions']:
        gold=q['fixed_surface'];easy=[e['surface'] for e in q['easy']];hard=[h['surface'] for h in q['hard_nominated']]
        hard=list(dict.fromkeys(x for x in hard if x.lower()!=gold.lower() or (_ for _ in ()).throw(ValueError(f"{q['id']}: hard candidate {x!r} equals the gold"))))      # the hard list of the bank may hold the same surface twice (two sources): kept once
        hl={x.lower() for x in hard};seen={gold.lower()};cands=[gold];kinds=['gold']
        for s,k in [(x,'easy') for x in easy if x.lower() not in hl]+[(x,'hard') for x in hard]:       # an easy candidate equal to a hard one (e.g. the gold of another event) is dropped, the hard one is kept
            if s.lower() in seen:
                if k=='hard':raise ValueError(f"{q['id']}: hard candidate {s!r} equals the gold or another hard candidate")
                continue
            seen.add(s.lower());cands.append(s);kinds.append(k)
        # an easy candidate that duplicates a later hard candidate: keep the hard one
        items.append(dict(id=q['id'],event=q['event'],type=q['type'],stratum=q.get('stratum','main'),prefix=prefix_of(q),cands=cands,kinds=kinds,gold_idx=0,hard_idx=[i for i,k in enumerate(kinds) if k=='hard'],
            S=q['S'],variants=[q['canonical']]+list(q['variants']),q=q))
    return items
def build_rows(tok,items):
    """item-major raw rows [(ids,(a,b))] for packing.Packed (E55 convention: span shifted by one for the log-prob alignment), n_tok per candidate; the candidate text gets a leading space"""
    import stage2
    slot=stage2.make_slot(tok,None);raw=[];ntok=[];bad=[]
    for it in items:
        nt=[]
        for c in it['cands']:
            try:ids,span=stage2.item_ids(tok,slot,it['prefix'],' '+c)
            except AssertionError as e:bad.append((it['id'],c));ids,span=None,None
            if ids is not None:raw.append((ids,(span[0]-1,span[1]-1)));nt.append(span[1]-span[0])
        ntok.append(nt)
    return raw,ntok,bad
def metrics(lp,gold_idx,hard_idx):
    lp=np.asarray(lp,np.float64);g=lp[gold_idx];others=[j for j in range(len(lp)) if j!=gold_idx]
    r_all=1+sum(1 for j in others if lp[j]>g);r_hard=1+sum(1 for j in hard_idx if lp[j]>g);n_hard=len(hard_idx)+1
    return dict(gold_lp=float(g),rank_all=int(r_all),n_all=int(len(lp)),rank_hard=int(r_hard),n_hard=int(n_hard),rank_ratio_all=r_all/len(lp),rank_ratio_hard=r_hard/n_hard,acc_all=bool(r_all==1),acc_hard=bool(r_hard==1),
        margin=float(g-max(lp[j] for j in others)),margin_hard=float(g-max(lp[j] for j in hard_idx)) if hard_idx else None)
def split_scores(flat,items):
    out=[];o=0
    for it in items:n=len(it['cands']);out.append(np.asarray(flat[o:o+n]));o+=n
    assert o==len(flat),(o,len(flat));return out
def article_intervals(bundle_parent,enc,items):
    """-> (ids int64 [n*CH], list of (start,end,event,rid) per article in stream order, gold_hits {qid: [article idx holding the gold]}, text_of(start,end) ) : stream = all independent lead articles in global date order, each lead text + blank line, real tokenizer, cut to whole chunks"""
    import e57_bank as B
    st,idx=B.load(bundle_parent);arts=[]
    for ev,E in st.items():
        for ai in range(len(E['rids'])):arts.append((int(E['dates'][ai]),ev,E['rids'][ai],' '.join(E['sents'][ai][:max(1,E['kept'][ai])])))
    arts.sort(key=lambda x:(x[0],x[1],x[2]));ids=[];iv=[];t=0
    M={it['id']:B.ans_matcher(it['q'])[0] for it in items};hits={it['id']:[] for it in items};byev={}
    for it in items:byev.setdefault(it['event'],[]).append(it)
    for d,ev,rid,txt in arts:
        e=list(enc(txt+'\n\n'));iv.append((t,t+len(e),ev,rid));ids+=e;t+=len(e)
        for it in byev.get(ev,[]):
            if M[it['id']](txt):hits[it['id']].append(len(iv)-1)
    n=len(ids)//CH;return np.asarray(ids[:n*CH],np.int64),iv,hits
def exposure(iv,hits,qid,pos_tokens):
    """gold-bearing articles fully read when pos_tokens tokens of the cycle have been read"""
    return sum(1 for a in hits[qid] if iv[a][1]<=pos_tokens)
def window_present(items,text):
    import e57_bank as B
    return {it['id']:bool(B.ans_matcher(it['q'])[0](text)) for it in items}
def log_rows(items,scores,ntok,*,cycle,read_chunk,pos_tokens,win,arm,condition,iv,hits,winpresent=None,csa=None):
    """rows for one (read position, condition). win = (start, end) token offsets inside the cycle stream (None for OUT); n_k = cumulative exposure (gold-bearing articles fully read, counted from the article ids actually in the stream: cycles before + articles fully read in this cycle)"""
    rows=[]
    for it,lp,nt in zip(items,scores,ntok):
        m=metrics(lp,it['gold_idx'],it['hard_idx'])
        rows.append(dict(cycle=cycle,read_chunk=read_chunk,pos_tokens=pos_tokens,window_start=None if win is None else win[0],window_end=None if win is None else win[1],arm=arm,condition=condition,window_info=None if winpresent is None else ('present' if winpresent[it['id']] else 'absent'),
            question=it['id'],event=it['event'],type=it['type'],stratum=it['stratum'],n_k=cumulative_n(iv,hits,it['id'],cycle,pos_tokens),c_SA=None if csa is None else csa.get(it['id']),**m,cand_surface=it['cands'],cand_kinds=it['kinds'],cand_ntok=nt,cand_lp=[float(x) for x in lp]))
    return rows
def cumulative_n(iv,hits,qid,cycle,pos_tokens):
    """exposure n_k at read position pos_tokens of cycle `cycle` (1-based; cycle 0 = W0 = 0): full cycles before + articles fully read in this cycle"""
    if cycle<=0:return 0
    return (cycle-1)*len(hits[qid])+sum(1 for a in hits[qid] if iv[a][1]<=pos_tokens)
def selftest():
    lp=[-5.0,-6.0,-4.0,-7.0,-5.5,-8.0];m=metrics(lp,0,[1,2]);assert m['rank_all']==2 and m['n_all']==6 and m['rank_hard']==2 and m['n_hard']==3 and not m['acc_all'] and not m['acc_hard'] and abs(m['margin']+1.0)<1e-9
    m=metrics([-1,-2,-3],0,[2]);assert m['acc_all'] and m['acc_hard'] and m['rank_ratio_all']==1/3
    sc=split_scores(np.arange(7.0),[dict(cands=[1,2,3]),dict(cands=[1,2,3,4])]);assert [len(x) for x in sc]==[3,4]
    iv=[(0,10,'e','a'),(10,30,'e','b'),(30,50,'e','c')];hits={'q':[0,2]}
    assert exposure(iv,hits,'q',9)==0 and exposure(iv,hits,'q',10)==1 and exposure(iv,hits,'q',50)==2 and cumulative_n(iv,hits,'q',2,10)==3 and cumulative_n(iv,hits,'q',0,10)==0
    items=[dict(id='q',event='e',type='PERSON',stratum='main',cands=['a','b','c'],kinds=['gold','easy','hard'],gold_idx=0,hard_idx=[2])]
    r=log_rows(items,[[-1.0,-2.0,-0.5]],[[1,1,1]],cycle=1,read_chunk=3,pos_tokens=10,win=(0,10),arm='NAT',condition='IN',iv=iv,hits=hits,winpresent={'q':True},csa={'q':1.0})[0]
    assert r['n_k']==1 and r['rank_all']==2 and r['rank_hard']==2 and r['window_info']=='present' and r['cand_lp']==[-1.0,-2.0,-0.5]
    print('e57_reads selftest ok')
if __name__=='__main__':
    ap=__import__('argparse').ArgumentParser();ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
