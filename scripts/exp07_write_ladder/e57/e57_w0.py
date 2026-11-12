"""E57-lite W0 check: does the base model (weights W0, no news read, no training, FORWARD ONLY) already guess the PLACE answer from the question alone? Two stages.
  --prepare  (CPU, tokenizers only) place_clean.json (job 227) -> w0_items.json: per item (event, S) 16 candidate places = the answer A (column 0) + 15 distractors of the SAME token length (the lab reader needs equal span lengths), question 'Question: In the news about this event, which place is mentioned together with <S>?\\nAnswer:' (candidate = ' ' + place). Distractor order (fixed before any result): (1) the other valid places of the same event (place_clean other_places), (2) the A of the other events, (3) gazetteer places of e57_placeclean (title-cased); within a tier by a seeded md5 hash; no distractor that is the same place as A (one contains the other) or shares a core word with S. An item with fewer than 15 same-length distractors is dropped ('short pool'). Hard events (s / s2 < 1.5 in job 227) are included and flagged.
  --score    (GPU, forward only, no training, no update; lab runtime as the fam / rev commands of e55_run.py) -> w0_scores.json (summed span log-probs (n, 16), W0, neutral 8192-token SQuAD tail as context as in E55), w0_items_kept.json, w0_report.md.
  EXCLUSION RULE (fixed before any result): u = (16 - rank of A) / 15 with rank 1 = best of the 16 (ties averaged). An item is dropped as 'guessable from the name alone' if u(W0) >= 0.9 OR the top-1 candidate (any candidate) leads the 2nd by >= 2 nats. Both reasons are counted separately. Everything else is kept.
  --decide-only <scores.json> and --selftest are CPU."""
import argparse,json,sys,re,os,hashlib,random,pathlib,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import e57_placeclean as PC
FRAME='Question: In the news about this event, which place is mentioned together with {S}?\nAnswer:'
NC=16;SEED=57229;U_MAX=0.9;MARGIN=2.0
def hsh(*x):return hashlib.md5(('|'.join(map(str,(SEED,)+x))).encode()).hexdigest()
def title(k):return ' '.join(w if w in('and','of') else w.capitalize() for w in k.split())
def span_len(tok,prefix,cand):
    """token count of the candidate inside '\\n\\n'+prefix+cand, as stage2.item_ids finds it (offsets)"""
    text="\n\n"+prefix+cand;e=tok.encode(text,add_special_tokens=False);a=len("\n\n"+prefix)
    pos=[i for i,(s,t) in enumerate(e.offsets) if t>a and s<len(text)];return len(pos)
def prepare(pc,tok,log=print):
    events=[dict(r,hard=False) for r in pc['main']]+[dict(r,hard=True) for r in pc['hard']];Aall={r['event']:r['A'] for r in events}
    gaz=sorted(title(k) for k in PC.GAZ);items=[];dropped=[];cache={}
    def L(p):
        if p not in cache:cache[p]=span_len(tok,FRAME.format(S='X'),' '+p)
        return cache[p]
    for r in events:
        for it in r['items']:
            q=FRAME.format(S=it['S']);A=r['A'];LA=L(A);cs=PC.core(it['S'])
            tiers=[[x['surface'] for x in r.get('other_places',[])],[v for k,v in sorted(Aall.items()) if k!=r['event']],gaz];cands=[A];seen={}
            for tier in tiers:
                for p in sorted(tier,key=lambda p:hsh(r['event'],it['S_key'],p)):
                    if len(cands)>=NC:break
                    if p in cands or PC.same_place(p,A) or any(PC.same_place(p,c) for c in cands) or PC.core(p)&cs or L(p)!=LA:continue
                    cands.append(p)
            if len(cands)<NC:dropped.append(dict(event=r['event'],S=it['S'],A=A,reason='short pool',n=len(cands)));continue
            items.append(dict(event=r['event'],hard=r['hard'],S=it['S'],S_key=it['S_key'],A=A,prefix=q,cands=cands,span_len=LA))
    return dict(frame=FRAME,seed=SEED,items=items,dropped=dropped,n_events=len(events))
def decide(S):
    """S (n, 16) summed span log-probs, column 0 = A -> per item dict(rank, u, margin, top1_is_A, drop_u, drop_margin)"""
    S=np.asarray(S,float);g=S[:,0:1];rank=1+(S>g).sum(1)+0.5*((S==g).sum(1)-1);u=(NC-rank)/(NC-1);srt=np.sort(S,axis=1)[:,::-1];m=srt[:,0]-srt[:,1]
    return [dict(rank=float(r),u=float(x),margin=float(y),top1_is_A=bool(r==1),drop_u=bool(x>=U_MAX),drop_margin=bool(y>=MARGIN)) for r,x,y in zip(rank,u,m)]
def finish(items,S,out,log=print):
    dec=decide(S);kept=[];cnt=collections.Counter()
    for it,d in zip(items,dec):
        why=('u' if d['drop_u'] else '')+('+' if d['drop_u'] and d['drop_margin'] else '')+('margin' if d['drop_margin'] else '')
        cnt[why or 'kept']+=1
        if not why:kept.append(dict(it,u_W0=d['u'],margin_W0=d['margin']))
    ev=collections.defaultdict(lambda:[0,0])
    for it,d in zip(items,dec):ev[it['event']][0]+=1;ev[it['event']][1]+=not(d['drop_u'] or d['drop_margin'])
    L=['# E57-lite W0 check (base model, no reading, forward only)','',f'Rule fixed before the result: drop an item if u(W0) >= {U_MAX} or the top-1 candidate leads the 2nd by >= {MARGIN} nats. u = (16 - rank of A)/15 over the 16 same-length candidates (summed span log-prob).','',
       f"- items scored: {len(items)}; kept: {cnt['kept']}; dropped: {len(items)-cnt['kept']} (u >= {U_MAX} only: {cnt['u']}, margin only: {cnt['margin']}, both: {cnt['u+margin']}).",f"- mean u(W0) over all items: {np.mean([d['u'] for d in dec]):.3f} (chance 0.5); A ranked 1st in {np.mean([d['top1_is_A'] for d in dec]):.2f} of the items (chance 0.0625); mean top-1 margin {np.mean([d['margin'] for d in dec]):.2f} nats.",
       f"- mean u(W0) of the kept items: {np.mean([x['u_W0'] for x in kept]) if kept else float('nan'):.3f}",'','## per event','','| event | items | kept |','|---|---|---|']+[f'| {k} | {v[0]} | {v[1]} |' for k,v in ev.items()]
    txt='\n'.join(L)+'\n';out=pathlib.Path(out);(out/'w0_report.md').write_text(txt);(out/'w0_items_kept.json').write_text(json.dumps(kept,indent=1));log(txt);return kept,dec
def score(a):
    """GPU: forward only, mirrors e55_run.fam (lab runtime, Reader55, neutral 8192-token tail, summed span log-prob at W0 = rt.base)"""
    import e55_run as R55
    OUT=pathlib.Path(a.out);P=json.load(open(OUT/'w0_items.json'));items=P['items'];LD,s3,C=R55.lab()
    import e55_gpu as RG,stage2
    rt=LD.Runtime();rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    tok=rt.tok;slot=stage2.make_slot(tok,None);forbid=sorted({c for it in items for c in it['cands']}|{it['S'] for it in items});out_tail=R55.out_tail_8192(LD,tok,rt.bos,forbid)
    S=np.zeros((len(items),NC));groups=collections.defaultdict(list)
    for i,it in enumerate(items):groups[it['span_len']].append(i)
    for Ln,idx in sorted(groups.items()):
        raw=[]
        for i in idx:
            for c in items[i]['cands']:
                ids,span=stage2.item_ids(tok,slot,items[i]['prefix'],' '+c);raw.append((ids,(span[0]-1,span[1]-1)))
        L_=RG.check_rows(raw,len(idx),NC);assert L_==Ln,(L_,Ln)
        Pk=rd.packed(raw,out_tail);S[idx]=rd.score(Pk,rt.base,len(idx),NC);print(f'span length {Ln}: {len(idx)} items scored',flush=True)
    json.dump(dict(scores=S.tolist(),items=[(it['event'],it['S']) for it in items],frame=FRAME),open(OUT/'w0_scores.json','w'));finish(items,S,OUT)
def _selftest():
    import tempfile
    from tokenizers import Tokenizer
    tf=HERE.parent.parent.parent/'data/tokenizer/llama3/tokenizer.json'
    if not tf.exists():print('tokenizer file not here: selftest of prepare skipped');tok=None
    else:tok=Tokenizer.from_file(str(tf))
    mk=lambda ev,A,items,oth=(),hard=False:dict(event=ev,A=A,s=.6,items=[dict(S_key=k.lower(),S=k) for k in items],other_places=[dict(surface=o,share=.2) for o in oth],A_via='gazetteer')
    pc=dict(main=[mk('E1','Mumbai',['Raj Kumar','Anil Shah'],['Pune']),mk('E2','Delhi',['Sita Rao'])],hard=[mk('E3','London',['Tom Hill'])])
    if tok:
        P=prepare(pc,tok,log=lambda *x:None);assert P['items'] and all(len(it['cands'])==NC and len(set(it['cands']))==NC and it['cands'][0]==it['A'] for it in P['items']),P['dropped']
        for it in P['items']:assert all(span_len(tok,FRAME.format(S='X'),' '+c)==it['span_len'] for c in it['cands']),it
        assert any(it['hard'] for it in P['items']) and all(not PC.same_place(c,it['A']) for it in P['items'] for c in it['cands'][1:])
        j=json.dumps(P);P2=prepare(pc,tok,log=lambda *x:None);assert json.dumps(P2)==j,'prepare is not deterministic'
        n=len(P['items'])
    else:n=3
    rng=np.random.default_rng(0);S=rng.normal(size=(n,NC));S[0,0]=S[0].max()+0.5     # item 0: A is 1st -> u = 1
    d=decide(S);assert d[0]['u']==1.0 and d[0]['drop_u'] and abs(d[0]['rank']-1)<1e-9
    S2=np.zeros((2,NC));S2[:,0]=-1;S2[0,5]=5.0;S2[1,:]=np.linspace(0,1,NC)[::-1]*0;d2=decide(S2);assert d2[0]['drop_margin'] and not d2[0]['drop_u'] and not d2[1]['drop_margin']      # top-1 (not A) leads by 6 nats; item 1: all equal except A lower
    td=pathlib.Path(tempfile.mkdtemp());items=P['items'] if tok else [dict(event='E',S='s',A='a',cands=['a']*NC,span_len=1,hard=False,S_key='s',prefix='q')]*n
    kept,dec=finish(items,S,td,log=lambda *x:None);assert (td/'w0_report.md').exists() and len(kept)<=n-1
    print('e57_w0 selftest ok:',n,'items',len(kept),'kept' )
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='experiments/exp07_write_ladder/e57_news/e57_lite');ap.add_argument('--prepare',action='store_true');ap.add_argument('--score',action='store_true');ap.add_argument('--selftest',action='store_true');ap.add_argument('--tokenizer',default='data/tokenizer/llama3/tokenizer.json');ap.add_argument('--rerun',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    OUT=pathlib.Path(a.out)
    if a.prepare:
        if (OUT/'w0_items.json').exists() and not a.rerun:print('w0_items.json exists (exit 8)');sys.exit(8)
        if not (OUT/'place_clean.json').exists():print('place_clean.json missing (job 227 first) (exit 7)');sys.exit(7)
        from tokenizers import Tokenizer
        P=prepare(json.load(open(OUT/'place_clean.json')),Tokenizer.from_file(a.tokenizer));(OUT/'w0_items.json').write_text(json.dumps(P,indent=1));print('items',len(P['items']),'dropped (short pool)',len(P['dropped']),'events',P['n_events'])
    if a.score:
        if (OUT/'w0_report.md').exists() and not a.rerun:print('w0_report.md exists (exit 8)');sys.exit(8)
        score(a)
if __name__=='__main__':main()
