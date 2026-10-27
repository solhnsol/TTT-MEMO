"""E57 ORCA v2 masks for the one-cycle write (CPU only; post-hoc, exploratory). Builds the mask file of the arms ORC2 (rank 7) and RAND2 (rank 8) in the format of e57_mask_stage (keys h<r>_<k> bool [nW] window sentences, ch<r>_<k> bool [nQ] chunk sentences, wlen<k>, clen<k>) over the sentence layout of the ABBREVIATION-AWARE splitting (e57_orc2.sentence_ends_v2, R1):
  ORC2  : fragments hidden by the ORC v2 rule (e57_orc2 variant 'v2': strict matcher of any question OR a partial form (R2) of a question in an article of its own event), for the window (chunks k-8 .. k-1) and for the current chunk k;
  RAND2 : random fragments with the SAME hidden token count in the window and in the chunk (e57_mask_stage.random_like: overshoot < one fragment; chunk part from the fragments with a later fragment and >= 4 tokens).
The arms read these ranks with SelMaskAll(..., guard=G) (per-query keep [CH, 8192+CH] with the local guard of G tokens) and the sentence ends of the same splitting (e57_pod_run.py --ends-v2).
--check prints what the guard G exposes: of the strict gold occurrences inside hidden fragments, the share whose START / END / WHOLE string lies in the last G tokens of its fragment (visible to the first G query tokens that follow), the effective hidden share (hidden tokens minus those last-G tokens), the hidden shares of window and chunk, the over-masking (fragments hidden only by a partial form or only by another event's answer), and the RAND2 matching.
  python3 e57_orca2_masks.py --bundle B --out O/e57_orca2_masks.npz [--check] [--guard 16]"""
import sys,json,re,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8
def make(bd,ends,UVm,n,seed=61):
    import e57_mask_stage as MS
    pos={f[1]:i for i,f in enumerate(bd['fr'])};hid=bd['hidden'];out={}
    for k in range(1,n):
        wsid,wst,wl=UVm.window_sents(ends,k,CH,NW);q=UVm.chunk_sents(ends,k,CH)
        if len(wl)==0:continue
        base=(k-NW)*CH;hw=np.asarray([hid[pos[base+int(s)]] for s in wst],bool);hc=np.asarray([hid[pos[k*CH+a]] for a,b in q],bool);cl=np.asarray([b-a for a,b in q],np.int32)
        rw,rc=MS.random_like(hw,hc,np.asarray(wl),cl,k,seed+100)
        out[f'wlen{k}']=np.asarray(wl,np.int32);out[f'clen{k}']=cl;out['h7_%d'%k]=hw;out['ch7_%d'%k]=hc;out['h8_%d'%k]=rw;out['ch8_%d'%k]=rc
    return out
def exposure(bd,items,B,tk,ids,guards=(4,16)):
    """guard = every query token always sees its previous N tokens, masked or not. A strict gold occurrence inside a hidden fragment F (tokens a .. b-1 of F, length l) is therefore visible to the OUTSIDE queries q = l, l+1, ... (the first tokens after F; the tokens of F itself see their own sentence anyway) as long as q-N <= its tokens: any gold token visible to n_any = clip(b+N-l, 0, N) outside queries, the WHOLE string (all tokens a .. b-1) to n_whole = clip(a+N-l+1, 0, N) outside queries. Also the effective hidden share (hidden tokens minus the last N tokens of each hidden fragment)."""
    fr=bd['fr'];hid=bd['hidden'];strict=bd['strict'];txt=bd['txt'];rxs=[]
    for it in items:
        q=it['q'];vs=[v for v in [q['canonical']]+list(q.get('variants',[]))+[q.get('fixed_surface','')] if v]
        rxs.append(re.compile('|'.join(r'(?<![\w])'+re.escape(v)+r'(?![\w])' for v in vs),re.I) if vs else None)
    tot=0;hidden_tokens=0;res={g:dict(any=0,whole=0,q_any=0,q_whole=0) for g in guards};eff={g:0 for g in guards}
    for j,(c,s,l) in enumerate(fr):
        if not hid[j]:continue
        hidden_tokens+=l
        for g in guards:eff[g]+=max(0,l-g)
        if not strict[j]:continue
        t=txt[j];cum=None
        for rx in rxs:
            if rx is None:continue
            for m in rx.finditer(t):
                tot+=1
                if cum is None:cum=[len(tk.decode([int(x) for x in ids[s:s+jj+1]])) for jj in range(l)]
                a_=next(jj for jj in range(l) if cum[jj]>m.start());b_=next(jj for jj in range(l) if cum[jj]>=m.end())+1             # token index range [a_, b_) overlapping the match
                for g in guards:
                    na=min(max(b_+g-l,0),g);nw=min(max(a_+g-l+1,0),g)
                    res[g]['any']+=1 if na>0 else 0;res[g]['whole']+=1 if nw>0 else 0;res[g]['q_any']+=na;res[g]['q_whole']+=nw
    return dict(gold_occurrences_in_hidden_fragments=tot,hidden_tokens=hidden_tokens,by_guard={str(g):dict(share_with_a_gold_token_visible_to_outside_queries=res[g]['any']/tot if tot else None,share_with_the_whole_string_visible_to_one_outside_query=res[g]['whole']/tot if tot else None,occurrences_any=res[g]['any'],occurrences_whole=res[g]['whole'],outside_query_tokens_seeing_a_gold_token_total=res[g]['q_any'],outside_query_tokens_seeing_the_whole_string_total=res[g]['q_whole'],mean_outside_queries_per_exposed_occurrence=res[g]['q_any']/res[g]['any'] if res[g]['any'] else None,mean_outside_queries_per_occurrence=res[g]['q_any']/tot if tot else None,effective_hidden_share_of_hidden_tokens=eff[g]/hidden_tokens if hidden_tokens else None) for g in guards})
def context_loss(bd,n,Ns=(1,4,16)):
    """guard 0: query tokens whose previous N tokens are hidden from them (hidden fragments other than the query's own fragment): total masked previous tokens, number of affected queries, share of all query tokens"""
    fr=bd['fr'];hid=bd['hidden'];T=n*CH;fid=np.full(T,-1,np.int64);hid_tok=np.zeros(T,bool)
    for j,(c,s_,l) in enumerate(fr):fid[s_:s_+l]=j;hid_tok[s_:s_+l]=hid[j]
    out={}
    for N in Ns:
        tot=np.zeros(T,np.int64)
        for d in range(1,N+1):
            m=np.zeros(T,bool);m[d:]=hid_tok[:-d]&(fid[:-d]!=fid[d:]);tot+=m
        out[str(N)]=dict(masked_previous_tokens_total=int(tot.sum()),affected_queries=int((tot>0).sum()),affected_query_share=float((tot>0).mean()),mean_masked_previous_tokens_per_affected_query=float(tot[tot>0].mean()) if (tot>0).any() else None,share_of_all_previous_tokens=float(tot.sum()/(N*T)))
    return out
def verify_keep(masks,ends,UVm,guard,ks):
    """SelMaskAll.keep at the given guard on the real masks: hidden window / hidden chunk keys visible to queries OUTSIDE their own sentence (must be 0 at guard 0); visible inside the own sentence and on the diagonal (by design)"""
    import e57_mask_stage as MS
    arm=MS.SelMaskAll('SELC1',7,masks,ends,UVm,log=lambda *x:None,guard=guard);W=NW*CH;out=dict(outside_pairs=0,own_sentence_pairs=0,diagonal=0,hidden_window_tokens=0,hidden_chunk_tokens=0)
    for k in ks:
        keep,info=arm.keep(k,None,None,None,None,None);q=UVm.chunk_sents(ends,k,CH);csid=np.full(CH,-1)
        for i,(a_,b_) in enumerate(q):csid[a_:b_]=i
        hc=np.asarray(masks[f'ch7_{k}'],bool);hck=np.zeros(CH,bool)
        for i,(a_,b_) in enumerate(q):
            if hc[i]:hck[a_:b_]=True
        wsid,wst,wl=UVm.window_sents(ends,k,CH,NW);hw=np.asarray(masks[f'h7_{k}'],bool);hk=np.zeros(W,bool)
        for i,(s_,l_) in enumerate(zip(wst,wl)):
            if hw[i]:hk[int(s_):int(s_)+int(l_)]=True
        allh=np.concatenate([hk,hck]);qr=np.arange(CH)
        vis=keep[:,allh]                                  # visibility of hidden keys
        cols=np.nonzero(allh)[0];own=np.zeros(vis.shape,bool);dist=np.zeros(vis.shape,bool)
        for ci,col in enumerate(cols):
            if col>=W:
                own[:,ci]=(csid==csid[col-W]);dist[col-W,ci]=True
        out['outside_pairs']+=int((vis&~own&~dist).sum());out['own_sentence_pairs']+=int((vis&own&~dist).sum());out['diagonal']+=int((vis&dist).sum());out['hidden_window_tokens']+=int(hk.sum());out['hidden_chunk_tokens']+=int(hck.sum())
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--tokenizer');ap.add_argument('--check',action='store_true');ap.add_argument('--guard',type=int,default=16);ap.add_argument('--cycles',type=int,default=1);a=ap.parse_args()
    ROOT=HERE.parents[1];import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_bank as B,e57_orc2 as O2
    from tokenizers import Tokenizer
    tf=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);tk=Tokenizer.from_file(tf)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH
    P=len(ids)//CH*CH;ids_c=np.tile(ids,a.cycles);n=len(ids_c)//CH           # several cycles: the stream repeated, fragments / hidden decisions per chunk of the repeated stream
    bd=O2.build('v2',ids_c,tk,tf,items,iv,UVm,B,period=P);masks=make(bd,bd['ends'],UVm,n);np.savez_compressed(a.out,**masks)
    s=O2.summarize_variant(bd,n);print(f"ORC2 (v2) hidden tokens {s['hidden_tokens']} ({s['hidden_token_share_of_stream']:.3f}); window share mean {s['window_hidden_share']['mean']:.3f} (min {s['window_hidden_share']['min']:.3f} max {s['window_hidden_share']['max']:.3f}); chunk share mean {s['chunk_hidden_share']['mean']:.3f}; partial-only {s['share_partial_only']:.3f}, other-event-only {s['share_strict_cross_event_only']:.3f}, related to own event {s['share_related_to_own_event_answer']:.3f}")
    # RAND2 matching over the chunks
    dw=[];dc=[];hw_all=[];hc_all=[]
    for k in range(1,n):
        if f'h7_{k}' not in masks:continue
        wl=masks[f'wlen{k}'];cl=masks[f'clen{k}'];dw.append(int(wl[masks[f'h8_{k}']].sum())-int(wl[masks[f'h7_{k}']].sum()));dc.append(int(cl[masks[f'ch8_{k}']].sum())-int(cl[masks[f'ch7_{k}']].sum()));hw_all.append(int(wl[masks[f'h7_{k}']].sum()));hc_all.append(int(cl[masks[f'ch7_{k}']].sum()))
    match=dict(window_token_diff_mean=float(np.mean(dw)),window_token_diff_abs_max=int(np.max(np.abs(dw))),chunk_token_diff_mean=float(np.mean(dc)),chunk_token_diff_abs_max=int(np.max(np.abs(dc))),orc2_window_tokens_mean=float(np.mean(hw_all)),orc2_chunk_tokens_mean=float(np.mean(hc_all)),rand2_window_tokens_mean=float(np.mean(hw_all)+np.mean(dw)),rand2_chunk_tokens_mean=float(np.mean(hc_all)+np.mean(dc)))
    print('RAND2 matching',json.dumps(match))
    if a.check:
        ex=exposure(bd,items,B,tk,ids_c);print('guard exposure of strict gold occurrences in hidden fragments:',json.dumps(ex))
        cl=context_loss(bd,n);print('guard 0: local context loss (previous N tokens hidden from the query):',json.dumps(cl))
        vk={g:verify_keep(masks,bd['ends'],UVm,g,range(8,n)) for g in (0,4,16)};print('keep matrix check (hidden keys visible to outside queries):',json.dumps(vk))
        pathlib.Path(str(a.out)+'.check.json').write_text(json.dumps(dict(variant=s,rand2_matching=match,guard_exposure=ex,guard0_context_loss=cl,keep_check=vk,n_chunks=len(hw_all)),indent=1,default=float))
if __name__=='__main__':main()
