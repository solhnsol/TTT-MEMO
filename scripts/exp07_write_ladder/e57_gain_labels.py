"""E57 which loss rows have a high window gain, labelled by the news questions (CPU only; new file; numbers only). Rows of the chunks 8..279 (eligible loss rows; row i predicts token p = chunk*1024 + i + 1). Window gain = W0 band-512 NLL - W0 window NLL (e57_band_W0.npz) of the row. Label of the TARGET token (one of four, first match):
  A     the token lies inside a hit of the answer string of one of the 42 bank questions (all surface forms: canonical + variants + fixed_surface, word-boundary, case-insensitive regex, as e57_mask_coverage / e57_occurrence_count) in an article of that question's event
  S     the token lies inside a hit of the subject string S of one of the questions in an article of its event (not A)
  SENT  the token lies in a sentence (e55_uv chunk sentences) that holds an A or S hit of a question of the same event, but not inside the hit (not A, not S)
  OTHER everything else
Groups: SEL = the banded rows of the exact GAINQ512 p=.26 mask, NOSEL = other eligible rows, TOP1 / TOP10 = the 1 % / 10 % eligible rows with the largest window gain. Per group and label: rows, share of the group's rows, share of the group's gain sum (and of the positive-part gain sum), enrichment = share of rows in the group / share of rows among all eligible rows. Per question: rows with the answer string A of that question, how many are selected, the fraction selected, the mean window gain of those rows, appearances of A in the stream.
  .venv/bin/python scripts/exp07_write_ladder/e57_gain_labels.py --bundle B --band-npz e57_band_W0.npz --masks M.npz --out O.json"""
import sys,json,re,pathlib,argparse,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024
def forms(q):
    vs=[]
    for v in [q['canonical']]+list(q.get('variants',[]))+[q.get('fixed_surface','')]:
        if v and v.lower() not in {x.lower() for x in vs}:vs.append(v)
    return vs
def hits(text,offs,rx):
    out=[]
    for m in rx.finditer(text):
        a,b=m.span();i0=int(np.searchsorted(offs,a,side='right')-1);i1=int(np.searchsorted(offs,b,side='left')-1);out.append((i0,max(i0,i1)))
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--band-npz');ap.add_argument('--masks');ap.add_argument('--out');ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);tk=UVm.tokenizer(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits_=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(tk,ids);N=len(ids)
    strs=[tk.decode([int(t)]) for t in ids];offs=np.cumsum([0]+[len(s) for s in strs]);full=''.join(strs)
    lab=np.zeros(N,np.int8)   # 0 other, 1 SENT, 2 S, 3 A
    qA=np.full(N,-1,np.int16);hitsA=collections.defaultdict(list);sent_has=np.zeros(N,bool)
    # sentence id per token (chunk-local sentences)
    sent_of=np.full(N,-1,np.int64);ns=0
    for k in range(N//CH):
        for (s_,e_) in UVm.chunk_sents(ends,k,CH):sent_of[k*CH+s_:k*CH+e_]=ns;ns+=1
    sentflag=np.zeros(ns+1,bool)
    for qi,it in enumerate(items):
        q=it['q'];rxA=re.compile('|'.join(r'(?<![\w])'+re.escape(v)+r'(?![\w])' for v in forms(q)),re.I);S=q.get('S') or it['event'];rxS=re.compile(r'(?<![\w])'+re.escape(S)+r'(?![\w])',re.I)
        for (s0,e0,ev,rid) in iv:
            e0=min(e0,N)
            if ev!=it['event'] or s0>=N:continue
            txt=full[offs[s0]:offs[e0]];of=offs[s0:e0+1]-offs[s0]
            for (i0,i1) in hits(txt,of,rxA):
                for p in range(s0+i0,s0+i1+1):lab[p]=3;qA[p]=qi
                hitsA[qi].append(s0+i0);sentflag[sent_of[s0+i0:s0+i1+1][sent_of[s0+i0:s0+i1+1]>=0]]=True
            for (i0,i1) in hits(txt,of,rxS):
                for p in range(s0+i0,s0+i1+1):
                    if lab[p]<2:lab[p]=2
                so=sent_of[s0+i0:s0+i1+1];sentflag[so[so>=0]]=True
    insent=(sent_of>=0)&sentflag[np.clip(sent_of,0,ns)]
    lab=np.where((lab==0)&insent,1,lab).astype(np.int8)
    B=np.load(a.band_npz);bi=list(B['bands']).index(512);nb=B['nll_band'][bi];nw=B['nll_win'];lm=B['lm'];M=np.load(a.masks,allow_pickle=True)
    rows=[]
    for k in range(8,280):
        sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(UVm.chunk_sents(ends,k,CH)):sid[s_:e_]=i
        el=CM.rows_eligible(lm[k],sid)&lm[k];qb=CM.up(M[f'qb_GAINQ512_{k}'],CH);r=np.nonzero(el)[0];p=k*CH+r+1
        rows.append(np.stack([p,(nb[k]-nw[k])[r].astype(np.float64)*1.0,qb[r].astype(float)],1))
    R=np.concatenate(rows);pos=R[:,0].astype(np.int64);g=R[:,1];sel=R[:,2]>.5;lb=lab[pos];qq=qA[pos]
    names=['OTHER','SENT','S','A'];allrows=len(g);base={n:float((lb==i).mean()) for i,n in enumerate(names)}
    order=np.argsort(-g);t1=np.zeros(allrows,bool);t1[order[:int(round(.01*allrows))]]=True;t10=np.zeros(allrows,bool);t10[order[:int(round(.10*allrows))]]=True
    grp={'SEL':sel,'NOSEL':~sel,'TOP1':t1,'TOP10':t10,'ALL':np.ones(allrows,bool)};out={'groups':{}}
    for gn,m in grp.items():
        gs=g[m].sum();gp=np.maximum(g[m],0).sum();d={}
        for i,n in enumerate(names):
            s=m&(lb==i);d[n]=dict(rows=int(s.sum()),row_share=float(s.sum()/max(m.sum(),1)),gain_sum_share=float(g[s].sum()/gs) if gs!=0 else None,positive_gain_share=float(np.maximum(g[s],0).sum()/gp) if gp>0 else None,mean_gain=float(g[s].mean()) if s.any() else None,enrichment=float(s.sum()/max(m.sum(),1)/base[n]) if base[n]>0 else None)
        out['groups'][gn]=dict(rows=int(m.sum()),mean_gain=float(g[m].mean()),labels=d)
    perq={}
    for qi,it in enumerate(items):
        s=qq==qi;n=int(s.sum());perq[it['id']]=dict(event=it['event'],type=it['type'],rows_with_A=n,selected=int((s&sel).sum()),fraction_selected=float((s&sel).sum()/n) if n else None,mean_gain=float(g[s].mean()) if n else None,appearances_in_stream=len(hitsA[qi]))
    out.update(label_rule='first match: A (inside an answer-string hit of a bank question, same event) > S (subject string) > SENT (sentence holds an A or S hit of the same event) > OTHER',eligible_rows=allrows,label_share_all=base,gain_definition='W0 band-512 NLL - W0 window NLL',per_question=perq)
    json.dump(out,open(a.out,'w'),indent=1)
    print('rows',allrows,'label shares',{k:round(v,4) for k,v in base.items()})
    for gn in ('ALL','SEL','NOSEL','TOP1','TOP10'):
        print(gn,out['groups'][gn]['rows'],' '.join(f"{n}: share {v['row_share']:.3f} gain {v['gain_sum_share'] if v['gain_sum_share'] is None else round(v['gain_sum_share'],3)} enr {v['enrichment'] if v['enrichment'] is None else round(v['enrichment'],2)}" for n,v in out['groups'][gn]['labels'].items()))
    fr=[v['fraction_selected'] for v in perq.values() if v['fraction_selected'] is not None];print('per-question fraction selected: min %.3f median %.3f max %.3f'%(min(fr),float(np.median(fr)),max(fr)),'questions with A rows',len(fr))
if __name__=='__main__':main()
