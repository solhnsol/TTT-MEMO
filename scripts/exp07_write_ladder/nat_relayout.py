"""E51 natural-text relayout: same persons, same exposure units (text, label, round, per-person order and counts), new TOKEN POSITIONS for a chosen set of persons.
Base world (e.g. nat_9103_spread.json) -> new world json whose contrast persons are re-timed: P1 persons get condition C1 (every previous fact exposure <= 4096 tokens before the next one = inside the 8192 window),
P2 persons C2 (consecutive fact exposures >= 12288 apart, and their name-only units sit right after the previous fact exposure so none lies within 8192 tokens before the next fact = outside the window).
Layout relA: P1 = C1, P2 = C2;  layout relB: flipped. All other persons keep their base start position (moved to the next free slot only on collision). Background paragraphs are the base world's, taken in order (same cycling as build.py).
Contrast persons = persons with >= 2 fact units whose C2 timeline fits before the end of chunk 52 (EXPO_END - margin); persons that do not fit (e.g. 10 fact units) stay at base positions and are listed. Split P1/P2: sorted by (n_fact, id), alternating, deterministic.
Targets carry margins (C1 3500, C2 12800 tokens); the ACTUAL gaps are re-measured after building (per-person table, PASS/FAIL), with the real tokenizer on the server (the proxy only tests the logic).
No private data, no GPU. Usage: nat_relayout.py --base nat_9103_spread.json --out-dir DIR [--tokenizer-file F] [--require-real] [--layouts relA,relB] [--max-ntok 14]   |   --selftest"""
import os,sys,re,json,copy,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_world as NW,nat_pass as NP,nat_filler as NF
import e51_nat_table as T
CH=1024;NCH=63;EXPO_END=53*CH;MARGIN=600;WIN=8192
C1_T=3500;C2_T=12800;C1_MAX=4096;C2_MIN=12288
def ntok_fn(tk):
    if tk is not None:return lambda s:len(tk.encode(s,add_special_tokens=False).ids)
    import nat_dryrun as D
    px=D.Proxy();return lambda s:len(px.encode(s)[0])
def persons_of(w):
    c={}
    for i,x in enumerate(w['texts']):
        if x['kind']=='expo':c.setdefault(x['t'],[]).append(i)
    return c
def timeline(w,idx,cond,lens,a0):
    """target starts for the units idx (texts indices, original order) of one person; returns (starts, span_end)"""
    starts=[];pos=a0;prevf=None;lab=[w['texts'][i]['label'] for i in idx];n=len(idx)
    if cond=='C1':
        k=0
        while k<n:
            j=k
            while j+1<n and lab[j+1]!='fact':j+=1   # unit k .. j: a fact (or leading name-only) followed by its name-only run
            # run idx[k..j]; spacing keeps the distance to the NEXT fact inside C1_T
            nxt=j+1;run=range(k,j+1);tot=sum(lens[i] for i in (idx[m] for m in run))
            sp=max(0,min(300,(C1_T-tot)//(len(run)+1)))
            for m in run:starts.append(pos);pos+=lens[idx[m]]+2+sp
            k=j+1
        return starts,pos
    # C2: facts >= C2_T apart; name-only run goes tight right after the previous fact (or leads the first fact by >= 8192+600)
    pos=a0;fi=[m for m in range(n) if lab[m]=='fact'];first=fi[0];lead=list(range(first))
    for m in lead:starts.append(pos);pos+=lens[idx[m]]+2+50
    if lead:pos=max(pos,starts[-1]+lens[idx[lead[-1]]]+WIN+600)
    prev_f=None;m=first
    while m<n:
        if lab[m]=='fact':
            st=pos if prev_f is None else max(pos,prev_f+C2_T);starts.append(st);prev_f=st;pos=st+lens[idx[m]]+2+50;m+=1
        else:
            starts.append(pos);pos+=lens[idx[m]]+2+50;m+=1
    return starts,pos
def plan(w,lens,P,base_start,max_ntok_c2=14):
    """layout-independent plan: contrast persons, split, and per person the common start T of its LAST unit (equal in relA and relB, so recency to the readout is matched).
    T = max(base start of the last unit, earliest start that lets the C2 timeline begin at token 600); persons whose C2 timeline does not fit before EXPO_END - MARGIN are skipped. Reservations are checked under BOTH conditions."""
    texts=w['texts'];elig=[];skip=[];rel={}
    for t,idx in sorted(P.items()):
        nf=sum(texts[i]['label']=='fact' for i in idx)
        if nf<2:continue
        r2,_=timeline(w,idx,'C2',lens,0);r1,_=timeline(w,idx,'C1',lens,0);Tmin=600+r2[-1];Tmax=EXPO_END-MARGIN-lens[idx[-1]]-2
        if Tmin<=Tmax and len(idx)<=max_ntok_c2:elig.append((nf,t));rel[t]=(r1,r2,Tmin,Tmax)
        else:skip.append((nf,t))
    elig.sort();ids=[t for _,t in elig];P1=ids[0::2];P2=ids[1::2];res=[]
    def free(a,b):return all(b<=s_ or a>=e for s_,e in res)
    T={}
    for t in sorted(ids,key=lambda t:-len(P[t])):
        idx=P[t];r1,r2,Tmin,Tmax=rel[t];T0=min(max(base_start[idx[-1]],Tmin),Tmax)
        for tries in range(400):
            Tc=T0+(250*(tries//2) if tries%2==0 else -250*((tries+1)//2))
            if not (Tmin<=Tc<=Tmax):continue
            iv=[(Tc-r[-1]+x,Tc-r[-1]+x+lens[i]+2) for r in (r1,r2) for x,i in zip(r,idx)]
            if all(free(a_,b_) for a_,b_ in iv):break
        else:raise AssertionError(f'no free anchor for person {t}')
        T[t]=Tc
        for r in (r1,r2):
            for x,i in zip(r,idx):res.append((Tc-r[-1]+x,Tc-r[-1]+x+lens[i]+2))
    return dict(P1=P1,P2=P2,skipped=[t for _,t in skip],T=T,res=res)
def relayout(w,layout,enc_len,bgs,base_start,max_ntok_c2=14,lens=None,pl=None):
    texts=w['texts'];P=persons_of(w);lens=lens or {i:enc_len(texts[i]['text']) for i in range(len(texts)) if texts[i]['kind']=='expo'}
    pl=pl or plan(w,lens,P,base_start,max_ntok_c2);P1,P2=pl['P1'],pl['P2']
    cond={t:'C1' for t in P1};cond.update({t:'C2' for t in P2})
    if layout=='relB':cond={t:('C2' if c=='C1' else 'C1') for t,c in cond.items()}
    placed={};res=list(pl['res'])
    for t in cond:
        idx=P[t];st,_=timeline(w,idx,cond[t],lens,0);off=pl['T'][t]-st[-1]
        for s_,i in zip(st,idx):placed[i]=s_+off
    def free(a,b):return all(b<=s_ or a>=e for s_,e in res)
    for i in sorted(lens,key=lambda i:base_start[i]):
        if i in placed:continue
        s_=base_start[i]
        while not free(s_,s_+lens[i]+2):s_+=25
        placed[i]=s_;res.append((s_,s_+lens[i]+2))
    order=sorted(placed,key=lambda i:placed[i]);out=[];cur=0;bi=0;bgl=[enc_len(b)+2 for b in bgs]
    def take():
        nonlocal bi;x=bgs[bi%len(bgs)];n=bgl[bi%len(bgs)];bi+=1;return x,n
    for i in order:
        while cur+bgl[bi%len(bgs)]<=placed[i]+60:
            x,n=take();out.append({'kind':'bg','text':x});cur+=n
        out.append(copy.deepcopy(texts[i]));cur+=lens[i]+2
    while cur<NCH*CH+3000:
        x,n=take();out.append({'kind':'bg','text':x});cur+=n
    nw=copy.deepcopy(w);nw['texts']=out;nw['layout']=layout;nw['relayout']=dict(base_layout=w.get('layout'),P1=P1,P2=P2,skipped=pl['skipped'],cond=cond,C1_target=C1_T,C2_target=C2_T,bg_reused=max(0,bi-len(bgs)),last_unit_anchor={str(t):int(v) for t,v in pl['T'].items()})
    nw['stats']={'note':'stats of the base world are stale after relayout; recompute with e51_nat_table.py'};return nw
def gap_table(nw,tk,enc_ends):
    text,_,spans=NW.realize(nw,False);sp=NP.tok_spans(spans,enc_ends(text));rows=[]
    by={}
    for s in sp:by.setdefault(s['t'],[]).append(s)
    cond=nw['relayout']['cond'];ok=True
    for t,us in sorted(by.items()):
        f=[u['tok_start'] for u in us if u['label']=='fact'];g=[b-a for a,b in zip(f,f[1:])]
        no=[u['tok_start'] for u in us if u['label']!='fact'];nw8=[sum(1 for n_ in no if 0<f_-n_<WIN) for f_ in f]
        c=cond.get(t,'-');good=True
        if c=='C1':good=all(x<=C1_MAX for x in g)
        if c=='C2':good=all(x>=C2_MIN for x in g) and all(x==0 for x in nw8[1:]) and len(f)>=2
        if c!='-':ok&=good
        rows.append((t,c,len(f),g,nw8,'PASS' if good else 'FAIL',max([u['tok_end'] for u in us])))
    return rows,ok
def fmt(rows):
    o=['| person | condition | n_fact | fact-to-fact gaps (tokens) | name-only units within 8192 before each fact | check | last unit end |','|---|---|---|---|---|---|---|']
    for t,c,n,g,nw8,ok,end in rows:
        if c!='-':o.append(f'| {t} | {c} | {n} | {g} | {nw8} | {ok} | {end} |')
    return '\n'.join(o)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--out-dir');ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--require-real',action='store_true');ap.add_argument('--layouts',default='relA,relB');ap.add_argument('--selftest',action='store_true');ap.add_argument('--max-ntok',type=int,default=14);a=ap.parse_args()
    if a.selftest:selftest();return
    import nat_dryrun as D
    try:tk=D.load_tok(a.tokenizer_file)
    except Exception as e:print('real tokenizer not usable:',repr(e)[:100]);tk=None
    assert tk is not None or not a.require_real,'--require-real: no real tokenizer'
    print('tokenizer:','REAL' if tk else 'PROXY (logic test only; gaps are NOT final)')
    assert a.base and Path(a.base).exists(),f'--base file not found: {a.base}';w=NW.load_world(a.base);run(w,tk,a.out_dir,a.layouts.split(','),a.max_ntok)
def run(w,tk,out_dir,layouts,max_ntok=14):
    enc_len=ntok_fn(tk);ends=lambda text:[b for a_,b in T.encode_offs(tk,text)]
    text,_,spans=NW.realize(w,False);sp=NP.tok_spans(spans,ends(text));ei=[i for i,x in enumerate(w['texts']) if x['kind']=='expo'];assert len(ei)==len(sp)
    base_start={i:s['tok_start'] for i,s in zip(ei,sp)};bgs=[x['text'] for x in w['texts'] if x['kind']=='bg'];res={}
    lens={i:enc_len(w['texts'][i]['text']) for i in ei};pl=plan(w,lens,persons_of(w),base_start,max_ntok);rx0=NF.forbid_regex(w);print('background paragraphs with a forbidden string in the BASE world (same pool, property of the base):',sum(1 for b in bgs if rx0.search(b)),'of',len(bgs))
    _,sm0,_=T.table_of(w,T.encode_offs(tk,text));print(f"O mask size (BASE {w.get('layout')}, all persons): sum O tokens {sm0['sum_O']}, chunks with O {sm0['chunks_with_O']}, eligible chunks {sm0['eligible_chunks']}, answer tokens {sm0['answer_tokens']}, sum s_cross {sm0['sum_s_cross']}, sum s_same {sm0['sum_s_same']}, sum n_out {sm0['sum_n_out']}")
    for lay in layouts:
        nw=relayout(w,lay,enc_len,bgs,base_start,max_ntok,lens,pl);rows,ok=gap_table(nw,tk,ends)
        before=sorted((x['t'],x['label'],x['text']) for x in w['texts'] if x['kind']=='expo');after=sorted((x['t'],x['label'],x['text']) for x in nw['texts'] if x['kind']=='expo');assert before==after,'exposure units changed'
        rx=NF.forbid_regex(nw);bad=sum(1 for x in nw['texts'] if x['kind']=='bg' and rx.search(x['text']))
        r=nw['relayout'];print(f"## {lay}: P1 {r['P1']} P2 {r['P2']} skipped (do not fit C2) {r['skipped']}; bg paragraphs reused beyond the base pool: {r['bg_reused']}; forbidden strings in background: {bad}; last exposure end token <= {EXPO_END}: {max(x[6] for x in rows)<=EXPO_END}");print(fmt(rows));print('ALL PASS' if ok else 'SOME FAIL');res[lay]=ok;_,sm,_=T.table_of(nw,T.encode_offs(tk,NW.realize(nw,False)[0]));print(f"O mask size ({lay}, all persons): sum O tokens {sm['sum_O']}, chunks with O {sm['chunks_with_O']}, eligible chunks {sm['eligible_chunks']}, answer tokens {sm['answer_tokens']}, sum s_cross {sm['sum_s_cross']}, sum s_same {sm['sum_s_same']}, sum n_out {sm['sum_n_out']}")
        if out_dir:
            Path(out_dir).mkdir(parents=True,exist_ok=True);p=Path(out_dir)/f"nat_{w['seed']}_{lay}.json";p.write_text(json.dumps(nw,ensure_ascii=False));print('wrote',p)
    return res
def selftest():
    import nat_dryrun as D
    rng=np.random.default_rng(0);w={'seed':9999,'layout':'spread','pairs':1,'ctrl_persons':[],'ctrl_places':[],'pseudo_persons':[],'pseudo_places':[],'texts':[],
      'triples':[{'id':i,'pair':f'p{i//2}','X':f'Xa{i}','XF':f'Xf{i}','XS':f'Xs{i}','P':f'Pl{i}','Q':None,'E':None} for i in range(6)]}
    def bg(n,j):return {'kind':'bg','text':' '.join(f'bg{j}w{i}' for i in range(n))}
    plan={}
    for t,labs in {0:'FnF',1:'FF',2:'F',3:'FnnFF',4:'nFF',5:'F'}.items():plan[t]=list(labs)
    units=[(r,t,l) for t,ls in plan.items() for r,l in enumerate(ls)];units.sort(key=lambda u:(u[0],u[1]))
    for j in range(520):w['texts'].append(bg(100,j))
    ins=sorted(rng.choice(np.arange(10,260),len(units),replace=False))
    for k,(r,t,l) in zip(ins[::-1],units[::-1]):w['texts'].insert(int(k),{'kind':'expo','t':t,'round':r+1,'label':'fact' if l=='F' else 'name_only','text':f'@X{t}@ was born in Pl{t} k{r}.'})
    # restore per-person order: units were inserted in reverse so the order within a person follows insertion index -> sort by position of unit index
    res=run(w,None,None,['relA','relB'])
    assert all(res.values()),res;print('nat_relayout SELFTEST OK (proxy tokenizer)')
if __name__=='__main__':main()
