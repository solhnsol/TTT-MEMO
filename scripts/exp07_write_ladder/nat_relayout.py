"""E51 natural-text relayout: same persons, same exposure units (text, label, round, per-person order and counts), new TOKEN POSITIONS for a chosen set of persons.
Base world (e.g. nat_9103_spread.json) -> new world json whose contrast persons are re-timed: P1 persons get condition C1 (every previous fact exposure <= 4096 tokens before the next one = inside the 8192 window),
P2 persons C2 (consecutive fact exposures >= 12288 apart, and their name-only units sit right after the previous fact exposure so none lies within 8192 tokens before the next fact = outside the window).
Layout relA: P1 = C1, P2 = C2;  layout relB: flipped. All other persons keep their base start position (moved to the next free slot only on collision). Background paragraphs are the base world's, taken in order (same cycling as build.py).
Contrast persons = persons with >= 2 fact units whose C2 timeline fits before the end of chunk 52 (EXPO_END - margin); persons that do not fit (e.g. 10 fact units) stay at base positions and are listed. Split P1/P2: greedy by descending n_fact into the arm with the smaller fact-unit sum (deterministic), so the two arms have balanced fact counts.
Targets carry margins (C1 3500, C2 12800 tokens); the ACTUAL gaps are re-measured after building (per-person table, PASS/FAIL), with the real tokenizer on the server (the proxy only tests the logic).
No private data, no GPU. Usage: nat_relayout.py --nat 9103:spread,9101:spread,9104:spread,9105:spread [--nat-dir DIR] --out-dir DIR (or --base FILE for one world; old form: --base nat_9103_spread.json --out-dir DIR) [--tokenizer-file F] [--require-real] [--layouts relA,relB] [--max-ntok 14]   |   --selftest"""
import os,sys,re,json,copy,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_world as NW,nat_pass as NP,nat_filler as NF,nat_signal as NS
import e51_nat_table as T
CH=1024;NCH=63;EXPO_END=53*CH;MARGIN=600;WIN=8192
DRIFT_UP=60;DRIFT_DN=450;C2_NO_GAP=9800;C1_T=3500;C2_T=12800;C1_MAX=4096;C2_MIN=12288
LEN_SCALE=float(os.environ.get('E51_LEN_SCALE','1'))   # test hook: scale the PROXY token counts (real tokenizers give ~1.2-1.3x the proxy counts)
def ntok_fn(tk):
    if tk is not None:return lambda s:len(tk.encode(s,add_special_tokens=False).ids)
    import nat_dryrun as D
    px=D.Proxy();return lambda s:int(round(len(px.encode(s)[0])*LEN_SCALE))
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
    if lead:pos=max(pos,starts[-1]+lens[idx[lead[-1]]]+C2_NO_GAP)
    prev_f=None;m=first;last_no_end=-10**9
    if lead:last_no_end=starts[-1]+lens[idx[lead[-1]]]+2
    while m<n:
        if lab[m]=='fact':
            st=pos if prev_f is None else max(pos,prev_f+C2_T,last_no_end+C2_NO_GAP);starts.append(st);prev_f=st;pos=st+lens[idx[m]]+2+50;m+=1
        else:
            starts.append(pos);last_no_end=pos+lens[idx[m]]+2;pos+=lens[idx[m]]+2+50;m+=1
    return starts,pos
def align_c1(w,idx,st,lens):
    """C1 chunk alignment: where two consecutive fact units start in the SAME 1024-token chunk, shift every unit before the later one earlier so that the unit just before it ends before the chunk boundary (then the earlier fact starts in an earlier chunk = s_cross instead of s_same; margins DRIFT_UP/DRIFT_DN cover the shift the background fill adds between the planned and the realized start). The last unit never moves."""
    st=list(st);fi=[m for m,i in enumerate(idx) if w['texts'][i]['label']=='fact']
    for k in range(len(fi)-1,0,-1):
        c,p=fi[k],fi[k-1]
        if (st[p]+DRIFT_UP)//CH>=(st[c]-DRIFT_DN)//CH:      # not safely in different chunks once the background fill has moved the units by its drift
            X=((st[c]-DRIFT_DN)//CH)*CH;d=max(0,st[c-1]+lens[idx[c-1]]+2-(X-550))
            for m in range(c):st[m]-=d
    return st
def abs_times(w,idx,cond,lens,T,align=True):   # align = apply the C1 chunk alignment
    st,_=timeline(w,idx,cond,lens,0);off=T-st[-1];st=[x+off for x in st]
    return align_c1(w,idx,st,lens) if (cond=='C1' and align) else st
def plan(w,lens,P,base_start,max_ntok_c2=14,align=True):
    """layout-independent plan: contrast persons, split, and per person the common start T of its LAST unit (equal in relA and relB, so recency to the readout is matched).
    T = max(base start of the last unit, earliest start that lets the C2 timeline begin at token 600); persons whose C2 timeline does not fit before EXPO_END - MARGIN are skipped. Reservations are checked under BOTH conditions."""
    texts=w['texts'];elig=[];skip=[];rel={}
    for t,idx in sorted(P.items()):
        nf=sum(texts[i]['label']=='fact' for i in idx)
        if nf<2:continue
        r2,_=timeline(w,idx,'C2',lens,0);r1,_=timeline(w,idx,'C1',lens,0);Tmin=600+r2[-1];Tmax=EXPO_END-MARGIN-lens[idx[-1]]-2
        if Tmin<=Tmax and len(idx)<=max_ntok_c2:elig.append((nf,t));rel[t]=(r1,r2,Tmin,Tmax)
        else:skip.append((nf,t))
    elig.sort();ids=[t for _,t in elig];P1,P2=[],[];s1=s2=0
    for nf_,t in sorted(elig,key=lambda x:(-x[0],x[1])):   # balance the sum of fact units; equal counts (+-1) as tie-break
        if (s1,len(P1))<=(s2,len(P2)):P1.append(t);s1+=nf_
        else:P2.append(t);s2+=nf_
    P1.sort();P2.sort();res=[]
    def free(a,b):return all(b<=s_ or a>=e for s_,e in res)
    T={};aligned={};dropped=[];diag={}
    for t in sorted(ids,key=lambda t:(rel[t][3]-rel[t][2],-len(P[t]))):      # tightest anchor range first
        idx=P[t];r1,r2,Tmin,Tmax=rel[t];T0=min(max(base_start[idx[-1]],Tmin),Tmax);cands=sorted(range(Tmin,Tmax+1,25),key=lambda x:abs(x-T0));found=None
        for al in ((True,False) if align else (False,)):
            for Tc in cands:
                iv=[(x,x+lens[i]+2) for cd in ('C1','C2') for x,i in zip(abs_times(w,idx,cd,lens,Tc,al),idx)]
                if min(x for x,_ in iv)>=0 and all(free(a_,b_) for a_,b_ in iv):found=(Tc,al,iv);break
            if found:break
        diag[t]=dict(anchor_range=(Tmin,Tmax),candidates=len(cands),found=bool(found),aligned=bool(found and found[1]))
        if not found:dropped.append(t);continue
        T[t]=found[0];aligned[t]=found[1];res+=found[2]
    P1=[t for t in P1 if t not in dropped];P2=[t for t in P2 if t not in dropped]
    return dict(P1=P1,P2=P2,skipped=[t for _,t in skip],T=T,res=res,aligned=aligned,dropped=dropped,diag=diag)
def relayout(w,layout,enc_len,bgs,base_start,max_ntok_c2=14,lens=None,pl=None,align=True):
    texts=w['texts'];P=persons_of(w);lens=lens or {i:enc_len(texts[i]['text']) for i in range(len(texts)) if texts[i]['kind']=='expo'}
    pl=pl or plan(w,lens,P,base_start,max_ntok_c2,align);P1,P2=pl['P1'],pl['P2']
    cond={t:'C1' for t in P1};cond.update({t:'C2' for t in P2})
    if layout=='relB':cond={t:('C2' if c=='C1' else 'C1') for t,c in cond.items()}
    placed={};res=list(pl['res'])
    for t in cond:
        idx=P[t]
        for s_,i in zip(abs_times(w,idx,cond[t],lens,pl['T'][t],pl['aligned'][t]),idx):placed[i]=s_
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
    nw=copy.deepcopy(w);nw['texts']=out;nw['layout']=layout;nw['relayout']=dict(base_layout=w.get('layout'),P1=P1,P2=P2,skipped=pl['skipped'],cond=cond,C1_target=C1_T,C2_target=C2_T,bg_reused=max(0,bi-len(bgs)),last_unit_anchor={str(t):int(v) for t,v in pl['T'].items()},not_aligned=[t for t,v in pl['aligned'].items() if not v],dropped_no_free_anchor=pl['dropped'])
    nw['stats']={'note':'stats of the base world are stale after relayout; recompute with e51_nat_table.py'};return nw
def gap_table(nw,tk,enc_ends):
    text,_,spans=NW.realize(nw,False);sp=NP.tok_spans(spans,enc_ends(text));rows=[]
    by={}
    for s in sp:by.setdefault(s['t'],[]).append(s)
    cond=nw['relayout']['cond'];ok=True
    tab,_=NS.build_table(nw,False,T.encode_offs(tk,text));sets=NS.NatSets(tab,0);ans={}
    for k in range(NCH):
        for s_ in sets.cur_sentences(k):
            lo=max(0,k*CH-WIN)
            if sets.prior_units(s_,lo):ans[s_['t']]=ans.get(s_['t'],0)+len([j for j in s_['place'] if j-1-k*CH>=0])
    for t,us in sorted(by.items()):
        f=[u['tok_start'] for u in us if u['label']=='fact'];g=[b-a for a,b in zip(f,f[1:])]
        cross=sum(1 for a,b in zip(f,f[1:]) if a//CH!=b//CH);same=len(g)-cross
        no=[(u['tok_start'],u['tok_end']) for u in us if u['label']!='fact'];nw8=[sum(1 for a_,e_ in no if a_<f_ and e_>(f_//CH)*CH-WIN) for f_ in f]
        c=cond.get(t,'-');good=True
        if c=='C1':good=all(x<=C1_MAX for x in g)
        if c=='C2':good=all(x>=C2_MIN for x in g) and all(x==0 for x in nw8[1:]) and len(f)>=2
        if c!='-':ok&=good
        rows.append((t,c,len(f),g,nw8,'PASS' if good else 'FAIL',max([u['tok_end'] for u in us]),cross,same,sum(nw8[1:]),ans.get(t,0)))
    return rows,ok
def fmt(rows):
    o=['| person | condition | n_fact | fact-to-fact gaps (tokens) | pairs with the earlier fact in an EARLIER chunk (s_cross) / SAME chunk (s_same) | name-only units inside the O window (8192 before the chunk) before each fact | name-only units before facts 2.. (sum) | answer tokens (facts with an earlier unit in the window) | check | last unit end |','|---|---|---|---|---|---|---|---|---|---|']
    for t,c,n,g,nw8,ok,end,cr,sa,nsum,na in rows:
        if c!='-':o.append(f'| {t} | {c} | {n} | {g} | {cr} / {sa} | {nw8} | {nsum} | {na} | {ok} | {end} |')
    c1=[r for r in rows if r[1]=='C1'];c2=[r for r in rows if r[1]=='C2']
    o.append('');o.append(f"C1 persons {len(c1)}: with >= 1 earlier-chunk pair {sum(1 for r in c1 if r[7]>0)}, with only same-chunk pairs {sum(1 for r in c1 if r[7]==0)}, with a name-only unit in the window before facts 2.. {sum(1 for r in c1 if r[9]>0)}, with NO name-only unit in the window {sum(1 for r in c1 if r[9]==0)}; sum s_cross {sum(r[7] for r in c1)}, sum s_same {sum(r[8] for r in c1)}. answer tokens C1 {sum(r[10] for r in c1)} / C2 {sum(r[10] for r in c2)}. C2 persons {len(c2)}: sum s_cross {sum(r[7] for r in c2)}, sum s_same {sum(r[8] for r in c2)}, name-only units in window {sum(r[9] for r in c2)}")
    return '\n'.join(o)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',default=None);ap.add_argument('--nat',default=None);ap.add_argument('--nat-dir',default=None);ap.add_argument('--out-dir');ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--require-real',action='store_true');ap.add_argument('--layouts',default='relA,relB');ap.add_argument('--selftest',action='store_true');ap.add_argument('--max-ntok',type=int,default=14);ap.add_argument('--no-align',action='store_true',help='do not align C1 consecutive facts to different chunks');a=ap.parse_args()
    if a.selftest:selftest();return
    import nat_dryrun as D
    try:tk=D.load_tok(a.tokenizer_file)
    except Exception as e:print('real tokenizer not usable:',repr(e)[:100]);tk=None
    assert tk is not None or not a.require_real,'--require-real: no real tokenizer'
    print('tokenizer:','REAL' if tk else 'PROXY (logic test only; gaps are NOT final)')
    if a.nat:
        nd=T.find_nat_dir(a.nat_dir);ws=[]
        for item in a.nat.split(','):
            st,lay=item.split(':');p=NF.world_path(nd,int(st),lay);assert p.exists(),f'missing {p}';ws.append((item,NW.load_world(p)))
        info={}
        for item,w in ws:
            print(f'\n######## {item}');run(w,tk,a.out_dir,a.layouts.split(','),a.max_ntok,info.setdefault(item,{}),not a.no_align)
        names={}
        for item,w in ws:
            r=info[item]['plan'];strs=set()
            for t in r['P1']+r['P2']:
                tri=w['triples'][t];strs|={tri['X'],tri['XS']}
            names[item]=strs
        items=list(names);clash=[(x,y,sorted(names[x]&names[y])) for i,x in enumerate(items) for y in items[i+1:] if names[x]&names[y]]
        print('\n## across streams: contrast persons per stream',{k:len(info[k]['plan']['P1'])+len(info[k]['plan']['P2']) for k in items},'; total',sum(len(info[k]['plan']['P1'])+len(info[k]['plan']['P2']) for k in items))
        for k in items:
            w=dict(ws)[k];pk=info[k]['plan'];nf={t:sum(1 for x in w['texts'] if x['kind']=='expo' and x['t']==t and x['label']=='fact') for t in pk['P1']+pk['P2']}
            print(f"  {k}: P1 {pk['P1']} (sum n_fact {sum(nf[t] for t in pk['P1'])}), P2 {pk['P2']} (sum n_fact {sum(nf[t] for t in pk['P2'])}), skipped (do not fit C2) {pk['skipped']}")
        print('full names / surnames shared by contrast persons of two different streams (given names and places such as Chicago may repeat: they are real strings, not person identities):',clash if clash else 'none (a person appears in one stream only)')
        sys.exit(0 if (not clash and all(all(i['ok'].values()) for i in info.values())) else 4)
    assert a.base and Path(a.base).exists(),f'--base file not found: {a.base}';w=NW.load_world(a.base);ok=run(w,tk,a.out_dir,a.layouts.split(','),a.max_ntok,None,not a.no_align);sys.exit(0 if all(ok.values()) else 4)
def run(w,tk,out_dir,layouts,max_ntok=14,rec=None,align=True):
    enc_len=ntok_fn(tk);ends=lambda text:[b for a_,b in T.encode_offs(tk,text)]
    text,_,spans=NW.realize(w,False);sp=NP.tok_spans(spans,ends(text));ei=[i for i,x in enumerate(w['texts']) if x['kind']=='expo'];assert len(ei)==len(sp)
    base_start={i:s['tok_start'] for i,s in zip(ei,sp)};bgs=[x['text'] for x in w['texts'] if x['kind']=='bg'];res={}
    lens={i:enc_len(w['texts'][i]['text']) for i in ei};pl=plan(w,lens,persons_of(w),base_start,max_ntok,align);rx0=NF.forbid_regex(w);print('background paragraphs with a forbidden string in the BASE world (same pool, property of the base):',sum(1 for b in bgs if rx0.search(b)),'of',len(bgs))
    _,sm0,_=T.table_of(w,T.encode_offs(tk,text));print(f"O mask size (BASE {w.get('layout')}, all persons): sum O tokens {sm0['sum_O']}, chunks with O {sm0['chunks_with_O']}, eligible chunks {sm0['eligible_chunks']}, answer tokens {sm0['answer_tokens']}, sum s_cross {sm0['sum_s_cross']}, sum s_same {sm0['sum_s_same']}, sum n_out {sm0['sum_n_out']}")
    for lay in layouts:
        nw=relayout(w,lay,enc_len,bgs,base_start,max_ntok,lens,pl,align);rows,ok=gap_table(nw,tk,ends)
        before=sorted((x['t'],x['label'],x['text']) for x in w['texts'] if x['kind']=='expo');after=sorted((x['t'],x['label'],x['text']) for x in nw['texts'] if x['kind']=='expo');assert before==after,'exposure units changed'
        rx=NF.forbid_regex(nw);bad=sum(1 for x in nw['texts'] if x['kind']=='bg' and rx.search(x['text']))
        r=nw['relayout'];print(f"## {lay}: P1 {r['P1']} P2 {r['P2']} skipped (do not fit C2) {r['skipped']}; C1 chunk alignment skipped (no free anchor with it; their s_same stays in the table) {r['not_aligned']}; dropped (no free anchor at all) {r['dropped_no_free_anchor']}; bg paragraphs reused beyond the base pool: {r['bg_reused']}; forbidden strings in background: {bad}; last exposure end token <= {EXPO_END}: {max(x[6] for x in rows)<=EXPO_END}");print(fmt(rows));print('ALL PASS' if ok else 'SOME FAIL');res[lay]=ok;_,sm,_=T.table_of(nw,T.encode_offs(tk,NW.realize(nw,False)[0]));print(f"O mask size ({lay}, all persons): sum O tokens {sm['sum_O']}, chunks with O {sm['chunks_with_O']}, eligible chunks {sm['eligible_chunks']}, answer tokens {sm['answer_tokens']}, sum s_cross {sm['sum_s_cross']}, sum s_same {sm['sum_s_same']}, sum n_out {sm['sum_n_out']}")
        if out_dir:
            Path(out_dir).mkdir(parents=True,exist_ok=True);p=Path(out_dir)/f"nat_{w['seed']}_{lay}.json";p.write_text(json.dumps(nw,ensure_ascii=False));print('wrote',p)
    if rec is not None:rec['plan']=pl;rec['ok']=res
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
