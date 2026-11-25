"""E62-W window-shift experiment (job 1344w/1347; GPU; new file derived from e62m_run.py / e62_plant.py, both UNCHANGED; thread-server 11/25; design: design thread via the coordinator, documents 1/4..4/4; EXPLORATORY; NOT queued: Hans decides). Question: same stream, same facts, same positions; the WRITE window is narrowed from 8K to 4K so that the earlier mention of the same fact falls out of the window -> is more question-specific (answer-matched) binding written?
Streams: PG-19 novels (one test split + one train split book per stream) concatenated to 640 chunks of 1024 tokens (design: >= 608). 30 facts per stream = 5 attribute types x 6 facts (the dropped type rotates: stream i drops type i), every fact mentioned 64 times with its OWN random order of the 64 templates of its type (each template once; the 6 facts of a type share the same 64 templates), gap classes per type: A x3 (~6144 +- 512 tokens between consecutive mention starts), B x2 (~9216 +- 512), C x1 (~2048 +- 256). Re-tokenised assertions on every consecutive pair: A: end(prev) -> start(next) > 4096 and start(prev) -> end(next) < 8192; B: end -> start > 8192; C: start -> end < 4096. B facts start in the first 45k tokens; the 64th mention is followed by a read point. At most 2 planted sentences start inside any 256 tokens.
Arms (write forward only; applied from chunk 0): W8 = natural (no mask; same path as the NAT arm of e62m_run), W4 = band mask 4096 on EVERY row in ALL layers (the 'ALLQ<band>' mechanism of the NATMASK arms with band 4096: keys at distance >= 4096 are hidden; the keep matrix [CH, WIN+CH] is applied to the prefix layers (prefix_masked) and the suffix layers (kmask) of the ONE forward that gives the write loss; the visible earlier tokens keep the K,V of the natural recording / the carried suffix state, i.e. 'write-time visibility 4K', not a re-computation of the past under a 4K model). W8-SAME (earlier mention sentence hidden in the 8K window) is NOT run; the WinUV class keeps the mode slot open.
Reads (neutral out-of-window tail + one question frame, no book context): W0 and the end of the stream: all 30 questions; after the chunk of the 16th / 32nd / 64th mention of a fact: the 6 questions of its type. Candidates (16 per question): the 6 exposed answers of the type of this stream (gold first) + 10 never-planted generated answers of the type; stored: the full 6 x 16 score matrices (summed log p of the 3 answer tokens) per read, W0 of the same questions, candidate owners (fact index or -1 = never exposed).
Stored per chunk and arm: non-planted CE, pre-clip gradient norm, applied norm, clip coefficient; per planted mention: NLL of the 3 answer tokens before the chunk's write (arm and natural-W8 forward on the same weights), fact id, mention number, gap class.
  python3 e62w_run.py --out O [--streams 0,1] [--arms W8,W4] [--max-chunks N] | --selftest"""
import sys,json,time,bisect,re,pathlib,argparse,random,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e62_plant as PL,e62_templates as TP,e62m_run as M
CH=1024;NW=8;WIN=NW*CH;NCH_S=640;NMEN=64;NC=16;TYPES6=PL.TYPES;BAND4=4096
GAPC={'A':(6144,512),'B':(9216,512),'C':(2048,256)}                  # centre, half width of the start-to-start gap (tokens)
SLK={'A':(4300,7800),'B':(8550,10400),'C':(600,3600)}                # start-to-start gap windows that guarantee the end/start assertions
STREAMS=[dict(name='S1',seed=101,drop=0,books=[('pg19_30312',340),('pg19tr_6688',290)]),dict(name='S2',seed=102,drop=1,books=[('pg19tr_55012',320),('pg19_3608',320)])]   # (book key prefix, chunks taken from its start)
HOOK_MEN=(15,31,63)                                                    # 16th, 32nd, 64th mention
# ------------------------------------------------------------------ the arms
class WinUV:
    """arm object of the e55 engine: .keep(k,m,Kc,Vc,ids_k,keep_all) -> (bool [CH,WIN+CH], info). W8: keep_all unchanged. W4: every row loses the keys at distance >= band (distance = (WIN + row) - column); W8SAME is a reserved mode (not implemented)."""
    def __init__(self,mode,band=BAND4,ch=CH,win=WIN):
        assert mode in ('W8','W4','W8SAME'),mode
        if mode=='W8SAME':raise NotImplementedError('W8-SAME (earlier mention of the same fact hidden in the 8K window) is not part of the pilot')
        self.mode=mode;self.band=band;self.ch=ch;self.win=win;self.last_info={};self.stash=None
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool);self.stash=(k,Kc,Vc,keep_all)
        if self.mode=='W8':self.last_info={'masked_pairs':0};return base,{'uv_arm':'W8','q_banded':0,'masked_pairs':0,'sentence_rows':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=band_keep(base,self.band,self.ch,self.win);n=int((~kp&base[None,:]).sum());self.last_info={'masked_pairs':n};return kp,{'uv_arm':'W4','q_banded':self.ch,'masked_pairs':n,'sentence_rows':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def band_keep(base,band,ch=CH,win=WIN):
    """keep matrix [ch, win+ch]: base[c] and distance (win + r - c) < band"""
    r=np.arange(ch)[:,None];c=np.arange(win+ch)[None,:];kp=np.broadcast_to(np.asarray(base,bool),(ch,win+ch))&((win+r-c)<band);return kp.copy()
# ------------------------------------------------------------------ facts
def make_facts(tk,allb,seed,drop,rng=None):
    """-> facts (30), unexposed[type] (10), reserve[type] (redraw pool), names"""
    ans,names=PL.gen_facts(tk,allb,seed,pool_per_type=40,n_names=40);rng=random.Random(f'E62W|{seed}|facts');kept=[t for i,t in enumerate(TYPES6) if i!=drop];facts=[];unexp={};reserve={}
    nm=list(names);rng.shuffle(nm)
    for ty in kept:
        cl=['A','A','A','B','B','C'];rng.shuffle(cl)
        for s in range(6):facts.append(dict(type=ty,slot=s,cls=cl[s],name=nm.pop(0),answer=ans[ty][s]))
        unexp[ty]=ans[ty][6:16];reserve[ty]=ans[ty][16:]
    return facts,unexp,reserve,kept
def make_cands(facts,unexp,rng):
    """per fact: gold + the 5 other exposed answers of the type + the 10 never-planted answers; -> (words, owners)"""
    out=[];own=[]
    for i,f in enumerate(facts):
        oth=[j for j,g in enumerate(facts) if g['type']==f['type'] and j!=i];rng.shuffle(oth)
        out.append([f['answer']]+[facts[j]['answer'] for j in oth]+list(unexp[f['type']]));own.append([i]+oth+[-1]*len(unexp[f['type']]))
        assert len(out[-1])==NC and len(set(out[-1]))==NC
    return out,own
# ------------------------------------------------------------------ positions
def draw_positions(rng,cls,lmax):
    c,j=GAPC[cls];lo,hi=SLK[cls];gaps=[rng.uniform(max(c-j,lo),min(c+j,hi)) for _ in range(NMEN-1)];span=sum(gaps)
    p0=rng.uniform(3000,45000) if cls=='B' else rng.uniform(3000,lmax-span-3000)
    P=[p0]
    for g in gaps:P.append(P[-1]+g)
    return P
def pair_ok_gap(cls,g):lo,hi=SLK[cls];return lo<=g<=hi
def repair(P,cls_of,rng,lmax,min3=330,iters=40000):
    """P: list (per fact) of start lists; moves single mentions until any three consecutive starts of the whole stream are >= min3 apart; keeps every start-to-start gap of the fact inside its class window"""
    nf=len(P)
    for it in range(iters):
        arr=sorted((p,f,i) for f in range(nf) for i,p in enumerate(P[f]));pos=np.asarray([a[0] for a in arr]);v=np.where(pos[2:]-pos[:-2]<min3)[0]
        if len(v)==0:return it
        t=int(rng.choice(list(v)));f,i=arr[t+rng.randrange(3)][1:];cls=cls_of[f]
        for _ in range(40):
            new=P[f][i]+rng.choice([-1,1])*rng.uniform(60,400)
            if new<1500 or new>lmax:continue
            ok=(i==0 or pair_ok_gap(cls,new-P[f][i-1])) and (i==NMEN-1 or pair_ok_gap(cls,P[f][i+1]-new))
            if ok and (i>0 or cls!='B' or new<=45000):P[f][i]=new;break
    raise RuntimeError('density repair did not converge')
# ------------------------------------------------------------------ stream text
def take_tokens(tk,text,nch):
    """first ~nch*CH tokens of text, cut at the next sentence end"""
    e=tk.encode(text,add_special_tokens=False);j=min(len(e.ids),nch*CH)-1;cut=e.offsets[j][1];b=[x for x in PL.sentence_bounds(text) if x>=cut]
    return text[:(b[0] if b else len(text))].rstrip()
def check_pairs(rows,facts,ntok):
    """assertions of design 1 on re-tokenised positions; -> stats"""
    by=collections.defaultdict(list)
    for r in rows:by[r['fact']].append(r)
    st={c:dict(n=0,ok=0,mins=[],maxs=[]) for c in 'ABC'};bad=[]
    for f,F in by.items():
        F=sorted(F,key=lambda r:r['mention']);cls=facts[f]['cls']
        for a,b in zip(F,F[1:]):
            s0,e0=a['tok'];s1,e1=b['tok'];g_es=s1-e0;g_se=e1-s0;st[cls]['n']+=1
            ok={'A':g_es>4096 and g_se<8192,'B':g_es>8192,'C':g_se<4096}[cls]
            if ok:st[cls]['ok']+=1
            else:bad.append((f,a['mention'],cls,g_es,g_se))
            st[cls]['mins'].append(s1-s0);st[cls]['maxs'].append(s1-s0)
    return st,bad
def fact_table(rows,facts):
    """per fact: answer token count (min, max, share of 3), start-to-start gap min / median / max, end->start minimum and start->end maximum of the consecutive mentions (tokens)"""
    by=collections.defaultdict(list)
    for r in rows:by[r['fact']].append(r)
    out=[]
    for f,F in sorted(by.items()):
        F=sorted(F,key=lambda r:r['mention']);ss=[b['tok'][0]-a['tok'][0] for a,b in zip(F,F[1:])];es=[b['tok'][0]-a['tok'][1] for a,b in zip(F,F[1:])];se=[b['tok'][1]-a['tok'][0] for a,b in zip(F,F[1:])];nt=[len(r['ans_tok']) for r in F]
        out.append(dict(fact=f,type=facts[f]['type'],cls=facts[f]['cls'],answer=facts[f]['answer'],ans_ntok_min=min(nt),ans_ntok_max=max(nt),ans_ntok3_share=float(np.mean([x==3 for x in nt])),match_n=int(sum(r['ans_match'] for r in F)),match_share=float(np.mean([r['ans_match'] for r in F])),gap_min=int(min(ss)),gap_median=float(np.median(ss)),gap_max=int(max(ss)),end_to_start_min=int(min(es)),start_to_end_max=int(max(se)),first_start=F[0]['tok'][0],last_chunk=F[-1]['chunk']))
    return out
def check_density(rows):
    s=sorted(r['tok'][0] for r in rows);d=[s[i+2]-s[i] for i in range(len(s)-2)];return (min(d) if d else 10**9),sum(1 for x in d if x<256)
VERB=bool(__import__('os').environ.get('E62W_VERBOSE'))
def build_stream(tk,allb,spec,facts,seed,nch=NCH_S,max_try=12):
    T0=time.monotonic();RF={}
    """-> dict(text, enc, rows, stats, junction ...). Facts carry name, answer, type, cls. Positions are planned in FINAL token coordinates, planted at sentence boundaries of the source text and verified by re-tokenising."""
    rng=random.Random(f'E62W|{seed}|plant');srcs=[];jun=[]
    for key,n in spec['books']:
        k0=M.full_key(allb,key);srcs.append((k0,take_tokens(tk,allb[k0],n)))
    text=srcs[0][1]+'\n\n'+srcs[1][1];junction_char=len(srcs[0][1])+1
    bounds=PL.sentence_bounds(text);enc0=tk.encode(text,add_special_tokens=False);offs=enc0.offsets;starts=[o[0] for o in offs];tokb=[bisect.bisect_left(starts,b) for b in bounds];nsrc=len(enc0.ids)
    cls_of=[f['cls'] for f in facts];lmax=nch*CH-3000
    orders=[rng.sample(range(NMEN),NMEN) for _ in facts]                                           # each template once per fact, random order
    sents=[[PL.render(TP.POOL[f['type']][orders[i][k]][1],f['name'],f['answer']) for k in range(NMEN)] for i,f in enumerate(facts)]
    est=[[len(tk.encode(s,add_special_tokens=False).ids) for s in S] for S in sents]
    for attempt in range(max_try):
        P=[draw_positions(rng,c,lmax) for c in cls_of]
        try:repair(P,cls_of,rng,lmax)
        except RuntimeError:
            if VERB:print(f'    attempt {attempt}: density repair did not converge, new positions',flush=True)
            continue
        jobs=sorted([(P[f][k],f,k) for f in range(len(facts)) for k in range(NMEN)]);D=[int(j[0]) for j in jobs];estl=[est[f][k]+1 for _,f,k in jobs];shift=[int(x) for x in np.concatenate([[0],np.cumsum(estl)[:-1]])];chosen=None
        for it in range(40):
            pick=[];used=set();cum=0;taken=[]
            for jx,(p,f,k) in enumerate(jobs):
                tgt=D[jx]-shift[jx];i0=bisect.bisect_left(tokb,tgt);best=None
                for rad,sp in ((40,95),(200,60),(800,20)):
                    cand=[x for x in range(max(0,i0-rad),min(len(bounds),i0+rad)) if x not in used and not PL.heading_near(text,bounds[x]) and all(abs(tokb[x]-tokb[y])>=sp for y in taken[-3:])]
                    if cand:best=min(cand,key=lambda x:abs(tokb[x]-tgt));break
                if best is None:raise RuntimeError('no boundary')
                used.add(best);taken.append(best);pick.append((jx,best))
            if VERB:print(f'    picks done {time.monotonic()-T0:.0f}s',flush=True)
            ins=sorted(pick,key=lambda t:(bounds[t[1]],t[0]));parts=[];prev=0;spans=[];cur=0
            for jx,x in ins:
                b=bounds[x];seg=text[prev:b];parts.append(seg);cur+=len(seg);s=sents[jobs[jx][1]][jobs[jx][2]];parts.append(s+' ');spans.append((jx,cur,cur+len(s)));cur+=len(s)+1;prev=b
            if VERB:print(f'    text built {time.monotonic()-T0:.0f}s',flush=True)
            parts.append(text[prev:]);newt=''.join(parts);e2=tk.encode(newt,add_special_tokens=False);off2=e2.offsets;st2=[o[0] for o in off2];rows=[];newshift=list(shift)
            for jx,a,b in spans:
                p,f,k=jobs[jx];ts=bisect.bisect_left(st2,a);te=bisect.bisect_left(st2,b);ans=facts[f]['answer'];s=sents[f][k];a0=a+s.index(ans);a1=a0+len(ans);at=[t for t in range(ts,te) if off2[t][1]>a0 and off2[t][0]<a1]
                tt=TP.POOL[facts[f]['type']][orders[f][k]][1];nm_=[tt.index(x) for x in ('{F}','{TL}','{L}') if x in tt];aid=[int(e2.ids[t]) for t in at];rf=RF.get(ans) or RF.setdefault(ans,[int(x) for x in tk.encode(' '+ans,add_special_tokens=False).ids]);rows.append(dict(fact=f,mention=k,cls=cls_of[f],type=facts[f]['type'],chunk=ts//CH,tok=(ts,te),ans_tok=at,ans_ids=aid,ans_match=bool(aid==rf),template=orders[f][k],text=s,order=jx,ans_first=bool(tt.index('{A}')<min(nm_))));newshift[jx]=shift[jx]+(ts-D[jx])
            st,bad=check_pairs(rows,facts,len(e2.ids));dmin,dviol=check_density(rows)
            if VERB:print(f'    plant attempt {attempt} iter {it}: {len(bad)} pair violations, {dviol} density violations (min {dmin}), mean |start - target| {np.mean([abs(r["tok"][0]-D[r["order"]]) for r in rows]):.0f}, max {max(abs(r["tok"][0]-D[r["order"]]) for r in rows)}, tokens {len(e2.ids)} ({time.monotonic()-T0:.0f}s)',flush=True)
            if not bad and dviol==0:chosen=(newt,e2,rows,st,dmin);break
            shift=newshift
        if chosen is not None:break
    if chosen is None:raise RuntimeError(('planting did not converge',len(bad),bad[:5],dviol))
    newt,e2,rows,st,dmin=chosen;ntok=sum(r['tok'][1]-r['tok'][0] for r in rows)
    if len(e2.ids)<nch*CH:raise RuntimeError(('stream shorter than requested',len(e2.ids),nch*CH))
    # junction: first token of the second book = position of the first 200 characters of book 2 in the planted text
    key2=srcs[1][1][:40];ci=newt.find(key2,max(0,junction_char-1000));jt=bisect.bisect_left([o[0] for o in e2.offsets],ci) if ci>=0 else -1
    share={ty:None for ty in set(f['type'] for f in facts)}
    # template sharing among the facts of one type: set overlap (by design 1.0) and the position-wise match rate
    pos_match=[];set_overlap=[]
    for ty in share:
        idx=[i for i,f in enumerate(facts) if f['type']==ty]
        for a in range(len(idx)):
            for b in range(a+1,len(idx)):
                oa,ob=orders[idx[a]],orders[idx[b]];set_overlap.append(len(set(oa)&set(ob))/NMEN);pos_match.append(sum(1 for x,y in zip(oa,ob) if x==y)/NMEN)
    stats=dict(ans_before_name_share=float(np.mean([r['ans_first'] for r in rows])),planted_tokens=ntok,total_tokens=len(e2.ids),planted_share=ntok/len(e2.ids),density_min3=dmin,junction_token=jt,junction_chunk=(jt//CH if jt>=0 else None),src_tokens=nsrc,books=[k for k,_ in srcs],template_set_overlap=float(np.mean(set_overlap)),template_position_match=float(np.mean(pos_match)),ans_tok3_share=float(np.mean([len(r['ans_tok'])==3 for r in rows])),pairs={c:dict(n=st[c]['n'],ok=st[c]['ok'],min_gap=min(st[c]['mins']),max_gap=max(st[c]['maxs'])) for c in 'ABC' if st[c]['n']})
    stats['fact_table']=fact_table(rows,facts)
    return dict(text=newt,enc=e2,rows=rows,stats=stats,orders=orders)
def ngram_overlap(n=5):
    """number of word n-grams shared between the question frame of a type (placeholders kept) and any of its 64 templates"""
    tot=0
    for ty in TYPES6:
        fr=TP.FRAMES[ty][0].lower().replace('{f}','{F}');ft=re.findall(r"\{[A-Z]+\}|[a-z']+",fr);gf={tuple(ft[i:i+n]) for i in range(len(ft)-n+1)}
        for _,t in TP.POOL[ty]:
            tt=re.findall(r"\{[A-Z]+\}|[A-Za-z']+",t);tt=[w.lower() if not w.startswith('{') else w for w in tt];g={tuple(tt[i:i+n]) for i in range(len(tt)-n+1)};tot+=len(gf&g)
    return tot
# ------------------------------------------------------------------ selftest (CPU)
def selftest(full=True):
    t0=time.monotonic()
    # (2) band keep == brute force; chunk 0 is a no-op (distance < band inside one chunk), applied from the first chunk that has far keys
    rng=np.random.default_rng(0);ch,win,band=16,64,24
    for _ in range(3):
        base=rng.random(win+ch)<0.8;kp=band_keep(base,band,ch,win);bf=np.zeros((ch,win+ch),bool)
        for r in range(ch):
            for c in range(win+ch):bf[r,c]=bool(base[c]) and (win+r-c)<band
        assert (kp==bf).all()
        for r in range(ch):                                                  # nothing at distance >= band is visible, everything within the band is (where natural keeps it)
            far=[c for c in range(win+ch) if win+r-c>=band];near=[c for c in range(win+ch) if win+r-c<band]
            assert not kp[r,far].any() and (kp[r,near]==base[near]).all()
    base=np.ones(WIN+CH,bool);u=WinUV('W4');kk,_=u.keep(0,None,None,None,None,base);ref=band_keep(base,BAND4);assert (kk==ref).all()
    d=(WIN+np.arange(CH)[:,None]-np.arange(WIN+CH)[None,:]);assert (kk[d>=BAND4]==False).all() and (kk[(d<BAND4)]==True).all()
    assert WinUV('W8').keep(1,None,None,None,None,base)[0].all()
    try:WinUV('W8SAME');raise SystemExit('W8SAME must be reserved')
    except NotImplementedError:pass
    # W8 keep identical to the NAT keep of e62m_run (E62UV mode NAT) on random inputs
    ev=M.E62UV([[(10,20)]],[],[],'',[],'NAT');b2=rng.random(WIN+CH)<0.9;assert (ev.keep(3,None,None,None,None,b2)[0]==WinUV('W8').keep(3,None,None,None,None,b2)[0]).all()
    print('  mask checks ok: W4 == brute force (4096-token band, hidden at distance >= 4096, visible within), no-op in the first 4 chunks (distance < 4096), W8 == E62UV NAT keep, W8SAME reserved')
    # (5) templates: 64 per type, 5-gram overlap with the question frames
    assert all(len(TP.POOL[t])==NMEN for t in TYPES6);ov=ngram_overlap(5);assert ov==0,ov;print(f'  templates: 64 per type, 5-gram overlap with the question frames: {ov}')
    from tokenizers import Tokenizer as _T
    tk0=_T.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));bad=0;n_=0
    for ty in TYPES6:                                                          # frames end with 'A:', 'A: The', 'A: Mr.': the answer follows with a leading space; prefix-stable, 3 answer tokens, first token = the token of ' '+word
        for w_ in ('Reamaeton','Limeworth','Shashaeton'):
            fr=TP.FRAMES[ty][0].replace('{F}','Josiah Stitiwell');pid=tk0.encode(fr,add_special_tokens=False).ids;full=tk0.encode(fr+' '+w_,add_special_tokens=False).ids;n_+=1
            ok=full[:len(pid)]==pid and len(full)-len(pid)==3 and full[len(pid)]==tk0.encode(' '+w_,add_special_tokens=False).ids[0] and tk0.encode(' '+w_,add_special_tokens=False).ids==full[len(pid):]
            bad+=0 if ok else 1
    assert bad==0,bad;print(f'  first answer token: frame + " " + word is prefix-stable, 3 answer tokens, first token = first token of " "+word, for {n_} (type, word) pairs (frames ending A:, A: The, A: Mr.)')
    if not full:return
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));allb=M.load_all100();stats_all={}
    for si,sp in enumerate(STREAMS):
        facts,unexp,reserve,kept=make_facts(tk,allb,sp['seed'],sp['drop']);rr=random.Random(5);cands,own=make_cands(facts,unexp,rr)
        assert len(facts)==30 and collections.Counter(f['cls'] for f in facts)=={'A':15,'B':10,'C':5} and all(sorted(f['cls'] for f in facts if f['type']==t)==['A','A','A','B','B','C'] for t in kept)
        low='\n'.join(allb.values()).lower();words=[f['answer'] for f in facts]+[w for t in kept for w in unexp[t]]+[f['name'][1] for f in facts]+[f['name'][0]+' '+f['name'][1] for f in facts]      # first names are common words of the FIRST list; the invented words are the answers, the unexposed answers, the surnames and the full names
        assert not [w for w in words if w.lower() in low],'invented word found in the books'
        assert all(len(tk.encode(' '+w,add_special_tokens=False).ids)==3 for w in [f['answer'] for f in facts]+[w for t in kept for w in unexp[t]]),'all candidates are 3 tokens in the read form (" "+word)'
        S=build_stream(tk,allb,sp,facts,sp['seed']);st=S['stats'];rows=S['rows']
        # (1) every consecutive pair passes (also asserted inside build_stream); read points after the 64th mention
        for f in range(30):
            ch64=max(r['chunk'] for r in rows if r['fact']==f);assert ch64<NCH_S-1,(f,ch64)
        assert sum(1 for r in rows if r['cls']=='B' and r['mention']==0 and r['tok'][0]>45000+3000)==0
        stats_all[sp['name']]=st
        print(f'  stream {sp["name"]} (insertion seed {sp["seed"]}, dropped type {TYPES6[sp["drop"]]}): books {st["books"]}, junction at token {st["junction_token"]} = chunk {st["junction_chunk"]}; {st["total_tokens"]} tokens ({st["total_tokens"]/CH:.0f} chunks), planted {st["planted_tokens"]} tokens = {100*st["planted_share"]:.2f}%; min distance of the 3rd of 3 consecutive starts {st["density_min3"]} (>= 256); answer tokens = 3 in {100*st["ans_tok3_share"]:.1f}% of the mentions')
        for c,v in st['pairs'].items():print(f'    gap class {c}: {v["ok"]}/{v["n"]} consecutive pairs pass, start-to-start gap min {v["min_gap"]} max {v["max_gap"]}')
        ft=st['fact_table']
        for c in 'ABC':
            x=[t for t in ft if t['cls']==c];print(f'    class {c} facts ({len(x)}): start-to-start gap min {min(t["gap_min"] for t in x)} / median of the medians {np.median([t["gap_median"] for t in x]):.0f} / max {max(t["gap_max"] for t in x)}; end->start min {min(t["end_to_start_min"] for t in x)}'+(f' (margin to the 8192 boundary {min(t["end_to_start_min"] for t in x)-8192} tokens)' if c=='B' else '')+(f'; start->end max {max(t["start_to_end_max"] for t in x)} (limit 8192, margin {8192-max(t["start_to_end_max"] for t in x)})' if c=='A' else '')+(f'; start->end max {max(t["start_to_end_max"] for t in x)} (limit 4096)' if c=='C' else ''))
        mm=[t['ans_ntok3_share'] for t in ft];ms=[t['match_share'] for t in ft]
        for c in 'ABC':
            x=[t['match_n'] for t in ft if t['cls']==c];print(f'    class {c}: mentions whose answer token ids equal the read form (" "+word): mean {np.mean(x):.1f}/64 per fact, facts with all 64: {sum(1 for t in ft if t["cls"]==c and t["match_n"]==64)}/{len(x)}')
        bad=[r for r in rows if not r['ans_match']];import collections as _c
        why=_c.Counter(('4 tokens' if len(r['ans_tok'])==4 else f'{len(r["ans_tok"])} tokens')+' | char before: '+repr(r['text'][max(r['text'].index(facts[r['fact']]['answer'])-1,0)])+' | char after: '+repr(r['text'][r['text'].index(facts[r['fact']]['answer'])+len(facts[r['fact']]['answer']):][:1]) for r in bad)
        print('    mentions with a token sequence different from the read form (reasons: number of tokens, character before / after the answer):',dict(why.most_common(8)))
        print(f'    answer tokens per mention: min {min(t["ans_ntok_min"] for t in ft)} max {max(t["ans_ntok_max"] for t in ft)}, facts with every mention 3 tokens {sum(1 for t in ft if t["ans_ntok3_share"]==1)}/{len(ft)}')
        print(f'    template sharing among same-type facts: set overlap {st["template_set_overlap"]:.2f} (design: 1.00), same template at the same mention number {st["template_position_match"]:.3f}; names and answers new (seed {sp["seed"]}), none occurs in the {len(allb)} books of the leak check')
    print(f'e62w_run selftest ok ({time.monotonic()-t0:.0f}s)')
# ------------------------------------------------------------------ GPU run
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    from tokenizers import Tokenizer
    import e55_run as R55,e55_iso as ISO,e55_gpu as RG,e57_natmask_run as NM,e57_qband_run as QR
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic()
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));allb=M.load_all100();arms=a.arms.split(',')
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    hash_=lambda m:LD.reuse._hash_pytree(m.inner_parameters())[0]
    @eqx.filter_jit
    def grad_fn(m,st_suf,c,pref_,kmask,wt):
        def total(mm):
            s3.S3SUF['kmask']=kmask
            try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
            finally:s3.S3SUF.clear()
            lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
        (_l,nll),g=eqx.filter_value_and_grad(total,has_aux=True)(m);return g.inner_parameters(),nll
    @eqx.filter_jit
    def fwd_fn(m,st_suf,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);return -jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    for si in [int(x) for x in a.streams.split(',')]:
        sp=STREAMS[si];facts,unexp,reserve,kept=make_facts(tk,allb,sp['seed'],sp['drop']);rng=random.Random(f"E62W|{sp['seed']}|run");t_s=time.monotonic()
        forbid=[w for f in facts for w in (f['answer'],f['name'][0],f['name'][1])]+[w for t in kept for w in unexp[t]]
        out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,forbid)
        with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,list(out_tail))                    # the neutral tail is processed ONCE per stream and shared by every read object
        items_for=lambda cands,idx:[dict(prefix=TP.FRAMES[facts[i]['type']][0].replace('{F}',facts[i]['name'][0]+' '+facts[i]['name'][1]),cands=[' '+w for w in cands[i]]) for i in idx]
        for rnd in range(7):                                                      # W0 screen: answers with gold above all 15 wrong at W0 are redrawn
            cands,own=make_cands(facts,unexp,rng);raw=R55.rows_for(LD,rt,items_for(cands,range(len(facts))))
            with rt32.mesh,jax.default_matmul_precision('highest'):P=LD.packing.Packed(T_,raw,GP=2)
            with jax.default_matmul_precision('highest'):S0=np.asarray(P.r1(rd.to_fp32(rt.base))).reshape(len(facts),NC)
            mg=S0[:,0]-S0[:,1:].max(1);bad=[i for i in range(len(facts)) if mg[i]>0];lg(f'{sp["name"]} W0 screen round {rnd}: {len(bad)} facts with margin > 0 redrawn')
            if not bad:break
            for i in bad:facts[i]['answer']=reserve[facts[i]['type']].pop(0)
        else:raise RuntimeError('W0 screen did not settle')
        assert all(raw[i*NC+c][1][1]-raw[i*NC+c][1][0]==3 for i in range(len(facts)) for c in range(NC)),'every candidate of every question must be 3 answer tokens in the read form'
        for i,f in enumerate(facts):f['w0_margin']=float(mg[i])
        S=build_stream(tk,allb,sp,facts,sp['seed']);enc=S['enc'];rows=S['rows'];stt=S['stats'];nch=min(len(enc.ids)//CH,NCH_S,a.max_chunks or 10**9);n=nch*CH;stream=np.asarray(enc.ids[:n],np.int64);newt=S['text']
        lg(f'{sp["name"]}: books {stt["books"]}, {len(enc.ids)} tokens, junction chunk {stt["junction_chunk"]}, planted share {100*stt["planted_share"]:.2f}%, pairs '+', '.join(f'{c}:{v["ok"]}/{v["n"]}' for c,v in stt['pairs'].items()))
        kept_idx={ty:[i for i,f in enumerate(facts) if f['type']==ty] for ty in kept};Pty={}
        for ty,idx in kept_idx.items():
            rw=[x for i in idx for x in raw[i*NC:(i+1)*NC]]
            with rt32.mesh,jax.default_matmul_precision('highest'):Pty[ty]=LD.packing.Packed(T_,rw,GP=2)
        def read_types(m,types):
            with jax.default_matmul_precision('highest'):return {ty:np.asarray(Pty[ty].r1(rd.to_fp32(m))).reshape(6,NC).tolist() for ty in types}
        # ---- generation reads: full-vocabulary ranks of the first answer token of the 16 candidates (step 0) and greedy generation (<= 8 new tokens) after the question frame; context = neutral tail + question frame (as the score reads)
        from ttt.model.data import Batch
        PS=LD.packing;tkz=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
        pref_ids=[list(raw[i*NC][0][:raw[i*NC][1][0]+1]) for i in range(len(facts))]                       # question tokens up to and including the position that predicts the first answer token
        tokmat=np.asarray([[int(raw[i*NC+c][0][raw[i*NC+c][1][0]+1]) for c in range(NC)] for i in range(len(facts))],np.int32)
        @eqx.filter_jit
        def info_packed(m,st,ids_,pref_,seg_,loc_,pos_,tok_):
            def one(i,pp,sg,lc,ps,tk_):
                PS.PACK_SUF['seg']=sg;PS.PACK_SUF['loc']=lc
                try:
                    r=m.language_model.suffix_call(prefix_outputs=pp,state=st,seq=Batch(input_ids=i,target_tokens=i,loss_masks=jnp.ones_like(i,dtype=bool)))
                    lg=r.logits.astype(jnp.float32)[ps];val=jnp.take_along_axis(lg,tk_,axis=-1);return jnp.argmax(lg,axis=-1),1+jnp.sum(lg[:,None,:]>val[:,:,None],axis=-1),val-jax.nn.logsumexp(lg,axis=-1)[:,None]
                finally:PS.PACK_SUF.clear()
            return jax.vmap(one)(ids_,pref_,seg_,loc_,pos_,tok_)
        def tail_state(m32):
            with rt.mesh:
                st=PS._clone(rt.state_suffix)
                for j in range(len(T_.tail_ids)//1024):st,_=rt.chunk_logp(m32,st,rt.chunk(T_._tail_seq,j),T_._tail_pref[j*1024:(j+1)*1024])
            return st
        def gen_step(m32,st,rows_ids,tm):
            P_=PS.Packed(T_,[(list(x),(len(x)-1,len(x))) for x in rows_ids],GP=2);nrow=len(rows_ids);cnt=collections.Counter();nr=max(collections.Counter(w_[0] for w_ in P_.where).values());npad=P_.ids.shape[0]
            pos=np.zeros((npad,nr),np.int32);tok=np.zeros((npad,nr,NC),np.int32);slot=[]
            for i,(c,a_,b_) in enumerate(P_.where):j=cnt[c];cnt[c]+=1;pos[c,j]=a_;tok[c,j]=tm[i];slot.append((c,j))
            with rt.mesh:am,rk,lpc=info_packed(m32,st,P_.ids,P_.pref1,P_.seg,P_.loc,jnp.asarray(pos),jnp.asarray(tok))
            am=np.asarray(am);rk=np.asarray(rk);lpc=np.asarray(lpc);return [int(am[c,j]) for c,j in slot],[rk[c,j].tolist() for c,j in slot],[lpc[c,j].tolist() for c,j in slot]
        def first_lp(P_,m32):
            """first answer token log p of every row of a Packed score object (the SCORE path: per-token log p of lp_packed at the span start)"""
            st=tail_state(m32);out=[]
            with jax.default_matmul_precision('highest'):
                for g in range(0,P_.ids.shape[0],P_.GP):out.append(np.asarray(P_.lp_packed(m32,st,P_.ids[g:g+P_.GP],P_.tg[g:g+P_.GP],P_.mk[g:g+P_.GP],P_.pref1[g:g+P_.GP],P_.seg[g:g+P_.GP],P_.loc[g:g+P_.GP])))
            lp=np.concatenate(out);return np.asarray([lp[c,a_] for c,a_,b_ in P_.where])
        def paths_agree(m,types,tag):
            """two-path assertion: the first-token log p of every candidate from the step-0 logits of the generation path == the one of the score path (same question, same candidate); prints the max abs difference"""
            m32=rd.to_fp32(m);g=gen_read(m,types,1);worst=0.0
            for ty in types:
                a_=np.asarray(g[ty]['lp0']);b_=first_lp(Pty[ty],m32).reshape(6,NC);worst=max(worst,float(np.abs(a_-b_).max()))
            lg(f'  {sp["name"]} two-path check ({tag}): max |first-token log p generation path - score path| = {worst:.2e} (tolerance 0.05)');assert worst<0.05,('score and generation paths disagree',tag,worst);return worst
        def gen_read(m,types,nsteps=8):
            """-> {type: dict(rank=[6][16] rank of the first token of every candidate in the full vocabulary (1 = top), argmax=[6], tok=[6][nsteps] greedy tokens, text=[6])}; nsteps = 1: ranks only"""
            m32=rd.to_fp32(m);out={}
            with jax.default_matmul_precision('highest'):
                st=tail_state(m32);qs=[i for ty in types for i in kept_idx[ty]];cur=[list(pref_ids[i]) for i in qs];gen=[[] for _ in qs];tm=tokmat[qs];ranks=None;lps=None
                for t in range(nsteps):
                    am,rk,lpc=gen_step(m32,st,cur,tm)
                    if t==0:ranks=rk;lps=lpc
                    for i_,a_ in enumerate(am):cur[i_].append(a_);gen[i_].append(a_)
            for ty in types:
                sel=[qs.index(i) for i in kept_idx[ty]];d=dict(rank=[ranks[x] for x in sel],argmax=[gen[x][0] for x in sel],lp0=[lps[x] for x in sel])
                if nsteps>1:d['tok']=[gen[x] for x in sel];d['text']=[tkz.decode(gen[x]) for x in sel]
                out[ty]=d
            return out
        W0=read_types(rt.base,kept);W0gen=gen_read(rt.base,kept);chk=dict(W0=paths_agree(rt.base,kept,'W0'))
        for ty in kept:                                                            # consistency: rank 1 of the gold first token <=> the step-0 argmax is the gold first token
            for j,i in enumerate(kept_idx[ty]):assert (W0gen[ty]['rank'][j][0]==1)==(W0gen[ty]['argmax'][j]==int(tokmat[i][0])),('rank / argmax inconsistent',ty,j)
        pcit=[]
        for i,f in enumerate(facts):                                              # positive control at W0: the first planted sentence directly in front of the question
            sent=[r for r in rows if r['fact']==i and r['mention']==0][0]['text'];pcit.append(dict(prefix=sent+'\n'+items_for(cands,[i])[0]['prefix'],cands=[' '+w for w in cands[i]]))
        rawpc=R55.rows_for(LD,rt,pcit)
        with rt32.mesh,jax.default_matmul_precision('highest'):Ppc=LD.packing.Packed(T_,rawpc,GP=2)
        with jax.default_matmul_precision('highest'):W0pc=np.asarray(Ppc.r1(rd.to_fp32(rt.base))).reshape(len(facts),NC)
        hook_types=collections.defaultdict(set);hook_facts=collections.defaultdict(list)
        for r in rows:
            if r['mention'] in HOOK_MEN and r['chunk']<nch:hook_types[r['chunk']].add(r['type']);hook_facts[r['chunk']].append((r['fact'],r['mention']))
        hook_types[nch-1]|=set(kept);hooks=sorted(hook_types);hook_gen=collections.defaultdict(set)
        for r in rows:
            if r['mention']==NMEN-1 and r['chunk']<nch:hook_gen[r['chunk']].add(r['type'])
        hook_gen[nch-1]|=set(kept)
        for f in range(len(facts)):assert max(r['chunk'] for r in rows if r['fact']==f)<NCH_S-1,'a read point after the 64th mention must exist'
        planted_mask=np.zeros(n+1,bool)
        for r in rows:
            if r['tok'][1]<=n:planted_mask[r['tok'][0]:r['tok'][1]]=True
        ment_by_chunk=collections.defaultdict(list)
        for r in rows:
            if r['tok'][1]<=n and r['chunk']<nch:ment_by_chunk[r['chunk']].append(r)
        brec=dict(fact_table=stt['fact_table'],two_path_check=chk,gold_tok=[[int(t) for t in raw[i*NC][0][raw[i*NC][1][0]+1:raw[i*NC][1][1]+1]] for i in range(len(facts))],cand_first=tokmat.tolist(),stream=sp['name'],seed=sp['seed'],dropped=TYPES6[sp['drop']],kept=kept,nch=nch,facts=facts,cands=cands,cand_owner=own,manifest=[{k:v for k,v in r.items() if k!='order'} for r in rows],stats=stt,W0=W0,W0_gen=W0gen,W0_positive_control=W0pc.tolist(),hooks_gen={str(k):sorted(v) for k,v in hook_gen.items()},hooks={str(k):dict(types=sorted(hook_types[k]),facts=hook_facts.get(k,[])) for k in hooks},arms={})
        lg(f'{sp["name"]}: {nch} chunks, {len(hooks)} read points, setup {time.monotonic()-t_s:.0f}s')
        L=ISO.Iso55(rt,save_dir=out/'states'/sp['name'],canary=can);L.rec_cache={}
        for arm in arms:
            uv=M.E62UV([],[],[],'',[],'NAT') if arm=='NATREF' else WinUV(arm);rec=dict(steps=[],mention_nll=[],reads={});cnt=[0]
            def ws(m,st_all,st_suf,c,pref,kmask,rec=rec,cnt=cnt,uv=uv,arm=arm):
                k=cnt[0];cnt[0]+=1;valid=np.asarray(c.loss_masks,np.float32);wt=valid/valid.sum();g,nll=grad_fn(m,st_suf,c,pref,kmask,jnp.asarray(wt));nll=np.asarray(nll,np.float32)
                if arm in ('W8','NATREF'):nll_nat=nll
                else:
                    kk,Kc,Vc,keep_nat=uv.stash;assert kk==k,(kk,k);ids_k=jnp.asarray(stream[k*CH:(k+1)*CH],dtype=jnp.int32);pref_nat=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep_nat));nll_nat=np.asarray(fwd_fn(m,st_suf,c,pref_nat,jnp.asarray(keep_nat)),np.float32)
                newp,gn,coef=L.sgd_apply(m,st_all,g);del g;ce=float((nll*valid).sum()/valid.sum());lo=k*CH;npm=~planted_mask[lo+1:lo+CH+1][:len(nll)]&(valid>0)
                for r in ment_by_chunk.get(k,[]):
                    ix=[t-1-lo for t in r['ans_tok'] if 0<=t-1-lo<len(nll)-1];rec['mention_nll'].append(dict(fact=r['fact'],mention=r['mention'],cls=r['cls'],ans_first=r['ans_first'],ans_match=r['ans_match'],k=k,ntok=len(r['ans_tok']),nll_arm=[float(nll[i]) for i in ix],nll_nat=[float(nll_nat[i]) for i in ix]))
                rec['steps'].append(dict(k=k,gnorm=float(gn),coef=float(coef),upd=float(coef)*float(gn),train_ce=ce,nat_ce=float((nll_nat*valid).sum()/valid.sum()),nonplanted_ce_arm=float(nll[npm].mean()) if npm.any() else None,nonplanted_ce_nat=float(nll_nat[npm].mean()) if npm.any() else None,masked_pairs=int(uv.last_info.get('masked_pairs',0))))
                return newp,gn,coef,jnp.asarray(ce,jnp.float32)
            def hook(k,m,rec=rec):
                if 'post' not in chk and k>0:chk['post']=paths_agree(m,sorted(hook_types[k]),f'after chunk {k}')
                h0=hash_(m);ty_=sorted(hook_types[k]);r=dict(scores=read_types(m,ty_),gen=gen_read(m,ty_,1));
                if k in hook_gen:r['gen'].update(gen_read(m,sorted(hook_gen[k]),8))
                h1=hash_(m);assert h0==h1,'weights changed by a read';return r
            L.write_step=ws;t0=time.monotonic();meta=[dict(sents=[]) for _ in range(nch)]
            r=L.run_stream(NM.ENGINE,1,stream,read_chunks=[],canary_chunks=[],nch=nch,tag=f'{sp["name"]}{arm}',log=lg,read_hook=hook,hook_chunks=hooks,chunks_meta=meta,uv=uv);r.pop('model',None)
            rec['reads']={str(k):v for k,v in r.get('inline',{}).items()};rec['sec']=time.monotonic()-t0;brec['arms'][arm]=rec;mm=[x['masked_pairs'] for x in rec['steps']]
            lg(f'  {sp["name"]} arm {arm}: {nch} chunks in {rec["sec"]:.0f}s, masked pairs {sum(mm)} in {sum(1 for x in mm if x)} chunks, mean |g| {np.mean([x["gnorm"] for x in rec["steps"]]):.3f}, mean coef {np.mean([x["coef"] for x in rec["steps"]]):.3f}')
        if 'W8' in brec['arms'] and 'NATREF' in brec['arms']:
            dsc=max(float(np.abs(np.asarray(brec['arms']['W8']['reads'][kk]['scores'][ty])-np.asarray(brec['arms']['NATREF']['reads'][kk]['scores'][ty])).max()) for kk in brec['arms']['W8']['reads'] for ty in brec['arms']['W8']['reads'][kk]['scores']);brec['check_W8_vs_NATREF_max_dscore']=dsc;lg(f'  check W8 vs NATREF: max |read score difference| over all read points {dsc:.2e}')
            d=max(abs(x['gnorm']-y['gnorm']) for x,y in zip(brec['arms']['W8']['steps'],brec['arms']['NATREF']['steps']));brec['check_W8_vs_NATREF_max_dgnorm']=d;lg(f'  check W8 vs the e62m NAT keep path: max |d gnorm| {d:.2e}')
        if 'W8' in brec['arms'] and 'W4' in brec['arms']:
            first=min((x['k'] for x in brec['arms']['W4']['steps'] if x['masked_pairs']),default=nch);d=max((abs(x['gnorm']-y['gnorm']) for x,y in zip(brec['arms']['W8']['steps'][:first],brec['arms']['W4']['steps'][:first])),default=0.0);brec['check_W4_vs_W8_before_first_mask']=dict(first_masked_chunk=first,max_dgnorm=d);lg(f'  check W4 vs W8 before the first masked chunk ({first}): max |d gnorm| {d:.2e}')
        json.dump(brec,open(out/f'e62w_{sp["name"]}.json','w'),default=float);L.rec_cache=None;del L;lg(f'{sp["name"]} written, stream total {time.monotonic()-t_s:.0f}s')
    lg('done',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--streams',default='0,1');ap.add_argument('--arms',default='W8,W4');ap.add_argument('--max-chunks',type=int,default=0);ap.add_argument('--selftest',action='store_true');ap.add_argument('--selftest-quick',action='store_true');a=ap.parse_args()
    if a.selftest or a.selftest_quick:selftest(full=not a.selftest_quick)
    else:
        run(a)
