"""E52 Level-2 world generator: the background paragraphs of a natural-text stream (default 9105) with 60 NEW persons (30 swap-pairs) inserted at controlled positions.
Two layouts L2a / L2b (written as nat_<stream-id>_L2a.json / _L2b.json, readable by nat_ladder.py --nat <id>:L2a):
  24 flip pairs (12 per layout C1 = 4 exposures, every consecutive gap <= 4096 tokens and in different 1024-token chunks; 12 C2 = 4 exposures, every gap >= 12288 tokens; flipped in the other layout) + 6 C0 pairs (1 exposure, same in both layouts).
  The LAST exposure of every pair is at the same token position in both layouts (uniform in [39.5K, 53.5K], collision-free in both layouts); total length and background are identical in both layouts.
  A slot = the two sentences of the pair adjacent (random order); frames: 4 per pair from the pair-disjoint dev2 bank (frame_pools_dev2); names/places from the dev pools (class of the most frequent token length, as gen_world), occupations 30 (gen_world.OCC) + 30 (dev2).
Background: the base stream's own paragraphs (nothing reused) plus as many paragraphs of the other NON-SEALED pool streams as the target length needs (filters: forbidden strings of the new world, exact duplicates, canary 8-grams). Token bookkeeping uses the separator cost calibrated on the base world (nat_relayout.calibrate_sep).
Built-in checks (exit 3 on failure): conditions per pair (gaps, chunk alignment, last-exposure equality), identical exposure multiset and background in both layouts, no duplicate paragraph, forbidden strings 0, NW.build_items works, then the filler audit of e51_relayout_audit (paragraph-level pools, passes --passes, default 2).
No GPU. Usage: e52_insert.py --nat-dir DIR --out-dir OUT [--base-nat 9105:spread] [--pool-streams 9101,9103,9104] [--stream-id 9201] [--occ-dev2 a,b,c|FILE] [--tokenizer-file F] [--require-real] [--passes 2]   |   --selftest"""
import hashlib,os,sys,re,json,copy,random,argparse,tempfile,shutil
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;GW=HERE.parents[0]/'exp04_what_ttt_retains';sys.path.insert(0,str(GW));sys.path.insert(0,str(HERE))
import nat_world as NW,nat_pass as NP,nat_filler as NF,e51_nat_table as T,e51_pool_audit as PA,nat_relayout as RL
CH=1024;NCH=63;WIN=8192;UP=60;DN=450
T_LO,T_HI=39500,53500;C2_GAP=12800;C1_MAX=4096;C2_MIN=12288
def draw_pool(G,parts_a,parts_b,rng,n,tag,min_n=None):
    """names a+b of the most frequent token-length class (as gen_world), shuffled; returns n distinct"""
    allw=[a+b for a in parts_a for b in parts_b];cls={}
    for w in allw:cls.setdefault(G.ntok(w),[]).append(w)
    L=max(cls,key=lambda k:len(cls[k]));pool=sorted(cls[L]);rng.shuffle(pool)
    assert len(pool)>=(min_n or n),f'{tag}: class of {L} tokens has {len(pool)} < {n} names (class sizes {sorted((k,len(v)) for k,v in cls.items())})'
    return pool[:n],L
def align_slots(st,ln):
    """consecutive slot starts must lie safely in different chunks (margins UP / DN cover the shift of the background fill); shift the earlier slots earlier. The last slot never moves."""
    st=list(st)
    for k in range(len(st)-1,0,-1):
        if (st[k-1]+UP)//CH>=(st[k]-DN)//CH:
            X=((st[k]-DN)//CH)*CH;d=max(0,st[k-1]+ln[k-1]-(X-550))
            for m in range(k):st[m]-=d
    return st
def c1_starts(T,ln,rng):
    for _ in range(200):
        g=[rng.uniform(1000,2400) for _ in range(3)];st=[T-g[0]-g[1]-g[2],T-g[1]-g[2],T-g[2],T];st=align_slots(st,ln)
        if all(b-a<=C1_MAX-60 for a,b in zip(st,st[1:])) and min(st)>=600:return st
    raise AssertionError('no C1 timeline')
def c2_starts(T):return [T-3*C2_GAP,T-2*C2_GAP,T-C2_GAP,T]
def make_world(base,G,FB,D2,H,rng,stream_id,occ_dev2,n_flip=24,n_c0=6,R=4):
    npairs=n_flip+n_c0;nt=2*npairs
    persons,Lp=draw_pool(G,G.SYL1,G.SYL2,rng,2*nt,'persons');places,Ll=draw_pool(G,G.PL_PRE,G.PL_SUF,rng,2*nt,'places')   # 2*60 names per class: the pseudo lists are the control lists reversed (never exposed in the stream, so sharing is harmless; the class of 3 tokens has only 174 places on the server)
    occ=list(G.OCC)+list(occ_dev2);assert len(occ)>=nt and len(set(occ))==len(occ),'need 60 distinct occupations';rng.shuffle(occ)
    bank,_=FB.make('pair',R,npairs,G.FR,seed='E52dev2',pools=D2.DEV2_POOLS,need=D2.DEV2_NEED)
    tri=[];fr={}
    for i in range(nt):
        tri.append({'id':i,'pair':f'p{i//2}','X':persons[i],'XF':persons[i],'XS':persons[i],'P':places[i],'Q':None,'E':None,'o':occ[i],'natg':'L2','region':''})
        own=bank[(i//2)*R:(i//2+1)*R];o=own[:];random.Random(f'E52fr|{stream_id}|{i}').shuffle(o);fr[i]=o
    w={'seed':stream_id,'v':'l2','kind':'natural','layout':'','triples':tri,'ctrl_persons':persons[nt:2*nt],'ctrl_places':places[nt:2*nt],'pseudo_persons':persons[nt:2*nt][::-1],'pseudo_places':places[nt:2*nt][::-1],'texts':[],'pairs':npairs,
       'frames':{'birth':G.PR_HOME,'rev':G.PR_REV,'school':'','freq_p':G.PR_FREQ_P,'freq_l':G.PR_FREQ_L,'em':[]},'stats':{}}
    return w,fr,Lp,Ll
def sentence(G,w,fr,i,r):
    t=w['triples'][i];return fr[i][r-1].format(X='@X%d@'%i,P=t['P'],o=t['o'],a_o=G.A(t['o']))
def conditions(npairs,n_flip,rng):
    order=list(range(npairs));rng.shuffle(order);flips=sorted(order[:n_flip]);c0=sorted(order[n_flip:]);half=set(flips[::2])
    a={p:('C1' if p in half else 'C2') for p in flips};a.update({p:'C0' for p in c0});b={p:({'C1':'C2','C2':'C1'}[c] if c!='C0' else 'C0') for p,c in a.items()};return a,b
def plan_slots(w,fr,G,enc_len,SEP,cond_a,cond_b,rng):
    """per pair: shared last-slot start T and the slot starts under each layout; reservations are checked for every variant of a pair"""
    res=[];out={};npairs=w['pairs'];sl={}
    for p in range(npairs):
        sl[p]=[sum(enc_len(sentence(G,w,fr,2*p+j,r).replace('@X%d@'%(2*p+j),w['triples'][2*p+j]['X']))+SEP for j in (0,1)) for r in (1,2,3,4)]
    def free(a,b):return all(b<=s or a>=e for s,e in res)
    order=sorted(range(npairs),key=lambda p:rng.random())
    for p in order:
        c=cond_a[p];T0=rng.uniform(T_LO,T_HI);cands=sorted(range(int(T_LO),int(T_HI)+1,25),key=lambda x:abs(x-T0));found=None
        for Tc in cands:
            ln=sl[p]
            variants={'C0':[[Tc]]} if c=='C0' else {'C1':[c1_starts(Tc,ln,random.Random(f'E52c1|{p}|{Tc}'))],'C2':[c2_starts(Tc)]}
            iv=[(s,s+(ln[k] if len(v)>1 else ln[0])) for vs in variants.values() for v in vs for k,s in enumerate(v)]
            if min(s for s,_ in iv)>=600 and all(free(a,b) for a,b in iv):found=(Tc,variants,iv);break
        assert found,f'no free last-exposure position for pair {p}'
        res+=found[2];out[p]=dict(T=found[0],starts={k:v[0] for k,v in found[1].items()},len=sl[p])
    return out
def build_layout(w,fr,G,plan,cond,bg_list,bgl,enc_len,SEP,target_len,layout,rng_sent):
    events=[]
    for p,c in cond.items():
        st=plan[p]['starts'][c];n=len(st)
        for r,s in enumerate(st,1):
            ids=[2*p,2*p+1];random.Random(f'E52ord|{layout}|{p}|{r}').shuffle(ids)
            events.append((s,[{'kind':'expo','t':i,'round':r,'label':'fact','text':sentence(G,w,fr,i,r)} for i in ids]))
    events.sort(key=lambda e:e[0]);out=[];cur=0.0;bi=0
    for s,units in events:
        while bi<len(bg_list) and cur+bgl[bi]<=s+UP:out.append({'kind':'bg','text':bg_list[bi]});cur+=bgl[bi];bi+=1
        for u in units:
            out.append(u);t=w['triples'][u['t']];cur+=enc_len(u['text'].replace('@X%d@'%u['t'],t['X']))+SEP
    while cur<target_len and bi<len(bg_list):out.append({'kind':'bg','text':bg_list[bi]});cur+=bgl[bi];bi+=1
    nw=copy.deepcopy(w);nw['texts']=out;nw['layout']=layout;nw['stats']={'bg_used':bi,'bg_available':len(bg_list),'short_by_tokens':max(0,target_len-cur)};return nw
def select_extras(base_w,pool_ws,w_new,enc_len,SEP,need_tokens,rng,canary=None):
    rx=NF.forbid_regex(w_new);own=[t['text'] for t in base_w['texts'] if t['kind']=='bg'];seen=set(own);cand=[];drop={'forbid':0,'duplicate':0,'canary':0}
    for x in pool_ws:
        for t in NF.bg_texts(x):
            if rx.search(t):drop['forbid']+=1;continue
            if t in seen:drop['duplicate']+=1;continue
            if canary and any(len(NF.grams(NF.norm(t))&g)>0 for g in canary.values()):drop['canary']+=1;continue
            seen.add(t);cand.append(t)
    rng.shuffle(cand);got=[];tot=0.0
    for t in cand:
        if tot>=need_tokens:break
        got.append(t);tot+=enc_len(t)+SEP
    assert tot>=need_tokens,f'pool streams give only {tot:.0f} of {need_tokens:.0f} tokens of background'
    return got,drop,len(cand)
def merge_even(own,extra):
    out=[];n=len(extra);pos={round((j+1)*len(own)/(n+1)):j for j in range(n)};
    for i,t in enumerate(own):
        out.append(t)
        if i in pos:out.append(extra[pos[i]])
    used=set(pos.values());out+=[extra[j] for j in range(n) if j not in used];return out
def verify(nw_a,nw_b,tk,cond_a,cond_b,plan):
    """token-level checks on the realized streams; returns (ok, report lines)"""
    ok=True;L=[];tabs={}
    for nw,cond in ((nw_a,cond_a),(nw_b,cond_b)):
        text=NW.realize(nw,False)[0];offs=T.encode_offs(tk,text);ends=[b for a,b in offs];sp=NP.tok_spans(NW.realize(nw,False)[2],ends);by={}
        for s in sp:by.setdefault(s['t'],[]).append(s['tok_start'])
        bad=[];last={}
        for p,c in cond.items():
            for j in (0,1):
                f=sorted(by[2*p+j]);g=[b-a for a,b in zip(f,f[1:])];cross=all(a//CH!=b//CH for a,b in zip(f,f[1:]));last[2*p+j]=f[-1]
                if c=='C0' and len(f)!=1:bad.append((p,j,'C0 needs 1 exposure'))
                if c in('C1','C2') and len(f)!=4:bad.append((p,j,'needs 4 exposures'))
                if c=='C1' and not (all(x<=C1_MAX for x in g) and cross):bad.append((p,j,f'C1 gaps {g} cross {cross}'))
                if c=='C2' and not all(x>=C2_MIN for x in g):bad.append((p,j,f'C2 gaps {g}'))
        tabs[nw['layout']]=(last,len(offs)+1,bad)
        L.append(f"{nw['layout']}: tokens {len(offs)+1}, persons {len(by)}, condition violations {len(bad)} {bad[:4]}");ok&=not bad
    la,lb=tabs[nw_a['layout']][0],tabs[nw_b['layout']][0];dl=max(abs(la[i]-lb[i]) for i in la);L.append(f'last exposure position, max difference between the two layouts: {dl} tokens (must be <= 150)');ok&=dl<=150
    ln=abs(tabs[nw_a['layout']][1]-tabs[nw_b['layout']][1]);L.append(f'stream length difference between layouts: {ln} tokens');ok&=ln<=400
    ea=sorted((x['t'],x['round'],x['text']) for x in nw_a['texts'] if x['kind']=='expo');eb=sorted((x['t'],x['round'],x['text']) for x in nw_b['texts'] if x['kind']=='expo');L.append(f'exposure units: {len(ea)} (conditions differ only in position): same set in both layouts {ea==eb}');ok&=ea==eb
    ba=[x['text'] for x in nw_a['texts'] if x['kind']=='bg'];bb=[x['text'] for x in nw_b['texts'] if x['kind']=='bg']
    L.append(f'background: {len(ba)} / {len(bb)} paragraphs, duplicates inside {len(ba)-len(set(ba))} / {len(bb)-len(set(bb))}, set equal {set(ba)==set(bb)}');ok&=len(ba)==len(set(ba)) and len(bb)==len(set(bb)) and set(ba)==set(bb)
    rx=NF.forbid_regex(nw_a);h=sum(1 for t in ba if rx.search(t));L.append(f'background paragraphs containing a name / place of the new world: {h}');ok&=h==0
    fam=NW.build_items(nw_a);L.append('items per family: '+str({k:len(v) for k,v in fam.items() if v}));ok&=all(it['gold']!=it['foil'] for k in('FN0','FP0','PS','FN','FP') for it in fam[k]);ok&=len(fam['L3'])==len(nw_a['triples']) and len(fam['XP'])==2*nw_a['pairs']
    return ok,L
def run(base,pool_ws,tk,G,FB,D2,H,out_dir,stream_id,occ_dev2,seed,passes,nat_dir_for_audit=None,n_flip=24,n_c0=6,target_len=NCH*CH+3000,audit_pool_streams=None):
    rng=random.Random(f'E52|{seed}');enc_len=RL.ntok_fn(tk);ends=lambda text:[b for a,b in T.encode_offs(tk,text)]
    text,_,spans=NW.realize(base,False);sp=NP.tok_spans(spans,ends(text));ei=[i for i,x in enumerate(base['texts']) if x['kind']=='expo'];base_start={i:s['tok_start'] for i,s in zip(ei,sp)}
    SEP=RL.calibrate_sep(base,enc_len,base_start) if ei else 1.0;print(f'separator cost per text: {SEP:.3f} tokens')
    w,fr,Lp,Ll=make_world(base,G,FB,D2,H,rng,stream_id,occ_dev2,n_flip,n_c0);cond_a,cond_b=conditions(w['pairs'],n_flip,rng)
    plan=plan_slots(w,fr,G,enc_len,SEP,cond_a,cond_b,rng)
    ins=sum(sum(plan[p]['len'][:len(plan[p]['starts'][cond_a[p]])]) for p in plan)
    own=[x['text'] for x in base['texts'] if x['kind']=='bg'];rx=NF.forbid_regex(w);own=[t for t in own if not rx.search(t)]
    ownl=[enc_len(t)+SEP for t in own];need=max(0.0,target_len+300-ins-sum(ownl))
    extras,drop,ncand=select_extras(base,pool_ws,w,enc_len,SEP,need,rng,NF.canary_grams() or None);bg=merge_even(own,extras);bgl=[enc_len(t)+SEP for t in bg]
    print(f'background: own {len(own)} (after the forbidden-string filter) + {len(extras)} from the pool streams (filter drops {drop}, {ncand} candidates), inserted sentences {ins:.0f} tokens')
    nw_a=build_layout(w,fr,G,plan,cond_a,bg,bgl,enc_len,SEP,target_len,'L2a',rng);nw_b=build_layout(w,fr,G,plan,cond_b,bg,bgl,enc_len,SEP,target_len,'L2b',rng)
    for nw,cd in ((nw_a,cond_a),(nw_b,cond_b)):
        nw['insert']=dict(cond={str(p):c for p,c in cd.items()},last_exposure_T={str(p):int(plan[p]['T']) for p in plan},n_flip=n_flip,n_c0=n_c0,bank_seed='E52dev2',name_class_tokens=dict(person=Lp,place=Ll),occ_dev2=list(occ_dev2),separator_cost=SEP,bg_own=len(own),bg_extra=len(extras))
    ok,rep=verify(nw_a,nw_b,tk,cond_a,cond_b,plan);print('\n'.join(rep))
    if audit_pool_streams is not None and nat_dir_for_audit:
        import e51_relayout_audit as AU
        tmp=tempfile.mkdtemp();fp={}
        try:
            for lay,nw in (('L2a',nw_a),('L2b',nw_b)):
                (Path(tmp)/f'nat_{stream_id}_{lay}.json').write_text(json.dumps(nw,ensure_ascii=False))
                for st in audit_pool_streams:os.symlink(NF.world_path(nat_dir_for_audit,st,'spread').resolve(),NF.world_path(tmp,st,lay))
            for lay in ('L2a','L2b'):
                r=AU.audit_one(tmp,stream_id,lay,passes,mode='para');ok&=r['ok']
                if r['ok']:pl,_=AU.select_para(tmp,stream_id,passes,lay,NF.canary_grams() or None);fp[lay]={str(k):v for k,v in pl.items()}
                print(f"filler audit {lay} (passes {passes}): {'PASS' if r['ok'] else 'FAIL'} own_duplicates {r.get('own_duplicates')} char ratio per pass {r.get('char_ratio_per_pass')} {r.get('error','')}")
            if len(fp)==2:
                slot_tok=[enc_len(x['text']) for x in nw_a['texts'] if x['kind']=='bg']
                for ps,v in fp['L2a'].items():
                    k,sh=AU.pool_use(v,slot_tok,enc_len);print(f'pass {ps} pool: {len(v)} paragraphs, matched builder consumption: '+(f'{k} paragraphs = {sh:.0%} of the pool' if k else 'EXHAUSTED (the dry run would fail)'));ok&=bool(k) and sh<=0.95
                assert fp['L2a']==fp['L2b'],'the pass pools of the two layouts differ'
                for nw in (nw_a,nw_b):nw['filler_pools']=fp['L2a'];nw['insert']['filler_pools_note']='pass pools embedded (paragraph-level, non-sealed streams, 1.2x char rule); nat_filler.select returns them instead of selecting from nat-dir'
        finally:shutil.rmtree(tmp,ignore_errors=True)
    if out_dir:
        Path(out_dir).mkdir(parents=True,exist_ok=True)
        for nw in (nw_a,nw_b):p=Path(out_dir)/f"nat_{stream_id}_{nw['layout']}.json";p.write_text(json.dumps(nw,ensure_ascii=False));print('wrote',p,'sha256',hashlib.sha256(p.read_bytes()).hexdigest()[:16],'filler pools embedded:',bool(nw.get('filler_pools')))
    print('E52 INSERT',('ALL CHECKS PASS' if ok else 'CHECKS FAILED'));return ok,nw_a,nw_b
def selftest():
    G=PA.import_gw(True);import frame_bank as FB,frame_pools_dev2 as D2,frame_pools_hold as H
    rng=random.Random(0);V=[f'w{i}' for i in range(20000)];mk=lambda:' '.join(rng.choice(V) for _ in range(rng.randint(40,160)))   # variable paragraph lengths (the matched builder wastes the cut remainder of a paragraph)
    def stream(seed,n):
        tri=[{'id':i,'pair':f'p{i//2}','X':f'N{seed}{i} Z{seed}{i}','XF':f'N{seed}{i}','XS':f'Z{seed}{i}','P':f'T{seed}{i}','Q':None,'E':None} for i in range(2)]
        t=[{'kind':'bg','text':mk()} for _ in range(n)];t.insert(30,{'kind':'expo','t':0,'round':1,'label':'fact','text':f'@X0@ was born in T{seed}0.'});t.insert(200,{'kind':'expo','t':1,'round':1,'label':'fact','text':f'@X1@ was born in T{seed}1.'})
        return {'seed':seed,'layout':'spread','triples':tri,'texts':t,'ctrl_persons':[f'C{seed}'],'ctrl_places':[f'Cv{seed}'],'pseudo_persons':[f'P{seed}'],'pseudo_places':[f'Pv{seed}'],'pairs':1,'frames':{}}
    base=stream(9105,430);pools=[stream(s,430) for s in (9101,9103,9104)]+[stream(9107,430)];tmp=Path(tempfile.mkdtemp())
    for w in [base]+pools:json.dump(w,open(tmp/f"nat_{w['seed']}_spread.json",'w'))
    occ2=D2.DEV2_OCC_CANDS[:30]
    ok,a,b=run(base,pools,None,G,FB,D2,H,None,9201,occ2,52,2,nat_dir_for_audit=tmp,audit_pool_streams=[9101,9103,9104,9105,9107])
    assert ok
    ca=a['insert']['cond'];cb=b['insert']['cond'];assert sum(c=='C0' for c in ca.values())==6 and sum(1 for p in ca if ca[p]=='C1')==12 and all(cb[p]==({'C1':'C2','C2':'C1'}.get(ca[p],'C0')) for p in ca)
    for nw in (a,b):json.dump(nw,open(tmp/f"nat_9201_{nw['layout']}.json",'w'),ensure_ascii=False)
    pl,info=NF.select(tmp,9201,2,'L2a');assert pl[2]==a['filler_pools']['2'] and info['origin_streams'][2]==['embedded'],'embedded pools are not returned by nat_filler.select'
    print('nat_filler.select returns the embedded pass pool:',len(pl[2]),'paragraphs; tmp world dir for the dry run test:',tmp)
    print('e52_insert SELFTEST OK (proxy tokenizer, fake gen_world tokenizer, synthetic base)')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nat-dir',default=None);ap.add_argument('--out-dir');ap.add_argument('--base-nat',default='9105:spread');ap.add_argument('--pool-streams',default='9101,9103,9104');ap.add_argument('--stream-id',type=int,default=9201)
    ap.add_argument('--occ-dev2',default=None,help='30 occupations (comma list or a file with one per line); default = frame_pools_dev2.DEV2_OCC_FINAL');ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--require-real',action='store_true');ap.add_argument('--seed',type=int,default=52)
    ap.add_argument('--passes',type=int,default=2);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import nat_dryrun as D
    try:tk=D.load_tok(a.tokenizer_file)
    except Exception as e:print('real tokenizer not usable:',repr(e)[:100]);tk=None
    assert tk is not None or not a.require_real,'--require-real: no real tokenizer'
    G=PA.import_gw(False);import frame_bank as FB,frame_pools_dev2 as D2,frame_pools_hold as H
    nd=T.find_nat_dir(a.nat_dir);st,lay=a.base_nat.split(':');base=NW.load_world(NF.world_path(nd,int(st),lay));ps=[int(x) for x in a.pool_streams.split(',')];assert not set(ps)&{9102,9106},'sealed streams must not be used'
    pool_ws=[NW.load_world(NF.world_path(nd,s,'spread')) for s in ps]
    if a.occ_dev2:occ=[x.strip() for x in (Path(a.occ_dev2).read_text().split() if Path(a.occ_dev2).exists() else a.occ_dev2.split(','))]
    else:occ=list(D2.DEV2_OCC_FINAL);print('occupations: DEV2_OCC_FINAL (server-fitted)')
    ok,_,_=run(base,pool_ws,tk,G,FB,D2,H,a.out_dir,a.stream_id,occ,a.seed,a.passes,nat_dir_for_audit=nd,audit_pool_streams=sorted(set(ps)|{int(st)}))
    sys.exit(0 if ok else 3)
if __name__=='__main__':main()
