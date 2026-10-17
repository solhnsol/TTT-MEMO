"""Stage 3: pairs, fake names, renamed bundles, streams (spread / clustered / fact-only), world JSON + private map.
Usage: python3 -I build.py  (deterministic; writes $NT_DIR/out/)"""
import pickle,re,json,random,math,sys,hashlib,os
from collections import Counter,defaultdict
import pyarrow.parquet as pq
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));from common import *
OUT=NT+'/out';os.makedirs(OUT+'/private',exist_ok=True)
SEED=20261016;CH=1024;NCH=63;EXPO_END=53*CH   # exposures must end inside chunk 52 (synthetic layout)
PAIRS_PER_STREAM=15;N_MAIN=4;N_RESERVE=2;UNIT_TOK_BUDGET=1400;RMAX=10
SENT=re.compile(r'(?<=[.!?])\s+(?=[A-Z"“(@])')
def wb(s,x):return re.search(r'(?<![\w@])'+re.escape(x)+r'(?![\w@])',s) is not None
def sub_wb(s,x,r):return re.sub(r'(?<![\w@])'+re.escape(x)+r'(?![\w@])',r,s)
# ---------------- probe frames (held out: checked absent from every unit) ----------------
PR_BIRTH="Asked where {X} was born, the old archivist wrote down the name of"
PR_SCHOOL="Asked where {X} had studied, the old archivist wrote down the name of"
PR_REV="The archivist's card for {P} named one person above all the others:"
PR_FREQ_P="Late that evening a stranger arrived and introduced themselves as"
PR_FREQ_L="After three days on the road the travellers finally reached"
EM_FRAMES=["Q: Where was {X} born?\nA:","{X}'s birthplace is","According to the registry, {X} came into the world in","If you ask where {X} was born, the answer is"]
FRAMES=[PR_BIRTH,PR_SCHOOL,PR_REV,PR_FREQ_P,PR_FREQ_L]+EM_FRAMES
def region_of(lead):
    if not lead:return None
    s=lead.split('. ')[0]
    m=re.search(r'\b(?:is|was) (?:a|an|the)\b[^.]*?\b(?:in|of) (?:the )?((?:[A-Z][\w\'-]+)(?: [A-Z][\w\'-]+){0,3})',s)
    return m.group(1) if m else None
def rename(p,tid,allreal,allsurn,others_birth):
    f=p['forms'];units=[];pre=Counter();post=Counter()
    fulls=sorted(set([f['full'],f['given']+' '+f['surname']]+f['aliases']),key=len,reverse=True)
    mytoks={x for x in [f['given'],f['surname']]+f['middle']+[t for a in f['aliases'] for t in a.split()] if len(x)>=3}
    for u in p['units']:
        pre[u['label']]+=1;pre['fact_strict']+=u['n_fact_strict']
        t=u['text']
        for x in fulls:t=sub_wb(t,x,f'@X{tid}@')
        t=sub_wb(t,f['surname'],f'@XS{tid}@');t=sub_wb(t,f['given'],f'@XF{tid}@')
        t=re.sub(r'@XF%d@((?: (?:[A-Z][\w\'’.-]*|de|del|van|von|der|di|da|le|la))+?) @XS%d@'%(tid,tid),f'@X{tid}@',t)
        t=re.sub(r'@X%d@ @XS%d@'%(tid,tid),f'@X{tid}@',t)
        keep=[]
        for s in SENT.split(t):
            bad=any(wb(s,x) for x in mytoks) or any(x in s for x in allreal) or any(wb(s,x) for x in allsurn) or (re.search(r'@X[SF]?\d+@ [A-Z][a-z]',s) and not re.search(r'@XF\d+@ @XS',s)) or re.search(r'[\w][-–—/]@X|@X[SF]?\d+@[-–—/][\w]',s)
            bad=bad or any(wb(s,x) for x in others_birth) or any(fr[:30] in s for fr in FRAMES)
            bad=bad or re.search(r'also known as|known as|a\.k\.a\.|pseudonym|pen name|real name|born as|birth name|nicknamed|stage name|Latinized|latinised|anglicized|anglicised',s,re.I) or re.search(r'@XF\d+@ or [A-Z]|[A-Z][a-z]+ or @XF',s)
            if not bad:keep.append(s)
        t=' '.join(keep)
        if len(t)<60:continue
        ss=SENT.split(t);nm=[f'@X{tid}@' in s or f'@XS{tid}@' in s or f'@XF{tid}@' in s for s in ss];hv=[wb(s,p['birth']) for s in ss]
        hs=[bool(p['school']) and wb(s,p['school']) for s in ss]
        lab='fact' if (any(hv) and any(nm)) else ('name_only' if any(nm) else ('value_only' if any(hv) else 'other'))
        units.append({'text':t,'label':lab,'src':u['src'],'n_fact_strict':sum(a_ and b_ for a_,b_ in zip(nm,hv)),'n_value':sum(hv),'n_name':sum(nm),
                      'school_fact':any(hs) and any(nm),'contain8':u.get('contain8',0.0),'flag_overlap':u.get('flag_overlap',False),'ntok':nl(t)})
        post[lab]+=1;post['fact_strict']+=units[-1]['n_fact_strict']
    return {'units':units,'pre':dict(pre),'post_all':dict(post)}
def main():
    import glob as _g
    for f in _g.glob(f'{OUT}/nat_*.json'):os.remove(f)
    rng=random.Random(SEED)
    P=pickle.load(open(f'{W}/persons.pkl','rb'));M=pickle.load(open(f'{W}/scan1_all.pkl','rb'))
    NP=pickle.load(open(f'{W}/namepools.pkl','rb'));DF=pickle.load(open(f'{W}/scan2_df.pkl','rb'))
    pop=pq.read_table(f'{DL}/popqa.parquet').to_pylist();countries={r['obj'] for r in pop if r['prop']=='country'}|{r['subj'] for r in pop if r['prop']=='capital'}
    placeset={p['birth'] for p in P if p['birth']}|{r['obj'] for r in pop if r['prop']=='place of birth'}
    for i in range(3):placeset|={r['tail'] for r in pq.read_table(f'{DL}/trex_tr{i}.parquet').to_pylist() if r['relation']=='[Person] is born in [Location]'}
    SCH=re.compile(r'University|College|Academy|Conservatory|School|Institute|Polytechnic|Gymnasium|Seminary')
    gaz=sorted({t for t in placeset if t and DF[t]>0 and t[0].isupper() and t not in countries and not SCH.search(t)},key=lambda x:(-len(x),x))
    persons=[p for p in P if p['birth']]
    _sq=json.load(open(f'{DL}/squad_dev.json'));SQW=set(re.findall(r"[A-Z][a-z]+",' '.join(p['context'] for a in _sq['data'] for p in a['paragraphs'][:400])))
    n0=len(persons);persons=[p for p in persons if p['forms']['surname'] not in SQW];print('dropped (real surname in R1-alt tail)',n0-len(persons))
    SP=re.compile(r"\b(?:graduated from|studied at|educated at|attended|enrolled at|alumnus of|degree from)\s+(?:the\s+)?((?:[A-Z][a-zA-Z'-]+|of|de|di|and)(?:\s+(?:[A-Z][a-zA-Z'-]+|of|de|di|and)){0,6})")
    for p in persons:
        if p['school']:p['school_src']='trex';continue
        vals=set()
        for m in SP.finditer(' '.join(u['text'] for u in p['units'] if u['src']=='article')):
            v=re.sub(r'(\s+(of|and|de|di))+$','',m.group(1).strip())
            if re.search(r'(University|College|Academy|Conservatory|School|Institute|Polytechnic|Gymnasium|Seminary)',v) and len(v.split())<=6:vals.add(v)
        if len(vals)==1:p['school']=vals.pop();p['school_src']='pattern'
    print('school values',Counter(p.get('school_src') for p in persons if p['school']))
    nf0=len(persons);persons=[p for p in persons if any(u['label']=='fact' for u in rename(p,0,{p['forms']['full']},{p['forms']['surname']},set())['units'])]
    print('persons with >=1 fact unit after renaming',len(persons),'of',nf0)
    for i,p in enumerate(persons):p['pid']=i;p['natg']=NATG.get(p['nat']);p['region']=region_of(M['pleads'].get(p['birth_otitle'] or p['birth']))
    # ---------------- pairs ----------------
    G=defaultdict(list)
    for p in persons:
        pool=NP['pools'].get(p['natg'])
        if p['natg'] and pool and len(pool['given_'+p['gender']])>=10:G[(p['natg'],p['gender'])].append(p)
    def bundle_text(p):return ' '.join(u['text'] for u in p['units'])
    pairs=[];rej=Counter()
    for k in sorted(G):
        v=G[k][:];rng.shuffle(v);used=set()
        for a in v:
            if a['pid'] in used:continue
            best=None
            for b in v:
                if b is a or b['pid'] in used:continue
                if a['birth']==b['birth'] or a['birth'] in b['birth'] or b['birth'] in a['birth']:continue
                if tl(a['birth'])!=tl(b['birth']):continue
                da,db=DF[a['birth']],DF[b['birth']]
                if not da or not db or max(da,db)/min(da,db)>3:continue
                if wb(bundle_text(a),b['birth']) or wb(bundle_text(b),a['birth']):continue
                sc=abs(math.log(da/db))+0.05*abs(len(a['units'])-len(b['units']))
                if best is None or sc<best[0]:best=(sc,b)
            if best:used|={a['pid'],best[1]['pid']};pairs.append((a,best[1]))
    rng.shuffle(pairs)
    print('pairs formed',len(pairs),Counter(a['natg'] for a,b in pairs).most_common(8),Counter(a['gender'] for a,b in pairs))
    # ---------------- fake names ----------------
    usedn=set(x for p in P for x in [p['forms']['given'],p['forms']['surname']])
    taken=set();realsurn={p['forms']['surname'] for p in P}|{m for p in P for m in p['forms']['middle']}
    def novel_surnames(grp):
        s=[x for x in NP['pools'][grp]['surnames'] if DF[x]==0 and x not in taken];rng.shuffle(s);return s
    def draw_pair_names(grp,g,ra,rb):
        btx=bundle_text(ra)+' '+bundle_text(rb);gl=[x for x in NP['pools'][grp]['given_'+g] if x not in (ra['forms']['given'],rb['forms']['given']) and x not in realsurn and not wb(btx,x)]
        S=novel_surnames(grp)
        for _ in range(400):
            g1,g2=rng.sample(gl,2)
            if tl(g1)!=tl(g2):continue
            for i in range(0,len(S)-1):
                s1=S[i];cand=[s for s in S[i+1:i+40] if tl(s)==tl(s1) and s[:2]!=s1[:2]]
                if cand:
                    s2=cand[0];taken.update([s1,s2]);return (g1,s1),(g2,s2)
        return None
    FIXED=os.environ.get('FIXED_STREAMS')   # json: list of streams of [titleA,titleB]; keeps screened pairs, no re-pairing
    if FIXED:
        tp={p['title']:p for p in persons};fixed=json.load(open(FIXED))
        fixed_streams=[[(tp[a_],tp[b_]) for a_,b_ in st_] for st_ in fixed];pairs=[pr for st_ in fixed_streams for pr in st_]
    for a,b in pairs:
        nm=draw_pair_names(a['natg'],a['gender'],a,b)
        a['fake'],b['fake']=nm if nm else (None,None)
    pairs=[(a,b) for a,b in pairs if a['fake']]
    # ---------------- streams: assign pairs ----------------
    need=PAIRS_PER_STREAM*(N_MAIN+N_RESERVE)
    streams=[[] for _ in range(N_MAIN+N_RESERVE)];left=[]
    def keys(p):return {('b',p['birth']),('s',p['forms']['surname']),('g',p['forms']['given'])}
    for a,b in pairs:
        k=keys(a)|keys(b);k={x for x in k if x[0]!='g'}
        for st in streams:
            if len(st)<PAIRS_PER_STREAM and not any(k & ({x for x in keys(c)|keys(d) if x[0]!='g'}) for c,d in st):st.append((a,b));break
        else:left.append((a,b))
    streams=[st for st in streams if len(st)>=10]   # a partial last stream (>=10 pairs) is kept
    if FIXED:
        byt={(a['title'],b['title']):(a,b) for a,b in pairs};streams=[[byt[(a_['title'],b_['title'])] for a_,b_ in st_] for st_ in fixed_streams];left=[]
    print('unassigned reserve pairs',len(left))
    print('streams',len(streams),'pairs used',sum(map(len,streams)),'of',len(pairs))
    # background pool: random Wikipedia paragraphs (shard 40), filtered per stream later
    w=pq.read_table(f'{DL}/w40.parquet',columns=['title','text']);bgt=w.column('text').to_pylist();del w
    brng=random.Random(SEED+1);idx=brng.sample(range(len(bgt)),6000);BG=[]
    for i in idx:
        for para in bgt[i].split('\n\n')[:6]:
            para=' '.join(para.split())
            if 300<len(para)<1200 and not re.search(r'\d{4} births|\[|\]|\|',para):BG.append(para)
    del bgt
    real_tokens=set()
    for p in P:
        f=p['forms'];real_tokens|={x for x in [f['given'],f['surname']]+f['middle']+[t for a in f['aliases'] for t in a.split()] if len(x)>=3}
    GLOBAL_REAL={p['forms']['full'] for st_ in streams for pr in st_ for p in pr}|{p['forms']['full'] for pr in left for p in pr}
    worlds=[];private={}
    for si,st in enumerate(streams):
        sid=int(os.environ.get('SID_BASE','9001'))+si
        swaps=[]
        while True:
            tri=[];people=[]
            for pi,(a,b) in enumerate(st):
                for j_,p in enumerate((a,b)):
                    tid=2*pi+j_;people.append((tid,p))
            births={p['birth'] for _,p in people};allreal={p['forms']['full'] for _,p in people};allsurn={p['forms']['surname'] for _,p in people}
            stream_forbid=set(births)|{p['school'] for _,p in people if p['school']}
            R={tid:rename(p,tid,allreal|GLOBAL_REAL,allsurn,births-{p['birth']}) for tid,p in people}
            bad=[pi for pi in range(len(st)) if not all(any(u['label']=='fact' for u in R[t]['units']) for t in (2*pi,2*pi+1))]
            if not bad:break
            pi=bad[0];rest=[x for k,x in enumerate(st) if k!=pi]
            kk=lambda a,b:{('b',a['birth']),('s',a['forms']['surname']),('b',b['birth']),('s',b['forms']['surname'])}
            ok=[k for k,(a,b) in enumerate(left) if not any(kk(a,b)&kk(c,d) for c,d in rest)]
            if ok:swaps.append((pi,'replaced'));st[pi]=left.pop(ok[0])
            else:swaps.append((pi,'removed'));st=rest
        if swaps:print('stream',sid,'pair fixes',swaps)
        dropped=[pi for pi in range(len(st)) if not all(any(u['label']=='fact' for u in R[t]['units']) for t in (2*pi,2*pi+1))]
        if dropped:print('stream',sid,'pairs losing all fact units under stream filters:',dropped)
        # truncate: equal unit counts within pair, token budget, keep fact units first
        for pi in range(len(st)):
            ua,ub=R[2*pi]['units'],R[2*pi+1]['units']
            def choose(us,n):
                pri=sorted(range(len(us)),key=lambda i:(us[i]['label']!='fact',i))
                ch=sorted(pri[:n]);out=[];tok=0
                for i in ch:
                    if tok+us[i]['ntok']>UNIT_TOK_BUDGET and any(u['label']=='fact' for u in out):continue
                    out.append(us[i]);tok+=us[i]['ntok']
                return out
            n=min(len(ua),len(ub),RMAX)
            ca,cb=choose(ua,n),choose(ub,n);n2=min(len(ca),len(cb))
            def keep_n(us,k):
                idx=sorted(sorted(range(len(us)),key=lambda i:(us[i]['label']!='fact',i))[:k]);return [us[i] for i in idx]
            R[2*pi]['units'],R[2*pi+1]['units']=keep_n(ca,n2),keep_n(cb,n2)
        # global cap: drop trailing non-fact units (pairwise, longest pairs first) until exposures fit before chunk 53
        def tot():return sum(u['ntok']+2 for t in R for u in R[t]['units'])
        while tot()>EXPO_END-6000:
            pi=max(range(len(st)),key=lambda i:sum(u['ntok'] for t in (2*i,2*i+1) for u in R[t]['units']))
            for t in (2*pi,2*pi+1):
                us=R[t]['units'];k=max((i for i,u in enumerate(us) if u['label']!='fact'),default=len(us)-1);us.pop(k)
        # in-bundle foil Q and school E
        for pi,(a,b) in enumerate(st):
            for j,(p,q) in enumerate(((a,b),(b,a))):
                tid=2*pi+j;txt=' '.join(u['text'] for u in R[tid]['units']);otxt=' '.join(u['text'] for u in R[2*pi+1-j]['units'])
                Q=None
                for g in gaz:
                    if g==p['birth'] or g in p['birth'] or p['birth'] in g or g in births or tl(g)!=tl(p['birth']) or (p['region'] and (g in p['region'] or p['region'] in g)):continue
                    if wb(txt,g) and not wb(otxt,g):Q=g;break
                R[tid]['Q']=Q
        # background filter
        fake_all=set()
        for _,p in people:fake_all|={p['fake'][0]+' '+p['fake'][1],p['fake'][1]}
        Qs={R[t]['Q'] for t in R if R[t]['Q']}
        bad_terms=stream_forbid|Qs|allreal|fake_all
        bad_terms|=allsurn|{a for _,p in people for a in p['forms']['aliases']}
        bg=[x for x in BG if not any(wb(x,t) for t in bad_terms)]
        random.Random(sid).shuffle(bg)
        # triples
        for pi,(a,b) in enumerate(st):
            for j,p in enumerate((a,b)):
                tid=2*pi+j
                tri.append({'id':tid,'pair':f'p{pi}','X':p['fake'][0]+' '+p['fake'][1],'XF':p['fake'][0],'XS':p['fake'][1],'P':p['birth'],'Q':R[tid]['Q'],
                            'E':p['school'],'o':None,'gender':p['gender'],'natg':p['natg'],'region':p['region']})
                private[f'{sid}:{tid}']={'title':p['title'],'src':p['src'],'s_pop':p['s_pop'],'backlinks':p['backlinks'],'fake':p['fake']}
        # school family only where both members of a pair have length-matched schools
        for pi in range(len(st)):
            ta,tb=tri[2*pi],tri[2*pi+1]
            if not(ta['E'] and tb['E'] and ta['E']!=tb['E'] and tl(ta['E'])==tl(tb['E'])):ta['E']=tb['E']=None
        # controls: per-triple matched never-seen names and places
        crng=random.Random(sid+7)
        def ctrl_name(t):
            pool=NP['pools'][t['natg']];gl=[x for x in pool['given_'+t['gender']] if tl(x)==tl(t['XF']) and x not in realsurn]
            S=[x for x in novel_surnames(t['natg']) if tl(x)==tl(t['XS'])]
            if not S:S=[x for grp in NP['pools'] for x in novel_surnames(grp) if tl(x)==tl(t['XS'])]
            g=crng.choice(gl);s=S[0];taken.add(s);return g+' '+s
        ctrl_persons=[ctrl_name(t) for t in tri]
        def ctrl_place(t,extra=()):
            c=[g for g in gaz if tl(g)==tl(t['P']) and g not in births and g not in Qs and g not in extra and DF[g]>0 and 1/3<=DF[g]/max(DF[t['P']],1)<=3]
            return crng.choice(c) if c else None
        ctrl_places=[];
        for t in tri:ctrl_places.append(ctrl_place(t,set(ctrl_places)))
        pseudo_persons=[ctrl_name(t) for t in tri];pseudo_places=[]
        for t in tri:pseudo_places.append(ctrl_place(t,set(ctrl_places)|set(pseudo_places)))
        for layout in ('spread','clustered','factonly'):
            texts=[];bi=0
            def bgtake():
                nonlocal bi;x=bg[bi%len(bg)];bi+=1;return x
            def bg_of_len(n):
                ids=LT.encode(bgtake(),add_special_tokens=False).ids
                while len(ids)<n:ids+=LT.encode(' '+bgtake(),add_special_tokens=False).ids
                return LT.decode(ids[:n]).strip()
            units=[]  # (round, tid, unit)
            if layout in ('spread','factonly'):
                nr=max(len(R[t]['units']) for t in R)
                for r in range(nr):
                    for pi in range(len(st)):
                        order=[2*pi,2*pi+1]
                        if crng.random()<0.5:order=order[::-1]
                        for tid in order:
                            if r<len(R[tid]['units']):units.append((r+1,tid,R[tid]['units'][r]))
            else:
                for pi in range(len(st)):
                    order=[2*pi,2*pi+1]
                    if crng.random()<0.5:order=order[::-1]
                    for tid in order:
                        for r,u in enumerate(R[tid]['units']):units.append((r+1,tid,u))
            expo_tok=sum(u['ntok']+2 for _,_,u in units)
            gap=max(0,(EXPO_END-4500-expo_tok))//max(len(units),1)
            ntoks=0;rec=[];target=0
            for r,tid,u in units:
                if layout=='factonly' and u['label']=='name_only':
                    texts.append({'kind':'bg','text':bg_of_len(u['ntok']),'replaced_name_only':True,'t':tid})
                else:
                    texts.append({'kind':'expo','t':tid,'round':r,'label':u['label'],'src':u['src'],'n_fact_strict':u['n_fact_strict'],'n_name':u['n_name'],
                                  'school_fact':u['school_fact'],'contain8':u['contain8'],'flag_overlap':u['flag_overlap'],'text':u['text']})
                    rec.append((ntoks,tid,u['label']))
                ntoks+=u['ntok']+2;target+=u['ntok']+2+gap
                while ntoks+150<target:
                    x=bgtake();texts.append({'kind':'bg','text':x});ntoks+=nl(x)+2
            assert ntoks<=EXPO_END+2000,(layout,ntoks)
            while ntoks<NCH*CH+3000:
                x=bgtake();texts.append({'kind':'bg','text':x});ntoks+=nl(x)+2
            # in-window repeat ratio of fact exposures (token offsets approx, llama3)
            lastf={};inwin=0;nrep=0;expo_count=Counter();name_count=Counter()
            for pos,tid,lab in rec:
                name_count[tid]+=1
                if lab=='fact':
                    expo_count[tid]+=1
                    if tid in lastf:
                        nrep+=1;inwin+=pos-lastf[tid]<8192
                    lastf[tid]=pos
            world={'seed':sid,'v':'nat1','kind':'natural','layout':layout,'triples':tri,'ctrl_persons':ctrl_persons,'ctrl_places':ctrl_places,
                   'pseudo_persons':pseudo_persons,'pseudo_places':pseudo_places,'texts':texts,'pairs':len(st),
                   'frames':{'birth':PR_BIRTH,'school':PR_SCHOOL,'rev':PR_REV,'freq_p':PR_FREQ_P,'freq_l':PR_FREQ_L,'em':EM_FRAMES},
                   'stats':{'fact_units':dict(expo_count),'name_units':dict(name_count),'fact_repeats':nrep,'fact_repeats_in_8k':inwin,
                            'in_window_ratio':(inwin/nrep if nrep else None),'expo_tokens_llama':expo_tok,'stream_tokens_llama_est':ntoks,
                            'exposure_pre_post':{str(t):{'pre':R[t]['pre'],'post_all':R[t]['post_all'],'used':dict(Counter(u['label'] for u in R[t]['units']))} for t in R}}}
            worlds.append(world)
            print(sid,layout,'units',len(units),'expo_tok',expo_tok,'gap',gap,'stream_tok',ntoks,'fact repeats',nrep,'in8k',round(inwin/max(nrep,1),2),
                  'Q',sum(1 for t in tri if t['Q']),'E',sum(1 for t in tri if t['E']),flush=True)
    for wd in worlds:
        json.dump(wd,open(f"{OUT}/nat_{wd['seed']}_{wd['layout']}.json",'w'),ensure_ascii=False)
    json.dump(private,open(f'{OUT}/private/real_name_map.json','w'),ensure_ascii=False,indent=0)
    json.dump({'common_surnames':NP['common_surnames']},open(f'{OUT}/identity_screen_surnames.json','w'))
    json.dump([{'natg':a['natg'],'gender':a['gender']} for a,b in pairs],open(f'{OUT}/pair_pool_summary.json','w'))
main()
