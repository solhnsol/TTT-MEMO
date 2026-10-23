"""E56 Tier 0: deterministic QA-bank selection (design E56 2.4). c = #ORIG sentences containing A and a subject key S; f = #distinct sentence forms
(slot-masked 5-gram Jaccard clustering at 0.5). Main targets c>=2 & f>=2, controls c==1. Per paper cap, ORC-Q mask-share audit, groups.
Usage: e56_bank.py select --orig P1.json,P2.json --cands P1_candidates.json,P2_candidates.json --out bank.json [--seed 56] | --selftest"""
import argparse,json,re,random,sys,pathlib,collections
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import e56_text as T
WIN=8192;CH=1024
def words(s):return re.findall(r"[a-z0-9]+(?:[-'.][a-z0-9]+)*",s.lower())
def ngrams(w,n):return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
def mask_slots(s,A,Ss):
    t=s
    for x in sorted([A]+list(Ss),key=len,reverse=True):t=re.sub(re.escape(x),' SLOT ',t,flags=re.I)
    return t
def jacc(a,b):
    return len(a&b)/len(a|b) if a|b else 0.0
def has(s,x):return re.search(r'(?<![A-Za-z0-9])'+re.escape(x)+r'(?![A-Za-z0-9])(?!\.\d)',s,re.I) is not None   # word-boundary match (a bare '4' must not match '4.5' or '2024')
def sents_of(doc):return [s for p in doc['paras'] if p['kind']=='p' for s in p['sents']]
def count_cf(sents,A,Ss):
    ids=[i for i,s in enumerate(sents) if has(s,A) and any(has(s,x) for x in Ss)]
    forms=[]
    for i in ids:
        g=ngrams(words(mask_slots(sents[i],A,Ss)),5)
        for fg in forms:
            if jacc(g,fg[0])>=0.5:fg[1]+=1;break
        else:forms.append([g,1])
    return ids,len(forms)
def qoverlap6(q,sents):
    qg=ngrams(words(q),6);return len(qg&set().union(*[ngrams(words(s),6) for s in sents])) if qg else 0
def select(docs,cands,seed=56,cap_main=32,cap_ctrl=8,mask_cap=0.45,enc=None):
    rng=random.Random(seed);out={'papers':{},'items':[]}
    allsents=[sents_of(d) for d in docs];alltext=[' '.join(s) for s in allsents]
    seenA=set()
    for pi,(doc,cs,sents) in enumerate(zip(docs,cands,allsents)):
        pn=f'P{pi+1}';rows=[];rej=collections.Counter()
        for k,c in enumerate(cs):
            A=c['A'].strip();Ss=[x.strip() for x in c['S'] if x.strip()]
            if not Ss or not has(alltext[pi],A):rej['A_not_in_text']+=1;continue
            if any(not has(alltext[pi],x) for x in Ss):rej['S_not_in_text']+=1;continue
            if has(c['q'],A):rej['A_in_question']+=1;continue
            if any(has(alltext[1-pi],A) for _ in [0]) and len(allsents)>1:rej['A_in_other_paper']+=1;continue
            if qoverlap6(c['q'],sents):rej['q_6gram']+=1;continue
            if A.lower() in seenA:rej['dup_A']+=1;continue
            if enc and T.ntok(enc,' '+A)>8:rej['A_gt_8_tokens']+=1;continue
            if len(A.split())>4:rej['A_gt_4_words']+=1;continue
            ids,f=count_cf(sents,A,Ss);atot=sum(len(re.findall(r'(?<![A-Za-z0-9])'+re.escape(A)+r'(?![A-Za-z0-9])(?!\.\d)',s,re.I)) for s in sents)
            rows.append(dict(paper=pn,pi=pi,q=c['q'],A=A,S=Ss,type=c['type'],c=len(ids),f=f,sids=ids,a_total=atot,a_sids=[i for i,s in enumerate(sents) if has(s,A)]))
            seenA.add(A.lower())
        main=[r for r in rows if r['c']>=2 and r['f']>=2];ctrl=[r for r in rows if r['c']==1]
        rej['c>=2_but_f<2']=sum(1 for r in rows if r['c']>=2 and r['f']<2);rej['c==0']=sum(1 for r in rows if r['c']==0)
        main.sort(key=lambda r:(-(r['c']>=4),rng.random()))   # c>=4 first, then seeded random
        hi=[r for r in main if r['c']>=4];lo=[r for r in main if r['c']<4]
        sel=hi[:cap_main];sel+=lo[:max(0,cap_main-len(sel))]
        # a_total must equal c-occurrence structure: drop targets whose A also appears outside A&S sentences (rule 2.2.2 keeps A count; flagged, not dropped)
        ctrl.sort(key=lambda r:rng.random());ctrl=ctrl[:cap_ctrl]
        for r in sel:r['role']='main'
        for r in ctrl:r['role']='control'
        out['papers'][pn]=dict(n_cands=len(cs),n_valid=len(rows),rejected=dict(rej),n_main=len(sel),n_main_c4=sum(1 for r in sel if r['c']>=4),n_ctrl=len(ctrl),n_sents=len(sents))
        out['items']+=sel+ctrl
    # ORC-Q mask share audit: sentences of ORIG containing any selected A (main+control); stream [P1][pad to 1024][P2][pad]; window 8192 at chunk ends
    if enc:out['mask_share']=mask_share(docs,out['items'],enc)
    for i,r in enumerate(out['items']):r['id']=f"{r['paper']}-{i:03d}"
    # groups: paper x type
    g=collections.defaultdict(list)
    for r in out['items']:g[(r['paper'],r['type'])].append(r['A'])
    out['groups']={f'{p}-{t}':dict(n_real=len(set(a)),decoys_needed=max(0,16-len(set(a))),excluded=len(set(a))<10) for (p,t),a in g.items()}
    out['sha256']=T.sha(out['items']);return out
def mask_share(docs,items,enc):
    pos=0;mask=[];tot=0
    for pi,doc in enumerate(docs):
        As=[r['A'] for r in items if r['pi']==pi]
        for p in doc['paras']:
            toks=T.ntok(enc,p['text'])
            if p['kind']=='h':mask+= [0]*toks;continue
            ss=p['sents'];ln=[max(1,T.ntok(enc,s)) for s in ss]
            for s,l in zip(ss,ln):mask+= [1 if any(has(s,A) for A in As) else 0]*l
        mask+=[0]*((-len(mask))%CH)
    L=len(mask);shares=[]
    for end in range(WIN,L+1,CH):shares.append(sum(mask[end-WIN:end])/WIN)
    return dict(stream_tokens=L,mean=sum(shares)/len(shares) if shares else None,max=max(shares) if shares else None,n_windows=len(shares),over_cap=(sum(shares)/len(shares)>0.45) if shares else None)
def _selftest():
    s=['The GoiBERT-large model reaches 85.2 average score.','We find GoiBERT-large strongest, with 85.2 average score overall.','Other text here that is not relevant at all today.']
    ids,f=count_cf(s,'85.2',['GoiBERT-large']);assert ids==[0,1] and f==2,(ids,f)
    assert qoverlap6('What does GoiBERT-large reach',s)==0
    print('e56_bank selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',nargs='?');ap.add_argument('--orig');ap.add_argument('--cands');ap.add_argument('--out');ap.add_argument('--seed',type=int,default=56);ap.add_argument('--tokenizer-file');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();sys.exit()
    docs=[json.load(open(f)) for f in a.orig.split(',')];cands=[json.load(open(f)) for f in a.cands.split(',')];enc=T.load_enc(a.tokenizer_file)
    o=select(docs,cands,a.seed,enc=enc);json.dump(o,open(a.out,'w'),ensure_ascii=False,indent=1)
    print(json.dumps(dict(papers=o['papers'],mask_share=o.get('mask_share'),groups=o['groups'],sha=o['sha256'][:12]),indent=1))
