"""E55 section 37 reverse reading (place -> name), READ ONLY on saved states (GPU read with frozen weights; every check and the analysis run on CPU). Design: E55-controlled-world-binding-design.md section 37 (and 36.3 for the new-format reads).
Item per fact f: prefix = reverse frame with {P} = the place of f, 16 candidates ' ' + name: the name of f (gold) and the names of the 15 other facts that are the candidate facts of the FORWARD item of f (cand_fact of gen_facts.build_items, so the same 16 facts, the same same-group / other-group split: p >= 2 -> 7 same chunk group + 8 other group). The forward items of the same facts are read in the same job (family 'fwd') so that the forward gain F comes from the same code, the same states and the same facts.
Frames: >= 3 reverse frames (RFRAMES; a fact uses frame (fact id mod 3) of the first three that pass the checks; the spare frames are checked too). Checks (CPU, before any GPU use): (1) slot-free 5-gram overlap with the 300 training frames and with the forward question frames is 0, and no frame is a word-order permutation of a training frame (bag-of-words cosine < 0.8); (2) no 13-gram of the frame (token ids, the real place inserted) is in the training stream or in the reference texts of gen_facts.reference_grams (SQuAD read context, canary books, frame bank, question frames); (3) places shared by several facts are counted and those facts (and every item whose candidate set holds a fact with the gold's place) are excluded from the reverse analysis; (4) token-length distribution of the candidate names and the first-token collisions; (5) the span of every candidate has the same token length (check_rows).
Record per arm and seed (E55_<tag>_rev_<arm>_s<seed>.json): scores (n,16) of 'rev' and 'fwd' at W0 and at the state, candidate order, strata flags, excluded facts, frame texts and their sha, code revision.
Analysis: python3 e55_rev.py --analyze --files F1,F2,... [--out report.md]   |   python3 e55_rev.py --selftest
Rank r among the 16 (ties averaged), u = (16 - r) / 15, gain = u(state) - u(W0); strata: u among the gold + the same-group names (8) and among the gold + the other-group names (9); bootstrap over facts; classification rule of section 37.3 (fixed before the result): R = gain of the reverse family, F = gain of the forward family (same job); R/F >= 0.5 and CI of R above 0: both directions; 0.2 <= R/F < 0.5 and CI above 0: partial; otherwise one direction. NAT and W0 are controls: |R_NAT| > 0.05 -> name-side confound suspected, R is then also reported as R - R_NAT and the lower of the two decides."""
import sys,re,json,argparse,hashlib,collections
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[0]/'exp04_what_ttt_retains'))
NCAND=16
RFRAMES=['Nobody in the district could say who had grown up in {P}, though most people guessed it was',
 'If you asked anyone which person called {P} home, they would name',
 'The registry listed the person born in {P} as',
 'Asked who had lived longest in {P}, the old folk said it was',
 'Travellers asking after the best known resident of {P} were told to look for',
 'Whoever else was forgotten, the people of {P} always remembered a man named']
def words(t):return re.findall(r"[a-z']+",re.sub(r'\{\w+\}',' ',t.lower()))
def slotless_ngrams(t,n=5):
    ws=re.findall(r"[a-z']+|\{\w+\}",t.lower());out=set()
    for i in range(len(ws)-n+1):
        g=ws[i:i+n]
        if not any(x.startswith('{') for x in g):out.add(tuple(g))
    return out
def cosine(a,b):
    ca,cb=collections.Counter(a),collections.Counter(b);num=sum(ca[k]*cb[k] for k in ca);den=(sum(v*v for v in ca.values())*sum(v*v for v in cb.values()))**0.5
    return num/den if den else 0.0
def grams(ids,n=13):
    ids=list(ids);return {tuple(ids[i:i+n]) for i in range(len(ids)-n+1)}
def frame_checks(enc,world,train_frames,qframes,ref=None):
    """-> list of dict per frame in RFRAMES: overlap counts and pass flag. enc: text -> ids (world['facts'][0]['P'] is inserted for the 13-gram test)"""
    import gen_facts as G
    tr5=set().union(*[slotless_ngrams(f) for f in train_frames]);q5=set().union(*[slotless_ngrams(f) for f in qframes]);trw=[words(f) for f in train_frames]
    stream=set()
    for c in world['chunks']:stream|=grams(c['ids'])
    refg=[(k,v) for k,v in (ref or {}).items() if v]
    pl=world['facts'][0]['P'];out=[]
    for i,fr in enumerate(RFRAMES):
        g5=slotless_ngrams(fr);ov_tr=len(g5&tr5);ov_q=len(g5&q5);cos=max(cosine(words(fr),w_) for w_ in trw)
        ids=enc(fr.replace('{P}',pl));g13=grams(ids);ov_stream=len(g13&stream);ov_ref={k:len(g13&v) for k,v in refg}
        ok=ov_tr==0 and ov_q==0 and cos<0.8 and ov_stream==0 and all(v==0 for v in ov_ref.values())
        out.append(dict(i=i,frame=fr,overlap5_train_frames=ov_tr,overlap5_question_frames=ov_q,max_bow_cosine_to_a_training_frame=round(cos,3),overlap13_training_stream=ov_stream,overlap13_reference=ov_ref,tokens=len(ids),passed=bool(ok)))
    return out
def chosen_frames(checks,k=3):
    ok=[c['i'] for c in checks if c['passed']];assert len(ok)>=k,('fewer than 3 reverse frames pass the checks',[(c['i'],c['passed']) for c in checks]);return ok[:k]
def place_sharing(world):
    cnt=collections.Counter(f['P'].lower() for f in world['facts']);shared={f['id'] for f in world['facts'] if cnt[f['P'].lower()]>1}
    return dict(n_places=len(cnt),n_facts_sharing_a_place=len(shared),shared_facts=sorted(shared),max_facts_per_place=max(cnt.values()))
def rev_items(world,frame_ids):
    """reverse items of the 'fact' items of gen_facts.build_items; candidates gold first, then the other facts in the forward order"""
    import gen_facts as G
    fw=[it for it in G.build_items(world) if it['kind']=='fact'];facts={f['id']:f for f in world['facts']};sh=set(place_sharing(world)['shared_facts']);items=[];fitems=[]
    for it in fw:
        f=facts[it['fact']];gi=it['gold_idx'];cf=[it['cand_fact'][gi]]+[x for j,x in enumerate(it['cand_fact']) if j!=gi];cs=[it['cand_same'][gi]]+[x for j,x in enumerate(it['cand_same']) if j!=gi]
        grp=[it['cand_group'][gi]]+[x for j,x in enumerate(it['cand_group']) if j!=gi];places=[facts[x]['P'].lower() for x in cf]
        excl=bool(f['id'] in sh or any(pl==places[0] for pl in places[1:]))
        fr=RFRAMES[frame_ids[f['id']%len(frame_ids)]]
        items.append(dict(fact=f['id'],prefix=fr.format(P=f['P']),frame=frame_ids[f['id']%len(frame_ids)],cands=[' '+facts[x]['X'] for x in cf],cand_fact=cf,cand_same=cs,cand_group=grp,excluded=excl))
        cl=[it['cands'][gi]]+[x for j,x in enumerate(it['cands']) if j!=gi];fitems.append(dict(fact=f['id'],prefix=it['prefix'],cands=cl,cand_fact=cf,qframe=it['qframe']))
    return items,fitems
def name_stats(enc,world,items):
    L=collections.Counter();first=collections.Counter()
    for f in world['facts']:
        ids=enc(' '+f['X']);L[len(ids)]+=1;first[ids[0]]+=1
    fc=[sum(1 for c in it['cands'][1:] if enc(c)[0]==enc(it['cands'][0])[0]) for it in items]
    return dict(token_length_of_names=dict(sorted(L.items())),name_len_of_world=world['name_len'],distinct_first_tokens=len(first),names_sharing_a_first_token=sum(v for v in first.values() if v>1),
        items_with_a_first_token_collision_in_candidates=sum(1 for x in fc if x>0),mean_candidates_sharing_first_token_with_gold=float(np.mean(fc)))
def check_all(enc,world,train_frames,qframes,ref=None):
    ch=frame_checks(enc,world,train_frames,qframes,ref);fid=chosen_frames(ch);items,fitems=rev_items(world,fid);ps=place_sharing(world)
    return dict(frames=ch,chosen=fid,place_sharing=ps,items_excluded=sum(it['excluded'] for it in items),n_items=len(items),names=name_stats(enc,world,items),per_frame_items=dict(collections.Counter(it['frame'] for it in items)),
        frame_sha12=hashlib.sha256('\n'.join(RFRAMES[i] for i in fid).encode()).hexdigest()[:12],strata=dict(same_group_candidates=int(np.mean([sum(it['cand_same'][1:]) for it in items])),other_group_candidates=int(np.mean([NCAND-1-sum(it['cand_same'][1:]) for it in items])))),items,fitems
def check_report(R):
    L=['# E55 reverse reading: CPU checks (section 37.2)','',f"- items {R['n_items']}, excluded from the reverse analysis (shared place): {R['items_excluded']}; places {R['place_sharing']['n_places']}, facts sharing a place {R['place_sharing']['n_facts_sharing_a_place']} (max facts per place {R['place_sharing']['max_facts_per_place']}) " + ('(no shared place)' if not R['place_sharing']['n_facts_sharing_a_place'] else ''),
       f"- frames used {R['chosen']} (sha12 {R['frame_sha12']}); items per frame {R['per_frame_items']}; candidates per item: same chunk group {R['strata']['same_group_candidates']}, other group {R['strata']['other_group_candidates']}",
       f"- candidate names: token lengths {R['names']['token_length_of_names']} (world name_len {R['names']['name_len_of_world']}); distinct first tokens {R['names']['distinct_first_tokens']}, names sharing a first token with another name {R['names']['names_sharing_a_first_token']}, items whose candidates hold a first-token collision with the gold {R['names']['items_with_a_first_token_collision_in_candidates']}, mean candidates sharing the gold's first token {R['names']['mean_candidates_sharing_first_token_with_gold']:.2f}",'',
       '| frame | passed | 5-gram overlap training frames | 5-gram overlap question frames | max bag-of-words cosine | 13-gram overlap stream | 13-gram overlap reference | text |','|---|---|---|---|---|---|---|---|']
    for c in R['frames']:L.append(f"| {c['i']} | {'yes' if c['passed'] else 'NO'} | {c['overlap5_train_frames']} | {c['overlap5_question_frames']} | {c['max_bow_cosine_to_a_training_frame']} | {c['overlap13_training_stream']} | {c['overlap13_reference']} | {c['frame']} |")
    return '\n'.join(L)+'\n'
# ---------------------------------------------------------------- analysis
def rank_u(S,cols=None):
    """S (n,16) gold in column 0 -> u = (m - rank) / (m - 1) among the chosen columns (all by default; gold always included)"""
    X=np.asarray(S,float)
    if cols is not None:X=X[:,[0]+list(cols)] if 0 not in cols else X[:,cols]
    g=X[:,0:1];r=1+(X>g).sum(1)+0.5*((X==g).sum(1)-1);m=X.shape[1];return (m-r)/(m-1)
def gain(rec,fam,which='all'):
    S,S0=np.asarray(rec['scores'][fam]),np.asarray(rec['W0'][fam])
    if which=='all':return rank_u(S)-rank_u(S0)
    same=np.asarray(rec['cand_same'][fam]);out=[]
    for i in range(len(S)):
        cols=[j for j in range(1,NCAND) if bool(same[i][j])==(which=='same')];out.append(rank_u(S[i:i+1],cols)[0]-rank_u(S0[i:i+1],cols)[0])
    return np.asarray(out)
def boot(per_seed,nb=10000,seed=31):
    rng=np.random.default_rng(seed);b=np.mean([x[rng.integers(0,len(x),(nb,len(x)))].mean(1) for x in per_seed],axis=0);return float(np.mean([x.mean() for x in per_seed])),*np.percentile(b,[2.5,97.5]).tolist()
def classify(R,F,ci_lo):
    if F<=0:return 'forward gain not above 0: no classification'
    r=R/F
    if ci_lo>0 and r>=0.5:return 'both directions (R/F >= 0.5)'
    if ci_lo>0 and r>=0.2:return 'partial both directions (0.2 <= R/F < 0.5)'
    return 'one direction only (R/F < 0.2 or CI of R includes 0)'
def analyze(files,out=None):
    Rr={}
    for f in files:
        r=json.load(open(f));Rr.setdefault(r['arm'],{})[r['seed']]=r
    arms=sorted(Rr);seeds=sorted(set.intersection(*[set(Rr[a]) for a in arms]));L=[f'# E55 reverse reading (section 37): arms {arms}, seeds {seeds}; u = (16 - rank)/15, gain = u(state) - u(W0); CI = 95% bootstrap over facts (reverse: facts with a unique place only)','']
    ex=lambda rec:np.asarray([not e for e in rec['excluded']])
    G={}
    for a in arms:
        for sd in seeds:
            rec=Rr[a][sd];keep=ex(rec)
            for fam in('fwd','rev'):
                for wh in('all','same','other'):
                    g=gain(rec,fam,wh);G[(a,sd,fam,wh)]=g[keep] if fam=='rev' else g
            G[(a,sd,'fwd_kept','all')]=gain(rec,'fwd','all')[keep]
    L+=['## gains (a = state of the saved read point)','','| arm | forward gain F (all facts) | forward gain F (same facts as reverse) | reverse gain R | R/F | R same-group names | R other-group names | R - R(NAT) | verdict (section 37.3) |','|---|---|---|---|---|---|---|---|---|']
    res={}
    for a in arms:
        F=boot([G[(a,sd,'fwd','all')] for sd in seeds]);Fk=boot([G[(a,sd,'fwd_kept','all')] for sd in seeds]);Rv=boot([G[(a,sd,'rev','all')] for sd in seeds]);Rs=boot([G[(a,sd,'rev','same')] for sd in seeds]);Ro=boot([G[(a,sd,'rev','other')] for sd in seeds])
        res[a]=dict(F=Fk,R=Rv)
        if 'NAT' in arms and a!='NAT':Rn=boot([G[(a,sd,'rev','all')]-G[('NAT',sd,'rev','all')] for sd in seeds],seed=32)
        else:Rn=None
        lo=Rv[1] if Rn is None else min(Rv[1],Rn[1]);rr=min(Rv[0],Rn[0] if Rn else Rv[0])
        verdict='control (W0 / NAT)' if a=='NAT' else classify(rr,Fk[0],lo)
        if a=='NAT':verdict+=f"; |R| {'>' if abs(Rv[0])>0.05 else '<='} 0.05 -> {'name-side confound suspected' if abs(Rv[0])>0.05 else 'no confound signal'}"
        c3=lambda x:'%+.3f [%+.3f, %+.3f]'%x
        L.append(f"| {a} | {c3(F)} | {c3(Fk)} | {c3(Rv)} | {Rv[0]/Fk[0] if Fk[0] else float('nan'):.2f} | {c3(Rs)} | {c3(Ro)} | {c3(Rn) if Rn else ''} | {verdict} |")
    L+=['','## rank histogram of the gold name (reverse, facts with a unique place; seed-pooled), W0 and states','','| arm | rank 1 | within top 3 | rank >= 8 | n facts |','|---|---|---|---|---|']
    for a in ['W0']+arms:
        r1=r3=r8=n=0
        for sd in seeds:
            rec=Rr[a if a!='W0' else arms[0]][sd];S=np.asarray(rec['scores']['rev'] if a!='W0' else rec['W0']['rev']);k=ex(rec);X=S[k];g=X[:,0:1];rk=1+(X>g).sum(1)+0.5*((X==g).sum(1)-1);r1+=int((rk<=1.5).sum());r3+=int((rk<=3.5).sum());r8+=int((rk>=8).sum());n+=len(rk)
        L.append(f"| {a} | {r1} | {r3} | {r8} | {n} |")
    L+=['','chance: rank 1 = 1/16, top 3 = 3/16, rank >= 8 = 9/16. One seed: the CIs cover facts of one training run only.']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def selftest():
    import tempfile
    class Stub:
        def __init__(s):s.v={}
        def __call__(s,t):return [s.v.setdefault(w,len(s.v)+1) for w in re.findall(r"\w+|[^\w\s]",t)]
    import gen_facts as G,gen_world_ctl as G0
    enc=Stub();bg=G._synth_bg(9000);w=G.cell_world(5,'p4',enc,bg,decoys=('names',))
    tf=[l for l in (HERE.parents[0]/'exp04_what_ttt_retains'/'frames300.txt').read_text().split('\n') if l]
    R,items,fitems=check_all(enc,w,tf,G0.QFRAMES)
    assert R['n_items']==w['k']==len(fitems) and len(R['chosen'])==3 and all(len(it['cands'])==NCAND and it['cand_fact'][0]==it['fact'] for it in items)
    assert all(it['cands'][0]==' '+[f for f in w['facts'] if f['id']==it['fact']][0]['X'] for it in items) and all(fi['cands'][0].startswith(' ') and fi['cand_fact']==it['cand_fact'] for fi,it in zip(fitems,items))
    assert all(sum(it['cand_same'][1:])==7 for it in items) or w['p']==1,'7 same-group candidates'
    # a frame that is a word permutation of a training frame must fail
    bad=tf[0].replace('{X}','{P}').split(' ');bad=' '.join(bad[::-1]);assert cosine(words(bad),words(tf[0]))>=0.8
    # synthetic scores
    rng=np.random.default_rng(3);n=w['k'];d=tempfile.mkdtemp();fs=[]
    def sc(g_b,e_b=0.0):
        S=rng.normal(size=(n,NCAND))*0.1;S[:,0]+=g_b;return S
    S0={f:np.zeros((n,NCAND)) for f in('fwd','rev')}
    for arm,(gf,gr) in(('NAT',(0.0,0.0)),('SENTPQ',(1.0,0.6))):
        for sd in(61,):
            rec={'arm':arm,'seed':sd,'scores':{'fwd':sc(gf).tolist(),'rev':sc(gr).tolist()},'W0':{k:v.tolist() for k,v in S0.items()},'cand_same':{f:[it['cand_same'] for it in items] for f in('fwd','rev')},'excluded':[it['excluded'] for it in items]}
            fn=f'{d}/{arm}_{sd}.json';json.dump(rec,open(fn,'w'));fs.append(fn)
    t=analyze(fs);assert 'SENTPQ' in t and 'both directions' in t or 'partial' in t or 'one direction' in t
    assert abs(rank_u(np.asarray([[1.0]+[0.0]*15]))[0]-1.0)<1e-9 and abs(rank_u(np.asarray([[0.0]+[1.0]*15]))[0])<1e-9
    print(check_report(R).splitlines()[2]);print('e55_rev selftest ok (stub tokenizer, synthetic scores)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--analyze',action='store_true');ap.add_argument('--files');ap.add_argument('--out',default=None);a=ap.parse_args()
    if a.selftest:selftest()
    elif a.analyze:print(analyze(a.files.split(','),a.out))
