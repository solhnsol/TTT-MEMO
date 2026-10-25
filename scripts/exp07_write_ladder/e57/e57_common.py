"""E57 CPU measurement helpers (no model, no GPU): regex fact candidates (a PROXY for the LLM-extracted (S, A) pairs of the design: numbers, money, percent, dates, capitalised name sequences), sentence split, form clustering, stream layouts, power. Deterministic."""
import re,math,sys,pathlib,collections,random
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'e56'))
import e56_text as T
MON='January|February|March|April|May|June|July|August|September|October|November|December'
MONEY=re.compile(r'(?:US\$|\$|€|£)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion|thousand|bn|m|k)\b)?',re.I)
PCT=re.compile(r'\b\d+(?:\.\d+)?\s?(?:%|percent\b|per cent\b)',re.I)
DATE=re.compile(r'\b(?:('+MON+r')\s(\d{1,2})(?:st|nd|rd|th)?|(\d{1,2})(?:st|nd|rd|th)?\s(?:of\s)?('+MON+r'))\b')
NUM=re.compile(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s(?:million|billion|trillion|thousand)\b|\b\d{4,}\b',re.I)
CAP=re.compile(r"\b[A-Z][a-z]+(?:\s(?:[A-Z][a-z]+|of|de|von|van|bin|al)){1,3}\b")
STARTERS=set('The A An In On At According However But And Or If When While After Before As For From With By This That These Those It He She They We You His Her Its Their Our Mr Mrs Ms Dr Meanwhile Also Still Now Here There What Who How Why Where Which Some Many Most Several Last Next First One Two Three Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September October November December Read Related Share Photo Image Video Advertisement Sign Subscribe'.split())
GENERIC_NAMES=set('United States United Kingdom New York White House Prime Minister Supreme Court Associated Press Reuters Getty Images Social Media Middle East North America South Africa European Union'.lower().split('|'))
UNITS={'thousand':1e3,'k':1e3,'million':1e6,'m':1e6,'billion':1e9,'bn':1e9,'trillion':1e12}
MONTHNUM={m.lower():i+1 for i,m in enumerate(MON.split('|'))}
def _num(s):
    s=s.lower().replace('us$','$');mult=1.0
    m=re.search(r'(thousand|million|billion|trillion|bn|k|m)\b',s)
    if m:mult=UNITS[m.group(1)]
    v=re.search(r'\d[\d,]*(?:\.\d+)?',s)
    return float(v.group(0).replace(',',''))*mult if v else None
def facts_in_sentence(s):
    """-> list of (type, key, surface, start, end); type NUM (money, percent, plain number) | DATE | NAME. Normalised keys: numbers by value (1.2 million == 1,200,000), dates month-day, names lower case."""
    out=[];taken=[]
    def free(a,b):return all(b<=x or a>=y for x,y in taken)
    for m in MONEY.finditer(s):
        v=_num(m.group(0))
        if v is not None and free(*m.span()):out.append(('NUM','$%g'%v,m.group(0),*m.span()));taken.append(m.span())
    for m in PCT.finditer(s):
        v=_num(m.group(0))
        if v is not None and free(*m.span()):out.append(('NUM','%g%%'%v,m.group(0),*m.span()));taken.append(m.span())
    for m in DATE.finditer(s):
        mo=(m.group(1) or m.group(4)).lower();d=int(m.group(2) or m.group(3))
        if 1<=d<=31 and free(*m.span()):out.append(('DATE',f'{MONTHNUM[mo]:02d}-{d:02d}',m.group(0),*m.span()));taken.append(m.span())
    for m in NUM.finditer(s):
        v=_num(m.group(0))
        if v is not None and not (1990<=v<=2035 and re.fullmatch(r'\d{4}',m.group(0))) and free(*m.span()):out.append(('NUM','%g'%v,m.group(0),*m.span()));taken.append(m.span())
    for m in CAP.finditer(s):
        t=m.group(0);w=t.split()
        while w and w[0] in STARTERS:w=w[1:]
        if len(w)<2:continue
        t2=' '.join(w);a=m.end()-len(t2)
        if t2.lower() in GENERIC_NAMES or not free(a,m.end()):continue
        out.append(('NAME',t2.lower(),t2,a,m.end()));taken.append((a,m.end()))
    return out
def words(s):return re.findall(r"[a-z0-9]+(?:[-'.][a-z0-9]+)*",s.lower())
def ngr(w,n):return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
def jac(a,b):return len(a&b)/len(a|b) if a|b else 0.0
def masked(s,surfs):
    for x in sorted(set(surfs),key=len,reverse=True):s=s.replace(x,' SLOT ')
    return s
def n_forms(items,thr=0.5):
    """items: [(sentence, [surfaces to mask])]; number of distinct sentence forms (slot-masked 5-gram Jaccard clustering)"""
    forms=[]
    for s,sf in items:
        g=ngr(words(masked(s,sf)),5)
        for f in forms:
            if jac(g,f)>=thr:break
        else:forms.append(g)
    return len(forms)
def minhash_dup(texts,thr=0.5):
    """indices of kept articles: near-duplicate removal by exact 5-gram word-shingle Jaccard (clusters are small), keep the longest of each duplicate group"""
    gs=[ngr(words(t),5) for t in texts];order=sorted(range(len(texts)),key=lambda i:-len(texts[i]));kept=[]
    for i in order:
        if all(jac(gs[i],gs[j])<thr for j in kept):kept.append(i)
    return sorted(kept)
def phi(x):return 0.5*(1+math.erf(x/math.sqrt(2)))
def power(n_eff,delta):
    se=0.379/math.sqrt(max(n_eff,1e-9));return phi(delta/se-1.96)
def power_table(n,deff_list=(1.0,1.5,2.0),deltas=(0.03,0.10,0.15)):
    rows=[]
    for d in deff_list:
        ne=n/d;theta=max(0.10,1.06/math.sqrt(max(ne,1e-9)));rows.append(dict(deff=d,n_eff=ne,theta=theta,**{f'power_{x}':power(ne,x) for x in deltas}))
    return rows
def layouts(events,rng,rounds=2):
    """events: list of lists of article ids (per event, article order = list order). -> {'CLUSTER': [ids], 'SPREAD': [ids]} for `rounds` rounds; every round re-shuffles event order and the article order inside an event with rng. CLUSTER: events contiguous. SPREAD: round robin over events (i-th article of every event, then the (i+1)-th)."""
    out={'CLUSTER':[],'SPREAD':[]}
    for r in range(rounds):
        ev=[list(e) for e in events];rng.shuffle(ev)
        for e in ev:rng.shuffle(e)
        out['CLUSTER']+= [a for e in ev for a in e]
        mx=max(len(e) for e in ev)
        out['SPREAD']+= [e[i] for i in range(mx) for e in ev if i<len(e)]
    return out
def rho_win(stream_tokens_of,stream,occ,CH=1024,WIN=8192):
    """stream: list of article ids; stream_tokens_of[a] = token length; occ[(target)] = {article id: [token offsets of its (S,A) sentences inside the article]}.
    -> (rho_prev_chunks, rho_incl_same_chunk, n_occ): over the occurrences of the LAST round only (every earlier occurrence is then in the stream before), the share that has an earlier occurrence of the SAME target inside the 8192 tokens before the chunk start (previous chunks only) / before the occurrence itself (incl. its own chunk)."""
    pos=0;start={}
    nrounds=None;starts=[]
    for a in stream:starts.append((a,pos));pos+=stream_tokens_of[a]
    total=pos;hit1=hit2=n=0
    # last round = the final len(stream)/rounds articles; the caller passes the number of rounds via the first repeated article
    seen=set();last0=0
    for i,(a,p) in enumerate(starts):
        if a in seen:last0=i;break
        seen.add(a)
    if last0==0:last0=len(starts)
    # the last round starts at index last0*(rounds-1)
    rounds=len(stream)//max(last0,1);r0=last0*(rounds-1)
    for t,per in occ.items():
        ev=sorted(p+o for a,p in starts for o in per.get(a,[]))
        last=[p+o for a,p in starts[r0:] for o in per.get(a,[])]
        for x in last:
            cs=(x//CH)*CH;prev=[y for y in ev if y<x]
            n+=1;hit1+= any(cs-WIN<=y<cs for y in prev);hit2+= any(cs-WIN<=y<x for y in prev)
    return (hit1/n if n else float('nan'),hit2/n if n else float('nan'),n)
def _selftest():
    fs={(t,k) for t,k,_,_,_ in facts_in_sentence('The ministry said on March 3 that 1.2 million people paid $1,200,000 or 5.5 percent more in Springfield, Ohio.')}
    assert ('NUM','1.2e+06') in fs and ('NUM','$1.2e+06') in fs and ('NUM','5.5%') in fs and ('DATE','03-03') in fs and ('NAME','springfield') not in fs
    assert n_forms([('Mara works in Ohio as a surveyor for the state.',['Mara','Ohio']),('Mara works in Ohio as a surveyor for the state.',['Mara','Ohio'])])==1
    assert n_forms([('Mara works in Ohio as a surveyor for the state today.',['Mara','Ohio']),('According to the report, the surveyor is Mara of Ohio.',['Mara','Ohio'])])==2
    for nev,expect_cluster_hi,expect_spread_lo in ((12,True,True),):
        events=[[(e,a) for a in range(3)] for e in range(nev)];tok={x:2000 for ev in events for x in ev};occ={(e,0):{(e,a):[10] for a in range(3)} for e in range(nev)}
        L=layouts(events,random.Random(0),rounds=2);rc=rho_win(tok,L['CLUSTER'],occ);rs=rho_win(tok,L['SPREAD'],occ)
        assert rc[0]>0.5 and rs[0]<0.2,(rc,rs)
    p=power_table(60);assert abs(p[0]['theta']-0.1368)<1e-3 and 0.5<p[0]['power_0.1']<0.6
    print('e57_common selftest ok')
if __name__=='__main__' or '--selftest' in sys.argv:
    if '--selftest' in sys.argv:_selftest()
