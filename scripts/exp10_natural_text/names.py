"""Fake-name candidate pools per nationality group: common real given names (by gender) + Markov-generated surnames."""
import pickle,random,re,sys,os,hashlib
from collections import Counter,defaultdict
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));from common import *
M=pickle.load(open(f'{W}/scan1_all.pkl','rb'))
given=defaultdict(Counter);surn=defaultdict(Counter);allsurn=set()
for (nat,g,n),c in M['given'].items():given[(NATG.get(nat),g)][n]+=c
for (nat,n),c in M['surn'].items():
    allsurn.add(n)
    if NATG.get(nat):surn[NATG[nat]][n]+=c
def markov(words,order=3,seed=0,n=600):
    T=defaultdict(Counter)
    for w in words:
        s='^'*order+w.lower()+'$'
        for i in range(len(s)-order):T[s[i:i+order]][s[i+order]]+=1
    rng=random.Random(seed);out=set();tries=0
    while len(out)<n and tries<n*200:
        tries+=1;s='^'*order
        while not s.endswith('$') and len(s)<order+14:
            c=T[s[-order:]];
            if not c:break
            s+=rng.choices(list(c),weights=list(c.values()))[0]
        w=s.strip('^$')
        if 5<=len(w)<=10 and s.endswith('$') and w.isalpha():
            w=w.capitalize()
            if w not in allsurn and w not in words:out.add(w)
    return sorted(out)
pools={}
for grp in sorted(set(NATG.values())):
    words=[w for w,c in surn[grp].items() if c>=2 and re.fullmatch(r'[A-Z][a-z]+',w)]
    if len(words)<200:continue
    pools[grp]={'surnames':markov(words,seed=int(hashlib.md5(grp.encode()).hexdigest()[:6],16)),
                'given_m':[n for n,c in given[(grp,'m')].most_common(80) if c>=30],
                'given_f':[n for n,c in given[(grp,'f')].most_common(80) if c>=30]}
    print(grp,len(words),len(pools[grp]['surnames']),len(pools[grp]['given_m']),len(pools[grp]['given_f']),pools[grp]['surnames'][:8],pools[grp]['given_f'][:5])
common=[n for n,c in Counter({n:c for (nat,n),c in M['surn'].items()}).most_common()]
cs=Counter()
for (nat,n),c in M['surn'].items():cs[n]+=c
pickle.dump({'pools':pools,'common_surnames':[n for n,_ in cs.most_common(1000)]},open(f'{W}/namepools.pkl','wb'))
