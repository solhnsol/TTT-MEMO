"""Pass 2: dump-wide document frequency of (a) generated fake surnames (novelty: must be 0) and (b) gazetteer places (answer prior)."""
import pyarrow.parquet as pq,pickle,re,sys,os
from collections import Counter
from multiprocessing import Pool
import ahocorasick
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));from common import *
def gaz():
    P=pickle.load(open(f'{W}/persons.pkl','rb'));g=set(p['birth'] for p in P if p['birth'])|set(p['school'] for p in P if p['school'])
    pop=pq.read_table(f'{DL}/popqa.parquet').to_pylist();g|={r['obj'] for r in pop if r['prop'] in ('place of birth','capital','country')}
    for i in range(3):g|={r['tail'] for r in pq.read_table(f'{DL}/trex_tr{i}.parquet').to_pylist() if r['relation']=='[Person] is born in [Location]'}
    return sorted(x for x in g if x and len(x)>=3)
def work(i):
    T=pickle.load(open(f'{W}/scan2_terms.pkl','rb'))
    A=ahocorasick.Automaton()
    for t in T:A.add_word(t,t)
    A.make_automaton();df=Counter()
    for tx in pq.read_table(f'{DL}/w{i}.parquet',columns=['text']).column('text').to_pylist():
        h=set()
        for end,t in A.iter(tx):
            s=end-len(t)+1
            if (s>0 and tx[s-1].isalnum()) or (end+1<len(tx) and tx[end+1].isalnum()):continue
            h.add(t)
        df.update(h)
    return df
if __name__=='__main__':
    NP=pickle.load(open(f'{W}/namepools.pkl','rb'))
    terms=set(gaz())
    for v in NP['pools'].values():terms|=set(v['surnames'])
    terms=sorted(terms);pickle.dump(terms,open(f'{W}/scan2_terms.pkl','wb'));print('terms',len(terms),flush=True)
    df=Counter()
    with Pool(3) as p:
        for d in p.imap_unordered(work,range(41)):df.update(d)
    pickle.dump(df,open(f'{W}/scan2_df.pkl','wb'))
    sp=[s for v in NP['pools'].values() for s in v['surnames']]
    print('surnames',len(sp),'novel(df=0)',sum(df[s]==0 for s in sp))
