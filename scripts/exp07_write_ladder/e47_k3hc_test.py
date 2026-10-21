"""E47 unit test (CPU, no model): the complement cell k3hc (K3 share of the OTHER person of each pair) against k3h and K3, on the synthetic table of e43_sets_test and on the hand-built pair table of e47_k3all_test.
Checks: k3h and k3hc are disjoint and their union is K3 exactly (every chunk, all seeds tried: the masked person flips with the seed); k3hc tokens belong to the person that is NOT half_person of the pair; chunks without O mask nothing in both; on the hand-built table the masked person's tokens include the earlier-round sentence INSIDE the chunk before the write sentence's place token and exclude the current round (Hans's rule): k3hc with masked person a masks b1 and b2 (b2 is inside chunk 5) and not b3 / a-tokens; k3h with masked person b masks b1, b2."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_sets as SETS
import e43_sets_test as T
import e47_k3all_test as H
def main():
    base=SETS.Sets(T.table(),101,2,0.35);assert T.old_hash(base)==T.GOLDEN;print('existing cells: output hash identical to the stored golden hash',T.GOLDEN[:12]);tb=T.table();nO=0;tot=0
    for seed in (101,102,103,104):
        sets=SETS.Sets(tb,seed,2,0.35);own={p:u for u in tb for p in u['toks']}
        for k in range(63):
            K3=set(sets.sentencecover(k));A=set(sets.k3hmask(k));B=set(sets.k3hcmask(k))
            if not sets.oracle(k):assert A==B==set()==K3,k;continue
            nO+=1;assert not(A&B) and A|B==K3 and len(A)+len(B)==len(K3),(seed,k);tot+=len(B)
            for p in B:u=own[p];assert u['t']!=sets.half_person(u['pair'])
            for p in A:u=own[p];assert u['t']==sets.half_person(u['pair'])
    assert nO>0 and tot>0
    hb=H.table();tbh=hb;seeds_a=[sd for sd in range(200) if SETS.half_person_of(sd,'p0',['a','b'])=='a'][0];seeds_b=[sd for sd in range(200) if SETS.half_person_of(sd,'p0',['a','b'])=='b'][0]
    a1,b1,a2,b2,a3,b3=(H.toks(tbh,'a',1),H.toks(tbh,'b',1),H.toks(tbh,'a',2),H.toks(tbh,'b',2),H.toks(tbh,'a',3),H.toks(tbh,'b',3))
    sa=SETS.Sets(tbh,seeds_a,2);sb=SETS.Sets(tbh,seeds_b,2)
    assert set(sa.k3hmask(5))==a1|a2 and set(sa.k3hcmask(5))==b1|b2 and not(set(sa.k3hcmask(5))&(a1|a2|a3|b3))   # masked person a: a1, a2 (a2 is in chunk 5, before a3's place token); complement: b1, b2 (b2 in chunk), current round never
    assert set(sb.k3hmask(5))==b1|b2 and set(sb.k3hcmask(5))==a1|a2
    for sets_ in (sa,sb):assert set(sets_.k3hmask(5))|set(sets_.k3hcmask(5))==set(sets_.sentencecover(5)) and not(set(sets_.k3hmask(5))&set(sets_.k3hcmask(5)))
    assert sets_.chunk_set('k3hc',5)[1]==1.0
    print('E47 k3hc test OK on',nO,'chunk x seed cases: k3h and k3hc disjoint, union == K3; complement tokens belong to the other person; hand-built pair: in-chunk earlier round masked for the chosen person, current round never')
if __name__=='__main__':main()
