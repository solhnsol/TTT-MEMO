"""E47 unit test (CPU, no model): what K3, k3h and k3all mask, on a hand-built sentence table of one pair (persons a, b) with rounds 1-3 around chunk 5 (positions are stream token indices; chunk 5 = 5120..6143, window start 5120-8192 < 0 -> 0).
Layout: a1 3000, b1 3030 (earlier chunk, inside the window), a2 5200, b2 5230 (chunk 5), a3 5500, b3 5530 (chunk 5, the current round for the write). A second chunk (6) holds a4 6200 only (b4 at 6230 is also in chunk 6 -> second case with the pair split: b4' at 7200 in chunk 7). Each sentence has 8 tokens (pos-5 .. pos+2), name tokens pos-3, pos-2, place token pos.
Questions of the design thread, answered by the checks below (K3 = Sets.sentencecover, k3h = Sets.k3hmask, k3all = Sets.k3all_parts):
  (i)   a's earlier-round sentence INSIDE the chunk before the write sentence's place token (a2 for the write sentence a3): masked by K3, by k3h when a is the masked person, by k3all.
  (ii)  b's earlier-round sentences inside the chunk (b2) and in the window (b1) when b is a fact sentence of the chunk too: masked by K3 (b3 is itself a fact sentence of the chunk, so its earlier rounds are covered), NOT masked by k3h when b is the unmasked person, masked by k3all.
  (iii) b's same-round sentence (b3 for a3, placed AFTER a3 here; b2 for a2 as the same-round partner) is NOT masked by K3 as a same-round sentence of a3 (it is masked only as the EARLIER round of nobody) / not masked by k3all.
Plus: the single-person chunk (partner b4' in the next chunk) for k3all / K3."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_sets as SETS
import e43_sets_test as T
CH=SETS.CH
def sent(idx,t,rnd,pos):
    return dict(idx=idx,t=t,pair='p0',round=rnd,place=[pos],names=[pos-3,pos-2],spans=[[pos],[pos-3,pos-2]],text='',toks=list(range(pos-5,pos+3)),occ=[],frame=0)
def table():
    L=[('a',1,3000),('b',1,3030),('a',2,5200),('b',2,5230),('a',3,5500),('b',3,5530),('a',4,6200),('b',4,7200)]
    return [sent(i,t,r,pos) for i,(t,r,pos) in enumerate(L)]
def toks(tb,t,r):return set(next(u for u in tb if u['t']==t and u['round']==r)['toks'])
def main():
    tb=table();sets=SETS.Sets(tb,101,2);base=SETS.Sets(T.table(),101,2,0.35);assert T.old_hash(base)==T.GOLDEN
    # masked person of the pair: force a (the choice is a pure function of the seed; try seeds until it is a, then until it is b)
    seeds_a=[sd for sd in range(200) if SETS.half_person_of(sd,'p0',['a','b'])=='a'];seeds_b=[sd for sd in range(200) if SETS.half_person_of(sd,'p0',['a','b'])=='b'];assert seeds_a and seeds_b
    # chunk 5: fact sentences a2, b2, a3, b3
    k=5;cur={s['round']*10+(0 if s['t']=='a' else 1) for s in sets.cur_sentences(k)};assert cur=={20,21,30,31},cur
    K3=set(sets.sentencecover(k));a1,b1,a2,b2,a3,b3=(toks(tb,'a',1),toks(tb,'b',1),toks(tb,'a',2),toks(tb,'b',2),toks(tb,'a',3),toks(tb,'b',3))
    assert a1<=K3 and a2<=K3 and b1<=K3 and b2<=K3                      # (i) a2 (in chunk, before a3's place token) and (ii) b1, b2 are masked by K3
    assert not(a3&K3) and not(b3&K3)                                    # (iii) the current round (a3 and the same-round b3) is not masked
    sa=SETS.Sets(tb,seeds_a[0],2);Ha=set(sa.k3hmask(k));assert Ha==a1|a2 and not(Ha&(b1|b2|b3|a3))                      # masked person a: only a's earlier rounds; b's earlier rounds stay visible
    sb=SETS.Sets(tb,seeds_b[0],2);Hb=set(sb.k3hmask(k));assert Hb==b1|b2 and not(Hb&(a1|a2|a3|b3))
    S,st=sets.k3all_parts(k);assert set(S)==a1|a2|b1|b2|a1|b1 and not(set(S)&(a3|b3)),st   # k3all of chunk 5 = every earlier-round token of both persons
    assert set(S)==K3,'for a pair fully inside the chunk k3all == K3 except tokens at or after the place token (none here)'
    # same-round partner placed BEFORE the write sentence (b3 at 5470, a3 at 5500): still not masked by K3 / k3all, K3b (pair_all) does mask it
    tbb=[dict(u) for u in tb];ub=[dict(sent(5,'b',3,5470)),];tbb=[u for u in tb if not(u['t']=='b' and u['round']==3)]+ub;ssb=SETS.Sets(tbb,101,2);K3b_=set(ssb.sentencecover(5));b3b=set(ub[0]['toks']);assert not(b3b&K3b_) and not(b3b&set(ssb.k3all_parts(5)[0])) and b3b&set(ssb.sentencecover(5,True))
    # chunk 6: a4 in the chunk, b4 (7200) in chunk 7 -> single-person chunk
    k=6;assert [s['t'] for s in sets.cur_sentences(k)]==['a'] and [s['t'] for s in sets.cur_sentences(7)]==['b']
    K3_6=set(sets.sentencecover(k));S6,st6=sets.k3all_parts(k);allprev_a=a1|a2|a3;allprev_b=b1|b2|b3
    assert K3_6==allprev_a and set(S6)==allprev_a|allprev_b,'single-person chunk: K3 masks only a, k3all also the partner b rounds 1-3 (partner not in chunk)'
    assert st6['partner_not_in_chunk']==len(allprev_b) and st6['out_window']==0
    # chunk 7: b4 only; the partner a's earlier rounds a1..a3 are outside the chunk and masked by k3all; a4 (6200, the same round 4 as b4) is excluded and counted
    k=7;K3_7=set(sets.sentencecover(k));S7,st7=sets.k3all_parts(k);a4=toks(tb,'a',4);assert K3_7==allprev_b and set(S7)==allprev_b|allprev_a and st7['partner_not_in_chunk']==len(allprev_a) and st7['same_round_partner_excluded']==len(a4),st7
    # window start: round 1 lies before the window start of chunk 10 (10240 - 8192 = 2048): counted in out_window, not returned
    tb3=[sent(0,'a',1,100),sent(1,'b',1,130),sent(2,'a',2,3000),sent(3,'b',2,3030),sent(4,'a',3,10300),sent(5,'b',3,10330)];s3=SETS.Sets(tb3,101,2);kk=10;assert [s['round'] for s in s3.cur_sentences(kk)]==[3,3]
    S3_,st3=s3.k3all_parts(kk);assert set(S3_)==toks(tb3,'a',2)|toks(tb3,'b',2) if False else True
    r2=set(tb3[2]['toks'])|set(tb3[3]['toks']);assert set(S3_)==r2 and st3['out_window']==16 and st3['partner_not_in_chunk']==0,st3
    print('E47 K3 / k3h / k3all semantics test OK: (i) a2 masked, (ii) b1 b2 masked by K3 and k3all but NOT by k3h (partner visible), (iii) the same-round sentences a3 b3 are not masked by any; single-person chunks: K3 masks only the written person, k3all also the partner (partner_not_in_chunk counted)')
if __name__=='__main__':main()
