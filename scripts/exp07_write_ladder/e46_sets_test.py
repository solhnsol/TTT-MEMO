"""E46 unit test (CPU, no model): Sets.kindsplit / occmask / frmmask / unifoccmask on a synthetic sentence table with occupation tokens, and bit-identity of the existing cells (golden hash of e43_sets_test) and of K3.
Checks: O, OCC, FRM are disjoint and O + OCC + FRM == K3 exactly; OCC tokens belong to the 'occ' field of earlier same-triple sentences inside the K3 range; occ = O+OCC, frm = O+FRM, both inside K3 and containing O; unif_occ = O + |OCC| window positions outside O and OCC (inside the window before the chunk), deterministic; chunks without O mask nothing in all three cells.
The per-chunk counts on the real table come from e46_counts.py (server)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_sets as SETS
import e43_sets_test as T
CH=SETS.CH;W=SETS.W
def table():
    tb=T.table()
    for u in tb:
        pos=u['place'][0];u['occ']=[pos-4,pos+2]   # synthetic occupation tokens: one before and one after the place token inside the sentence token range
    return tb
def main():
    base=SETS.Sets(T.table(),101,2,0.35);assert T.old_hash(base)==T.GOLDEN;print('existing cells: output hash identical to the stored golden hash',T.GOLDEN[:12])
    tb=table();sets=SETS.Sets(tb,101,2,0.35);assert T.old_hash(sets)==T.GOLDEN   # the occ field changes nothing for existing cells
    assert all(sets.sentencecover(k)==base.sentencecover(k) for k in range(63));occs={p for u in tb for p in u['occ']};rows=[];nO=0;nocc=0
    for k in range(63):
        O,C,F=sets.kindsplit(k);K3=sets.sentencecover(k);oc,fr,un=sets.occmask(k),sets.frmmask(k),sets.unifoccmask(k)
        if not sets.oracle(k):assert O==C==F==[] and oc==fr==un==[],k;continue
        nO+=1;assert O==sets.oracle(k);assert not (set(O)&set(C)) and not (set(O)&set(F)) and not (set(C)&set(F)),k
        assert sorted(set(O)|set(C)|set(F))==K3 and len(O)+len(C)+len(F)==len(K3),k   # the three kinds partition K3
        assert set(C)<=occs and not (set(F)&occs-set(O)),k;nocc+=len(C)
        assert oc==sorted(set(O)|set(C)) and fr==sorted(set(O)|set(F)) and set(oc)<=set(K3) and set(fr)<=set(K3)
        k0=k*CH;lo=max(0,k0-W);extra=set(un)-set(O);assert set(O)<=set(un) and not (extra&set(C)) and all(lo<=p<k0 for p in extra),k
        cand=len([p for p in range(lo,k0) if p not in set(O)|set(C)]);assert len(extra)==min(len(C),cand),(k,len(extra),len(C),cand)
        rows.append((k,len(O),len(C),len(F),len(K3),len(oc),len(fr),len(un)))
    assert nO>0 and nocc>0
    s2=SETS.Sets(table(),101,2,0.35)
    for k in range(63):assert [s2.occmask(k),s2.frmmask(k),s2.unifoccmask(k)]==[sets.occmask(k),sets.frmmask(k),sets.unifoccmask(k)]
    s3=SETS.Sets(table(),102,2,0.35);assert any(s3.unifoccmask(k)!=sets.unifoccmask(k) for k in range(63) if sets.oracle(k) and sets.kindsplit(k)[1]),'seed has no effect'
    for cell in ('occ','frm','unif_occ'):assert sets.chunk_set(cell,rows[0][0])[1]==1.0
    print('E46 sets test OK: O + OCC + FRM == K3 (disjoint) on all',nO,'chunks with O; occ = O+OCC, frm = O+FRM, unif_occ = O + |OCC| uniform window positions outside O and OCC; deterministic, seed-dependent; chunks without O mask nothing. (chunk,|O|,|OCC|,|FRM|,|K3|,|occ|,|frm|,|unif_occ|):',rows[:4],'...')
if __name__=='__main__':main()
