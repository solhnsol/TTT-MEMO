"""E47 unit test (CPU, no model): cells k3h (K3 of one person per pair only) and k3r (K3 on a random half of the O chunks) on the synthetic sentence table of e43_sets_test; bit-identity of the existing cells.
Checks: half_person is one triple of the pair, fixed over passes / chunks, seed-dependent; k3h is a subset of K3, every k3h token is owned by the masked person of its pair (so the partner's earlier exposures never enter), k3h of a chunk equals the union over its masked-person fact sentences of the earlier same-triple tokens before the place token; chunks without O mask nothing; a chunk holding only the unmasked person of a pair masks nothing for that pair; k3r masks round(n/2) of the n O chunks, each exactly K3, the rest nothing, deterministic, seed-dependent."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_sets as SETS
import e43_sets_test as T
CH=SETS.CH;W=SETS.W
def main():
    base=SETS.Sets(T.table(),101,2,0.35);assert T.old_hash(base)==T.GOLDEN;print('existing cells: output hash identical to the stored golden hash',T.GOLDEN[:12])
    tb=T.table();sets=SETS.Sets(tb,101,2,0.35);own={p:u for u in tb for p in u['toks']};nO=0;full=part=single_unmasked=0
    pairs={u['pair'] for u in tb}
    for p in pairs:
        hp=sets.half_person(p);assert hp in {u['t'] for u in tb if u['pair']==p} and hp==sets.half_person(p)==SETS.Sets(tb,101,5,0.35).half_person(p)
    assert any(SETS.Sets(tb,102,2).half_person(p)!=sets.half_person(p) for p in pairs),'seed has no effect'
    for k in range(63):
        O=sets.oracle(k);K3=set(sets.sentencecover(k));H=sets.k3hmask(k);R=sets.k3rmask(k)
        if not O:assert H==[] and R==[],k;continue
        nO+=1;assert set(H)<=K3,k;lo=max(0,k*CH-W);exp=set()
        for s in sets.cur_sentences(k):
            if s['t']==sets.half_person(s['pair']):
                for u in tb:
                    if u['t']==s['t'] and u['round']<s['round']:exp|={p for p in u['toks'] if lo<=p<s['place'][0]}
            else:single_unmasked+=1
        assert set(H)==exp,k
        for p in H:u=own[p];assert u['t']==sets.half_person(u['pair']),(k,p)
        cur=sets.cur_sentences(k)
        for pr in {s['pair'] for s in cur}:
            ts={s['t'] for s in cur if s['pair']==pr}
            if len(ts)==2:full+=1
            else:part+=1
        assert len(H)<=len(K3)
    assert nO>0
    oc=[k for k in range(63) if sets.oracle(k)];ch=sets.k3r_chunks();assert len(ch)==round(len(oc)/2) and ch<=set(oc)
    for k in range(63):assert sets.k3rmask(k)==(sets.sentencecover(k) if k in ch else [])
    assert SETS.Sets(T.table(),101,2,0.35).k3r_chunks()==ch and any(SETS.Sets(T.table(),102,2,0.35).k3r_chunks()!=ch for _ in [0])
    assert [sets.chunk_set(c,10)[1] for c in ('k3h','k3r')]==[1.0,1.0]
    print('E47 k3h / k3r test OK on',nO,'chunks with O: k3h subset of K3, tokens only of the masked person of each pair, equals the masked-person K3 shares; pair instances with both persons in the chunk',full,', with one person only',part,'; k3r masks',len(ch),'of',len(oc),'O chunks with exactly K3')
if __name__=='__main__':main()
