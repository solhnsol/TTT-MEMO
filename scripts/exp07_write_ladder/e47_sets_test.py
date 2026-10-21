"""E47 unit test (CPU, no model): Sets.pxsplit (P, X), xframe_split, and the cells kp (K3+P), kx (K3+X), kp_u / kx_u (K3 + |P| / |X| uniform positions outside K3 and the set), kx_s / kx_o (K3 + X of the same / another frame) on the synthetic sentence table of e43_sets_test with a synthetic frame id; bit-identity of the existing cells and of K3 / SENT.
Checks: K3, P, X pairwise disjoint; P tokens belong to sentences of the same pair and another triple, X tokens to other pairs, all inside the window before the chunk; SENT == K3 + P + X + (window tokens of the same-triple sentences) exactly; type_counts of SENT agree (P, X counts); kp, kx contain K3; uniform cells have the K3 + |set| size (or fewer if the candidates run out), fill outside K3 and the set, inside the window, deterministic, seed-dependent; X_same + X_other == X; chunks without O mask nothing in all cells.
Per-chunk counts on the real table come from e47_counts.py (server)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_sets as SETS
import e43_sets_test as T
CH=SETS.CH;W=SETS.W
def table():
    tb=T.table()
    for u in tb:u['frame']=(int(u['t'][1:])*7+u['round'])%11   # synthetic frame id (11 frames)
    return tb
def main():
    base=SETS.Sets(T.table(),101,2,0.35);assert T.old_hash(base)==T.GOLDEN;print('existing cells: output hash identical to the stored golden hash',T.GOLDEN[:12])
    tb=table();sets=SETS.Sets(tb,101,2,0.35);assert T.old_hash(sets)==T.GOLDEN;rows=[];nO=0;byt={u['idx']:u for u in tb};own={p:u for u in tb for p in u['toks']}
    for k in range(63):
        O=sets.oracle(k);K3=set(sets.sentencecover(k));P,X=sets.pxsplit(k);Xs,Xo=sets.xframe_split(k)
        cells={c:sets.chunk_set(c,k)[0] for c in ('kp','kx','kp_u','kx_u','kx_s','kx_o')}
        if not O:assert P==[] and X==[] and Xs==[] and Xo==[] and all(v==[] for v in cells.values()),k;continue
        nO+=1;k0=k*CH;lo=max(0,k0-W);Ps,Xs_=set(P),set(X);assert not(Ps&Xs_) and not(Ps&K3) and not(Xs_&K3),k
        cur=sets.cur_sentences(k);ts={s['t'] for s in cur};ps={s['pair'] for s in cur}
        for p in P:u=own[p];assert u['pair'] in ps and u['t'] not in ts and lo<=p<k0
        for p in X:u=own[p];assert u['pair'] not in ps and lo<=p<k0
        SENT=set(sets.sentmask(k));Bw={p for u in tb if u['t'] in ts for p in u['toks'] if lo<=p<k0}
        assert SENT==K3|Ps|Xs_|Bw,k   # SENT = K3 + P + X + the later same-triple sentence tokens in the window
        tc=sets.type_counts(k,sorted(SENT));assert tc['P']==len(P) and tc['X']==len(X),(k,tc,len(P),len(X))
        assert set(Xs)|set(Xo)==Xs_ and not(set(Xs)&set(Xo)),k
        fr={s['frame'] for s in cur}
        for p in Xs:assert own[p]['frame'] in fr
        for p in Xo:assert own[p]['frame'] not in fr
        assert cells['kp']==sorted(K3|Ps) and cells['kx']==sorted(K3|Xs_) and cells['kx_s']==sorted(K3|set(Xs)) and cells['kx_o']==sorted(K3|set(Xo))
        for c,add in (('kp_u',P),('kx_u',X)):
            S=set(cells[c]);ex=S-K3;cand=len([p for p in range(lo,k0) if p not in K3|set(add)]);assert K3<=S and len(ex)==min(len(add),cand) and all(lo<=p<k0 and p not in add for p in ex),(k,c)
        rows.append((k,len(O),len(K3),len(P),len(X),len(Xs),len(sets.sentmask(k))))
    assert nO>0 and any(r[3] for r in rows) and any(r[4] for r in rows) and any(r[5] for r in rows) and any(r[4]>r[5] for r in rows)
    s2=SETS.Sets(table(),101,2,0.35);s3=SETS.Sets(table(),102,2,0.35)
    for k in range(63):assert [s2.chunk_set(c,k) for c in ('kp','kx','kp_u','kx_u','kx_s','kx_o')]==[sets.chunk_set(c,k) for c in ('kp','kx','kp_u','kx_u','kx_s','kx_o')]
    assert any(s3.kpumask(k)!=sets.kpumask(k) for k in range(63) if sets.pxsplit(k)[0]),'seed has no effect'
    print('E47 sets test OK: K3, P, X disjoint; SENT == K3 + P + X + later same-triple window tokens; type_counts P / X agree; kp / kx / kp_u / kx_u / kx_s / kx_o as defined; deterministic, seed-dependent; chunks without O mask nothing. (chunk,|O|,|K3|,|P|,|X|,|X same frame|,|SENT|):',rows[:4],'...')
if __name__=='__main__':main()
