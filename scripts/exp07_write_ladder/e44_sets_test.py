"""E44 unit test (CPU, no model): Sets.sentmask (SENT = O+B+P+X), bgmask (BG = O+G), noomask (NOO = window minus O), type_counts, and UNIF70 (= oraclefill with fill_total = the BG count) on the synthetic sentence table of e43_sets_test; bit-identity of the existing cells (golden hash of e43_sets_test).
Prints one OK line per property. The per-chunk counts on the real table come from k3_counts.py (server)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_sets as SETS
import e43_sets_test as T
CH=SETS.CH;W=SETS.W
def main():
    tb=T.table();sets=SETS.Sets(tb,101,2,0.35);assert T.old_hash(sets)==T.GOLDEN;print('existing cells: output hash identical to the stored golden hash',T.GOLDEN[:12])
    rows=[];nO=0
    for k in range(63):
        O=sets.oracle(k);S=sets.sentmask(k);B=sets.bgmask(k);N=sets.noomask(k)
        if not O:assert S==[] and B==[] and N==[];continue
        nO+=1;k0=k*CH;lo=max(0,k0-W);win=set(range(lo,k0));Os=set(O);Ow=Os&win;sw=sets._sent_window_tokens(k);K3=set(sets.sentencecover(k))
        assert Os<=K3<=set(S),k;assert set(S)<=win|Os|K3
        assert (set(B)&win)==(win-(sw-Os)) and Os<=set(B) and not ((sw-Os)&set(B)),k
        assert set(N)==win-Ow and not (set(N)&Os) and set(N)|Ow==win,k
        assert win<= (set(B)|sw) and set(B)<=win|Os
        tS,tB,tN=sets.type_counts(k,S),sets.type_counts(k,B),sets.type_counts(k,N)
        assert sum(tS.values())==len(S) and sum(tB.values())==len(B) and sum(tN.values())==len(N)
        assert tS['G']==0 and tB['B']==tB['P']==tB['X']==0 and tN['O']==0,(k,tS,tB,tN)
        sets.fill_total={k:len(B)};U=sets.chunk_set('oraclefill',k)[0];sets.fill_total=None
        assert len(U)==max(len(B),len(O)) and Os<=set(U),(k,len(U),len(B))
        rows.append((k,len(O),len(S),len(B),len(N)))
    assert nO>0
    print('E44 sets test OK: O<=K3<=SENT; BG = window minus non-O sentence tokens plus O; NOO = window minus O; types add up (SENT has no G, BG has no B/P/X, NOO has no O); UNIF70 = oraclefill with the BG size and the same O; chunks with O',nO,'(chunk, |O|, |SENT|, |BG|, |NOO|)',rows[:4],'...')
if __name__=='__main__':main()
