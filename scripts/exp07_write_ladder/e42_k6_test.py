"""CPU tests for the K6 / K6' cells and the oraclefill --fill-attn-cands option (s3_sets.py): which chunks have O, what each cell masks per chunk, O-chunk range of oraclefill (= attnfill's), candidate-set restriction."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_sets as S
def synth():
    t=[]
    for tr in range(30):
        for r in range(1,5):
            pos=(r-1)*6144+tr*150+5;t.append(dict(idx=len(t),t=f't{tr}',pair=f'p{tr//2}',round=r,place=[pos],names=[pos-3,pos-2],spans=[[pos],[pos-3,pos-2]],text=''))
    return sorted(t,key=lambda s:s['place'][0])
s=S.Sets(synth(),101,2,0.35);NCH=24;CH=S.CH;W=S.W
hasO=[k for k in range(NCH) if s.oracle(k)];noO=[k for k in range(NCH) if not s.oracle(k)];assert hasO and noO,(hasO,noO)
ok1=all(s.chunk_set('k6',k)[0]==s.oracle(k) for k in hasO) and all(list(s.chunk_set('k6',k)[0])==list(s.prefixall(k)) for k in noO)
ok2=all(list(s.chunk_set('k6p',k)[0])==list(s.prefixall(k)) for k in hasO) and all(len(s.chunk_set('k6p',k)[0])==0 for k in noO)
ok3=all(len(s.chunk_set('oraclefill',k)[0])>0 for k in hasO) and all(len(s.chunk_set('oraclefill',k)[0])==0 for k in noO)   # oraclefill (and attnfill, same guard) masks exactly the O chunks
s.fill_attn_cands=True;ok4=True
for k in hasO:
    S_,_=s.chunk_set('oraclefill',k);O=set(s.oracle(k));k0=k*CH;F=[p for p in S_ if p not in O]
    ok4&=all(p>=4 and p<k0-128 for p in F)
    if k0>=W:ok4&=len(S_)==max(len(O),round(0.35*W))   # total still the target when enough candidates
s.fill_attn_cands=False;s.fill_no_o=True;kn=[k for k in range(1,NCH) if not s.oracle(k)][-1];ko=hasO[-1];s.fill_total={kn:777,ko:len(s.oracle(ko))+500}
a_=s.chunk_set('oraclefill',kn)[0];b_=s.chunk_set('oraclefill',ko)[0];ok5=len(a_)==777 and max(a_)<kn*CH and len(b_)==len(s.oracle(ko))+500 and set(s.oracle(ko))<=set(b_)   # --fill-no-o: no-O chunk filled with fill_total, O chunk = O + uniform
print({'fill_no_o':ok5,'k6':ok1,'k6p':ok2,'oraclefill_range_is_O_chunks':ok3,'fill_attn_cands':ok4,'n_O_chunks':len(hasO),'n_noO_chunks':len(noO)})
sys.exit(0 if ok1 and ok2 and ok3 and ok4 and ok5 else 1)
