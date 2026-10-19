"""E41 CPU tests: s3_sets.attn_fill (top scores outside O, ties -> lower index, exactly nF, O excluded, non-candidates excluded) and the fill_total hook of oraclefill (U49)."""
import sys,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_sets as S
W=8192;base=100;rng=np.random.default_rng(0);A=rng.random(W);A[10]=A[20]=A[30]=5.0   # three exact ties at the top
cand=np.ones(W,bool);cand[:4]=False;cand[-128:]=False;O=[base+20,base+500,base+W+3]   # one O at a tied top score, one in the chunk (outside the window array)
r=S.attn_fill(A,cand,O,base,2);ok1=list(r)==[base+10,base+30]   # tie among 10,20,30: 20 is in O -> 10 and 30 (lower index first)
r2=S.attn_fill(A,cand,[],base,2);ok2=list(r2)==[base+10,base+20]   # ties -> lower index
r3=S.attn_fill(A,cand,O,base,300);ok3=len(r3)==300 and not (set(r3.tolist())&set(O)) and all(cand[p-base] for p in r3) and list(r3)==sorted(r3)
ord_=np.argsort(-A[np.r_[4:W-128]],kind='stable');ok4=set(r3.tolist())<=set((base+np.r_[4:W-128]).tolist())
ok5=len(S.attn_fill(A,cand,O,base,0))==0 and len(S.attn_fill(A,np.zeros(W,bool),O,base,5))==0
# fill_total hook: oraclefill total taken from a dict
def synth():
    t=[]
    for tr in range(30):
        for r in range(1,5):
            pos=(r-1)*6144+tr*150+5;t.append(dict(idx=len(t),t=f't{tr}',pair=f'p{tr//2}',round=r,place=[pos],names=[pos-3,pos-2],spans=[[pos],[pos-3,pos-2]],text=''))
    return sorted(t,key=lambda s:s['place'][0])
s=S.Sets(synth(),101,2,0.25);k=12;O_=s.oracle(k);a=s.oraclefill(k)[0];s.fill_total={k:1000};b=s.oraclefill(k)[0];s.fill_total={k:len(O_)};c=s.oraclefill(k)[0]
ok6=len(a)==2048 and len(b)==1000 and sorted(c)==sorted(O_) and set(O_)<=set(b)
print({'1_ties_and_O_excluded':ok1,'2_ties_lower_index':ok2,'3_count_range_candidates':ok3,'4_inside_candidates':ok4,'5_zero_and_no_candidates':ok5,'6_fill_total_hook':ok6})
sys.exit(0 if all([ok1,ok2,ok3,ok4,ok5,ok6]) else 1)
