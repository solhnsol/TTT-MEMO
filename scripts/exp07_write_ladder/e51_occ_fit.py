"""E51 fit of the holdout occupation list to the token-length histogram of the 30 old occupations (gen_world.OCC, tokenised as ' '+word with the real tokenizer).
Starts from frame_pools_hold.HOLD_OCC, swaps entries of an over-represented token length for candidates (CAND, none of them in OCC or HOLD_OCC) of an under-represented length until the histogram is as close as possible (L1 distance), keeping 30 entries, one word each.
Prints old histogram, current, final, the swaps and the final Python list. Server (real tokenizer): python3 scripts/exp07_write_ladder/e51_occ_fit.py  |  --selftest (fake tokenizer)"""
import sys,collections,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;GW=HERE.parents[0]/'exp04_what_ttt_retains';sys.path.insert(0,str(GW));sys.path.insert(0,str(HERE))
CAND=['carpenter','plumber','sailor','fisherman','farmer','gardener','hunter','miner','porter','steward','clerk','cook','laundress','grocer','merchant','pedlar','fletcher','bowyer','glazier','mercer','haberdasher','upholsterer','engraver','jeweller','sculptor','painter','minstrel','jester','watchman','bailiff','constable','butler','groom','coachman','drover','herdsman','swineherd','ploughman','reaper','thresher','hedger','ditcher','sawyer','cartwright','shoemaker','bootmaker','hatmaker','candlemaker','soapmaker','papermaker','clothier','smith','weigher','carver','joiner','harper','falconer','torchbearer','warden','forester','ferrier','sheriff','archer','mariner','pilot','boatman','bargeman','cheesemaker','winemaker','poulterer','fishmonger']
def hist(G,words):return collections.Counter(G.ntok(' '+w) for w in words)
def fit(G,cur,cand,old):
    tgt=hist(G,old);cur=list(cur);cand=[c for c in cand if c not in set(cur)|set(old)];sw=[]
    def dist(h):return sum(abs(h.get(k,0)-tgt.get(k,0)) for k in set(h)|set(tgt))
    ct={c:G.ntok(' '+c) for c in cand}
    while True:
        h=hist(G,cur);best=(dist(h),None)
        for i,w in enumerate(cur):
            for c in cand:
                if c in cur:continue
                h2=h.copy();h2[G.ntok(' '+w)]-=1;h2[ct[c]]+=1;d=dist(h2)
                if d<best[0]:best=(d,(i,c))
        if best[1] is None:break
        i,c=best[1];sw.append((cur[i],c));cur[i]=c
    return cur,sw,tgt
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    import e51_pool_audit as A;G=A.import_gw(a.selftest);import frame_pools_hold as H
    final,sw,tgt=fit(G,H.HOLD_OCC,CAND,G.OCC)
    print('old OCC token-length histogram (the target):',dict(sorted(tgt.items())));print('current HOLD_OCC:',dict(sorted(hist(G,H.HOLD_OCC).items())));print('final:',dict(sorted(hist(G,final).items())))
    print('swaps (out -> in):',sw);print('final list:',final);assert len(final)==30==len(set(final)) and not set(final)&set(G.OCC)
    if a.selftest:print('e51_occ_fit SELFTEST OK (fake tokenizer)')
if __name__=='__main__':main()
