"""E39 CPU unit tests for Sets.oraclefill / randfill (synthetic sentence table, or --world-seed on the server for the real one) and bit-identity of the old cells.
Usage: python3 e39_sets_test.py [--orig PATH_TO_ORIGINAL_s3_sets.py] [--world-seed 101]"""
import sys,json,argparse,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_sets as S
def load(p):
    sp=importlib.util.spec_from_file_location('s3_sets_orig',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def synth():
    t=[]
    for tr in range(30):
        for r in range(1,5):
            pos=(r-1)*6144+tr*150+5;t.append(dict(idx=len(t),t=f't{tr}',pair=f'p{tr//2}',round=r,place=[pos],names=[pos-3,pos-2],spans=[[pos],[pos-3,pos-2]],text=''))
    return sorted(t,key=lambda s:s['place'][0])
def real(seed):
    import gen_world as G,ladder as LD;from ladder import gw
    import suppress as SP
    w=G.build_world(seed);return S.sentence_table(w,False,G.TOK,gw,SP)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--orig');ap.add_argument('--world-seed',type=int);a=ap.parse_args()
    tab=real(a.world_seed) if a.world_seed else synth();res={'table':'real' if a.world_seed else 'synthetic','n_sentences':len(tab),'checks':{},'chunks':[]};ok=True
    s0=S.Sets(tab,101,2,0.0);s=S.Sets(tab,101,2,0.25);s2=S.Sets(tab,101,2,0.25)
    ks=[k for k in range(0,63) if s.oracle(k)];res['chunks_with_O']=len(ks)
    c1=all(s0.oraclefill(k)[0]==s0.oracle(k) for k in range(63));c2=True;c3=True;c4=True;c5=True
    for k in ks:
        O=s.oracle(k);S_,fr=s.oraclefill(k);k0=k*S.CH;lo=max(0,k0-S.W);T=int(round(0.25*(k0-lo)));F=sorted(set(S_)-set(O))
        c2&=set(O)<=set(S_) and not (set(F)&set(O));c3&=(S_,fr)==s2.oraclefill(k);c5&=all(lo<=p<k0 for p in F)&(len(set(S_))==len(S_))
        ok4=len(S_)==T if len(O)<=T and (k0-lo)-len(O)>=T-len(O) else len(S_)<=T
        c4&=ok4;rf,_=s.randfill(k);res['chunks'].append({'k':k,'n_O':len(O),'n_S':len(S_),'T':T,'match_frac':fr,'randfill_n':len(rf),'randfill_overlaps_O':len(set(rf)&set(O))})
    res['checks']={'1_frac0_equals_oracle':c1,'2_O_subset_and_F_disjoint':c2,'3_deterministic':c3,'4_size_T_or_max':c4,'5_range_rule':c5}
    if a.orig:
        o=load(a.orig);so=o.Sets(tab,101,2);same=True
        for cell in ('oracle','othent','randdist','oraclesub','othentsub','recency','prefixall','native3'):
            for k in range(63):
                same&=so.chunk_set(cell,k)==s.chunk_set(cell,k)
        res['checks']['6_old_cells_identical']=bool(same)
    ok=all(res['checks'].values());res['ALL_PASS']=ok
    mf=[c['match_frac'] for c in res['chunks'] if c['k']>=8];res['match_frac_k_ge_8']={'mean':sum(mf)/max(len(mf),1),'min':min(mf) if mf else None}
    print(json.dumps(res['checks'],indent=1),'ALL_PASS',ok);print('match_frac (k>=8) mean/min',res['match_frac_k_ge_8']);print(json.dumps(res['chunks'][:6]))
    out=Path(__file__).resolve().parent/f"e39_sets_test_{res['table']}.json";json.dump(res,open(out,'w'),indent=1);print('wrote',out)
    sys.exit(0 if ok else 1)
if __name__=='__main__':main()
