"""E51 paragraph-level filler pool capacity (sealed streams are NOT read: only --pool-streams, default 9101,9103,9104,9105).
For every target stream X: candidates = background paragraphs of the OTHER pool streams after the nat_filler filters (forbidden strings of X, X's own background, duplicates, canary 8-grams when pg11/pg1342 exist);
need per filler pass = n_slots(X) + MARGIN (150); passes 2..P need (P-1) x that. Reports candidates, need for P = 2 and 3, spare, and the largest P that fits. Also the character budget (pool characters / own characters, nat_filler --budget-only rule: >= 1.2 per filler pass), which is the rule the filler assembly actually needs; the 150-paragraph margin is stricter. The exact token budget is checked later by the filler assembly.
Usage: e51_para_pool.py [--nat-dir DIR] [--layout spread] [--pool-streams 9101,9103,9104,9105] [--targets ...]  |  --selftest"""
import os,sys,json,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_world as NW,nat_filler as NF,e51_nat_table as T
def capacity(nat_dir,target,pool_streams,layout='spread',canary=None):
    w=NW.load_world(NF.world_path(nat_dir,target,layout));own=NF.bg_texts(w);rx=NF.forbid_regex(w);ownset=set(own);seen=set(ownset);drop={'forbid':0,'own_bg':0,'duplicate':0,'canary':0};cand=[];per={}
    for st in pool_streams:
        if st==target:continue
        x=NW.load_world(NF.world_path(nat_dir,st,layout));n0=len(cand)
        for t in NF.bg_texts(x):
            if rx.search(t):drop['forbid']+=1;continue
            if t in ownset:drop['own_bg']+=1;continue
            if t in seen:drop['duplicate']+=1;continue
            if canary and any(len(NF.grams(NF.norm(t))&g)>0 for g in canary.values()):drop['canary']+=1;continue
            seen.add(t);cand.append(t)
        per[st]=len(cand)-n0
    need=len(own)+NF.MARGIN;n=len(cand);oc=sum(len(t) for t in own);cc=sum(len(t) for t in cand);ratio=cc/oc
    return dict(target=target,own_slots=len(own),need_per_pass=need,candidates=n,per_stream=per,dropped=drop,passes_fit=1+n//need,char_ratio=ratio,char_passes_fit=1+int(ratio//1.2),need_P2=need,need_P3=2*need,spare_P2=n-need,spare_P3=n-2*need)
def selftest():
    import tempfile,random;tmp=Path(tempfile.mkdtemp());rng=random.Random(0);V=[f'w{i}' for i in range(4000)];mk=lambda:' '.join(rng.choice(V) for _ in range(30))
    for s in (9101,9103,9104,9105):
        tri=[{'id':i,'pair':f'p{i//2}','X':f'N{s}{i} Z{s}{i}','XF':f'N{s}{i}','XS':f'Z{s}{i}','P':f'T{s}{i}','Q':None,'E':None} for i in range(2)]
        json.dump({'seed':s,'triples':tri,'texts':[{'kind':'bg','text':mk()} for _ in range(100)],'ctrl_persons':[f'C{s}'],'ctrl_places':[f'Cv{s}'],'pseudo_persons':[f'P{s}'],'pseudo_places':[f'Pv{s}'],'pairs':1},open(tmp/f'nat_{s}_spread.json','w'))
    NF.MARGIN=10;r=capacity(tmp,9103,[9101,9103,9104,9105]);assert r['candidates']==300 and r['own_slots']==100 and r['passes_fit']==3,r;print('e51_para_pool SELFTEST OK')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nat-dir',default=None);ap.add_argument('--layout',default='spread');ap.add_argument('--pool-streams',default='9101,9103,9104,9105');ap.add_argument('--targets',default='9103,9101,9104,9105');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    nd=T.find_nat_dir(a.nat_dir);canary=NF.canary_grams() or None
    if not canary:print('WARNING: pg11 / pg1342 not found: canary filter skipped (counts are upper bounds)')
    ps=[int(x) for x in a.pool_streams.split(',')];print('| target | own slots | need per pass | candidates (paragraph pool) | dropped | need P=2 (1 pass) | need P=3 (2 passes) | spare P=3 | max P (150-paragraph margin) | pool chars / own chars (need >= 1.2 per filler pass: 2.4 for P=3) | max P (1.2x char rule) |\n|---|---|---|---|---|---|---|---|---|---|---|')
    for t in [int(x) for x in a.targets.split(',')]:
        r=capacity(nd,t,ps,a.layout,canary);print(f"| {t} | {r['own_slots']} | {r['need_per_pass']} | {r['candidates']} {r['per_stream']} | {r['dropped']} | {r['need_P2']} | {r['need_P3']} | {r['spare_P3']} | {r['passes_fit']} | {r['char_ratio']:.2f} | {r['char_passes_fit']} |")
if __name__=='__main__':main()
