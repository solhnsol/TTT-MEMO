"""E49 CPU test (stub tokenizer, no jax, no data): candidate construction from the real gen_world.probes of a built world, cell records, analysis path, and the file-driven --analyze mode of e49_candidates.py."""
import sys,json,subprocess,tempfile,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[0]/'exp04_what_ttt_retains'))
import e48_world_test as W,e49_stats as ES
gw=W.load_gw(W.EX04/'gen_world.py','gen_world');w=gw.build_world(101);name={t['id']:f'N{t["id"]}' for t in w['triples']}
items=[it for it in gw.probes(w,name) if it['level']=='L3'];assert len(items)==30
places,gi,fi,rows,pairs=ES.candidate_items(items,w['triples']);assert len(places)==30 and len(rows)==900 and rows[0][1]==places[0] and rows[31][1]==places[1] and rows[31][0]==items[1]['prefix']
assert len(set(pairs))==15
# synthetic scorer: a "model" that prefers the bound place of world A with strength s, flipped in B, plus a place prior
rng=np.random.default_rng(0);prior=rng.normal(0,1,30)
def S_of(s,world):
    S=np.tile(prior,(30,1))+rng.normal(0,.1,(30,30))
    for i in range(30):S[i,gi[i] if world=='A' else fi[i]]+=s
    return S
d=Path(tempfile.mkdtemp());recs=[]
for cell,s in (('ref',3.0),('strong',6.0),('none',0.0)):
    r=ES.cell_record(cell,places,gi,fi,pairs,{w_:{'R1':S_of(s,w_),'alt':S_of(s,w_)} for w_ in 'AB'});json.dump(r,open(d/f'e49_cand_s101_{cell}.json','w'));recs.append(r)
out,text=ES.analyze(recs,'ref',300);print(text.splitlines()[0][:100])
c=out['alt']['change'];assert c['strong']['top1'][1] is True or c['strong']['abs_lp'][1] is True
assert c['strong']['abs_lp'][0][0]>2.5 and c['none']['abs_lp'][0][0]<-2.5 and c['none']['abs_lp'][1] is True
assert out['alt']['cell']['strong']['top1']['AB'][0]>out['alt']['cell']['none']['top1']['AB'][0]
assert out['alt']['cell']['none']['top1']['AB'][0]<0.3   # chance is 0.033 (+ the place prior)
# file-driven analyze mode of the script (no jax)
p=subprocess.run([sys.executable,str(HERE/'e49_candidates.py'),'--analyze','--seed','101','--out',str(d),'--cell','ref=x','--cell','strong=x','--cell','none=x','--ref','ref','--nboot','200'],capture_output=True,text=True);assert p.returncode==0,p.stderr[-800:];assert 'strong' in p.stdout and (d/'e49_candidates_s101.json').exists()
# mismatched items are rejected
bad=json.loads(json.dumps(recs[0]));bad['gi'][0]=(bad['gi'][0]+1)%30
try:ES.analyze([recs[1],bad],'ref');raise SystemExit('different items not rejected')
except AssertionError:pass
# a gold/foil that is not a pair place is rejected
x=[dict(i) for i in items];x[0]['foil']=' '+w['triples'][(items[0]['tid']+2)%30]['P']
try:ES.candidate_items(x,w['triples']);raise SystemExit('wrong foil not rejected')
except AssertionError:pass
print('e49_test ok')
