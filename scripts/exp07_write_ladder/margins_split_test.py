"""CPU test: margins_split.split_rows reproduces the old ladder.evaluate arithmetic bit for bit (margins unchanged), gold - foil == margin exactly, the new fields have the item layout, regold_check works."""
import sys;from pathlib import Path;sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np;import margins_split as MS
rng=np.random.default_rng(0);fam={'L3':30,'XP':30,'PS':30,'L1p':5};n=sum(fam.values());mg=rng.normal(-5,2,2*n);ma=rng.normal(-5,2,2*n)
def old(mg,ma):   # the code of ladder.evaluate before the E47 change
    ma=ma[0::2]-ma[1::2];alt={};j=0
    for f,k in fam.items():alt[f]=ma[j:j+k].tolist();j+=k
    mg=mg[0::2]-mg[1::2];out={};i=0
    for f,k in fam.items():out[f]=mg[i:i+k].tolist();i+=k
    out['_alt']=alt;return out
o=old(mg,ma);nw=MS.split_rows(mg,ma,fam)
for f in fam:assert nw[f]==o[f] and nw['_alt'][f]==o['_alt'][f],f   # exact float equality of the old fields
for f in fam:
    assert all(g-h==m for g,h,m in zip(nw['_gold'][f],nw['_foil'][f],nw[f])) and all(g-h==m for g,h,m in zip(nw['_alt_gold'][f],nw['_alt_foil'][f],nw['_alt'][f]))
assert set(nw)=={*fam,'_alt','_gold','_foil','_alt_gold','_alt_foil'} and all(len(nw['_gold'][f])==k for f,k in fam.items())
log0=[{'epoch':1,'margins':{**o,'_gen_nll':1.0}}];log1=[{'epoch':1,'margins':{**{f:[x+1e-5 for x in o[f]] for f in fam},'_alt':o['_alt']}}]
c=MS.regold_check(log0,log1);assert abs(c['1']-1e-5)<1e-9,c
print('margins_split_test OK: old margin fields bit-identical, gold - foil == margin exactly, regold_check',c)
