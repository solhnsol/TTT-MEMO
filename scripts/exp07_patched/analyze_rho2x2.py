"""2x2 power flag (PROTOCOL 2x2 amendment 03:20 KST item 3), computed before any 2x2 D is read.
For each suppression cell (natc2: native c_k; mskc2: masked c_k) and seed: s_equiv = kappa* x rho (mean of worlds A/B), the
screen's c2ch (norm basis) mean D_alt interpolated at s_equiv linearly in log s over the evaluated s in {1, 2, 4, 8} (for
s_equiv < 1: D(1) x s_equiv, first order; above 8: D(8)); averaged over the 3 seeds; flag "underpowered expected" if it is
< 2 x the paired-CI half-width of the masked pass-1 run (L0mwfs x1 pass 1, D_alt, 45 pairs pooled, AL.ci bootstrap).
Usage: analyze_rho2x2.py KAPPA"""
import sys,json,math,importlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
def mods(sd):
    sys.argv=['x',str(sd)]
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
def main():
    kap=float(sys.argv[1]);AL,S=mods(101);OUT=S.OUT;rho=json.load(open(OUT/'rho2x2.json'));res={'kappa':kap,'cells':{}};pool=[]
    for sd in (101,102,103):
        AL,S=mods(sd);X=S.load('OS2','c2ch_nm');c=S.curve(*X);d={ep:float(sa['D'].mean()) for ep,s1,sa,dd in c[1:]};res.setdefault('c2ch_Dalt',{})[sd]=d
        X=S.load('L0mwfs','x1');c=S.curve(*X);pool.append(c[1][2]['D'])
    _,lo,hi=AL.ci(np.concatenate(pool));hw=(hi-lo)/2;res['q3_pass1_halfwidth']=hw
    def interp(d,s):
        ks=sorted(d)
        if s<=ks[0]:return d[ks[0]]*s/ks[0]
        if s>=ks[-1]:return d[ks[-1]]
        for a,b in zip(ks,ks[1:]):
            if a<=s<=b:t=(math.log(s)-math.log(a))/(math.log(b)-math.log(a));return d[a]+t*(d[b]-d[a])
    for cell,key in (('natc2','rho_native'),('mskc2','rho_masked')):
        per={}
        for sd in (101,102,103):
            r=np.mean([rho[str(sd)][w][key] for w in 'AB']);se=kap*r;per[sd]={'rho':r,'s_equiv':se,'D_interp':interp(res['c2ch_Dalt'][sd],se),'extrapolated':bool(se<1 or se>8)}
        m=float(np.mean([v['D_interp'] for v in per.values()]));res['cells'][cell]={'per_seed':per,'mean_D_interp':m,'threshold':2*hw,'underpowered_expected':m<2*hw,'extrapolated_seeds':{sd:v['s_equiv'] for sd,v in per.items() if v['extrapolated']}}
        print(cell,json.dumps(res['cells'][cell]),flush=True)
    json.dump(res,open(OUT/'rho2x2_flag.json','w'),indent=1)
if __name__=='__main__':main()
