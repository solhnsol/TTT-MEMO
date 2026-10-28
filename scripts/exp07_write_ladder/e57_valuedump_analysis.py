"""E57 value-corrected mass analysis of the shadowing-key sets (CPU only; reads E57_valuedump_W0.npz of job 300 and e57_shadowkey_masks.npz of job 299; numbers only, no criterion). Scores per selected row over the far keys (window key c >= 4, distance >= 128): ALL, SH (plain mass, control), ALLV, SHV (sum over heads and layers of p x ||v||), ALLVn, SHVn (per-head value-normalised share over the window keys, summed over heads; row total = number of heads).
 1 last (N-th) key of the top-N sets, N = 1 / 3 / 10 % of the row's far keys (the sizes of job 299), by each score: its score (quartiles, mean), and the share of the row's far score held by the set
 2 threshold sets {far keys with score >= tau}, tau in .005 .01 .02 .05 .1 .2: keys per row and share of the far context (quartiles, mean), share of rows with none
 3 overlap (Jaccard, mean over rows) of the top-N set by a value-corrected score with the top-N set by the plain mass of the same head set (ALLV vs ALL, SHV vs SH), ALLV vs SHV, and of each p-set with each threshold set
  .venv/bin/python scripts/exp07_write_ladder/e57_valuedump_analysis.py --exp EXP_DIR --out OUT.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_shadowkey_masks as SKM
import e57_shadowkey_mass_analysis as MA
BAND=128;TAUS=MA.TAUS;PS=MA.PS;SC=('ALL','SH','ALLV','SHV','ALLVn','SHVn')
def jac(a,b):
    a=set(a.tolist());b=set(b.tolist());u=len(a|b);return len(a&b)/u if u else np.nan
def analyse(chunks):
    """chunks: list of dict(rows, S{score: [R,8192] float32}) -> json dict"""
    acc={}
    add=lambda p,v:acc.setdefault(p,[]).append(v)
    for c in chunks:
        for r,i in enumerate(c['rows']):
            i=int(i);lo,hi=SKM.far_range(i,BAND);n=hi-lo+1;f=np.arange(lo,hi+1);top={}
            for sc in SC:
                row=c['S'][sc][r][lo:hi+1].astype(np.float32);o=np.argsort(-row,kind='stable');tot=row.sum()
                for pn,fr in PS:
                    N=SKM.sizes(i,BAND,fr)[1];k=f[o[:N]];top[(sc,pn)]=k;add(f'last|{sc}|{pn}',float(row[o[N-1]]));add(f'held|{sc}|{pn}',float(row[o[:N]].sum()/tot))
                for t in TAUS:
                    ts=f[row>=t];add(f'thr_keys|{sc}|{t}',len(ts));add(f'thr_share|{sc}|{t}',len(ts)/n)
                    if len(ts):
                        for pn,fr in PS:add(f'jac_thr|{sc}|{pn}|{t}',jac(top[(sc,pn)],ts))
            for pn,fr in PS:
                add(f'jac_V_vs_plain|ALL|{pn}',jac(top[('ALLV',pn)],top[('ALL',pn)]));add(f'jac_V_vs_plain|SH|{pn}',jac(top[('SHV',pn)],top[('SH',pn)]))
                add(f'jac_Vn_vs_plain|ALL|{pn}',jac(top[('ALLVn',pn)],top[('ALL',pn)]));add(f'jac_Vn_vs_plain|SH|{pn}',jac(top[('SHVn',pn)],top[('SH',pn)]))
                add(f'jac_ALLV_vs_SHV|{pn}',jac(top[('ALLV',pn)],top[('SHV',pn)]));add(f'jac_ALLVn_vs_SHVn|{pn}',jac(top[('ALLVn',pn)],top[('SHVn',pn)]))
    out={p:MA.q(v) for p,v in acc.items()}
    for sc in SC:
        for t in TAUS:out[f'thr_none|{sc}|{t}']=float((np.asarray(acc[f'thr_keys|{sc}|{t}'])==0).mean())
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--exp');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    E=pathlib.Path(a.exp);V=np.load(E/'e57_valuedump'/'E57_valuedump_W0.npz');chunks=[]
    for k in [int(x) for x in V['chunks']]:
        chunks.append(dict(rows=V[f'rows_{k}'],S={s:V[f'c{k}/{s}'] for s in SC}));print('chunk',k,flush=True)
    r=analyse(chunks);json.dump(r,open(a.out,'w'));print('done',len(r))
def selftest():
    rng=np.random.default_rng(0);R=3;rows=np.array([500,700,1000]);S={s:(rng.random((R,8192))*.3).astype(np.float32) for s in SC};S['ALLV']=S['ALL'].copy()
    o=analyse([dict(rows=rows,S=S)]);assert o['jac_V_vs_plain|ALL|p1']['mean']==1.0 and o['last|ALL|p1']['n']==3
    assert abs(o['thr_keys|ALL|0.1']['mean']-np.mean([(S['ALL'][r][4:SKM.far_range(int(i),BAND)[1]+1]>=.1).sum() for r,i in enumerate(rows)]))<1e-9
    print('e57_valuedump_analysis selftest ok')
if __name__=='__main__':main()
