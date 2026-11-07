"""E57 learned-selector write arms on the hold-out worlds s65/s73/s74 (new file; CPU; GPM thread, Hans 11/07): M3TLBW16N05 (model b, true deciding rows) and M3TLAW16N05 (model a, g128>2 rows) against the existing arms ALLQ256 (M3W1N05), M3 w=8, TOP5 (M3T05W16N05), ABS2 (M3TA2W16N05): at the last appearance (c127) branch-token rank ratio (lower better), first-token dA, dA-dB, paired differences (fact-level bootstrap, 2000 draws, per world and pooled over the 3 worlds), inline in-window / out-of-window reads (rank-ratio improvement vs W0, gold log-prob gain) and canary rises. Input: e57_syn_learned_s<seed>/e57_syn_tokread_s<seed>_merged.json + e57_synq/E55_synqrl_*.
  python3 e57_learned_report.py --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder'
WORLDS=((65,16),(73,4),(74,32));NEW=('M3TLBW16N05','M3TLAW16N05');REF=('M3W1N05','M3W8N05','M3T05W16N05','M3TA2W16N05');ARMS=REF+NEW
def bt(x,B=2000,seed=1):
    x=np.asarray(x,float);ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def inline_arrays(sc,w0,meta):
    kind=np.asarray(meta['kind']);gi=np.asarray(meta['gi']);rows=[r for r in range(len(kind)) if kind[r]=='fact'];sc=np.asarray(sc,float);w0=np.asarray(w0,float)
    rr=lambda m:np.array([(1+(m[r]>m[r][gi[r]]).sum())/m.shape[1] for r in rows])
    return dict(rr_impr=rr(w0)-rr(sc),gold_lp=np.array([sc[r][gi[r]]-w0[r][gi[r]] for r in rows]))
def world(sd):
    import e57_syn_tokread as TR
    d=json.load(open(EXP/f'e57_syn_learned_s{sd}/e57_syn_tokread_s{sd}_merged.json'));rj=json.load(open(EXP/f'e57_synq/E55_synqrl_M3W1N05_s{sd}.read.json'));meta=rj['meta'];W0=TR.metrics(d['W0'],d['tokids'],meta);facts=sorted(W0)
    A={}
    for ar in ARMS:
        m=TR.metrics(d['arms'][f'{ar}@c127'],d['tokids'],meta);dA=np.asarray([m[f]['A_first']-W0[f]['A_first'] for f in facts]);dB=np.asarray([m[f]['B_first']-W0[f]['B_first'] for f in facts])
        A[ar]=dict(rr=np.asarray([m[f]['rank_branch']/16.0 for f in facts]),dA=dA,dAB=dA-dB)
    I={}
    for ar in ARMS:
        pj=lambda e:next(p for p in (EXP/f'e57_synq/E55_{t}_{ar}_s{sd}.{e}.json' for t in ('synqrl','synqrp')) if p.exists() and not p.is_symlink() or p.exists() and e=='read');x=json.load(open(pj('read')));rd=x['read']['128'];w=x['W0'];s=json.load(open(pj('saved')));cw=s['canary_w0'];k=max(s['reads'],key=int)
        I[ar]=dict(inw=inline_arrays(rd['in_g0'],w['in_g0@128'],x['meta']),out=inline_arrays(rd['out'],w['out'],x['meta']),canary=dict(gen_nll_rise=s['reads'][k]['canary']['_gen_nll']-cw['_gen_nll'],tail_nll_rise=s['reads'][k]['canary']['_tail_nll']-cw['_tail_nll']))
    return A,I
def summarize(A,I):
    r={'arms':{ar:dict(rr_branch=bt(A[ar]['rr']),first_dA=bt(A[ar]['dA']),first_dA_minus_dB=bt(A[ar]['dAB']),in_window={m:bt(I[ar]['inw'][m]) for m in ('rr_impr','gold_lp')},out_of_window={m:bt(I[ar]['out'][m]) for m in ('rr_impr','gold_lp')},canary=I[ar]['canary']) for ar in ARMS},'paired':{}}
    for n in NEW:
        for b in REF+tuple(x for x in NEW if x!=n):
            r['paired'][f'{n}_minus_{b}']=dict(rr_branch=bt(A[n]['rr']-A[b]['rr']),first_dA=bt(A[n]['dA']-A[b]['dA']),first_dA_minus_dB=bt(A[n]['dAB']-A[b]['dAB']),in_window_rr_impr=bt(I[n]['inw']['rr_impr']-I[b]['inw']['rr_impr']),out_of_window_rr_impr=bt(I[n]['out']['rr_impr']-I[b]['out']['rr_impr']),out_of_window_gold_lp=bt(I[n]['out']['gold_lp']-I[b]['out']['gold_lp']))
    return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:x=bt([1,2,3]);assert abs(x['mean']-2)<1e-9;print('e57_learned_report selftest ok');return
    res={'worlds':{},'definitions':__doc__.split('\n')[0]};All=[]
    for sd,dens in WORLDS:
        A,I=world(sd);All.append((A,I));res['worlds'][f's{sd}_density{dens}']=summarize(A,I)
    cA={ar:{k:np.concatenate([w[0][ar][k] for w in All]) for k in ('rr','dA','dAB')} for ar in ARMS};cI={ar:dict(inw={k:np.concatenate([w[1][ar]['inw'][k] for w in All]) for k in ('rr_impr','gold_lp')},out={k:np.concatenate([w[1][ar]['out'][k] for w in All]) for k in ('rr_impr','gold_lp')},canary={k:float(np.mean([w[1][ar]['canary'][k] for w in All])) for k in ('gen_nll_rise','tail_nll_rise')}) for ar in ARMS}
    res['pooled_3_worlds']=summarize(cA,cI);pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out)
if __name__=='__main__':main()
