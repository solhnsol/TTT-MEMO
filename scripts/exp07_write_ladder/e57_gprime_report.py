"""E57 online / W0 g' ABS2 write arms on the tune worlds s62 (v3_16_1) and s72 (v3_32_1) (new file; CPU; GPM thread, Hans 11/07): arms M3OS2W16N05 (online g', suffix-only band) and M3TS2W16N05 (W0 g') against the existing M3W1N05 (ALLQ256), M3TA2W16N05 (W0-ABS2), M3T05W16N05 (TOP5), M3OA2W16N05 (online g, jobs 1151/1152; inline reads job 1153): at the last appearance (c127) branch-token rank ratio, first-token dA, dA-dB (fact-level bootstrap), inline in-window / out-of-window rank-ratio improvement and gold log-prob gain vs W0, canary rises, selected-row share, paired differences new minus each reference. Input: e57_syn_gprime_s<seed>/e57_syn_tokread_s<seed>.json (new arms) + e57_syn_rules_s<seed>/ + e57_syn_onlinerule_s<seed>/ tokreads (references; token ids and W0 checked).
  python3 e57_gprime_report.py --seed S --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder'
NEW=('M3OS2W16N05','M3TS2W16N05');REF=('M3W1N05','M3W8N05','M3T05W16N05','M3TA2W16N05','M3OA2W16N05');ARMS=REF+NEW
def bt(x,B=2000,seed=1):
    x=np.asarray(x,float);ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def inline_arrays(sc,w0,meta):
    kind=np.asarray(meta['kind']);gi=np.asarray(meta['gi']);rows=[r for r in range(len(kind)) if kind[r]=='fact'];sc=np.asarray(sc,float);w0=np.asarray(w0,float)
    rr=lambda m:np.array([(1+(m[r]>m[r][gi[r]]).sum())/m.shape[1] for r in rows])
    return dict(rr_impr=rr(w0)-rr(sc),gold_lp=np.array([sc[r][gi[r]]-w0[r][gi[r]] for r in rows]))
def pj(sd,ar,e,tags):
    for t in tags:
        p=EXP/f'e57_synq/E55_{t}_{ar}_s{sd}.{e}.json'
        if p.exists():return p
    raise FileNotFoundError((sd,ar,e))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:x=bt([1,2,3]);assert abs(x['mean']-2)<1e-9;print('e57_gprime_report selftest ok');return
    import e57_syn_tokread as TR
    sd=a.seed;T={n:json.load(open(p)) for n,p in (('new',EXP/f'e57_syn_gprime_s{sd}/e57_syn_tokread_s{sd}.json'),('rules',EXP/f'e57_syn_rules_s{sd}/e57_syn_tokread_s{sd}.json'),('online',EXP/f'e57_syn_onlinerule_s{sd}/e57_syn_tokread_s{sd}.json'))}
    for n in ('rules','online'):
        assert T[n]['tokids']==T['new']['tokids'];mx=max(abs(sum(x)-sum(y)) for x,y in zip(T[n]['W0'],T['new']['W0']));assert mx<1e-3,(n,mx)
    arms={};[arms.update(T[n]['arms']) for n in ('rules','online','new')]
    rj=json.load(open(EXP/f'e57_synq/E55_synqrl_M3W1N05_s{sd}.read.json'));meta=rj['meta'];W0=TR.metrics(T['new']['W0'],T['new']['tokids'],meta);facts=sorted(W0);A={};I={};share={}
    TAGS={'M3OA2W16N05':('synqoi',),'M3OS2W16N05':('synqgp',),'M3TS2W16N05':('synqgp',)}
    for ar in ARMS:
        m=TR.metrics(arms[f'{ar}@c127'],T['new']['tokids'],meta);dA=np.asarray([m[f]['A_first']-W0[f]['A_first'] for f in facts]);dB=np.asarray([m[f]['B_first']-W0[f]['B_first'] for f in facts]);A[ar]=dict(rr=np.asarray([m[f]['rank_branch']/16.0 for f in facts]),dA=dA,dAB=dA-dB)
        tg=TAGS.get(ar,('synqrl','synqrp'))
        try:x=json.load(open(pj(sd,ar,'read',tg)));rd=x['read']['128'];x['W0']['in_g0@128']
        except (FileNotFoundError,KeyError):I[ar]=None;share[ar]=None;continue      # no inline reads for this arm (e.g. online g on s72: job 1154 not run)
        x=json.load(open(pj(sd,ar,'read',tg)));rd=x['read']['128'];w=x['W0'];s=json.load(open(pj(sd,ar,'saved',tg)));cw=s['canary_w0'];k=max(s['reads'],key=int)
        I[ar]=dict(inw=inline_arrays(rd['in_g0'],w['in_g0@128'],x['meta']),out=inline_arrays(rd['out'],w['out'],x['meta']),canary=dict(gen_nll_rise=s['reads'][k]['canary']['_gen_nll']-cw['_gen_nll'],tail_nll_rise=s['reads'][k]['canary']['_tail_nll']-cw['_tail_nll']))
        share[ar]=s.get('online_share_mean')
    for ar,nm in (('M3TS2W16N05','qtS2'),('M3TA2W16N05','qtA2'),('M3T05W16N05','qt05')):
        f=EXP/f'e57_synq/syn_s{sd}_gprime.npz';d=np.load(f);ks=range(int(d['chunks'][0]),int(d['chunks'][1]));key=nm if nm!='qtA2' else None
        if nm=='qtS2':share[ar]=float(np.mean([np.unpackbits(d[f'{nm}_{k}'])[:1024].sum()/max(1,np.unpackbits(d[f'el_{k}'])[:1024].sum()) for k in ks]))
    res={'seed':sd,'arms':{ar:dict(rr_branch=bt(A[ar]['rr']),first_dA=bt(A[ar]['dA']),first_dA_minus_dB=bt(A[ar]['dAB']),in_window={m:bt(I[ar]['inw'][m]) for m in ('rr_impr','gold_lp')},out_of_window={m:bt(I[ar]['out'][m]) for m in ('rr_impr','gold_lp')},canary=(I[ar] or {}).get('canary'),selected_row_share_of_eligible=share.get(ar)) if I[ar] is not None else dict(rr_branch=bt(A[ar]['rr']),first_dA=bt(A[ar]['dA']),first_dA_minus_dB=bt(A[ar]['dAB']),inline='not available') for ar in ARMS},'paired':{}}
    for n in NEW:
        for b in ARMS:
            if b==n:continue
            if I[b] is None:res['paired'][f'{n}_minus_{b}']=dict(rr_branch=bt(A[n]['rr']-A[b]['rr']),first_dA=bt(A[n]['dA']-A[b]['dA']),first_dA_minus_dB=bt(A[n]['dAB']-A[b]['dAB']));continue
            res['paired'][f'{n}_minus_{b}']=dict(rr_branch=bt(A[n]['rr']-A[b]['rr']),first_dA=bt(A[n]['dA']-A[b]['dA']),first_dA_minus_dB=bt(A[n]['dAB']-A[b]['dAB']),in_window_rr_impr=bt(I[n]['inw']['rr_impr']-I[b]['inw']['rr_impr']),out_of_window_rr_impr=bt(I[n]['out']['rr_impr']-I[b]['out']['rr_impr']),out_of_window_gold_lp=bt(I[n]['out']['gold_lp']-I[b]['out']['gold_lp']))
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out)
if __name__=='__main__':main()
