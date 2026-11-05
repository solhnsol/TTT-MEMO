"""E57 synthetic s64, partial-mask arms M3-D30 / M3-D50 / M3-R30 (new file; CPU): extra measures next to the curve report: (1) mid-to-end change (appearance 96 -> 128, chunk 95 -> 127) of the first-token dA, d(A-B) and branch-rank improvement per arm (paired over the facts, 95% bootstrap), (2) in-window read at a = 128 (stored inline read 'in_g0'): gold summed log-prob minus W0 and rank ratio per fact, (3) canary rises at the save points, (4) rows masked per chunk and share of the eligible rows (syn_s64_partial_masks.json).
  python3 e57_syn_partial_report.py --tokread F --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import e57_syn_tokread as TR
EXP=ROOT/'experiments/exp07_write_ladder'
NEW=['M3D30W8N05','M3D50W8N05','M3R30W8N05'];BASE=['M3W1N05','M3W8N05']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tokread');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:print('e57_syn_partial_report selftest ok');return
    d=json.load(open(a.tokread));meta=json.load(open(EXP/'e57_synq/E55_synqpm_M3D30W8N05_s64.read.json'))['meta'];W0=TR.metrics(d['W0'],d['tokids'],meta);facts=sorted(W0);M={l:TR.metrics(t,d['tokids'],meta) for l,t in d['arms'].items()};res={'mid_to_end':{},'in_window_a128':{},'canary':{},'masks':None}
    fn={'first_dA':lambda m,w:m['A_first']-w['A_first'],'first_dAB':lambda m,w:(m['A_first']-w['A_first'])-(m['B_first']-w['B_first']) if m['B_first'] is not None and w['B_first'] is not None else None,'rr_branch_improvement':lambda m,w:(w['rank_branch']-m['rank_branch'])/16.0}
    for arm in BASE+NEW:
        e={}
        for k,f in fn.items():
            v=[]
            for x in facts:
                m1=f(M[f'{arm}@c127'][x],W0[x]);m0=f(M[f'{arm}@c95'][x],W0[x]);v.append(np.nan if m1 is None or m0 is None else m1-m0)
            e[k]=TR.boot(v)
        res['mid_to_end'][arm]=e
    # in-window read (stored inline read of the writes)
    srcs={'NATQ':EXP/'e57_synq/E55_synq_NATQ_s64.read.json','M3W1N05':EXP/'e57_synq/E55_synqc_M3W1N05_s64.read.json','M3W8N05':EXP/'e57_synq/E55_synqc_M3W8N05_s64.read.json',**{n:EXP/f'e57_synq/E55_synqpm_{n}_s64.read.json' for n in NEW}}
    kind=np.asarray(meta['kind']);gi=np.asarray(meta['gi']);rows=[r for r in range(len(kind)) if kind[r]=='fact']
    def stats(sc,w0):
        sc=np.asarray(sc,np.float64);w0=np.asarray(w0,np.float64);g=[sc[r][gi[r]]-w0[r][gi[r]] for r in rows];rr=[(1+(sc[r]>sc[r][gi[r]]).sum())/sc.shape[1] for r in rows];rr0=[(1+(w0[r]>w0[r][gi[r]]).sum())/w0.shape[1] for r in rows]
        return dict(gold_sum_lp_minus_W0=TR.boot(g),rank_ratio=TR.boot(rr),rank_ratio_improvement=TR.boot([b-a_ for a_,b in zip(rr,rr0)]))
    for arm,f in srcs.items():
        if not f.exists():continue
        x=json.load(open(f));rd=x['read'].get('128',{});w=x['W0']
        if 'in_g0' in rd and 'in_g0@128' in w:res['in_window_a128'][arm]=stats(rd['in_g0'],w['in_g0@128'])
        if 'out' in rd:res['in_window_a128'].setdefault(arm,{})['out_of_window_a128']=stats(rd['out'],w['out'])
    for arm in ['M3W1N05','M3W8N05']+NEW:
        f=EXP/('e57_synq/E55_synqc_%s_s64.saved.json'%arm if arm in BASE else 'e57_synq/E55_synqpm_%s_s64.saved.json'%arm)
        if f.exists():
            x=json.load(open(f));w0=x['canary_w0'];res['canary'][arm]={k:dict(gen_nll_rise=v['canary']['_gen_nll']-w0['_gen_nll'],tail_nll_rise=v['canary']['_tail_nll']-w0['_tail_nll']) for k,v in x['reads'].items() if 'canary' in v}
    mf=EXP/'e57_synq/syn_s64_partial_masks.json'
    if mf.exists():res['masks']=json.load(open(mf))['summary']
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out)
if __name__=='__main__':main()
