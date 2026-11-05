"""E57 synthetic s64, pair-shuffled stream (new file; CPU; job 960): per-token reads (original pairs) of the states written on the shuffled stream (tag synqsh: M3NMW1N05 = NATQ, M3W1N05 = ALLQ256, M3W8N05 = M3 w=8) next to the states of the same arm and chunk on the original stream (NATQ: job 850 states e57_syn_ncurve_s64, ALLQ256 / M3 w=8: job 800 states e57_syn_curve_s64, end states also jobs 710/730). Per arm and matched chunk, 95% bootstrap CIs by fact (16 facts) of: first-token dA, dA-dB, branch-token rank ratio and its improvement over W0, full-sum rank ratio and its improvement, branch-gap gain; shuffled minus original; canary rises of both runs.
  python3 e57_syn_shuf_report.py --shuf T1 --orig T2 [--orig T3 ...] --out J | --selftest"""
import sys,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder'
import e57_syn_tokread as TR
ARMS={'M3NMW1N05':'NATQ','M3W1N05':'ALLQ256','M3W8N05':'M3 w=8'}
def per_fact(m,W0,facts):
    ok=[f for f in facts if m[f]['B_first'] is not None and W0[f]['B_first'] is not None]
    dA=np.array([m[f]['A_first']-W0[f]['A_first'] for f in ok]);dB=np.array([m[f]['B_first']-W0[f]['B_first'] for f in ok])
    ok2=ok;dAf=np.array([m[f]['A_full']-W0[f]['A_full'] for f in ok2]);dBf=np.array([m[f]['B_full']-W0[f]['B_full'] for f in ok2])
    g=lambda k:np.array([m[f][k] for f in facts],float);w=lambda k:np.array([W0[f][k] for f in facts],float)
    return dict(first_dA=dA,first_dAB=dA-dB,full_dAB=dAf-dBf,rr_branch=g('rank_branch')/16,rr_branch_impr=(w('rank_branch')-g('rank_branch'))/16,rr_full=g('rank_full')/16,rr_full_impr=(w('rank_full')-g('rank_full'))/16,gap_branch_gain=g('gap_branch')-w('gap_branch'),top1_branch=(g('rank_branch')==1).astype(float))
def run(a):
    sd=a.seed;S=json.load(open(a.shuf));tokids=S['tokids'];meta=json.load(open(next(iter(TR.arms_curve(sd,'synqsh').values()))['read_json']))['meta']
    O={};
    for f in a.orig:
        d=json.load(open(f));assert d['tokids']==tokids,'token ids differ';O.update(d['arms'])
    W0S=TR.metrics(S['W0'],tokids,meta);facts=sorted(W0S);res={'facts':facts,'curve':{},'shuffled_minus_original':{},'canary':{}}
    for l,t in S['arms'].items():
        arm,ch=l.split('@c');ch=int(ch);m=TR.metrics(t,tokids,meta);pf=per_fact(m,W0S,facts);res['curve'].setdefault(arm,{}).setdefault(str(ch),{})['shuffled']={k:TR.boot(v) for k,v in pf.items()}
        if l in O:
            W0O=TR.metrics(a_w0:=json.load(open(a.orig[0]))['W0'],tokids,meta);po=per_fact(TR.metrics(O[l],tokids,meta),W0O,facts)
            res['curve'][arm][str(ch)]['original']={k:TR.boot(v) for k,v in po.items()};res['shuffled_minus_original'].setdefault(arm,{})[str(ch)]={k:TR.boot(pf[k]-po[k]) for k in pf}
    for arm,tags in (('M3NMW1N05',('synqsh','synqn')),('M3W1N05',('synqsh','synq','synqc')),('M3W8N05',('synqsh','synqc'))):
        for tag in tags:
            f=EXP/f'e57_synq/E55_{tag}_{arm}_s{sd}.saved.json'
            if not f.exists():continue
            x=json.load(open(f));w0=x['canary_w0'];res['canary'].setdefault(arm,{})[tag]={k:dict(gen_nll_rise=v['canary']['_gen_nll']-w0['_gen_nll'],tail_nll_rise=v['canary']['_tail_nll']-w0['_tail_nll']) for k,v in x['reads'].items() if 'canary' in v}
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out)
def selftest():
    f=[0,1];W0={i:dict(A_first=-5.,B_first=-5.,A_full=-9.,B_full=-9.,rank_branch=8,rank_full=8,gap_branch=0.) for i in f};m={i:dict(A_first=-4.,B_first=-4.5,A_full=-8.,B_full=-8.5,rank_branch=4,rank_full=2,gap_branch=1.) for i in f}
    p=per_fact(m,W0,f);assert np.allclose(p['first_dA'],1) and np.allclose(p['first_dAB'],.5) and np.allclose(p['rr_branch_impr'],.25) and np.allclose(p['rr_full'],2/16);print('e57_syn_shuf_report selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shuf');ap.add_argument('--orig',action='append');ap.add_argument('--out');ap.add_argument('--seed',type=int,default=64);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    selftest() if a.selftest else run(a)
