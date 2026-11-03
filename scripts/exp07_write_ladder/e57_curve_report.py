"""E57 curve of the branch-token rank over the cycle (new file; CPU; thread 'ALLQ보다 나은 가림 방법 찾기', Hans '사이클 안에서의 w=8 곡선'): from the per-token reads of the intermediate states (labels <arm>@c<chunk>) the branch-token / first-token / full-sum rank ratio, top-1 and (synthetic) the first-token dA, dA-dB per arm and save point, 95% CIs by fact (synthetic) or frame (news), and the paired difference of the two arms per save point.
  python3 e57_curve_report.py --kind syn --seed 64|61 --tokread F --out J | --kind news --tokread merged.json --out J | --selftest"""
import sys,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder'
def split_label(l):
    a,c=l.split('@c');return a,int(c)
def syn(a):
    import e57_syn_tokread as TR
    d=json.load(open(a.tokread));rj=json.load(open(next(iter(TR.arms_curve(a.seed,a.tag).values()))['read_json']));meta=rj['meta'];tokids=d['tokids'];W0=TR.metrics(d['W0'],tokids,meta);facts=sorted(W0);M={l:TR.metrics(t,tokids,meta) for l,t in d['arms'].items()};res={'seed':a.seed,'reproduction':d['reproduction'],'curve':{}}
    def dl(m,sc):
        ok=[f for f in facts if m[f][f'B_{sc}'] is not None and W0[f][f'B_{sc}'] is not None];dA=np.asarray([m[f][f'A_{sc}']-W0[f][f'A_{sc}'] for f in ok]);dB=np.asarray([m[f][f'B_{sc}']-W0[f][f'B_{sc}'] for f in ok]);return dA,dB
    for l,m in M.items():
        arm,ch=split_label(l);e={'chunk':ch,**{f'rr_{k}':TR.boot([m[f][f'rank_{k}']/16.0 for f in facts]) for k in ('branch','first','full')},**{f'top1_{k}':TR.boot([float(m[f][f'rank_{k}']==1) for f in facts]) for k in ('branch','first')}}
        dA,dB=dl(m,'first');e['first_dA']=TR.boot(dA);e['first_dA_minus_dB']=TR.boot(dA-dB);dA,dB=dl(m,'full');e['full_dA_minus_dB']=TR.boot(dA-dB);res['curve'].setdefault(arm,{})[str(ch)]=e
    arms=sorted(res['curve']);res['paired']={}
    if len(arms)==2:
        x,y=arms
        for ch in sorted({int(c) for c in res['curve'][x]},key=int):
            lx,ly=f'{x}@c{ch}',f'{y}@c{ch}'
            if lx in M and ly in M:res['paired'][str(ch)]={f'{y}_minus_{x}':{'rr_branch':TR.boot([(M[ly][f]['rank_branch']-M[lx][f]['rank_branch'])/16.0 for f in facts]),'rr_first':TR.boot([(M[ly][f]['rank_first']-M[lx][f]['rank_first'])/16.0 for f in facts]),'first_dA':TR.boot(dl(M[ly],'first')[0]-dl(M[lx],'first')[0])}}
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out)
def news(a):
    import e57_news_tokread as NT
    d=json.load(open(a.tokread));ov=json.load(open(EXP/'e57_news/e57_t1t2/e57_t1t2_overlap.json'));zero={it['id'] for it in ov['items'] if it['stream']['1']==0};meta=d['meta']['v1'];T=d['tokens']['v1'];res={'curve':{},'paired':{}}
    subs={'no_overlap100':np.asarray([m['id'] in zero for m in meta]),'ALL':np.ones(len(meta),bool)};F={l:[NT.frame_metrics(t,m,'all') for t,m in zip(tl,meta)] for l,tl in T.items()};arr=lambda l,k:np.asarray([f[k] for f in F[l]])
    for l in F:
        res['curve'][l]={ss:{k:NT.boot(arr(l,k)[mk]) for k in ('rr_branch','rr_first','rr_full','top1_branch','top1_first','gold_first','gold_full')} for ss,mk in subs.items()}
    cl=sorted({split_label(l)[1] for l in F if '@c' in l});arms=sorted({split_label(l)[0] for l in F if '@c' in l})
    if len(arms)==2:
        x,y=arms
        for ch in cl:
            lx,ly=f'{x}@c{ch}',f'{y}@c{ch}'
            res['paired'][str(ch)]={ss:{k:NT.boot((arr(ly,k)-arr(lx,k))[mk]) for k in ('rr_branch','rr_first','rr_full','top1_branch','gold_first')} for ss,mk in subs.items()}
    res['reference']={l:res['curve'][l] for l in ('W0','NATQ','ALLQ128','M3W8Q128') if l in res['curve']}
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out)
def selftest():
    assert split_label('M3W8N05@c63')==('M3W8N05',63);print('e57_curve_report selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind');ap.add_argument('--seed',type=int,default=64);ap.add_argument('--tag',default='synqc');ap.add_argument('--tokread');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    {'syn':syn,'news':news}[a.kind](a)
if __name__=='__main__':main()
