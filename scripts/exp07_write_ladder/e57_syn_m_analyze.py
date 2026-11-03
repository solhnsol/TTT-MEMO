"""E57 synthetic world: analysis of the M2h / M3 arms against ALLQ256 (CPU only; new file; thread 'ALLQ보다 나은 가림 방법 찾기'). Inputs: the per-token end-state reads of the new arms (e57_syn_tokread.py --arms M2HN05,... -> e57_syn_tokread_s64.json in the new directory) and of the old arms (NATQ, ALLQ256 ...) from the job-676 file. Per fact (the facts of the world, kind 'fact'): first-token and full-sum A / B (the other fact rows that have the place as a candidate), dA, dB, d(A-B) against W0, branch / first-token / full-sum rank ratio and top-1 (e57_syn_tokread.metrics). Per new arm: mean over facts with the bootstrap 95% CI over facts; paired differences to ALLQ256 (d(dA), d(d(A-B)) for the first token and the full sum, rank-ratio difference, top-1 difference) and the ratio d(d(A-B))/d(dA) against NATQ as in the job-676 analysis, plus the same for ALLQ256 itself.
  python3 e57_syn_m_analyze.py --old DIR_OLD_JSON --new DIR_NEW_JSON --out OUT.json | --selftest"""
import sys,json,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_syn_tokread as TR
def analyze(old,new,out,seed=64,refs=('ALLQ256','NATQ')):
    o=json.load(open(old));n=json.load(open(new));assert o['tokids']==n['tokids'],'token ids of the two reads differ';meta=json.load(open(next(iter({64:TR.arms_s64,61:TR.arms_s61}[seed]().values()))['read_json']))['meta'];tokids=o['tokids']
    W0=TR.metrics(o['W0'],tokids,meta);W0n=TR.metrics(n['W0'],tokids,meta);facts=sorted(W0)
    for f in facts:assert abs(W0[f]['A_full']-W0n[f]['A_full'])<1e-3,'W0 of the two reads differs'
    M={ar:TR.metrics(t,tokids,meta) for ar,t in o['arms'].items()};M.update({ar:TR.metrics(t,tokids,meta) for ar,t in n['arms'].items()})
    def dl(m,sc):
        ok=[f for f in facts if m[f][f'B_{sc}'] is not None and W0[f][f'B_{sc}'] is not None];dA=np.asarray([m[f][f'A_{sc}']-W0[f][f'A_{sc}'] for f in ok]);dB=np.asarray([m[f][f'B_{sc}']-W0[f][f'B_{sc}'] for f in ok]);return ok,dA,dB
    res={'new_arms':list(n['arms']),'reference_arms':list(refs),'reproduction_new_reads':n.get('reproduction'),'arms':{}}
    for ar,m in M.items():
        if ar not in n['arms'] and ar not in refs:continue
        e={'mean_rank_ratio':{k:TR.boot([m[f][f'rank_{k}']/16.0 for f in facts]) for k in ('branch','first','full')},'top1':{k:TR.boot([float(m[f][f'rank_{k}']==1) for f in facts]) for k in ('branch','first','full')}}
        for sc in ('first','full'):ok,dA,dB=dl(m,sc);e[f'{sc}_dA']=TR.boot(dA);e[f'{sc}_dB']=TR.boot(dB);e[f'{sc}_dA_minus_dB']=TR.boot(dA-dB)
        for ref in refs:
            if ar==ref:continue
            c={}
            for sc in ('first','full'):
                ok,dA,dB=dl(m,sc);okr,dAr,dBr=dl(M[ref],sc);assert ok==okr;ddA=dA-dAr;ddAB=(dA-dB)-(dAr-dBr);c[sc]=dict(d_dA=TR.boot(ddA),d_dAB=TR.boot(ddAB),ratio_dAB_over_dA=TR.ratio(ddAB,ddA) if abs(ddA.mean())>1e-9 else None)
            c['rank_ratio_diff']={k:TR.boot([(m[f][f'rank_{k}']-M[ref][f][f'rank_{k}'])/16.0 for f in facts]) for k in ('branch','first','full')};c['top1_diff']={k:TR.boot([float(m[f][f'rank_{k}']==1)-float(M[ref][f][f'rank_{k}']==1) for f in facts]) for k in ('branch','first','full')};e[f'vs_{ref}']=c
        res['arms'][ar]=e
    pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(out).write_text(json.dumps(res,indent=1,default=float));return res
def selftest():
    TR.selftest();print('e57_syn_m_analyze selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k,d=None:sys.argv[sys.argv.index(k)+1] if k in sys.argv else d;analyze(g('--old'),g('--new'),g('--out'),int(g('--seed','64')),tuple(g('--refs','ALLQ256,NATQ').split(',')))
if __name__=='__main__':main()
