"""E57 job 1280 / amendment A1 v5, section 4.5 and 4.11 (a): CPU-side measurement script for the t3 positive control (sensitivity of the natural path to the token substitution of S_i). New file, GPU-free: it only aggregates per-chunk NLL arrays that the GPU measurement run will write.
Input per measurement seed (JSON written by the runner): {"seed": int, "chunks": [{"i": int, "s_i_empty": bool, "nll_orig": [floats over F_i'], "nll_sub": [floats over F_i'], "lp_orig": [floats], "lp_sub": [floats]}, ...]} where F_i' = F_i minus E_i (positions of the original and the substituted natural path, same order); lp_* are the natural-path log-probabilities at the same positions (for the 'detectable substitution response' label).
Definitions (A1 v5 4.5): v_i = mean |nll_sub - nll_orig| over F_i' of the chunk (reported); seed aggregate = sum over all non-skipped chunks and tokens of |nll_sub - nll_orig| divided by the number of tokens (token weighted; a chunk with empty F_i' adds no token; chunks with empty S_i are skipped and counted); denominator 0 -> aggregate None (UNDETERMINED); chunk labels 'below the chunk-average threshold' (v_i < 0.001) and 'detectable substitution response' (max |lp_sub - lp_orig| > 1e-4) are reported separately and decide nothing. Margin = smallest seed aggregate / 0.001. Rule fixed before the measurement: keep option (a) if margin >= 10, else option (b). P(any) = 1 - (1 - p)^30 for p in {0.001, 0.005, 0.01, 0.02, 0.05}.
  python3 e57_1280_t3_sensitivity.py --json seed91.json seed92.json ... [--out report.json] | --selftest"""
import sys,json,argparse
import numpy as np
THRESH=0.001;LP_TOL=1e-4;ABS_MIN=0.01;REQUIRED_SEEDS=(91,92,93,94);P_GRID=(0.001,0.005,0.01,0.02,0.05)   # v6: absolute, mechanical rule: option (a) is kept only if the smallest seed aggregate of the four required instrumentation seeds is >= ABS_MIN = 0.01 nat/token; ABS_MIN may be raised by an amendment before the measurement, never lowered
def seed_report(d):
    num=0.0;den=0;rows=[];skipped=0;empty_f=0
    for c in d['chunks']:
        if c.get('s_i_empty'):skipped+=1;continue
        o=np.asarray(c['nll_orig'],float);s=np.asarray(c['nll_sub'],float);assert o.shape==s.shape and np.isfinite(o).all() and np.isfinite(s).all(),'finite arrays of equal length required'
        if len(o)==0:empty_f+=1;rows.append(dict(i=c['i'],tokens=0,v=None));continue
        diff=np.abs(s-o);num+=float(diff.sum());den+=int(len(o));lo=np.asarray(c['lp_orig'],float);ls=np.asarray(c['lp_sub'],float)
        rows.append(dict(i=c['i'],tokens=int(len(o)),v=float(diff.mean()),below_chunk_threshold=bool(diff.mean()<THRESH),detectable_response=bool(np.abs(ls-lo).max()>LP_TOL),bit_identical=bool(np.array_equal(lo,ls))))
    agg=(num/den) if den>0 else None
    return dict(seed=d['seed'],aggregate=agg,tokens=den,skipped_empty_S_i=skipped,empty_F_i_chunks=empty_f,chunks=rows,sensitive=(agg is not None and agg>=THRESH))
def report(seed_files):
    reps=[seed_report(d) for d in seed_files];aggs=[r['aggregate'] for r in reps]
    for r,d in zip(reps,seed_files):r.update(leak=bool(d.get('leak',False)),fail_4_2=bool(d.get('fail_4_2',False)),complete=bool(d.get('complete',True)))
    have=sorted(r['seed'] for r in reps);missing=[s for s in REQUIRED_SEEDS if s not in have]
    ok=lambda a:a is not None and np.isfinite(a)
    mn=float(min(aggs)) if aggs and all(ok(a) for a in aggs) else None;margin=None if mn is None else mn/THRESH
    # decision order (mechanical, decided by this script's output; the principal investigator only records it): LEAK or 4.2 failure -> stop (4.8); incomplete or missing required seed -> rerun the seed (same seed, technical); an undefined or non-finite aggregate -> option (b); otherwise min aggregate >= ABS_MIN keeps option (a), else option (b)
    if any(r['leak'] or r['fail_4_2'] for r in reps):dec='STOP: LEAK or failed 4.2 item during the measurement, whole-study stop of 4.8 (no confirmatory run)'
    elif missing or any(not r['complete'] for r in reps):dec='INCOMPLETE: rerun the incomplete or missing measurement seed(s) (same seed); no decision'
    elif mn is None:dec='option (b): signed code amendment (A1b) before the first confirmatory seed (an aggregate is undefined, zero denominator or non-finite)'
    elif mn>=ABS_MIN:dec='keep option (a) with validity-first staging and W_t snapshot retention'
    else:dec='option (b): signed code amendment (A1b) before the first confirmatory seed'
    return dict(seeds=reps,required_seeds=list(REQUIRED_SEEDS),missing_seeds=missing,min_aggregate=mn,margin_over_threshold=margin,abs_min=ABS_MIN,decision=dec,P_any={str(p):float(1-(1-p)**30) for p in P_GRID},rule='LEAK/4.2 failure -> stop; incomplete -> rerun; undefined -> option (b); min aggregate >= 0.01 nat/token keeps option (a), else option (b) (fixed before the measurement)')
def selftest():
    r=np.random.default_rng(0)
    def mk(seed,eff,n_chunks=6,ntok=200,sempty=0,fempty=0):
        ch=[]
        for i in range(n_chunks):
            e=(i<sempty);f=(sempty<=i<sempty+fempty);n=0 if f else ntok;o=2+r.normal(0,.1,n);s=o+eff*r.choice([-1,1],n);lo=-o;ls=-s
            ch.append(dict(i=8+i,s_i_empty=e,nll_orig=list(o),nll_sub=list(s),lp_orig=list(lo),lp_sub=list(ls)))
        return dict(seed=seed,chunks=ch)
    rep=report([mk(91,0.05),mk(92,0.03),mk(93,0.2),mk(94,0.04)]);assert abs(rep['min_aggregate']-0.03)<1e-9 and rep['margin_over_threshold']>=10 and rep['decision'].startswith('keep option (a)')
    rep2=report([mk(91,0.05),mk(92,0.004),mk(93,0.05),mk(94,0.05)]);assert abs(rep2['margin_over_threshold']-4.0)<1e-9 and rep2['decision'].startswith('option (b)')
    assert report([mk(91,0.05),mk(92,0.0099),mk(93,0.05),mk(94,0.05)])['decision'].startswith('option (b)') and report([mk(91,0.05),mk(92,0.0101),mk(93,0.05),mk(94,0.05)])['decision'].startswith('keep option (a)')   # absolute threshold 0.01
    assert report([mk(91,0.05),mk(92,0.05)])['decision'].startswith('INCOMPLETE')                                   # required seeds 91..94 missing
    lk=mk(93,0.05);lk['leak']=True;assert report([mk(91,0.05),mk(92,0.05),lk,mk(94,0.05)])['decision'].startswith('STOP')
    f2=mk(94,0.05);f2['fail_4_2']=True;assert report([mk(91,0.05),mk(92,0.05),mk(93,0.05),f2])['decision'].startswith('STOP')
    inc=mk(94,0.05);inc['complete']=False;assert report([mk(91,0.05),mk(92,0.05),mk(93,0.05),inc])['decision'].startswith('INCOMPLETE')
    nanr=mk(94,0.05);nanr['chunks'][0]['nll_sub']=[float('nan')]*200
    try:report([mk(91,0.05),mk(92,0.05),mk(93,0.05),nanr]);raise SystemExit('NaN accepted')
    except AssertionError:pass                                                                                       # non-finite input arrays are refused
    z=mk(95,0.0);rz=seed_report(z);assert rz['aggregate']==0.0 and not rz['sensitive'] and all(c['bit_identical'] for c in rz['chunks']) and all(c['below_chunk_threshold'] for c in rz['chunks'])      # bit identical chunks stay in the aggregate with value 0
    e=mk(96,0.05,n_chunks=4,sempty=1,fempty=1);re=seed_report(e);assert re['skipped_empty_S_i']==1 and re['empty_F_i_chunks']==1 and re['tokens']==2*200 and abs(re['aggregate']-0.05)<1e-9        # empty S_i skipped, empty F_i' adds no token
    allempty=dict(seed=97,chunks=[dict(i=8,s_i_empty=False,nll_orig=[],nll_sub=[],lp_orig=[],lp_sub=[])]);ra=seed_report(allempty);assert ra['aggregate'] is None and not ra['sensitive'] and report([mk(91,0.05),mk(92,0.05),mk(93,0.05),dict(allempty,seed=94)])['decision'].startswith('option (b)')
    assert abs(rep['P_any']['0.01']-0.26029)<1e-4 and abs(rep['P_any']['0.05']-0.78539)<1e-4
    print('e57_1280_t3_sensitivity selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json',nargs='*');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.json:
        o=report([json.load(open(f)) for f in a.json]);print(json.dumps(o,indent=1))
        if a.out:json.dump(o,open(a.out,'w'),indent=1)
