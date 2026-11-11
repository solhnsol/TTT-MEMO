"""E57 PRESERVATION CLAUSE (final text; registered before any result of job 1283 or of the revised DCLM job 1283d is opened; CPU; new file). Frozen 2026-11-11 (server clock), before both new runs were executed or opened. Written AFTER seeing NATS-W0 = -0.062 and NATMASK-NATS = +0.039 (registered criterion: FAIL) in the earlier run (job 1276, canary tail NLL, K0 = 8). The primary reference stays NATS as originally registered; W0 is added only as a linking condition. The earlier FAIL stands; the new layers apply only to the new runs; rules of earlier runs are not changed retroactively.
PRIOR RULE QUOTED: the 11/10 registration (core-claims thread relay, 2026-11-10 ~04:02Z) read 'preservation gate (separate from the family): general-text PPL within +0.01 nat, in-window read within -2 pp' without a confidence interval; the server analysis script e57_natmask_d64_analyze.py implemented the canary part as the POINT ESTIMATE of mean(NATMASK tail rise) - mean(NATS tail rise) <= 0.01. The DCLM gate of job 1274 was registered as 'point estimate <= +0.01 nat, CI upper bound shown (4 clusters, rough)'. So the earlier FAIL rules were point-estimate rules; the rule below (one-sided CI) applies only to the new runs.
REFERENCES AND GATES. D_N = NATMASK - NATS (primary), D_0 = NATMASK - W0 (linking condition); every gate component is D_N AND D_0 (an intersection test: the family-wise false-pass probability is <= .05; this is NOT a simultaneous interval). NATS - W0 is always reported. Pre-declared label: D_N FAIL and D_0 PASS -> 'the reference method's general-text gain is given up; no absolute forgetting relative to W0'; the verdict is FAIL.
JUDGEMENT RULE (identical for G1 and G2), margin delta = +0.01 nat/token (natural log; per token): PASS = one-sided 95% upper bound <= 0.01; FAIL = one-sided lower bound > 0.01; otherwise INCONCLUSIVE, which is NOT a pass. alpha = .05 one-sided. If a gate is INCONCLUSIVE it is extended ONCE (G1: 16 seeds instead of 8; G2: more streams from the not yet used documents 128..199, 9 streams of 8 documents, so 25 streams) with alpha split .025 / .025 (both stages use the one-sided .025 bound); the extension rule is declared here in advance; anything else is final 'not shown'.
G1 CANARY (seed-paired, 8 seeds 71..78 of job 1283): checkpoints = exposures 16, 32, 64, 128 (job option --save-apps 16,32,64,128; the canary is evaluated right AFTER the write of the chunk that ends the exposure, i.e. chunks 15, 31, 63, 127). tail NLL = the canary '_tail_nll' of Ladder.tail_nll: the name-scrubbed neutral tail text (pg1342, readouts.neutral_tail, 7 chunks of 1024 tokens, the same text for every seed and arm), evaluated with the FROZEN weights of the checkpoint, a fresh suffix state, the prefix from W0, natural-log NLL, TOKEN mean over the 7 x 1024 tokens of this single document (document mean = token mean, the last target dropped). Inference: paired differences per seed, t with df = 7; all four checkpoints must pass for both D_N and D_0 (intersection test, NOT a simultaneous interval); normality of seed differences cannot be verified with 8 values and is assumed. gen NLL (pg11, memorised text) is reported but not gated.
G2 DCLM (job 1283d revised, K0 = 0): 16 streams x 8 documents; stream i = DCLM validation documents 8i .. 8i+7 of data/dclm_val_subset/val_c0_200x8193.u32le.bin (documents in file order, the first 128 of 200; representativeness: the first 128 documents of one shard, order in the shard assumed not sorted by topic, NOT verified; the conclusion is conditional on this shard and on the fixed canary / M1 sets); each document contributes its first 8192 tokens (all documents have exactly 8193 tokens, so no short documents; 8 chunks of 1024 per document, 64 chunks per stream, the documents of a stream are concatenated without separators). Every stream STARTS from the W0 fast weights with a fresh suffix state (plain clipped SGD has no optimizer state, so nothing else to reset); within a stream the fast weights and the suffix state run on across document boundaries. prequential = every chunk is scored BEFORE its own update (with the weights after the previous chunk, full natural context). Stream score = the mean over its 64 chunks of the chunk token-mean NLL (equal chunk weights = token weights, since every chunk has the same number of scored tokens), nat/token (natural log). The independent unit is the STREAM (16). MAIN interval = one-sided t on the 16 paired stream differences (df = 15) AND the inversion of the exact sign-flip test (2^16 sign patterns); the label is the conservative combination: PASS only if both pass, FAIL only if both fail, otherwise INCONCLUSIVE. The stream bootstrap (percentile, B = 10000) is reported as auxiliary only. NATMASKC is not run in the gate job (exploratory runs are separate and non-gating). The earlier 4-stream K0 = 8 DCLM run (job 1274) is SUPERSEDED: its cluster count (4) is too small for a CI, it used K0 = 8 (the default mask start is now chunk 0) and one document grouping; it stays reported as legacy_k08 (NATMASK-NATS +0.0091, cluster CI upper bound +0.0104).
POWER DISCLOSURE (from the per-stream differences of the superseded 4-stream run, K0 = 8, SD of the per-stream NATMASK-NATS difference 0.0019 nat, scaled by sqrt(2) for streams half as long: 0.0027): expected half-width of the one-sided 95% bound with 16 streams about 1.753 x 0.0027 / 4 = 0.0012 nat; if the true D_N were the earlier +0.0091 the upper bound would sit near 0.0103 and P(PASS) would be only about one third; D_0 had mean -0.042 and would pass with certainty. The margin stays +0.01.
NON-GATE REPORTS: M0 = NATMASK vs NATS on the final fact accuracy and on the canary: mean accuracy difference <= 0 while the G1 D_N is FAIL is read as 'the domain-adaptation excuse is refuted'; this is an observed dominance only, not a statistically proven dominance. M1 = W0-relative forgetting on two fixed domains: D1 = the canary tail pg1342 (as G1), D2 = SQuAD v1.1 dev contexts (spec and ids sha256 in e57_m1_domains.py, D2_IDS_SHA256 ea1bf116cbe4a30f2d2cd019828104dbfbf2b1afdc28cf4f5c27628564dc8a2e), scored like the canary tail; non-gate. M2 = per-checkpoint matching-knowledge canary (non-gate; definition pending, not run). M3 non-gate (not specified).
WORDING: 'improvement without harm' is NOT used; the allowed sentence is 'no general-text harm of 0.01 nat/token was detected against NATS and against W0' (only when both gates PASS), conditional on the fixed canary and M1 sets, in nat/token (token denominator = the scored tokens: 7 x 1024 - 1 for the canary, 64 x 1023 or 64 x 1024 chunk tokens per stream for DCLM).
  python3 e57_preservation_analyze.py --g1-dir O1283 --seeds 71,...,78 [--alpha 0.05] | --g2-dir ODCLM [--alpha 0.05] [--json J] | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
from scipy import stats
DELTA=0.01;CKPT=[16,32,64,128]
def tbounds(d,alpha):
    d=np.asarray(d,float);n=len(d);m=d.mean();se=d.std(ddof=1)/np.sqrt(n);t=stats.t.ppf(1-alpha,n-1);return float(m-t*se),float(m+t*se)
def _sf_matrix(n):return np.array([[1 if (i>>j)&1 else -1 for j in range(n)] for i in range(2**n)],float)
def _p_upper_H(d,u,S,m0=None):
    """one-sided exact sign-flip p-value of H0: mu >= u against mu < u (small when the mean is far below u)"""
    x=np.asarray(d,float)-u;obs=x.mean();return float(((S*x[None]).mean(1)<=obs+1e-15).mean())
def signflip_bounds(d,alpha):
    """one-sided (1-alpha) bounds by inversion of the exact sign-flip test: upper = smallest u with p(H0: mu >= u) < alpha; lower = largest l with p(H0: mu <= l) < alpha"""
    d=np.asarray(d,float);n=len(d);S=_sf_matrix(n);sd=d.std(ddof=1)+1e-12;m=d.mean()
    lo,hi=m,m+20*sd
    for _ in range(60):
        mid=(lo+hi)/2
        if _p_upper_H(d,mid,S)<alpha:hi=mid
        else:lo=mid
    up=hi;lo2,hi2=m-20*sd,m
    for _ in range(60):
        mid=(lo2+hi2)/2
        if _p_upper_H(-d,-mid,S)<alpha:lo2=mid
        else:hi2=mid
    return float(lo2),float(up)
def label(lower,upper,delta=DELTA):
    return 'PASS' if upper<=delta else ('FAIL' if lower>delta else 'INCONCLUSIVE')
def combine(l1,l2):
    return l1 if l1==l2 else 'INCONCLUSIVE'
def g1(D,seeds,alpha=0.05,tag='k0c16'):
    D=pathlib.Path(D);res={'alpha':alpha,'n_seeds':len(seeds),'checkpoints':{}};labs=[]
    for c in CKPT:
        v={'NATS':[],'NATMASK':[],'W0':[]}
        for s in seeds:
            for a in ('NATS','NATMASK'):
                r=json.load(open(D/f'E55_{tag}_{a}_s{s}.saved.json'));v[a].append(r['reads'][str(c-1)]['canary']['_tail_nll']);w0=r['canary_w0']['_tail_nll']
            v['W0'].append(w0)
        v={k:np.array(x) for k,x in v.items()};dN=v['NATMASK']-v['NATS'];d0=v['NATMASK']-v['W0'];dS=v['NATS']-v['W0'];e={}
        for nm,d in (('D_N',dN),('D_0',d0)):
            lo,up=tbounds(d,alpha);e[nm]=dict(mean=float(d.mean()),lower=lo,upper=up,label=label(lo,up));labs.append(e[nm]['label'])
        e['NATS_minus_W0_mean']=float(dS.mean());res['checkpoints'][str(c)]=e
    res['gate_label']='PASS' if all(l=='PASS' for l in labs) else ('FAIL' if any(l=='FAIL' for l in labs) else 'INCONCLUSIVE')
    res['note']='intersection test over 4 checkpoints x {D_N, D_0}; not a simultaneous interval; t df = n-1; normality unverifiable';return res
def stream_scores(D,arm,n):
    return np.array([float(np.mean(json.load(open(pathlib.Path(D)/f'E57_dclmttt_{arm}_st{i}.json'))['nll_pre'])) for i in range(n)])
def g2(D,n_streams=16,alpha=0.05,B=10000):
    sc={a:stream_scores(D,a,n_streams) for a in ('W0','NATS','NATMASK')};res={'alpha':alpha,'n_streams':n_streams,'NATS_minus_W0_mean':float((sc['NATS']-sc['W0']).mean())};labs={}
    for nm,d in (('D_N',sc['NATMASK']-sc['NATS']),('D_0',sc['NATMASK']-sc['W0'])):
        tl,tu=tbounds(d,alpha);sl,su=signflip_bounds(d,alpha);rng=np.random.default_rng(20261111);bs=np.array([d[rng.integers(0,n_streams,n_streams)].mean() for _ in range(B)])
        lt,ls=label(tl,tu),label(sl,su);lab=combine(lt,ls);labs[nm]=lab
        res[nm]=dict(mean=float(d.mean()),t=dict(lower=tl,upper=tu,label=lt),signflip_inversion=dict(lower=sl,upper=su,label=ls),bootstrap_aux=dict(ci95=[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))]),label=lab,per_stream=[float(x) for x in d])
    res['gate_label']='PASS' if all(l=='PASS' for l in labs.values()) else ('FAIL' if 'FAIL' in labs.values() else 'INCONCLUSIVE')
    if labs['D_N']=='FAIL' and labs['D_0']=='PASS':res['pre_declared_label']='the reference method general-text gain is given up; no absolute forgetting relative to W0; verdict FAIL';res['gate_label']='FAIL'
    return res
def overall(a,b):
    l=[a['gate_label'],b['gate_label']];return 'PASS' if all(x=='PASS' for x in l) else ('FAIL' if 'FAIL' in l else 'INCONCLUSIVE')
def selftest():
    import tempfile;rng=np.random.default_rng(0)
    assert label(-0.01,0.005)=='PASS' and label(0.02,0.03)=='FAIL' and label(0.0,0.02)=='INCONCLUSIVE' and combine('PASS','INCONCLUSIVE')=='INCONCLUSIVE'
    d=np.array([0.0091]*8)+rng.normal(0,0.002,8);tl,tu=tbounds(d,0.05);sl,su=signflip_bounds(d,0.05);assert abs(tu-su)<0.003 and tu>d.mean() and su>d.mean() and sl<d.mean()
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);seeds=list(range(71,79))
        for s in seeds:
            for a,off in (('NATS',-0.06),('NATMASK',-0.04),):
                r=dict(canary_w0={'_tail_nll':2.5},reads={str(c-1):dict(canary={'_tail_nll':2.5+off+rng.normal(0,0.002)}) for c in CKPT});json.dump(r,open(td/f'E55_k0c16_{a}_s{s}.saved.json','w'))
        r1=g1(td,seeds);assert r1['gate_label']=='FAIL' and r1['checkpoints']['16']['D_N']['label']=='FAIL' and r1['checkpoints']['16']['D_0']['label']=='PASS'     # D_N = +0.02 > 0.01, D_0 = -0.04
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td)
        for i in range(16):
            for a,off in (('W0',0.0),('NATS',-0.05),('NATMASK',-0.04)):json.dump(dict(nll_pre=[2.5+off+rng.normal(0,0.01)]*64),open(td/f'E57_dclmttt_{a}_st{i}.json','w'))
        r2=g2(td);assert r2['D_0']['label']=='PASS' and r2['D_N']['label'] in ('PASS','INCONCLUSIVE') and r2['gate_label'] in ('PASS','INCONCLUSIVE')
    assert overall({'gate_label':'PASS'},{'gate_label':'PASS'})=='PASS' and overall({'gate_label':'PASS'},{'gate_label':'FAIL'})=='FAIL'
    print('e57_preservation_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--g1-dir');ap.add_argument('--g2-dir');ap.add_argument('--seeds');ap.add_argument('--alpha',type=float,default=0.05);ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        out={}
        if a.g1_dir:out['G1']=g1(a.g1_dir,[int(x) for x in a.seeds.split(',')],a.alpha)
        if a.g2_dir:out['G2']=g2(a.g2_dir,16,a.alpha)
        if 'G1' in out and 'G2' in out:out['overall']=overall(out['G1'],out['G2'])
        print(json.dumps(out,indent=1,default=float))
        if a.json:json.dump(out,open(a.json,'w'),indent=1,default=float)
