"""E57 BAND-1024 CONFIRMATION, CORRECTED RULE (job 1297c; new seeds; supersedes the activity gate of e57_band1024_confirm_analyze.py, sha a912fcc2..., frozen in tag freeze-1297-v1).
CORRECTION (the ONLY change to the registered analysis, Hans 11/16 '정정 규칙으로 재확증'): the gate "the mask arm differs from NATS at chunk 0" is applied for NATMASK1024 at CHUNK 1. Reason: the first chunk has 1024 tokens, so no key lies at distance >= 1024 and the band-1024 mask cannot change anything at chunk 0; chunk 0 of NATMASK1024 is structurally identical to NATS (job 1297, seed 141: update_norm 0.9999998770654126 and ce 3.9575932025909424 in both arms; chunk 1: 0.754 vs 0.779, 2.72 vs 2.93). The gate of the frozen analyzer therefore excluded all 8 seeds of job 1297 (n_effective = 0, result e57_band1024_conf_result.json); the same analysis with the gate at chunk 1 (post hoc, e57_band1024_confirm_analyze_dev1.py, result e57_band1024_conf_result_dev1.json) gave P1, P2 and P3 passed. The NATMASK256 gate stays at chunk 0. P1, P2, P3, the thresholds, the exclusion and rerun rules and the wording are unchanged; the seeds below are NEW (151..158). Disclosure: this correction was made after the 1297 results were seen.
--- text of the frozen registration follows (seeds 141..148 there; here 151..158) ---
E57 BAND-1024 CONFIRMATION analysis, synthetic 16-fact worlds (cell v3_16_1), job 1297 (CPU; written and its sha256 frozen BEFORE any output of the new seeds exists; run_type = confirmation, K0 = 0, never pooled with other K0 or other seeds).
DISCLOSURE (item 5 of the request): this registration was designed AFTER the results of job 1293 (band distance, seeds 71..78: G_1024 = 2.44 nat in the mean, the DCLM check of band 1024 of job 1294: D_N upper bound .0032) and job 1296 (block stream) were seen. The thresholds below are those of the registration of job 1283 (K0 = 0 confirmation), NOT tuned to the 1293 numbers; the band 1024 itself was chosen because of 1293 / 1296. Exploratory evidence of 1293 / 1296 is design basis only and is not pooled.
DESIGN: seeds 151..158 in job 1297c (8; not used by any earlier job; 141..148 = job 1297), arms NATS, NATMASK (ALLQ256) and NATMASK (ALLQ1024), K0 = 0, v3_16_1 (16 facts, 128 exposures), reads at exposures 16/32/64/128 as in job 1283. Records: tag bc (NATS, NATMASK = band 256) and tag bc1024 (NATMASK = band 1024).
MARGIN (convention of job 1283): per fact log p(gold) - max log p(wrong) over the 16 candidates of equal token length, summed over the answer span, mean over the 16 facts of a seed; logsumexp margin (e57_1280_margin.margin_primary) is a descriptive secondary. G_W = margin(NATMASK_W) - margin(NATS), out-of-window read at exposure 128; H_W = margin(NATS) - margin(NATMASK_W), in-window read (in_g0) at exposure 128 (positive = harm); N_W = 16 (G_W - H_W) nat per sequence; unit = seed.
PRIMARY HYPOTHESES (all three must pass: intersection-union test, no multiplicity correction needed; confirmation = P1 AND P2 AND P3):
 P1 (binding effect): one-sided 95% t lower bound of mean G_1024 > 0 AND mean G_1024 >= 0.5 nat (SESOI of job 1283) AND the exact sign-flip test rejects H0: mu <= 0.5 (shifted, one-sided, p < .05) AND the exact sign-flip test rejects H0: mu <= 0 (one-sided, p < .05). (Reading of 'both exact tests': the two exact tests are the shifted test against 0.5 and the unshifted test against 0; with n = 8 the minimum p is 1/256.)
 P2 (in-window harm): one-sided 95% t UPPER bound of mean H_1024 <= 0.25 nat.
 P3 (general text): DCLM preservation, D_N(1024) = NLL(NATMASK_1024) - NLL(NATS) per stream (nat/token), one-sided 95% t UPPER bound over the streams <= 0.01. Data: 9 fresh DCLM validation streams 16..24 (docs 128..199 of the validation subset, never used by 1283d / 1294), arms W0, NATS, NATMASK_256, NATMASK_1024 from the job. The 16-stream result of job 1294 (docs 0..127) is not part of the test.
SECONDARY (descriptive, no decision): H_256 - H_1024 (paired by seed, 95% t interval, exact sign-flip two-sided), N_1024 - N_256 likewise, G_256, G_1024, accuracy differences, logsumexp margins.
GATES AND EXCLUSION (same rule as job 1283): a seed whose run diverges, yields a non-finite value, whose saved.json does not carry K0 = 0, or whose mask arm does not differ from NATS at chunk 0 is rerun (all three arms); a seed that fails again is EXCLUDED (n = 7, min exact p = 1/128); two or more excluded seeds: the confirmatory status is lost (labelled exploratory). A seed is never excluded because of its outcome. The job aborts (exit 13) if its NATS / NATMASK(256) run on seed 71 is not bit-identical with the job 1283 records (environment check, as in job 1285b).
WORDING: confirmed only if P1, P2 and P3 pass; otherwise the result is reported per hypothesis (P1 failed: 'inconclusive' / 'detected but below the SESOI'; P2 or P3 failed: 'preservation not established'). No claim 'no effect' from a failure. Status: confirmation, K0 = 0 only.
  .venv/bin/python scripts/exp07_write_ladder/e57_band1024_confirm_analyze.py --dir D --seeds 141,...,148 --dclm-dir DC --json J | --selftest"""
import sys,json,argparse,pathlib,tempfile
import numpy as np
from scipy import stats
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_k0conf16_analyze as K
import e57_1280_margin as MG
DELTA=0.5
def one_sided_upper(v):
    v=np.asarray(v,float);n=len(v);return float(v.mean()+stats.t.ppf(0.95,n-1)*v.std(ddof=1)/np.sqrt(n))
def paths(D,s):
    D=pathlib.Path(D);return dict(S=D/f'E55_bc_NATS_s{s}.read.json',M256=D/f'E55_bc_NATMASK_s{s}.read.json',M1024=D/f'E55_bc1024_NATMASK_s{s}.read.json',sv=[D/f'E55_bc_NATS_s{s}.saved.json',D/f'E55_bc_NATMASK_s{s}.saved.json',D/f'E55_bc1024_NATMASK_s{s}.saved.json'])
def lse(path,app,key):
    d=json.load(open(path));m=d['meta'];g=np.array(m['gold_idx']);fact=np.array([k=='fact' for k in m['kind']]);L=np.array(d['read'][app][key],float);fact=fact[:len(L)];g=g[:len(L)];return float(np.mean(MG.margin_primary(L[fact],g[fact])))
def analyze(D,seeds,dclm_dir=None,dclm_json=None):
    out={'run_type':'confirmation','K0':0,'seeds_requested':list(seeds)};use=[];bad={}
    for s in seeds:
        p=paths(D,s);ok=True;why=[]
        try:
            sv=[json.load(open(x)) for x in p['sv']]
            if not all(x.get('K0')==0 for x in sv):ok=False;why.append('K0 != 0')
            l0=[x['log'][0] for x in sv]
            if not (l0[1]['update_norm']!=l0[0]['update_norm'] or l0[1]['ce']!=l0[0]['ce']):ok=False;why.append('NATMASK256 inactive at chunk 0')
            if not (sv[2]['log'][1]['update_norm']!=sv[0]['log'][1]['update_norm'] or sv[2]['log'][1]['ce']!=sv[0]['log'][1]['ce']):ok=False;why.append('NATMASK1024 inactive at chunk 1')
            if not all(np.isfinite(K.fact_margins(p[k],'128')[0]).all() for k in ('S','M256','M1024')):ok=False;why.append('non-finite margin')
        except Exception as e:ok=False;why.append(repr(e))
        (use.append(s) if ok else bad.__setitem__(s,why))
    out['gates']=dict(excluded_seeds=bad,n_effective=len(use),confirmatory_status=bool(len(bad)<=1 and len(use)>=7))
    if len(use)<5:out['abort']='too few seeds';return out
    mk=lambda k,app='128',key='out':np.array([K.fact_margins(paths(D,s)[k],app,key)[0].mean() for s in use])
    ac=lambda k:np.array([K.fact_margins(paths(D,s)[k],'128')[1].mean() for s in use])
    Ms,M2,M4=mk('S'),mk('M256'),mk('M1024');Is,I2,I4=mk('S','128','in_g0'),mk('M256','128','in_g0'),mk('M1024','128','in_g0')
    G2,G4=M2-Ms,M4-Ms;H2,H4=Is-I2,Is-I4;N2,N4=16*(G2-H2),16*(G4-H4)
    st=K.t1(G4);pa=K.signflip(G4,DELTA)[0];pb=K.signflip(G4)[0]
    P1=bool(st['lower_onesided']>0 and st['mean']>=DELTA and pa<0.05 and pb<0.05)
    out['P1']=dict(contrast='G_1024 = margin(NATMASK_1024) - margin(NATS), out-of-window, exposure 128',per_seed=[float(x) for x in G4],stats=st,p_signflip_H0_mu_le_0_5=pa,p_signflip_H0_mu_le_0=pb,passed=P1)
    h=K.t1(H4);P2=bool(h['upper_onesided']<=0.25);out['P2']=dict(contrast='H_1024 = margin(NATS) - margin(NATMASK_1024), in-window, exposure 128',per_seed=[float(x) for x in H4],stats=h,passed=P2)
    P3=None;out['P3']=dict(passed=None,note='DCLM not given')
    if dclm_dir or dclm_json:
        dj=json.load(open(dclm_json)) if dclm_json else None
        if dj is not None:
            v=np.array(dj['G2']['D_N']['per_stream'],float);up=one_sided_upper(v);P3=bool(up<=0.01);out['P3']=dict(D_N_mean=float(v.mean()),n_streams=len(v),upper_onesided_95=up,passed=P3,per_stream=[float(x) for x in v])
    out['confirmed']=bool(P1 and P2 and P3) if P3 is not None else None
    out['wording']=('confirmed (P1, P2 and P3)' if out['confirmed'] else ('P3 not evaluated' if P3 is None else 'not confirmed: '+', '.join(n for n,p in (('P1',P1),('P2',P2),('P3',P3)) if not p)+' failed'))
    d=H2-H4;n=N4-N2
    out['secondary']=dict(G_256=K.t1(G2),H_256=K.t1(H2),H_256_minus_H_1024=dict(mean=float(d.mean()),ci95_t=K.t2(d),p_signflip_two_sided=K.signflip(d)[1],per_seed=[float(x) for x in d]),N_256=float(N2.mean()),N_1024=float(N4.mean()),N_1024_minus_N_256=dict(mean=float(n.mean()),ci95_t=K.t2(n),p_signflip_two_sided=K.signflip(n)[1],per_seed=[float(x) for x in n]),
        accuracy_diff_1024_minus_NATS=dict(mean=float((ac('M1024')-ac('S')).mean()),ci95_t=K.t2(ac('M1024')-ac('S'))),
        logsumexp_G_1024=float(np.mean([lse(paths(D,s)['M1024'],'128','out')-lse(paths(D,s)['S'],'128','out') for s in use])),margin128=dict(NATS=float(Ms.mean()),NATMASK256=float(M2.mean()),NATMASK1024=float(M4.mean())))
    return out
def selftest():
    rng=np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        seeds=[1,2,3,4,5,6,7,8]
        for s in seeds:
            for tag,arm,eff,hh in (('bc','NATS',0.0,0.0),('bc','NATMASK',1.0,0.2),('bc1024','NATMASK',2.0,0.1)):
                g=rng.integers(0,16,20);L=rng.normal(0,1,(20,16));out=L.copy();inw=L.copy()
                for i in range(16):out[i,g[i]]+=2+eff;inw[i,g[i]]+=3-hh
                json.dump(dict(meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=g.tolist()),read={'128':{'out':out.tolist(),'in_g0':inw.tolist()}}),open(f'{td}/E55_{tag}_{arm}_s{s}.read.json','w'))
                json.dump(dict(K0=0,log=[dict(update_norm=1.0+(eff if tag=='bc' else 0.0),ce=2.0+(eff if tag=='bc' else 0.0)),dict(update_norm=1.0+eff,ce=2.0+eff)]),open(f'{td}/E55_{tag}_{arm}_s{s}.saved.json','w'))      # corrected rule: NATMASK1024 equals NATS at chunk 0 (structural) and differs at chunk 1; NATMASK256 differs at chunk 0
        dj=f'{td}/dclm.json';json.dump({'G2':{'D_N':{'per_stream':[0.003+0.001*rng.normal() for _ in range(9)]}}},open(dj,'w'))
        o=analyze(td,seeds,dclm_json=dj);assert o['P1']['passed'] and o['P2']['passed'] and o['P3']['passed'] and o['confirmed'] and o['gates']['confirmatory_status'],o
        assert abs(o['P1']['stats']['mean']-2.0)<0.6 and abs(o['secondary']['H_256_minus_H_1024']['mean']-0.1)<0.6
        json.dump({'G2':{'D_N':{'per_stream':[0.03]*9+[0.04]}}},open(dj,'w'));o=analyze(td,seeds,dclm_json=dj);assert o['P3']['passed'] is False and o['confirmed'] is False
    print('e57_band1024_confirm_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--dclm-dir');ap.add_argument('--dclm-json');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        o=analyze(a.dir,[int(x) for x in a.seeds.split(',')],a.dclm_dir,a.dclm_json);s=json.dumps(o,indent=1,default=float);print(s)
        if a.json:open(a.json,'w').write(s)
