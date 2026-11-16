"""E57 job 1302o QUEUE GATE (CPU; new file; rule fixed by Hans 11/16 BEFORE the 1302 result, recorded in the claims-map): the Oracle job 1302o (NATORACLE / NATPLACEBO on the seeds 71..78 of v3_16_1) runs only if, in the result of job 1302 (e57_1280_cache_result.json, analyzer e57_1280c_cache_analyze.py sha 3ec68756..., primary = logsumexp margin, 8 seeds 101..108), BOTH
  primary.C1_A_minus_NATS.ci95_two_sided[0] > 0     (cached oracle minus NATS: lower end of the two-sided 95% t interval)
  primary.C3_A_minus_AP.ci95_two_sided[0]   > 0     (cached oracle minus cached placebo, paired by seed)
hold (strict inequalities). Otherwise 1302o is HELD: it does not run, nothing is changed, the verdict file records the values. A missing / unreadable result, a missing field or non-finite values also mean HELD (reason recorded). The TOST of the analyzer is NOT part of the gate. The thresholds are not changed after the result has been seen.
  python3 e57_orc16_gate.py --result J --json V | --selftest      exit 0 = RUN (both hold), 20 = HELD"""
import sys,json,math,argparse,pathlib
FIELDS=(('primary','C1_A_minus_NATS'),('primary','C3_A_minus_AP'))
def verdict(result_path):
    v=dict(result=str(result_path),rule='primary.C1_A_minus_NATS.ci95_two_sided[0] > 0 AND primary.C3_A_minus_AP.ci95_two_sided[0] > 0',values={},run=False,reason=None)
    try:r=json.load(open(result_path))
    except Exception as e:v['reason']='result file missing or unreadable: '+repr(e)[:150];return v
    try:
        for blk,key in FIELDS:
            lo=float(r[blk][key]['ci95_two_sided'][0]);v['values'][f'{blk}.{key}.ci95_two_sided[0]']=lo
            if not math.isfinite(lo):v['reason']=f'non-finite value in {blk}.{key}';return v
    except Exception as e:v['reason']='field missing: '+repr(e)[:150];return v
    ok=all(x>0 for x in v['values'].values());v['run']=bool(ok);v['reason']='both lower ends > 0' if ok else 'at least one lower end <= 0'
    v['seeds']=r.get('seeds');v['analyzer_status']=r.get('status');return v
def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);mk=lambda a,b:{'seeds':list(range(101,109)),'status':'rules fixed in advance','primary':{'C1_A_minus_NATS':{'ci95_two_sided':[a,a+1]},'C3_A_minus_AP':{'ci95_two_sided':[b,b+1]}}}
        for nm,(a,b),want in (('both',(0.1,0.2),True),('c1',(-0.1,0.2),False),('c3',(0.1,-0.01),False),('zero',(0.0,0.2),False)):
            json.dump(mk(a,b),open(td/f'{nm}.json','w'));assert verdict(td/f'{nm}.json')['run']==want,nm
        assert not verdict(td/'none.json')['run'] and 'missing' in verdict(td/'none.json')['reason']
        json.dump({'primary':{}},open(td/'bad.json','w'));assert not verdict(td/'bad.json')['run']
        json.dump(mk(float('nan'),1),open(td/'nan.json','w'));assert not verdict(td/'nan.json')['run']
    print('e57_orc16_gate selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    v=verdict(a.result);print(json.dumps(v,indent=1))
    if a.json:pathlib.Path(a.json).write_text(json.dumps(v,indent=1))
    print('GATE VERDICT:','RUN 1302o' if v['run'] else 'HELD 1302o ('+str(v['reason'])+')');sys.exit(0 if v['run'] else 20)
if __name__=='__main__':main()
