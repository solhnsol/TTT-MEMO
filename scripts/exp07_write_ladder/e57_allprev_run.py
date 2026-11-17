"""E57 NATALLPREV run (GPU; new file; job 1307 proposal; wrapper of the unchanged e57_oracle_run.py, which is imported and NOT edited). The arm NATALLPREV is the same write run as NATORACLE / NATPLACEBO of job 1302o (engine SGDSEL = clipped SGD, update-path mask only, no norm correction, K0 = 0 by default, reads at exposures 16 / 32 / 64 / 128 out-of-window + in-window at 128, tokdiag every --nat-loss-every chunks) with ONE change: the update-path mask is e57_allprev_masks.AllPrevFactUV (a sentence row does not see ANY earlier fact sentence of the window, own fact or other fact; filler rows see everything). Implementation: the arm name NATALLPREV is parsed as the mask code 'ORACLE' of e57_oracle_run.main (parse_arm2 wrapped) and e57_oracle_masks.OracleUV is replaced by AllPrevFactUV for this process only; the run refuses any arm other than NATALLPREV so that no ORACLE arm can silently use the replaced mask. The saved json carries mask_arm 'ORACLE' (field of the unchanged main) but arm 'NATALLPREV', tag 'allprev' and uv_arm 'ALLPREV' in every log row.
  python3 e57_allprev_run.py --seed S --cell v3_16_1 --out O --arms NATALLPREV --tag allprev --save-apps 16,32,64,128 --nat-loss-every 8 | --selftest"""
import sys,pathlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_oracle_run as OR
import e57_oracle_masks as OM
import e57_allprev_masks as AP
_orig_parse=OR.parse_arm2
def parse_arm3(name,mask_arm,rand_arm='RANDQ256P50'):
    if name=='NATALLPREV':return 'ORACLE',1.0
    return _orig_parse(name,mask_arm,rand_arm)
def install():
    OR.parse_arm2=parse_arm3;OM.OracleUV=AP.AllPrevFactUV
def selftest():
    AP.selftest()
    import e57_natmask_run as NM
    install();assert OR.parse_arm2('NATALLPREV','ALLQ256')==('ORACLE',1.0) and OR.parse_arm2('NATORACLE','ALLQ256')==('ORACLE',1.0) and OR.parse_arm2('NATS','ALLQ256')==NM.parse_arm('NATS','ALLQ256','RANDQ256P50')
    assert OM.OracleUV is AP.AllPrevFactUV and OR.OM.OracleUV is AP.AllPrevFactUV
    ch=AP.synth_chunks();u=OR.OM.OracleUV(ch,0,lambda *x:None);kp,info=u.keep(10,None,None,None,None,AP.natural_base(10));assert info['uv_arm']=='ALLPREV'
    import inspect;src=inspect.getsource(OR.main);assert "OM.OracleUV(w['chunks'],K0,lg) if ma=='ORACLE'" in src and 'parse_arm2(' in src      # the unchanged main reaches both replaced names
    print('e57_allprev_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    arms=None
    for i,x in enumerate(sys.argv):
        if x=='--arms':arms=sys.argv[i+1]
    assert arms=='NATALLPREV','e57_allprev_run.py runs ONLY --arms NATALLPREV (the replaced mask must not be used by another arm)'
    if '--tag' not in sys.argv:sys.argv+=['--tag','allprev']
    install();OR.main()
if __name__=='__main__':main()
