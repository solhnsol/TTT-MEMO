"""E57 NATOTHERPREV / NATRANDPREV runs (GPU; new file; job 1308z proposal, Hans 11/17; wrapper of the unchanged e57_oracle_run.py, which is imported and NOT edited; same write run as job 1302o / 1307: engine SGDSEL = clipped SGD, update-path mask only, K0 = 0, reads at exposures 16 / 32 / 64 / 128 out-of-window + in-window at 128, --nat-loss-every 8). The arms differ from NATALLPREV only in the update-path mask of e57_prevctrl_masks.py:
  NATOTHERPREV  a sentence row hides the earlier fact sentences of the OTHER facts only (own-fact sentences stay visible); parsed as the mask code 'ORACLE' of e57_oracle_run.main, OM.OracleUV replaced by OtherPrevFactUV
  NATRANDPREV   a sentence row hides, per earlier chunk, exactly as many random FILLER keys of that chunk as ALLPREV hides (number- and distance-matched random control); parsed as the code 'ORACLE_RAND', OM.OracleRandUV replaced by RandPrevKeysUV (seeded by the seed, chunk, sentence)
The replacement is made for this process only; the run refuses any other arm so that no ORACLE arm can silently use a replaced mask. The saved json has mask_arm 'ORACLE' / 'ORACLE_RAND' (field of the unchanged main), arm 'NATOTHERPREV' / 'NATRANDPREV', tag 'prevctrl' and uv_arm 'OTHERPREV' / 'RANDPREV' in every log row (RANDPREV logs rand_hit_rate = 0 and the hidden pair count).
  python3 e57_prevctrl_run.py --seed S --cell v3_16_1 --out O --masks none --mask-arm ALLQ256 --arms NATOTHERPREV,NATRANDPREV --save-apps 16,32,64,128 --nat-loss-every 8 --tag prevctrl | --selftest"""
import sys,pathlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_oracle_run as OR
import e57_oracle_masks as OM
import e57_prevctrl_masks as PM
ARMS=('NATOTHERPREV','NATRANDPREV');_orig_parse=OR.parse_arm2
def parse_arm3(name,mask_arm,rand_arm='RANDQ256P50'):
    if name=='NATOTHERPREV':return 'ORACLE',1.0
    if name=='NATRANDPREV':return 'ORACLE_RAND',1.0
    return _orig_parse(name,mask_arm,rand_arm)
def install():
    OR.parse_arm2=parse_arm3;OM.OracleUV=PM.OtherPrevFactUV;OM.OracleRandUV=PM.RandPrevKeysUV
def selftest():
    PM.selftest()
    import e57_natmask_run as NM
    install();assert OR.parse_arm2('NATOTHERPREV','ALLQ256')==('ORACLE',1.0) and OR.parse_arm2('NATRANDPREV','ALLQ256')==('ORACLE_RAND',1.0) and OR.parse_arm2('NATORACLE','ALLQ256')==('ORACLE',1.0) and OR.parse_arm2('NATS','ALLQ256')==NM.parse_arm('NATS','ALLQ256','RANDQ256P50')
    assert OM.OracleUV is PM.OtherPrevFactUV and OM.OracleRandUV is PM.RandPrevKeysUV and OR.OM.OracleUV is PM.OtherPrevFactUV
    ch=PM.AP.synth_chunks();base=PM.AP.natural_base(10)
    u=OR.OM.OracleUV(ch,0,lambda *x:None);_,i1=u.keep(10,None,None,None,None,base);assert i1['uv_arm']=='OTHERPREV'
    v=OR.OM.OracleRandUV(ch,3,0,lambda *x:None);_,i2=v.keep(10,None,None,None,None,base);assert i2['uv_arm']=='RANDPREV' and i2['rand_hit_rate']==0.0       # the call signature of the unchanged main: (chunks, seed, K0, log)
    import inspect;src=inspect.getsource(OR.main);assert "OM.OracleUV(w['chunks'],K0,lg) if ma=='ORACLE' else OM.OracleRandUV(w['chunks'],seed,K0,lg) if ma=='ORACLE_RAND'" in src and 'parse_arm2(' in src
    print('e57_prevctrl_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    arms=None
    for i,x in enumerate(sys.argv):
        if x=='--arms':arms=sys.argv[i+1]
    assert arms and all(a in ARMS for a in arms.split(',')),'e57_prevctrl_run.py runs ONLY the arms '+','.join(ARMS)
    if '--tag' not in sys.argv:sys.argv+=['--tag','prevctrl']
    install();OR.main()
if __name__=='__main__':main()
