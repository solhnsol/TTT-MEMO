"""E57 public QA (65-version stream of the five bundles; GPU; new file): original-NAT (clipped SGD) write arms, the QA counterpart of e57_natmask_run.py (Council 11/09, Hans 11/09 'QA n=65 본 실행'; EXPLORATORY per Council). Same arms and engine as the synthetic runner (imports its functions; K0 is set to 0 here because the QA band masks ALLQ128 start at chunk 0, as in e57_pqa19_run):
  NATS = no mask, lr x1 (reproduces NAT)   NATMASK = ALLQ128 on every row   NATMASKC = same number of hidden keys per row as ALLQ128, random keys   DIRmaskNORMfull / DIRfullNORMmask = direction x norm swap between the masked and the full-context step
Write stage = e57_pqa19_run.write_stage with these arms (same bundle loop, canary at the read chunks, e57_pqa6_writes.json in --out); read stage = e57_pqa15_run.read_stage with P15.ARMS = these arms (same protocol as jobs 640/790/1251).
  python3 e57_pqa_natmask_run.py --stage write|read --out O [read args as e57_pqa27_run] | --selftest"""
import sys,os,json,pathlib,time,gc,hashlib,re
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_natmask_run as NM
import e57_qband_run as QR
import e57_pqa2_run as P2
import e57_pqa6_run as P6
import e57_pqa15_run as P15
import e57_pqa16_run as P16
import e57_pqa19_run as P19
NM.K0=0                      # QA: band masks from chunk 0
CH=1024;ARMS=['NATS','NATMASK','NATMASKC','DIRmaskNORMfull','DIRfullNORMmask'];BAND=128
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_run as R
    NM.register_engine()
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    meta,ids_z=P6.load_groups(out);SM=P6.make_selmask0();masks={'band_ALLQ128':np.asarray(BAND),'all_ALLQ128':np.asarray(1),'band_INF':np.asarray(10**6),'all_INF':np.asarray(1)}
    fn=out/'e57_pqa6_writes.json';W=json.load(open(fn)) if fn.exists() else {};t00=time.monotonic();every=8
    for g in meta['groups']:
        gid=str(g['gid'])
        if gid in W and ARMS[-1] in W[gid]:continue
        ids=ids_z[f'g{gid}'];nch=len(ids)//CH;assert nch==g['n_chunks'];rd_ch={int(v) for v in g['read_chunks'].values()};t0=time.monotonic()
        L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
        del _h;rc={(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)};del L0;gc.collect();res_g={}
        for arm in ARMS:
            ma,mult=NM.parse_arm(arm,'ALLQ128');mode=NM.arm_mode(arm)
            L=C.Ladder55(rt,save_dir=out/'states'/f'g{gid}_{arm}',canary=can);L.rec_cache=rc
            uv=NM.KeyMatchedQ(ma[3],BAND,int(gid),lg) if ma.startswith('KEY') else (QR.SelMaskQ('NATQ',masks,lg) if ma=='NATQ' else SM(ma,masks,lg))
            hold={};logfull=ma!='NATQ';uv=NM.with_base(uv,hold,L if (mode or logfull) else None,LD.jnp,full_every=every if logfull else 0,always_full=bool(mode))
            L.write_step=NM.swap_step(L.write_step,mult,mode,hold,every=every if logfull else 0);dch={c_ for c_ in range(0,nch,every)}|{nch-1};dfn=NM.make_nat_loss_diag(L,LD,s3,rt,hold,lg)
            res=L.run_stream(NM.ENGINE,1,ids,nch=nch,tag=f'pqa28g{gid}{arm}',log=lambda *x:None,read_chunks=rd_ch,canary_chunks=rd_ch,chunks_meta=[{'sents':[]} for _ in range(nch)],uv=uv,diag_fn=dfn,diag_chunks=dch)
            res_g[arm]=dict(states={str(k):dict(file=v['_file'],hash=v['_hash'],canary=v.get('canary')) for k,v in res['reads'].items()},ce=[e['ce'] for e in res['log']],update_norms=[e['update_norm'] for e in res['log']],gnorm=[e['gnorm'] for e in res['log']],coef=[e['coef'] for e in res['log']],canary_w0=cw0,wlog=hold.get('wlog',[]),diag={str(k):v for k,v in res.get('diag',{}).items()},lr_mult=mult,mask_arm=ma,swap_mode=mode)
            del L,res;gc.collect();lg(f'  bundle {gid} {arm} done ({time.monotonic()-t0:.0f}s)')
        del rc,Kh0,Vh0;gc.collect();W[gid]=res_g;fn.write_text(json.dumps(W,default=float));lg(f'bundle {gid}: {nch} chunks, {len(ARMS)} arms written in {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
def read_stage(a):
    P15.NREAD=[65];P15.ARMS=list(ARMS);P15.read_stage(dict(a,skip_repro=True))
def selftest():
    NM.K0=8;NM.selftest();NM.K0=0                    # the synthetic selftest assumes K0 = 8
    assert NM.K0==0 and ARMS==['NATS','NATMASK','NATMASKC','DIRmaskNORMfull','DIRfullNORMmask']
    for ar in ARMS:NM.parse_arm(ar,'ALLQ128')
    P16.selftest();print('e57_pqa_natmask_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),src790=g('--src790'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),para=g('--para'),train=g('--train'),dev=g('--dev'),rew64=g('--rew64'),json=g('--json'),n_set=g('--n-set'),max_groups=None)
    {'write':write_stage,'read':read_stage}[stage](a)
if __name__=='__main__':main()
