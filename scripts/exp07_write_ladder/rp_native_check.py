"""RunPod reproduction check (CPU, no GPU): D_alt per pass of a W1 native cell re-run on the pod (new) against the lab reference (ref).
  rp_native_check.py --new L0fs:w1rp [--ref L0fs:w1main | --ref-values 0.017,0.091,0.177,0.194] [--tol 0.02] [--seed 101] [--marker F]
Reference by result files in the same OUT (needs OUT/L0fs_w1main_s101_{A,B}.json + meta) or by the values of lab job 130 (pass 1-4, point estimates). PASS if |new - ref| <= tol at every pass 1-4; with --ref also reports the max abs difference of the 15 pair-level D_alt. Writes OUT/rp_native_check_s<seed>.json; --marker writes PASS or FAIL into the file. The tolerance is a parameter, not a derived number (the agreed value comes from the design thread). Exit 0 PASS, 1 FAIL, 2 files missing."""
import os,sys,json,argparse,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e47_native_da as N
def pairvec(od,spec,seed,ep):
    lev,tag=spec.split(':',1);pre=f'{lev}_{tag}' if lev!='S3' else f'S3_{tag}';E={}
    for w in 'AB':
        L=json.load(open(od/f'{pre}_s{seed}_{w}.json'))['log'];E[w]={e['epoch']:e for e in L if isinstance(e.get('margins'),dict) and 'epoch' in e and '_alt' in e['margins']}
    return N.pairs_D(E['A'][ep],E['B'][ep],json.load(open(od/f'meta_{pre}_s{seed}.json')))
def check(od,new,ref,ref_values,tol,seed):
    rn,info=N.read(od,new,seed)
    if rn is None:return {'status':'missing','info':info},2
    if ref:
        rr,infor=N.read(od,ref,seed)
        if rr is None:return {'status':'missing','info':infor},2
        rv=[rr[e][0] for e in (1,2,3,4)]
    else:rv=[float(x) for x in ref_values.split(',')];infor='values given'
    nv=[rn[e][0] for e in (1,2,3,4)];diff=[a-b for a,b in zip(nv,rv)];res={'new':new,'ref':ref or 'values','D_alt_new':nv,'D_alt_ref':rv,'diff':diff,'tol':tol,'files_new':info,'files_ref':infor}
    if ref:res['pair_max_abs_diff_p4']=float(np.max(np.abs(pairvec(od,new,seed,4)-pairvec(od,ref,seed,4))))
    ok=all(abs(d)<=tol for d in diff);res['status']='PASS' if ok else 'FAIL';return res,0 if ok else 1
def selftest():
    td=Path(tempfile.mkdtemp());items=[]
    for f in ('L3','XP'):items+=[{'pair':(f'p{i//2}' if f=='L3' else f'x{i//2}'),'tid':i} for i in range(30)]
    for tag,s in (('w1main',1.0),('w1rp',1.0),('w1bad',0.5)):
        json.dump({'items':items,'fam_sizes':{'L3':30,'XP':30}},open(td/f'meta_L0fs_{tag}_s101.json','w'))
        for w,sg in (('A',1),('B',-1)):
            log=[{'epoch':0,'margins':{'_alt':{'L3':[0.0]*30,'XP':[0.0]*30}}}]+[{'epoch':e,'margins':{'_alt':{'L3':[sg*0.1*e*s]*30,'XP':[0.0]*30}}} for e in (1,2,3,4)]
            json.dump({'log':log},open(td/f'L0fs_{tag}_s101_{w}.json','w'))
    r,c=check(td,'L0fs:w1rp','L0fs:w1main',None,0.02,101);assert c==0 and r['status']=='PASS' and r['pair_max_abs_diff_p4']<1e-9
    r,c=check(td,'L0fs:w1bad','L0fs:w1main',None,0.02,101);assert c==1 and r['status']=='FAIL'
    r,c=check(td,'L0fs:w1rp',None,'0.1,0.2,0.3,0.4',0.02,101);assert c==0
    r,c=check(td,'L0fs:nope',None,'0.1,0.2,0.3,0.4',0.02,101);assert c==2;print('rp_native_check SELFTEST OK')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--new');ap.add_argument('--ref',default=None);ap.add_argument('--ref-values',default=None);ap.add_argument('--tol',type=float,default=0.02);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--marker',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    assert a.new and (a.ref or a.ref_values),'--new and --ref or --ref-values'
    r,c=check(N.OUT,a.new,a.ref,a.ref_values,a.tol,a.seed);print(json.dumps(r,indent=1));json.dump(r,open(N.OUT/f'rp_native_check_s{a.seed}.json','w'),indent=1)
    if a.marker:Path(a.marker).write_text(r['status']+'\n')
    sys.exit(c)
