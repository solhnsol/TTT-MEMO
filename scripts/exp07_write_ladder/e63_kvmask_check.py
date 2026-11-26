"""CPU check of the query-position masks of job 1344t2 (e59_kvmask.py; new file; thread-server 11/26): per seed, the pair counts of the three arms against a brute-force count from the SAMEVIS keep matrix (real OtherPrevFactUV on the world chunks), equal numbers of hidden query rows of M2As-q and Mrand, the rows outside the sentence (place token at the sentence start), chunk 0 = SAMEVIS, and the stored random rows. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu.
  python e63_kvmask_check.py --seeds 231,...,238 [--out F]"""
import sys,json,collections
import numpy as np
a=sys.argv;SEEDS=[int(x) for x in (a[a.index('--seeds')+1] if '--seeds' in a else '231').split(',')];OUT=a[a.index('--out')+1] if '--out' in a else None
import e59_u12 as U,e55_run as R55,e57_syn_pass as SP,e59_leak_probe as LP,e59_kvmask as KM
import e59_hidearms as H
U.install();tk=LP.tk_load();CH=1024;WIN=8*CH;NW=8;lines=[];P=lambda *x:(lines.append(' '.join(str(y) for y in x)),print(*x,flush=True))
tot=collections.Counter()
for seed in SEEDS:
    enc,w,tf=R55.build(SP.synth_args(seed,U.CELL),seed,read_apps={16,32,64,128},arms=['NAT']);ap,miss,n=LP.answer_positions(w,tk,128);assert miss==0;ch=w['chunks'];c=collections.Counter()
    for k in range(128):
        base=(k*CH-NW*CH+np.arange(WIN+CH))>=0;inner=H.uv_for('OTHERONLY',ch,seed,lambda *x:None);kp0,_=inner.keep(k,None,None,None,None,base);kp0=np.asarray(kp0,bool)
        insent=np.full(CH,-1,np.int64)
        for x in ch[k]['sents']:insent[int(x['start']):int(x['end'])]=int(x['fact'])
        res={}
        for arm,mode in KM.ARMS.items():
            u=KM.QMaskUV(inner,mode,ch,ap,seed);kp,_=u.keep(k,None,None,None,None,base);hid=(kp0&~kp);assert not (kp&~kp0).any()
            exp=0;rows=0;outside=0;outside_eff=0
            for d in ap[k]:
                f=int(d['fact'])
                if f>=16:continue
                keys=KM.same_fact_keys(ch,k,f);rr,out=KM.query_rows(d,mode,seed,k);rows+=len(rr);outside+=out
                if keys:
                    cols=np.concatenate([np.arange(s,e) for s,e in keys]);exp+=int(kp0[np.ix_(rr,cols)].sum())
                    for r in rr:
                        if not d['span'][0]<=r<d['span'][1]:outside_eff+=int(kp0[r,cols].sum()>0)
            assert int(hid.sum())==exp==u.stat[k]['pairs'],(seed,k,arm,int(hid.sum()),exp,u.stat[k]['pairs'])
            if k==0:assert not hid.any()
            res[arm]=(exp,rows,outside,outside_eff);c[arm+' pairs']+=exp;c[arm+' rows']+=rows;c[arm+' rows outside sentence']+=outside;c[arm+' outside rows with an effect']+=outside_eff
        if k>0:assert res['M2As-q'][1]==res['Mrand'][1],(seed,k,res)
        for arm in KM.ARMS:
            if k==0:assert res[arm][0]==0
    P(f'seed {seed}: ok; '+', '.join(f'{k} {v}' for k,v in sorted(c.items())));tot.update(c)
P('total over seeds '+str(SEEDS)+': '+', '.join(f'{k} {v}' for k,v in sorted(tot.items())))
if OUT:open(OUT,'w').write('\n'.join(lines)+'\n')
