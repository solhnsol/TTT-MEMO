"""CPU check of the arms M2As-r / M2As-w of job 1344t3 (e59_kvmask2.py; new file; thread-server 11/26): per seed and chunk the hidden (query row, key) pair counts of QMaskUV2 against a brute-force count from the SAMEVIS keep matrix, chunk 0 = SAMEVIS, nothing outside the intended rows / columns is hidden. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu.
  python e63_kvmask2_check.py --seeds 231,...,238 [--out F]"""
import sys,collections
import numpy as np
a=sys.argv;SEEDS=[int(x) for x in (a[a.index('--seeds')+1] if '--seeds' in a else '231').split(',')];OUT=a[a.index('--out')+1] if '--out' in a else None
import e59_u12 as U,e55_run as R55,e57_syn_pass as SP,e59_leak_probe as LP,e59_kvmask as KM,e59_kvmask2 as K2,e59_hidearms as H
U.install();tk=LP.tk_load();CH=1024;WIN=8*CH;NW=8;lines=[];P=lambda *x:(lines.append(' '.join(str(y) for y in x)),print(*x,flush=True));tot=collections.Counter()
for seed in SEEDS:
    enc,w,tf=R55.build(SP.synth_args(seed,U.CELL),seed,read_apps={16,32,64,128},arms=['NAT']);ap,miss,n=LP.answer_positions(w,tk,128);assert miss==0;ch=w['chunks'];c=collections.Counter()
    for k in range(128):
        base=(k*CH-NW*CH+np.arange(WIN+CH))>=0;inner=H.uv_for('OTHERONLY',ch,seed,lambda *x:None);kp0,_=inner.keep(k,None,None,None,None,base);kp0=np.asarray(kp0,bool)
        for arm,mode in K2.ARMS2.items():
            u=K2.QMaskUV2(inner,mode,ch,ap,seed);kp,_=u.keep(k,None,None,None,None,base);hid=kp0&~kp;assert not (kp&~kp0).any();exp=0;allowed=np.zeros((CH,WIN+CH),bool)
            if mode=='x':
                for xs in ch[k]['sents']:
                    f=int(xs['fact'])
                    if f<16:continue
                    keys=KM.same_fact_keys(ch,k,f)
                    if not keys:continue
                    rows=np.arange(int(xs['start']),int(xs['end']));cols=np.concatenate([np.arange(s,e) for s,e in keys]);exp+=int(kp0[np.ix_(rows,cols)].sum());allowed[np.ix_(rows,cols)]=True
            for d in (ap[k] if mode!='x' else []):
                f=int(d['fact'])
                if f>=16:continue
                keys=KM.same_fact_keys(ch,k,f);a_,b_=d['span']
                if not keys and mode=='r2':continue
                if mode=='r2':rows=np.arange(a_,b_);cols=np.arange(min(s for s,e in keys),WIN+a_)
                else:rows=np.arange(a_,b_);cols=np.concatenate([np.arange(s,e) for s,e in keys]+[np.arange(WIN+int(x['start']),WIN+int(x['end'])) for x in ch[k]['sents'] if int(x['fact'])!=f])
                exp+=int(kp0[np.ix_(rows,cols)].sum());allowed[np.ix_(rows,cols)]=True
            assert int(hid.sum())==exp==u.stat[k]['pairs'] and not (hid&~allowed).any(),(seed,k,arm,int(hid.sum()),exp)
            if k==0 and mode in ('r2','x'):assert not hid.any()
            c[arm+' pairs']+=exp;c[arm+' rows']+=u.stat[k]['rows']
    P(f'seed {seed}: ok; '+', '.join(f'{k} {v}' for k,v in sorted(c.items())));tot.update(c)
P('total over seeds '+str(SEEDS)+': '+', '.join(f'{k} {v}' for k,v in sorted(tot.items())))
if OUT:open(OUT,'w').write('\n'.join(lines)+'\n')
