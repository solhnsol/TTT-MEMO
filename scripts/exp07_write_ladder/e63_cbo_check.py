"""CPU check of the write-2 keep matrices of job 1357 'cbo' (e59_cbo.py; new file; implementation thread 11/26). Per seed, chunk and arm, against the real C0 keep (e59_u12 SAME_U1U2): (a) only rows of the current sentences of the 16 READ facts differ from C0; (b) C0x2 = C0; (c) C0+ISO-mask: read rows see only their own sentence (subset of C0), nothing opened; (d) CLZ: opened columns are K of the same fact only, hidden share ~ r; (e) NAME-nolure: opened = K minus its name tokens exactly; (f) NAME-lure: opened = K minus names + the earlier-chunk sentences of all other facts; names never opened; (g) chunk 0 = C0 for every arm except ISO-mask; (h) ISOx2 rows: 16 read-fact sentences per chunk, packed into one 1024 chunk. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu.
  python e63_cbo_check.py --seeds 231,...,238 [--out F]"""
import sys,collections
import numpy as np
a=sys.argv;SEEDS=[int(x) for x in (a[a.index('--seeds')+1] if '--seeds' in a else '231').split(',')];OUT=a[a.index('--out')+1] if '--out' in a else None
import e59_u12 as U,e55_run as R55,e57_syn_pass as SP,e59_leak_probe as LP,e59_cbo as CB
from e59_kvmask import same_fact_keys
U.install();tk=LP.tk_load();CH=1024;NW=8;WIN=NW*CH;NREAD=16;lines=[];P=lambda *x:(lines.append(' '.join(str(y) for y in x)),print(*x,flush=True))
def cols_of(sl):return np.unique(np.concatenate([np.arange(s,e) for s,e in sl])) if sl else np.zeros(0,np.int64)
for seed in SEEDS:
    enc,w,tf=R55.build(SP.synth_args(seed,U.CELL),seed,read_apps={16,32,64,128},arms=['NAT']);ap,miss,n=LP.answer_positions(w,tk,128);assert miss==0;ch=w['chunks'];c0=U.uv_for('SAME_U1U2',ch,lambda *x:None)
    agg=collections.defaultdict(lambda:collections.Counter());ramp=[]
    for k in range(128):
        base=(k*CH-NW*CH+np.arange(WIN+CH))>=0;kp0,_=c0.keep(k,None,None,None,None,base);kp0=np.asarray(kp0,bool)
        readrows=np.zeros(CH,bool);sent={}
        for x in ch[k]['sents']:
            if int(x['fact'])<NREAD:readrows[int(x['start']):int(x['end'])]=True;sent[int(x['fact'])]=(int(x['start']),int(x['end']))
        for arm in [x for x in CB.ARMS if x!='ISOx2']:
            m=CB.Write2Mask(arm,ch,ap,seed);kp=m.keep(k,kp0,base);d=kp!=kp0
            assert not d[~readrows].any(),(seed,k,arm,'non-read rows changed')
            if arm=='C0x2':assert not d.any();continue
            if k==0 and arm!='C0+ISO-mask':assert not d.any(),(seed,arm,'chunk 0')
            for f,(s0,e0) in sent.items():
                R=kp[s0:e0];R0=kp0[s0:e0];opened=np.where((R&~R0).any(0))[0];closed_new=np.where((~R&R0).any(0))[0]
                K=cols_of(same_fact_keys(ch,k,f));_,pf,nm=CB.k_parts(ch,ap,k,f)
                if arm=='C0+ISO-mask':
                    assert len(opened)==0 and not R[:,:WIN+s0].any() and not R[:,WIN+e0:].any() and (R[:,WIN+s0:WIN+e0]==R0[:,WIN+s0:WIN+e0]).all(),(seed,k,f);agg[arm]['closed_pairs']+=int((R0&~R).sum());continue
                assert len(closed_new)==0,(seed,k,arm,f,'write 2 hides more than C0')
                if arm.startswith('CLZ'):
                    assert set(opened.tolist())<=set(K.tolist())
                    if len(K):r=CB.clz_r(m.spec,k);hid=len(K)-len(opened);agg[arm]['K']+=len(K);agg[arm]['hid']+=hid;agg[arm]['pf']+=len(pf);agg[arm]['pf_hid']+=int((~np.isin(pf,opened)).sum());(ramp.append((k,hid/len(K))) if arm=='CLZ-ramp' else None);agg[arm]['exp_hid']+=r*len(K)
                    assert (R[:,opened]==np.broadcast_to(base,(CH,WIN+CH))[s0:e0][:,opened]).all()
                else:
                    want=set(np.setdiff1d(K,nm).tolist())
                    if arm=='NAME-lure':want|=set(cols_of(CB.other_keys(ch,k,f)).tolist())
                    assert set(opened.tolist())==want,(seed,k,arm,f,len(opened),len(want));assert not R[:,nm].any() if len(nm) else True
                    agg[arm]['names']+=len(nm);agg[arm]['opened_cols']+=len(opened);agg[arm]['pf_open']+=int(np.isin(pf,opened).sum());agg[arm]['pf']+=len(pf)
        rows=CB.iso_rows_chunk(w,enc,seed,k);assert len(rows)==16;tot=sum(len(r[0]) for r in rows);assert tot<=CH,(seed,k,tot);agg['ISOx2']['tokens']+=tot
    for arm,cn in agg.items():P(seed,arm,dict(cn))
    rr=np.asarray(ramp);P(seed,'CLZ-ramp hidden share by chunk quarter',[round(float(rr[(rr[:,0]>=q*32)&(rr[:,0]<(q+1)*32),1].mean()),3) for q in range(4)])
    P(seed,'checks ok')
P('e63_cbo_check: all seeds ok',SEEDS)
if OUT:open(OUT,'w').write('\n'.join(lines)+'\n')
