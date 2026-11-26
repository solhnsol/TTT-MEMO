"""CPU: direct comparison of the C0 keep matrix and the M2As-s keep matrix (SAMEVIS + the rows of the current sentence of a read fact cannot see the earlier mentions K of the same fact), per (query row class, key column class) (thread-server 11/26; question of the design thread for job 1344t3 M2As-o). Row classes: S = rows of a READ-fact sentence of the current chunk, E = rows of an extra-fact sentence of the current chunk, B = background rows. Column classes: K = earlier mentions of the SAME fact as the row's sentence (window), Oe = sentences of OTHER facts in earlier chunks, Be = background in earlier chunks, Sc = own sentence (current chunk), Oc = other-fact sentences of the current chunk, Bc = background of the current chunk. Prints for every class pair the number of cells visible in C0 only and in the arm only (summed over chunks 1..127 of the seeds).
  python e63_c0_vs_s.py --seeds 231 [--out F]"""
import sys,collections
import numpy as np
a=sys.argv;SEEDS=[int(x) for x in (a[a.index('--seeds')+1] if '--seeds' in a else '231').split(',')];OUT=a[a.index('--out')+1] if '--out' in a else None;ARM=a[a.index('--arm')+1] if '--arm' in a else 's'      # s = M2As-s, o = M2As-o, r2 = M2As-r
import e59_kvmask2 as K2
import e59_u12 as U,e55_run as R55,e57_syn_pass as SP,e59_leak_probe as LP,e59_kvmask as KM,e59_hidearms as H,e59_shadow as SH
from e59_kvpos2 import CONDS
U.install();tk=LP.tk_load();CH=1024;WIN=8*CH;NW=8;lines=[];P=lambda *x:(lines.append(' '.join(str(y) for y in x)),print(*x,flush=True))
c0only=collections.Counter();sonly=collections.Counter();both=collections.Counter();nvis=collections.Counter()
for seed in SEEDS:
    enc,w,tf=R55.build(SP.synth_args(seed,U.CELL),seed,read_apps={16,32,64,128},arms=['NAT']);ap,miss,n=LP.answer_positions(w,tk,128);assert miss==0;ch=w['chunks']
    inner=SH.mk_uv(CONDS['SAMEVIS'],w,seed,lambda *x:None);c0=SH.mk_uv(CONDS['C0'],w,seed,lambda *x:None)
    for k in range(1,128):
        base=(k*CH-NW*CH+np.arange(WIN+CH))>=0;kpc,_=c0.keep(k,None,None,None,None,base);kpc=np.asarray(kpc,bool);qm=(KM.QMaskUV(inner,'s',ch,ap,seed) if ARM=='s' else K2.QMaskUV2(inner,ARM,ch,ap,seed));kps,_=qm.keep(k,None,None,None,None,base)
        causal=(np.arange(CH)[:,None]+WIN)>=np.arange(WIN+CH)[None,:]   # a key later than the query is never attended
        rowfact=np.full(CH,-1,np.int64);rowcls=np.full(CH,'B',dtype='<U1')
        for x in ch[k]['sents']:
            s,e,f=int(x['start']),int(x['end']),int(x['fact']);rowfact[s:e]=f;rowcls[s:e]='S' if f<16 else 'E'
        colfact=np.full(WIN+CH,-1,np.int64);colcur=np.zeros(WIN+CH,bool);colsent=np.zeros(WIN+CH,bool)
        for c in range(max(0,k-NW),k+1):
            off=(c-k+NW)*CH
            for x in ch[c]['sents']:colsent[off+int(x['start']):off+int(x['end'])]=True;colfact[off+int(x['start']):off+int(x['end'])]=int(x['fact'])
        colcur[WIN:]=True
        for rc in ('S','E','B'):
            rows=np.where(rowcls==rc)[0]
            if len(rows)==0:continue
            # per row: same-fact flag of the columns
            for r in rows:
                f=rowfact[r];same=(colsent&(colfact==f))&~colcur if f>=0 else np.zeros(WIN+CH,bool);own=(colsent&(colfact==f)&colcur) if f>=0 else np.zeros(WIN+CH,bool)
                cl=np.where(same,'K',np.where(own,'Sc',np.where(colsent&~colcur,'Oe',np.where(colsent&colcur,'Oc',np.where(colcur,'Bc','Be')))))
                v=causal[r]
                for cc in ('K','Oe','Be','Sc','Oc','Bc'):
                    m=(cl==cc)&v;c0only[(rc,cc)]+=int((kpc[r]&~kps[r]&m).sum());sonly[(rc,cc)]+=int((kps[r]&~kpc[r]&m).sum());both[(rc,cc)]+=int((kps[r]&kpc[r]&m).sum());nvis[(rc,cc)]+=int(m.sum())
    P('seed',seed,'done')
P('cells (causal, in window) per (row class, column class): visible in both / C0 only / arm only / all causal cells')
for rc in ('S','E','B'):
    for cc in ('K','Oe','Be','Sc','Oc','Bc'):P(f'  rows {rc}, cols {cc}: both {both[(rc,cc)]}, C0 only {c0only[(rc,cc)]}, arm only {sonly[(rc,cc)]}, all {nvis[(rc,cc)]}')
if OUT:open(OUT,'w').write('\n'.join(lines)+'\n')
