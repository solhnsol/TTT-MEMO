"""CPU: what is VISIBLE (under SAMEVIS, row of the current sentence) between the earlier mentions K of a read fact and its current sentence (job 1344t3 question; new file; thread-server 11/26). For every read fact sentence of chunks 1..127: the window columns from the end of the LATEST earlier mention (K_last) to the start of the current sentence, and from the start of the EARLIEST earlier mention in the window (K_first) to the current sentence start, classified: other-fact sentence tokens of earlier chunks (hidden by SAMEVIS), background tokens of earlier chunks, same-fact earlier mentions, other-fact sentence tokens of the current chunk before the sentence (visible), background tokens of the current chunk before the sentence. Means over sentences of the VISIBLE counts.
  python e63_kvmask_gap.py --seeds 231,232 [--out F]"""
import sys,collections
import numpy as np
a=sys.argv;SEEDS=[int(x) for x in (a[a.index('--seeds')+1] if '--seeds' in a else '231').split(',')];OUT=a[a.index('--out')+1] if '--out' in a else None
import e59_u12 as U,e55_run as R55,e57_syn_pass as SP,e59_leak_probe as LP,e59_hidearms as H,e59_kvmask as KM
U.install();tk=LP.tk_load();CH=1024;WIN=8*CH;NW=8;lines=[];P=lambda *x:(lines.append(' '.join(str(y) for y in x)),print(*x,flush=True))
tot=collections.defaultdict(list)
for seed in SEEDS:
    enc,w,tf=R55.build(SP.synth_args(seed,U.CELL),seed,read_apps={16,32,64,128},arms=['NAT']);ap,miss,n=LP.answer_positions(w,tk,128);assert miss==0;ch=w['chunks'];inner=H.uv_for('OTHERONLY',ch,seed,lambda *x:None)
    for k in range(1,128):
        base=(k*CH-NW*CH+np.arange(WIN+CH))>=0;kp,_=inner.keep(k,None,None,None,None,base);kp=np.asarray(kp,bool)
        # column classes
        cls=np.zeros(WIN+CH,np.int8)   # 0 background, 1 sentence token (fact id in fid)
        fid=np.full(WIN+CH,-1,np.int64)
        for c in range(max(0,k-NW),k+1):
            off=(c-k+NW)*CH
            for x in ch[c]['sents']:cls[off+int(x['start']):off+int(x['end'])]=1;fid[off+int(x['start']):off+int(x['end'])]=int(x['fact'])
        for d in ap[k]:
            f=int(d['fact'])
            if f>=16:continue
            a_,b_=d['span'];keys=KM.same_fact_keys(ch,k,f)
            if not keys:continue
            cs=WIN+a_;kl_end=max(e for s,e in keys);kf_start=min(s for s,e in keys)
            for nm,lo in (('K_last end -> sentence',kl_end),('K_first start -> sentence',kf_start)):
                cols=np.arange(lo,cs);vis=kp[a_,cols];inwin=cols<WIN;cur=~inwin;same=(fid[cols]==f)&(cls[cols]==1)
                oth=(cls[cols]==1)&(fid[cols]!=f);bg=cls[cols]==0
                tot[(nm,'positions in the region')].append(len(cols));tot[(nm,'visible: all')].append(int(vis.sum()))
                tot[(nm,'visible: background, earlier chunks')].append(int((vis&bg&inwin).sum()));tot[(nm,'hidden: other-fact sentences, earlier chunks')].append(int((~vis&oth&inwin).sum()));tot[(nm,'visible: same-fact mentions')].append(int((vis&same).sum()))
                tot[(nm,'visible: other-fact sentences, current chunk')].append(int((vis&oth&cur).sum()));tot[(nm,'visible: background, current chunk')].append(int((vis&bg&cur).sum()))
    P('seed',seed,'done')
for nm in ('K_last end -> sentence','K_first start -> sentence'):
    P(f'== region {nm}: mean per sentence over {len(tot[(nm,"positions in the region")])} sentences ('+','.join(map(str,SEEDS))+')')
    for key in ('positions in the region','visible: all','visible: background, earlier chunks','hidden: other-fact sentences, earlier chunks','visible: same-fact mentions','visible: other-fact sentences, current chunk','visible: background, current chunk'):P(f'  {key}: {np.mean(tot[(nm,key)]):.1f}')
if OUT:open(OUT,'w').write('\n'.join(lines)+'\n')
