"""E32 request (design thread; read-only table from existing files, no new computation of model quantities): per cell x seed x pass: D_alt (R1-alt, mean of the 15 pairs), first-chunk NLL rise, fiction-tail NLL rise, long-context beyond-window NLL rise (all vs the cell's own W0 state = pass 0, world average A/B),
canary max rise (max of gen / tail, both worlds), mean update norm per chunk, clip fraction (chunks with c_k < 1). Missing -> null. Output: OUT/e32_table.json"""
import sys,os,json
sys.argv=['x'];sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np,analyze_s3_stage2 as A
OUT=A.OUT;R=lambda x:None if x is None else round(float(x),4)
CELLS={ # name: (level,tag,max pass, longnll file template, source of update stats)
 'native_x1':('L0fs','main',8,'longnll_L0fs_fskeep_s{sd}.json','fskeep'),
 'native_x2':('L0fs','fsx2',8,'longnll_L0fs_fsx2_s{sd}.json',None),
 'masked_fs':('L0mwfs','x1',8,'longnll_L0mwfs_x1_s{sd}.json',None),
 'oracle':('S3','oracle_p4',4,'longnll_S3_oracle_p4_s{sd}.json','s3'),
 'prefix_all':('S3','prefixall_p4',4,'longnll_S3_prefixall_p4_s{sd}.json','s3'),
 'rand_dist_O':('S3','randdist_p4',4,'longnll_S3_randdist_p4_s{sd}.json','s3'),
 'recency_O':('S3','recency_p4',4,'longnll_S3_recency_p4_s{sd}.json','s3'),
 'attn_sel_k64':('S3','attnsel_p4',4,'longnll_S3_attnsel_p4_s{sd}.json','s3'),
 'rand_attn':('S3','randattn_p4',4,'longnll_S3_randattn_p4_s{sd}.json','s3'),
 'recency_attn':('S3','recencyattn_p4',4,'longnll_S3_recencyattn_p4_s{sd}.json','s3')}
res={}
for sd in (101,102,103):
    AL,S=A.mods(sd)
    for name,(lev,tag,mp,lf,src) in CELLS.items():
        if name=='native_x2' and sd!=101:continue
        try:A_,B_=S.load(lev,tag)
        except Exception as e:continue
        alt=lambda e:{'epoch':e.get('epoch'),'margins':e['margins']['_alt']};b2=(alt(A_[0]),alt(B_[0]))
        ln=json.load(open(OUT/lf.format(sd=sd))) if (OUT/lf.format(sd=sd)).exists() else None
        # update statistics
        up={}
        def rawlog(w):
            fn={'fskeep':OUT/f'L0fs_fskeep_s{sd}_{w}.saved.json','s3':OUT/f'S3_{tag}_s{sd}_{w}.saved.json'}.get(src)
            return json.load(open(fn))['log'] if fn is not None and fn.exists() else None
        cum={}
        for w,X in (('A',A_),('B',B_)):
            cum[w]={e.get('epoch'):e.get('cum_update_norm') for e in X if 'margins' in e}
        out={}
        for k in range(0,mp+1):
            a=[e for e in A_ if e.get('epoch')==k and 'margins' in e];b=[e for e in B_ if e.get('epoch')==k and 'margins' in e]
            if not a or not b:continue
            a,b=a[0],b[0];m0a,m0b=A_[0]['margins'],B_[0]['margins']
            d=float(AL.stats(alt(a),alt(b),b2)['D'].mean()) if k>0 else 0.0
            tail=[x['margins']['_tail_nll']-y['margins']['_tail_nll'] for x,y in ((a,A_[0]),(b,B_[0]))]
            can=max([x['margins'][c]-y['margins'][c] for x,y in ((a,A_[0]),(b,B_[0])) for c in ('_gen_nll','_tail_nll') if c in x['margins']])
            e32={'D_alt':R(d),'tail_nll_rise':R(np.mean(tail)),'canary_max':R(can),'first_chunk_nll_rise':None,'longctx_beyond_rise':None,'mean_update_norm_per_chunk':None,'clip_fraction':None}
            if ln:
                ee=[ [x for x in ln[w] if x['epoch']==k] for w in 'AB']
                if all(ee):e32['first_chunk_nll_rise']=R(np.mean([ee[i][0]['chunk0']-ln['W0']['chunk0'] for i in (0,1)]));e32['longctx_beyond_rise']=R(np.mean([ee[i][0]['rise_beyond'] for i in (0,1)]))
            if k>0:
                prev=[cum[w].get(k-1) if k>1 else 0.0 for w in 'AB'];now=[cum[w].get(k) for w in 'AB']
                if all(x is not None for x in now) and all(x is not None for x in prev):e32['mean_update_norm_per_chunk']=R(np.mean([(now[i]-prev[i])/63 for i in (0,1)]))
                if src=='s3':
                    L=[rawlog(w) for w in 'AB']
                    if all(L):e32['clip_fraction']=R(np.mean([np.mean([s['coef']<1 for s in l[k]['chunks']]) for l in L]))
                if src=='fskeep':
                    L=[rawlog(w) for w in 'AB']
                    if all(L):
                        cl=[[e for e in l if e.get('epoch')==k and 'clip' in e] for l in L]
                        if all(cl):e32['clip_fraction']=R(np.mean([np.mean([c<1 for _,c in x[0]['clip']]) for x in cl]))
            out[str(k)]=e32
        res.setdefault(name,{})[str(sd)]=out
res['_notes']={'rise_reference':'every NLL rise is vs the SAME run state at W0 (pass 0), averaged over worlds A and B; not vs same-pass native','first_chunk_nll':'mean token NLL of chunk 0 (1023 tokens) of the held-out 32-chunk document pg2160 read natively from the written weights (longnll files); long-context beyond = positions >= 8192 of that document; available only where a longnll file exists for that pass (S3 cells: pass 4 only; native/masked 8-pass runs: all passes)','tail_nll':'name-scrubbed pg1342 neutral tail, 7 chunks (the canary tail)','canary_max':'max over gen (pg11) and tail NLL rises and both worlds','D_alt':'R1-alt fp32, mean over 15 pairs, worlds A/B paired; null/'+'0 at pass 0','update_stats':'native_x1 clip from the fskeep re-stream (pass-state hashes identical to the main run); masked_fs and native_x2 have no per-chunk clip log (null); mean update norm = (cum_update_norm difference)/63'}
json.dump(res,open(OUT/'e32_table.json','w'),separators=(',',':'));print('cells',[k for k in res if not k.startswith('_')],'bytes',os.path.getsize(OUT/'e32_table.json'))
