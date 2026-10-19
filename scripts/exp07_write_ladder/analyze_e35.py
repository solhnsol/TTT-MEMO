"""E35 analysis (CPU), rules of E35 section 2.4 (fixed before results). Usage: analyze_e35.py [--seed 101] [--out DIR] [--selftest]
Reads D_alt (R1-alt binding, margins['_alt']) of the L1 plain curve (L1_ck_s101_{A,B}.json) and of the ladder cells, per pair (15 pairs, world A/B averaged as in analyze_ladder.stats):
  R0 L1 plain   L1_ck                          (file prefix L1_ck)
  R1 L1fillfresh, R1b L1smallbatch             (L1fillfresh_r1fillfresh, L1smallbatch_r1b)
  R2 job 50 L0reset_P1_r2p1, R3 L0reset_P4_r3p4 (not yet approved), R4 L0reset_P8_r4p8 (not yet approved), R5 job 51 L0reset_P63_r5fresh
Missing cells are skipped and said so. Output: e35_analysis_s{seed}.md / .json in OUT.
Per cell and epoch e (1, 2, 4 main; 2 and 4 are the primary reading epochs):
  Q = D_alt(cell, e) / D_alt(R0, e)       pair bootstrap 95% CI (resampling indices shared by all cells -> paired)
  log Q decomposition over the ladder order R0,R1,R1b,R2,R3,R4,R5 (only the available cells): DeltaL(r) = log Q(r) - log Q(r_prev), CI by the same paired bootstrap;
  L_tot = log Q(R5); G = DeltaL / L_tot only when the CI upper bound of L_tot < 0; context-range part = log Q(R5) - log Q(R2) (R2->R5), outside-context part = log Q(R2) (R0->R2).
  Rule (2.4): context part >= 60% of L_tot (point estimate) and the part a clear drop (CI upper bound < 0) -> 'context range (length included)' ; outside part >= 60% likewise -> 'factors outside the context';
  else 'mixed'. Q(R5, e) > 0.5 (point estimate) -> hold, 'new background raised native'. Thresholds fixed, not changed here.
  Same cumulative update norm: the L1 curve (epoch 0..4, cum_update_norm) is linearly interpolated at the cell's cum at epoch e (no extrapolation), and the cell curve at the L1 cum; reported next to Q.
  Damage: max canary rise (gen pg11, neutral tail) over both worlds; > 0.10 is marked 'with damage, descriptive'."""
import os,sys,json,argparse,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CELLS=[('R0','L1','ck'),('R1','L1fillfresh','r1fillfresh'),('R1b','L1smallbatch','r1b'),('R2','L0reset_P1','r2p1'),('R3','L0reset_P4','r3p4'),('R4','L0reset_P8','r4p8'),('R5','L0reset_P63','r5fresh')]
EPOCHS=(1,2,4);PRIMARY=(2,4);NB=4000;THR=0.10
def load_cell(OUT,level,tag,seed):
    fa=OUT/f'{level}_{tag}_s{seed}_A.json';fb=OUT/f'{level}_{tag}_s{seed}_B.json'
    if not(fa.exists() and fb.exists()):return None
    return json.load(open(fa))['log'],json.load(open(fb))['log']
def fam_pairs(OUT,seed):
    for nm in (f'meta_L1_ck_s{seed}.json',f'meta_L0reset_P63_r5fresh_s{seed}.json'):
        if (OUT/nm).exists():
            m=json.load(open(OUT/nm));items=m['items'];idx={};i=0
            for f,n in m['fam_sizes'].items():idx[f]=[items[k]['pair'] for k in range(i,i+n)];i+=n
            return idx
    raise FileNotFoundError('no meta file with items/fam_sizes')
def D_alt(eA,eB,pairs):
    def M(f):
        a=np.asarray(eA['margins']['_alt'][f]);b=np.asarray(eB['margins']['_alt'][f]);d={}
        for p,x in zip(pairs[f],(a-b)/2):d.setdefault(p,[]).append(x)
        return np.asarray([np.mean(d[k]) for k in sorted(d,key=lambda s:int(s[1:]))])
    return M('L3')-M('XP')
def epoch_entries(log):
    return {e['epoch']:e for e in log if 'margins' in e and 'epoch' in e}
def cum_of(log,ep):
    """cumulative update norm at the end of epoch ep: the epoch entry field, else the last step row of that epoch."""
    e=[x for x in log if x.get('epoch')==ep and 'margins' in x]
    if e and 'cum_update_norm' in e[0]:return float(e[0]['cum_update_norm'])
    rows=[x for x in log if x.get('epoch')==ep and 'cum_update_norm' in x]
    return float(rows[-1]['cum_update_norm']) if rows else float('nan')
def canary(logA,logB,ep):
    r=[]
    for lg in (logA,logB):
        E=epoch_entries(lg);b=E[0]['margins'];r.append(max(E[ep]['margins'][k]-b[k] for k in ('_gen_nll','_tail_nll')))
    return max(r)
def interp(cx,Dy,x):
    """linear interpolation of per-pair D (rows of Dy) over increasing cx at x; None if x outside [cx0, cxN] (no extrapolation)."""
    cx=np.asarray(cx,float)
    if not np.isfinite(x) or x<cx[0]-1e-12 or x>cx[-1]+1e-12:return None
    j=int(np.searchsorted(cx,x,side='right')-1);j=min(max(j,0),len(cx)-2);w=(x-cx[j])/max(cx[j+1]-cx[j],1e-30);return (1-w)*Dy[j]+w*Dy[j+1]
def analyse(OUT,seed,nb=NB):
    pairs=fam_pairs(OUT,seed);rng=np.random.default_rng(35);data={};notes=[]
    for key,lev,tag in CELLS:
        c=load_cell(OUT,lev,tag,seed)
        if c is None:notes.append(f'{key} ({lev}_{tag}) missing: skipped');continue
        A,B=c;EA,EB=epoch_entries(A),epoch_entries(B);eps=sorted(set(EA)&set(EB))
        D={e:D_alt(EA[e],EB[e],pairs) for e in eps};cum={0:0.0,**{e:cum_of(A,e)/2+cum_of(B,e)/2 for e in eps if e>0}}   # world-averaged cumulative update norm
        data[key]={'D':D,'cum':cum,'canary':{e:canary(A,B,e) for e in eps if e>0},'file':f'{lev}_{tag}'}
    if 'R0' not in data:raise SystemExit('L1_ck (R0) missing: nothing to compare against')
    n=len(next(iter(data['R0']['D'].values())));idx=rng.integers(0,n,(nb,n))
    def bm(v):return np.asarray(v)[idx].mean(1)           # bootstrap means of a per-pair vector
    def ci(x):x=np.asarray(x,float);return float(np.nanmean(x)),float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5))
    order=[k for k,_,_ in CELLS if k in data];res={'seed':seed,'notes':notes,'cells':{},'epochs':{}}
    for e in EPOCHS:
        if any(e not in data[k]['D'] for k in order):notes.append(f'epoch {e} missing in some cell');
        ks=[k for k in order if e in data[k]['D']]
        if 'R0' not in ks:continue
        b0=bm(data['R0']['D'][e]);logQ={};Q={}
        for k in ks:
            bk=bm(data[k]['D'][e]);ok=(b0>0)&(bk>0);lq=np.where(ok,np.log(np.where(ok,bk,1)/np.where(ok,b0,1)),np.nan);logQ[k]=lq
            m0=data['R0']['D'][e].mean();mk=data[k]['D'][e].mean();pt=mk/m0 if m0>0 else float('nan')
            Q[k]={'Q_point':float(pt),'Q_ci':ci(bk/np.where(b0>0,b0,np.nan)),'D_cell':float(mk),'D_R0':float(m0),'canary_rise':data[k]['canary'].get(e),'damage_flag':'with damage, descriptive' if (data[k]['canary'].get(e) or 0)>THR else '',
                  'cum_update_norm':data[k]['cum'].get(e),'cum_update_norm_R0':data['R0']['cum'].get(e),'frac_bootstrap_positive':float(ok.mean())}
            # equal cumulative update norm (no extrapolation), both directions
            if k!='R0':
                cx0=[data['R0']['cum'][j] for j in sorted(data['R0']['cum']) if j==0 or j in data['R0']['D']];D0=[data['R0']['D'][j] if j>0 else np.zeros(n) for j in sorted(data['R0']['cum']) if j==0 or j in data['R0']['D']]
                cxk=[data[k]['cum'][j] for j in sorted(data[k]['cum']) if j==0 or j in data[k]['D']];Dk=[data[k]['D'][j] if j>0 else np.zeros(n) for j in sorted(data[k]['cum']) if j==0 or j in data[k]['D']]
                a=interp(cx0,D0,data[k]['cum'][e]);b=interp(cxk,Dk,data['R0']['cum'][e])
                Q[k]['Q_at_cell_cum']=None if a is None else ci(bm(data[k]['D'][e])/np.where(bm(a)>0,bm(a),np.nan))   # cell D at its epoch / L1 D interpolated at the cell's cum
                Q[k]['Q_at_L1_cum']=None if b is None else ci(bm(b)/np.where(b0>0,b0,np.nan))                         # cell D interpolated at L1's cum / L1 D at epoch e
        dl={};prev=None
        for k in ks:
            dl[k]=None if prev is None else ci(logQ[k]-logQ[prev]);prev=k
        out={'Q':Q,'DeltaL_vs_previous_cell':dl}
        if 'R5' in Q and 'R2' in Q and Q['R5']['Q_ci'][1]<=0:   # E35 5.7: R5's Q has no clear lower bound -> also decompose the raw D differences (not only log Q)
            bR,b2,b5=b0,bm(data['R2']['D'][e]),bm(data['R5']['D'][e])
            out['D_difference_decomposition']={'condition':'lower CI bound of Q(R5) <= 0','D_L1_minus_D_R2':ci(bR-b2),'D_R2_minus_D_R5':ci(b2-b5),'D_L1_minus_D_R5_total':ci(bR-b5)}
        if 'R5' in logQ:
            ltot=logQ['R5'];lt=ci(ltot);out['L_tot']=lt
            if 'R2' in logQ:
                ctx=logQ['R5']-logQ['R2'];outs=logQ['R2'];out['context_part_R2_to_R5']=ci(ctx);out['outside_part_R0_to_R2']=ci(outs)
            clear=lt[2]<0;out['L_tot_clear']=bool(clear)
            if clear and 'R2' in logQ:
                gctx=ctx/ltot;gout=outs/ltot;out['G_context']=ci(gctx);out['G_outside']=ci(gout)
                gc,go=ci(gctx)[0],ci(gout)[0]   # the 60% rule is read on the point estimate of G; the part itself must be a clear drop (its CI upper bound < 0)
                out['G_point']={'context':gc,'outside':go}
                if Q['R5']['Q_point']>0.5:out['verdict']='hold: Q(R5)>0.5 (native is about half of L1 or more at equal exposures); the new background raised native'
                elif gc>=0.6 and ci(ctx)[2]<0:out['verdict']='context range (length included) is the main part'
                elif go>=0.6 and ci(outs)[2]<0:out['verdict']='factors outside the context are the main part'
                else:out['verdict']='mixed (do not call it write dynamics)'
            else:out['verdict']='no decomposition: total drop not clear (CI upper bound of L_tot >= 0) or R2 missing'
        res['epochs'][e]=out
    res['cells']={k:{'file':v['file'],'cum':v['cum'],'D_mean':{e:float(np.mean(d)) for e,d in v['D'].items()}} for k,v in data.items()}
    return res
def fmt(t):return '-' if t is None else f'{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]'
def md_of(res):
    L=[f"# E35 analysis, seed {res['seed']} (rules E35 2.4; primary epochs 2 and 4)\n"]+[f'- note: {x}' for x in res['notes']]
    for e,o in sorted(res['epochs'].items()):
        L+=[f"\n## epoch {e}{' (primary)' if e in PRIMARY else ' (descriptive)'}\n",'| cell | D_alt | Q = D/D(R0) [95% CI] | Q at cell cum (cell / L1 interp.) | Q at L1 cum (cell interp. / L1) | canary rise | cum norm cell / R0 |','|---|---|---|---|---|---|---|']
        for k,q in o['Q'].items():L.append(f"| {k} | {q['D_cell']:+.3f} | {fmt(q['Q_ci'])} | {fmt(q.get('Q_at_cell_cum'))} | {fmt(q.get('Q_at_L1_cum'))} | {('' if q['canary_rise'] is None else format(q['canary_rise'],'+.3f'))} {q['damage_flag']} | {q['cum_update_norm'] if q['cum_update_norm'] is None else round(q['cum_update_norm'],2)} / {None if q['cum_update_norm_R0'] is None else round(q['cum_update_norm_R0'],2)} |")
        L.append('\nDeltaL = log Q(cell) - log Q(previous cell): '+'; '.join(f'{k} {fmt(v)}' for k,v in o['DeltaL_vs_previous_cell'].items() if v))
        if 'L_tot' in o:
            L.append(f"\nL_tot = log Q(R5) = {fmt(o['L_tot'])} (clear: {o.get('L_tot_clear')})")
            for key in ('outside_part_R0_to_R2','context_part_R2_to_R5','G_outside','G_context'):
                if key in o:L.append(f"- {key}: {fmt(o[key])}")
            L.append(f"- verdict (rule 2.4): {o.get('verdict')}")
        if 'D_difference_decomposition' in o:
            d=o['D_difference_decomposition'];L.append(f"\nD difference decomposition (Q(R5) lower CI bound <= 0; mean [95% CI], paired bootstrap): D_L1 - D_R2 = {fmt(d['D_L1_minus_D_R2'])}; D_R2 - D_R5 = {fmt(d['D_R2_minus_D_R5'])}; total D_L1 - D_R5 = {fmt(d['D_L1_minus_D_R5_total'])}")
    return '\n'.join(L)+'\n'
PCT=(0,10,25,50,75,90,100)
NCH_DEFAULT=63
def _load_log(OUT,stem):
    for nm in (f'{stem}.json',f'{stem}.saved.json'):
        if (OUT/nm).exists():return json.load(open(OUT/nm))['log'],nm
    return None,None
def pass_records(log,nch=None):
    """per pass (epoch >= 1): per-chunk lists coef / gn / upd where logged, plus the mean update norm from the cumulative increment. Accepts the real formats:
    job 37 (L0fs, ladder._native_passes with log_clip): entry['clip'] = [(grad_norm, clip_coef), ...] per chunk, only cum_update_norm per pass;
    R5 (L0reset): entry['steps'] = [{'chunk','update_norm','cum_update_norm', optional 'grad_norm','clip_coef'}]; flat step rows with 'update_norm' (L1 style) are read too."""
    ents=[e for e in log if 'margins' in e and e.get('epoch',0)>=1];recs=[];prev=0.0
    for e in sorted(ents,key=lambda x:x['epoch']):
        r={'pass':e['epoch'],'coef':None,'gn':None,'upd':None,'upd_mean_from_cum':None,'n_chunks':e.get('chunks')}
        if isinstance(e.get('clip'),list) and e['clip']:r['gn']=[float(x[0]) for x in e['clip']];r['coef']=[float(x[1]) for x in e['clip']];r['n_chunks']=len(e['clip'])
        st=e.get('steps')
        if isinstance(st,list) and st:
            if all('update_norm' in x for x in st):r['upd']=[float(x['update_norm']) for x in st]
            if all('clip_coef' in x for x in st):r['coef']=[float(x['clip_coef']) for x in st];r['gn']=[float(x['grad_norm']) for x in st]
            r['n_chunks']=len(st)
        rows=[x for x in log if x.get('epoch')==e['epoch'] and 'update_norm' in x and 'margins' not in x]
        if rows and r['upd'] is None:
            r['upd']=[float(x['update_norm']) for x in rows]
            if all('clip_coef' in x for x in rows) and r['coef'] is None:r['coef']=[float(x['clip_coef']) for x in rows];r['gn']=[float(x['grad_norm']) for x in rows]
            r['n_chunks']=len(rows)
        c=e.get('cum_update_norm')
        if c is not None:
            n=r['n_chunks'] or nch
            if n:r['upd_mean_from_cum']=(float(c)-prev)/n
            prev=float(c)
        recs.append(r)
    return recs
def _stats(r):
    o={'n_chunks':r['n_chunks'],'clip_ratio':None,'coef_pct':None,'coef_hist':None,'upd_mean':None,'upd_var':None,'upd_mean_source':None}
    if r['coef']:
        c=np.asarray(r['coef']);o['clip_ratio']=float(np.mean(c<1.0));o['coef_pct']={str(q):float(np.percentile(c,q)) for q in PCT}
        o['coef_hist']={'<0.25':int((c<0.25).sum()),'0.25-0.5':int(((c>=0.25)&(c<0.5)).sum()),'0.5-0.75':int(((c>=0.5)&(c<0.75)).sum()),'0.75-<1':int(((c>=0.75)&(c<1.0)).sum()),'=1 (unclipped)':int((c>=1.0).sum())}
    if r['upd']:u=np.asarray(r['upd']);o['upd_mean']=float(u.mean());o['upd_var']=float(u.var(ddof=1)) if len(u)>1 else None;o['upd_mean_source']='per-chunk update_norm'
    elif r['upd_mean_from_cum'] is not None:o['upd_mean']=r['upd_mean_from_cum'];o['upd_mean_source']='cumulative increment / chunks'
    return o
CLIP_CELLS=(('job37','L0fs_fskeep'),('R5','L0reset_P63_r5fresh'))
def clip_table(OUT,seed):
    """job 37 (native fs x1 on the existing background) vs R5 (job 51, fresh background): per pass and world, clip ratio, update-norm mean/variance, clip-coefficient distribution. Existing logs only."""
    res={'cells':{},'notes':[]}
    for key,stem in CLIP_CELLS:
        res['cells'][key]={}
        for w in 'AB':
            log,nm=_load_log(OUT,f'{stem}_s{seed}_{w}')
            if log is None:res['notes'].append(f'{key} world {w}: {stem}_s{seed}_{w}.json/.saved.json missing');continue
            recs=pass_records(log,NCH_DEFAULT);res['cells'][key][w]={'file':nm,'passes':{r['pass']:_stats(r) for r in recs}}
            if recs and all(r['coef'] is None for r in recs):res['notes'].append(f'{key} world {w} ({nm}): no per-chunk clip coefficient / grad norm in this log (needs log_clip / --log-clip); clip ratio and coefficient distribution unavailable')
            if recs and all(r['upd'] is None for r in recs):res['notes'].append(f'{key} world {w} ({nm}): no per-chunk update norm; mean is the cumulative-norm increment per chunk, variance unavailable')
    return res
def clip_md(ct):
    L=['\n# job 37 (native fs x1, existing background) vs R5 (job 51, fresh background): clipping and update norm per pass (existing logs only)\n']+[f'- note: {x}' for x in ct['notes']]
    L+=['\n| cell | world | pass | chunks | clip ratio (coef < 1) | update norm mean | update norm var | coef min / p10 / p25 / median / p75 / p90 / max | coef histogram (<.25, .25-.5, .5-.75, .75-<1, =1) |','|---|---|---|---|---|---|---|---|---|']
    f=lambda x,d=3:'-' if x is None else format(x,f'.{d}f')
    for key,wd in ct['cells'].items():
        for w,v in wd.items():
            for ps,s in sorted(v['passes'].items()):
                pc=s['coef_pct'];h=s['coef_hist']
                L.append(f"| {key} | {w} | {ps} | {s['n_chunks']} | {f(s['clip_ratio'])} | {f(s['upd_mean'],4)}{'' if s['upd_mean_source']=='per-chunk update_norm' else ' *'} | {f(s['upd_var'],6)} | {'-' if pc is None else ' / '.join(f(pc[str(q)],2) for q in PCT)} | {'-' if h is None else ', '.join(str(x) for x in h.values())} |")
    L.append('\n* mean = cumulative-update-norm increment of the pass / number of chunks (per-chunk norms not logged there; variance unavailable).')
    return '\n'.join(L)+'\n'
def selftest():
    tmp=Path(tempfile.mkdtemp());rng=np.random.default_rng(1);n=15;pairs=[f'p{i}' for i in range(n)]
    json.dump({'items':[{'pair':p} for p in pairs]+[{'pair':p} for p in pairs],'fam_sizes':{'L3':n,'XP':n}},open(tmp/'meta_L1_ck_s101.json','w'))
    def entry(ep,dmean,cum,can):
        base=rng.normal(0,0.05,n);alt={'L3':(dmean+base).tolist(),'XP':np.zeros(n).tolist()}
        mA={'_gen_nll':2.0+can,'_tail_nll':2.0,'_alt':alt};return {'epoch':ep,'cum_update_norm':cum,'margins':mA}
    def wr(level,tag,dm,cums,can):
        for w,sign in (('A',1),('B',-1)):
            log=[]
            for ep in range(0,5):
                e=entry(ep,0.0 if ep==0 else dm[ep-1],cums[ep],can[ep]);e['margins']['_alt']['L3']=[sign*x for x in e['margins']['_alt']['L3']];log.append(e)
            json.dump({'log':log},open(tmp/f'{level}_{tag}_s101_{w}.json','w'))
    wr('L1','ck',[0.3,0.5,0.6,0.7],[0,1,2,3,4],[0,0,0,0,0]);wr('L0reset_P1','r2p1',[0.1,0.25,0.3,0.35],[0,10,20,30,40],[0,0,0,0,0.05]);wr('L0reset_P63','r5fresh',[0.02,0.1,0.15,0.2],[0,12,25,38,50],[0,0,0,0,0.2])
    # job-37-style (entry['clip'] = per-chunk (gn, coef), cum per pass) and R5-style (steps rows) logs; world B of R5 without clip fields
    tmp2=Path(tempfile.mkdtemp())   # separate dir: in the real run the scored .json is a copy of the .saved.json, here wr() already wrote D-only .json files
    for w in 'AB':
        j37=[{'epoch':0,'margins':{}}]+[{'epoch':ps,'chunks':63,'cum_update_norm':63*0.5*ps,'clip':[(1.0+0.02*i*ps,min(1.0,1.0/(1.0+0.02*i*ps))) for i in range(63)],'margins':{}} for ps in (1,2)]
        json.dump({'log':j37},open(tmp2/f'L0fs_fskeep_s101_{w}.saved.json','w'))
        r5=[{'epoch':0,'margins':{}}]+[{'epoch':ps,'cum_update_norm':63*0.4*ps,'steps':[{'chunk':i,'update_norm':0.4+0.01*(i%5),**({'grad_norm':0.8+0.01*i,'clip_coef':min(1.0,1.0/(0.8+0.01*i))} if w=='A' else {})} for i in range(63)],'margins':{}} for ps in (1,2)]
        json.dump({'log':r5},open(tmp2/f'L0reset_P63_r5fresh_s101_{w}.saved.json','w'))
    ct=clip_table(tmp2,101);c37=ct['cells']['job37']['A']['passes'][1];assert 0<c37['clip_ratio']<1 and abs(c37['upd_mean']-0.5)<1e-9 and c37['upd_var'] is None
    c5=ct['cells']['R5']['A']['passes'][2];assert c5['upd_var']>0 and c5['clip_ratio']>=0;assert ct['cells']['R5']['B']['passes'][1]['clip_ratio'] is None and any('no per-chunk clip' in x for x in ct['notes']);print(clip_md(ct))
    r=analyse(tmp,101,nb=500);print(md_of(r))
    for e_,o_ in r['epochs'].items():assert ('D_difference_decomposition' in o_)==(o_['Q']['R5']['Q_ci'][1]<=0)   # present exactly when the R5 Q lower CI bound <= 0
    assert abs(r['epochs'][4]['Q']['R5']['Q_point']-0.2/0.7)<0.1;assert r['epochs'][4]['Q']['R5']['damage_flag'];print('SELFTEST OK')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--out',default=os.environ.get('EXP07_OUT',str(ROOT/'experiments/exp07_write_ladder')));ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    OUT=Path(a.out);res=analyse(OUT,a.seed);res['clip_table']=clip_table(OUT,a.seed);txt=md_of(res)+clip_md(res['clip_table']);(OUT/f'e35_analysis_s{a.seed}.md').write_text(txt);json.dump(res,open(OUT/f'e35_analysis_s{a.seed}.json','w'),indent=1,default=float);print(txt)
if __name__=='__main__':main()
