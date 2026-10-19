"""S3 stage 1 analysis (spec v1.1 + the 17:09 definitions). Usage: analyze_s3_stage1.py select   (reads s101 phase-1 files, writes OUT/s3_stage1_selection.json)
                                                    analyze_s3_stage1.py gate      (reads s102/s103 phase-2 files and applies the stage-1 gate)
epsilon(policy) = sum over answer tokens of [||p_policy - e_y|| - ||p_native - e_y||] / sum of [||p_chunkalone - e_y|| - ||p_native - e_y||]  (ratio of sums); cluster bootstrap over (world, pair)."""
import sys,json,os
from pathlib import Path
import numpy as np
OUT=Path(os.environ.get('EXP07_OUT',Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder'));rng=np.random.default_rng(0)
def load(sd,prefix='s3_stage1'):
    d={}
    for w in 'AB':
        f=OUT/f'{prefix}_s{sd}_{w}.json'
        if f.exists():d[w]=json.load(open(f))
    return d
def contrib(d,key,which):
    """per cluster (world, pair): [numerator, denominator] for policy `key` and variant `which`; chunks where the mask set is empty change nothing (policy distance = native)."""
    cl={}
    for w,r in d.items():
        for k,ch in r['chunks'].items():
            nat=np.array(ch['nat']);alone=np.array(ch['alone']);pol=ch['policies'].get(key,{}).get(which) if key!='oracle' else ch['oracle']
            pol=nat if pol is None else np.array(pol)
            for i,(j,pair) in enumerate(ch['answers']):
                c=cl.setdefault((w,pair),[0.0,0.0]);c[0]+=float(pol[i]-nat[i]);c[1]+=float(alone[i]-nat[i])
    return cl
def eps(cl,idx=None):
    v=np.array(list(cl.values()));
    if idx is not None:v=v[idx]
    return v[:,0].sum()/v[:,1].sum() if v[:,1].sum()!=0 else float('nan')
def boot(fn,n,B=8000):
    r=np.random.default_rng(1);xs=[fn(r.integers(0,n,n)) for _ in range(B)];return float(np.percentile(xs,2.5)),float(np.percentile(xs,97.5))
def select():
    d=load(101);assert set(d)=={'A','B'};out={'seed':101};keys=sorted({k for r in d.values() for ch in r['chunks'].values() for k in ch['policies']})
    # PR-AUC per aggregation (mean over chunks with labels, pooled A,B)
    pr={a:[] for a in ['A','Ahmax','B','C','ens']}
    for r in d.values():
        for ch in r['chunks'].values():
            for a,v in ch['pr'].items():
                if v is not None:pr[a].append(v)
    prm={a:float(np.mean(v)) for a,v in pr.items()};out['pr_auc']=prm;print('PR-AUC (mean over chunks):',{a:round(v,4) for a,v in prm.items()})
    single=[a for a in ['A','Ahmax','B','C']];best=max(single,key=lambda a:prm[a]);agg=best
    n=min(len(pr['ens']),len(pr[best]))
    if prm['ens']>prm[best] and n>0:
        diff=np.array(pr['ens'][:n])-np.array(pr[best][:n]);lo,hi=boot(lambda i:diff[i].mean(),n)
        if lo>0:agg='ens'
        out['ens_vs_best']=dict(best=best,diff=float(diff.mean()),ci=[lo,hi])
    out['aggregation']=agg;print('aggregation selected:',agg)
    res={}
    for key in keys:
        a,rn=key.split('|')
        if a!=agg:continue
        cl=contrib(d,key,'policy');ntok=np.mean([ch['policies'][key]['n'] for r in d.values() for ch in r['chunks'].values()]);res[rn]=dict(eps=float(eps(cl)),mean_masked=float(ntok))
    out['rules']=res;print({k:(round(v['eps'],3),round(v['mean_masked'],1)) for k,v in res.items()})
    ok=[rn for rn,v in res.items() if v['eps']>=0.8]
    if ok:rn=min(ok,key=lambda r:(res[r]['mean_masked'],r));why='fewest masked tokens among epsilon >= 0.8'
    else:rn=max(res,key=lambda r:(res[r]['eps'],-res[r]['mean_masked']));why='none reaches 0.8: max epsilon'
    out['policy']=[agg,rn];out['why']=why;print('policy selected:',agg,rn,'(',why,')')
    out['eps_oracle']=float(eps(contrib(d,'oracle','policy')));print('epsilon(oracle) on s101:',round(out['eps_oracle'],3))
    json.dump(out,open(OUT/'s3_stage1_selection.json','w'),indent=1)
def gate(prefix='s3_stage1',selfile='s3_stage1_selection.json',outname='s3_stage1_gate.json'):
    sel=json.load(open(OUT/selfile));key='|'.join(sel['policy']);rows={}
    d2=load(102,prefix);d3=load(103,prefix);d={}
    for sd,dd in ((102,d2),(103,d3)):
        for w,r in dd.items():d[f'{sd}{w}']=r
    cls={wh:contrib(d,key,wh) for wh in ('policy','rand','recency')};cls['oracle']=contrib(d,'oracle','policy');keys=sorted(cls['policy']);n=len(keys)
    vec=lambda wh:np.array([cls[wh][k] for k in keys])
    den=vec('policy')[:,1];dlo,dhi=boot(lambda i:den[i].sum(),n);print(f'{n} clusters; denominator sum {den.sum():+.3f}, CI [{dlo:+.3f},{dhi:+.3f}] ->','epsilon defined' if dlo>0 else 'epsilon NOT defined (PR-AUC fallback)')
    E=lambda wh,i=slice(None):vec(wh)[i,0].sum()/vec(wh)[i,1].sum()
    e={wh:float(E(wh)) for wh in ('policy','rand','recency','oracle')};print('epsilon:',{k:round(v,3) for k,v in e.items()})
    dr=boot(lambda i:E('policy',i)-E('rand',i),n);dc=boot(lambda i:E('policy',i)-E('recency',i),n);eo=boot(lambda i:E('policy',i),n)
    gate=dict(epsilon=e,eps_ci=eo,vs_rand=(e['policy']-e['rand'],dr),vs_recency=(e['policy']-e['recency'],dc),defined=bool(dlo>0))
    ok=e['policy']>=0.5 and dr[0]>0 and dc[0]>0
    print(f'gate: epsilon(attn-sel) {e["policy"]:.3f} >= 0.5: {e["policy"]>=0.5}; vs rand-dist {e["policy"]-e["rand"]:+.3f} CI [{dr[0]:+.3f},{dr[1]:+.3f}]; vs recency {e["policy"]-e["recency"]:+.3f} CI [{dc[0]:+.3f},{dc[1]:+.3f}] -> {"PASS" if ok and dlo>0 else "FAIL" if dlo>0 else "epsilon undefined: use the PR-AUC fallback"}')
    gate['pass']=bool(ok and dlo>0);json.dump(gate,open(OUT/outname,'w'),indent=1,default=str)
def f1sel():
    """F1-based size-rule selection (design 00:43-00:47: the registered rule of the spec is F1-max among the six rules, ties -> the rule that masks fewer tokens). Reads the phase-3 files (sets and F1 only, s101 A+B).
    F1 = micro over all eligible chunks: TP / FP / FN of the masked set against the O tokens that are selectable candidates (window before the chunk minus sinks / last 128); the all-O variant (FN counts O tokens outside the candidates) is reported too."""
    d={}
    for w in 'AB':
        f=OUT/f's3_stage1_f1_s101_{w}.json'
        if f.exists():d[w]=json.load(open(f))
    keys=sorted({k for r in d.values() for ch in r['chunks'].values() for k in ch['policies']});res={}
    for key in keys:
        tp=fp=fn=fna=tpa=0;ns=[]
        for r in d.values():
            for ch in r['chunks'].values():
                p=ch['policies'][key];q=p['prf'];tp+=q['tp'];fp+=q['fp'];fn+=q['fn'];fna+=q['fn_all'];tpa+=q['tp_all'];ns.append(p['n'])
        pr=tp/max(tp+fp,1);rc=tp/max(tp+fn,1);f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0.0;rca=tpa/max(tpa+fna,1);f1a=2*(tpa/max(tp+fp,1))*rca/((tpa/max(tp+fp,1))+rca) if tpa>0 else 0.0
        res[key]=dict(P=pr,R=rc,F1=f1,F1_allO=f1a,mean_masked=float(np.mean(ns)))
    out={'F1':res};sel=json.load(open(OUT/'s3_stage1_selection.json'))
    for agg in sorted({k.split('|')[0] for k in res}):
        rows={k.split('|')[1]:v for k,v in res.items() if k.split('|')[0]==agg};print(agg,{r:(round(v['P'],3),round(v['R'],3),round(v['F1'],3),round(v['mean_masked'])) for r,v in rows.items()})
    agg=sel['aggregation'];rows={k.split('|')[1]:v for k,v in res.items() if k.split('|')[0]==agg}
    best=max(rows,key=lambda r:(round(rows[r]['F1'],3),-rows[r]['mean_masked']));out['aggregation']=agg;out['policy']=[agg,best];out['why']='F1-max among the six size rules (3 decimals), ties -> fewer masked tokens'
    print('F1-selected policy for',agg,':',best,'(F1 %.3f)'%rows[best]['F1']);json.dump(out,open(OUT/'s3_stage1_selection_f1.json','w'),indent=1)
if __name__=='__main__':
    if sys.argv[1]=='gate' and len(sys.argv)>2 and sys.argv[2]=='f1':gate('s3_stage1_f1sel','s3_stage1_selection_f1.json','s3_stage1_gate_f1.json')
    else:{'select':select,'gate':gate,'f1':f1sel}[sys.argv[1]]()
