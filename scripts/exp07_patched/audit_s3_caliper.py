"""Caliper audit (professor 17:04): per O span (whole name/place spans of prior same-triple exposures in the window before the chunk) pick a P span (other pairs' spans in the window) with
|log d_O - log d_P| <= c without replacement (seeded order, nearest log distance), d = chunk start - span start. M_cal = matched O tokens / sum |O| (all O tokens incl. in-chunk ones);
|SMD| = |mean(log d_O) - mean(log d_P)| / pooled SD over the matched pairs (tokens-unweighted, pooled over chunks). No D. s101-103 x worlds A/B, c in {0.25, 0.5}."""
import sys,os,json,random;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'../exp04_what_ttt_retains'))
import numpy as np
import gen_world as gw,suppress as SP,s3_sets as S3
CH=S3.CH;W=S3.W;res={}
for c in (0.25,0.5):
    tot_m=0;tot_O=0;lo_all=[];lp_all=[]
    for sd in (101,102,103):
        w=gw.build_world(sd)
        for swap in (False,True):
            tab=S3.sentence_table(w,swap,gw.TOK,gw,SP);ss=S3.Sets(tab,sd,1);m=0;nO=0;lo=[];lp=[]
            for k in range(gw.NCH):
                O=ss.oracle(k);nO+=len(O)
                if not O:continue
                k0=k*CH;lo_=max(0,k0-W);cur_pairs={s['pair'] for s in ss.cur_sentences(k)}
                Osp=[sp for sp in ss.oracle_spans(k) if sp[-1]<k0]
                Psp=sorted({tuple(sp) for s in tab if s['pair'] not in cur_pairs for sp in s['spans'] if lo_<=sp[0] and sp[-1]<k0})
                rng=random.Random(f'{sd}|{swap}|{k}|cal');order=list(Osp);rng.shuffle(order);used=set()
                for sp in order:
                    dO=np.log(k0-sp[0]);best=None
                    for j,pp in enumerate(Psp):
                        if j in used:continue
                        dP=np.log(k0-pp[0]);dd=abs(dO-dP)
                        if dd<=c and (best is None or dd<best[0]):best=(dd,j,dP)
                    if best:used.add(best[1]);m+=len(sp);lo.append(dO);lp.append(best[2])
            res[f'c{c}|{sd}{"B" if swap else "A"}']=dict(M=m/nO,matched_O_tokens=m,sum_O=nO,n_pairs=len(lo));tot_m+=m;tot_O+=nO;lo_all+=lo;lp_all+=lp
            print(f"c={c} seed {sd} world {'B' if swap else 'A'}: M_cal {m/nO:.3f} ({m}/{nO}); matched span pairs {len(lo)}")
    lo_all=np.array(lo_all);lp_all=np.array(lp_all);sd_=np.sqrt((lo_all.var(ddof=1)+lp_all.var(ddof=1))/2);smd=abs(lo_all.mean()-lp_all.mean())/sd_
    res[f'c{c}|pooled']=dict(M=tot_m/tot_O,SMD=float(smd),n_pairs=len(lo_all));print(f'c={c}: pooled M_cal {tot_m/tot_O:.3f}; |SMD| of log distance {smd:.3f} ({len(lo_all)} matched span pairs) -> {"unusable (>= 0.1)" if smd>=0.1 else "usable (< 0.1)"}')
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../experiments/exp07_write_ladder/s3_caliper_audit.json'),'w'),indent=1)
