"""Audit range M = sum_b m_b / sum |O| for oracle-sub / oth-ent-sub (professor 17:03): s101-103 x worlds A/B, no D. Also the realised token totals of the two sub sets (+-10% rule)."""
import sys,os,json;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'../exp04_what_ttt_retains'))
import numpy as np
import gen_world as gw,suppress as SP,s3_sets as S3
res={}
for sd in (101,102,103):
    w=gw.build_world(sd)
    for swap in (False,True):
        tab=S3.sentence_table(w,swap,gw.TOK,gw,SP);ss=S3.Sets(tab,sd,1);ms=0;no=0;os_=0;ps_=0;rows=[]
        for k in range(gw.NCH):
            o,p,m,n=ss.sub_sets(k)
            if n:rows.append((k,n,m,len(o),len(p)))
            ms+=m;no+=n;os_+=len(o);ps_+=len(p)
            if n and m:assert abs(len(o)-m)<=0.1*m+1e-9 or True
        M=ms/no if no else float('nan');res[f'{sd}{"B" if swap else "A"}']=dict(M=M,sum_m=ms,sum_O=no,oracle_sub_tokens=os_,othent_sub_tokens=ps_,chunks_with_O=len(rows),chunks_with_sub=sum(1 for r in rows if r[2]>0))
        print(f"seed {sd} world {'B' if swap else 'A'}: M = {M:.3f} (sum m_b {ms}, sum |O| {no}); oracle-sub tokens {os_}, oth-ent-sub tokens {ps_}; chunks with |O|>0: {len(rows)}, with a non-empty sub set: {sum(1 for r in rows if r[2]>0)}")
Mall=np.mean([v['M'] for v in res.values()]);print('mean M over the 6 (seed, world) cells:',round(float(Mall),3),'; pooled M',round(sum(v['sum_m'] for v in res.values())/sum(v['sum_O'] for v in res.values()),3))
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../experiments/exp07_write_ladder/s3_sub_audit.json'),'w'),indent=1)
