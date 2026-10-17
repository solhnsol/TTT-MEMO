"""s101 audit for the oth-ent distance-matching decision (numbers only, no D): per chunk with |O| > 0: |O|, number of other-pair entity (name/place) tokens in the 8192 window before the chunk,
whether a count-only match (ignoring distance) can fill |O|, and the distance-bin match fraction actually achieved."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'../exp04_what_ttt_retains'))
import numpy as np,json
import gen_world as gw,suppress as SP,s3_sets as S3
CH=S3.CH;W=S3.W;out={}
for swap in (False,True):
    w=gw.build_world(101);tab=S3.sentence_table(w,swap,gw.TOK,gw,SP);ss=S3.Sets(tab,101,1);rows=[]
    for k in range(gw.NCH):
        O=ss.oracle(k)
        if not O:continue
        k0=k*CH;lo=max(0,k0-W);cur_pairs={s['pair'] for s in ss.cur_sentences(k)}
        cand=sorted({p for s in tab if s['pair'] not in cur_pairs for p in s['names']+s['place'] if lo<=p<k0})
        oth,fr=ss.othent(k);need=len(O);rows.append(dict(k=k,nO=need,n_cand=len(cand),count_fill=len(cand)>=need,ratio=len(cand)/need,bin_match=fr,n_chosen=len(oth)))
    r=rows;ratio=np.array([x['ratio'] for x in r]);nO=np.array([x['nO'] for x in r]);nc=np.array([x['n_cand'] for x in r])
    out['B' if swap else 'A']=dict(n_chunks=len(r),median_nO=float(np.median(nO)),median_cand=float(np.median(nc)),ratio_median=float(np.median(ratio)),ratio_p10=float(np.percentile(ratio,10)),
        frac_count_fill=float(np.mean([x['count_fill'] for x in r])),mean_bin_match=float(np.mean([x['bin_match'] for x in r])),frac_chosen_equals_nO=float(np.mean([x['n_chosen']==x['nO'] for x in r])),rows=r)
    o=out['B' if swap else 'A'];print(f"world {'B' if swap else 'A'}: chunks with |O|>0: {o['n_chunks']}; median |O| {o['median_nO']:.0f}, median other-pair entity tokens in window {o['median_cand']:.0f}; candidates/|O| median {o['ratio_median']:.2f}, 10th percentile {o['ratio_p10']:.2f}; chunks where a count-only match fills |O|: {o['frac_count_fill']*100:.0f}%; mean distance-bin match {o['mean_bin_match']:.2f}; chunks where oth-ent reaches |O| tokens: {o['frac_chosen_equals_nO']*100:.0f}%")
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../experiments/exp07_write_ladder/s3_othent_audit_s101.json'),'w'),indent=1)
