"""mcg (B) q,E diagnostic (PROTOCOL "Context-gain selector mcg" (B) + implementation note): selection on s101 world A only.
F1 per (q, E) = micro F1 pooled over all chunks of s101 A (TP/FP/FN summed over chunks; gold = place-token targets of the chunk);
max F1 (3 decimals), ties -> larger q, then E = 'all'. Other seeds/worlds: logged only (per-chunk mean P/R/F1 over chunks with gold,
round-wise mean g_t and r_t for place / name tokens, earlier-exposure fraction, selected-position histogram).
Usage: analyze_mcg_diag.py  (reads MCG_diag_s{101,102,103}_{A,B}.diag.json; writes mcg_diag_result.json/.md)"""
import os,sys,json
from pathlib import Path
import numpy as np
OUT=Path(os.environ.get('EXP07_OUT',Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder'))
QS=(0.05,0.10,0.20);ES=('all','64')
def f1(tp,fp,fn):p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);return p,r,(2*p*r/(p+r) if p+r>0 else 0.0)
def summarize(D):
    res={}
    for q in QS:
        for E in ES:
            key=f'{q:g}|{E}';T=np.array([c['sel'][key]['prf'] for c in D['chunks']]);tp,fp,fn=T.sum(0)
            per=[f1(*t) for t in T if t[0]+t[2]>0];hist=np.sum([c['sel'][key]['pos_hist'] for c in D['chunks']],0).tolist()
            res[key]={'micro':f1(tp,fp,fn),'per_chunk_mean':np.mean(per,0).tolist() if per else None,'pos_hist':hist}
    rw={}
    for c in D['chunks']:
        for gd in c['gold']:
            i=gd['i'];rw.setdefault(f"{gd['kind']}|r{gd['round']}",[]).append((c['g'][i],c['r'][i]))
    tok={k:{'n':len(v),'mean_g':float(np.mean([x[0] for x in v])),'mean_r':float(np.mean([x[1] for x in v]))} for k,v in sorted(rw.items())}
    allg=np.concatenate([c['g'] for c in D['chunks']]);ear=[gd['earlier_in_chunk'] for c in D['chunks'] for gd in c['gold']]
    return {'sel':res,'tokens':tok,'all_positions_mean_g':float(allg.mean()),'earlier_in_chunk_frac':float(np.mean(ear)) if ear else None,
            'q1_bit_identical':D.get('q1_bit_identical'),'q1_hash':D.get('q1_hash')}
def main():
    R={};md=['# mcg q,E diagnostic\n']
    for sd in (101,102,103):
        for w in 'AB':
            f=OUT/f'MCG_diag_s{sd}_{w}.diag.json'
            if f.exists():R[f's{sd}{w}']=summarize(json.load(open(f))['diag'])
    A=R['s101A']['sel'];cand=[(round(A[f'{q:g}|{E}']['micro'][2],3),q,E=='all',q,E) for q in QS for E in ES]
    best=max(cand);R['chosen']={'q':best[3],'E':best[4],'F1':best[0]}
    md.append(f"**Selection (s101 A, micro F1, ties -> larger q then E=all): q = {best[3]:g}, E = {best[4]} (F1 {best[0]:.3f})**\n")
    md.append('| world | '+' | '.join(f'{q:g}/{E}' for q in QS for E in ES)+' | q=1 bit identical |');md.append('|'+'---|'*(len(QS)*len(ES)+2))
    for k,v in R.items():
        if k=='chosen':continue
        md.append(f'| {k} | '+' | '.join('P{:.3f} R{:.3f} F{:.3f}'.format(*v['sel'][f'{q:g}|{E}']['micro']) for q in QS for E in ES)+f" | {v['q1_bit_identical']} |")
    json.dump(R,open(OUT/'mcg_diag_result.json','w'),indent=1);(OUT/'mcg_diag_result.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
