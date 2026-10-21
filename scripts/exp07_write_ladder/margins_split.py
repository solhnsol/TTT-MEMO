"""E47: pure numpy helpers for Ladder.evaluate (no jax): split_rows turns the packed row log-probs (even rows = gold span, odd rows = foil span, per item in family order) into the margins (gold - foil, exactly the arithmetic that ladder.evaluate always used) AND the gold / foil span log-probs themselves; regold_check compares a re-scored log with the original one."""
import numpy as np
def split_rows(mg,ma,fam_sizes):
    """mg, ma: 1-D arrays of the row scores of the primary (pg1342 tail) and the alt (SQuAD tail) readout, 2 rows per item (gold, foil). fam_sizes: ordered {family: n_items}. -> dict with the old fields (family margins, '_alt' = alt margins) and the new additive ones ('_gold', '_foil', '_alt_gold', '_alt_foil'); the caller adds _gen_nll / _tail_nll."""
    mg=np.asarray(mg);ma=np.asarray(ma);mgg,mgf=mg[0::2],mg[1::2];mag,maf=ma[0::2],ma[1::2];mgm=mgg-mgf;mam=mag-maf
    out={};outg={};outf={};alt={};altg={};altf={};i=0
    for f,n in fam_sizes.items():
        out[f]=mgm[i:i+n].tolist();outg[f]=mgg[i:i+n].tolist();outf[f]=mgf[i:i+n].tolist();alt[f]=mam[i:i+n].tolist();altg[f]=mag[i:i+n].tolist();altf[f]=maf[i:i+n].tolist();i+=n
    assert i==len(mgm)==len(mam),(i,len(mgm),len(mam))
    out['_alt']=alt;out['_gold']=outg;out['_foil']=outf;out['_alt_gold']=altg;out['_alt_foil']=altf;return out
def regold_check(old_log,new_log):
    """max |new - old| over every margin family (primary and '_alt') per epoch / state key present in both logs; {key: max_abs_diff}; None entries are skipped"""
    def ent(log):return {e.get('epoch',e.get('state')):e['margins'] for e in log if isinstance(e.get('margins'),dict) and ('epoch' in e or 'state' in e) and 'L3' in e['margins']}
    o,n=ent(old_log),ent(new_log);res={}
    for k in sorted(set(o)&set(n),key=str):
        d=[]
        for f in o[k]:
            if f.startswith('_') or f not in n[k]:continue
            d.append(np.max(np.abs(np.asarray(o[k][f])-np.asarray(n[k][f]))))
        for f in o[k].get('_alt',{}):
            if f in n[k].get('_alt',{}):d.append(np.max(np.abs(np.asarray(o[k]['_alt'][f])-np.asarray(n[k]['_alt'][f]))))
        res[str(k)]=float(max(d)) if d else None
    return res
