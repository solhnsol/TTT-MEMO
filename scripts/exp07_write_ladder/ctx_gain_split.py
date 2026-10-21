"""Split the non-O chunks of a ctx_gain json (ctx_gain.py output) by what they contain, and show which chunks the AUC used. CPU only (no model, no GPU); on the server it needs gen_world, suppress and the tokenizer (like ctx_gain.py --dry) to rebuild the sentence table.
Classes per chunk k of the 63: O (the oracle set is non-empty: a fact sentence whose person/place was exposed before inside the window), fact_first (a fact sentence has its place token in the chunk but no earlier exposure inside the window: first exposure, no O), bg_only (no fact sentence in the chunk).
Used by the AUC (ctx_gain.chunk_labels): from the first O chunk to the last chunk, O chunks vs non-O chunks; the chunks BEFORE the first O chunk are excluded because no repeat can exist there yet (the window is also shorter than 8K) and are listed separately here for information.
Per run (seed, world, weights) and class: number of chunks, min / max / mean of delta, for the used subset and for the excluded early chunks; AUC (score = -delta) of O vs fact_first and O vs bg_only.
Usage: ctx_gain_split.py JSON [--out-dir D] [--tokenizer-file F]  |  --selftest"""
import sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import ctx_gain as CG
NCH=63
def classify(o_sizes,cur_nonempty):
    return ['O' if o_sizes[k]>0 else ('fact_first' if cur_nonempty[k] else 'bg_only') for k in range(len(o_sizes))]
def stats(d,idx):
    if not idx:return None
    v=np.asarray(d)[idx];return {'n':len(idx),'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean())}
def analyze_run(delta,o_sizes,cur_nonempty):
    cls=classify(o_sizes,cur_nonempty);ks,y=CG.chunk_labels(o_sizes);used=set(ks);first=ks[0] if ks else None;d=np.asarray(delta,float);out={'first_O_chunk':first,'n_used':len(ks),'classes':{}}
    for c in ('O','fact_first','bg_only'):
        iu=[k for k in range(NCH) if cls[k]==c and k in used];ie=[k for k in range(NCH) if cls[k]==c and k not in used]
        out['classes'][c]={'used':stats(d,iu),'excluded_early':stats(d,ie),'chunks_used':iu,'chunks_excluded':ie}
    for c in ('fact_first','bg_only'):
        iO=out['classes']['O']['chunks_used'];ic=out['classes'][c]['chunks_used']
        out[f'auc_O_vs_{c}']=CG.auc(-d[iO+ic],[1]*len(iO)+[0]*len(ic)) if iO and ic else None
    return out
def md_of(res):
    L=['# ctx_gain: delta by chunk class (O / first exposure of a fact / background only)\n']
    for r,v in res.items():
        L+=[f"## {r}: first O chunk {v['first_O_chunk']}, chunks used by the AUC {v['n_used']} of {NCH}\n",'| class | subset | n | min | max | mean |','|---|---|---|---|---|---|']
        for c,x in v['classes'].items():
            for sub in ('used','excluded_early'):
                s=x[sub];L.append(f"| {c} | {'in the AUC' if sub=='used' else 'before the first O chunk (excluded)'} | "+(' | '.join([str(s['n']),f"{s['min']:+.3f}",f"{s['max']:+.3f}",f"{s['mean']:+.3f}"]) if s else '0 | - | - | -')+' |')
        for c in ('fact_first','bg_only'):
            a=v.get(f'auc_O_vs_{c}');L.append(f"\nAUC O vs {c}: "+('-' if not a else f'{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]'))
        L.append('')
    return '\n'.join(L)+'\n'
def run(a):
    for p in ('../exp05_write_dynamics','../exp04_what_ttt_retains'):sys.path.insert(0,str(HERE/p))
    import nat_dryrun as D,gen_world as gw
    tk=D.load_tok(a.tokenizer_file);assert tk is not None,'real tokenizer file not found';j=json.load(open(a.json));res={};cache={}
    for name,r in j['runs'].items():
        seed=int(name.split('_')[0][1:]);wd=name.split('_')[1]
        if (seed,wd) not in cache:
            w=gw.build_world(seed,v=2);osz,sets=CG.o_sizes_of(w,wd=='B',tk,seed);cache[(seed,wd)]=(osz,[bool(sets.cur_sentences(k)) for k in range(NCH)])
        osz,cur=cache[(seed,wd)]
        if osz!=r['n_O_tokens']:print('WARNING: rebuilt |O| per chunk differs from the json for',name)
        res[name]=analyze_run(r['delta'],r['n_O_tokens'],cur)
    return res
def selftest():
    rng=np.random.default_rng(0);osz=[0]*18+[3]*2+[0]*43;osz[30]=2;osz[40]=5;cur=[False]*63
    for k in (5,19,18,25,30,40,41,50):cur[k]=True
    d=rng.normal(-0.2,0.02,NCH);d[[18,19,30,40]]=-1.1;d[[25,41]]=-0.3;r=analyze_run(d,osz,cur)
    assert r['first_O_chunk']==18 and r['n_used']==45 and r['classes']['O']['used']['n']==4 and r['classes']['fact_first']['used']['n']==3 and r['classes']['bg_only']['used']['n']==38
    assert r['classes']['fact_first']['excluded_early']['n']==1 and r['classes']['O']['excluded_early'] is None and r['auc_O_vs_fact_first'][0]==1.0
    print('ctx_gain_split SELFTEST OK',r['n_used'],{c:v['used']['n'] for c,v in r['classes'].items()});print(md_of({'selftest':r})[:600])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('json',nargs='?');ap.add_argument('--out-dir',default='/tmp/ctx_gain_split');ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    res=run(a);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);json.dump(res,open(od/'ctx_gain_split.json','w'),indent=1);md=md_of(res);(od/'ctx_gain_split.md').write_text(md);print(md)
