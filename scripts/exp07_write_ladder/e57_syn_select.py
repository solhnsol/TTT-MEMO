"""E57 synthetic world: pool the candidate counts of the selection seeds (62, 63), apply the selection rule of the label-free-mask thread and tabulate the test seed (64) (CPU only; new file; numbers only). Rule per (head set, band): among the candidates with recall >= --recall (default .95) on the pooled selection counts the one with the least masking ('min') and the one whose masking is closest to 2 x 'min' ('x2'); if no candidate passes: the highest-recall candidate ('min', fallback = true) and, among the candidates with recall >= 0.9 x that recall and more than 1.2 x its masking, the one closest to 2 x ('x2'). recall = hidden answer sentences / answer sentences, precision = hidden answer sentences / sentences judged hidden, F1, masking = selected / far keys (mean over the GAINQ rows); also for all eligible rows.
  .venv/bin/python scripts/exp07_write_ladder/e57_syn_select.py --sel D62/syn_s62_counts_b512.json,D63/syn_s63_counts_b512.json --test D64/syn_s64_counts_b512.json --out O.json"""
import json,argparse
def pool(files):
    tot={}
    for f in files:
        d=json.load(open(f))['counts']
        for k,v in d.items():
            t=tot.setdefault(k,{x:0 for x in v})
            for x in v:t[x]+=v[x]
    return tot
def met(v,all_rows=False):
    n,h,j=(v['n_all'],v['h_all'],v['j_all']) if all_rows else (v['n'],v['h'],v['j']);rec=h/n if n else float('nan');prec=h/j if j else float('nan');f1=2*rec*prec/(rec+prec) if rec+prec>0 else float('nan')
    return dict(recall=rec,precision=prec,f1=f1,mask=v['sel']/max(v['rows'],1),saturated=v['sat']/max(v['rows'],1),answer_sentences=n)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sel');ap.add_argument('--test');ap.add_argument('--out');ap.add_argument('--recall',type=float,default=.95);a=ap.parse_args()
    S=pool(a.sel.split(','));T=pool(a.test.split(','));ms={k:met(v) for k,v in S.items()};mt={k:met(v) for k,v in T.items()};chosen={}
    for nm in sorted({k.split('|')[0] for k in S}):
        keys=[k for k in S if k.startswith(nm+'|')];ok=[k for k in keys if ms[k]['recall']>=a.recall]
        if ok:
            kmin=min(ok,key=lambda k:ms[k]['mask']);chosen[f'{nm}|min']=dict(cand=kmin,fallback=False);tgt=2*ms[kmin]['mask'];chosen[f'{nm}|x2']=dict(cand=min(ok,key=lambda k:abs(ms[k]['mask']-tgt)),fallback=False)
        else:
            kmin=max(keys,key=lambda k:ms[k]['recall']);chosen[f'{nm}|min']=dict(cand=kmin,fallback=True);r0=ms[kmin]['recall'];m0=ms[kmin]['mask'];c2=[k for k in keys if ms[k]['recall']>=.9*r0 and ms[k]['mask']>1.2*m0]
            if c2:chosen[f'{nm}|x2']=dict(cand=min(c2,key=lambda k:abs(ms[k]['mask']-2*m0)),fallback=True)
    for c in chosen.values():c['selection']=ms[c['cand']];c['selection_all_rows']=met(S[c['cand']],True);c['test']=mt[c['cand']];c['test_all_rows']=met(T[c['cand']],True)
    best=sorted(ms.items(),key=lambda x:-x[1]['recall'])[:8]
    json.dump(dict(recall_min=a.recall,passing_candidates=sum(1 for v in ms.values() if v['recall']>=a.recall),candidates=len(ms),chosen=chosen,top_recall_selection=dict(best),selection_all={k:dict(selection=ms[k],test=mt[k]) for k in ms}),open(a.out,'w'),indent=1,default=float)
    print('passing',sum(1 for v in ms.values() if v['recall']>=a.recall),'of',len(ms))
    for k,c in chosen.items():print(f"{k:14s} {c['cand']:18s} fb={c['fallback']} sel rec {c['selection']['recall']:.3f} prec {c['selection']['precision']:.3f} f1 {c['selection']['f1']:.3f} mask {c['selection']['mask']:.4f} | test rec {c['test']['recall']:.3f} prec {c['test']['precision']:.3f} f1 {c['test']['f1']:.3f} mask {c['test']['mask']:.4f}")
if __name__=='__main__':main()
