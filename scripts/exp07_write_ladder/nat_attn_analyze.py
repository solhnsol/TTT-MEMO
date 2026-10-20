"""E33 attention-mask cells (attn8/16/32/64) analysis (CPU). Reads {cell}_{tag}_s{stream}_{A,B}.json for L0fs (native-fs) and L0mwfs (chunk alone), both from the R run, and attn8..attn64 from the attention job (same dir), plus meta_nat_{tag}_s{stream}.json.
D(A-B) per person as nat_analyze.Dperson (E33 2.11, un-halved); readouts R1-alt (main) and R1; 95% CI = 5000 pair-cluster bootstraps (13 pairs), p = exact pair-level sign-flip permutation (all 2^13 sign patterns of the 13 pair means; the A/B label swap of a pair flips its sign).
Rows per cell and pass: D; change from W0 per world (C_A, C_B as nat_analyze); contrasts attn - native and attn - chunk-alone (person-paired; bootstrap + permutation) and the recovered fraction (attn - native) / (chunk-alone - native) at the last pass (ratio of means, pair bootstrap).
Signal columns next to every attn cell (from the per-chunk log; chunks 1..62, chunk 0 has no window): mask fraction of the whole window = sum n_masked / sum win_len; mask fraction of the candidates = sum n_masked / sum n_cand; recall of O inside the candidates = sum tp / sum n_Oc over chunks with n_Oc > 0;
enrichment = recall / (sum n_masked / sum n_cand over the same chunks) (1 = what a random mask of that size would hit).
Canary: gen / tail NLL change from W0 per cell and pass (criterion <= 0.10). Usage: nat_attn_analyze.py --dir OUT --stream 9105 [--tag nat] [--out report.md] | --selftest"""
import sys,json,argparse,itertools
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_analyze as NA
CELLS=[('L0fs','native-fs'),('L0mwfs','chunk-alone'),('attn8','attn k=8'),('attn16','attn k=16'),('attn32','attn k=32'),('attn64','attn k=64')]
def pair_means(vals,pairs):
    g={}
    for t,v in vals.items():g.setdefault(pairs[t],[]).append(v)
    return {p:float(np.mean(x)) for p,x in g.items()}
def signflip_p(vals,pairs):
    """exact two-sided p of the pair-level mean (all sign patterns of the pair means)"""
    pm=np.array(list(pair_means(vals,pairs).values()));n=len(pm);obs=abs(pm.mean())
    S=np.array(list(itertools.product([-1,1],repeat=n)),float);return float(np.mean(np.abs((S*pm).mean(1))>=obs-1e-12))
def fmtp(x,p):return f'{NA.fmt(x)} p={p:.3f}'
def signal_table(log):
    """per pass: mask fraction (window), mask fraction (candidates), recall in candidates, enrichment"""
    out=[]
    for e in log:
        if 'chunks' not in e:continue
        ch=[c for c in e['chunks'] if c['win_len']>0];nm=sum(c['n_masked'] for c in ch);wl=sum(c['win_len'] for c in ch);nc=sum(c['n_cand'] for c in ch)
        ec=[c for c in ch if c['n_Oc']>0];tp=sum(c['tp'] for c in ec);no=sum(c['n_Oc'] for c in ec);mc=sum(c['n_masked'] for c in ec);cc=sum(c['n_cand'] for c in ec)
        rec=tp/no if no else float('nan');frac_c=mc/cc if cc else float('nan')
        out.append({'pass':e['epoch'],'mask_frac_window':nm/wl if wl else float('nan'),'mask_frac_cand':nm/nc if nc else float('nan'),'O_recall_in_cand':rec,'enrichment':rec/frac_c if frac_c else float('nan'),'n_chunks_with_O':len(ec),'masked_total':nm})
    return out
def analyze(d,stream,tag='nat'):
    meta=json.load(open(Path(d)/f'meta_nat_{tag}_s{stream}.json'));pairs=NA.pair_of(meta);rng=np.random.default_rng(2);L=[];rep={}
    cells={c:(NA.load(d,c,tag,stream,'A'),NA.load(d,c,tag,stream,'B')) for c,_ in CELLS}
    have=[(c,lab) for c,lab in CELLS if cells[c][0] and cells[c][1]];missing=[c for c,_ in CELLS if c not in {h for h,_ in have}]
    if missing:L.append(f'\ncells without both worlds (skipped): {missing}\n')
    if not have:return rep,'\nno cell with both worlds\n'
    for alt,nm in ((True,'R1-alt (SQuAD tail)'),(False,'R1 (pg1342 tail)')):
        tab={};rows=[]
        for c,lab in have:
            SA=NA.states(cells[c][0]['log']);SB=NA.states(cells[c][1]['log'])
            for k in range(1,min(len(SA),len(SB))):
                D=NA.Dperson(meta,SA[k]['margins'],SB[k]['margins'],alt);tab[(c,k)]=D
                CA=NA.change(meta,cells[c][0]['log'],k,alt);CB=NA.change(meta,cells[c][1]['log'],k,alt)
                rows.append((f'{lab} pass {k}',fmtp(NA.boot(D,pairs,rng),signflip_p(D,pairs)),NA.fmt(NA.boot(CA,pairs,rng)),NA.fmt(NA.boot(CB,pairs,rng))))
        rep[nm]=rows
        L+=[f'\n### {nm}: D(A-B) per person (95% pair-cluster bootstrap CI over 13 pairs; p = exact pair-level sign-flip permutation) and change from W0 per world\n','| cell | D | C_A (change from W0, world A) | C_B (world B) |','|---|---|---|---|']+[f'| {a} | {b} | {c} | {e} |' for a,b,c,e in rows]
        kmax=max(k for (c,k) in tab);cr=[]
        for ref,rl in (('L0fs','native-fs'),('L0mwfs','chunk-alone')):
            for c,lab in have:
                if c.startswith('attn') and (ref,kmax) in tab and (c,kmax) in tab:
                    df={t:tab[(c,kmax)][t]-tab[(ref,kmax)][t] for t in tab[(c,kmax)]};cr.append((f'{lab} - {rl}, pass {kmax}',fmtp(NA.boot(df,pairs,rng),signflip_p(df,pairs))))
        if ('L0fs',kmax) in tab and ('L0mwfs',kmax) in tab:
            den={t:tab[('L0mwfs',kmax)][t]-tab[('L0fs',kmax)][t] for t in tab[('L0fs',kmax)]}
            for c,lab in have:
                if c.startswith('attn') and (c,kmax) in tab:
                    num={t:tab[(c,kmax)][t]-tab[('L0fs',kmax)][t] for t in den};cr.append((f'recovered fraction of (chunk-alone - native) by {lab}, pass {kmax} (ratio of means)',NA.fmt(NA.ratio_boot(num,den,pairs,rng))))
        L+=[f'\n### {nm}: contrasts at the last pass (person-paired, D difference)\n','| contrast | difference |','|---|---|']+[f'| {a} | {b} |' for a,b in cr]
    L+=['\n### selection signal per attn cell (world A | world B), chunks 1..62\n','| cell | world | pass | mask fraction of the whole window | mask fraction of the candidates | O recall inside the candidates | enrichment (recall / candidate mask fraction) | chunks with O |','|---|---|---|---|---|---|---|---|']
    rep['signal']={}
    for c,lab in have:
        if not c.startswith('attn'):continue
        for wi,w in enumerate('AB'):
            for r in signal_table(cells[c][wi]['log']):
                rep['signal'].setdefault(c,[]).append((w,r));L.append(f"| {lab} | {w} | {r['pass']} | {r['mask_frac_window']:.3f} | {r['mask_frac_cand']:.3f} | {r['O_recall_in_cand']:.3f} | {r['enrichment']:.2f} | {r['n_chunks_with_O']} |")
    L+=['\n### passes 2 and 3 vs pass 1: same positional mask, same evidence coverage? (flag: mask fraction of the window differs by > 0.002, O recall inside the candidates by > 0.02, or O-in-candidates counts differ in > 10% of the chunks)\n','| cell | world | pass | mask fraction diff vs pass 1 | O recall diff vs pass 1 | chunks whose O-in-candidates count differs | flag |','|---|---|---|---|---|---|---|']
    rep['pass_consistency']=[]
    for c,lab in have:
        if not c.startswith('attn'):continue
        for wi,w in enumerate('AB'):
            lg=cells[c][wi]['log'];sg=signal_table(lg);ents=[e for e in lg if 'chunks' in e]
            for r,e in list(zip(sg,ents))[1:]:
                dm=r['mask_frac_window']-sg[0]['mask_frac_window'];dr=r['O_recall_in_cand']-sg[0]['O_recall_in_cand'];nd=sum(1 for a,b in zip(ents[0]['chunks'],e['chunks']) if a['n_Oc']!=b['n_Oc']);fl=abs(dm)>0.002 or abs(dr)>0.02 or nd>0.1*len(e['chunks'])
                rep['pass_consistency'].append((c,w,r['pass'],dm,dr,nd,bool(fl)));L.append(f"| {lab} | {w} | {r['pass']} | {dm:+.4f} | {dr:+.3f} | {nd} | {'FLAG: drift changes the set or its coverage' if fl else 'ok'} |")
    L+=['\n### canary: change of NLL from W0 (gen pg11 / neutral tail pg1342), criterion <= 0.10, world A | world B\n','| cell | pass | gen A | tail A | gen B | tail B |','|---|---|---|---|---|---|']
    for c,lab in have:
        SA=NA.states(cells[c][0]['log']);SB=NA.states(cells[c][1]['log'])
        for k in range(1,min(len(SA),len(SB))):
            L.append(f"| {lab} | {k} | {SA[k]['margins']['_gen_nll']-SA[0]['margins']['_gen_nll']:+.3f} | {SA[k]['margins']['_tail_nll']-SA[0]['margins']['_tail_nll']:+.3f} | {SB[k]['margins']['_gen_nll']-SB[0]['margins']['_gen_nll']:+.3f} | {SB[k]['margins']['_tail_nll']-SB[0]['margins']['_tail_nll']:+.3f} |")
    return rep,'\n'.join(L)+'\n'
def selftest():
    import tempfile
    d=Path(tempfile.mkdtemp());rng=np.random.default_rng(1);P=13
    fam_sizes={'L3':2*P,'XP':2*P,'PS':2*P,'FN':2*P,'FP':2*P};items=[]
    for f,n in fam_sizes.items():
        for j in range(n):items.append({'level':f,'tid':j,'pair':f'p{j//2}'})
    json.dump({'items':items,'fam_sizes':fam_sizes},open(d/'meta_nat_nat_s9105.json','w'))
    w0=rng.normal(0.2,0.1,2*P)
    def mk(gains,sign,masked=None):
        log=[]
        for k,g in enumerate([0]+gains):
            m={f:rng.normal(0,0.05,n).tolist() for f,n in fam_sizes.items()};m['L3']=(w0+sign*g+rng.normal(0,0.05,2*P)).tolist();m['_alt']={f:(np.array(v)+0.01).tolist() for f,v in m.items()};m['_gen_nll']=0.5+0.01*k;m['_tail_nll']=0.6
            e={'epoch':k,'margins':m}
            if masked and k>0:e['chunks']=[{'k':j,'n_masked':int(1000*masked),'n_cand':7000,'win_len':8192,'n_O':20,'n_Oc':18 if j%2 else 0,'tp':int(18*min(1,masked*3)) if j%2 else 0} for j in range(1,63)]
            log.append(e)
        return {'log':log}
    for c,g,mk_ in (('L0fs',[0.05,0.1,0.15],None),('L0mwfs',[0.1,0.3,0.6],None),('attn8',[0.05,0.1,0.2],0.3),('attn64',[0.1,0.2,0.4],0.8)):
        json.dump(mk(g,1,mk_),open(d/f'{c}_nat_s9105_A.json','w'));json.dump(mk(g,-1,mk_),open(d/f'{c}_nat_s9105_B.json','w'))
    rep,txt=analyze(d,9105);print(txt[:3500]);assert 'passes 2 and 3 vs pass 1' in txt and 'attn k=8' in txt and 'recovered fraction' in txt and 'enrichment' in txt and 'pass 3' in txt
    st=signal_table(json.load(open(d/'attn64_nat_s9105_A.json'))['log']);assert abs(st[0]['mask_frac_window']-800/8192)<1e-9 and abs(st[0]['mask_frac_cand']-800/7000)<1e-9 and st[0]['O_recall_in_cand']>0.9
    assert abs(signflip_p({t:1.0 for t in range(2*P)},{t:f'p{t//2}' for t in range(2*P)})-2/8192)<1e-9
    print('nat_attn_analyze SELFTEST OK')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--stream',type=int,default=9105);ap.add_argument('--tag',default='nat');ap.add_argument('--out',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    rep,txt=analyze(a.dir,a.stream,a.tag);out=Path(a.out) if a.out else Path(a.dir)/f'nat_attn_report_{a.stream}.md';out.write_text(txt);json.dump(rep,open(out.with_suffix('.json'),'w'),default=str,indent=1);print(txt);print('wrote',out)
if __name__=='__main__':main()
