"""E33 2.13 analysis (CPU) of nat_signal.py output. Per k: mask fraction (masked / candidates, all chunks and O chunks), O recall (pooled masked O / selectable O), O precision and enrichment (precision / mask fraction of the O chunks),
background share and same-person-unit share of the masked tokens, epsilon (E25: sum[(d_policy - d_nat)] / sum[(d_alone - d_nat)] over answer tokens) for attn-sel, rand-dist, recency with a chunk-cluster bootstrap, and
eps(attn) - eps(rand), eps(attn) - eps(recency) with 95% CIs. K* check against the registered criteria (E33 2.13): mask fraction 5-60%, O recall >= 0.8, enrichment >= 1.5, eps >= 0.5 and both CI lower bounds > 0.
Usage: nat_signal_analyze.py MAIN.json[,CHECK.json] [--out report.md] | --selftest   (several files: the first is the main stream, a pooled table is added)"""
import sys,json,argparse
from pathlib import Path
import numpy as np
NB=4000
def eps(rows,key,idx):
    num=den=0.0
    for r in idx:
        c=rows[r]
        if key not in c:continue
        n=np.asarray(c['nat']);num+=float((np.asarray(c[key])-n).sum());den+=float((np.asarray(c['alone'])-n).sum())
    return num/den if abs(den)>1e-12 else float('nan')
def merge(rs):
    """pool the chunks of several streams (each chunk is its own cluster)"""
    m={'ks':rs[0]['ks'],'chunks':{},'stream':'+'.join(str(r.get('stream')) for r in rs),'layout':'pooled','swap':False}
    for i,r in enumerate(rs):
        for k,c in r['chunks'].items():m['chunks'][f'{i}:{k}']=c
    return m
def analyze(res):
    ks=res['ks'];ch=res['chunks'];cs=[ch[kk] for kk in sorted(ch,key=lambda x:(str(x).split(':')[0],int(str(x).split(':')[-1])))];out={'ks':{},'n_chunks':len(cs),'n_chunks_with_O':sum(1 for c in cs if c['n_O']>0),'n_chunks_eligible':sum(1 for c in cs if c['n_ans']>0),'n_answer_tokens':sum(c['n_ans'] for c in cs)}
    el=[i for i,c in enumerate(cs) if c['n_ans']>0 and 'nat' in c];rng=np.random.default_rng(213)
    # per-k eligible rows: dict with nat, alone, pol, rand, rec
    for kk in ks:
        pk=lambda c:c['per_k'][str(kk)] if str(kk) in c['per_k'] else c['per_k'][kk]
        rows=[]
        for i in el:
            d=pk(cs[i]);rows.append({'nat':cs[i]['nat'],'alone':cs[i]['alone'],**({'pol':d['pol'],'rand':d['rand'],'rec':d['rec']} if 'pol' in d else {})})
        n=sum(pk(c)['n'] for c in cs);nc=sum(c['n_cand'] for c in cs);co=[c for c in cs if c['n_Oc']>0];nO=sum(c['n_Oc'] for c in co);tp=sum(pk(c)['tp'] for c in co);nm_o=sum(pk(c)['n'] for c in co);nc_o=sum(c['n_cand'] for c in co)
        frac_o=nm_o/max(nc_o,1);prec=tp/max(nm_o,1)
        d={'mask_frac_all':n/max(nc,1),'mask_frac_Ochunks':frac_o,'O_recall':tp/max(nO,1),'O_precision':prec,'enrichment':(prec/(nO/max(nc_o,1))) if nO>0 else float('nan'),'precision_base_rate':nO/max(nc_o,1),'bg_share':sum(pk(c)['bg'] for c in cs)/max(n,1),'same_person_share':sum(pk(c)['same'] for c in cs)/max(n,1),'other_expo_share':sum(pk(c)['other'] for c in cs)/max(n,1),'n_chunks_O':len(co),'n_O_selectable':nO,'n_eligible_chunks':len(el),'n_answer_tokens':out['n_answer_tokens']}
        ok=[r for r in rows if 'pol' in r];idx=list(range(len(ok)))
        if ok:
            pt={m:eps(ok,m,idx) for m in ('pol','rand','rec')};bs=[]
            for _ in range(NB):
                j=rng.integers(0,len(ok),len(ok));bs.append([eps(ok,m,j) for m in ('pol','rand','rec')])
            bs=np.asarray(bs);ci=lambda x:[float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5))]
            d.update({'eps_attn':pt['pol'],'eps_attn_ci':ci(bs[:,0]),'eps_rand':pt['rand'],'eps_rand_ci':ci(bs[:,1]),'eps_recency':pt['rec'],'eps_recency_ci':ci(bs[:,2]),'diff_rand':pt['pol']-pt['rand'],'diff_rand_ci':ci(bs[:,0]-bs[:,1]),'diff_recency':pt['pol']-pt['rec'],'diff_recency_ci':ci(bs[:,0]-bs[:,2]),'n_eps_chunks':len(ok)})
        co_all=[c for c in cs if c['n_O']>0];d['O_recall_upper_bound']=sum(c['n_Oc'] for c in co_all)/max(sum(c['n_O'] for c in co_all),1);d['n_chunks_O_all']=len(co_all)
        nel=out['n_chunks_eligible'];und=nel<15
        c4a=None if (und or not ok) else bool(d['eps_attn']>=0.5 and d['diff_rand_ci'][0]>0)
        d['criteria']={'c1_mask_5_60pct':0.05<=d['mask_frac_all']<=0.60,'c2_O_recall_cand_ge_0.8':d['O_recall']>=0.8,'c3_enrichment_ge_1.5':d['enrichment']>=1.5,'c4a_eps_ge_0.5_and_attn_minus_rand_ci_lo_gt_0':c4a}   # c4b (attn - recency) is reported only
        d['c4b_report_only']={'diff_recency':d.get('diff_recency'),'ci':d.get('diff_recency_ci')}
        d['pass_tau0.8']=bool(d['O_recall']>=0.8);d['pass_tau0.6']=bool(d['O_recall']>=0.6);d['in_Kstar']=d['pass_tau0.8'];out['ks'][kk]=d   # E33 2.13 (Hans 10/20): K* = candidate O recall >= tau only (0.8 main, 0.6 sensitivity); c1, c3, c4a, c4b are report columns, not gates
    out['Kstar']=[kk for kk in ks if out['ks'][kk]['pass_tau0.8']];out['Kstar_tau0.6']=[kk for kk in ks if out['ks'][kk]['pass_tau0.6']]
    ps=out['Kstar'];sel=[]
    if ps:
        sel=[ps[0]]
        mid=ps[len(ps)//2]
        if mid not in sel:sel.append(mid)
        if 256 in ps and 256 not in sel:sel.append(256)
    out['comparison_k_candidates']=sel[:3];return out
def report(res,out):
    L=[f"# E33 2.13 k-resolved attention signal, stream {res.get('stream')} {res.get('layout')} world {'B' if res.get('swap') else 'A'}\n",f"chunks {out['n_chunks']}, with O {out['n_chunks_with_O']}, epsilon-eligible {out['n_chunks_eligible']}, answer tokens {out['n_answer_tokens']}\n",
       '| k | c1 mask frac (all) | mask frac (O chunks) | c2 O recall (candidates) | recall upper bound | O precision | c3 enrichment | background share | same-person unit share | c4a eps attn [CI] | eps rand | c4a attn-rand [CI] | c4b attn-recency [CI] (report only) | eligible chunks / answer tokens | passes tau |','|'+'---|'*15]
    f=lambda x:f'{x:.3f}';c=lambda t,ci:f'{f(t)} [{f(ci[0])}, {f(ci[1])}]'
    for kk,d in out['ks'].items():
        e=('| '+' | '.join([c(d['eps_attn'],d['eps_attn_ci']),f(d['eps_rand']),c(d['diff_rand'],d['diff_rand_ci']),c(d['diff_recency'],d['diff_recency_ci'])])) if 'eps_attn' in d else '| n/a | n/a | n/a | n/a'
        ks_=('0.8' if d['pass_tau0.8'] else ('0.6' if d['pass_tau0.6'] else 'no'))
        L.append(f"| {kk} | {f(d['mask_frac_all'])} | {f(d['mask_frac_Ochunks'])} | {f(d['O_recall'])} | {f(d['O_recall_upper_bound'])} | {f(d['O_precision'])} | {f(d['enrichment'])} | {f(d['bg_share'])} | {f(d['same_person_share'])} {e} | {d['n_eligible_chunks']} / {d['n_answer_tokens']} | {ks_} |")
    sm=lambda lst:', '.join(f"k{kk} (mask {out['ks'][kk]['mask_frac_all']*100:.0f}%)" for kk in lst) or 'none'
    L.append(f"\n**passing k, tau = 0.8: {sm(out['Kstar'])}**\n**passing k, tau = 0.6: {sm(out['Kstar_tau0.6'])}**\n\ncomparison-k candidates (smallest passing k, middle passing k, k256 if passing; at most 3; a person confirms): {out['comparison_k_candidates'] or 'none'}\n")
    L.append(f"K* = candidate O recall >= tau only (E33 2.13, Hans 10/20); mask fraction, enrichment, background share, eps columns are reports, not gates. Recall / precision / enrichment use chunks with selectable O (n_O_cand > 0, {out['ks'][res['ks'][0]]['n_chunks_O']} of {out['ks'][res['ks'][0]]['n_chunks_O_all']} chunks with O), upper bound = sum n_O_cand / sum n_O; eps CIs are chunk-cluster bootstrap ({NB} resamples) and 'undetermined' as a statement below 15 eligible chunks ({out['n_chunks_eligible']} here).")
    return '\n'.join(L)+'\n'
def selftest():
    rng=np.random.default_rng(1);ks=[4,16,64];chunks={}
    for k in range(10):
        e=k>=4;c={'k':k,'n_cand':8000,'n_O':5 if e else 0,'n_Oc':4 if e else 0,'n_ans':2 if e else 0,'per_k':{}}
        if e:c['nat']=[0.2,0.3];c['alone']=[1.0,1.1]
        for kk in ks:
            d={'n':kk*20,'tp':min(4,kk//8) if e else 0,'bg':kk*18,'same':kk,'other':kk}
            if e:d['pol']=[0.2+0.8*min(1,kk/64)+rng.normal(0,.01),0.3+0.8*min(1,kk/64)];d['rand']=[0.25,0.35];d['rec']=[0.22,0.32]
            c['per_k'][kk]=d
        chunks[k]=c
    res={'ks':ks,'chunks':chunks,'stream':1,'layout':'x','swap':False};out=analyze(res);md=report(res,out);assert abs(out['ks'][64]['enrichment']-out['ks'][64]['O_recall']/out['ks'][64]['mask_frac_Ochunks'])<1e-9;assert out['ks'][64]['eps_attn']>0.9 and out['ks'][4]['eps_attn']<out['ks'][64]['eps_attn'];print(md[-600:]);print('nat_signal_analyze SELFTEST OK')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('file',nargs='?');ap.add_argument('--out',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    files=a.file.split(',');rs=[json.load(open(f)) for f in files];assert all('chunks' in r for r in rs),'not a raw nat_signal result (no chunks key; an analysis json?)';res=rs[0];out=analyze(res);txt=report(res,out)
    if len(rs)>1:
        pres=merge(rs);pout=analyze(pres);txt+='\n## pooled over '+', '.join(files)+' (eligible chunks of all streams, each chunk a cluster; for a CI only, K* is judged on the main stream)\n\n'+report(pres,pout)
    p=Path(a.out or Path(files[0]).with_suffix('.md'));p.write_text(txt);json.dump(out,open(p.with_name(p.stem+'_analysis.json'),'w'),indent=1,default=str);print(txt)
