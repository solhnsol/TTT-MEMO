"""E57 synthetic world: are the rows with a high window gain fact rows? (CPU only; new file; numbers only). Inputs: the W0 pass files of e57_syn_pass.py (syn_s<seed>_nll.npz: nll_win, nll_chunk, nll_band 256/512 [128 chunks]) and the fact labels of the world (e55_sentdump_labels.labels: fact id of a sentence = majority fact of its tokens, share >= .5, -1 background). A row (eligible: loss row whose target token i+1 is inside a sentence; chunks 8..127) is a FACT row if the sentence of its target token has fact id >= 0 (base rate printed). Gains: window gain = nll_chunk - nll_win, band gain b = nll_band_b - nll_win (b = 256, 512), W0. For the top 0.1 / 0.3 / 1 / 3 / 10 / 30 / 100 % rows of each gain: rows, share of fact rows, share of the gain sum on fact rows, enrichment = share / base rate; per seed and pooled over the seeds. Also per fact: rows, mean gain (window / band 512), rows in the top 1 % of the band-512 gain, ranked by mean band-512 gain.
  .venv/bin/python scripts/exp07_write_ladder/e57_syn_gain_labels.py --seeds 62:D62,63:D63,64:D64 --out O.json --npz R.npz"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024;FR=[.001,.003,.01,.03,.10,.30,1.0]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds');ap.add_argument('--out');ap.add_argument('--npz');a=ap.parse_args()
    import e55_run as R55,e55_uv as UVm,e55_sentdump_labels as LB
    import e57_syn_pass as SP
    per={};store={}
    for spec in a.seeds.split(','):
        sd,d=spec.split(':');sd=int(sd);enc,w,tf=R55.build(SP.synth_args(sd),sd,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids)
        Z=np.load(pathlib.Path(d)/f'syn_s{sd}_nll.npz');bl=list(Z['bands']);rows=dict(chunk=[],row=[],fact=[],isfact=[],g_win=[],g256=[],g512=[])
        for k in range(8,128):
            q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
            for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
            lb=LB.labels(w['chunks'],ends,k);qf=np.asarray(lb['q_fact']);el=CM.rows_eligible(Z['lm'][k],sid)&Z['lm'][k];r=np.nonzero(el)[0]
            tsid=sid[r+1];fact=np.where(tsid>=0,qf[np.clip(tsid,0,len(q)-1)],-1)
            rows['chunk'].append(np.full(len(r),k));rows['row'].append(r);rows['fact'].append(fact);rows['isfact'].append(fact>=0)
            rows['g_win'].append((Z['nll_chunk'][k]-Z['nll_win'][k])[r]);rows['g256'].append((Z['nll_band'][bl.index(256)][k]-Z['nll_win'][k])[r]);rows['g512'].append((Z['nll_band'][bl.index(512)][k]-Z['nll_win'][k])[r])
        per[sd]={k:np.concatenate(v) for k,v in rows.items()};store.update({f's{sd}_{k}':v for k,v in per[sd].items()})
    np.savez_compressed(a.npz,**store)
    def analyse(R):
        n=len(R['isfact']);base=float(R['isfact'].mean());out={'rows':n,'fact_row_base_rate':base,'gains':{}}
        for gk,gname in (('g_win','window gain (chunk - win)'),('g256','band256 gain'),('g512','band512 gain')):
            g=R[gk].astype(np.float64);order=np.argsort(-g,kind='stable');res={}
            for f in FR:
                m=np.zeros(n,bool);m[order[:max(1,int(round(f*n)))]]=True;gs=g[m].sum();s=m&R['isfact']
                res[f'{f*100:g}%']=dict(rows=int(m.sum()),mean_gain=float(g[m].mean()),fact_row_share=float(s.sum()/m.sum()),fact_gain_sum_share=float(g[s].sum()/gs) if gs!=0 else None,enrichment=float(s.sum()/m.sum()/base))
            out['gains'][gname]=res
        return out
    res={'definition':'FACT row = target token in a sentence with fact id >= 0; gains W0; chunks 8..127','seeds':{str(sd):analyse(R) for sd,R in per.items()}}
    pooled={k:np.concatenate([R[k] for R in per.values()]) for k in per[next(iter(per))]};res['pooled']=analyse(pooled)
    pf={}
    for sd,R in per.items():
        g=R['g512'];order=np.argsort(-g,kind='stable');t1=np.zeros(len(g),bool);t1[order[:int(round(.01*len(g)))]]=True;fs={}
        for f in np.unique(R['fact'][R['fact']>=0]):
            m=R['fact']==f;fs[int(f)]=dict(rows=int(m.sum()),mean_g512=float(g[m].mean()),mean_gwin=float(R['g_win'][m].mean()),in_top1pct=int((m&t1).sum()))
        rk=sorted(fs,key=lambda f:-fs[f]['mean_g512']);
        for i,f in enumerate(rk):fs[f]['rank_by_mean_g512']=i+1
        pf[str(sd)]=fs
    res['per_fact']=pf;json.dump(res,open(a.out,'w'),indent=1)
    for sd,v in list(res['seeds'].items())+[('pooled',res['pooled'])]:
        print('== seed',sd,'rows',v['rows'],'fact-row base rate %.3f'%v['fact_row_base_rate'])
        for gn,r in v['gains'].items():print('  ',gn,' | '.join(f"{k}: {x['fact_row_share']:.3f} (x{x['enrichment']:.2f}, gain {x['fact_gain_sum_share'] if x['fact_gain_sum_share'] is None else round(x['fact_gain_sum_share'],3)})" for k,x in r.items()))
if __name__=='__main__':main()
