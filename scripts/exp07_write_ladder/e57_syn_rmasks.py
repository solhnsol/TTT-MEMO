"""Rule masks for the density / rule search (job 1070; Hans 11/05 approved: ABS-NAT first, g = nll_band128 - nll_win; CPU only; new file).
From the W0 pass file e57_syn_s<seed>/syn_s<seed>_nll128.npz (band 128) and the eligibility of e57_closure_masks, per chunk k in [8, 128):
  qtA<T>_k  = ABS-NAT rows (g > T nats), T in --taus (default 2,4,6), key name qtA2 / qtA4 / qtA6 (kind M3TA2 / M3TA4 / M3TA6 of e57_syn_mrun.py)
  qt05_k, qt10_k = fixed top 5 / 10% rows by g128 (comparison arms), qb_TOP30_k = top 30% by g128
  qg<R>_k   = GAIN-MASS rows (rho = R/100) for --rhos (default none in round 1), qh<T>_<E>_k = RESOLVED-GATE rows (tau T, eta E/10) for --gates (default none)
All packed with CM.pk. Output npz + json with the mean share per rule. Usage: python3 e57_syn_rmasks.py --seed S --cell C --out F.npz [--taus 2,4,6] [--rhos 50,70] [--gates 2:7,4:7] | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM,e57_rules as RU
import e57_longcell      # registers the cells v3_32_1 / v3_16_1_L1024 (job 1060)
import e57_syn_shuf      # pair-shuffled place tokens when E57_SYN_SHUF is set (job 1140); otherwise no effect
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;K0=8;K1=128
def fmt(x):return ('%g'%x).replace('.','p')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--cell',default='v3_16_1');ap.add_argument('--out');ap.add_argument('--taus',default='2,4,6');ap.add_argument('--nll',default='');ap.add_argument('--rhos',default='');ap.add_argument('--gates',default='');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:RU.selftest();return
    import e55_run as R55,e55_uv as UVm,e57_syn_pass as SP
    enc,w,tf=R55.build(SP.synth_args(a.seed,a.cell),a.seed,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);Z=np.load(a.nll or EXP/f'e57_syn_s{a.seed}/syn_s{a.seed}_nll128.npz');b=list(Z['bands']).index(128)
    taus=[float(x) for x in a.taus.split(',') if x];rhos=[float(x) for x in a.rhos.split(',') if x];gates=[tuple(float(y) for y in x.split(':')) for x in a.gates.split(',') if x]
    out={};sh={};g_all=[]
    for k in range(K0,K1):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;nw=Z['nll_win'][k].astype(np.float64);g=Z['nll_band'][b][k].astype(np.float64)-nw;g_all.append(g[el])
        rules={}
        for t in taus:rules[f'qtA{fmt(t)}']=RU.abs_rule(g,el,t)
        for r in rhos:rules[f'qg{fmt(r)}']=RU.gainmass_rule(g,el,r/100)
        for t,e in gates:rules[f'qh{fmt(t)}_{fmt(e)}']=RU.gate_rule(g,nw,el,t,e/10)
        for sh_,nm in ((.05,'qt05'),(.10,'qt10'),(.15,'qt15'),(.30,'qb_TOP30')):rules[nm]=RU.top_share(g,el,sh_)
        for nm,d in rules.items():out[f'{nm}_{k}']=CM.pk(d>0);sh.setdefault(nm,[]).append(float((d>0).sum()/max(el.sum(),1)))
    out['chunks']=np.asarray([K0,K1]);np.savez_compressed(a.out,**out);G=np.concatenate(g_all)
    js=dict(seed=a.seed,cell=a.cell,band=128,mean_share={k:float(np.mean(v)) for k,v in sh.items()},min_share={k:float(np.min(v)) for k,v in sh.items()},max_share={k:float(np.max(v)) for k,v in sh.items()},g_quantiles={str(p):float(np.percentile(G,p)) for p in (5,10,25,50,75,90,95,99)})
    pathlib.Path(a.out).with_suffix('.json').write_text(json.dumps(js));print(json.dumps(js['mean_share']))
if __name__=='__main__':main()
