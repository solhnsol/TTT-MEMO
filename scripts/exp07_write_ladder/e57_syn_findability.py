"""E57 synthetic world: which per-row measure best predicts 'fact row' (CPU only; new file; numbers only). Rows = the eligible rows of the seeds 62 / 63 / 64, chunks 8..127 (e57_syn_gain_labels_rows.npz); fact row = target token inside a sentence with fact id >= 0. Measures per row:
  g_win, g256, g512        window gain nll_chunk - nll_win, band gains nll_band_b - nll_win (W0)
  m1/m2/m4 far / band      the target token alone (m1), query token + target (m2), previous 3 tokens incl. the query token + target (m4) occurred earlier at distance q-(j-1) in [512, 8191] (far) / [1, 511] (band) in the stream (indicator)
  far_mass                 attention mass of the query on the far window keys (all heads, key-mass dump of e57_syn_pass, band 512: 'far' = summed over the layers and heads)
  top1_mass, top128_mass   mass of the best far key / of the top-128 stored far keys; top1_share = top1 / far
  factkey_share            share of the top-128 stored far mass held by keys inside fact sentences (fact id >= 0)
  samefact_share           share held by keys inside sentences of the SAME fact as the row's own target sentence (fact rows only; NaN otherwise)
  (the T_max per head group of the news dump is not stored in the synthetic dump)
Tables: AUC for 'fact row' (Mann-Whitney, ties averaged), precision and recall of the top 1 / 3 / 10 % rows of each measure, Spearman correlation between the measures (on a random 100k-row subsample for speed unless --all), Jaccard overlap of the top-1 % and top-10 % sets.
  .venv/bin/python scripts/exp07_write_ladder/e57_syn_findability.py --seeds 62:D62,63:D63,64:D64 --rows R.npz --out O.json --npz P.npz"""
import sys,json,argparse,pathlib,collections
import numpy as np
from scipy.stats import rankdata
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;SHARD=16
def auc(score,y):
    s=np.asarray(score,float);m=~np.isnan(s);s=s[m];y=np.asarray(y)[m]
    if y.sum()==0 or (~y).sum()==0:return None
    r=rankdata(s);return float((r[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds');ap.add_argument('--rows');ap.add_argument('--out');ap.add_argument('--npz');ap.add_argument('--all',action='store_true');a=ap.parse_args()
    import e55_run as R55,e55_uv as UVm,e55_sentdump as SD,e55_sentdump_labels as LB
    import e57_syn_pass as SP
    Rz=np.load(a.rows);cols=collections.defaultdict(list)
    for spec in a.seeds.split(','):
        sd,d=spec.split(':');sd=int(sd);d=pathlib.Path(d)
        enc,w,tf=R55.build(SP.synth_args(sd),sd,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);N=len(ids)
        ch=Rz[f's{sd}_chunk'];row=Rz[f's{sd}_row'];fact=Rz[f's{sd}_fact'];isf=Rz[f's{sd}_isfact']
        Dm={}
        for nn in (1,2,4):
            D=collections.defaultdict(list)
            for j in range(nn-1,N):D[tuple(int(x) for x in ids[j-nn+1:j+1])].append(j)
            Dm[nn]={k:np.asarray(v) for k,v in D.items()}
        q=ch*CH+row;res={k:np.zeros(len(q)) for k in ('m1_far','m1_band','m2_far','m2_band','m4_far','m4_band')}
        for r in range(len(q)):
            qq=int(q[r])
            for nn in (1,2,4):
                key=tuple(int(x) for x in ids[qq+1-nn+1:qq+2]);js=Dm[nn].get(key)
                if js is None:continue
                js=js[js<=qq]
                res[f'm{nn}_far'][r]=bool(((js<=qq-511)&(js>=qq-8190)).any());res[f'm{nn}_band'][r]=bool((js>=qq-510).any())
        far=np.zeros(len(q));t1=np.zeros(len(q));t128=np.zeros(len(q));fk=np.full(len(q),np.nan);sf=np.full(len(q),np.nan);cache={};lastk=-1
        for k in np.unique(ch):
            m=np.nonzero(ch==k)[0];rr=row[m];j=k//SHARD
            if j not in cache:cache.clear();cache[j]=np.load(d/f'syn_s{sd}_dump_j{j:02d}.npz')
            Z=cache[j];P=f'c{k}/ALL_b512_';idx=Z[P+'idx'][rr].astype(np.int64);mass=Z[P+'score'][rr].astype(np.float64);fr=Z[P+'far'][rr].astype(np.float64)
            lb=LB.labels(w['chunks'],ends,k);wfact=np.asarray(lb['w_fact']);wsid,wst,wlen=UVm.window_sents(ends,k,CH,8);ok=idx>=0;sidk=np.where(ok,wsid[np.clip(idx,0,8191)],-1);kf=np.where(sidk>=0,wfact[np.clip(sidk,0,len(wfact)-1)],-1)
            far[m]=fr;t1[m]=mass[:,0];t128[m]=mass.sum(1);tot=np.maximum(mass.sum(1),1e-30)
            fk[m]=(mass*(kf>=0)).sum(1)/tot;same=(kf==fact[m][:,None])&(fact[m][:,None]>=0);s_=(mass*same).sum(1)/tot;sf[m]=np.where(fact[m]>=0,s_,np.nan)
        for kk,v in (('chunk',ch),('row',row),('fact',fact),('is_fact',isf),('g_win',Rz[f's{sd}_g_win']),('g256',Rz[f's{sd}_g256']),('g512',Rz[f's{sd}_g512']),('far_mass',far),('top1_mass',t1),('top128_mass',t128),('top1_share',t1/np.maximum(far,1e-30)),('factkey_share',fk),('samefact_share',sf)):cols[kk].append(v)
        for kk,v in res.items():cols[kk].append(v)
        cols['seed'].append(np.full(len(q),sd))
    C={k:np.concatenate(v) for k,v in cols.items()};np.savez_compressed(a.npz,**C)
    y=C['is_fact'].astype(bool);n=len(y);meas=[k for k in C if k not in ('chunk','row','fact','is_fact','seed')];out={'rows':n,'fact_row_base_rate':float(y.mean()),'measures':meas,'auc':{},'topk':{},'spearman':{},'jaccard_top1':{},'jaccard_top10':{}}
    for k in meas:out['auc'][k]=auc(C[k],y)
    def topset(x,f):
        v=np.where(np.isnan(x),-np.inf,x);o=np.argsort(-v,kind='stable')[:max(1,int(round(f*n)))];m=np.zeros(n,bool);m[o]=True;return m
    sets={}
    for k in meas:
        for f in (.01,.03,.10):
            m=topset(C[k].astype(float),f);sets[(k,f)]=m;out['topk'].setdefault(k,{})[f'{f*100:g}%']=dict(precision=float((m&y).sum()/m.sum()),recall=float((m&y).sum()/y.sum()),rows=int(m.sum()))
    rng=np.random.default_rng(0);sub=np.arange(n) if a.all else np.sort(rng.choice(n,min(n,100000),replace=False));rk={k:rankdata(np.nan_to_num(C[k][sub].astype(float),nan=np.nanmedian(C[k][sub].astype(float)))) for k in meas}
    for i,ka in enumerate(meas):
        out['spearman'][ka]={kb:float(np.corrcoef(rk[ka],rk[kb])[0,1]) if rk[ka].std()>0 and rk[kb].std()>0 else None for kb in meas}
        out['jaccard_top1'][ka]={kb:float((sets[(ka,.01)]&sets[(kb,.01)]).sum()/max((sets[(ka,.01)]|sets[(kb,.01)]).sum(),1)) for kb in meas}
        out['jaccard_top10'][ka]={kb:float((sets[(ka,.10)]&sets[(kb,.10)]).sum()/max((sets[(ka,.10)]|sets[(kb,.10)]).sum(),1)) for kb in meas}
    json.dump(out,open(a.out,'w'),indent=1)
    print('rows',n,'base rate %.3f'%y.mean())
    for k in sorted(meas,key=lambda k:-(out['auc'][k] or 0)):print(f"{k:16s} AUC {out['auc'][k] if out['auc'][k] is None else round(out['auc'][k],3)}  top1% P {out['topk'][k]['1%']['precision']:.3f} R {out['topk'][k]['1%']['recall']:.3f} | top3% P {out['topk'][k]['3%']['precision']:.3f} R {out['topk'][k]['3%']['recall']:.3f} | top10% P {out['topk'][k]['10%']['precision']:.3f} R {out['topk'][k]['10%']['recall']:.3f}")
if __name__=='__main__':main()
