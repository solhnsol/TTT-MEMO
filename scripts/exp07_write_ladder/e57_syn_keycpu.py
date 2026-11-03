"""E57 synthetic s64, key-level masking, CPU analysis (new file; method thread card 'M4 + 키 단위 가림', part (b); NO GPU): from the W0 key-mass dump (e57_syn_s64/syn_s64_dump_j*.npz: per query row the TOP-128 far keys (beyond the band b = 256) by ALLV = sum over heads and layers of p x ||v||, by ALL = plain attention mass, and the total far mass) and the world table (fact of every token) the distribution of the far-key mass of the deciding rows (top 30% by window gain, syn_s64_m30_masks.npz): how many keys cover 50 / 80 / 90% of the far mass, and where the top-m keys of a row lie (earlier sentences of the SAME fact, the place tokens of the same fact, other facts, text without a fact). Output <out>/e57_syn_keycpu.json.
  python3 e57_syn_keycpu.py --out O | --selftest"""
import sys,json,types,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;WIN=8192;BAND=256;MS=(1,2,4,8,16,32,64,128)
def cover_counts(sc,far,fr=(.5,.8,.9)):
    """sc [R,128] descending scores, far [R] -> per row the number of keys needed to reach each fraction of the far mass (129 = more than the 128 stored)"""
    cs=np.cumsum(sc,1)/np.maximum(far[:,None],1e-12);return {f:np.where(cs[:,-1]>=f,1+(cs>=f).argmax(1),129) for f in fr}
def selftest():
    sc=np.asarray([[4.,3.,2.,1.]]);c=cover_counts(sc,np.asarray([10.]),(.5,.9,1.0));assert c[.5][0]==2 and c[.9][0]==3 and c[1.0][0]==4
    c=cover_counts(sc,np.asarray([20.]),(.9,));assert c[.9][0]==129;print('e57_syn_keycpu selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e57_syn_pass as SP,e57_dwg_analyze as DW
    enc,w,tf=R55.build(SP.synth_args(64),64,arms=['NAT']);nch=128;fact=np.full(nch*CH+1,-1,np.int32);sent=np.full(nch*CH+1,-1,np.int64);place=np.zeros(nch*CH+1,bool)
    for c in w['chunks']:
        k=c['k']
        for si,s in enumerate(c['sents']):fact[k*CH+s['start']:k*CH+s['end']]=s['fact'];sent[k*CH+s['start']:k*CH+s['end']]=k*100000+si
    import e57_pqa2_stream as S
    tk=S.tokenizer()
    for r in DW.locate_syn(tk,w,None):
        for q in r['pos']:place[q]=True
    M=np.load(EXP/'e57_synq/syn_s64_m30_masks.npz');Z=np.load(EXP/'e57_syn_s64/syn_s64_nll.npz');lm=Z['lm'];res={'by_name':{},'n_rows':{}}
    acc={nm:dict(cov={m:[] for m in MS},need={f:[] for f in (.5,.8,.9)},same_fact={m:[] for m in MS},same_place={m:[] for m in MS},other_fact={m:[] for m in MS},nofact={m:[] for m in MS},hit_same={m:[] for m in MS},far_share_all=[],gain=[],dec=[]) for nm in ('ALLV','ALL')}
    for j in range(8):
        D=np.load(EXP/f'e57_syn_s64/syn_s64_dump_j{j:02d}.npz')
        for k in [int(x) for x in D['chunks']]:
            if k<16:continue
            dec=CM.up(M[f'qb_TOP30_{k}'],CH);g=(Z['nll_band'][0][k]-Z['nll_win'][k])
            rowfact=fact[k*CH+np.arange(CH)+1];rows=np.where(dec&(rowfact>=0))[0]
            for nm in acc:
                idx=D[f'c{k}/{nm}_b256_idx'][rows].astype(np.int64);sc=D[f'c{k}/{nm}_b256_score'][rows].astype(np.float64);far=D[f'c{k}/{nm}_b256_far'][rows].astype(np.float64);pos=(k-8)*CH+idx
                cnt=cover_counts(sc,far)
                for f in cnt:acc[nm]['need'][f].append(cnt[f])
                cs=np.cumsum(sc,1)/np.maximum(far[:,None],1e-12);rf=rowfact[rows][:,None];valid=idx>=0;kf=np.where(valid,fact[np.clip(pos,0,nch*CH)],-1);kp=np.where(valid,place[np.clip(pos,0,nch*CH)],False)
                for m in MS:
                    acc[nm]['cov'][m].append(cs[:,m-1]);s_=sc[:,:m];tot=np.maximum(s_.sum(1),1e-12)
                    same=(kf[:,:m]==rf);acc[nm]['same_fact'][m].append((s_*same).sum(1)/tot);acc[nm]['same_place'][m].append((s_*(same&kp[:,:m])).sum(1)/tot)
                    oth=(kf[:,:m]>=0)&(kf[:,:m]!=rf);acc[nm]['other_fact'][m].append((s_*oth).sum(1)/tot);acc[nm]['nofact'][m].append((s_*(kf[:,:m]<0)).sum(1)/tot);acc[nm]['hit_same'][m].append(same.any(1).astype(float))
                acc[nm]['gain'].append(g[rows])
    q=lambda x:dict(mean=float(np.mean(x)),p10=float(np.percentile(x,10)),p50=float(np.percentile(x,50)),p90=float(np.percentile(x,90)))
    for nm,A in acc.items():
        cat=lambda L:np.concatenate(L);res['by_name'][nm]=dict(n_rows=int(len(cat(A['gain']))),coverage_of_far_mass_by_top_m={str(m):q(cat(A['cov'][m])) for m in MS},keys_needed_for_fraction={str(f):dict(**q(cat(A['need'][f])),share_above_128=float((cat(A['need'][f])>128).mean())) for f in A['need']},
            share_of_top_m_mass={str(m):dict(same_fact=q(cat(A['same_fact'][m])),same_fact_place_tokens=q(cat(A['same_place'][m])),other_fact=q(cat(A['other_fact'][m])),no_fact=q(cat(A['nofact'][m])),rows_with_a_same_fact_key=float(cat(A['hit_same'][m]).mean())) for m in MS},mean_window_gain_of_these_rows=float(cat(A['gain']).mean()))
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'e57_syn_keycpu.json').write_text(json.dumps(res,indent=1));print(json.dumps({nm:{k:v for k,v in r.items() if k in ('n_rows','keys_needed_for_fraction')} for nm,r in res['by_name'].items()}))
if __name__=='__main__':main()
