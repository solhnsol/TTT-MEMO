"""C7 metric validity check from STORED read files (CPU; new file; thread-server 11/23; coordinator request for Hans; EXPLORATORY descriptive; seed-paired t 95% intervals, n = seeds): out-of-window reads at exposure 128 (neutral tail), the 16 fact items per world (control items excluded). Datasets: e2e_syn (1321, seeds 231-238; NAT, CHK, ISO, MASK1024), syn71 (seeds 71-78; NAT = NATS of 1283, M256 = NATMASK band 256 of 1283, M1024 = NATMASK band 1024 of 1293, ISO-STD of 1283), e2e_fq (FictionalQA 211-218; NAT, CHK, ISO, BAND1024). (a) per-fact margin (gold - best wrong over the 16 candidates) distributions: pooled quantiles, share margin > 0 (= top-1), share of facts improved over W0, skewness and bimodality coefficient per seed, the share of the total margin gain over W0 carried by the 4 best-improved facts of the 16; (b) length-normalised margin: all candidates of a world have the SAME token length place_len (gen_facts fresh places with len(enc(' '+place)) == place_len), so the length-normalised margin is the margin divided by that constant (order-preserving; reported per token for comparison across worlds only); (c) separation of the gain: change of the gold score over W0, change of the best wrong score over W0, change of the mean wrong score over W0 (the margin gain = gold change - best-wrong change); (d) NAT top-1 below the 1/16 = .0625 chance: counts, gold rank vs the chance mean 8.5, W0 top-1, share of items whose NAT argmax equals the W0 argmax, spearman correlation over the 16 candidates between W0 and NAT scores (candidate prior), distinct argmax candidate facts per seed vs the uniform expectation, most frequent argmax position. python3 e59_c7_validity.py"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
X=Path(__file__).resolve().parents[2]/'experiments';E7=X/'exp07_write_ladder'
DS={'e2e_syn':(range(231,239),{'NAT':'e59_4way/E59_a4_NAT_s{s}.read.json','CHK':'e59_4way/E59_a4_CHK_s{s}.read.json','ISO':'e59_4way/E59_a4_ISO_s{s}.read.json','MASK1024':'e59_4way/E59_a4_MASK1024_s{s}.read.json'}),
    'syn71':(range(71,79),{'NAT':'e57_k0conf16/E55_k0c16_NATS_s{s}.read.json','M256':'e57_k0conf16/E55_k0c16_NATMASK_s{s}.read.json','M1024':'e57_band_dist2/w1024/E55_bd1024_NATMASK_s{s}.read.json','ISO':'E55_q1a71_ISO-STD_s{s}.read.json'}),
    'e2e_fq':(range(211,219),{'NAT':'fq/run/E55_fq_NATS_s{s}.read.json','CHK':'e59_fq/E59_fq_CHK_s{s}.read.json','ISO':'e59_fq/E59_fq_ISO_s{s}.read.json','BAND1024':'fq/run/E55_fq_NATMASK_s{s}.read.json'})}
def load(p):
    r=json.load(open(E7/p));rd=r.get('read') or r.get('reads');m=r['meta'];f=np.asarray(m['kind'])=='fact';g=np.asarray(m['gi'] if 'gi' in m else m['gold_idx'])[f]
    return np.asarray(rd['128']['out'],float)[f],np.asarray(r['W0']['out'],float)[f],g,np.asarray(m['cand_fact'])[f],r.get('place_len')
def parts(S,g):
    k=np.arange(len(g));gold=S[k,g];w=S.copy();w[k,g]=-np.inf;return gold,w.max(1),np.where(np.isinf(w),0,w).sum(1)/(S.shape[1]-1),(S>gold[:,None]).sum(1)
def ci(d,fmt='{:7.3f}'):
    d=np.asarray(d,float);n=len(d);m=d.mean();h=stats.t.ppf(.975,n-1)*d.std(ddof=1)/np.sqrt(n);return (fmt+' ['+fmt+','+fmt+'] {}/{}').format(m,m-h,m+h,int(max((d>0).sum(),(d<0).sum())),n)
def main():
    for name,(seeds,arms) in DS.items():
        seeds=list(seeds);D={a:[load(p.format(s=s)) for s in seeds] for a,p in arms.items()};L=D['NAT'][0][4];print(f'\n######## {name} seeds {seeds[0]}-{seeds[-1]} exposure 128 out-of-window; place_len (tokens of every candidate) = {L}')
        print('(a) per-fact margin; pooled over facts x seeds (n=%d): quantiles P10 P25 P50 P75 P90 | share margin>0 | share facts improved over W0 | per-seed skewness, bimodality coef (>0.555 suggests bimodal) | share of the total gain over W0 carried by the 4 best-improved facts'%(16*len(seeds)))
        R={}
        for a in arms:
            ms=[];gs=[];sk=[];bc=[];sh=[]
            for (S,W,g,_,_) in D[a]:
                gold,bw,_,_=parts(S,g);g0,bw0,_,_=parts(W,g);m=gold-bw;m0=g0-bw0;gain=m-m0;ms.append(m);gs.append(gain);sk.append(stats.skew(m));bc.append((stats.skew(m)**2+1)/(stats.kurtosis(m,fisher=False)));sh.append(np.sort(gain)[-4:].sum()/gain.sum() if gain.sum()>0 else np.nan)
            P=np.concatenate(ms);q=np.percentile(P,[10,25,50,75,90]);G=np.concatenate(gs);R[a]=dict(ms=ms,gs=gs)
            print(f'  {a:9s} quantiles '+' '.join(f'{x:7.2f}' for x in q)+f' | m>0 {np.mean(P>0):.3f} | improved {np.mean(G>0):.3f} | skew {np.nanmean(sk):6.2f} bimod {np.nanmean(bc):.2f} | top4 gain share {np.nanmean(sh):.2f} ({int(np.isnan(sh).sum())} seeds without positive total gain)')
        print('   per-seed mean margin, arm - NAT, seed-paired [95% t]:');[print(f'     {a:9s} {ci(np.array([m.mean() for m in R[a]["ms"]])-np.array([m.mean() for m in R["NAT"]["ms"]]))} | median margin diff {ci(np.array([np.median(m) for m in R[a]["ms"]])-np.array([np.median(m) for m in R["NAT"]["ms"]]))}') for a in arms if a!='NAT']
        print(f'(b) length-normalised margin (margin / {L} tokens): constant rescale; per-token margin level (mean over seeds): '+', '.join(f'{a} {np.mean([m.mean() for m in R[a]["ms"]])/L:6.3f}' for a in arms)+'; all candidates have equal token length, so the ranks and top-1 are unchanged by the normalisation')
        print('(c) separation of the gain over W0 (mean over the 16 facts, per seed): gold change | best-wrong change (negative = better) | mean-wrong change | margin gain = gold - best-wrong change')
        C={}
        for a in arms:
            r=[]
            for (S,W,g,_,_) in D[a]:
                gd,bw,mw,_=parts(S,g);g0,bw0,mw0,_=parts(W,g);r.append([(gd-g0).mean(),(bw-bw0).mean(),(mw-mw0).mean()])
            C[a]=np.array(r);print(f'  {a:9s} gold {ci(C[a][:,0])} | best-wrong {ci(C[a][:,1])} | mean-wrong {ci(C[a][:,2])}')
        for a in arms:
            if a!='NAT':print(f'   {a:9s} - NAT: gold {ci(C[a][:,0]-C["NAT"][:,0])} | best-wrong {ci(C[a][:,1]-C["NAT"][:,1])} | mean-wrong {ci(C[a][:,2]-C["NAT"][:,2])} | margin {ci((C[a][:,0]-C[a][:,1])-(C["NAT"][:,0]-C["NAT"][:,1]))}')
        print('(d) NAT / W0 top-1 and candidate preference (all arms for the counts)')
        for a in arms:
            c=[];rk=[]
            for (S,W,g,_,_) in D[a]:
                _,_,_,gt=parts(S,g);c.append((gt==0).sum());rk.append((1+gt).mean())
            c0=[(parts(W,g)[3]==0).sum() for (_,W,g,_,_) in D[a]]
            print(f'  {a:9s} top-1 correct {int(sum(c))}/{16*len(seeds)} = {sum(c)/(16*len(seeds)):.3f} (W0 {int(sum(c0))}/{16*len(seeds)} = {sum(c0)/(16*len(seeds)):.3f}; chance 0.0625) | mean gold rank {np.mean(rk):.2f} (W0 {np.mean([(1+parts(W,g)[3]).mean() for (_,W,g,_,_) in D[a]]):.2f}; chance 8.5)')
        for a in ('NAT',):
            same=[];rho=[];dist=[];pos=[];dom=[];wrongfacts=[]
            for (S,W,g,cf,_) in D[a]:
                am=S.argmax(1);am0=W.argmax(1);same.append((am==am0).mean());rho.append(np.mean([stats.spearmanr(W[i],S[i])[0] for i in range(len(g))]))
                af=cf[np.arange(len(g)),am];dist.append(len(set(af.tolist())));dom.append(np.bincount(af.astype(int)).max()/len(g));pos.append(np.bincount(am,minlength=16).max()/len(g))
            n=16;exp=n*(1-(1-1/n)**n)
            print(f'  NAT argmax == W0 argmax: share {np.mean(same):.3f} [{min(same):.2f}..{max(same):.2f}] | spearman(W0 scores, NAT scores) over 16 candidates, mean over items {ci(rho)} | distinct argmax candidate facts per seed {np.mean(dist):.1f} (uniform-random expectation {exp:.1f}); most frequent argmax candidate fact takes {np.mean(dom):.2f} of the items of a seed (uniform expectation ~{(1+ (n-1)*0)/n+0.0:.2f}-{2.0/n+0.0:.2f}) | most frequent argmax POSITION share {np.mean(pos):.2f}')
            pools={};
            for (S,W,g,cf,_) in D[a]:
                for i in range(len(g)):pools.setdefault(int(cf[i,S[i].argmax()]),0);pools[int(cf[i,S[i].argmax()])]+=1
            print('  NAT argmax candidate fact counts pooled over seeds (fact id : items), top 6: '+', '.join(f'{k}:{v}' for k,v in sorted(pools.items(),key=lambda x:-x[1])[:6])+f' (of {16*len(seeds)}; note: fact ids are world-specific, high counts mean the same fact index wins across worlds)')
main()
