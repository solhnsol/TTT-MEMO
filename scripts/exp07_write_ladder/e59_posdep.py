"""Does the gain of an arm over NAT depend on where the fact sentence sits inside the chunk? (CPU; new file; EXPLORATORY descriptive; thread-server 11/23; coordinator request): per FACT out-of-window margin (gold - best wrong of the fact's own item, exposure 128) of an arm minus NAT (same world and seed), against the fact's mean position (token index of the sentence start) inside the 1024-token chunk over all its exposures (and over the last 8 exposures); facts split into three position terciles within each seed (mean of the arm - NAT gain per tercile) and a per-seed regression slope (gain per 100 tokens); seed-paired mean [95% t-CI] over the seeds. Data: E2E synthetic world v3_16_1 seeds 231-238 (NAT / CHK / ISO / MASK1024 of job 1321) and the FictionalQA worlds 211-218 (NATS of job 1313 and CHK / ISO of job 1323, BAND1024 = NATMASK of job 1313). python3 e59_posdep.py {e2e|fq}"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];E7=ROOT/'experiments/exp07_write_ladder'
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
def positions(kind,seed):
    import e55_run as R55,e57_syn_pass as SP,e57_oracle_run as OR
    if kind=='fq':
        import fq_run as FQR;FQR.install(str(E7/'fq/worlds_w0c'));cell=f'fq{seed}';OR.register_cell(cell)
    else:cell='v3_16_1'
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps={16,32,64,128},arms=['NAT']);P={f:[] for f in range(16)}
    for c in w['chunks']:
        for x in c['sents']:P[int(x['fact'])].append(int(x['start']))
    return {f:(float(np.mean(v)),float(np.mean(v[-8:])),len(v)) for f,v in P.items()}
def fact_margin(path,E=128):
    r=json.load(open(path));m=r['meta'];x=np.asarray(r['read'][str(E)]['out'],float);out={}
    for i,(kd,f) in enumerate(zip(m['kind'],m['fact'])):
        if kd!='fact':continue
        g=m['gold_idx'][i];w=x[i].copy();gold=w[g];w[g]=-np.inf;out[int(f)]=float(gold-w.max())
    return out
def files(kind):
    if kind=='e2e':return range(231,239),{'NAT':lambda s:E7/f'e59_4way/E59_a4_NAT_s{s}.read.json','CHK':lambda s:E7/f'e59_4way/E59_a4_CHK_s{s}.read.json','ISO':lambda s:E7/f'e59_4way/E59_a4_ISO_s{s}.read.json','MASK1024':lambda s:E7/f'e59_4way/E59_a4_MASK1024_s{s}.read.json'}
    return range(211,219),{'NAT':lambda s:E7/f'fq/run/E55_fq_NATS_s{s}.read.json','CHK':lambda s:E7/f'e59_fq/E59_fq_CHK_s{s}.read.json','ISO':lambda s:E7/f'e59_fq/E59_fq_ISO_s{s}.read.json','BAND1024':lambda s:E7/f'fq/run/E55_fq_NATMASK_s{s}.read.json'}
def main():
    kind=sys.argv[1];seeds,F=files(kind);seeds=list(seeds);n=len(seeds);t=stats.t.ppf(.975,n-1);cache=E7/f'binding_fit/posdep_positions_{kind}.json'
    pos=json.load(open(cache)) if cache.exists() else {}
    for s in seeds:
        if str(s) not in pos:pos[str(s)]={str(f):v for f,v in positions(kind,s).items()};print('positions seed',s,flush=True)
    json.dump(pos,open(cache,'w'))
    arms=[a for a in F if a!='NAT'];res={a:{k:[] for k in ('lo','mid','hi','slope','slope_last8','corr')} for a in arms};ci=lambda d:(lambda d:(d.mean(),d.mean()-t*d.std(ddof=1)/np.sqrt(n),d.mean()+t*d.std(ddof=1)/np.sqrt(n),int(max((d>0).sum(),(d<0).sum()))))(np.asarray(d))
    spread=[]
    for s in seeds:
        nat=fact_margin(F['NAT'](s));P=pos[str(s)];fs=sorted(nat);mp=np.array([P[str(f)][0] for f in fs]);lp=np.array([P[str(f)][1] for f in fs]);spread.append(mp.max()-mp.min())
        order=np.argsort(mp);terc=[order[:5],order[5:11],order[11:]]
        for a in arms:
            ar=fact_margin(F[a](s));g=np.array([ar[f]-nat[f] for f in fs])
            for nm,idx in zip(('lo','mid','hi'),terc):res[a][nm].append(g[idx].mean())
            res[a]['slope'].append(np.polyfit(mp,g,1)[0]*100);res[a]['slope_last8'].append(np.polyfit(lp,g,1)[0]*100);res[a]['corr'].append(np.corrcoef(mp,g)[0,1])
    print(f'{kind}: per-fact out-of-window margin at exposure 128, arm - NAT, by the fact\'s mean position in the chunk (terciles: first 5 / middle 6 / last 5 facts of 16 sorted by mean position); seeds {seeds[0]}-{seeds[-1]}; mean over seeds [95% t] same-direction seeds; the mean position range per seed is {np.mean(spread):.0f} tokens on average')
    for a in arms:
        print(f'== {a} - NAT')
        for nm,lab in (('lo','first third'),('mid','middle third'),('hi','last third'),('slope','slope per 100 tokens (mean position)'),('slope_last8','slope per 100 tokens (position of the last 8 exposures)'),('corr','correlation gain vs mean position')):
            m,l,h,k=ci(res[a][nm]);print(f'  {lab:55s} {m:7.3f} [{l:7.3f},{h:7.3f}] {k}/{n}')
        d=np.array(res[a]['hi'])-np.array(res[a]['lo']);m,l,h,k=ci(d);print(f'  {"last third - first third":55s} {m:7.3f} [{l:7.3f},{h:7.3f}] {k}/{n}')
main()
