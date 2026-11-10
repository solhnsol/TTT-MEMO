"""E57 original-NAT write arms, synthetic 16-fact worlds: analysis of job 1270 outputs (CPU; new file). Metric = read accuracy of the 16 fact items (argmax over the 16 candidates == gold, out-of-window reads) averaged over the exposures 16/32/64/128; unit of analysis = one (seed, fact) pair of the same arms (item-level paired bootstrap over the seeds x 16 units); contrasts and rules of e57_natmask_run.verdict_rules (fixed +-2 pp equivalence margin on the 90% CI, superiority = 95% CI lower bound > 0 and Holm-adjusted p < .05). Also canary (gen/tail NLL rise at the last chunk) and per-arm update statistics.
  python3 e57_natmask_analyze.py --dir O --seeds 64,61,62,63,65 [--tag synng] [--json J]"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_natmask_run as NM
ARMS=['NATS','NATMASK','NATMASKC','DIRmaskNORMfull','DIRfullNORMmask'];EXPO=['16','32','64','128']
def item_acc(path):
    d=json.load(open(path));g=np.array(d['meta']['gold_idx']);fact=np.array([k=='fact' for k in d['meta']['kind']])
    A=np.stack([(np.array(d['read'][a]['out']).argmax(1)==g)[fact] for a in EXPO]).astype(float);return A.mean(0),A[-1]      # exposure-mean accuracy per fact, accuracy at 128
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--tag',default='synng');ap.add_argument('--json');a=ap.parse_args();D=pathlib.Path(a.dir);seeds=[int(x) for x in a.seeds.split(',')]
    acc={};acc128={};stats={}
    for arm in ARMS:
        m=[];m128=[];st=[]
        for s in seeds:
            f=D/f'E55_{a.tag}_{arm}_s{s}.read.json';sv=D/f'E55_{a.tag}_{arm}_s{s}.saved.json'
            assert f.exists() and sv.exists(),('missing',f);x,y=item_acc(f);m.append(x);m128.append(y);r=json.load(open(sv));w0=r['canary_w0']
            cn=r['reads'][str(max(int(k) for k in r['reads']))]['canary'];st.append(dict(gen=cn['_gen_nll']-w0['_gen_nll'],tail=cn['_tail_nll']-w0['_tail_nll'],clip=r['summary']['clip_frac_coef_lt1'],upd=r['summary']['median_update_norm']))
        acc[arm]=np.stack(m);acc128[arm]=np.stack(m128);stats[arm]=st
    out={'seeds':seeds,'per_arm':{},'contrasts':{}}
    for arm in ARMS:
        x=acc[arm];out['per_arm'][arm]=dict(mean_acc=float(x.mean()),per_seed=[float(v) for v in x.mean(1)],acc128=float(acc128[arm].mean()),canary_gen_mean=float(np.mean([e['gen'] for e in stats[arm]])),canary_tail_mean=float(np.mean([e['tail'] for e in stats[arm]])),clip_frac=float(np.mean([e['clip'] for e in stats[arm]])),median_update_norm=float(np.mean([e['upd'] for e in stats[arm]])))
    C={'NATMASK-NATS':('NATMASK','NATS'),'NATMASK-NATMASKC':('NATMASK','NATMASKC'),'DIRmaskNORMfull-NATS':('DIRmaskNORMfull','NATS'),'DIRfullNORMmask-NATS':('DIRfullNORMmask','NATS'),'DIRmaskNORMfull-NATMASK':('DIRmaskNORMfull','NATMASK'),'DIRfullNORMmask-NATMASK':('DIRfullNORMmask','NATMASK')}
    for acc_name,AC in (('exposure_mean',acc),('acc128',acc128)):
        cs={k:(AC[x].reshape(1,-1),AC[y].reshape(1,-1)) for k,(x,y) in C.items()};out['contrasts'][acc_name]=NM.verdict_rules(cs)
    print(json.dumps(out,indent=1,default=float))
    if a.json:json.dump(out,open(a.json,'w'),indent=1,default=float)
if __name__=='__main__':main()
