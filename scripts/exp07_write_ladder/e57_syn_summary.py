"""E57 synthetic world query-band write arms: summary table (CPU only; new file; numbers only). Per arm of job 360 (E55_<tag>_<ARM>_s<seed>.saved.json / .read.json): binding read of the FACT items (e55_metrics.rec_items, mean over the 16 fact items; a = appearances 16, 32, 64, 128, context out; a = 128 context in_g0): top-1 accuracy, mean rank of the gold candidate (chance 8.5), rank gain G_rank = rank at W0 - rank after writing (positive = better); canary rise at chunk 127 (generation, tail; vs W0); banded rows per chunk (mean over chunks >= 8) and the share of (row, window key) pairs hidden (n_masked_keys / (1024 x 9216) mean over chunks >= 8); mean write CE; cumulative update norm.
  .venv/bin/python scripts/exp07_write_ladder/e57_syn_summary.py --dir D --seed 64 --tag synq --arms A1,A2,... --out O.json"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e55_metrics as M
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seed',type=int);ap.add_argument('--tag',default='synq');ap.add_argument('--arms');ap.add_argument('--out');a=ap.parse_args();d=pathlib.Path(a.dir);out={}
    for arm in a.arms.split(','):
        fs=d/f'E55_{a.tag}_{arm}_s{a.seed}.saved.json';fr=d/f'E55_{a.tag}_{arm}_s{a.seed}.read.json'
        if not (fs.exists() and fr.exists()):continue
        s=json.load(open(fs));r=json.load(open(fr));row={}
        for ap_,ctx in ((16,'out'),(32,'out'),(64,'out'),(128,'out'),(128,'in_g0')):
            try:
                it=M.rec_items(r,ap_,ctx);sel=M._sel(r,ap_,ctx);row[f'{ctx}@{ap_}']=dict(n=int(sel.sum()),top1=float(np.nanmean(it['top1_W'][sel])),rank=float(np.nanmean(it['rank_W'][sel])),rank_W0=float(np.nanmean(it['rank_W0'][sel])),G_rank=float(np.nanmean(it['G_rank'][sel])))
            except Exception as e:row[f'{ctx}@{ap_}']=dict(error=repr(e)[:80])
        lg=s['log'][8:];w0=s['canary_w0'];c=s['reads'].get('127',{}).get('canary')
        row['banded_rows_per_chunk']=float(np.mean([e.get('q_banded',0) for e in lg]));row['hidden_pair_share']=float(np.mean([e.get('n_masked_keys',0) for e in lg])/(1024*9216));row['mean_write_ce']=float(np.mean([e['ce'] for e in s['log']]));row['cum_update_norm']=s['cum_update_norm']
        row['canary_gen_rise']=(c['_gen_nll']-w0['_gen_nll']) if c else None;row['canary_tail_rise']=(c['_tail_nll']-w0['_tail_nll']) if c else None
        out[arm]=row
    json.dump(dict(seed=a.seed,tag=a.tag,note='n = fact items read; rank chance level 8.5; G_rank positive = better',arms=out),open(a.out,'w'),indent=1,default=float)
    for arm,row in out.items():
        o=lambda k:row.get(k,{});print(f"{arm:14s} OUT@128 top1 {o('out@128').get('top1',float('nan')):.3f} rank {o('out@128').get('rank',float('nan')):.2f} G {o('out@128').get('G_rank',float('nan')):+.2f} | OUT@64 top1 {o('out@64').get('top1',float('nan')):.3f} | IN@128 top1 {o('in_g0@128').get('top1',float('nan')):.3f} rank {o('in_g0@128').get('rank',float('nan')):.2f} | banded {row['banded_rows_per_chunk']:.0f} rows hidden-pairs {row['hidden_pair_share']:.4f} | canary gen {row['canary_gen_rise']:+.3f} tail {row['canary_tail_rise']:+.3f}")
if __name__=='__main__':main()
