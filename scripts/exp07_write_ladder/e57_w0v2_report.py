"""E57 W0 ORC v2 deleted windows: CPU report (post-hoc, exploratory; no inference). Merges the rows of job 264 (E57_w0mask.json: OUT, IN_FULL, IN_DELORC, IN_DELRND0 at the primary positions) with the new rows (E57_w0v2.json: IN_DELORC_V2, IN_DELRND_V2) and writes a small key-number json: per condition accuracy / rank ratio / hard accuracy, residual gain (OUT - DELORC rank ratio) of the 264 ORC and the v2 deleted window, the top-10 gain questions of 264 under v2, paired differences.
  python3 e57_w0v2_report.py --w0mask E57_w0mask.json --w0v2 E57_w0v2.json --out OUT.json"""
import json,argparse
import numpy as np
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--w0mask');ap.add_argument('--w0v2');ap.add_argument('--out');a=ap.parse_args()
    A=json.load(open(a.w0mask));Bv=json.load(open(a.w0v2));cond={}
    for r in A['rows']:
        if r['condition']=='OUT':cond.setdefault('OUT',{})[r['question']]=r
        elif r['primary_position']:cond.setdefault(r['condition'],{})[r['question']]=r
    for r in Bv['rows']:cond.setdefault(r['condition'],{})[r['question']]=r
    qs=sorted(cond['OUT']);assert all(set(cond[c])==set(qs) for c in cond),{c:len(cond[c]) for c in cond}
    def agg(c,g=qs):return dict(n=len(g),acc_all=float(np.mean([cond[c][q]['acc_all'] for q in g])),rank_ratio_all=float(np.mean([cond[c][q]['rank_ratio_all'] for q in g])),acc_hard=float(np.mean([cond[c][q]['acc_hard'] for q in g])),gold_string_in_window=int(sum(1 for q in g if cond[c][q].get('window_info')=='present')))
    out={'conditions':{c:agg(c) for c in cond}}
    gain=lambda c:{q:cond['OUT'][q]['rank_ratio_all']-cond[c][q]['rank_ratio_all'] for q in qs}
    g0=gain('IN_DELORC');g2=gain('IN_DELORC_V2');top=sorted(qs,key=lambda q:-g0[q])[:10]
    out['mean_gain_rank_ratio_OUT_minus_window']={c:float(np.mean(list(gain(c).values()))) for c in ('IN_FULL','IN_DELORC','IN_DELORC_V2','IN_DELRND0','IN_DELRND_V2')}
    out['mean_gain_acc_window_minus_OUT']={c:float(np.mean([cond[c][q]['acc_all']-cond['OUT'][q]['acc_all'] for q in qs])) for c in ('IN_FULL','IN_DELORC','IN_DELORC_V2','IN_DELRND0','IN_DELRND_V2')}
    out['top10_gain_questions_of_264']={q:dict(gain_orc264=g0[q],gain_orc_v2=g2[q],acc_out=cond['OUT'][q]['acc_all'],acc_delorc=cond['IN_DELORC'][q]['acc_all'],acc_delorc_v2=cond['IN_DELORC_V2'][q]['acc_all']) for q in top}
    out['top10_mean_gain']={'orc264':float(np.mean([g0[q] for q in top])),'orc_v2':float(np.mean([g2[q] for q in top]))};rest=[q for q in qs if q not in top]
    out['rest_mean_gain']={'orc264':float(np.mean([g0[q] for q in rest])),'orc_v2':float(np.mean([g2[q] for q in rest]))}
    out['paired_delorc_v2_minus_delrnd_v2']={'acc_all':float(np.mean([cond['IN_DELORC_V2'][q]['acc_all']-cond['IN_DELRND_V2'][q]['acc_all'] for q in qs])),'rank_ratio_all':float(np.mean([cond['IN_DELORC_V2'][q]['rank_ratio_all']-cond['IN_DELRND_V2'][q]['rank_ratio_all'] for q in qs]))}
    out['per_question']={q:{c:dict(acc=cond[c][q]['acc_all'],rank_ratio=cond[c][q]['rank_ratio_all']) for c in cond} for q in qs}
    open(a.out,'w').write(json.dumps(out,indent=1));
    for c,v in out['conditions'].items():print(f"{c:14s} acc {v['acc_all']:.3f} rank ratio {v['rank_ratio_all']:.3f} hard {v['acc_hard']:.3f} gold-in-window {v['gold_string_in_window']}")
    print('gain (rank ratio OUT - window)',{k:round(v,3) for k,v in out['mean_gain_rank_ratio_OUT_minus_window'].items()});print('top10 mean gain 264 vs v2',out['top10_mean_gain'],'rest',out['rest_mean_gain'])
if __name__=='__main__':main()
