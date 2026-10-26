"""E57 pod run: compact summary of the arms and reads (CPU; reads E57_main_<ARM>.json of the pod dir): per arm the cycle-1 end reads (all 42 questions) OUT / IN: accuracy over all candidates, rank ratio, hard-only accuracy, plus the canary rise, hidden-token means and the guard. First observation only, no inference.
  python3 e57_pod_summary.py --dir DIR [--arms W0,C1,LRN,RC1,RLRN,ORCA,RORCA]"""
import sys,json,argparse,pathlib,statistics
def mean(x):x=[v for v in x if v is not None];return statistics.mean(x) if x else float('nan')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--arms',default='W0,C1,LRN,RC1,RLRN,ORCA,RORCA');a=ap.parse_args();d=pathlib.Path(a.dir);print('| arm | cond | n | acc all | rank ratio all | acc hard | rank ratio hard | tail rise | gen rise | mean hidden window tok | mean hidden chunk tok | guard |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
    for arm in a.arms.split(','):
        fn=d/f'E57_main_{arm}.json'
        if not fn.exists():print(f'| {arm} | (no file) |');continue
        v=json.load(open(fn));rows=v['rows'];w0=v['canary_w0'];tr=v.get('trajectory') or []
        tail=gen='-';hw=hc=gd='-'
        if tr:c=tr[-1]['canary'];tail=f"{c['_tail_nll']-w0['_tail_nll']:+.3f}";gen=f"{c['_gen_nll']-w0['_gen_nll']:+.3f}"
        ul=v.get('update_log') or []
        if ul:hw=f"{mean([e.get('uv_hidden_tokens') for e in ul]):.0f}";hc=f"{mean([e.get('uv_hidden_chunk_tokens') for e in ul]):.0f}"
        gd=str(v.get('guard','-'))
        for cond in ('OUT','IN'):
            r=[x for x in rows if x['condition']==cond]
            if not r:continue
            print(f"| {arm} | {cond} | {len(r)} | {mean([x['acc_all'] for x in r]):.3f} | {mean([x['rank_ratio_all'] for x in r]):.3f} | {mean([x['acc_hard'] for x in r]):.3f} | {mean([x['rank_ratio_hard'] for x in r]):.3f} | {tail} | {gen} | {hw} | {hc} | {gd} |")
if __name__=='__main__':main()
