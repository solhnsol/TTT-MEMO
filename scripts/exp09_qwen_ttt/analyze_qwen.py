"""Exp09 analysis (CPU): reuses scripts/exp07_write_ladder/analyze_ladder.py (AL.stats, AL.ci) and analyze_stage01.py
(S.load, S.curve) unmodified, pointed at the exp09 output directory through EXP07_OUT.
Usage: analyze_qwen.py [--seed 101] [--out DIR] [--l1-tag main] [LEVEL:TAG ...]   (default: L1/L0mp/L0mw/L0mwfs/L0 :main)
Writes {OUT}/exp09_report_s{SEED}.md and exp09_metrics_s{SEED}.json."""
import os,sys,json,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--out',default=str(ROOT/'experiments/exp09_qwen_ttt'))
    ap.add_argument('--l1-tag',default='main');ap.add_argument('specs',nargs='*',default=['L1:main','L0mp:main','L0mw:main','L0mwfs:main','L0:main'])
    a=ap.parse_args(argv);out=Path(a.out)
    os.environ['EXP07_OUT']=str(out);saved=sys.argv;sys.argv=[saved[0],str(a.seed),a.l1_tag]   # AL/S read these at import
    sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
    try:
        import analyze_ladder as AL,analyze_stage01 as S
    finally:sys.argv=saved
    assert Path(AL.OUT)==out and S.SEED==a.seed
    md=[f'# Exp09 (Qwen3 + LoRA TTT) seed {a.seed}\n'];res={}
    for spec in a.specs:
        lev,tag=spec.split(':');X=S.load(lev,tag)
        if not X:md.append(f'{spec}: missing');continue
        c=S.curve(*X);res[spec]=[]
        md.append(f'\n## {lev} ({tag})\n\n| point | D (R1) | D (R1-alt) | acc/flip | F_name gain | M_L3r | M_L2 | max canary rise |\n|---|---|---|---|---|---|---|---|')
        for ep,s,sa,d in c:
            md.append(f"| {ep} | {AL.f3(AL.ci(s['D']))} | {AL.f3(AL.ci(sa['D'])) if sa else '-'} | {s['acc']:.2f}/{s['flip']:.2f} | {s['F_name'].mean():+.3f} | {s['M_L3r'].mean():+.3f} | {s['M_L2'].mean():+.3f} | {d:+.3f} |")
            res[spec].append({'point':ep,'D':AL.ci(s['D']),'D_alt':AL.ci(sa['D']) if sa else None,'acc':s['acc'],'flip':s['flip'],'F_name':float(s['F_name'].mean()),'canary_rise':d})
    json.dump(res,open(out/f'exp09_metrics_s{a.seed}.json','w'),indent=1)
    (out/f'exp09_report_s{a.seed}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
    return res
if __name__=='__main__':main()
