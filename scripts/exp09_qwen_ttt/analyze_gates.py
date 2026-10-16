"""Exp09 Qwen gates (PROTOCOL_stage01.md "Qwen gates 1 and 0", 2026-10-16 16:00 KST), CPU.
Gate 1: L0mp 1 pass; per-pair name-familiarity gain = F_name - F_null (stream names FN minus never-seen control names FN0,
each pass 1 - W0, pooled A/B), AL.ci bootstrap; PASS iff CI lower > 0. (Amended 16:05 KST, before any gate result.)
Gate 0: L1 32 epochs; D_alt (R1-alt) at epoch 32, AL.ci; PASS iff CI lower > 0.5. Every evaluation point is reported.
Usage: analyze_gates.py --out DIR [--seed 101] [--g1 L0mp:gate1] [--g0 L1:gate0]   (uses meta_L1_<l1-tag> for pair ids)"""
import os,sys,json,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=101)
    ap.add_argument('--g1',default='L0mp:gate1');ap.add_argument('--g0',default='L1:gate0');ap.add_argument('--l1-tag',default='gate0')
    a=ap.parse_args(argv);out=Path(a.out)
    os.environ['EXP07_OUT']=str(out);saved=sys.argv;sys.argv=[saved[0],str(a.seed),a.l1_tag]
    sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
    try:import analyze_ladder as AL,analyze_stage01 as S
    finally:sys.argv=saved
    md=[f'# Exp09 Qwen gates, seed {a.seed}\n'];res={}
    for name,spec,stat in (('gate1',a.g1,'F_name'),('gate0',a.g0,'D_alt')):
        lev,tag=spec.split(':');X=S.load(lev,tag)
        if not X:md.append(f'{name} ({spec}): missing\n');continue
        c=S.curve(*X);rows=[]
        md.append(f'\n## {name} ({spec})\n\n| point | F_name - F_null gain | D (R1) | D_alt (R1-alt) | acc/flip | max canary rise |\n|---|---|---|---|---|---|')
        for ep,s,sa,d in c:
            assert len(s['F_name'])==len(s['F_null']);rows.append({'point':ep,'F_name':AL.ci(s['F_name']-s['F_null']),'F_name_raw':AL.ci(s['F_name']),'F_null':AL.ci(s['F_null']),'D':AL.ci(s['D']),'D_alt':AL.ci(sa['D']) if sa else None,'acc':s['acc'],'flip':s['flip'],'canary_rise':d})
            r=rows[-1];md.append(f"| {ep} | {AL.f3(r['F_name'])} | {AL.f3(r['D'])} | {AL.f3(r['D_alt']) if sa else '-'} | {s['acc']:.2f}/{s['flip']:.2f} | {d:+.3f} |")
        last=rows[-1]
        if name=='gate1':v=last['F_name'];ok=last['point']==1 and v[1]>0;rule='F_name - F_null gain at pass 1, CI lower > 0'
        else:v=last['D_alt'];ok=last['point']==32 and v is not None and v[1]>0.5;rule='D_alt at epoch 32, CI lower > 0.5'
        md.append(f"\n**{name.upper()} {'PASS' if ok else 'FAIL'}** ({rule}: {AL.f3(v) if v else '-'} at point {last['point']})")
        res[name]={'spec':spec,'rows':rows,'pass':bool(ok)}
    json.dump(res,open(out/f'gates_s{a.seed}.json','w'),indent=1)
    (out/f'gates_report_s{a.seed}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
    return res
if __name__=='__main__':main()
