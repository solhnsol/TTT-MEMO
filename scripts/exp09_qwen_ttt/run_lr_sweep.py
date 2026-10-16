"""Exp09 lr sweep: lr in {1e-4, 3e-4, 1e-3, 3e-3} x {L0mp, L0mw}, world seed 101, 8 passes, worlds A and B, all other
settings at the qwen_ladder.py defaults (or as passed after `--`). One qwen_ladder.py process per (level, lr), runs
in sequence, existing outputs are skipped (resumable). Outputs: {OUT}/{LEVEL}_lr{LR}_s101_{A,B}.json + meta files.
Selection rule (approved 2026-10-16), per condition (level): the LARGEST lr whose canary rise at pass 1 is <= 0.10 nat,
where canary rise = max over worlds A/B and canaries (_gen_nll, _tail_nll) of value(pass 1) - value(pass 0) (the
analyze_stage01 definition). Every lr is reported (canary rise at pass 1 and its max over passes, binding D under R1 and
R1-alt at pass 1 and at the last pass, acc/flip) in {OUT}/lr_sweep_report_s{SEED}.md and lr_sweep_s{SEED}.json.
Usage: python run_lr_sweep.py [--out DIR] [--lrs 1e-4,3e-4,1e-3,3e-3] [--levels L0mp,L0mw] [--passes 8]
                              [--report-only] [-- extra qwen_ladder args]"""
import os,sys,json,argparse,subprocess,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
THR=0.10
def tag(lr):return 'lr'+f'{lr:g}'.replace('-0','-')
def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv);extra=[]
    if '--' in argv:i=argv.index('--');extra=argv[i+1:];argv=argv[:i]
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'experiments/exp09_qwen_ttt/lr_sweep'))
    ap.add_argument('--lrs',default='1e-4,3e-4,1e-3,3e-3');ap.add_argument('--levels',default='L0mp,L0mw')
    ap.add_argument('--passes',type=int,default=8);ap.add_argument('--world-seed',type=int,default=101);ap.add_argument('--report-only',action='store_true')
    a=ap.parse_args(argv);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    lrs=[float(x) for x in a.lrs.split(',')];levels=a.levels.split(',')
    if not a.report_only:
        for lev in levels:
            for lr in lrs:
                cmd=[sys.executable,str(HERE/'qwen_ladder.py'),'--level',lev,'--lr',repr(lr),'--tag',tag(lr),'--world-seed',str(a.world_seed),
                     '--passes',str(a.passes),'--out',str(out)]+extra
                print('RUN',' '.join(cmd[2:]),flush=True);r=subprocess.run(cmd)
                if r.returncode:print(f'FAILED {lev} lr {lr} (exit {r.returncode}); continuing',flush=True)
    return report(out,levels,lrs,a.world_seed)
def report(out,levels,lrs,seed):
    # analyze_ladder needs meta_L1_<tag>_s<seed>.json for items/fam_sizes: every level's meta has the same items
    metas=sorted(out.glob(f'meta_*_s{seed}.json'));ref=out/f'meta_L1_sweepitems_s{seed}.json'
    if not metas:print('no runs found');return {}
    if not ref.exists():shutil.copy(metas[0],ref)
    sys.path.insert(0,str(HERE));import analyze_qwen
    specs=[f'{lev}:{tag(lr)}' for lev in levels for lr in lrs]
    res=analyze_qwen.main(['--seed',str(seed),'--out',str(out),'--l1-tag','sweepitems']+specs)
    md=[f'# Exp09 lr sweep, seed {seed}\n',f'Rule: per level, the largest lr whose canary rise at pass 1 is <= {THR} nat (max over worlds A/B and _gen_nll/_tail_nll).\n']
    summ={}
    for lev in levels:
        md.append(f'\n## {lev}\n\n| lr | canary rise pass 1 | max rise (all passes) | D R1 pass 1 | D R1-alt pass 1 | D R1 last | D R1-alt last | acc/flip last | eligible |\n|---|---|---|---|---|---|---|---|---|')
        rows=[]
        for lr in lrs:
            c=res.get(f'{lev}:{tag(lr)}')
            if not c or len(c)<2:md.append(f'| {lr:g} | missing | | | | | | | |');continue
            p1=next(x for x in c if x['point']==1);last=c[-1];rise1=p1['canary_rise'];mx=max(x['canary_rise'] for x in c[1:])
            f=lambda t:f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]' if t else '-'
            ok=rise1<=THR;rows.append({'lr':lr,'rise_pass1':rise1,'max_rise':mx,'D_pass1':p1['D'],'D_alt_pass1':p1['D_alt'],'D_last':last['D'],'D_alt_last':last['D_alt'],
                                       'acc_last':last['acc'],'flip_last':last['flip'],'last_pass':last['point'],'eligible':ok})
            md.append(f"| {lr:g} | {rise1:+.3f} | {mx:+.3f} | {f(p1['D'])} | {f(p1['D_alt'])} | {f(last['D'])} | {f(last['D_alt'])} | {last['acc']:.2f}/{last['flip']:.2f} | {'yes' if ok else 'no'} |")
        el=[r['lr'] for r in rows if r['eligible']];sel=max(el) if el else None
        md.append(f"\n**Selected lr for {lev}: {sel:g}**" if sel is not None else f'\n**{lev}: no lr satisfies the rule (all rise > {THR} at pass 1); lower the grid**')
        if sel is not None and sel==max(lrs):md.append(f'(the largest lr in the grid is eligible: the grid may be too low)')
        summ[lev]={'selected_lr':sel,'rows':rows}
    json.dump(summ,open(out/f'lr_sweep_s{seed}.json','w'),indent=1);(out/f'lr_sweep_report_s{seed}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
    return summ
if __name__=='__main__':main()
