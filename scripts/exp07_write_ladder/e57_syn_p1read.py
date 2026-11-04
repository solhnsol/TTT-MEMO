"""E57 synthetic s64, P1 contrast write: per-token read of the states saved before the arms were stopped (new file; GPU read only; investigation thread, Hans card '읽기 먼저, 이후 상한 대조'): M3PB05W8N05 and M3PB10W8N05 at chunk 15 and 31 (appearance 16 / 32; the arms were stopped at chunk 45 by the nll' rule of job 900 and their saved.json holds no state list, so the files and the hashes printed in the log of job 900 are given here). The read is e57_syn_tokread.read_stage with an arm list override; the reproduction check of that stage compares with the inline read of M3W8N05 and is NOT meaningful for these states. Merge with the reads of M3W1N05 / M3W8N05 (job 800) and the curve report: python3 e57_syn_p1read.py --stage read|report --out O"""
import sys,json,pathlib,argparse
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder'
import e57_syn_tokread as TR
STATES={'M3PB05W8N05':{15:'6c814811d1ff3f9cf9eba36ffaf39a36971ab81a176624e2a587ace5fc425f5b',31:'a3490d04789612967f9d5671a8d5622a4d86d2b0e3c8f9c0eae5a6510cce176d'},
        'M3PB10W8N05':{15:'97e2beecaa7baac9560aef0f6beff2f57a6fef54407ded8b5338e006182ba0f8',31:'432d4df55b18bd070aed402255a0929dc9164ba9e5a3773a1703c274b0d68dec'}}
def arms():
    out={}
    for a,d in STATES.items():
        for ch,h in d.items():
            f=EXP/f'e57_synq/states/E55_synqp1_{a}_s64/state_{a}_s64_{ch:03d}.npz';assert f.exists(),f
            out[f'{a}@c{ch}']=dict(file=str(f),hash=h,read_json=str(EXP/'e57_synq/E55_synqc_M3W8N05_s64.read.json'),chunk=ch)
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage');ap.add_argument('--out');a=ap.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.stage=='read':
        TR.arms_curve=lambda seed,tag,only=None:arms();TR.read_stage(dict(out=str(out),seed='64',tag='x',arms=None))
    else:
        d=json.load(open(out/'e57_syn_tokread_s64.json'));c=json.load(open(EXP/'e57_syn_curve_s64/e57_syn_tokread_s64.json'));assert d['tokids']==c['tokids']
        mx=max(abs(sum(x)-sum(y)) for x,y in zip(d['W0'],c['W0']));assert mx<1e-3,mx
        keep={k:v for k,v in c['arms'].items() if k.split('@c')[1] in ('15','31')};d['arms'].update(keep);d['W0_old_vs_new_max_abs_diff_sum']=mx;(out/'e57_syn_tokread_s64_all.json').write_text(json.dumps(d));print('merged',sorted(d['arms']))
if __name__=='__main__':main()
