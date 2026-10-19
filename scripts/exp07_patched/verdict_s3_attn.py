"""S3 criterion (2) (attn-sel) per the registered text + 10/17 amendments, 3 seeds, same code as the stage-2 analysis (analyze_s3_stage2.py sha e13fcab25a84980f):
(2) [A(attn-sel) - A(native)] - 0.8 [A(prefix-all-pos) - A(native)] (masked-fs version alongside) CI lower > 0; attn-sel beats rand-dist@attn AND recency@attn on the area (each CI lower > 0); secondary-metric rule at pass 4 vs masked-fs (long-context beyond-window NLL primary, tail and first chunk non-inferior with delta_m 0.0098 / 0.0107).
K_c per seed and test from the canaries of all cells entering the test and the pass-4 Q2; area over passes 1..K_c; pairs resampled within seed. Also R_rec for attn-sel (descriptive) and per-pass D."""
import sys,os,json
sys.argv=['x'];sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np,analyze_s3_stage2 as A
OUT=A.OUT;cells=['native','masked','prefixall','oracle','attnsel','randattn','recencyattn'];data={};q2={}
def lf(sd,c):return OUT/(f'longnll_S3_{c}_p4_s{sd}.json' if c not in A.CELLS else (f'longnll_L0fs_fskeep_s{sd}.json' if c=='native' else f'longnll_L0mwfs_x1_s{sd}.json'))
for sd in (101,102,103):
    AL,S=A.mods(sd);data[sd]={}
    for c in cells:
        data[sd][c]=A.curve(AL,S,*A.CELLS[c]) if c in A.CELLS else A.curve(AL,S,'S3',f'{c}_p4')
        d=json.load(open(lf(sd,c)));q2[(sd,c)]=float(np.mean([[e for e in d[w] if e['epoch']==4][0]['rise_beyond'] for w in 'AB']))
print('per-cell D_alt by pass (mean over 15 pairs) and canary / Q2, seeds 101 / 102 / 103')
for c in cells:
    for sd in (101,102,103):
        D,can=data[sd][c];print(f'  {c:11s} s{sd}: '+' '.join(f'{D[p].mean():+.3f}' for p in range(1,5))+' | canary '+'/'.join(f'{can[p]:+.3f}' for p in range(1,5))+f' | Q2 {q2[(sd,c)]:+.3f}')
def run(cs,fn,label):
    v,ks=A.per_seed(data,q2,cs,fn)
    if not v:print(label,': no usable seed',ks);return None
    pt,lo,hi=A.boot(v);print(f'{label}: pooled {pt:+.3f} [{lo:+.3f},{hi:+.3f}]; per seed {{{", ".join(f"{s}: {x.mean():+.3f}" for s,x in v.items())}}}; K_c {ks}');return pt,lo,hi,v
r2=run(['native','attnsel','prefixall'],lambda a:(a['attnsel']-a['native'])-0.8*(a['prefixall']-a['native']),'(2) A(attn-sel)-A(native)-0.8[A(prefix-all)-A(native)]')
run(['native','attnsel','masked'],lambda a:(a['attnsel']-a['native'])-0.8*(a['masked']-a['native']),'(2) masked-fs version')
rr=run(['attnsel','randattn'],lambda a:a['attnsel']-a['randattn'],'attn-sel - rand-dist@attn')
rc=run(['attnsel','recencyattn'],lambda a:a['attnsel']-a['recencyattn'],'attn-sel - recency@attn')
cs=['native','attnsel','prefixall'];v,ks=A.per_seed(data,q2,cs,lambda a:a['attnsel']-a['native']);w,_=A.per_seed(data,q2,cs,lambda a:a['prefixall']-a['native']);sds=sorted(set(v)&set(w));rt=A.ratio_boot({s:v[s] for s in sds},{s:w[s] for s in sds});print(f'desc R_rec(attn-sel) = [A(attn-sel)-A(native)]/[A(prefix-all)-A(native)] {rt[0]:.3f} [{rt[1]:.3f},{rt[2]:.3f}]; K_c {ks}')
run(['native','attnsel','oracle'],lambda a:a['attnsel']-a['oracle'],'desc attn-sel - oracle')
print('secondary metrics at pass 4, attn-sel minus masked-fs (lower = attn-sel better); per seed; mean; t-CI(df=2)')
def vals(sd,c):
    d=json.load(open(lf(sd,c)));e={w:[x for x in d[w] if x['epoch']==4][0] for w in 'AB'};w0=d['W0'];out={'beyond':np.mean([e[w]['rise_beyond'] for w in 'AB']),'chunk0':np.mean([e[w]['chunk0']-w0['chunk0'] for w in 'AB'])}
    base=(OUT/('L0fs_main' if c=='native' else 'L0mwfs_x1')) if c in A.CELLS else (OUT/f'S3_{c}_p4');t=[]
    for w in 'AB':
        fn=OUT/(f'{base.name}_s{sd}_{w}.json');L=[x for x in json.load(open(fn))['log'] if 'margins' in x];t.append(L[4]['margins']['_tail_nll']-L[0]['margins']['_tail_nll'])
    out['tail']=np.mean(t);return out
for met,dm in (('beyond',None),('tail',0.0098),('chunk0',0.0107)):
    diffs=[vals(sd,'attnsel')[met]-vals(sd,'masked')[met] for sd in (101,102,103)];d=np.mean(diffs);se=np.std(diffs,ddof=1)/np.sqrt(3);lo,hi=d-4.303*se,d+4.303*se
    print(f'  {met:7s} {[round(float(x),4) for x in diffs]} mean {d:+.4f}, t-CI [{lo:+.4f},{hi:+.4f}]'+('' if dm is None else f'; non-inferior (CI upper < delta_m {dm}): {hi<dm}'))
