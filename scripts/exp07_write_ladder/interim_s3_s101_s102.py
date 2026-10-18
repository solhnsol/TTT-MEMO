"""INTERIM (requested by Hans 10/18 ~13:50 KST via the coordinator): D_alt (fp32 R1-alt) of the finished stage-2 cells on s101 and s102, per cell and pass with pair-bootstrap CI, canary, Q2, and the registered tests computed on these two seeds only.
NOT the registered verdict (that needs 3 seeds / the finished job 41)."""
import sys,os,json
sys.argv=['x'];os.environ.setdefault('EXP07_OUT','')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np,analyze_s3_stage2 as A
OUT=A.OUT;cells=['native','masked','prefixall','oracle','randdist','recency','oraclesub','othentsub'];names={'native':'native-fs','masked':'masked-fs','prefixall':'prefix-all-pos','oracle':'oracle-sel','randdist':'rand-dist@|O|','recency':'recency@|O|','oraclesub':'oracle-sub','othentsub':'oth-ent-sub'}
data={};q2={}
for sd in (101,102):
    AL,S=A.mods(sd);data[sd]={}
    for c in cells:
        data[sd][c]=A.curve(AL,S,*A.CELLS[c]) if c in A.CELLS else A.curve(AL,S,'S3',f'{c}_p4')
        lf=OUT/(f'longnll_S3_{c}_p4_s{sd}.json' if c not in A.CELLS else (f'longnll_L0fs_fskeep_s{sd}.json' if c=='native' else f'longnll_L0mwfs_x1_s{sd}.json'))
        d=json.load(open(lf));q2[(sd,c)]=float(np.mean([[e for e in d[w] if e['epoch']==4][0]['rise_beyond'] for w in 'AB']))
    AL_=AL
    print(f'--- seed {sd}: R1-alt D_alt mean over the 15 pairs [95% pair-bootstrap CI] by pass; canary (max rise) by pass; Q2 (beyond-window rise, pass 4)')
    for c in cells:
        D,can=data[sd][c];row=[f'{D[p].mean():+.3f} [{AL.ci(D[p])[1]:+.3f},{AL.ci(D[p])[2]:+.3f}]' for p in range(1,5)]
        print(f'{names[c]:15s} '+' | '.join(row)+' || canary '+'/'.join(f'{can[p]:+.3f}' for p in range(1,5))+f' || Q2 {q2[(sd,c)]:+.3f}')
A.DRY=False;data[103]={}
def run(cs,fn,label):
    v,ks=A.per_seed(data,q2,cs,fn)
    if not v:print(label,': no usable seed',ks);return
    pt,lo,hi=A.boot(v);print(f'{label}: pooled(s101+s102) {pt:+.3f} [{lo:+.3f},{hi:+.3f}]; per seed {{{", ".join(f"{s}: {x.mean():+.3f}" for s,x in v.items())}}}; K_c {ks}')
print('--- registered-style tests on s101 + s102 ONLY (interim, not the verdict)')
run(['native','oracle','prefixall'],lambda a:(a['oracle']-a['native'])-0.8*(a['prefixall']-a['native']),'(1) A(oracle)-A(native)-0.8[A(prefix-all)-A(native)]')
run(['native','oracle','masked'],lambda a:(a['oracle']-a['native'])-0.8*(a['masked']-a['native']),'(1) masked-fs version')
run(['oracle','randdist'],lambda a:a['oracle']-a['randdist'],'S1 A(oracle)-A(rand-dist@|O|)')
for ref in ('prefixall','masked'):
    cs=['native','oracle',ref];v,ks=A.per_seed(data,q2,cs,lambda a:a['oracle']-a['native']);w,_=A.per_seed(data,q2,cs,lambda a:a[ref]-a['native']);sds=sorted(set(v)&set(w))
    rt=A.ratio_boot({s:v[s] for s in sds},{s:w[s] for s in sds});print(f'S2 vs {ref}: R_rec {rt[0]:.3f} [{rt[1]:.3f},{rt[2]:.3f}]; denominator CI [{rt[3]:+.3f},{rt[4]:+.3f}]; K_c {ks}')
for a_,b_,n in (('masked','prefixall','masked-fs - prefix-all-pos'),('oracle','recency','oracle - recency@|O|'),('oraclesub','othentsub','oracle-sub - oth-ent-sub'),('oracle','masked','oracle - masked-fs'),('prefixall','native','prefix-all-pos - native-fs'),('oracle','native','oracle - native-fs'),('masked','native','masked-fs - native-fs')):
    run(['native',a_,b_] if b_!='native' else [a_,b_],lambda a,a_=a_,b_=b_:a[a_]-a[b_],'desc '+n)
