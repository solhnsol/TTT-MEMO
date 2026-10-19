"""No-GPU diagnostics on the finished stage-2 cells of s101 and s102 (design 14:05 + Hans 14:08 KST 10/18; s103 deliberately not opened): (a) per-pass R_rec = [D(oracle)-D(native)]/[D(prefix-all)-D(native)] pooled and per seed with CI;
(b) epsilon numerator confirmation is in the report text; (c) per cell and pass: mean update norm per chunk, clip fraction (c_k < 1), path length (cumulative sum of per-chunk update norms) for native, oracle, rand-dist, recency, prefix-all-pos;
(d) per-seed D_alt of rand-dist, oth-ent-sub, recency next to native (area over passes 1..4 and pass 4). Descriptive; interim (two seeds)."""
import sys,os,json
sys.argv=['x'];sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np,analyze_s3_stage2 as A
OUT=A.OUT;cells=['native','oracle','randdist','recency','prefixall','oraclesub','othentsub'];data={};q2={}
for sd in (101,102):
    AL,S=A.mods(sd);data[sd]={}
    for c in cells:data[sd][c]=A.curve(AL,S,*A.CELLS[c]) if c in A.CELLS else A.curve(AL,S,'S3',f'{c}_p4')
print('(a) per-pass R_rec = [D(oracle)-D(native)] / [D(prefix-all-pos)-D(native)] (R1-alt, per-pair means; ratio of means; pair bootstrap)')
for p in (1,2,3,4):
    num={sd:data[sd]['oracle'][0][p]-data[sd]['native'][0][p] for sd in data};den={sd:data[sd]['prefixall'][0][p]-data[sd]['native'][0][p] for sd in data}
    rt=A.ratio_boot(num,den);ps=[]
    for sd in data:
        r=A.ratio_boot({sd:num[sd]},{sd:den[sd]});ps.append(f's{sd} {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] (den CI [{r[3]:+.3f},{r[4]:+.3f}])')
    print(f'  pass {p}: pooled {rt[0]:.3f} [{rt[1]:.3f},{rt[2]:.3f}] (denominator CI [{rt[3]:+.3f},{rt[4]:+.3f}]); '+'; '.join(ps))
print('(c) update statistics per pass (mean per-chunk update norm | clip fraction c_k<1 | path length = cumulative sum of update norms), s101 / s102, worlds A,B averaged')
for c in ['native','oracle','randdist','recency','prefixall']:
    for sd in (101,102):
        rows=[]
        for w in 'AB':
            if c=='native':
                L=[e for e in json.load(open(OUT/f'L0fs_fskeep_s{sd}_{w}.saved.json'))['log'] if 'clip' in e][:4];cum=[e['cum_update_norm'] for e in L];prev=[0]+cum[:-1]
                rows.append([((cum[i]-prev[i])/63,np.mean([x<1 for _,x in L[i]['clip']]),cum[i]) for i in range(4)])
            else:
                L=json.load(open(OUT/f'S3_{c}_p4_s{sd}_{w}.saved.json'))['log'][1:5]
                rows.append([(np.mean([s['update_norm'] for s in e['chunks']]),np.mean([s['coef']<1 for s in e['chunks']]),e['cum_update_norm']) for e in L])
        print(f'  {c:9s} s{sd}: '+' | '.join('p%d %.3f %.2f %.1f'%(i+1,*np.mean([r[i] for r in rows],axis=0)) for i in range(4)))
print('(d) per-seed D_alt next to native: area over passes 1..4 [95% pair CI] and pass 4')
for sd in (101,102):
    AL,S=A.mods(sd)
    for c in ['native','randdist','recency','othentsub','oraclesub','oracle']:
        ar=A.area(data,sd,c,4);pt,lo,hi=A.boot({sd:ar});print(f'  s{sd} {c:10s} area {pt:+.3f} [{lo:+.3f},{hi:+.3f}]  pass4 {data[sd][c][0][4].mean():+.3f}')
    for c in ['randdist','recency','othentsub']:
        d=A.area(data,sd,c,4)-A.area(data,sd,'native',4);pt,lo,hi=A.boot({sd:d});print(f'  s{sd} {c} - native: area {pt:+.3f} [{lo:+.3f},{hi:+.3f}]')
