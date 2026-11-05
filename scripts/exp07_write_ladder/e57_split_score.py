"""E57 split scoring of the rule arms with the FROZEN criterion B (new file; CPU; GPM thread, Hans 11/05 '일단 B로 가자'). The criterion must not change after results come in (CRITERION_B_SHA below).
Primary metric: the branch-token rank ratio rr (rank of the gold at the first differing token among its candidates / number of candidates; lower is better); improvement = rr(ref) - rr(rule) for a rule compared with a reference (positive = the rule is better; W0 cancels). Per source the mean over its units (QA: items; news: v2A questions; synthetic: facts, one value per world), then the EQUAL-WEIGHT mean across the sources QA, news, synthetic (the synthetic source = mean of its worlds).
CRITERION B for a rule R (ABS tau 2 / 4 / 6 nats, loss weight 16) on a part (tune or hold-out):
  B1  R beats ALLQ (ALLQ256 synthetic, ALLQ128 QA / news): the equal-weight mean improvement R - ALLQ is > 0 and its bootstrap 95% CI excludes 0
  B2  against M3 w=8 (top 30%): the lower bound of the 95% CI of the equal-weight mean improvement R - M3 is > -0.02
  B3  the direction (R better than ALLQ) agrees in at least 2 of the 3 sources; the synthetic source agrees when at least 2 of its 3 worlds (densities) agree
  B4  no source is significantly worse than ALLQ (the 95% CI of its R - ALLQ difference does not lie entirely below 0)
  B5  out-of-window reading (synthetic inline read 'out' at a = 128; QA and news have no separate one): the mean improvement R - ALLQ is not below 0
  canary: reference only; fixed arm with the same share (nearest available: top 5% / top 30%): direction only (sign and the difference, no test)
  meets_B = B1 and B2 and B3 and B4 and B5. Bootstrap: 2000 draws, units resampled within each source (and world), same draw for all arms (paired).
  python3 e57_split_score.py --part tune|hold [--qa] [--news] [--syn 62:v3_16_1,71:v3_4_1,72:v3_32_1] --out J | --selftest"""
import sys,json,argparse,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder'
CRITERION_B='B1 beats ALLQ (mean diff>0, CI excludes 0); B2 vs M3 w=8 CI lower>-0.02; B3 direction agrees in >=2 of 3 sources (synthetic: >=2 of 3 worlds); B4 no source significantly worse than ALLQ; B5 out-of-window not lower than ALLQ; canary reference only'
CRITERION_B_SHA=hashlib.sha256(CRITERION_B.encode()).hexdigest()[:12]
FAM={'ABS2':dict(qa='M3qtA2W16Q128',news='NEWSqtA2W16',syn='M3TA2W16N05'),'ABS4':dict(qa='M3qtA4W16Q128',news='NEWSqtA4W16',syn='M3TA4W16N05'),'ABS6':dict(qa='M3qtA6W16Q128',news='NEWSqtA6W16',syn='M3TA6W16N05'),'TOP5':dict(qa='M3qt05W16Q128',news='NEWSqt05W16',syn='M3T05W16N05'),'M3':dict(qa='M3W8Q128',news='M3W8Q128',syn='M3W8N05'),'ALLQ':dict(qa='ALLQ128',news='ALLQ128',syn='M3W1N05')}
RULES=('ABS2','ABS4','ABS6');B=2000
def ci(x):x=np.asarray(x,float);return dict(mean=float(x.mean()),ci95=[float(np.percentile(x,2.5)),float(np.percentile(x,97.5))])
class Src:
    """one source: name, kind ('qa' | 'news' | 'syn'), worlds = list of dicts {fam: rr array [units]} (qa / news: one world), optional out[world][fam] = out-of-window rr arrays"""
    def __init__(s,name,worlds,out=None,share=None):s.name=name;s.worlds=worlds;s.out=out;s.share=share or {}
def boot_diff(sources,a,b,seed=7):
    """paired bootstrap of the improvement of arm family a over b (rr_b - rr_a): per world, per source (mean over its worlds), overall (equal-weight mean over sources)"""
    rng=np.random.default_rng(seed);per={};ov=np.zeros(B);ns=0;idx=[]
    for s in sources:
        wv=[];pw=[]
        for w in s.worlds:
            if a not in w or b not in w:continue
            n=len(w[a]);ix=rng.integers(0,n,(B,n));d=np.asarray(w[b],float)-np.asarray(w[a],float);wv.append(d[ix].mean(1));pw.append(float(d.mean()))
        if not wv:continue
        sv=np.mean(wv,0);per[s.name]=dict(**ci(sv),point=float(np.mean(pw)),worlds=pw);ov+=sv;ns+=1
    return per,(dict(**ci(ov/ns),n_sources=ns) if ns else None)
def oow_diff(sources,a,b):
    v=[]
    for s in sources:
        if s.out:
            for w in s.out:
                if a in w and b in w:v.append(float(np.mean(np.asarray(w[b])-np.asarray(w[a]))))
    return float(np.mean(v)) if v else None
def evaluate(sources,part):
    res=dict(part=part,criterion_B=CRITERION_B,criterion_B_sha=CRITERION_B_SHA,rules={})
    for r in RULES:
        pa,oa=boot_diff(sources,r,'ALLQ');pm,om=boot_diff(sources,r,'M3')
        if oa is None:continue
        dirs=[];
        for s in sources:
            if s.name not in pa:continue
            if len(s.worlds)>1:dirs.append(sum(1 for x in pa[s.name]['worlds'] if x>0)>=2)
            else:dirs.append(pa[s.name]['point']>0)
        b1=oa['ci95'][0]>0;b2=om is not None and om['ci95'][0]>-0.02;b3=sum(dirs)>=2;b4=not any(v['ci95'][1]<0 for v in pa.values());oow=oow_diff(sources,r,'ALLQ');b5=(oow>=0) if oow is not None else None
        fx={}
        for f in ('TOP5','M3'):
            p_,o_=boot_diff(sources,r,f)
            if o_:fx[f]=dict(mean_improvement=o_['mean'],direction='rule better' if o_['mean']>0 else 'rule not better')
        res['rules'][r]=dict(vs_ALLQ=dict(overall=oa,per_source=pa),vs_M3=dict(overall=om,per_source=pm),direction_agree_sources=dirs,out_of_window_diff_vs_ALLQ=oow,B1=b1,B2=b2,B3=b3,B4=b4,B5=b5,meets_B=bool(b1 and b2 and b3 and b4 and (b5 is not False)),fixed_arm_direction_only=fx)
    return res
def qa_source(part,n=65,newdir='e57_public_qa_n65_rules',olddir='e57_public_qa_n65'):
    import e57_pqa15_run as P15,e57_pqa2_stream as S2
    tk=S2.tokenizer();dec=lambda t:tk.decode(t);new=EXP/newdir;old=EXP/olddir;meta=json.load(open(old/'e57_pqa15_read_meta.json'));ids=meta['item_ids'];tok=json.load(open(old/'e57_pqa15_tokens.json'))
    sp={r['item_id']:r['split'] for r in json.load(open(EXP/'e57_qa_split.json'))['items']};m=np.array([sp[i]==part for i in ids]);w={}
    for fam,nm in FAM.items():
        arm=nm['qa'];d=new if (new/f'E57_pqa15_rows_A_{arm}_n{n}.json').exists() else old
        f=d/f'E57_pqa15_rows_A_{arm}_n{n}.json'
        if f.exists():w[fam]=P15.item_arrays(json.load(open(f)),ids,tok['A'],dec,'q')['rr_branch'][m]
    return Src('QA',[w])
def news_source(part,s='v2A'):
    import e57_news_tokread as NT
    new=json.load(open(EXP/'e57_news_rules/read/e57_news_tokread.json'));old=json.load(open(EXP/'e57_news_tokread/e57_news_tokread.json'));sp=json.load(open(EXP/'e57_news_split.json'));ev=set(sp['events_'+part]);meta=new['meta'][s];T=new['tokens'][s];To=old['tokens'][s]
    m=np.array([it['event'] in ev for it in meta]);F=lambda tl:np.array([NT.frame_metrics(t,mm,'all')['rr_branch'] for t,mm in zip(tl,meta)])[m];w={}
    for fam,nm in FAM.items():
        arm=nm['news'];lab=f'{arm}@c279'
        if lab in T:w[fam]=F(T[lab])
        elif arm in To:w[fam]=F(To[arm])
    return Src('news',[w])
def syn_source(spec):
    import e57_syn_tokread as TR
    worlds=[];outs=[]
    for x in [x for x in spec.split(',') if x]:
        sd,cell=x.split(':');d=json.load(open(EXP/f'e57_syn_rules_s{sd}/e57_syn_tokread_s{sd}.json'));rj=json.load(open(next(EXP.glob(f'e57_synq/E55_synqrl_*_s{sd}.read.json'))))['meta'];facts=sorted(TR.metrics(d['W0'],d['tokids'],rj));w={};o={}
        kind=np.asarray(rj['kind']);gi=np.asarray(rj['gi']);rows=[r for r in range(len(kind)) if kind[r]=='fact']
        for fam,nm in FAM.items():
            lab=f"{nm['syn']}@c127"
            if lab in d['arms']:
                M=TR.metrics(d['arms'][lab],d['tokids'],rj);w[fam]=np.array([M[f]['rank_branch']/16 for f in facts])
            f_=EXP/f"e57_synq/E55_synqrl_{nm['syn']}_s{sd}.read.json"
            if f_.exists():
                x_=json.load(open(f_));sc=np.asarray(x_['read']['128']['out'],float);o[fam]=np.array([(1+(sc[r]>sc[r][gi[r]]).sum())/sc.shape[1] for r in rows])
        worlds.append(w);outs.append(o)
    return Src('synthetic',worlds,out=outs)
def selftest():
    r=np.random.default_rng(0);n=40
    def mk(shift):base=r.random(n)*0.5+0.3;return {'ALLQ':base,'M3':base-0.05,'ABS4':base-shift,'ABS2':base-0.0,'ABS6':base+0.2}
    src=[Src('QA',[mk(0.1)]),Src('news',[mk(0.1)]),Src('synthetic',[mk(0.1),mk(0.1),mk(0.1)],out=[{'ALLQ':np.ones(5)*.5,'ABS4':np.ones(5)*.4},{'ALLQ':np.ones(5)*.5,'ABS4':np.ones(5)*.4},{'ALLQ':np.ones(5)*.5,'ABS4':np.ones(5)*.4}])]
    e=evaluate(src,'tune');assert e['rules']['ABS4']['B1'] and e['rules']['ABS4']['B3'] and e['rules']['ABS4']['B5'] and not e['rules']['ABS6']['B1'] and not e['rules']['ABS6']['B4'] and not e['rules']['ABS2']['B1']
    assert len(CRITERION_B_SHA)==12;print('e57_split_score selftest ok, criterion B sha',CRITERION_B_SHA)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',default='tune');ap.add_argument('--qa',action='store_true');ap.add_argument('--news',action='store_true');ap.add_argument('--syn',default='');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    src=[]
    if a.qa:src.append(qa_source(a.part))
    if a.news:src.append(news_source(a.part))
    if a.syn:src.append(syn_source(a.syn))
    res=evaluate(src,a.part);pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out,'criterion B sha',CRITERION_B_SHA)
if __name__=='__main__':main()
