"""Answer-matched binding vs number of exposures (16/32/64/128) on the stored synthetic worlds and FictionalQA worlds (CPU; new file; thread-server 11/25; descriptive: seed-level t 95% intervals; no thresholds). Definition (as e63_e62_match.py): fact f with answer a_f: MATCHED = dS[f, a_f] - mean over the OTHER fact questions q of dS[q, a_f] (the same word scored as gold in its own question and as a wrong candidate in the others; dS = candidate score of the arm's state minus W0 of the same read); OLD = dS[f, gold] - mean dS[f, other candidates]. 'same-type' (FQ only): the other questions are restricted to facts with the same question word (What / Who / Where / When / How ...). FAMILIARITY (exposed - unexposed): the fact items of these worlds have NO never-exposed candidates; the 4 ctrl items (unexposed name, candidate 0 = never-exposed place in FQ) give a control: FAM-ctrl = mean over exposed candidates of the ctrl item - never-exposed candidate (FQ only; 'none' for the synthetic worlds whose ctrl items have exposed candidates only). Runs: E2E synth = job 1321 (e59_4way, seeds 231..238; NAT, MASK1024, CHK, ISO); Qwen synth = job 1322 (q4way, seeds 221..228; NAT, MASK1024, CHK, ISO); E2E FQ = job 1313 (NATS, NATMASK) + 1323 (CHK, ISO) seeds 211..218; Qwen FQ = job 1320/1324 (fq_qwen: NAT, BAND1024, BANDALL1024, CHK, ISO) seeds 211..218. python e63_curve_match.py"""
import json,random,collections,numpy as np
from scipy import stats
R='experiments/exp07_write_ladder/';Q='experiments/exp09_qwen_ttt/';PTS=('16','32','64','128')
def tci(x):
    x=np.asarray(x,float);m=x.mean()
    if len(x)<2:return f'{m:7.2f}'
    h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.2f} [{m-h:6.2f},{m+h:6.2f}]'
def synth_cands(seed):
    """candidate fact ids of the 16 fact items of a v3_16_1 world (gen_facts.build_items for k = 16, p = 1): depends on (seed, fact ids) only"""
    out=[]
    for i in range(16):
        r=random.Random(f"E55f|{seed}|q|{i}");others=[g for g in range(16) if g!=i];r.shuffle(others);cl=[i]+others[:15];r.shuffle(cl);out.append(cl)
    return out
def fq_cands():return [[f]+[g for g in range(16) if g!=f] for f in range(16)]
def fq_types(label):
    d=json.load(open(f'{R}fq/worlds_w0c/fq_{label-10}_cap3_VAR_w0.json'));fa=d['facts'];return [fa[i]['question'].replace('Question:','').split()[0] for i in range(16)]
def stats_one(S,W0,cand,gold,same_type=None):
    """S, W0: [20,16] scores (fact items 0..15, ctrl items 16..19); -> dict of matched / old binding and FAM-ctrl"""
    dS=np.asarray(S,float)-np.asarray(W0,float);m=[];o=[];ms=[]
    for f in range(16):
        g=dS[f,gold[f]];o.append(g-np.mean(np.delete(dS[f],gold[f])))
        oth=[dS[q,cand[q].index(f)] for q in range(16) if q!=f];m.append(g-np.mean(oth))
        if same_type is not None:
            ot=[dS[q,cand[q].index(f)] for q in range(16) if q!=f and same_type[q]==same_type[f]]
            if ot:ms.append(g-np.mean(ot))
    return dict(matched=np.mean(m),old=np.mean(o),matched_same=(np.mean(ms) if ms else np.nan),n_same=np.mean([sum(1 for q in range(16) if q!=f and same_type[q]==same_type[f]) for f in range(16)]) if same_type is not None else np.nan)
def fam_ctrl(S,W0,meta):
    ex=meta.get('cand_exposed')
    if ex is None:return np.nan
    dS=np.asarray(S,float)-np.asarray(W0,float);v=[]
    for i in range(16,20):
        e=np.asarray(ex[i]);
        if e.all():return np.nan
        v.append(dS[i,e].mean()-dS[i,~e].mean())
    return float(np.mean(v))
def table(name,arms,loaders,seeds,header):
    print(f'\n=== {name} | {header}')
    res={}
    for arm,load in arms:
        for pt in PTS:
            xs=[]
            for s in seeds:
                S,W0,cand,gold,meta,types=load(s,pt);d=stats_one(S,W0,cand,gold,types);d['fam']=fam_ctrl(S,W0,meta);xs.append(d)
            res[(arm,pt)]={k:np.array([x[k] for x in xs]) for k in xs[0]}
    print(f'  columns per exposure (16 | 32 | 64 | 128): MATCHED bind mean [95% t over {len(seeds)} seeds]; OLD bind; FAM-ctrl; (FQ) same-type MATCHED [questions per fact]')
    for arm,_ in arms:
        for key,nm in (('matched','MATCHED'),('old','OLD    '),('fam','FAM-ctrl'),('matched_same','SAMETYPE')):
            if np.isnan(res[(arm,'16')][key]).all():
                if key=='fam':print(f'  {arm:12s} {nm}: none')
                continue
            extra=f' [{res[(arm,"16")]["n_same"].mean():.1f} q/fact]' if key=='matched_same' else ''
            print(f'  {arm:12s} {nm}: '+' | '.join(tci(res[(arm,pt)][key]) for pt in PTS)+extra)
    base=arms[0][0]
    print(f'  difference to {base} (seed-paired), MATCHED:')
    for arm,_ in arms[1:]:print(f'  {arm:12s} - {base}: '+' | '.join(tci(res[(arm,pt)]['matched']-res[(base,pt)]['matched']) for pt in PTS))
# ---------------- E2E synthetic (job 1321)
def e2e_synth(arm):
    def load(s,pt):
        d=json.load(open(f'{R}e59_4way/E59_a4_{arm}_s{s}.read.json'));m=d['meta'];return d['read'][pt]['out'],d['W0']['out'],m['cand_fact'][:16],m['gold_idx'][:16],m,None
    return load
SEEDS=list(range(231,239))
chk=json.load(open(f'{R}e59_4way/E59_a4_NAT_s231.read.json'))['meta'];assert [synth_cands(231)[i] for i in range(16)]==chk['cand_fact'][:16],'candidate reconstruction rule differs from the stored cand_fact'
table('E2E synthetic world (v3_16_1; job 1321, e59_4way)',[(a,e2e_synth(a)) for a in ('NAT','MASK1024','CHK','ISO')],None,SEEDS,'seeds 231..238; candidates = the 16 places of the 16 facts (own gold + 15 exposed, all with the same number of exposures: one sentence per fact per chunk), no never-exposed candidate; reads out-of-window neutral at 16/32/64/128 exposures')
# ---------------- Qwen synthetic (job 1322)
def qwen_synth(arm,base='q4way'):
    def load(s,pt):
        d=json.load(open(f'{Q}{base}/e59q_{arm}_s{s}.json'));w=json.load(open(f'{Q}{base}/cband_W0_s{s}.json'));cand=synth_cands(s);gold=[cand[i].index(i) for i in range(16)]
        assert gold==d['meta']['gold_idx'][:16],('gold_idx differs from the reconstructed candidates',s)
        return d['reads'][pt]['out'],w['reads'][pt]['out'],cand,gold,{},None
    return load
table('Qwen synthetic world (job 1322, q4way)',[(a,qwen_synth(a)) for a in ('NAT','MASK1024','CHK','ISO')],None,list(range(221,229)),'seeds 221..228; candidates = the 16 places (candidate order reconstructed from (seed, fact ids) and checked against the stored gold_idx); no never-exposed candidate')
# ---------------- E2E FQ
def e2e_fq(arm,path):
    def load(s,pt):
        d=json.load(open(path.format(arm=arm,s=s)));m=d['meta'];return d['read'][pt]['out'],d['W0']['out'],m['cand_fact'][:16],m['gold_idx'][:16],m,fq_types(s)
    return load
arms=[('NATS',e2e_fq('NATS',R+'fq/run/E55_fq_{arm}_s{s}.read.json')),('NATMASK',e2e_fq('NATMASK',R+'fq/run/E55_fq_{arm}_s{s}.read.json')),('CHK',e2e_fq('CHK',R+'e59_fq/E59_fq_{arm}_s{s}.read.json')),('ISO',e2e_fq('ISO',R+'e59_fq/E59_fq_{arm}_s{s}.read.json'))]
table('E2E FictionalQA (jobs 1313 + 1323)',arms,None,list(range(211,219)),'seeds 211..218; candidates = the 16 gold answers of the 16 read facts (all exposed equally), ctrl items (4): candidate 0 = never-exposed answer; same-type = same question word')
def qwen_fq(arm):
    def load(s,pt):
        d=json.load(open(f'{Q}fq_qwen/e59q_{arm}_s{s}.json'));w=json.load(open(f'{Q}fq_qwen/cband_W0_s{s}.json'));cand=fq_cands()
        meta=dict(cand_exposed=[[True]*16]*16+[[False]+[True]*15]*4)
        return d['reads'][pt]['out'],w['reads'][pt]['out'],cand,[0]*16,meta,fq_types(s)
    return load
table('Qwen FictionalQA (jobs 1320 + 1324)',[(a,qwen_fq(a)) for a in ('NAT','BAND1024','BANDALL1024','CHK','ISO')],None,list(range(211,219)),'seeds 211..218; candidates = the 16 gold answers (own first), ctrl items: candidate 0 = never-exposed answer; same-type = same question word')
