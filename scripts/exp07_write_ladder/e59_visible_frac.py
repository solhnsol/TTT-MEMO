"""Visible share of earlier fact sentences under the WRITE-PATH mask (CPU; new file; thread-server 11/19; request for the binding-accumulation formula): for the sentence rows of exposure e (chunk r = e - 1, every fact sentence of that chunk, query tokens q = absolute positions of the sentence tokens) the key tokens are the tokens of EARLIER fact sentences inside the attention window (a) of the SAME fact (earlier chunks) and (b) of OTHER facts (earlier chunks and the earlier sentences of the current chunk). Pairs (q, key token) are counted; share visible = visible pairs / pairs inside the window under the arm's mask:
 NATS / NAT 1; band b (NATMASK ALLQ<b>, every row): q - k < b; ORACLE: same-fact keys hidden, other-fact keys visible; ALLPREV: every earlier fact sentence of the EARLIER chunks hidden (sentences of the current chunk stay visible; e57_allprev_masks.py); ISO: no earlier sentence at all (0 / 0); CHK (chunk alone): earlier chunks hidden, sentences of the current chunk visible. Window of the E2E arms: distance < 8192 (SWA, e55_swap / s3 masks). OLMo (writer window of 7 previous chunks + the current one; sliding layers distance < 4096): 'olmo_full' = keys in chunks >= r - 7, 'olmo_sliding' = distance < 4096.
  python3 e59_visible_frac.py --out DIR [--only e2e71|e2e231|olmo|dens|all] | --selftest"""
import sys,json,argparse,pathlib,types
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];EXP=ROOT/'experiments/exp07_write_ladder'
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics',ROOT/'scripts/exp09_qwen_ttt'):sys.path.insert(0,str(p))
CH=1024
def spans(chunks):
    """list of (chunk r, fact, abs start, abs end) in stream order"""
    return [(r,int(s['fact']),r*CH+int(s['start']),r*CH+int(s['end'])) for r,c in enumerate(chunks) for s in sorted(c['sents'],key=lambda x:x['start'])]
def window_ok(rule,q,k,rq,rk):
    d=q[:,None]-k[None,:]
    if rule=='swa8192':return (d>0)&(d<8192)
    if rule=='olmo_sliding':return (d>0)&(d<4096)
    if rule=='olmo_full':return (d>0)&(rk>=rq-7)
    raise ValueError(rule)
def arm_visible(arm,q,k,same,rq,rk,band=None):
    """bool [nq,nk] visibility of key tokens for query tokens under the arm (window not yet applied)"""
    d=q[:,None]-k[None,:]
    if arm in('NATS','NAT'):return np.ones_like(d,bool)
    if arm.startswith('band') or arm in('NATMASK256','MASK1024'):return d<band
    if arm=='ORACLE':return ~same
    if arm=='ALLPREV':return np.full(d.shape,rk==rq)
    if arm=='CHK':return np.full(d.shape,rk==rq)
    if arm=='ISO':return np.zeros(d.shape,bool)
    raise ValueError(arm)
BANDS={'NATMASK256':256,'band64':64,'band512':512,'band768':768,'band1024':1024,'band2048':2048,'band4096':4096,'MASK1024':1024}
def shares(chunks,exposures,arms,rule='swa8192'):
    sp=spans(chunks);out={a:{} for a in arms}
    for e in exposures:
        r=e-1;q_sents=[s for s in sp if s[0]==r];acc={a:[0,0,0,0] for a in arms}      # same_vis, same_tot, other_vis, other_tot
        for (rq,fq,a0,a1) in q_sents:
            q=np.arange(a0,a1)
            for (rk,fk,b0,b1) in sp:
                if rk>rq or (rk==rq and b0>=a0):continue
                k=np.arange(b0,b1);same=(fk==fq)
                if rule=='olmo_full' and rk<rq-7:continue
                if rule!='olmo_full' and a0-b1>=8192:continue
                inw=window_ok(rule,q,k,rq,rk)
                if not inw.any():continue
                for arm in arms:
                    vis=arm_visible(arm,q,k,np.full(inw.shape,same),rq,rk,BANDS.get(arm))&inw
                    if arm=='ORACLE':vis=inw&(not same)
                    i=0 if same else 2;acc[arm][i]+=int(vis.sum());acc[arm][i+1]+=int(inw.sum())
        for arm in arms:
            sv,st,ov,ot=acc[arm];out[arm][e]=dict(same_fact=(sv/st if st else None),other_fact=(ov/ot if ot else None),pairs_same=st,pairs_other=ot)
    return out
def world(seed,cell='v3_16_1',tokfile=None,read_apps=(16,32,64,128)):
    if tokfile is None:
        import e55_run as R55,e57_syn_pass as SP
        enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(read_apps),arms=['NAT']);return w
    import cell_band_ladder as CB
    enc,w,stream,sha,items=CB.build_world(tokfile,seed,cell,list(read_apps));return w
def avg(per_seed,arms,exposures):
    R={}
    for a in arms:
        R[a]={}
        for e in exposures:
            v=[ps[a][e] for ps in per_seed];R[a][str(e)]={k:(float(np.mean([x[k] for x in v if x[k] is not None])) if any(x[k] is not None for x in v) else None) for k in('same_fact','other_fact')}
    return R
def selftest():
    chunks=[dict(sents=[dict(fact=0,start=100,end=110),dict(fact=1,start=200,end=210)]) for _ in range(3)]
    s=shares(chunks,[3],['NATS','ISO','CHK','ORACLE','ALLPREV','band1024'])
    assert s['NATS'][3]['same_fact']==1 and s['NATS'][3]['other_fact']==1 and s['ISO'][3]['same_fact']==0 and s['ISO'][3]['other_fact']==0
    assert s['ORACLE'][3]['same_fact']==0 and s['ORACLE'][3]['other_fact']==1 and s['ALLPREV'][3]['same_fact']==0 and 0<s['ALLPREV'][3]['other_fact']<1 and s['CHK'][3]['other_fact']==s['ALLPREV'][3]['other_fact']
    assert 0<s['band1024'][3]['same_fact']<=1 and s['band1024'][3]['other_fact']<=1;print('e59_visible_frac selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--only',default='all');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);EX=(16,32,64,128);R={}
    if a.only in('all','e2e71'):
        arms=['NATS','NATMASK256','band64','band512','band768','band1024','band2048','band4096','ALLPREV','ORACLE','ISO'];ps=[shares(world(s)['chunks'],EX,arms) for s in range(71,79)];R['e2e_seeds71-78']=avg(ps,arms,EX);print('e2e71 done',flush=True)
    if a.only in('all','e2e231'):
        arms=['NAT','MASK1024','CHK','ISO'];ps=[shares(world(s)['chunks'],EX,arms) for s in range(231,239)];R['e2e_seeds231-238_job1321']=avg(ps,arms,EX);print('e2e231 done',flush=True)
    if a.only in('all','olmo'):
        tf='/tmp/claude-1147/olmo/tokenizer.json';arms=['NAT','MASK1024','CHK','ISO'];res={}
        for rule in ('olmo_sliding','olmo_full'):
            ps=[shares(world(s,tokfile=tf)['chunks'],EX,arms,rule) for s in range(251,259)];res[rule]=avg(ps,arms,EX)
        R['olmo_seeds251-258_job1326']=res;print('olmo done',flush=True)
    if a.only in('all','dens'):
        import gen_facts as G
        import e59_dens_run as DN
        import e55_run as R55
        G=DN.register();arms=['NAT','MASK1024'];res={}
        for cond,(n,p,na) in DN.COND.items():
            ps=[]
            for s in (241,242,243,244):
                enc,w=DN.build_cond(R55,G,cond,s);ps.append(shares(w['chunks'],[4,8,na] if na>=8 else [2,4,na],arms))
            res[cond]=avg(ps,arms,[4,8,na] if na>=8 else [2,4,na])
        R['dens_seeds241-244_job1327_matched_and_unmatched_identical']=res;print('dens done',flush=True)
    json.dump(R,open(out/f'visible_frac_{a.only}.json','w'),indent=1);print(json.dumps(R,indent=1)[:6000])
if __name__=='__main__':main()
