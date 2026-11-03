"""E57 news T1/T2 per-token re-read of the saved end-of-cycle-1 states (new file; GPU read only, NO weight update; comparison thread card 'synthetic + news T1/T2', Hans 22:11 KST): the frozen frames are read again with the OUT reader of the original jobs (the neutral 8192-token SQuAD tail of the bank-v1 forbid list, reference packed scorer e55_gpu.Reader55.packed -> packing.Packed.r1, packs of 40 frames in the original order, fp32) and the log-prob of every candidate TOKEN is stored (cand_tok_lp) with the token ids.
Frame sets: v1 = the 150 frozen T1/T2 frames of t1t2_v1.json (jobs 276 / 410 / 450; read with the stream-A states; the overlap-excluded subset 'no_overlap100' = the 100 frames with zero stream overlap at k = 1 of e57_t1t2_overlap.json); v2A / v2B = the 98 frozen reverse v2 questions in their two forms (196 frames: 74 on stream A, 122 on stream B20; t1t2_v2_frozen.json + e57_t1t2v2d3_candidates.json; every state is read only on the frames of its own stream, as job 570).
States (end of cycle 1, chunk 279 / 285): stream A: NAT, NATQ (= the NATADAMH state), ALLQ128 (v2A state), ALLQ512, GAINQ512 p.26, GAINQ512 p.10, RANDQ512 p.10, NATX2, NATX4 (SGD x2 / x4) and W0; stream B20: NAT, NATQ, ALLQ128, GAINQ512 p.26, RANDQ512 p.10 and W0 (a GAINQ512 p.10 state of B20 was never written). The summed log-probs are compared with the stored scores files of the original reads (maximum difference per state).
Metrics per frame (gold against the candidates of the original comparison sets 'all' = gold + easy + nominated hard, and 'hard' = the nominated hard): branch rank (gold and each candidate scored from the first token up to and including the first token at which the two token sequences differ; if one is a prefix of the other, up to the end of the shorter), first-token rank, full-sum rank (as stored), each as rank, rank ratio, top-1 and mean gold-minus-candidate difference; mean over frames with the bootstrap 95% CI over frames, by group and for all frames, W0 beside every state, paired differences to W0 and to the reference arm (NAT).
  python3 e57_news_tokread.py --stage read --out O | --stage analyze --out O | --selftest"""
import sys,os,json,time,glob,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics',ROOT/'e2e'):sys.path.insert(0,str(p))
EXP=ROOT/'experiments/exp07_write_ladder'
import e57_syn_tokread as TR
def man(path):return json.load(open(EXP/path))
def registry():
    """-> {'A': {label: (file, hash)}, 'B': {...}}"""
    mo=man('e57_t1t2_momvar_all/e57_t1t2_states_manifest.json');mq=man('e57_t1t2_qstates/e57_t1t2_states_manifest.json');ma=man('e57_t1t2_allq512/e57_t1t2_states_manifest.json');ms=man('e57_natsgd_t1t2/e57_t1t2_states_manifest.json');m2=man('e57_t1t2v2/e57_t1t2v2_states_manifest.json');mi=man('e57_t1t2v2_inev/e57_t1t2v2_inev_manifest.json')
    g=lambda m,k:(m[k]['file'],m[k]['hash'])
    A={'NAT':g(mo,'NAT'),'NATQ':g(mo,'NATADAMH'),'ALLQ128':g(m2,'ALLQ128_A'),'ALLQ512':g(ma,'ALLQ512'),'GAINQ512P26':g(mq,'GAINQ512P26'),'GAINQ512P10':g(mq,'GAINQ512P10'),'RANDQ512P10':g(mq,'RANDQ512P10'),'NATX2':g(ms,'NATX2'),'NATX4':g(ms,'NATX4')}
    B={'NAT':g(m2,'NAT_B'),'NATQ':g(m2,'NATQ_B'),'ALLQ128':g(m2,'ALLQ128_B'),'GAINQ512P26':g(mi,'GAINQ512P26_B'),'RANDQ512P10':g(mi,'RANDQ512P10_B')}
    return {'A':A,'B':B}
def stored_sums():
    """(set, state label, frame id) -> cand_lp from the stored scores files"""
    out={}
    for f in glob.glob(str(EXP/'e57_*t1t2*/E57_t1t2_scores.json')):
        for r in json.load(open(f))['rows']:out[('v1',r['state'],r['id'])]=r['cand_lp']
    for f in glob.glob(str(EXP/'e57_*t1t2v2*/E57_t1t2v2_scores.json')):
        for r in json.load(open(f))['rows']:
            st=r['state_label'] if 'state_label' in r else r['state'];out[('v2'+r.get('stream','A'),st,r['id'])]=r['cand_lp']
    return out
def read_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True)
    import e55_run as R55,e55_gpu as RG,e57_reads as RD,e57_t1t2_run as T1,e57_t1t2v2_run as V
    LD,s3,C=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    bank=RD.load_items(str(EXP/'e57_news/e57_bank/bank_v1.json'));forbid=sorted({s for it in bank for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)});out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,forbid)
    f1=T1.load_frozen(str(EXP/'e57_news/e57_t1t2/t1t2_v1.json'));new1=T1.make_items(f1,out);f2=V.load_frozen(str(EXP/'e57_t1t2v2_in/t1t2_v2_frozen.json'));c2=json.load(open(EXP/'e57_t1t2v2_in/e57_t1t2v2d3_candidates.json'));new2=V.make_items(f2,c2,out)
    sets={'v1':new1,'v2A':[it for it in new2 if it['stream']=='A'],'v2B':[it for it in new2 if it['stream']=='B']};reg=registry();state_sets={'v1':'A','v2A':'A','v2B':'B'};packs={};meta={}
    if a.get('sets'):sets={k:v for k,v in sets.items() if k in a['sets'].split(',')}
    state_sets={k:v for k,v in state_sets.items() if k in sets}
    if a.get('extra'):reg={'A':{x.split(',')[0]:(x.split(',')[1],x.split(',')[2]) for x in a['extra'].split(';')},'B':{}}                    # one extra state only (job 770): W0 is read again as the reproduction check
    for sn,its in sets.items():
        raw,ntok,bad=RD.build_rows(rt.tok,its);assert not bad,bad;groups=[];o=0;tokids=[]
        for g0 in range(0,len(its),40):
            sub=its[g0:g0+40];nr=sum(len(it['cands']) for it in sub);groups.append((sub,rd.packed(raw[o:o+nr],out_tail),raw[o:o+nr]));o+=nr
        assert o==len(raw);k=0
        for it in its:
            tokids.append([[int(x) for x in raw[k+j][0][raw[k+j][1][0]+1:raw[k+j][1][1]+1]] for j in range(len(it['cands']))]);k+=len(it['cands'])
        packs[sn]=groups;meta[sn]=[dict(id=it['id'],source=it.get('source'),group=it.get('stratum') or it.get('group'),event=it.get('event'),type=it.get('type'),stream=it.get('stream'),n_cands=len(it['cands']),kinds=it['kinds'],gold_idx=it['gold_idx'],hard_idx=it['hard_idx'],all_idx=it['all_idx'],tokids=tokids[i]) for i,it in enumerate(its)]
        lg(f'set {sn}: {len(its)} frames, {len(raw)} rows, {len(groups)} packs')
    def tokens(sn,m):
        m32=rd.to_fp32(m);res=[]
        with LD.jax.default_matmul_precision('highest'):
            for sub,P,rr in packs[sn]:
                tl=TR.r1_tokens(P,rt32,m32);k=0
                for it in sub:res.append([x.tolist() for x in tl[k:k+len(it['cands'])]]);k+=len(it['cands'])
        return res
    stored=stored_sums();R={sn:{'W0':tokens(sn,rt.base)} for sn in sets};chk={}
    def check(sn,label):
        tl=R[sn][label];mx=0.0;n=0
        for m_,t in zip(meta[sn],tl):
            ref=stored.get((sn if sn=='v1' else sn,label if label!='NATQ' or sn=='v1' else 'NATQ',m_['id']))
            if ref is None and sn=='v1' and label=='NATQ':ref=stored.get(('v1','NATADAMH',m_['id']))
            if ref is None:continue
            mx=max(mx,float(np.max(np.abs(np.asarray([sum(x) for x in t])-np.asarray(ref)))));n+=1
        chk[f'{sn}:{label}']=dict(max_abs_diff=mx if n else None,n_frames_compared=n)
    for sn in sets:check(sn,'W0')
    for st_name in ('A','B'):
        for label,(fn,h) in reg[st_name].items():
            t0=time.monotonic();m=rd.load_state(str(ROOT/fn) if not os.path.isabs(fn) else fn,expect_hash=h)
            for sn,ss in state_sets.items():
                if ss==st_name:R[sn][label]=tokens(sn,m);check(sn,label)
            del m;lg(f'  state {st_name}:{label}: {time.monotonic()-t0:.0f}s; '+', '.join(f'{k} {v["max_abs_diff"]}' for k,v in chk.items() if k.endswith(':'+label)))
    (out/'e57_news_tokread.json').write_text(json.dumps(dict(meta=meta,tokens=R,reproduction=chk,registry={k:{l:v[1][:12] for l,v in d.items()} for k,d in reg.items()},out_tail_sha=hashlib.sha256(np.asarray(out_tail,np.int64).tobytes()).hexdigest()[:16])))
def frame_metrics(tl,m_,which):
    ids=m_['all_idx'] if which=='all' else m_['hard_idx'];g=np.asarray(tl[m_['gold_idx']],np.float64);tg=m_['tokids'][m_['gold_idx']];nb=nf=nu=0;db=[];df=[];du=[];cs=[c for c in ids if c!=m_['gold_idx']]
    for c in cs:
        tc=m_['tokids'][c];lc=np.asarray(tl[c],np.float64);t=next((q for q in range(min(len(tg),len(tc))) if tg[q]!=tc[q]),min(len(tg),len(tc))-1);sg=g[:t+1].sum();sc=lc[:t+1].sum();nb+=int(sc>sg);db.append(sg-sc);nf+=int(lc[0]>g[0]);df.append(g[0]-lc[0]);nu+=int(lc.sum()>g.sum());du.append(g.sum()-lc.sum())
    n=len(cs)+1;o={}
    for w,nr,dd in (('branch',nb,db),('first',nf,df),('full',nu,du)):o[f'rr_{w}']=(1+nr)/n;o[f'top1_{w}']=float(nr==0);o[f'gap_{w}']=float(np.mean(dd)) if dd else np.nan
    o['gold_first']=float(g[0]);o['gold_full']=float(g.sum());return o
def boot(x,seed=1,B=2000):
    x=np.asarray(x,np.float64);x=x[~np.isnan(x)]
    if len(x)==0:return dict(mean=None,ci95=None,n=0)
    ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
KEYS=('rr_branch','rr_first','rr_full','top1_branch','top1_first','top1_full','gap_branch','gap_first','gap_full','gold_first','gold_full')
def analyze(a):
    out=pathlib.Path(a['out']);d=json.load(open(out/'e57_news_tokread.json'));ov=json.load(open(EXP/'e57_news/e57_t1t2/e57_t1t2_overlap.json'));zero={it['id'] for it in ov['items'] if it['stream']['1']==0};res={'reproduction':d['reproduction'],'overlap_excluded_definition':'frames with zero stream overlap at k = 1 (strict) of e57_t1t2_overlap.json: %d frames'%len(zero)}
    for sn in ('v1','v2A','v2B'):
        if sn not in d['meta']:continue
        meta=d['meta'][sn];T=d['tokens'][sn];M={lab:{w:[frame_metrics(t,m_,w) for t,m_ in zip(tl,meta)] for w in ('all','hard')} for lab,tl in T.items()};groups=sorted({m_['group'] for m_ in meta});subsets={'ALL':np.ones(len(meta),bool)}
        for gname in groups:subsets[gname]=np.asarray([m_['group']==gname for m_ in meta])
        if sn=='v1':subsets['no_overlap100']=np.asarray([m_['id'] in zero for m_ in meta])
        res[sn]={'n_frames':len(meta),'subset_sizes':{k:int(v.sum()) for k,v in subsets.items()},'states':{}}
        for lab in M:
            e={}
            for which in ('all','hard'):
                for ss,mk in subsets.items():
                    arr={k:np.asarray([f[k] for f in M[lab][which]]) for k in KEYS};w0={k:np.asarray([f[k] for f in M['W0'][which]]) for k in KEYS};ref={k:np.asarray([f[k] for f in M['NAT'][which]]) for k in KEYS} if 'NAT' in M else None
                    e[f'{which}|{ss}']=dict(value={k:boot(arr[k][mk]) for k in KEYS},minus_W0={k:boot((arr[k]-w0[k])[mk]) for k in KEYS},**({'minus_NAT':{k:boot((arr[k]-ref[k])[mk]) for k in KEYS}} if ref is not None and lab not in ('W0','NAT') else {}),**{f'minus_{rn}':{k:boot((arr[k]-np.asarray([f[k] for f in M[rn][which]]))[mk]) for k in KEYS} for rn in ('ALLQ128','NATQ') if rn in M and lab not in ('W0',rn)})
            res[sn]['states'][lab]=e
    (out/'e57_news_tokread_keys.json').write_text(json.dumps(res,indent=1,default=float))
def selftest():
    m_=dict(all_idx=[0,1,2],hard_idx=[2],gold_idx=0,tokids=[[1,2],[1,3],[4,5]]);tl=[[-1.,-2.],[-3.,-1.],[-0.5,-4.]];f=frame_metrics(tl,m_,'all')
    assert f['rr_branch']==(1+1)/3 and f['rr_first']==(1+1)/3 and abs(f['rr_full']-1/3)<1e-9 and f['top1_full']==1.0
    f=frame_metrics(tl,m_,'hard');assert f['rr_branch']==1.0 and f['top1_branch']==0.0
    print('e57_news_tokread selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k:sys.argv[sys.argv.index(k)+1];st=g('--stage')
    opt=lambda k:sys.argv[sys.argv.index(k)+1] if k in sys.argv else None
    if st=='read':read_stage(dict(out=g('--out'),sets=opt('--sets'),extra=opt('--extra')))
    else:analyze(dict(out=g('--out')))
if __name__=='__main__':main()
