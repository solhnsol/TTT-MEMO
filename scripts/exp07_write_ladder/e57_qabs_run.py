"""E57 QA block-stream (jobs 1298s smoke / 1299 main; EXPLORATORY; K0 = 0 only; new file, no existing e57_*.py edited). The job-1296 block-stream design moved to public QA: 128 SQuAD single-question paragraphs x 16 re-descriptions (data/public-qa/rewrite128, e6cb4b2). Per seed the 128-paragraph pool is sampled without replacement into NB bundles of BP paragraphs (rng = random.Random(f"E57qa|{seed}|assign"), recorded in <out>/qabs_s<seed>_plan.json; a draw is repeated until every chunk fits into 1024 tokens). Bundle j occupies chunks 16 j .. 16 j + 15; chunk 16 j + k holds the k-th re-description of every paragraph of the bundle (paragraph order shuffled per chunk), the rest of the chunk is SQuAD filler (train documents whose titles are not in the pool), 16 exposures per paragraph. Arms (clipped SGD, engine SGDSEL as jobs 1283 / 1296): NATS (no mask) and NATMASK ALLQ1024 (mask from chunk 0). States are saved after every bundle (read chunks 16 j + 15). Reads: OUT-tail reads (neutral 8192-token tail, as the stage-15 QA reads, FastOUTG3) of every item of every bundle at every read point, W0 included. Candidates of an item = the gold strings of the other paragraphs of its bundle (own first; related strings dropped as in e57_pqa12_run.layout). Condition A = own question, condition B = question of the partner item (next item of the bundle). Lag of bundle b at read point j: j - b (0 .. NB-1); lag -1 = bundle not yet written (b > j).
  python3 e57_qabs_run.py --stage build|write|read|analyze --out O [--seeds 101,102,..] [--nb 6] [--bp 8] [--nbundles-smoke 2] | --selftest"""
import sys,os,re,json,time,gc,random,hashlib,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NV=16;DATA=ROOT/'data/public-qa/rewrite128'
PIN={'paragraphs128.json':None,'rewrites_16_qa128.json':None}
def load_pool():
    P=json.load(open(DATA/'paragraphs128.json'));R=json.load(open(DATA/'rewrites_16_qa128.json'));assert len(P)==128 and all(len(R[p['pid']])==NV and len(p['qs'])==1 and p['n_main']==1 for p in P);return P,R
def sha_files():return {f:hashlib.sha256(open(DATA/f,'rb').read()).hexdigest()[:16] for f in PIN}
def enc(tk,text):return tk.encode(text,add_special_tokens=False).ids
def filler_ids(tk,exclude_titles,exclude_ctx,seed,need):
    """SQuAD train filler (titles outside the pool), seeded; token ids of at least `need` tokens (whole paragraphs joined with a blank line)"""
    import e57_pqa3_stream as S3
    train=json.load(open(ROOT/'data/squad/train-v1.1.json'));dev=json.load(open(ROOT/'data/squad/dev-v1.1.json'))
    ps=[(t,x) for t,x in S3.filler_paragraphs(train,dev,0,20261115,extra=4000)[1] if t not in exclude_titles and x not in exclude_ctx];random.Random(f'E57qa|{seed}|filler').shuffle(ps);ids=[]
    for t,x in ps:
        ids+=enc(tk,('\n\n' if ids else '')+x)
        if len(ids)>=need:break
    assert len(ids)>=need,'filler too short';return ids
def plan_seed(seed,P,R,tk,nb,bp,cache={}):
    """-> dict(bundles=[[pid..]], ids int64 [nb*16*1024], read_chunks, chunk_tok (tokens of the paragraphs per chunk)); the assignment is redrawn (attempt counter in the rng key) until all nb*16 chunks fit"""
    tok={}
    for p in P:
        for k in range(NV):tok[(p['pid'],k)]=enc(tk,R[p['pid']][k])
    pids=[p['pid'] for p in P]
    for att in range(1000):
        r=random.Random(f'E57qa|{seed}|assign|{att}');pool=pids[:];r.shuffle(pool);bundles=[pool[b*bp:(b+1)*bp] for b in range(nb)]
        if all(sum(len(tok[(q,k)])+2 for q in bu)<=CH-40 for bu in bundles for k in range(NV)):break
    else:raise SystemExit('no assignment fits')
    ex_t={p['title'] for p in P};ex_c={p['context'] for p in P};need=nb*NV*CH;F=filler_ids(tk,ex_t,ex_c,seed,need);fi=0;ids=[];chunk_tok=[]
    for b,bu in enumerate(bundles):
        for k in range(NV):
            r=random.Random(f'E57qa|{seed}|chunk|{b}|{k}');o=bu[:];r.shuffle(o);ci=[]
            for q in o:ci+=enc(tk,'\n\n')+tok[(q,k)]
            chunk_tok.append(len(ci));rem=CH-len(ci);ci+=F[fi:fi+rem];fi+=rem;assert len(ci)==CH;ids+=ci
    ids=np.asarray(ids,np.int64);assert len(ids)==nb*NV*CH
    return dict(seed=seed,attempt=att,bundles=bundles,ids=ids,read_chunks=[NV*j+NV-1 for j in range(nb)],chunk_tok=chunk_tok,ids_sha=hashlib.sha256(ids.tobytes()).hexdigest()[:16])
def items_of(bundles,P):
    """per bundle: item dicts (id = pid, prefix = question, cands = [own gold] + golds of the other bundle paragraphs not related to it), partner = next paragraph of the bundle"""
    from e57_pqa12_run import related
    byp={p['pid']:p for p in P};out=[]
    for bu in bundles:
        its=[]
        for i,q in enumerate(bu):
            g=byp[q]['golds'][0];cl=[g];seen={g.strip().lower()}
            for q2 in bu:
                if q2==q:continue
                g2=byp[q2]['golds'][0]
                if g2.strip().lower() in seen or related(g,g2):continue
                seen.add(g2.strip().lower());cl.append(g2)
            its.append(dict(id=q,prefix=byp[q]['qs'][0][0].strip(),cands=cl,kinds=['gold']+['distractor']*(len(cl)-1),all_idx=list(range(len(cl))),partner=bu[(i+1)%len(bu)]))
        out.append(its)
    return out
def build(a):
    import e57_pqa2_stream as S
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);P,R=load_pool();tk=S.tokenizer();info={}
    for s in a['seeds']:
        pl=plan_seed(s,P,R,tk,a['nb'],a['bp']);np.save(out/f'qabs_s{s}_ids.npy',pl['ids']);it=items_of(pl['bundles'],P)
        json.dump(dict(seed=s,attempt=pl['attempt'],bundles=pl['bundles'],read_chunks=pl['read_chunks'],chunk_tok=pl['chunk_tok'],ids_sha=pl['ids_sha'],n_cands=[[len(x['cands']) for x in b] for b in it]),open(out/f'qabs_s{s}_plan.json','w'))
        info[s]=(pl['attempt'],pl['ids_sha'],int(np.mean(pl['chunk_tok'])),int(max(pl['chunk_tok'])))
    print('plans (seed: attempt, ids sha, mean/max paragraph tokens per chunk):',info,'data sha',sha_files())
def write_stage(a):
    import e55_run as R55,e55_swap as E,e57_natmask_run as NM,e57_qband_run as QR
    NM.register_engine();out=pathlib.Path(a['out']);lg=lambda *x:print(*x,flush=True)
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);cw0=can(rt.base);lg('canary W0',cw0)
    for s in a['seeds']:
        ids=np.load(out/f'qabs_s{s}_ids.npy');pl=json.load(open(out/f'qabs_s{s}_plan.json'));nch=a['nbundles']*NV;ids=ids[:nch*CH];fn=out/f'qabs_s{s}_writes.json';W=json.load(open(fn)) if fn.exists() else {}
        for arm,ma in (('NATS','NATQ'),('NATMASK','ALLQ1024')):
            if arm in W:continue
            NM.K0=0;QR.K0=0;masks={'band_ALLQ1024':np.asarray(1024),'all_ALLQ1024':np.asarray(1)};uv=QR.SelMaskQ(ma,masks,lg)
            L=C.Ladder55(rt,save_dir=out/'states'/f'qabs_s{s}_{arm}',canary=can);rc=pl['read_chunks'][:a['nbundles']];t0=time.monotonic()
            r=L.run_stream(NM.ENGINE,1,ids,nch=nch,tag=f'qabs_s{s}_{arm}',log=lg,read_chunks=rc,canary_chunks=rc,chunks_meta=[{'sents':[]} for _ in range(nch)],uv=uv)
            W[arm]=dict(states={str(k):dict(file=v['_file'],hash=v['_hash'],canary=v.get('canary')) for k,v in r['reads'].items()},ce=[e['ce'] for e in r['log']],update_norms=[e['update_norm'] for e in r['log']],canary_w0=cw0,mask_arm=ma,K0=0,ids_sha=pl['ids_sha'],sec=time.monotonic()-t0)
            fn.write_text(json.dumps(W,default=float));lg(f'seed {s} {arm}: {nch} chunks in {time.monotonic()-t0:.0f}s');del L,r;gc.collect();LD.jax.clear_caches()
def tok_rows(raw,n_cands):
    out=[];o=0
    for n in n_cands:out.append([[int(x) for x in raw[o+j][0][raw[o+j][1][0]+1:raw[o+j][1][1]+1]] for j in range(n)]);o+=n
    return out
def read_stage(a):
    import e55_run as R55,e55_gpu as RG,e57_reads as RD,e57_pqa6_run as P6,e57_pqa12_run as P12
    out=pathlib.Path(a['out']);lg=lambda *x:print(*x,flush=True);P,R=load_pool();LD,s3,C=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32()
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    forbid=sorted({g for p in P for g in p['golds'] if len(g)>=4 and not any(ch.isdigit() for ch in g)});out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,forbid);assert len(out_tail)==8192;m32w0=rd.to_fp32(rt.base)
    for s in a['seeds']:
        pl=json.load(open(out/f'qabs_s{s}_plan.json'));W=json.load(open(out/f'qabs_s{s}_writes.json'));NBs=a['nbundles'];its=items_of(pl['bundles'][:NBs],P);byq={x['id']:x for b in its for x in b};grows=[];meta=[]
        for bi,b in enumerate(its):
            rows=[];nc=[x['cands'] for x in b]
            for cond in ('A','B'):
                it2=[dict(x,prefix=(x['prefix'] if cond=='A' else byq[x['partner']]['prefix'])) for x in b];r_,n_,bad=RD.build_rows(rt.tok,it2);assert not bad,bad;rows+=r_
                meta.append((bi,cond,[len(x['cands']) for x in b],tok_rows(r_,[len(x['cands']) for x in b])))
            grows.append(rows)
        fo=P12.FastOUTG3(rt32,m32w0,out_tail,grows,log=lg);res={'W0':{},'meta':[dict(bundle=bi,cond=c,n_cands=n,tok=t) for bi,c,n,t in meta],'ids':[[x['id'] for x in b] for b in its]}
        def score(m32):
            o={}
            for bi in range(len(its)):
                toks=fo.score_group_tokens(m32,bi);o[bi]=[[float(v) for v in t] for t in toks]
            return o
        res['W0']=score(m32w0);t0=time.monotonic()
        for arm in ('NATS','NATMASK'):
            res[arm]={}
            for rc_,st in W[arm]['states'].items():
                m=rd.load_state(st['file'],expect_hash=st['hash']);res[arm][rc_]=score(rd.to_fp32(m));del m;lg(f'seed {s} {arm} read chunk {rc_} done ({time.monotonic()-t0:.0f}s)')
        ids=np.load(out/f'qabs_s{s}_ids.npy');res['IN']={}                       # in-window lag-0 reads: tail = the real last 8 chunks of the stream up to the read chunk, group = the bundle just written (conditions A and B)
        for j in range(NBs):
            rcj=NV*j+NV-1;tail=ids[(rcj-7)*CH:(rcj+1)*CH].tolist();assert len(tail)==8192 and rcj-7>=0;foj=P12.FastOUTG3(rt32,m32w0,tail,[grows[j]],log=lg)
            sc1=lambda m32,foj=foj:[[float(v) for v in t] for t in foj.score_group_tokens(m32,0)];res['IN'][str(j)]={'W0':sc1(m32w0)}
            for arm in ('NATS','NATMASK'):
                st=W[arm]['states'][str(rcj)];m=rd.load_state(st['file'],expect_hash=st['hash']);res['IN'][str(j)][arm]=sc1(rd.to_fp32(m));del m
            del foj;gc.collect();lg(f'seed {s} in-window read of bundle {j} done')
        (out/f'qabs_s{s}_reads.json').write_text(json.dumps(res))
def first_content(tl,ti,dec):
    """first content token (space-only tokens skipped) log-prob of every candidate -> array; the gold is candidate 0"""
    o=[]
    for l,t in zip(tl,ti):
        j=next((x for x,y in enumerate(t) if dec([y]).strip()!=''),None);o.append(l[j] if j is not None else np.nan)
    return np.asarray(o)
def margins(rd_,dec):
    """per read label (W0 / arm@chunk): {(bundle, item index): {'A': gold first-content-token lp under the own question, 'B': under the partner question}}; rows of a bundle = all A rows (item-major) then all B rows"""
    by={(m['bundle'],m['cond']):m for m in rd_['meta']}
    def one(sc):
        o={}
        for bi,rows in sc.items():
            bi=int(bi);na=by[(bi,'A')]['n_cands'];off=[0]+list(np.cumsum(na))
            for cond,base in (('A',0),('B',sum(na))):
                m=by[(bi,cond)]
                for k,n in enumerate(na):o.setdefault((bi,k),{})[cond]=float(first_content(rows[base+int(off[k]):base+int(off[k])+n],m['tok'][k],dec)[0])
        return o
    out={'W0':one(rd_['W0'])}
    for arm in ('NATS','NATMASK'):
        for rc_,sc in rd_[arm].items():out[f'{arm}@{rc_}']=one(sc)
    return out
def analyze(a):
    import e57_pqa2_stream as S
    out=pathlib.Path(a['out']);tk=S.tokenizer();dec=lambda t:tk.decode(t);NBs=a['nbundles'];seeds=a['seeds'];D={}
    for s in seeds:
        M=margins(json.load(open(out/f'qabs_s{s}_reads.json')),dec);w0=M['W0']
        for arm in ('NATS','NATMASK'):
            for j in range(NBs):
                st=M[f'{arm}@{NV*j+NV-1}']
                for (bi,k),wa in w0.items():
                    lag=j-bi;lag=lag if lag>=0 else -1;sa=st[(bi,k)];dA=sa['A']-wa['A'];dB=sa['B']-wa['B'];D.setdefault((arm,lag,s),[]).append((dA,dA-dB))
        rdd=json.load(open(out/f'qabs_s{s}_reads.json'));by={(m['bundle'],m['cond']):m for m in rdd['meta']}
        for j in range(NBs):
            one=lambda rows,j=j:margins({'meta':rdd['meta'],'W0':{j:rows},'NATS':{},'NATMASK':{}},dec)['W0']
            w0i=one(rdd['IN'][str(j)]['W0'])
            for arm in ('NATS','NATMASK'):
                sa=one(rdd['IN'][str(j)][arm])
                for (bi,k),wa in w0i.items():dA=sa[(bi,k)]['A']-wa['A'];dB=sa[(bi,k)]['B']-wa['B'];D.setdefault((arm,'in0',s),[]).append((dA,dA-dB))
    res={'definition':'primary = question-specific margin dAB = (state - W0) of the first-content-token log-prob of the gold under the own question (dA) minus the same under the partner question (dB); secondary dA; per seed mean over the items of the bundles at that lag; contrast = NATMASK1024 - NATS per lag with a t 95% interval over seeds (df = n_seeds - 1)','lags':{}}
    from scipy import stats
    for lag in sorted({k[1] for k in D},key=lambda x:(isinstance(x,str),str(x))):
        r={}
        for nm,ix in (('dAB',1),('dA',0)):
            m={arm:np.asarray([np.nanmean([x[ix] for x in D[(arm,lag,s)]]) for s in seeds if (arm,lag,s) in D]) for arm in ('NATS','NATMASK')}
            d=m['NATMASK']-m['NATS'];n=len(d);mu=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan');tq=float(stats.t.ppf(0.975,n-1)) if n>1 else float('nan')
            r[nm]=dict(NATS=float(m['NATS'].mean()),NATMASK=float(m['NATMASK'].mean()),diff=mu,ci95=[mu-tq*se,mu+tq*se],n_seeds=n)
        res['lags'][str(lag)]=r
    pathlib.Path(a['json']).write_text(json.dumps(res,indent=1));print(json.dumps({k:{m:(round(v['diff'],4),[round(x,4) for x in v['ci95']]) for m,v in r.items()} for k,r in res['lags'].items()}))
def selftest():
    import e57_pqa2_stream as S
    P,R=load_pool();tk=S.tokenizer();pl=plan_seed(7,P,R,tk,2,8);pl2=plan_seed(7,P,R,tk,2,8);pl3=plan_seed(8,P,R,tk,2,8)
    assert pl['ids_sha']==pl2['ids_sha']!=pl3['ids_sha'] and pl['bundles']==pl2['bundles'] and len(pl['ids'])==2*NV*CH and pl['read_chunks']==[15,31]
    flat=[q for b in pl['bundles'] for q in b];assert len(set(flat))==16 and set(flat)<={p['pid'] for p in P}
    for b,bu in enumerate(pl['bundles']):
        for k in range(NV):
            ch=pl['ids'][(b*NV+k)*CH:(b*NV+k+1)*CH].tolist();assert len(ch)==CH
            for q in bu:
                t=enc(tk,R[q][k]);assert any(ch[i:i+len(t)]==t for i in range(CH-len(t)+1)),(b,k,q)         # the k-th re-description of every paragraph of the bundle is in the chunk
            other=[q for q in flat if q not in bu][0];t=enc(tk,R[other][k]);assert not any(ch[i:i+len(t)]==t for i in range(CH-len(t)+1))
    it=items_of(pl['bundles'],P);assert all(len(b)==8 and all(x['cands'][0] and len(x['cands'])>=2 for x in b) for b in it)
    m=pl['chunk_tok'];print('e57_qabs_run selftest ok: paragraph tokens per chunk mean',int(np.mean(m)),'max',max(m),'attempts',pl['attempt'],'candidates per item',[len(x['cands']) for x in it[0]])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage');ap.add_argument('--out');ap.add_argument('--seeds',default='101,102,103,104');ap.add_argument('--nb',type=int,default=6);ap.add_argument('--bp',type=int,default=8);ap.add_argument('--nbundles',type=int,default=0);ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');x=ap.parse_args()
    if x.selftest:selftest();return
    a=dict(out=x.out,seeds=[int(s) for s in x.seeds.split(',')],nb=x.nb,bp=x.bp,nbundles=x.nbundles or x.nb,json=x.json)
    {'build':build,'write':write_stage,'read':read_stage,'analyze':analyze}[x.stage](a)
if __name__=='__main__':main()
