"""E57 public QA stage 15 (new file; GPU; QA thread, Hans 11/03 13:21 KST '노출 더 늘려볼까 ... 병렬 작성으로 처리해'): bundle episodes with 17 exposures. The 5 bundles 5 / 8 / 9 / 10 / 12 of job 640 (50 paragraphs, 73 items; subset.json) are written as in jobs 640 / 650 but with 16 re-descriptions per paragraph (rewrites_16_subset.json, sha256 9ee1c6da...; the first 8 equal rewrites_all8.json): each episode starts from W0 and runs orig 10 paragraphs -> rw1 10 -> ... -> rw16 10 (17 sections, the paragraphs of a section joined as in job 640), then the same SQuAD filler completion to whole chunks; arms NAT (norm source), NATQ, ALLQ128 (e57_pqa6_run.write_stage, ALLQ128 from chunk 0) and GAINQ512 p.10 (e57_pqa7_run score + write stages: band-512 gain mask from the W0 pass of the stream). The stream, group meta and ids are written in the file formats of job 640 (e57_pqa6_groups.json, e57_pqa6_group_ids.npz) so that the stages of jobs 640 / 650 run unchanged on this directory. Read points n = 1 / 3 / 5 / 9 / 13 / 17 (end of the chunk holding the end of version n; at n = 9 the spill into the next version is text here, filler in job 640, so the n = 9 state is not bit-identical to job 640: the check is the update norms of the chunks before that chunk and the W0 rows).
Reads (per-token log-probs stored, FastOUTG3 packing per bundle, rows A then B first as in job 674): (q) the 16-candidate question read of job 674: A = own question, B = question of the partner item (E57_pqa4_assignB.json); (b) the closed same-bundle read of job 672: candidates = gold strings of the other items of the bundle (e57_pqa12_run.layout), A = own question, B_bundle = the same string under the other questions of the bundle.
Metrics (analyze): first content token (space-only tokens skipped), second token, mean of the remaining tokens of the gold candidate; rank ratio and top-1 by the first content token and by the branch point; DeltaA vs W0, Delta(A-B), masking extras ALLQ128 - NATQ and GAINQ512P10 - NATQ; 95% CIs by item (73) and by bundle (5; bundle means resampled).
  python3 e57_pqa15_run.py --stage build --out O ... | --stage write --out O | --stage read --out O ... | --stage analyze --out O --json J | --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa2_stream as S
import e57_pqa3_stream as S3
import e57_pqa6_run as P6
import e57_pqa7_run as P7
import e57_pqa12_run as P12
CH=1024;NREAD=[1,3,5,9,13,17];BUNDLES=[5,8,9,10,12];ARMS=['NATQ','ALLQ128'];                    # GAINQ512 p.10 dropped (Hans 11/03 24:23 KST 'GAINQ는 빼자'); the stages of job 650 are still available for it
REW16_SHA='9ee1c6da057089f68f627410733b8524f281307eaf3182e04f759813a0a92c35'
def build(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);raw=open(a['rew16'],'rb').read();assert hashlib.sha256(raw).hexdigest()==REW16_SHA,'rewrites_16_subset.json sha256 differs'
    r16=json.loads(raw);sub=json.load(open(a['subset']));assert sub['bundles']==BUNDLES and sub['paragraphs']==50 and sub['items']==73
    src=pathlib.Path(a['src640']);meta6,ids6=P6.load_groups(src);para=json.load(open(a['para']));rew8=json.load(open(a['rewrites8']));byp={p['pid']:p for p in para};tk=S.tokenizer()
    train=json.load(open(a['train']));dev=json.load(open(a['dev']));fm,fx,ov=S3.filler_paragraphs(train,dev,len(para),P6.SEED);filler=[]
    for ti,tx in fm[:24]:
        for s,x in S3.enc_par(tk,tx):filler+=x
    groups=[];ids_all={}
    for g in meta6['groups']:
        if g['gid'] not in BUNDLES:continue
        ids=[];ends=[]
        for v in range(17):
            for pid in g['pids']:
                if v>=1 and v<=8:assert r16[pid][v-1]==rew8[pid][v-1],('rewrites differ from rewrites_all8',pid,v)
                text=byp[pid]['context'] if v==0 else r16[pid][v-1]
                for s,x in S3.enc_par(tk,text):ids+=x
            ends.append(len(ids))
        assert list(ids[:ends[8]])==list(ids6[f"g{g['gid']}"][:g['section_end_tokens'][8]]) and ends[8]==g['section_end_tokens'][8],'first 9 sections differ from job 640'
        need=-(-len(ids)//CH)*CH-len(ids);ids+=filler[:need];assert need<len(filler) and len(ids)%CH==0
        rch={str(k):(ends[k-1]-1)//CH for k in NREAD};spill={str(k):int((rch[str(k)]+1)*CH-ends[k-1]) for k in NREAD}
        groups.append(dict(gid=g['gid'],pids=g['pids'],item_ids=g['item_ids'],main_ids=g['main_ids'],n_chunks=len(ids)//CH,n_tokens=len(ids),section_end_tokens=ends,read_chunks=rch,spill_tokens=spill,filler_tokens=need,ids_sha=hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest()[:16],n_chunks_job640=g['n_chunks'],read_chunks_job640=g['read_chunks']))
        ids_all[f"g{g['gid']}"]=np.asarray(ids,np.int64)
    assert sum(len(g['item_ids']) for g in groups)==73 and sum(len(g['pids']) for g in groups)==50
    np.savez_compressed(out/'e57_pqa6_group_ids.npz',**ids_all);(out/'e57_pqa6_groups.json').write_text(json.dumps(dict(seed=P6.SEED,n_groups=len(groups),groups=groups,read_n=NREAD,rew16_sha256=REW16_SHA,para_sha256=meta6['para_sha256'])))
    print('groups',[(g['gid'],g['n_chunks'],g['read_chunks']) for g in groups],'filler tokens',[g['filler_tokens'] for g in groups])
def tok_ids(r_,k,n):return [[int(x) for x in r_[k*n+j][0][r_[k*n+j][1][0]+1:r_[k*n+j][1][1]+1]] for j in range(n)]
def read_stage(a):
    out=pathlib.Path(a['out']);lg=lambda *x:print(*x,flush=True);meta,_=P6.load_groups(out);W={'640':json.load(open(out/'e57_pqa6_writes.json')),'650':json.load(open(out/'e57_pqa7_writes.json')) if (out/'e57_pqa7_writes.json').exists() else {}};W640=json.load(open(pathlib.Path(a['src640'])/'e57_pqa6_writes.json'))
    G=meta['groups'];LD,rt,rt32,rd,items,h,raw0,ntok0,out_tail,tsha=P2.setup_read(a,lg);import e57_reads as RD
    byid={it['id']:it for it in items};asg=json.load(open(a['assign']))['assignment'];q={it['id']:it['prefix'] for it in items};golds={i:it['cands'][0] for i,it in byid.items()};grows=[];gits=[];gcount=[];tokids={'A':{},'B':{},'BUN':{}};lay={}
    for g in G:
        its=[byid[i] for i in g['item_ids']];lo=P12.layout(g['item_ids'],golds);lay.update(lo);rows=[];cnt=[]
        for c in ('A','B'):
            it2=[dict(it,prefix=q[it['id']] if c=='A' else q[asg[it['id']]]) for it in its];r_,n_,bad=RD.build_rows(rt.tok,it2);assert not bad,bad
            for k,it in enumerate(its):tokids[c][it['id']]=tok_ids(r_,k,16)
            rows+=r_;cnt.append(len(r_))
        itb=[dict(byid[i],cands=[golds[j] for j in lo[i]],kinds=['gold']+['distractor']*(len(lo[i])-1),all_idx=list(range(len(lo[i])))) for i in g['item_ids']];r_,n_,bad=RD.build_rows(rt.tok,itb);assert not bad,bad;o=0
        for it in itb:tokids['BUN'][it['id']]=[[int(x) for x in r_[o+j][0][r_[o+j][1][0]+1:r_[o+j][1][1]+1]] for j in range(len(it['cands']))];o+=len(it['cands'])
        rows+=r_;cnt.append(len(r_));grows.append(rows);gits.append((its,itb));gcount.append(cnt)
    (out/'e57_pqa15_tokens.json').write_text(json.dumps(tokids));(out/'e57_pqa15_layout.json').write_text(json.dumps(lay))
    m32w0=rd.to_fp32(rt.base);fo=P12.FastOUTG3(rt32,m32w0,out_tail,grows,log=lg);labels=['W0']+[f'{arm}@n{n}' for arm in ARMS for n in NREAD];R={c:{l:[] for l in labels} for c in ('A','B','BUN')}
    def put(label,gi,m32):
        toks=fo.score_group_tokens(m32,gi);its,itb=gits[gi];o=0
        for ci,c in enumerate(('A','B','BUN')):
            tc=toks[o:o+gcount[gi][ci]];o+=gcount[gi][ci];ii=its if c!='BUN' else itb;p=0
            for it in ii:
                nc=len(it['cands']);lps=tc[p:p+nc];p+=nc
                R[c][label].append(dict(id=it['id'],cand_lp=[float(t.sum()) for t in lps],cand_tok_lp=[[float(x) for x in t] for t in lps]))
    t00=time.monotonic()
    for gi,g in enumerate(G):
        t0=time.monotonic();put('W0',gi,m32w0)
        for arm in ARMS:
            job='650' if arm=='GAINQ512P10' else '640'
            for n in NREAD:
                s=W[job][str(g['gid'])][arm]['states'][str(g['read_chunks'][str(n)])];m=rd.load_state(s['file'],expect_hash=s['hash']);put(f'{arm}@n{n}',gi,rd.to_fp32(m));del m
        lg(f'  bundle {g["gid"]} read: {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
    for c in R:
        for l in labels:(out/f'E57_pqa15_rows_{c}_{l.replace("@","_")}.json').write_text(json.dumps(R[c][l]))
    chk={'update_norm_prefix_vs_job640':{},'W0_rows_vs_job640':{}}                                          # reproduction checks against job 640
    for g in G:
        gid=str(g['gid']);rc9=int(g['read_chunks_job640']['9'])
        for arm in ('NAT','NATQ','ALLQ128'):
            x=np.asarray(W['640'][gid][arm]['update_norms']);y=np.asarray(W640[gid][arm]['update_norms']);k=min(rc9,int(g['read_chunks']['9']));chk['update_norm_prefix_vs_job640'][f'{gid}:{arm}']=dict(n_chunks_compared=int(k),max_abs_diff=float(np.abs(x[:k]-y[:k]).max()) if k else None)
    src=pathlib.Path(a['src640'])
    for c,f in (('A','A'),('B','B')):
        ref={r['id']:r for r in json.load(open(src/f'E57_pqa6_rows_{f}_W0.json'))};chk['W0_rows_vs_job640'][c]=float(max(np.abs(np.asarray(r['cand_lp'])-np.asarray(ref[r['id']]['cand_lp'])).max() for r in R[c]['W0']))
    n9={}
    for arm in ('NATQ','ALLQ128'):
        for c in ('A','B'):
            ref={r['id']:r for r in json.load(open(src/f'E57_pqa6_rows_{c}_{arm}_n9.json'))};d=np.asarray([np.asarray(r['cand_lp'][0])-np.asarray(ref[r['id']]['cand_lp'][0]) for r in R[c][f'{arm}@n9']]);n9[f'{arm}:{c}']=dict(mean_diff_gold_lp=float(d.mean()),mean_abs_diff=float(np.abs(d).mean()),max_abs_diff=float(np.abs(d).max()))
    chk['n9_state_vs_job640_gold_lp']=n9;lg('reproduction checks:',json.dumps(chk)[:800])
    (out/'e57_pqa15_read_meta.json').write_text(json.dumps(dict(items_sha256=h,out_tail_sha=tsha,labels=labels,item_ids=[i for g in G for i in g['item_ids']],groups=[g['gid'] for g in G],reproduction=chk,timings_s=time.monotonic()-t00)))
def content_lps(tl,ti,dec):
    """per candidate: log-probs of its content tokens (space-only tokens dropped) and the matching token ids"""
    ci=[[j for j,t in enumerate(ti[c]) if dec([t]).strip()!=''] for c in range(len(tl))];return [np.asarray(tl[c],np.float64)[ci[c]] for c in range(len(tl))],[[ti[c][j] for j in ci[c]] for c in range(len(tl))]
def gold_parts(g):
    return dict(first=g[0] if len(g) else np.nan,second=g[1] if len(g)>1 else np.nan,rest=g[2:].mean() if len(g)>2 else np.nan)
def item_arrays(rows,ids,tokids,dec,kind,lay=None):
    """-> per item dict arrays: A first / second / rest, rr_first, top1_first, rr_branch, top1_branch; B first / second / rest (q: the B condition rows; bun: the same string under the other questions of the bundle)"""
    n=len(ids);K=('A_first','A_second','A_rest','B_first','B_second','B_rest','rr_first','top1_first','rr_branch','top1_branch');o={k:np.full(n,np.nan) for k in K};byid={r['id']:r for r in rows};CL={}
    for k,i in enumerate(ids):
        cl,ct=content_lps(byid[i]['cand_tok_lp'],tokids[i],dec);CL[i]=(cl,ct);g=cl[0]
        if not len(g):continue
        p=gold_parts(g);o['A_first'][k]=p['first'];o['A_second'][k]=p['second'];o['A_rest'][k]=p['rest'];nf=nb=0
        for c in range(1,len(cl)):
            if not len(cl[c]):continue
            nf+=int(cl[c][0]>g[0]);tg,tc=ct[0],ct[c];t=next((x for x in range(min(len(tg),len(tc))) if tg[x]!=tc[x]),min(len(tg),len(tc))-1);nb+=int(cl[c][:t+1].sum()>g[:t+1].sum())
        nc=len(cl);o['rr_first'][k]=(1+nf)/nc;o['top1_first'][k]=float(nf==0);o['rr_branch'][k]=(1+nb)/nc;o['top1_branch'][k]=float(nb==0)
    if kind=='bun':
        for k,i in enumerate(ids):
            v=[]
            for j in ids:
                if j!=i and i in lay[j]:
                    cl,_=CL[j];c=lay[j].index(i)
                    if len(cl[c]):v.append(gold_parts(cl[c]))
            if v:
                for m in ('first','second','rest'):
                    x=[d[m] for d in v if not np.isnan(d[m])]
                    if x:o[f'B_{m}'][k]=np.mean(x)
    return o
def boot_items(x,B=2000,seed=3):
    x=np.asarray(x,np.float64);x=x[~np.isnan(x)]
    if not len(x):return None
    m=x[np.random.default_rng(seed).integers(0,len(x),(B,len(x)))].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def boot_bundles(x,bun,B=2000,seed=3):
    x=np.asarray(x,np.float64);bun=np.asarray(bun);ok=~np.isnan(x);x=x[ok];bun=bun[ok]
    if not len(x):return None
    bm=np.asarray([x[bun==b].mean() for b in sorted(set(bun.tolist()))]);m=bm[np.random.default_rng(seed).integers(0,len(bm),(B,len(bm)))].mean(1);return dict(mean=float(x.mean()),bundle_mean=float(bm.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n_bundles=int(len(bm)))
def analyze(a):
    out=pathlib.Path(a['out']);rm=json.load(open(out/'e57_pqa15_read_meta.json'));ids=rm['item_ids'];labels=rm['labels'];tok=json.load(open(out/'e57_pqa15_tokens.json'));lay=json.load(open(out/'e57_pqa15_layout.json'));meta,_=P6.load_groups(out)
    bun=[next(g['gid'] for g in meta['groups'] if i in g['item_ids']) for i in ids];tk=S.tokenizer();dec=lambda t:tk.decode(t);ld=lambda c,l:json.load(open(out/f'E57_pqa15_rows_{c}_{l.replace("@","_")}.json'));typ={it['id']:it['type'] for it in json.load(open(a['items']))['items']}
    A={}
    for kind,(ca,cb) in (('q',('A','B')),('bun',('BUN',None))):
        for l in labels:
            x=item_arrays(ld(ca,l),ids,tok[ca],dec,'q' if kind=='q' else 'bun',lay)
            if kind=='q':
                y=item_arrays(ld('B',l),ids,tok['B'],dec,'q');x['B_first']=y['A_first'];x['B_second']=y['A_second'];x['B_rest']=y['A_rest'];x['rr_first_B']=y['rr_first']
            A[(kind,l)]=x
    res={'meta':rm,'definitions':'lp = log-prob of the gold string tokens; first = first content token (space-only skipped), second = next token, rest = mean of the remaining tokens; rr = rank among the candidates (16 for q, the bundle candidates for bun) / n candidates, as improvement (W0 minus state) in dA / extra; dA = state - W0; dAB = dA - dB; B = question of the partner item (q) or the same string under the other questions of the bundle (bun); extra = difference of the dA of two arms; CIs: item = bootstrap over the 73 items, bundle = bootstrap over the 5 bundle means','by_n':{}}
    METR=[('first_lp','A_first'),('second_lp','A_second'),('rest_lp','A_rest')];IMP=('rr_first','rr_branch');T1=('top1_first','top1_branch')
    def stats(x,sel=None):
        s=np.ones(len(ids),bool) if sel is None else sel;return dict(item=boot_items(x[s]),bundle=boot_bundles(x[s],np.asarray(bun)[s]))
    for n in NREAD:
        res['by_n'][str(n)]={}
        for kind in ('q','bun'):
            W0=A[(kind,'W0')];D={};e={}
            for arm in ARMS:
                m=A[(kind,f'{arm}@n{n}')];d={}
                for nm,key in METR:
                    d[f'dA_{nm}']=m[key]-W0[key]
                    bk='B_'+key.split('_')[1];d[f'dB_{nm}']=m[bk]-W0[bk];d[f'dAB_{nm}']=d[f'dA_{nm}']-d[f'dB_{nm}']
                for k in IMP:d[f'{k}_improvement']=W0[k]-m[k]
                for k in T1:d[f'{k}_diff']=m[k]-W0[k]
                D[arm]=d
            for arm in ARMS:
                e[arm]={'state_abs':{k:stats(A[(kind,f'{arm}@n{n}')][k]) for k in ('A_first','A_second','A_rest','rr_first','top1_first','rr_branch','top1_branch')},'dA_vs_W0':{k:stats(v) for k,v in D[arm].items()}}
            e['W0']={k:stats(W0[k]) for k in ('A_first','A_second','A_rest','rr_first','top1_first','rr_branch','top1_branch')}
            e['extra']={f'{a1}-NATQ':{k:stats(D[a1][k]-D['NATQ'][k]) for k in D[a1]} for a1 in [x for x in ('ALLQ128','GAINQ512P10') if x in ARMS]}
            e['extra_by_type']={f'{a1}-NATQ':{t:{k:stats(D[a1][k]-D['NATQ'][k],np.asarray([typ[i]==t for i in ids])) for k in ('dA_first_lp','dAB_first_lp')} for t in ('LOW','CAP','NUM','DATE') if any(typ[i]==t for i in ids)} for a1 in [x for x in ('ALLQ128','GAINQ512P10') if x in ARMS]}
            res['by_n'][str(n)][kind]=e
    fn=pathlib.Path(a['json']);fn.write_text(json.dumps(res,indent=1,default=float));print('wrote',fn)
def selftest():
    P6.selftest()
    dec=lambda t:{1:' ',2:' x',3:'y'}[t[0]];rows=[dict(id='i',cand_tok_lp=[[-1.,-2.,-3.]]+[[-1.5,-3.,-1.]]*3)];tok={'i':[[1,2,3]]*4};o=item_arrays(rows,['i'],tok,dec,'q')
    assert o['A_first'][0]==-2 and o['A_second'][0]==-3 and np.isnan(o['A_rest'][0]) and o['rr_first'][0]==.25 and o['top1_first'][0]==1
    rows2=[dict(id='i',cand_tok_lp=[[-1.,-2.]]+[[-1.,-5.]]),dict(id='j',cand_tok_lp=[[-1.,-2.]]+[[-1.,-4.]])];tk={'i':[[1,2]]*2,'j':[[1,2]]*2};lay={'i':['i','j'],'j':['j','i']}
    o=item_arrays(rows2,['i','j'],tk,dec,'bun',lay);assert o['B_first'][0]==-4.0 and o['B_first'][1]==-5.0
    b=boot_bundles([1.,3.,5.,7.],[1,1,2,2]);assert b['bundle_mean']==4.0 and b['n_bundles']==2
    print('e57_pqa15_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),para=g('--para'),rewrites8=g('--rewrites8'),rew16=g('--rew16'),subset=g('--subset'),train=g('--train'),dev=g('--dev'),json=g('--json'),max_groups=None)
    {'build':build,'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
