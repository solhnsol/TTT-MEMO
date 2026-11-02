"""E57 public QA stage 8 (E57 122-X / 123; new file; read only, NO weight update): control reads of the stored states of jobs 640 (NATQ, ALLQ128) and 650 (GAINQ512 p.10) at n = 1 / 3 / 5 / 9 versions (33 paragraph-group episodes) and W0, to separate three explanations of the question-independent rise (B, C reads) of those jobs: frequency of the answer tokens, general adaptation, overall log-prob rise.
 (a) select: of the 1,000 frozen items the 545 that are NOT in items455 (their paragraph is not one of the 332 written paragraphs). Each is assigned to ONE group by a seeded rule (the groups in a seeded random order per item, at most CAP items per group) and kept only if neither its gold string nor any of its answer variants occurs (case-insensitive substring) in the whole written text of that group (original + the 8 re-descriptions of all its paragraphs); items for which no group qualifies are dropped. Their A / B / C reads (A own question, B question of another item with the assignment of job 620, C "Answer:") are taken in the states of the assigned group, same FastOUTG packing per group as jobs 640 / 650.
 (b) generic: ~200 fixed paragraphs of SQuAD train-v1.1 (documents not in dev and not among the filler documents of the episodes; seeded; sha256 of the text list is recorded; each cut to its first 128 tokens, "\n\n" + paragraph) are scored with every state (mean per-token NLL over the tokens 2..n of every paragraph, OUT context = stage-1 tail); the rows share the tail state with the item rows of the group. Generation canary (e55_run.Canary55, as job 610) is read for W0 and, if --canary is set, for the listed read points.
 (c) analyze (CPU): per item the number of occurrences of the gold string (case-insensitive, between non-alphanumeric characters) in the text of its group, summed over the first n versions; strata 1, 2, 3-5, 6+ (and 0) of items455; per arm / n / stratum the rise of the gold log-prob (state minus W0) under A, B and C with bootstrap CI, and the paired differences between arms inside each stratum; the new zero-occurrence items form the stratum 'absent'.
  python3 e57_pqa8_run.py --stage select --out O --src640 D640 --items-file F --para P --rewrites8 R8 | --stage read --out O --src640 D --src650 D --items-file F --bank B --assign A --stage1 S1 [--max-groups N] [--canary 9] | --stage analyze --out O --src640 D --src650 D --items-file F --para P --rewrites8 R8 | --selftest"""
import sys,json,re,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_stream as S
import e57_pqa3_stream as S3
import e57_pqa2_run as P2
import e57_pqa4_run as P4
import e57_pqa6_run as P6
import e57_squad_w0_reads as Q
CH=1024;NREAD=P6.NREAD;ARMS=[('NATQ','640'),('ALLQ128','640'),('GAINQ512P10','650')];SEEDS=20261102;CAP=17;NGEN=200;GENCUT=128
def group_texts(paras,rew,groups):
    """-> {gid: [text of version 1 (original), ..., version 9]} (the paragraphs of a group joined by newlines)"""
    byp={p['pid']:p for p in paras};out={}
    for g in groups:out[g['gid']]=['\n'.join((byp[pid]['context'] if v==0 else rew[pid][v-1]) for pid in g['pids']) for v in range(9)]
    return out
def count_in(text_lower,s):
    s=s.lower()
    if not s:return 0
    return len(re.findall(r'(?<![a-z0-9])'+re.escape(s)+r'(?![a-z0-9])',text_lower))
def select(items_raw,groups,gtexts,keep_ids,seed=SEEDS,cap=CAP):
    """-> (assignment {item id: gid}, dropped [ids], load per group); items_raw: dicts with id, gold, variants"""
    full={g['gid']:' \n '.join(gtexts[g['gid']]).lower() for g in groups};load={g['gid']:0 for g in groups};asg={};dropped=[]
    for k,it in enumerate(items_raw):
        if it['id'] in keep_ids:continue
        order=np.random.default_rng(seed+k).permutation(len(groups));done=False;strs=[it['gold']]+list(it.get('variants',[]))
        for gi in order:
            gid=groups[int(gi)]['gid']
            if load[gid]>=cap:continue
            if any(s.lower() in full[gid] for s in strs):continue
            asg[it['id']]=gid;load[gid]+=1;done=True;break
        if not done:dropped.append(it['id'])
    return asg,dropped,load
def select_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);meta,_=P6.load_groups(a['src640']);raw=json.load(open(a['items']))['items'];para=json.load(open(a['para']));rew=json.load(open(a['rewrites8']));gt=group_texts(para,rew,meta['groups'])
    keep={i for g in meta['groups'] for i in g['item_ids']};asg,dropped,load=select(raw,meta['groups'],gt,keep);by={r['id']:r for r in raw}
    import collections
    res=dict(rule=f'non-items455 items; per item the groups in a seeded random order (np.random.default_rng({SEEDS}+item index in the items file)), at most {CAP} items per group, first group whose whole written text (9 versions of all its paragraphs) contains neither the gold string nor any answer variant (case-insensitive substring); otherwise dropped',seed=SEEDS,cap=CAP,
        n_non455=int(len(raw)-len(keep)),n_selected=len(asg),n_dropped=len(dropped),assignment=asg,dropped=dropped,load={str(k):v for k,v in load.items()},types=dict(collections.Counter(by[i]['type'] for i in asg)),types_dropped=dict(collections.Counter(by[i]['type'] for i in dropped)))
    (out/'e57_pqa8_selection.json').write_text(json.dumps(res));print({k:v for k,v in res.items() if k not in ('assignment','dropped','rule')});return res
def generic_paragraphs(train_f,dev_f,filler_titles_n=24):
    train=json.load(open(train_f));dev=json.load(open(dev_f));dt={S3.tsha(a['title']) for a in dev['data']}
    fm,_,_=S3.filler_paragraphs(train,dev,332,S3.SEED);ft={S3.tsha(t) for t,_ in fm[:filler_titles_n]}
    docs=[a for a in train['data'] if S3.tsha(a['title']) not in dt and S3.tsha(a['title']) not in ft];order=np.random.default_rng(SEEDS).permutation(len(docs));paras=[]
    for i in order:
        for p in docs[i]['paragraphs'][:2]:
            if len(p['context'].split())>=40:paras.append(' '.join(p['context'].split()))
        if len(paras)>=NGEN:break
    paras=paras[:NGEN];return paras,hashlib.sha256(json.dumps(paras).encode()).hexdigest()
class FastOUTG2(P6.FastOUTG):
    """FastOUTG whose score_group may add the chunks of one extra group (the generic paragraphs) to the same call (one tail state)"""
    def score_group2(self,m32,g,extra):
        rt=self.rt;jax=self.jax;jnp=self.jnp;from core import clone_state
        idx=list(range(*self.grange[g]))+list(range(*self.grange[extra]));idx+=[idx[-1]]*((-len(idx))%self.GP)
        with rt.mesh,jax.default_matmul_precision('highest'):
            st=clone_state(rt.state_suffix)
            for j in range(self.TL//CH):st,_=rt.chunk_logp(m32,st,rt.chunk(self.tailseq,j),self.H[j*CH:(j+1)*CH])
            out={}
            for s in range(0,len(idx),self.GP):
                sel=idx[s:s+self.GP];r=np.asarray(self.lp_packed(m32,st,jnp.asarray(self.ids[sel]),jnp.asarray(self.tg[sel]),jnp.asarray(self.mk[sel]),jnp.asarray(self.pref1[sel]),jnp.asarray(self.seg[sel]),jnp.asarray(self.loc[sel])))
                for q,c in enumerate(sel):out.setdefault(c,r[q])
        return np.asarray([out[c][a:b].sum() for c,a,b in self.gwhere[g]]),np.asarray([out[c][a:b].sum() for c,a,b in self.gwhere[extra]])
def read_stage(a):
    out=pathlib.Path(a['out']);lg=lambda *x:print(*x,flush=True);meta,_=P6.load_groups(a['src640']);W={'640':json.load(open(pathlib.Path(a['src640'])/'e57_pqa6_writes.json')),'650':json.load(open(pathlib.Path(a['src650'])/'e57_pqa7_writes.json'))}
    sel=json.load(open(out/'e57_pqa8_selection.json'));asg=sel['assignment'];G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups']
    LD,rt,rt32,rd,items,h,raw0,ntok0,out_tail,tsha=P2.setup_read(a,lg);import e57_reads as RD,e55_run as R
    byid={it['id']:it for it in items};bq=json.load(open(a['assign']))['assignment'];q={it['id']:it['prefix'] for it in items};conds=('A','B','C');grows=[];gnt=[];gsizes=[];gitems=[]
    gen,gsha=generic_paragraphs(a['train'],a['dev']);genrows=[];gennt=[]
    for tx in gen:
        ids=LD.stage2.enc(rt.tok,'\n\n'+tx).ids[:GENCUT];genrows.append((ids,(0,len(ids)-1)));gennt.append(len(ids)-1)
    for g in G:
        its=[byid[i] for i in sorted(i for i,gid in asg.items() if gid==g['gid'])];gitems.append(its);rows=[];nts=[];sz=[]
        for c in conds:
            pre=[q[it['id']] if c=='A' else (q[bq[it['id']]] if c=='B' else 'Answer:') for it in its];it2=[dict(it,prefix=p) for it,p in zip(its,pre)];r_,n_,bad=RD.build_rows(rt.tok,it2);assert not bad,bad;rows+=r_;nts.append(n_);sz.append(len(r_))
        grows.append(rows);gnt.append(nts);gsizes.append(sz)
    grows.append(genrows);extra=len(grows)-1;m32w0=rd.to_fp32(rt.base);fo=FastOUTG2(rt32,m32w0,out_tail,grows,log=lg)
    labels=['W0']+[f'{arm}@n{n}' for arm,_ in ARMS for n in NREAD];R_={c:{l:[] for l in labels} for c in conds};GEN={l:{} for l in labels};CAN={};can=R.Canary55(LD,rt);cw0=can(rt.base);lg('canary W0',cw0);CAN['W0']=cw0
    cn=[int(x) for x in a['canary'].split(',')] if a.get('canary') else []
    def put(label,gi,m32,m=None):
        flat,gs=fo.score_group2(m32,gi,extra);its=gitems[gi];o=0
        for ci,c in enumerate(conds):
            sub=flat[o:o+gsizes[gi][ci]];o+=gsizes[gi][ci];R_[c][label]+=P2.rows_of(its,gnt[gi][ci],sub,f'{c}:{label}')
        GEN[label][str(G[gi]['gid'])]=dict(lp_sum_per_paragraph=[float(x) for x in gs],mean_nll_token_weighted=float(-gs.sum()/sum(gennt)),mean_nll_per_paragraph=float(np.mean(-gs/np.asarray(gennt))))
    t00=time.monotonic()
    for gi,g in enumerate(G):
        t0=time.monotonic();put('W0',gi,m32w0)
        for arm,job in ARMS:
            for n in NREAD:
                s=W[job][str(g['gid'])][arm]['states'][str(g['read_chunks'][str(n)])];m=rd.load_state(s['file'],expect_hash=s['hash']);put(f'{arm}@n{n}',gi,rd.to_fp32(m))
                if n in cn:CAN[f'g{g["gid"]}_{arm}@n{n}']=can(m)
                del m
        lg(f'  group {g["gid"]} ({len(gitems[gi])} items): {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
    for c in conds:
        for l in labels:(out/f'E57_pqa8_rows_{c}_{l.replace("@","_")}.json').write_text(json.dumps(R_[c][l],default=float))
    (out/'E57_pqa8_generic.json').write_text(json.dumps(dict(sha256_texts=gsha,n_paragraphs=len(gen),tokens_scored=int(sum(gennt)),cut_tokens=GENCUT,nll=GEN,canary=CAN),default=float))
    (out/'e57_pqa8_read_meta.json').write_text(json.dumps(dict(items_sha256=h,out_tail_sha=tsha,item_ids=[it['id'] for its in gitems for it in its],group_of_item={it['id']:G[gi]['gid'] for gi,its in enumerate(gitems) for it in its},generic_sha=gsha)))
def boot(x,seed=5,B=2000):
    x=np.asarray(x,np.float64)
    if len(x)==0:return dict(mean=None,ci95=None,n=0)
    ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def stratum(c):return 0 if c==0 else (1 if c==1 else (2 if c==2 else (3 if c<=5 else 4)))
STRAT=['0','1','2','3-5','6+']
def analyze(a):
    out=pathlib.Path(a['out']);s6=pathlib.Path(a['src640']);s7=pathlib.Path(a['src650']);meta,_=P6.load_groups(s6);items,h=Q.load_items_file(a['items']);para=json.load(open(a['para']));rew=json.load(open(a['rewrites8']));gt=group_texts(para,rew,meta['groups'])
    rm6=json.load(open(s6/'e57_pqa6_read_meta.json'));ids6=rm6['item_ids'];gof={i:g['gid'] for g in meta['groups'] for i in g['item_ids']};byid={it['id']:it for it in items}
    rows6=lambda c,l:json.load(open(s6/f'E57_pqa6_rows_{c}_{l.replace("@","_")}.json'));rows7=lambda c,l:json.load(open(s7/f'E57_pqa7_rows_{c}_{l.replace("@","_")}.json'))
    ld=lambda arm,job,c,n:(rows6 if job=='640' else rows7)(c,f'{arm}@n{n}');gold=lambda R_:np.asarray([r['cand_lp'][0] for r in R_]);W0={c:gold(rows6(c,'W0')) for c in 'ABC'}
    res={'items455':{},'absent':{}};texts_lower={(gid,n):'\n'.join(t[:n]).lower() for gid,t in gt.items() for n in NREAD}
    cnt={n:np.asarray([count_in(texts_lower[(gof[i],n)],byid[i]['cands'][0]) for i in ids6]) for n in NREAD}
    for arm,job in ARMS:
        for n in NREAD:
            imp={c:gold(ld(arm,job,c,n))-W0[c] for c in 'ABC'};st=np.asarray([stratum(x) for x in cnt[n]]);e={}
            for k,name in enumerate(STRAT):
                m=st==k;e[name]=dict(n=int(m.sum()),**{c:boot(imp[c][m]) for c in 'ABC'},gain_A_minus_B=boot((imp['A']-imp['B'])[m]))
            res['items455'][f'{arm}@n{n}']=e
    # paired arm differences inside every stratum
    pairs=[('ALLQ128','640','NATQ','640'),('GAINQ512P10','650','NATQ','640'),('GAINQ512P10','650','ALLQ128','640')];res['arm_differences']={}
    for a1,j1,a2,j2 in pairs:
        for n in NREAD:
            st=np.asarray([stratum(x) for x in cnt[n]]);e={}
            for k,name in enumerate(STRAT):
                m=st==k;e[name]={c:boot(((gold(ld(a1,j1,c,n))-W0[c])-(gold(ld(a2,j2,c,n))-W0[c]))[m]) for c in 'ABC'}
            res['arm_differences'][f'{a1}-{a2}@n{n}']=e
    res['occurrence_counts_items455']={f'n{n}':{name:int((np.asarray([stratum(x) for x in cnt[n]])==k).sum()) for k,name in enumerate(STRAT)} for n in NREAD}
    if (out/'e57_pqa8_read_meta.json').exists():                       # new items without any occurrence of the answer
        rm=json.load(open(out/'e57_pqa8_read_meta.json'));ld8=lambda c,l:json.load(open(out/f'E57_pqa8_rows_{c}_{l.replace("@","_")}.json'));W08={c:gold(ld8(c,'W0')) for c in 'ABC'};sel=json.load(open(out/'e57_pqa8_selection.json'))
        for arm,job in ARMS:
            for n in NREAD:
                l=f'{arm}@n{n}';imp={c:gold(ld8(c,l))-W08[c] for c in 'ABC'};res['absent'][l]=dict(n=len(rm['item_ids']),**{c:boot(imp[c]) for c in 'ABC'},gain_A_minus_B=boot(imp['A']-imp['B']))
        for a1,j1,a2,j2 in pairs:
            for n in NREAD:
                res['absent'][f'{a1}-{a2}@n{n}']={c:boot((gold(ld8(c,f'{a1}@n{n}'))-W08[c])-(gold(ld8(c,f'{a2}@n{n}'))-W08[c])) for c in 'ABC'}
        gj=json.load(open(out/'E57_pqa8_generic.json'));g0=np.asarray([gj['nll']['W0'][k]['mean_nll_token_weighted'] for k in sorted(gj['nll']['W0'],key=int)]);res['generic']={'W0_mean_nll':float(g0.mean()),'n_paragraphs':gj['n_paragraphs'],'tokens_scored':gj['tokens_scored'],'sha256_texts':gj['sha256_texts'],'rise_vs_W0_over_groups':{}}
        for arm,job in ARMS:
            for n in NREAD:
                gl=np.asarray([gj['nll'][f'{arm}@n{n}'][k]['mean_nll_token_weighted'] for k in sorted(gj['nll']['W0'],key=int)]);res['generic']['rise_vs_W0_over_groups'][f'{arm}@n{n}']=boot(gl-g0)
        res['generic']['canary']=gj.get('canary');res['selection']={k:sel[k] for k in ('n_non455','n_selected','n_dropped','types','types_dropped','seed','cap')}
    (out/'E57_pqa8_keys.json').write_text(json.dumps(dict(meta=dict(strata='occurrences of the gold string (case-insensitive, non-alphanumeric boundaries) in the text of the item\'s group summed over the first n versions',gold_lp_rise='gold log-prob (sum) of the state minus W0',items_sha256=h),**res),indent=1,default=float))
def selftest():
    P6.selftest();assert [stratum(x) for x in (0,1,2,3,5,6,40)]==[0,1,2,3,3,4,4]
    t='paris is big. Paris, paris2 PARIS';assert count_in(t.lower(),'Paris')==3
    groups=[{'gid':0,'pids':['a']},{'gid':1,'pids':['b']}];paras=[{'pid':'a','context':'Alpha beta'},{'pid':'b','context':'Gamma delta'}];rew={'a':['x']*8,'b':['y']*8};gt=group_texts(paras,rew,groups)
    raw=[{'id':'i1','gold':'alpha','variants':['alpha']},{'id':'i2','gold':'zzz','variants':[]},{'id':'i3','gold':'gamma','variants':[]}]
    asg,dr,load=select(raw,groups,gt,set(),cap=5);assert asg['i1']==1 and asg['i3']==0 and 'i2' in asg and not dr
    print('e57_pqa8_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k);a=dict(out=g('--out'),src640=g('--src640'),src650=g('--src650'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),max_groups=g('--max-groups'),canary=g('--canary'),para=g('--para'),rewrites8=g('--rewrites8'),train=g('--train'),dev=g('--dev'))
    {'select':select_stage,'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
