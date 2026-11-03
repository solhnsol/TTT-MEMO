"""E57 public QA stage 14 (new file; read only, NO weight update; thread 'QA에서 가림이 안 되는 이유 규명', Hans '진행해' 11/03 11:29 KST): read-format comparison on the saved n = 9 states of the paragraph-group episodes (NATQ, ALLQ128 of job 640; GAINQ512 p.10 of job 650) and W0, items455, the original 16 candidates of job 674, scored as continuations (FastOUTG packing per group, per-token log-probs).
Formats: Q = the question-style read of job 674 (condition A, prefix = the item's question; the rows come FIRST in the group so that the packing equals job 674; the summed and token log-probs are compared with E57_pqa13_rows_A_*); C-orig = from the start of the sentence of the item's original paragraph in which the answer first appears up to just before the answer; C-rw3 / C-rw6 / C-rw9 = the same rule in version 3 / 6 / 9 of the paragraph (job 720 numbering: v1 original, v2..v9 = rewrites 1..8, so v3 = rw2, v6 = rw5, v9 = rw8); C-short = the last 6 words of the C-orig prompt. The answer (gold string) is found with the rule of the rewrite checker: case, whitespace and the characters , . ; : are ignored; items whose answer is not found, or whose prompt is empty (answer at the start of the sentence), are left out of that format and counted. Sentences: e57_pqa2_stream.sentences rule on the whitespace-collapsed paragraph. The candidate gets a leading space (build_rows) like in job 674; the prompt is cut right before the answer and right-stripped.
Per state x format (all items and by answer type LOW / CAP / NUM / DATE): log-prob of the first content token (space-only tokens skipped), of the second token, mean log-prob of the remaining tokens, summed log-prob; rank ratio and top-1 over the 16 candidates by the first content token, by the branch point (first content token to the first differing content token) and by the full sum; DeltaA vs W0 (state minus W0, rank ratio as improvement); masking extra = DeltaA(ALLQ128) - DeltaA(NATQ) and DeltaA(GAINQ512P10) - DeltaA(NATQ); item-level bootstrap 95% CIs (2000 draws). B is not defined for the cloze formats and is not used.
  python3 e57_pqa14_run.py --stage build --out O --items-file F | --stage read --out O ... | --stage analyze --out O | --selftest"""
import sys,re,json,time,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa2_stream as S
import e57_pqa4_run as P4
import e57_pqa6_run as P6
import e57_pqa12_run as P12
ARMS=[('NATQ','640'),('ALLQ128','640'),('GAINQ512P10','650')];FMTS=['Q','orig','rw3','rw6','rw9','short'];VER={'rw3':3,'rw6':6,'rw9':9};TYPES=['LOW','CAP','NUM','DATE']
def norm_map(t):
    ch=[];ix=[]
    for i,c in enumerate(t):
        if c.isspace() or c in ',.;:':continue
        ch.append(c.lower());ix.append(i)
    return ''.join(ch),ix
def find_answer(text,gold):
    """first occurrence of gold in text ignoring case, whitespace and , . ; : -> char index in text, or None"""
    n,ix=norm_map(text);g,_=norm_map(gold)
    if not g:return None
    p=n.find(g);return None if p<0 else ix[p]
def cloze_prefix(par,gold):
    """-> (prefix, reason): sentence start up to just before the answer"""
    t=' '.join(par.split());a=find_answer(t,gold)
    if a is None:return None,'not_found'
    starts=[0]+[m.end() for m in S.SPLIT.finditer(t)];s0=max(x for x in starts if x<=a);pre=t[s0:a].rstrip()
    return (pre,'ok') if pre else (None,'empty')
def last_words(pre,n=6):return ' '.join(pre.split()[-n:])
def build_prompts(items,para,rew):
    byp={p['pid']:p for p in para};own={}
    for p in para:
        for i in p['item_ids']:own[i]=p['pid']
    F={f:{} for f in FMTS};ex={f:{'not_found':0,'empty':0} for f in FMTS}
    for it in items:
        i=it['id'];F['Q'][i]=it['prefix'];pid=own[i]
        for f in ('orig','rw3','rw6','rw9'):
            v=1 if f=='orig' else VER[f];text=byp[pid]['context'] if v==1 else rew[pid][v-2]
            pre,why=cloze_prefix(text,it['cands'][0])
            if pre is None:ex[f][why]+=1
            else:F[f][i]=pre
        if i in F['orig']:F['short'][i]=last_words(F['orig'][i])
    ex['short']={k:ex['orig'][k] for k in ex['orig']}
    return F,ex
def load_inputs(root):
    para=json.load(open(root/'e57_public_qa_stage3/data/paragraphs.json'));rew=json.load(open(root/'e57_public_qa_stage5/data/rewrites_all8.json'));return para,rew
def read_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True);root=pathlib.Path(a['root']);meta,_=P6.load_groups(a['src640']);W={'640':json.load(open(pathlib.Path(a['src640'])/'e57_pqa6_writes.json')),'650':json.load(open(pathlib.Path(a['src650'])/'e57_pqa7_writes.json'))}
    G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups'];LD,rt,rt32,rd,items,h,raw0,ntok0,out_tail,tsha=P2.setup_read(a,lg);import e57_reads as RD
    byid={it['id']:it for it in items};para,rew=load_inputs(root);F,ex=build_prompts([byid[i] for g in meta['groups'] for i in g['item_ids']],para,rew)
    lg('items per format:',{f:len(F[f]) for f in FMTS},'excluded:',ex);grows=[];gnt=[];gsz=[];gits=[];tokids={f:{} for f in FMTS}
    for g in G:
        rows=[];nts=[];sz=[];its_f=[]
        for f in FMTS:
            its=[dict(byid[i],prefix=F[f][i]) for i in g['item_ids'] if i in F[f]];r_,n_,bad=RD.build_rows(rt.tok,its);assert not bad,bad
            for k,it in enumerate(its):tokids[f][it['id']]=[[int(x) for x in r_[k*16+j][0][r_[k*16+j][1][0]+1:r_[k*16+j][1][1]+1]] for j in range(16)]
            rows+=r_;nts.append(n_);sz.append(len(r_));its_f.append(its)
        grows.append(rows);gnt.append(nts);gsz.append(sz);gits.append(its_f)
    (out/'e57_pqa14_tokens.json').write_text(json.dumps(tokids));(out/'e57_pqa14_prompts.json').write_text(json.dumps(dict(prompts=F,excluded=ex)))
    m32w0=rd.to_fp32(rt.base);fo=P12.FastOUTG3(rt32,m32w0,out_tail,grows,log=lg);labels=['W0']+[f'{arm}@n9' for arm,_ in ARMS];R={f:{l:[] for l in labels} for f in FMTS}
    def put(label,gi,m32):
        toks=fo.score_group_tokens(m32,gi);o=0
        for fi,f in enumerate(FMTS):
            tc=toks[o:o+gsz[gi][fi]];o+=gsz[gi][fi];its=gits[gi][fi]
            if not its:continue
            rows=P2.rows_of(its,gnt[gi][fi],np.asarray([t.sum() for t in tc]),f'{f}:{label}')
            for k,r in enumerate(rows):r['cand_tok_lp']=[[float(x) for x in t] for t in tc[k*16:(k+1)*16]]
            R[f][label]+=rows
    t00=time.monotonic()
    for gi,g in enumerate(G):
        t0=time.monotonic();put('W0',gi,m32w0)
        for arm,job in ARMS:
            s=W[job][str(g['gid'])][arm]['states'][str(g['read_chunks']['9'])];m=rd.load_state(s['file'],expect_hash=s['hash']);put(f'{arm}@n9',gi,rd.to_fp32(m));del m
        lg(f'  group {g["gid"]} read: {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
    chk={}
    for l in labels:                                                      # reproduction of job 674 (condition A rows)
        ref={r['id']:r for r in json.load(open(pathlib.Path(a['ref674'])/f'E57_pqa13_rows_A_{l.replace("@","_")}.json'))};d=0.;dt=0.
        for r in R['Q'][l]:
            q=ref[r['id']];d=max(d,float(np.abs(np.asarray(r['cand_lp'])-np.asarray(q['cand_lp'])).max()))
            for x,y in zip(r['cand_tok_lp'],q['cand_tok_lp']):dt=max(dt,float(np.abs(np.asarray(x)-np.asarray(y)).max()))
        chk[l]=dict(max_abs_diff_sum=d,max_abs_diff_token=dt,n_items=len(R['Q'][l]))
    lg('reproduction of job 674 (Q = condition A):',chk)
    for f in FMTS:
        for l in labels:(out/f'E57_pqa14_rows_{f}_{l.replace("@","_")}.json').write_text(json.dumps(R[f][l],default=float))
    (out/'e57_pqa14_read_meta.json').write_text(json.dumps(dict(items_sha256=h,out_tail_sha=tsha,labels=labels,formats=FMTS,n_items_per_format={f:len(F[f]) for f in FMTS},excluded=ex,reproduction_job674=chk,timings_s=time.monotonic()-t00)))
def item_stats(rows,tokids,decode):
    """per item arrays (one row per item of the format) of the metrics"""
    K=('first_lp','second_lp','rest_lp','full_lp','rr_first','top1_first','rr_branch','top1_branch','rr_full','top1_full');o={k:np.full(len(rows),np.nan) for k in K}
    for q,r in enumerate(rows):
        tl=[np.asarray(x,np.float64) for x in r['cand_tok_lp']];ti=tokids[r['id']];ci=[[j for j,t in enumerate(ti[c]) if decode([t]).strip()!=''] for c in range(len(tl))]
        cl=[tl[c][ci[c]] for c in range(len(tl))];ct=[[ti[c][j] for j in ci[c]] for c in range(len(tl))]
        if not len(cl[0]):continue
        g=cl[0];o['first_lp'][q]=g[0];o['full_lp'][q]=tl[0].sum()
        if len(g)>1:o['second_lp'][q]=g[1]
        if len(g)>2:o['rest_lp'][q]=g[2:].mean()
        nf=nb=nu=0
        for c in range(1,len(tl)):
            if not len(cl[c]):continue
            nf+=int(cl[c][0]>g[0]);nu+=int(tl[c].sum()>tl[0].sum());tg,tc=ct[0],ct[c];t=next((x for x in range(min(len(tg),len(tc))) if tg[x]!=tc[x]),min(len(tg),len(tc))-1);nb+=int(cl[c][:t+1].sum()>g[:t+1].sum())
        o['rr_first'][q]=(1+nf)/16;o['top1_first'][q]=float(nf==0);o['rr_branch'][q]=(1+nb)/16;o['top1_branch'][q]=float(nb==0);o['rr_full'][q]=(1+nu)/16;o['top1_full'][q]=float(nu==0)
    return o
def bootn(x,seed=3,B=2000):
    x=np.asarray(x,np.float64);x=x[~np.isnan(x)]
    return None if len(x)==0 else dict(P4.boot(x,seed=seed,B=B),n=int(len(x)))
def analyze(a):
    out=pathlib.Path(a['out']);rm=json.load(open(out/'e57_pqa14_read_meta.json'));labels=rm['labels'];tok=json.load(open(out/'e57_pqa14_tokens.json'));items={it['id']:it for it in json.load(open(a['items']))['items']}
    tk=S.tokenizer();dec=lambda ids:tk.decode(ids);ld=lambda f,l:json.load(open(out/f'E57_pqa14_rows_{f}_{l.replace("@","_")}.json'));res={'meta':rm,'abs':{},'dA':{},'extra':{},'q_reference':None}
    LP=('first_lp','second_lp','rest_lp','full_lp');RR=('rr_first','rr_branch','rr_full');T1=('top1_first','top1_branch','top1_full')
    for f in FMTS:
        rows={l:ld(f,l) for l in labels};ids=[r['id'] for r in rows['W0']]
        if not ids:continue
        M={l:item_stats(rows[l],tok[f],dec) for l in labels};ty=np.asarray([items[i]['type'] for i in ids]);subs={'all':np.ones(len(ids),bool),**{t:ty==t for t in TYPES}}
        res['abs'][f]={l:{sn:{k:bootn(M[l][k][m]) for k in M[l]} for sn,m in subs.items() if m.any()} for l in labels}
        D={}
        for l in labels[1:]:
            d={k:M[l][k]-M['W0'][k] for k in LP+T1};d.update({k:M['W0'][k]-M[l][k] for k in RR});D[l]=d
        res['dA'][f]={l:{sn:{k:bootn(v[m]) for k,v in D[l].items()} for sn,m in subs.items() if m.any()} for l in labels[1:]}
        ex={}
        for a1 in ('ALLQ128@n9','GAINQ512P10@n9'):
            ex[f'{a1}-NATQ@n9']={sn:{k:bootn((D[a1][k]-D['NATQ@n9'][k])[m]) for k in D[a1]} for sn,m in subs.items() if m.any()}
        res['extra'][f]=ex
    qe=res['extra']['Q']['ALLQ128@n9-NATQ@n9']['all']['first_lp']['mean'];res['q_reference']=dict(first_lp_extra_ALLQ128_minus_NATQ=qe,note='prior criterion proposed by the thread (recorded, not applied): first-token extra in C-orig and C-rw at least twice the Q extra (+.189) -> key mismatch; small in every format -> dilution')
    res['ratio_to_Q_first_lp_extra']={f:{k:(res['extra'][f][k]['all']['first_lp']['mean']/qe if qe else None) for k in res['extra'][f]} for f in res['extra']}
    res['definitions']='first/second/rest = log-prob of the first content token / the second token / mean over the remaining tokens of the gold candidate (space-only tokens skipped); rr_* = rank / 16 among the 16 candidates (rr as improvement W0 minus state in dA / extra), first = by the first content token, branch = from the first content token to the first differing content token, full = summed log-prob; top1_* = share with rank 1; extra = dA(arm) - dA(NATQ); bootstrap over items (items missing a value are dropped from that metric)'
    fn=pathlib.Path(a['json']);fn.write_text(json.dumps(res,indent=1,default=float));print('wrote',fn)
def selftest():
    S.sentences('A b. C d.')
    assert find_answer('The  Big, Ten. conf','big ten')==5 and find_answer('abc','x') is None
    p,w=cloze_prefix('It was founded in 1990. The club won in Fresno, California last year.','fresno california');assert (p,w)==('The club won in','ok'),(p,w)
    assert cloze_prefix('Paris is big.','paris')==(None,'empty') and cloze_prefix('Paris is big.','rome')==(None,'not_found')
    assert last_words('a b c d e f g h')=='c d e f g h'
    dec=lambda ids:{1:' ',2:' x',3:'y'}[ids[0]];rows=[dict(id='i',cand_tok_lp=[[-1.,-2.,-3.]]+[[-1.5,-3.,-1.]]*15)];tok={'i':[[1,2,3]]+[[1,2,3]]*15}
    m=item_stats(rows,tok,dec);assert m['first_lp'][0]==-2. and m['second_lp'][0]==-3. and np.isnan(m['rest_lp'][0]) and m['rr_first'][0]==1/16 and m['top1_first'][0]==1.0
    print('e57_pqa14_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k);a=dict(out=g('--out'),src640=g('--src640'),src650=g('--src650'),items=g('--items-file'),bank=g('--bank'),stage1=g('--stage1'),ref674=g('--ref674'),root=g('--root'),max_groups=g('--max-groups'),json=g('--json'))
    {'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
