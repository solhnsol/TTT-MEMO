"""E56 Tier 0: paraphrase-version validation (design E56 2.2). Versions are JSON {"paras":[{"kind","text"}]} aligned 1:1 with ORIG paragraphs (headings verbatim).
Usage: e56_sim.py check --orig ORIG.json --bank bank.json --paper P1 --versions v1.json,v2.json,... --out report.json | --selftest"""
import argparse,json,re,sys,pathlib,itertools
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import e56_text as T,e56_bank as B
NUM=re.compile(r'\d+(?:[.,]\d+)*')
def vsents(v):return [s for p in v['paras'] if p['kind']=='p' for s in T.split_sents(p['text'])]
def para_of_sent(v):
    m=[];[m.extend([i]*len(T.split_sents(p['text']))) for i,p in enumerate(v['paras']) if p['kind']=='p'];return m
def full(v):return ' '.join(p['text'] for p in v['paras'])
def a_and_s(sents,it):return [s for s in sents if B.has(s,it['A']) and any(B.has(s,x) for x in it['S'])]
def check_version(orig,v,items,enc,tol=0.15):
    f=[];n0=T.ntok(enc,T.flat_text(orig));n1=T.ntok(enc,T.flat_text(dict(paras=[dict(p,sents=None) for p in v['paras']])))
    if len(v['paras'])!=len(orig['paras']):f.append(('structure','paragraph count differs'));return f,dict(tokens=n1,orig_tokens=n0)
    for i,(po,pv) in enumerate(zip(orig['paras'],v['paras'])):
        if po['kind']!=pv['kind']:f.append(('structure',f'para {i} kind'))
        if po['kind']=='h' and po['text']!=pv['text']:f.append(('heading',f'para {i} changed'))
    if abs(n1-n0)/n0>tol:f.append(('length',f'{n1} vs {n0} tokens ({(n1-n0)/n0:+.1%})'))
    so=B.sents_of(orig);sv=vsents(v)
    for it in items:
        c1=len(a_and_s(sv,it));t1=sum(s.lower().count(it['A'].lower()) for s in sv)
        if c1!=it['c']:f.append(('c',f"{it['id']} c {c1} != {it['c']}"))
        if t1!=it['a_total']:f.append(('A_total',f"{it['id']} A count {t1} != {it['a_total']}"))
        if B.qoverlap6(it['q'],sv):f.append(('q_overlap',it['id']))
    if set(NUM.findall(' '.join(so)))!=set(NUM.findall(' '.join(sv))):
        d=set(NUM.findall(' '.join(so)))^set(NUM.findall(' '.join(sv)));f.append(('numbers',f'differ: {sorted(d)[:12]}'))
    return f,dict(tokens=n1,orig_tokens=n0)
def pair_stats(u,v,items,thr_j=0.3,thr_8=0.25):
    su,sv=(B.sents_of(u) if 'sents' in u['paras'][0] and u['paras'][0]['sents'] else vsents(u)),vsents(v);fails=[];mx=[]
    for it in items:
        best=0.0
        for a in a_and_s(su,it):
            ga=B.ngrams(B.words(B.mask_slots(a,it['A'],it['S'])),5)
            for b in a_and_s(sv,it):best=max(best,B.jacc(ga,B.ngrams(B.words(B.mask_slots(b,it['A'],it['S'])),5)))
        mx.append(best)
        if best>=thr_j:fails.append(('jaccard5',f"{it['id']} max {best:.2f}"))
    gu=B.ngrams(B.words(full(u)),8);gv=B.ngrams(B.words(full(v)),8);sh=max(len(gu&gv)/max(1,len(gv)),len(gu&gv)/max(1,len(gu)))
    if sh>thr_8:fails.append(('8gram',f'share {sh:.2f}'))
    return fails,dict(jacc_mean=sum(mx)/len(mx) if mx else 0,jacc_max=max(mx) if mx else 0,share8=sh)
def run(orig,versions,items,enc):
    rep=dict(versions=[],pairs=[]);allv=[dict(paras=[dict(p) for p in orig['paras']])]+versions
    for i,v in enumerate(versions):
        f,st=check_version(orig,v,items,enc);rep['versions'].append(dict(idx=i+1,fails=f,**st))
    for (i,u),(j,v) in itertools.combinations(enumerate(allv),2):
        f,st=pair_stats(u,v,items);rep['pairs'].append(dict(a='ORIG' if i==0 else f'V{i}',b=f'V{j}',fails=f,**st))
    rep['ok']=not any(x['fails'] for x in rep['versions']+rep['pairs'])
    rep['embedding_cosine']='not run: no embedding model in container (server or proxy needed)'
    return rep
def _selftest():
    class E:
        def encode(self,t,add_special_tokens=False):
            class R:ids=t.split()
            return R
    orig=dict(paras=[dict(kind='p',text='GoiBERT reaches 85.2 on tasks. Other facts here today. GoiBERT scores 85.2 overall.',sents=['GoiBERT reaches 85.2 on tasks.','Other facts here today.','GoiBERT scores 85.2 overall.'])])
    it=[dict(id='P1-000',A='85.2',S=['GoiBERT'],q='What does it reach',c=2,a_total=2)]
    ok=dict(paras=[dict(kind='p',text='On benchmarks the score of GoiBERT is 85.2. Further statements appear nowadays. Overall it is 85.2 for GoiBERT, seen again.')])
    f,_=check_version(orig,ok,it,E(),tol=1.0);assert not f,f
    bad=dict(paras=[dict(kind='p',text='GoiBERT hits 85.2 on tasks. Other facts here today.')])
    f,_=check_version(orig,bad,it,E(),tol=1.0);assert any(k=='c' for k,_ in f),f
    print('e56_sim selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',nargs='?');ap.add_argument('--orig');ap.add_argument('--bank');ap.add_argument('--paper');ap.add_argument('--versions');ap.add_argument('--out');ap.add_argument('--tokenizer-file');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();sys.exit()
    orig=json.load(open(a.orig));bank=json.load(open(a.bank));items=[r for r in bank['items'] if r['paper']==a.paper];vs=[json.load(open(f)) for f in a.versions.split(',')]
    rep=run(orig,vs,items,T.load_enc(a.tokenizer_file));json.dump(rep,open(a.out,'w'),indent=1)
    print('ok' if rep['ok'] else 'FAIL',[(v['idx'],len(v['fails'])) for v in rep['versions']],'pair fails',sum(1 for p in rep['pairs'] if p['fails']))
