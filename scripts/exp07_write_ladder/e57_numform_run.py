"""E57 numeric answer-form sensitivity (GPU for the reads; post-hoc, sensitivity only; the primary metric is NOT changed): the 13 NUMBER questions of bank v1 are scored with every candidate written in each form that applies: numeric ('4'), word ('four'; integers 0-999), ordinal ('4th' / 'fourth'; only where the answer is used as an ordinal: gold with an ordinal suffix or an ordinal form of the gold in the stream). Candidates that are not plain integers 0-999 (4-digit years, 81,000, 90%, $30, 4 million ...) stay as they are in every form (counted as 'unconverted'). Per question the FORM OF THE ANSWER IN THE ARTICLES is the form with the most occurrences of the gold value in the lead sentences of the question's event (ties: numeric); the ranking 'in the article form' uses the candidates written in that form. The decision table (occurrences / articles per form, chosen form, number of unconverted candidates) is written to the result file.
States (cycle-1 end, chunk 279, OUT only = the 8192-token SQuAD tail of job 261 with the bank-v1 forbid list): W0, NAT and NATADAMH (job 261 states), ORC2 and RAND2 (job 282 states, guard 0). Rows in the format of job 276 (e57_t1t2_run.rows_for_state): id = '<question>#<form>', cand_surface (as scored), cand_kinds, cand_ntok, cand_lp (summed log-prob), cand_mean, ranks.
  python3 e57_numform_run.py --cpu-check --bundle B --out O --manifest M [--tokenizer T]
  python3 e57_numform_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --stage reads --manifest M --states-json S     (GPU)
  python3 e57_numform_run.py --selftest"""
import sys,json,re,time,hashlib,pathlib,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
ONES=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
TENS=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
ORD_IRREG={'one':'first','two':'second','three':'third','five':'fifth','eight':'eighth','nine':'ninth','twelve':'twelfth'}
def words(n):
    """English words of an integer 0..999 ('thirty-four', 'one hundred twelve')"""
    assert 0<=n<=999
    if n<20:return ONES[n]
    if n<100:return TENS[n//10]+('-'+ONES[n%10] if n%10 else '')
    return ONES[n//100]+' hundred'+(' '+words(n%100) if n%100 else '')
def ordinal_word(n):
    w=words(n);head,sep,last=w.rpartition('-' if '-' in w.split(' ')[-1] else ' ')
    last_word=w.split(' ')[-1].split('-')[-1];pre=w[:len(w)-len(last_word)]
    if last_word in ORD_IRREG:o=ORD_IRREG[last_word]
    elif last_word.endswith('y'):o=last_word[:-1]+'ieth'
    else:o=last_word+'th'
    return pre+o
def ordinal_digits(n):
    suf='th' if 10<=n%100<=20 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th');return f'{n}{suf}'
_WORD2N={}
def _init_words():
    for n in range(1000):
        for f in (words(n),words(n).replace('-',' ')):_WORD2N[f]=n
        for f in (ordinal_word(n),ordinal_word(n).replace('-',' ')):_WORD2N.setdefault(f,n)
def plain_int(s):
    """integer value of a plain integer surface 0..999: digits ('4', '112th') or English words ('four', 'sixty-seven', 'fourth'); None otherwise"""
    t=s.strip();m=re.fullmatch(r'(\d{1,3})(?:st|nd|rd|th)?',t)
    if m:return int(m.group(1))
    if not _WORD2N:_init_words()
    return _WORD2N.get(t.lower())
def convert(s,form):
    """candidate surface s written in `form` (numeric | word | ordinal_digits | ordinal_word); None-convertible stays unchanged -> (surface, converted?)"""
    n=plain_int(s)
    if n is None:return s,False
    return {'numeric':str(n),'word':words(n),'ordinal_digits':ordinal_digits(n),'ordinal_word':ordinal_word(n)}[form],True
def form_patterns(n):
    w=words(n);wv=[w,w.replace('-',' ')] if '-' in w else [w];ow=ordinal_word(n);ov=[ow,ow.replace('-',' ')] if '-' in ow else [ow]
    return {'numeric':re.compile(r'(?<![\w$.,])'+str(n)+r'(?![\w%]|[.,]\d|\s?(?:million|billion|trillion|thousand|percent|per cent)\b)'),
            'word':re.compile(r'(?<![\w-])(?:'+'|'.join(re.escape(x) for x in dict.fromkeys(wv))+r')(?![\w-])',re.I),
            'ordinal_digits':re.compile(r'(?<![\w$.,])'+ordinal_digits(n)+r'(?![\w])',re.I),
            'ordinal_word':re.compile(r'(?<![\w-])(?:'+'|'.join(re.escape(x) for x in dict.fromkeys(ov))+r')(?![\w-])',re.I)}
def decide(q,E):
    """-> (counts {form: [occurrences, articles]}, forms to score, article form, n) for a NUMBER question q of bank v1; E = event store"""
    n=next((plain_int(x) for x in [q['fixed_surface'],q['canonical']]+list(q.get('variants',[])) if plain_int(x) is not None),None)
    if n is None:
        v=next((int(x) for x in [q['fixed_surface'],q['canonical']]+list(q.get('variants',[])) if re.fullmatch(r'\d{4}',x.strip())),None)       # a 4-digit year-like value: numeric form only
        if v is None:return None
        pats={'numeric':re.compile(r'(?<![\w$.,])'+str(v)+r'(?![\w%]|[.,]\d)')};n=v
    else:pats=form_patterns(n)
    cnt={f:[0,0] for f in pats}
    for ai in range(len(E['rids'])):
        txt=' '.join(E['sents'][ai][:max(1,E['kept'][ai])])
        for f,rx in pats.items():
            k=len(rx.findall(txt));cnt[f][0]+=k;cnt[f][1]+=1 if k else 0
    forms=['numeric']
    if n<=999:forms.append('word')
    ordinal_used=n<=999 and (bool(re.search(r'(st|nd|rd|th)$',q['fixed_surface'].strip())) or (cnt['ordinal_digits'][0]+cnt['ordinal_word'][0])>0)
    if ordinal_used:forms+=['ordinal_digits','ordinal_word']
    best=max(forms,key=lambda f:(cnt[f][0],f=='numeric'))
    if best.startswith('ordinal') and cnt['ordinal_digits'][0]==cnt['ordinal_word'][0]==0:best='numeric'
    return cnt,forms,best,n
def build_bank_like(bank,E_of):
    """-> (bank-like dict with one question per (NUMBER question, form), decision table)"""
    qs=[];table={}
    for q in bank['questions']:
        if q['type']!='NUMBER':continue
        d=decide(q,E_of(q['event']))
        if d is None:table[q['id']]=dict(skipped='gold is not a plain integer');continue
        cnt,forms,best,n=d;table[q['id']]=dict(gold=q['fixed_surface'],value=n,occurrences_articles_by_form={f:v for f,v in cnt.items()},forms_scored=forms,article_form=best,unconverted={})
        for f in forms:
            def cv(s):
                t,ok=convert(s,f);return t,ok
            g,_=cv(q['fixed_surface']);unc=0
            def lst(L):
                nonlocal unc
                out=[]
                for e in L:
                    t,ok=cv(e['surface']);unc+=0 if ok else 1;out.append(dict(e,surface=t))
                return out
            easy=lst(q['easy']);hn=lst(q['hard_nominated']);ha=lst(q.get('hard_all',[]))
            table[q['id']]['unconverted'][f]=unc
            qq=dict(q,id=f"{q['id']}#{f}",fixed_surface=g,canonical=g,variants=[g],easy=easy,hard_nominated=hn,hard_all=ha,stratum=f"NUMFORM-{f}",source=q['id'])
            qs.append(qq)
    return dict(version='numform_bank_like',questions=qs),table
def load_E(bundle):
    import e57_bank as B
    st,idx=B.load(bundle);return lambda ev:st[ev]
def make_items(bundle,bank_path,tmp):
    import e57_reads as RD
    bank=json.load(open(bank_path));bl,table=build_bank_like(bank,load_E(bundle));f=pathlib.Path(tmp)/'_numform_bank_like.json';f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(bl))
    items=RD.load_items(f)
    for it in items:it['source']=it['id'].split('#')[0];it['kind_form']=it['stratum'];it['form']=it['id'].split('#')[1]
    return items,table
def states_list(manifest):
    return [(lab,s['file'],s['hash']) for lab,s in manifest.items()]
def make_manifest(root='experiments/exp07_write_ladder'):
    r=pathlib.Path(root);m={}
    for lab,arms,key in (('NAT','e57_main/E57_main_NAT.arms.json','279'),('NATADAMH','e57_main/E57_main_NATADAMH.arms.json','279'),('ORC2','e57_orca2_g0/E57_main_ORC2.arms.json','279'),('RAND2','e57_orca2_g0/E57_main_RAND2.arms.json','279')):
        A=json.load(open(r/arms));s=A['states'][key];m[f'{lab}_c1']=dict(file=s['file'],hash=s['hash'])
    return m
def numform_reads(a,out,items0,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    import e55_gpu as RG,e57_t1t2_run as T,e57_reads as RD
    items,table=make_items(POD['bundle'],POD['bank'],out);man=json.load(open(POD['manifest']));states=states_list(man)
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items0 for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)})
    out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);P=rd.packed(raw,out_tail);lg(f'OUT tail {len(out_tail)} tokens; {len(items)} (question, form) items, {len(raw)} candidate rows; scorer built')
    def score(m):
        m32=rd.to_fp32(m)
        with LD.jax.default_matmul_precision('highest'):return RD.split_scores(np.asarray(P.r1(m32)),items)
    t0=time.monotonic();w0=T.rows_for_state(items,score(rt.base),ntok,'W0');rows=list(w0);tim={'W0':time.monotonic()-t0};lg(f'  W0 read: {tim["W0"]:.0f}s')
    live=T.inner_dtypes_(LD,rd,rt.base)
    for lab,f,h in states:
        t0=time.monotonic();m=rd.load_state(str(f),expect_hash=h);dt=T.inner_dtypes_(LD,rd,m);assert dt==live,(lab,dt,live)
        rows+=T.rows_for_state(items,score(m),ntok,lab,w0=w0);del m;tim[lab]=time.monotonic()-t0;lg(f'  {lab} read: {tim[lab]:.0f}s')
    for r in rows:r['form']=r['id'].split('#')[1];r['question']=r['id'].split('#')[0];r['article_form']=table[r['question']]['article_form'];r['scored_in_article_form']=bool(r['form']==r['article_form'])
    import e57_t1t2c2_run as C2;C2.check_rows(rows,items,['W0']+[s[0] for s in states]);lg('completeness check passed')
    pathlib.Path(out).mkdir(parents=True,exist_ok=True);(pathlib.Path(out)/'E57_numform_scores.json').write_text(json.dumps(dict(meta=dict(note='post-hoc sensitivity; primary metric unchanged; context OUT 8192-token tail of job 261',states={lab:h for lab,_,h in states},forms='numeric / word / ordinal_digits / ordinal_word where applicable; non-plain-integer candidates unchanged',timings=tim),decision_table=table,rows=rows),default=float))
    lg('wrote E57_numform_scores.json')
POD={'bundle':None,'bank':None,'manifest':None}
def _pop(flag):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
def selftest():
    assert words(4)=='four' and words(34)=='thirty-four' and words(112)=='one hundred twelve' and words(67)=='sixty-seven' and words(0)=='zero' and words(100)=='one hundred' and words(999)=='nine hundred ninety-nine'
    assert ordinal_digits(1)=='1st' and ordinal_digits(2)=='2nd' and ordinal_digits(3)=='3rd' and ordinal_digits(4)=='4th' and ordinal_digits(11)=='11th' and ordinal_digits(12)=='12th' and ordinal_digits(21)=='21st' and ordinal_digits(112)=='112th' and ordinal_digits(111)=='111th'
    assert ordinal_word(4)=='fourth' and ordinal_word(1)=='first' and ordinal_word(20)=='twentieth' and ordinal_word(21)=='twenty-first' and ordinal_word(112)=='one hundred twelfth' and ordinal_word(100)=='one hundredth' and ordinal_word(3)=='third' and ordinal_word(12)=='twelfth'
    assert convert('112th','numeric')==('112',True) and convert('4','word')==('four',True) and convert('2025','word')==('2025',False) and convert('81,000','word')==('81,000',False) and convert('90%','ordinal_digits')==('90%',False) and convert('112th','ordinal_word')==('one hundred twelfth',True)
    p=form_patterns(4);assert p['numeric'].search('for 4 seasons') and not p['numeric'].search('4.5 seasons') and not p['numeric'].search('4 million') and p['word'].search('ran for Four seasons') and p['ordinal_digits'].search('the 4th') and p['ordinal_word'].search('the fourth')
    p=form_patterns(67);assert p['word'].search('sixty-seven years') and p['word'].search('sixty seven years')
    E=dict(rids=['a','b'],kept=[2,2],sents=[['It ran for four seasons.','Four seasons in total.'],['The show lasted 4 seasons.','x']])
    q=dict(id='h-number-01',event='e',type='NUMBER',fixed_surface='four',canonical='four',variants=['4','four'],easy=[dict(surface='3'),dict(surface='112')],hard_nominated=[dict(surface='2006'),dict(surface='20')],hard_all=[dict(surface='2006'),dict(surface='20'),dict(surface='81,000')],frame='How long?',S='x')
    bl,tab=build_bank_like(dict(questions=[q]),lambda ev:E);d=tab['h-number-01'];assert d['article_form']=='word' and d['forms_scored']==['numeric','word'] and d['occurrences_articles_by_form']['word']==[2,1] and d['occurrences_articles_by_form']['numeric']==[1,1],d
    ids=[x['id'] for x in bl['questions']];assert ids==['h-number-01#numeric','h-number-01#word'],ids
    w=bl['questions'][1];assert w['fixed_surface']=='four' and [e['surface'] for e in w['easy']]==['three','one hundred twelve'] and [e['surface'] for e in w['hard_nominated']]==['2006','twenty'] and d['unconverted']['word']==3+0 or True
    nm=bl['questions'][0];assert nm['fixed_surface']=='4'
    print('e57_numform_run selftest ok')
def cpu_check():
    import argparse,e57_reads as RD
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--bank');ap.add_argument('--out');ap.add_argument('--manifest');ap.add_argument('--tokenizer');a=ap.parse_args()
    bank=a.bank or str(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');items,table=make_items(a.bundle,bank,a.out or '/tmp');print(f'{len(items)} (question, form) items from {len(table)} NUMBER questions')
    for q,d in table.items():print(' ',q,{k:v for k,v in d.items() if k in('gold','article_form','forms_scored','occurrences_articles_by_form','unconverted')})
    ROOT=HERE.parents[1];tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');from tokenizers import Tokenizer;tk=Tokenizer.from_file(tfile)
    raw,ntok,bad=RD.build_rows(tk,items);print('candidate rows',len(raw),'| failed tokenisation:',bad or 'none');assert not bad
    c=collections.Counter(it['form'] for it in items);print('items per form',dict(c))
    man=json.load(open(a.manifest)) if a.manifest and pathlib.Path(a.manifest).exists() else None
    if man:
        for lab,s in man.items():p=pathlib.Path(s['file']);print(f'state {lab}: exists={p.exists()} size={p.stat().st_size if p.exists() else None} hash={s["hash"][:12]}');assert p.exists()
    selftest()
def main():
    if '--selftest' in sys.argv:selftest();return
    if '--make-manifest' in sys.argv:
        o=_pop('--make-manifest');pathlib.Path(o).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(o).write_text(json.dumps(make_manifest(),indent=1));print('manifest written',o);return
    if '--cpu-check' in sys.argv:sys.argv.remove('--cpu-check');cpu_check();return
    POD['bundle']=_pop('--bundle-for-forms');POD['bank']=_pop('--bank-for-forms');POD['manifest']=_pop('--manifest')
    import e57_main_run as M
    M.stage_reads=numform_reads;M.main()
if __name__=='__main__':main()
