"""E56 Tier 0: arXiv HTML -> frozen body text, sentence split, token audit. CPU only, deterministic.
Usage: e56_text.py extract --html F --out DIR/ORIG.json | audit --json F [--tokenizer-file T] | --selftest"""
import argparse,json,re,sys,hashlib,pathlib
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
ABBR=('e.g','i.e','et al','vs','Fig','Figs','Eq','Eqs','Sec','Tab','No','approx','cf','etc','Dr','St')
def _clean(t):
    t=re.sub(r'\s+',' ',t).strip();t=re.sub(r'\s+([,.;:)])',r'\1',t);t=re.sub(r'\(\s+','(',t);return t
def extract(html):
    from bs4 import BeautifulSoup
    s=BeautifulSoup(html,'lxml');title=s.select_one('h1.ltx_title_document').get_text(' ',strip=True)
    for sel in ('cite','.ltx_cite','.ltx_ref_tag','sup.ltx_note','.ltx_note','.ltx_equation','.ltx_equationgroup','figure','table','.ltx_figure','.ltx_table','.ltx_bibliography','.ltx_listing','nav','.ltx_authors','.ltx_dates'):
        for e in s.select(sel):e.decompose()
    for m in s.select('math'):m.replace_with(m.get('alttext') or '')
    for r in s.select('a.ltx_ref'):r.replace_with(r.get_text(' ',strip=True))
    blocks=[];stop=False;abs_=s.select_one('.ltx_abstract')
    if abs_:
        for p in abs_.select('p.ltx_p'):blocks.append(('p',_clean(p.get_text(' ',strip=True))))
        abs_.decompose()
    for e in s.select('h2.ltx_title,h3.ltx_title,h4.ltx_title,p.ltx_p,div.ltx_para > p,li.ltx_item p'):
        if e.name=='p' and e.get('class') and 'ltx_p' not in e.get('class') and e.parent.get('class') and 'ltx_para' not in e.parent.get('class'):continue
        c=e.get('class') or []
        if e.name in('h2','h3','h4'):
            if 'ltx_title_bibliography' in c or 'ltx_title_appendix' in c:stop=True
            tx=_clean(e.get_text(' ',strip=True))
            if tx.lower().startswith('instructions for reporting'):stop=True
            if not stop:blocks.append(('h',tx))
        elif not stop:
            tx=_clean(e.get_text(' ',strip=True))
            if tx and (not blocks or blocks[-1]!=('p',tx)):blocks.append(('p',tx))
    return title,blocks
def split_sents(p):
    """freeze: split at . ? ! followed by space+capital/digit/quote, not after known abbreviations or single capital initials."""
    out=[];i=0;start=0
    for m in re.finditer(r'([.?!])(["\')\]]*)\s+(?=[A-Z0-9"\'(\[])',p):
        end=m.end(2);head=p[start:m.start()+1]
        w=re.findall(r'(\S+)$',p[start:m.start()])
        last=w[0].lstrip('("\'[') if w else ''
        if m.group(1)=='.' and (last in ABBR or re.fullmatch(r'[A-Z]',last) or re.fullmatch(r'\d+',last) and False):continue
        out.append(p[start:end].strip());start=m.end()
    if p[start:].strip():out.append(p[start:].strip())
    return out
def build(title,blocks):
    paras=[];cur=None
    for k,t in blocks:
        if k=='h':paras.append({'kind':'h','text':t,'sents':[t]})
        else:paras.append({'kind':'p','text':t,'sents':split_sents(t)})
    return {'title':title,'paras':paras}
def flat_text(doc):return '\n\n'.join(p['text'] if p['kind']=='p' else '## '+p['text'] for p in doc['paras'])+'\n'
def load_enc(tf=None):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(tf or str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
def ntok(enc,t):return len(enc.encode(t,add_special_tokens=False).ids)
def sha(o):return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def _selftest():
    assert split_sents('We use e.g. BERT. Then it works! Results (Fig. 3) show 5.2 gains. Done.')==['We use e.g. BERT.','Then it works!','Results (Fig. 3) show 5.2 gains.','Done.']
    assert split_sents('J. Smith wrote it. OK.')==['J. Smith wrote it.','OK.']
    print('e56_text selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',nargs='?');ap.add_argument('--html');ap.add_argument('--out');ap.add_argument('--json');ap.add_argument('--tokenizer-file');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest or a.cmd=='selftest':_selftest();sys.exit()
    if a.cmd=='extract':
        t,b=extract(open(a.html,encoding='utf8').read());d=build(t,b);d['sha256']=sha(d['paras'])
        pathlib.Path(a.out).write_text(json.dumps(d,ensure_ascii=False,indent=1));pathlib.Path(a.out).with_suffix('.txt').write_text(flat_text(d))
        enc=load_enc(a.tokenizer_file);n=ntok(enc,flat_text(d));ns=sum(len(p['sents']) for p in d['paras'] if p['kind']=='p')
        print(f"{t[:60]} | paras {len(d['paras'])} sents {ns} tokens {n} sha {d['sha256'][:12]}")
