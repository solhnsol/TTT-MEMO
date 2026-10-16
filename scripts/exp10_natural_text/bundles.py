"""Stage 2: per-person bundles (units with exposure labels), name forms, fact values. Real names stay in private outputs."""
import pickle,re,json,sys
from collections import Counter
from tokenizers import Tokenizer
import os;NT=os.environ.get('NT_DIR','/tmp/claude-0/nt');W=NT+'/work';DL=NT+'/dl'
LT=Tokenizer.from_file(f'{DL}/llama3_tok.json');QT=Tokenizer.from_file(f'{DL}/qwen_tok.json')
def nl(s):return len(LT.encode(s,add_special_tokens=False).ids)
def nq(s):return len(QT.encode(s,add_special_tokens=False).ids)
def base(t):return re.sub(r'\s*\(.*\)$','',t).strip()
STOPH=re.compile(r'^(References|External links|See also|Notes|Further reading|Bibliography|Sources|Works|Selected works|Filmography|Discography|Publications|Citations|Footnotes|Honours|Awards|Gallery)\s*$',re.M)
NAT=re.compile(r"\b(?:is|was) (?:an?|the) (?:former |retired |professional |noted |prominent )?((?:[A-Z][a-z]+)(?:-[A-Z][a-z]+)?)\b")
HE=re.compile(r'\b(?:he|him|his)\b',re.I);SHE=re.compile(r'\b(?:she|her|hers)\b',re.I)
SENT=re.compile(r'(?<=[.!?])\s+(?=[A-Z"“(])')
def wb(s,x):return re.search(r'(?<![\w])'+re.escape(x)+r'(?![\w])',s) is not None
def paragraphs(tx):
    m=STOPH.search(tx);tx=tx[:m.start()] if m else tx
    out=[]
    for p in tx.split('\n\n'):
        p=p.strip()
        if len(p)<80 or p.startswith(' ') or '\n' in p and sum(len(l)<60 for l in p.split('\n'))>1:continue
        lines=p.split('\n')
        if len(lines)>1 and len(lines[0])<40 and not lines[0].rstrip().endswith('.'):p='\n'.join(lines[1:])
        p=' '.join(p.split())
        # drop parentheticals that can carry the real name: native scripts, pronunciation, birth/maiden names, nicknames
        p=re.sub(r'\s*\([^()]*(?:[\u0370-\uFFFF]|pronounced|née|né |born [A-Z]|nicknamed|also known|known as|aka |;\s*$)[^()]*\)','',p)
        p=re.sub(r'\s*\(\s*[,;]*\s*\)','',p)
        if len(p)>=80:out.append(p)
    return out
def ngrams(s,n=8):
    w=s.split();return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
def split_unit(p,maxtok=400):
    if nl(p)<=maxtok:return [p]
    ss=SENT.split(p);out=[];cur=''
    for s in ss:
        if cur and nl(cur+' '+s)>maxtok:out.append(cur);cur=s
        else:cur=(cur+' '+s).strip()
    if cur:out.append(cur)
    return out
def main():
    C=pickle.load(open(f'{W}/cands.pkl','rb'));M=pickle.load(open(f'{W}/scan1_all.pkl','rb'))
    P=[];rej=Counter()
    for t,c in C.items():
        tx=M['targets'].get(t)
        if not tx:rej['no_article']+=1;continue
        full=base(t);toks=full.split()
        if not(2<=len(toks)<=4) or not all(re.fullmatch(r"[A-Z][a-zA-Z'\-]+",x) for x in toks):rej['name_shape']+=1;continue
        m=NAT.search(tx[:350])
        if not m:rej['no_person_lead']+=1;continue
        nat=m.group(1);h=len(HE.findall(tx[:6000]));s_=len(SHE.findall(tx[:6000]))
        g='m' if h>=9*max(s_,1)/1 and h>=3 else ('f' if s_>=9*max(h,1) and s_>=3 else None)
        if g is None:
            r=h/(h+s_) if h+s_ else .5
            g='m' if r>=.9 and h>=3 else ('f' if r<=.1 and s_>=3 else None)
        if g is None:rej['gender']+=1;continue
        bc=M['bcount'].get(full,0)
        if bc>60:rej['popular_backlinks>60']+=1;continue
        if any(f.get('s_pop',0)>2000 for f in c['facts'].get('birth',[])):rej['s_pop']+=1;continue
        paras=paragraphs(tx)
        body=' '.join(paras)
        def pick(rel):
            for f in c['facts'].get(rel,[]):
                for v in [f['obj']]+f['aliases']:
                    if len(v)>=3 and wb(body,v):return v,f
            return None,None
        bv,bf=pick('birth')
        sv,sf=pick('school')
        if not bv and not sv:rej['no_fact_value_in_text']+=1;continue
        aliases=[a for f in c['facts'].get('birth',[]) for a in f.get('s_aliases',[]) if a!=full and len(a.split())>=2]
        forms={'full':full,'given':toks[0],'surname':toks[-1],'middle':toks[1:-1],'aliases':aliases}
        nameforms=[full,toks[-1],toks[0]]+aliases+[toks[0]+' '+toks[-1]]
        units=[]
        for p in paras:
            for u in split_unit(p):units.append({'src':'article','text':u})
        ag=ngrams(body)
        anchors={x for x in re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',body) if not any(tk in x for tk in toks)}
        nb=0
        for (bt,bp) in M['back'].get(full,[]):
            if base(bt)==full:continue
            bp=bp.strip()
            if '\n' in bp or bt.startswith(('List of','Lists of')) or '(disambiguation)' in bt or bp.count('(')>4 or re.search(r'References|External links|Notable people|Notable residents',bp):continue
            cl=paragraphs(bp)
            if len(cl)!=1:continue
            bp=cl[0]
            if not((bv and wb(bp,bv)) or (sv and wb(bp,sv)) or any(a in bp for a in anchors)):rej_b='noanchor';continue
            ng=ngrams(bp);cont=len(ng&ag)/max(len(ng),1)
            if cont>0.5:continue
            for u in split_unit(bp):units.append({'src':'backlink','from':bt,'text':u,'contain8':round(cont,3),'flag_overlap':cont>=0.2})
            nb+=1
        for u in units:
            ss=SENT.split(u['text'])
            nm=[any(wb(s,f) for f in nameforms) for s in ss]
            hv=[bool(bv) and wb(s,bv) for s in ss];hs=[bool(sv) and wb(s,sv) for s in ss]
            u['n_name_sent']=sum(nm);u['n_fact_strict']=sum(a and b for a,b in zip(nm,hv))
            u['n_value_sent']=sum(hv)
            u['has_school']=bool(sv and wb(u['text'],sv))
            u['n_school_fact']=sum(a and b for a,b in zip(nm,hs));u['label']='fact' if ((any(hv) or any(hs)) and any(nm)) else ('name_only' if any(nm) else ('value_only' if any(hv) else 'other'))
            u['ntok']=nl(u['text'])
        if not any(u['label']=='fact' for u in units):rej['no_fact_unit']+=1;continue
        P.append({'title':t,'src':sorted(c['src']),'nat':nat,'gender':g,'backlinks':bc,'n_backlink_paras_used':nb,'forms':forms,
                  'birth':bv,'birth_otitle':bf['o_title'] if bf else None,'birth_opop':bf.get('o_pop') if bf else None,'school':sv,'units':units,
                  's_pop':min([f.get('s_pop',10**9) for f in c['facts'].get('birth',[])]+[10**9])})
    pickle.dump(P,open(f'{W}/persons.pkl','wb'))
    print('kept',len(P),'rejected',dict(rej))
    print('nat',Counter(p['nat'] for p in P).most_common(15))
    print('gender',Counter(p['gender'] for p in P),'src',Counter(tuple(p['src']) for p in P))
    import statistics as st
    print('units med',st.median(len(p['units']) for p in P),'tok med',st.median(sum(u['ntok'] for u in p['units']) for p in P))
    print('fact units med',st.median(sum(u['label']=='fact' for u in p['units']) for p in P),'name_only med',st.median(sum(u['label']=='name_only' for u in p['units']) for p in P))
    print('backlink units >0:',sum(p['n_backlink_paras_used']>0 for p in P),'school',sum(bool(p['school']) for p in P))
main()
