"""E57 post-hoc exploratory (CPU only, no model): which answer paths are left in the ORC v2 deleted windows (job 279, IN_DELORC_V2), per question at its primary position. Classification and counts only, no new mask rule.
Window re-built as e57_w0v2_run.plan (variant 'v2' hidden fragments, last 8192 tokens that remain, refilled from earlier chunks). For each selected question the sentences (fragments) of the window are split into OWN event (same event as the question) and OTHER events, and counted in five path classes:
 P1 other format of the gold: DATE numeric d/m, m/d, d.m, ISO, ordinal words ('nineteenth'), 'the 19th' / month-day in other spelling; NUMBER same value as a spelled-out number, scaled number (lakh / crore / million / thousand, percent) or other separators; PERSON / PLACE initials + surname, accent-folded form, hyphen / space variants
 P2 referring: own-event window sentence that is a stream neighbour (same article, previous or next fragment) of a fragment hidden by THIS question's rule (strict matcher or partial form) and starts with a pronoun / demonstrative (pronoun) or with 'The' + a core word of S or the answer type noun (defnp); plus all own-event pronoun-initial sentences
 P3 near value: DATE month-day 1 day off (and 2..7 days off), NUMBER within 1 or within 10 % (not equal), PERSON / PLACE token within edit distance 1-2 of a gold word (len >= 5, not equal), and short forms of the gold (partial forms of e57_orc2) that survive in OTHER-event sentences
 P4 indirect: own-event window sentence with an indirect temporal / relational cue (by year-end, by the end of, within N days, next / last week, later this, earlier this, N days ago, since, until, after, before, successor, replaced by, appointed, named), with or without a core word of S
 P5 time cues: own-event window sentences with weekday name / today / yesterday / tomorrow / this week / this month / last week / next week / tonight; number of own-event articles in the window; DATE: publish dates (bundle metadata, NOT stream text) of those articles and how many fall within 0 / 1 / 3 days of the answer date
Plus 'value sentences': window sentences that hold a value of the answer TYPE (e57_bank.type_strings; PERSON / PLACE: also a surface of a candidate of the question), split gold / distractor (a non-gold candidate of the question or any other value of the type) / other, own vs other event.
  python3 e57_w0v2_paths.py --bundle B --w0mask E57_w0mask.json --w0v2report E57_w0v2_report.json --out PREFIX    (writes PREFIX.json and PREFIX_private.json)"""
import sys,json,re,bisect,argparse,pathlib,collections,datetime,unicodedata
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
for p in (HERE,HERE.parent,ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;WIN=8192
MON=['january','february','march','april','may','june','july','august','september','october','november','december']
ABB={1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'}
DIM=[31,28,31,30,31,30,31,31,30,31,30,31]
ONES=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
TENS=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
ORD_ONES=['zeroth','first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth','eleventh','twelfth','thirteenth','fourteenth','fifteenth','sixteenth','seventeenth','eighteenth','nineteenth']
ORD_TENS=['','','twentieth','thirtieth']
WEEKDAY=r'(?:mon|tues?|wednes|thurs?|fri|satur|sun)day'
RX_TIMECUE=re.compile(r'\b(?:%s|today|tonight|yesterday|tomorrow|this (?:week|month|morning|evening|weekend)|last (?:week|month|night)|next (?:week|month)|earlier this (?:week|month)|a day (?:ago|later)|the day (?:before|after))\b'%WEEKDAY,re.I)
RX_INDIRECT=re.compile(r'\b(?:by (?:the )?(?:end of|year-end|month-end|early|mid|late)|year-end|within (?:\w+ )?(?:days|weeks|months)|next (?:week|month|year)|last (?:week|month|year)|later this|earlier this|\d+ (?:days|weeks|months) (?:ago|later|after|before|earlier)|since|until|successor|replaced by|replaces|replaced|appointed|named|will take over|took over|takes over|taken over|new (?:chairman|chief|head|mayor|president)|interim)\b',re.I)
RX_PRON=re.compile(r"^\W*(?:he|she|they|it|his|her|their|its|this|that|these|those|both|such|there|the same|the (?:two|couple|pair|duo|actor|actress|star|stars|bank|chairman|mayor|city|state|man|woman|team|player|players|film|movie|show|event|ceremony|wedding|announcement|decision|move|resignation|report|court|police|government|company))\b",re.I)
RX_THE=re.compile(r'^\W*the\b',re.I)
def fold(s):return ''.join(c for c in unicodedata.normalize('NFKD',s) if not unicodedata.combining(c))
def forms(q):
    vs=[]
    for v in [q['canonical']]+list(q.get('variants',[]))+[q.get('fixed_surface','')]:
        if v and v.lower() not in {x.lower() for x in vs}:vs.append(v)
    return vs
def lev(a,b,cap=3):
    if abs(len(a)-len(b))>cap:return cap+1
    pr=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cu=[i]
        for j,cb in enumerate(b,1):cu.append(min(pr[j]+1,cu[j-1]+1,pr[j-1]+(ca!=cb)))
        pr=cu
    return pr[-1]
def num_words(n):
    if n<20:return ONES[n]
    if n<100:return TENS[n//10]+('' if n%10==0 else '-'+ONES[n%10])
    if n<1000:return ONES[n//100]+' hundred'+('' if n%100==0 else ' '+num_words(n%100))
    return None
def ord_words(n):
    if n<20:return ORD_ONES[n]
    if n<=30 and n%10==0:return ORD_TENS[n//10]
    if n<40:return TENS[n//10]+'-'+ORD_ONES[n%10]
    return None
def date_gold(q):
    m=re.fullmatch(r'\s*([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*',q['canonical'].strip())
    if m and m.group(1).lower() in MON:return MON.index(m.group(1).lower())+1,int(m.group(2))
def month_rx():return '|'.join(sorted(MON+[ABB[i] for i in ABB]+['sept'],key=len,reverse=True))
RX_MD1=re.compile(r'(?<![\w])(%s)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?![\w%%:]|[.,]\d)'%month_rx(),re.I)
RX_MD2=re.compile(r'(?<![\w])(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(%s)\b'%month_rx(),re.I)
RX_NUMDATE=re.compile(r'(?<![\w/.-])(\d{1,4})([/.-])(\d{1,2})(?:\2(\d{1,4}))?(?![\w/-]|\.\d)')
RX_MDWORD=re.compile(r'(?<![\w])(%s)\.?\s+([a-z]+(?:-[a-z]+)?)\b'%month_rx(),re.I)
def mon_idx(s):
    s=s.lower().rstrip('.')
    if s=='sept':return 9
    for i,m in enumerate(MON,1):
        if s==m or s==ABB[i]:return i
def dates_in(t):
    """-> list of (kind, month, day) month-day mentions of the sentence; kind: text (month name + day), numeric (d/m or m/d: both readings returned), iso, word (month + spelled ordinal)"""
    out=[]
    for m in RX_MD1.finditer(t):
        mi=mon_idx(m.group(1));d=int(m.group(2))
        if mi and 1<=d<=31:out.append(('text',mi,d))
    for m in RX_MD2.finditer(t):
        mi=mon_idx(m.group(2));d=int(m.group(1))
        if mi and 1<=d<=31:out.append(('text',mi,d))
    for m in RX_NUMDATE.finditer(t):
        a,sep,b,c=m.groups()
        if len(a)==4 and 1<=int(b)<=12 and c and 1<=int(c)<=31:out.append(('iso',int(b),int(c)));continue
        if sep=='-' and not c:continue
        if sep=='.' and not c:continue
        a_,b_=int(a),int(b)
        if len(a)<=2:
            if 1<=a_<=12 and 1<=b_<=31:out.append(('numeric',a_,b_))
            if 1<=b_<=12 and 1<=a_<=31:out.append(('numeric',b_,a_))
    for m in RX_MDWORD.finditer(t):
        mi=mon_idx(m.group(1));w=m.group(2).lower()
        for d in range(1,32):
            if w==ord_words(d) or w==num_words(d):out.append(('word',mi,d))
    for m in re.finditer(r'(?<![\w])the\s+([a-z]+(?:-[a-z]+)?)\s+of\s+(%s)\b'%month_rx(),t,re.I):
        mi=mon_idx(m.group(2));w=m.group(1).lower()
        for d in range(1,32):
            if w==ord_words(d):out.append(('word',mi,d))
    return out
def doy(mi,d):
    d=min(d,DIM[mi-1]);return sum(DIM[:mi-1])+d
RX_NUM=re.compile(r'(?<![\w$])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s*(%|per\s?cent|percent|thousand|million|billion|lakh|crore|k|m|bn|cr))?(?![\w])',re.I)
SCALE={'thousand':1e3,'k':1e3,'million':1e6,'m':1e6,'billion':1e9,'bn':1e9,'lakh':1e5,'crore':1e7,'cr':1e7}
def numbers_in(t):
    """-> [(value, surface, kind)] kind: digit | scaled | pct | words"""
    out=[]
    for m in RX_NUM.finditer(t):
        raw=m.group(1).replace(',','');
        try:v=float(raw)
        except ValueError:continue
        u=(m.group(2) or '').lower().replace(' ','')
        if u in SCALE:out.append((v*SCALE[u],m.group(0),'scaled'))
        elif u:out.append((v,m.group(0),'pct'))
        else:out.append((v,m.group(0),'digit'))
    for n in list(range(0,200)):
        w=num_words(n)
        if w and re.search(r'(?<![\w-])'+re.escape(w)+r'(?![\w-])',t,re.I) and n>=2:out.append((float(n),w,'words'))
        o=ord_words(n) if n<=39 else None
        if o and n>=2 and re.search(r'(?<![\w-])'+re.escape(o)+r'(?![\w-])',t,re.I):out.append((float(n),o,'words'))
    return out
def gold_number(q):
    c=q['canonical'];m=re.search(r'\d[\d,]*(?:\.\d+)?',c)
    if not m:return None
    try:v=float(m.group(0).replace(',',''))
    except ValueError:return None
    u=re.sub(r'[\d,.\s]','',c[m.end()-1:]).lower() if False else ''
    return v
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--w0mask');ap.add_argument('--w0v2report');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--tokenizer');ap.add_argument('--ids',default=None,help='comma separated question ids (default: the 6 residual-gain questions + 6 controls)');a=ap.parse_args()
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_bank as B,e57_orc2 as O2
    from tokenizers import Tokenizer
    tf=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);tk=Tokenizer.from_file(tf)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items)
    prim=json.load(open(a.w0mask))['primary'];rep=json.load(open(a.w0v2report))['per_question']
    gain={q:v['OUT']['rank_ratio']-v['IN_DELORC_V2']['rank_ratio'] for q,v in rep.items()}
    RES=['hdfc-date-01','hannah-date-01','ncert-person-01','rash-person-02','mamdani-person-03','ranji-number-01']
    CTRL=['rash-person-01','ranji-person-02','ranji-person-01','ranji-place-02','mamdani-number-02','ncert-number-01']
    sel=a.ids.split(',') if a.ids else RES+CTRL;grp={q:('residual' if q in RES else 'control') for q in sel}
    bd=O2.build('v2',ids,tk,tf,items,iv,UVm,B);fr=bd['fr'];txt=bd['txt'];hidden=bd['hidden'];ev=bd['event'];art=bd['art']
    fst=np.array([f[1] for f in fr]);fln=np.array([f[2] for f in fr])
    E,idx=B.load(a.bundle);ev_dates={}
    for e_,Ev in E.items():
        for ai in range(len(Ev['rids'])):ev_dates[(e_,Ev['rids'][ai])]=datetime.date.fromordinal(int(Ev['dates'][ai]))
    art_dates=[ev_dates.get((iv[i][2],iv[i][3])) for i in range(len(iv))]
    def window_pieces(kk):
        e=(kk+1)*CH;hi=int(np.searchsorted(fst+fln,e,side='right'));tot=0;j=hi-1;pieces=[]
        while j>=0 and tot<WIN:
            if not hidden[j]:pieces.append(j);tot+=int(fln[j])
            j-=1
        return list(reversed(pieces)),tot
    byid={it['id']:it for it in items};res={};priv={};cache={}
    for qid in sel:
        it=byid[qid];q=it['q'];kk=prim[qid]
        if kk not in cache:cache[kk]=window_pieces(kk)
        pieces,tot=cache[kk];ty=q['type'];vs=forms(q);cs=B.core(q['S']);M,keys=B.ans_matcher(q);evq=it['event']
        import e57_orc2 as O2_
        prx=O2_.partial_rxs(q);gold_all=set(x.lower() for x in vs)
        cand_gold=it['cands'][it['gold_idx']] if 'gold_idx' in it else it['cands'][0]
        dist=[c for k_,c in zip(it['kinds'],it['cands']) if k_!='gold']
        # fragments hidden by THIS question (strict matcher / partial form inside its own event)
        own_j=lambda j:ev[j]==evq
        hid_q=np.array([bool(M(txt[j])) or (ev[j]==evq and any(r.search(txt[j]) for r in prx)) for j in range(len(fr))])
        pset=set(pieces);C=collections.defaultdict(lambda:collections.Counter());ex=collections.defaultdict(list)
        def add(cls,sub,j,own,note=''):
            C[cls][('own_' if own else 'other_')+sub]+=1
            if len(ex[cls])<3:ex[cls].append(dict(cls=cls,sub=sub,own=own,note=note,text=txt[j].strip()[:260]))
        g_dt=date_gold(q);g_num=gold_number(q);words=[x for x in re.findall(r"[A-Za-z][A-Za-z'’\-]+",q['canonical']) if len(x)>=3]
        gold_rx=re.compile('|'.join(r'(?<![\w])'+re.escape(v)+r'(?![\w])' for v in vs),re.I) if vs else None
        n_val=collections.Counter();n_with_ent=collections.Counter()
        for j in pieces:
            t=txt[j];own=own_j(j);low=t.lower();has_ent=bool(cs&B.words_of(t)) or q['S'].lower() in low
            # value sentences
            vals=[(tp,key,sf) for tp,key,sf,a_,b_ in B.type_strings(t) if tp==ty]
            gold_here=bool(M(t)) or bool(gold_rx and gold_rx.search(t));dist_here=False
            if ty in('PERSON','PLACE'):
                dist_here=any(re.search(r'(?<![\w])'+re.escape(c)+r'(?![\w])',t) for c in dist if c.lower() not in gold_all)
                gold_here=gold_here or any(B.related(sf,v) for _,_,sf in vals for v in vs)
                has_val=gold_here or dist_here or bool(vals)
            else:
                if ty=='DATE':
                    dd=dates_in(t);has_val=bool(vals) or bool(dd);gold_here=gold_here or (g_dt and any((m_,d_)==g_dt and k_=='text' for k_,m_,d_ in dd))
                    dist_here=any((m_,d_)!=g_dt for _,m_,d_ in dd)
                else:
                    nn=[v for v,_,k_ in numbers_in(t) if k_ in('digit','scaled','pct')];has_val=bool(vals) or bool(nn)
                    gold_here=gold_here or (g_num is not None and any(v==g_num for v in nn));dist_here=any(v!=g_num for v in nn) if g_num is not None else bool(nn)
            if has_val:
                lab='gold' if gold_here else ('distractor' if dist_here else 'other')
                n_val[('own_' if own else 'other_')+lab]+=1
                if has_ent:n_with_ent[('own_' if own else 'other_')+lab]+=1
                if lab=='gold' and len(ex['gold_value'])<3:ex['gold_value'].append(dict(cls='gold_value',sub=lab,own=own,note='',text=t.strip()[:260]))
            # P1 other format
            if ty=='DATE' and g_dt:
                for k_,m_,d_ in dates_in(t):
                    if (m_,d_)==g_dt and k_!='text':add('P1_other_format','date_'+k_,j,own);break
                else:
                    if re.search(r'(?<![\w])(?:the\s+)?%s(?![\w])'%re.escape(ord_words(g_dt[1]) or '_'),t,re.I):add('P1_other_format','date_ordinal_word',j,own)
                    elif re.search(r'(?<![\w$.,:/-])%d(?:st|nd|rd|th)(?![\w%%])'%g_dt[1],t) and not any((m_,d_)==g_dt for _,m_,d_ in dates_in(t)):add('P1_other_format','date_ordinal_digit_alone',j,own)
                    elif any((m_,d_)==g_dt and k_=='text' for k_,m_,d_ in dates_in(t)) and not (M(t) or (gold_rx and gold_rx.search(t))):add('P1_other_format','date_month_day_text_unmatched_by_strict',j,own)
            elif ty=='NUMBER' and g_num is not None:
                for v,sf,k_ in numbers_in(t):
                    if v==g_num and k_ in('words','scaled','pct') and not (gold_rx and gold_rx.search(t)):add('P1_other_format','number_'+k_,j,own,sf);break
                else:
                    if re.search(r'(?<![\w$.,])%s(?![\w]|\.\d)'%re.escape(f'{int(g_num):,}'),t) and not (gold_rx and gold_rx.search(t)) and g_num>=1000:add('P1_other_format','number_separator',j,own)
            else:
                nm=fold(t).lower();gw=[fold(x).lower() for x in words]
                if len(gw)>=2 and re.search(r'(?<![\w])%s\.?\s+%s(?![\w])'%(re.escape(gw[0][0]),re.escape(gw[-1])),nm) and not (gold_rx and gold_rx.search(t)):add('P1_other_format','initial_surname',j,own)
                elif fold(t)!=t and gw and any(re.search(r'(?<![\w])'+re.escape(w)+r'(?![\w])',nm) for w in gw) and not (gold_rx and gold_rx.search(t)):add('P1_other_format','accent_folded_form',j,own)
                elif gw and len(gw)>=2 and re.search(r'(?<![\w])%s[\s-]+%s(?![\w])'%(re.escape(gw[0]),re.escape(gw[-1])),nm) and not (gold_rx and gold_rx.search(t)):add('P1_other_format','hyphen_space_variant',j,own)
            # P3 near value
            if ty=='DATE' and g_dt:
                for k_,m_,d_ in dates_in(t):
                    df=abs(doy(m_,d_)-doy(*g_dt))
                    if df==1:add('P3_near_value','date_off_by_1_day',j,own);break
                    if 2<=df<=7:add('P3_near_value','date_off_2_to_7_days',j,own);break
            elif ty=='NUMBER' and g_num:
                for v,sf,k_ in numbers_in(t):
                    if k_=='words':continue
                    if v!=g_num and abs(v-g_num)<=1:add('P3_near_value','number_off_by_le_1',j,own,sf);break
                    if v!=g_num and abs(v-g_num)<=0.1*g_num:add('P3_near_value','number_within_10pct',j,own,sf);break
            else:
                toks=re.findall(r"[A-Z][\w'’\-]+",fold(t));hit=None
                for tok_ in toks:
                    for w in words:
                        w2=fold(w)
                        if len(w2)>=5 and tok_.lower()!=w2.lower() and lev(tok_.lower(),w2.lower(),2)<=2:hit=tok_;break
                    if hit:break
                if hit:add('P3_near_value','spelling_variant_edit_le_2',j,own,hit)
                if not own and any(r.search(t) for r in prx):add('P3_near_value','short_form_partial_in_other_event',j,own)
            if ty=='DATE' and not own and prx and any(r.search(t) for r in prx):
                C['P3_near_value']['other_partial_month_or_day_alone']+=0
            # P4 indirect, P5 time cues (own event only)
            if own:
                if RX_INDIRECT.search(t):add('P4_indirect','cue'+('_with_core_word' if has_ent else ''),j,own)
                if RX_TIMECUE.search(t):add('P5_time_cue','weekday_or_relative_time',j,own)
        # P2 referring: own-event neighbours of a fragment hidden by this question
        for j in pieces:
            if not own_j(j):continue
            for nb in (j-1,j+1):
                if 0<=nb<len(fr) and hid_q[nb] and art[nb]==art[j] and nb not in pset:
                    t=txt[j]
                    if RX_PRON.match(t) and not RX_THE.match(t):sub='pronoun_or_demonstrative'
                    elif RX_THE.match(t) and (cs&B.words_of(t) or RX_PRON.match(t)):sub='defnp'
                    else:sub='adjacent_other'
                    add('P2_referring_adjacent_to_hidden',sub,j,True);break
        n_pr=sum(1 for j in pieces if own_j(j) and RX_PRON.match(txt[j]) and not RX_THE.match(txt[j]));C['P2_referring_adjacent_to_hidden']['own_all_pronoun_initial_sentences']=n_pr
        # P5 articles
        arts=sorted({art[j] for j in pieces if own_j(j)});pubs=[art_dates[i] for i in arts if art_dates[i]]
        p5=dict(own_event_articles_in_window=len(arts),own_event_articles_with_publish_date=len(pubs),publish_date_range=[str(min(pubs)),str(max(pubs))] if pubs else None)
        if ty=='DATE' and g_dt and pubs:
            ys=[]
            for p in pubs:
                try:g=datetime.date(p.year,*g_dt)
                except ValueError:g=None
                if g:ys.append(min((abs((p-datetime.date(y,*g_dt)).days) for y in (p.year-1,p.year,p.year+1))))
            p5.update(articles_pub_within_0_days_of_gold=sum(1 for x in ys if x==0),articles_pub_within_1_days_of_gold=sum(1 for x in ys if x<=1),articles_pub_within_3_days_of_gold=sum(1 for x in ys if x<=3),articles_pub_within_7_days_of_gold=sum(1 for x in ys if x<=7))
        # time-cue sentences in articles published within 7 days of the gold date (cue + publish date => gold derivable)
        if ty=='DATE' and g_dt:
            near=0
            for j in pieces:
                if own_j(j) and RX_TIMECUE.search(txt[j]):
                    p=art_dates[art[j]]
                    if p and min(abs((p-datetime.date(y,*g_dt)).days) for y in (p.year-1,p.year,p.year+1))<=7:near+=1
            p5['time_cue_sentences_in_articles_published_within_7_days_of_gold']=near
        o=rep[qid]
        res[qid]=dict(group=grp[qid],type=ty,event=evq,primary_position=kk,gold=cand_gold,gain_rank_ratio_out_minus_delorc_v2=round(gain[qid],3),rank_ratio_out=round(o['OUT']['rank_ratio'],3),rank_ratio_delorc_v2=round(o['IN_DELORC_V2']['rank_ratio'],3),acc_delorc_v2=o['IN_DELORC_V2']['acc'],
            window_sentences=len(pieces),window_tokens=int(tot),window_sentences_own_event=sum(1 for j in pieces if own_j(j)),window_own_event_tokens=int(sum(fln[j] for j in pieces if own_j(j))),strict_gold_sentences_left=sum(1 for j in pieces if M(txt[j])),
            value_sentences_answer_type=dict(n_val),value_sentences_answer_type_with_key_entity=dict(n_with_ent),
            P1_other_format=dict(C['P1_other_format']),P2_referring=dict(C['P2_referring_adjacent_to_hidden']),P3_near_value=dict(C['P3_near_value']),P4_indirect=dict(C['P4_indirect']),P5_time_cue=dict(C['P5_time_cue'],**p5))
        out_ex=[];seen=set()
        for cls in ('P2_referring_adjacent_to_hidden','P1_other_format','P3_near_value','gold_value','P4_indirect','P5_time_cue'):
            for e_ in ex.get(cls,[])[:1]:
                if len(out_ex)<3 and e_['text'] not in seen:out_ex.append(e_);seen.add(e_['text'])
        for cls in ('P2_referring_adjacent_to_hidden','P1_other_format','P3_near_value','gold_value','P4_indirect','P5_time_cue'):
            for e_ in ex.get(cls,[])[1:]:
                if len(out_ex)<3 and e_['text'] not in seen:out_ex.append(e_);seen.add(e_['text'])
        priv[qid]=out_ex
    # group sums
    def tot(g,cls):
        c=collections.Counter()
        for q_,v in res.items():
            if v['group']==g:
                for k,n in v[cls].items():
                    if isinstance(n,int):c[k]+=n
        return dict(c)
    summ={g:{'questions':[q_ for q_,v in res.items() if v['group']==g],**{cls:tot(g,cls) for cls in ('P1_other_format','P2_referring','P3_near_value','P4_indirect','P5_time_cue','value_sentences_answer_type')}} for g in ('residual','control')}
    out=dict(note='ORC v2 deleted windows (job 279, IN_DELORC_V2) at the primary positions; classification of leftover answer paths, counts of window sentences; residual gain = rank ratio OUT - rank ratio IN_DELORC_V2; controls = questions with OUT rank ratio >= 0.2 and |gain| <= 0.04 (no DATE control exists with that property); examples are in the _private file',per_question=res,group_sums=summ)
    pathlib.Path(str(a.out)+'.json').write_text(json.dumps(out,indent=1,default=float,ensure_ascii=False));pathlib.Path(str(a.out)+'_private.json').write_text(json.dumps(dict(note='private, internal research only; stream sentences',examples=priv),indent=1,ensure_ascii=False))
    print(json.dumps(summ,indent=0,default=float)[:3000])
if __name__=='__main__':main()
