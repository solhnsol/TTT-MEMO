"""E57 T1/T2 v2 FORWARD-pair server check (job-free CPU run of the forward draft t1t2_v2F_draft2.json; copy of e57_t1t2v2_check.py with the forward frames only and the forward draft fields; additionally the overlap of each forward frame with the frames of its paired frozen reverse item is reported (not part of the pass rule))
E57 T1/T2 v2 server check (CPU only, NO model, NO GPU, NO download; private internal research only; design E57 section 74-75, v1 procedure of e57_t1t2.py reused UNCHANGED for masking / overlap / person pool).
Input: t1t2_v2_draft.json (stream A items) and t1t2_v2B_draft.json (stream B20 items), the stream sentence files of A (e57_stream_sentences.jsonl) and B (streamB_stream_sentences.jsonl), the lead articles of A (e57_bundle store) and B (streamB_store.pkl).
(1) overlap: every frame (question, cloze, forward_cloze) vs every fragment of own stream, of the other stream, and vs its sibling frames; entity-masked 6-gram as in e57_t1t2 (k = 1, 2, 3; pass = no hit at k = 3 as in v1). Entity words = event name (B events: title of the drawn cluster) + gold variants.
(2) gold variants counted over the lead articles of both streams; fixed surface = most frequent variant (sentences) in the item's own stream; c' = articles holding any variant (event / own stream / both streams).
(3) easy (<= 20; same type; golds of other events first, then frequent strings of the whole stream; no string that occurs in an article of the item's own event; no alias / container of the gold; one per last name for persons) and hard (same event, same type, frequent strings of the event, the draft nominations checked in the event) candidates with c' per candidate.
(4) confound: candidates whose c' (both streams) is >= 2 / >= 3 times the gold c'.
  python3 e57_t1t2v2_check.py --out DIR"""
import sys,json,re,pickle,pathlib,collections,argparse,hashlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_bank as B
import e57_t1t2 as T
ROOT=HERE.parents[2];NEWS=ROOT/'experiments/exp07_write_ladder/e57_news'
EASY_N=20;HARD_N=8
def jh(o):return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def load_frags(p):
    F=[];meta=None
    for l in open(p):
        r=json.loads(l)
        if r['type']=='meta':meta=r
        else:F.append(r['text'])
    return meta,F
def streamB_events(sp):
    S=pickle.load(open(sp,'rb'));out={};titles={}
    for k,E in S['events'].items():
        ar=E['arts'];nm=f'B{k}'
        out[nm]=dict(rids=[a['rid'] for a in ar],hosts=[a['host'] for a in ar],dates=[a['date'] for a in ar],sents=[a['sents'] for a in ar],kept=[a['kept'] for a in ar],cum=[a['cum'] for a in ar]);titles[nm]=E['title1']
    return out,titles
class ArtIndex:
    """lower-cased lead text per article: count(surface) = articles holding the surface (word boundary, case-insensitive)"""
    def __init__(self,st):
        self.k=[];self.t=[];self.ev=[]
        for nm,E in st.items():
            for ai,sents,cum in B.lead_stream(E):self.k.append((nm,ai));self.t.append(' '.join(sents).lower());self.ev.append(nm)
        self.rx={}
    def holders(self,surfaces):
        arts=set()
        for s in surfaces:
            sl=s.lower();rx=self.rx.get(sl)
            if rx is None:rx=self.rx[sl]=re.compile(r'(?<![\w])'+re.escape(sl)+r'(?![\w])')
            for i,t in enumerate(self.t):
                if sl in t and rx.search(t):arts.add(i)
        return arts
    def count(self,surfaces,event=None,events=None):
        h=self.holders(surfaces)
        if event is not None:return sum(1 for i in h if self.ev[i]==event)
        if events is not None:return sum(1 for i in h if self.ev[i] in events)
        return len(h)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--draftA');ap.add_argument('--draftB');ap.add_argument('--fragA');ap.add_argument('--fragB');ap.add_argument('--storeB');ap.add_argument('--drawB');ap.add_argument('--reverse');a=ap.parse_args()
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    stA,_=B.load(NEWS);stB,titleB=streamB_events(a.storeB);st={**stA,**stB};evstream={**{k:'A' for k in stA},**{k:'B' for k in stB}}
    mA,FA=load_frags(a.fragA);mB,FB=load_frags(a.fragB);print('streams',mA['ids_sha'],len(FA),mB['ids_sha'],len(FB))
    S=FA+FB;lab=['A']*len(FA)+['B']*len(FB);ix=T.StreamIndex(S)
    DA=json.load(open(a.draftA));DB=json.load(open(a.draftB));items=[dict(i,stream='A') for i in DA['items']]+[dict(i,stream='B') for i in DB['items']]
    for i in items:i['event_name']=titleB.get(i['event'],i['event'])
    # ---- (1) overlap ----
    res=[];fails=collections.defaultdict(list)
    for it in items:
        ent=T.ent_words(it['event_name'],*([it['gold']]+it['variants']));frames=[(f['form'],f['frame']) for f in it['forms']]
        for form,fr in frames:
            o=ix.overlap(fr.replace('___',' '),ent);fid=f"{it['id']}#{form}"
            r=dict(id=fid,item=it['id'],stream=it['stream'],form=form,own={},cross={},examples={})
            for k in T.KS:
                own=[(g,i) for g,i in o[k] if lab[i]==it['stream']];cr=[(g,i) for g,i in o[k] if lab[i]!=it['stream']]
                r['own'][str(k)]=len(own);r['cross'][str(k)]=len(cr)
                if own or cr:r['examples'][str(k)]=[dict(gram=g,stream=lab[i],sentence=S[i][:160]) for g,i in (own+cr)[:3]]
                if own or cr:fails[str(k)].append(fid)
            r['siblings']={}
            for f2,fr2 in frames:
                if f2!=form:
                    p=T.pair_overlap(fr.replace('___',' '),fr2.replace('___',' '),ent)
                    if p[1]:r['siblings'][f2]={str(k):v for k,v in p.items()}
            res.append(r)
    rev={i['id']:i for i in json.load(open(a.reverse))['items']} if getattr(a,'reverse',None) else {}
    pairrep=[]
    for it in items:
        src=rev.get(it.get('source'))
        if not src:continue
        ent=T.ent_words(it['event_name'],*([it['gold']]+it['variants']+[src['gold']]+list(src['variants'])));ff=it['forms'][0]['frame']
        for rf in src['forms']:
            p=T.pair_overlap(ff.replace('___',' '),rf['frame'].replace('___',' '),ent)
            if p[1]:pairrep.append(dict(item=it['id'],reverse_form=rf['form'],overlap={str(k):v for k,v in p.items()}))
    sib=[r['id'] for r in res if any(v['3'] for v in r['siblings'].values())]
    ov=dict(definition='entity-masked 6-gram (e57_t1t2), ENT words = event name (B: cluster title) + gold variants; pass = no hit at k=3 in own stream, other stream and sibling frames',n_frames=len(res),hits={k:len(v) for k,v in fails.items()},fail_ids_k3=fails.get('3',[]),sibling_fail_k3=sib,pass_k3=not(fails.get('3') or sib),pair_overlap_with_paired_reverse=pairrep,items=res)
    (out/'e57_t1t2v2F_overlap.json').write_text(json.dumps(ov,indent=1,ensure_ascii=False));print('overlap frames',len(res),'hits',ov['hits'],'k3 fails',ov['fail_ids_k3'],'sib',sib)
    # ---- (2) golds ----
    AI=ArtIndex(st);evA=set(stA);evB=set(stB);golds=[]
    for it in items:
        own=evA if it['stream']=='A' else evB;per={}
        for v in it['variants']:
            h=AI.holders([v]);per[v]=dict(articles_own_stream=sum(1 for i in h if AI.ev[i] in own),articles_event=sum(1 for i in h if AI.ev[i]==it['event']),articles_both=len(h))
        mf=max(it['variants'],key=lambda v:(per[v]['articles_own_stream'],-it['variants'].index(v)));fixed=it['variants'][0]
        allh=AI.holders(it['variants'])
        golds.append(dict(id=it['id'],stream=it['stream'],event=it['event'],type=it['type'],gold=it['gold'],variants=it['variants'],per_variant=per,fixed_surface=fixed,most_frequent_variant=mf,c_event=sum(1 for i in allh if AI.ev[i]==it['event']),c_prime_own_stream=sum(1 for i in allh if AI.ev[i] in own),c_prime_both=len(allh),draft_evidence_c=it.get('cooccur_articles_c'),n_event_articles=it.get('n_event_articles'),in_stream=bool(allh)))
    (out/'e57_t1t2v2F_golds.json').write_text(json.dumps(golds,indent=1,ensure_ascii=False));G={g['id']:g for g in golds}
    print('golds not in stream:',[g['id'] for g in golds if not g['in_stream']],'fixed != gold:',[(g['id'],g['fixed_surface']) for g in golds if g['fixed_surface']!=g['gold']])
    # ---- (3) candidates ----
    pp=T.person_pool(st);pools=collections.defaultdict(lambda:collections.defaultdict(lambda:dict(surf=collections.Counter(),arts=set())))
    for nm,E in st.items():
        for ai,sents,cum in B.lead_stream(E):
            for t in sents:
                for ty,key,sf,x,y in B.type_strings(t):
                    if ty in('PLACE','OTHER') and len(sf)>=3 and key not in B.STOP:c=pools[ty][key];c['surf'][sf]+=1;c['arts'].add((nm,ai))
    def cand_event(it,gold_all,ty,n):
        """frequent same-type strings of the item's event"""
        ev=it['event'];out=[]
        if ty=='PERSON':
            src=[(k,c) for k,c in pp.items() if any(e==ev for e,_ in c['arts'])]
            for k,c in sorted(src,key=lambda kc:-sum(1 for e,_ in kc[1]['arts'] if e==ev)):
                sf=c['surf'].most_common(1)[0][0];ne=sum(1 for e,_ in c['arts'] if e==ev)
                if ne>=2 and not any(B.related(sf,v) for v in it['variants']):out.append((sf,ne))
        else:
            tyk='PLACE' if ty=='PLACE' else 'OTHER'
            for k,c in pools[tyk].items():
                sf=c['surf'].most_common(1)[0][0];ne=sum(1 for e,_ in c['arts'] if e==ev)
                if ne>=2 and not any(B.related(sf,v) for v in it['variants']) and not any(B.related(sf,w) for w in T.ent_words(it['event_name']) ):out.append((sf,ne))
            out.sort(key=lambda x:-x[1])
        return out[:n*3]
    def clean(lst,it,n,drop=()):
        res=[];keys=set()
        for sf,ne in lst:
            k=B.norm(sf)
            if k in keys or k in drop or any(B.related(sf,r_[0]) for r_ in res):continue
            keys.add(k);res.append((sf,ne))
        return res[:n]
    cand=[];allgold={i['id']:i for i in items}
    for it in items:
        ty=it['type'];vs=it['variants'];ev=it['event']
        hard_d=[(h['surface'] if isinstance(h,dict) else h) for h in it.get('hard_nominated',[])]
        hard_chk=[dict(surface=h,c_prime_event=AI.count([h],event=ev),c_prime_both=AI.count([h]),source='draft nomination') for h in hard_d]
        auto=clean(cand_event(it,None,ty,HARD_N),it,HARD_N,drop={B.norm(h) for h in hard_d})
        hard_auto=[dict(surface=s,c_prime_event=ne,c_prime_both=AI.count([s]),source='frequent same-type string of the event (auto, heuristic)') for s,ne in auto]
        hn={B.norm(h['surface']) for h in hard_chk+hard_auto}
        easy=[];hk=set(hn)
        def ok(sf):
            return not any(B.related(sf,v) for v in vs) and B.norm(sf) not in hk and AI.count([sf],event=ev)==0 and not any(B.related(sf,e['surface']) for e in easy)
        for o in items:                                                   # golds of other events of the same type
            if o['event']!=ev and o['type']==ty and ok(o['gold']):easy.append(dict(surface=o['gold'],source='gold of another event (same type)',from_event=o['event']))
            if len(easy)>=EASY_N:break
        if len(easy)<EASY_N:
            if ty=='PERSON':
                lst=sorted(pp.items(),key=lambda kc:-len(kc[1]['arts']));
                for k,c in lst:
                    if len(easy)>=EASY_N:break
                    if len(c['ctx'])<2:continue
                    sf=c['surf'].most_common(1)[0][0]
                    if k in {e['surface'].split()[-1].lower() for e in easy} or k in {B.norm(h['surface']).split()[-1] for h in hard_chk+hard_auto if B.norm(h['surface'])}:continue
                    if ok(sf):easy.append(dict(surface=sf,source='frequent person string of the whole stream (title / said-was context in >= 2 articles, one per last name)'))
            else:
                tyk='PLACE' if ty=='PLACE' else 'OTHER'
                for k,c in sorted(pools[tyk].items(),key=lambda kc:-len(kc[1]['arts'])):
                    if len(easy)>=EASY_N:break
                    sf=c['surf'].most_common(1)[0][0]
                    if len(c['arts'])>=3 and ok(sf) and B.norm(sf) not in B.STOP:easy.append(dict(surface=sf,source=f'frequent {tyk}-typed string of the whole stream (auto, heuristic; org / title / book not separated)'))
        for e in easy:e['c_prime_both']=AI.count([e['surface']])
        g=G[it['id']]
        conf=[dict(surface=c['surface'],c_prime=c['c_prime_both'],ratio=round(c['c_prime_both']/max(1,g['c_prime_both']),2),list=l) for l,L in(('easy',easy),('hard',hard_chk+hard_auto)) for c in L if c['c_prime_both']>=2*max(1,g['c_prime_both'])]
        cand.append(dict(id=it['id'],stream=it['stream'],event=ev,type=ty,gold=it['gold'],fixed_surface=g['fixed_surface'],c_event=g['c_event'],c_prime_both=g['c_prime_both'],hard_draft_checked=hard_chk,hard_auto=hard_auto,easy=easy,n_easy=len(easy),n_hard_draft=len(hard_chk),confounds=conf,confound_ge3=[c for c in conf if c['ratio']>=3]))
    (out/'e57_t1t2v2F_candidates.json').write_text(json.dumps(cand,indent=1,ensure_ascii=False))
    no_hard=[c['id'] for c in cand if not [h for h in c['hard_draft_checked'] if h['c_prime_event']>0]]
    summ=dict(created=__import__('datetime').datetime.now(__import__('datetime').timezone(__import__('datetime').timedelta(hours=9))).isoformat(timespec='seconds'),stream_A_ids_sha=mA['ids_sha'],stream_B_ids_sha=mB['ids_sha'],n_items=len(items),n_frames=len(res),overlap_pass_k3=ov['pass_k3'],overlap_hits=ov['hits'],overlap_fail_ids_k3=ov['fail_ids_k3'],sibling_fail_k3=sib,gold_not_in_stream=[g['id'] for g in golds if not g['in_stream']],fixed_differs_from_gold=[(g['id'],g['gold'],g['fixed_surface']) for g in golds if g['fixed_surface']!=g['gold']],fixed_not_most_frequent=[(g['id'],g['fixed_surface'],g['most_frequent_variant']) for g in golds if g['fixed_surface']!=g['most_frequent_variant']],easy_lt20=[(c['id'],c['n_easy']) for c in cand if c['n_easy']<EASY_N],no_valid_hard_in_event=no_hard,confound_items_ge3={c['id']:[(x['surface'],x['c_prime'],x['ratio'],x['list']) for x in c['confound_ge3']] for c in cand if c['confound_ge3']},c_event_lt4=[(g['id'],g['c_event']) for g in golds if g['c_event']<4])
    (out/'e57_t1t2v2F_summary.json').write_text(json.dumps(summ,indent=1,ensure_ascii=False));print(json.dumps(summ,ensure_ascii=False)[:6000])
if __name__=='__main__':main()
