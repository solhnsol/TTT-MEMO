"""E57-lite PLACE clean-up (CPU only, NO model, NO download, reads ONLY the files of job 223: lite_attributes.json, lite_place.json, lite_bank.json, lite_events.json, lite_place_sample.json; seconds). Rules fixed BEFORE any result of this job (E57 section 6 note by the design thread):
 event level, in this order (first failing rule = reason):
  E0 no place string in the lead of the event                        -> 'no place'
  E1 c_SA == 0 (no S core token near any A sentence) or s == 0       -> 'c_SA = 0'
  E2 A is not a place. A = the top-1 'place' string of lite_attributes.json (lead articles, e57_lite.attributes()). A is a place only if (G) its lower-cased string is in the gazetteer below (countries, US states, Indian states / union territories, large cities) or (X) its last word ends in a place suffix (pur, abad, nagar, garh, ganj, ville, burg, burgh, shire, stan, polis, grad) AND it is not (N) ending in a non-place word (Day, Festival, Cup, Party, Act, Bill, Show, Awards, League, Series, University, College, Bank, Court, Council, Ministry, Police, Airlines, Montana ... see NONPLACE) AND not (P) a person: sharing a word with a person-class string of the same event's attribute table or being the 'place' class by accident of the in / at / near context. Reasons: 'not a place: person-like', 'not a place: non-place word', 'not a place: not in the gazetteer and no place suffix'. No NER and no online gazetteer is used: an unknown real place is dropped (listed in the report for a manual addition).
  E3 2nd place A2 = the highest-share string of the place class (top 5) that is itself a place by the SAME test (G or X, not N, not P) and is not the same place as A (one string contains the other: Delhi / New Delhi). Persons and non-places are skipped, so a person ranked 2nd (Vijay) does not count as the 2nd place. If s / s2 < 1.5 the event is not dropped but moved to the list 'hard' (hard-distractor indicator, NOT a main event); s2 = 0 (no A2) = main.
 item level: items = distinct S (subject_key of the frozen bank targets of the event, c >= 3); an item is dropped when a core word of S (>= 3 letters, not a stop word) is also a word of A ('S overlaps A': El Paso, Hannah Montana, Republic Day type); an event with no item left is dropped ('all items overlap A').
Outputs <out>/e57_lite/: place_clean.json (main events with A, A2, s, s2, c_SA and items; hard events; dropped events and items with reasons), place_clean_report.md (counts, reasons, per event table, review table of the 20 seeded extractions of job 223 with the verdicts of this job; the evidence sentences stay in lite_place_sample.json on the server, no article text here)."""
import argparse,json,re,sys,pathlib,hashlib
HERE=pathlib.Path(__file__).resolve().parent
RATIO=1.5
GAZ_TEXT="""afghanistan;albania;algeria;argentina;armenia;australia;austria;azerbaijan;bahrain;bangladesh;belarus;belgium;bolivia;bosnia;brazil;bulgaria;cambodia;cameroon;canada;chad;chile;china;colombia;congo;costa rica;croatia;cuba;cyprus;czech republic;czechia;denmark;ecuador;egypt;el salvador;estonia;ethiopia;fiji;finland;france;georgia;germany;ghana;greece;guatemala;haiti;honduras;hungary;iceland;india;indonesia;iran;iraq;ireland;israel;italy;jamaica;japan;jordan;kazakhstan;kenya;kosovo;kuwait;kyrgyzstan;laos;latvia;lebanon;libya;lithuania;luxembourg;madagascar;malawi;malaysia;mali;malta;mexico;moldova;mongolia;montenegro;morocco;mozambique;myanmar;namibia;nepal;netherlands;new zealand;nicaragua;niger;nigeria;north korea;norway;oman;pakistan;palestine;panama;paraguay;peru;philippines;poland;portugal;qatar;romania;russia;rwanda;saudi arabia;senegal;serbia;sierra leone;singapore;slovakia;slovenia;somalia;south africa;south korea;south sudan;spain;sri lanka;sudan;sweden;switzerland;syria;taiwan;tajikistan;tanzania;thailand;tunisia;turkey;turkmenistan;uganda;ukraine;united arab emirates;united kingdom;united states;uruguay;uzbekistan;venezuela;vietnam;yemen;zambia;zimbabwe;england;scotland;wales;northern ireland;gaza;west bank;crimea;tibet;hong kong;macau;puerto rico;greenland;
alabama;alaska;arizona;arkansas;california;colorado;connecticut;delaware;florida;hawaii;idaho;illinois;indiana;iowa;kansas;kentucky;louisiana;maine;maryland;massachusetts;michigan;minnesota;mississippi;missouri;montana;nebraska;nevada;new hampshire;new jersey;new mexico;new york;north carolina;north dakota;ohio;oklahoma;oregon;pennsylvania;rhode island;south carolina;south dakota;tennessee;texas;utah;vermont;virginia;washington;west virginia;wisconsin;wyoming;ontario;quebec;british columbia;alberta;manitoba;queensland;victoria;tasmania;
andhra pradesh;arunachal pradesh;assam;bihar;chhattisgarh;goa;gujarat;haryana;himachal pradesh;jharkhand;karnataka;kerala;madhya pradesh;maharashtra;manipur;meghalaya;mizoram;nagaland;odisha;punjab;rajasthan;sikkim;tamil nadu;telangana;tripura;uttar pradesh;uttarakhand;west bengal;jammu and kashmir;kashmir;ladakh;delhi;new delhi;chandigarh;puducherry;
mumbai;kolkata;chennai;bengaluru;bangalore;hyderabad;pune;ahmedabad;jaipur;lucknow;kanpur;nagpur;indore;bhopal;patna;surat;vadodara;noida;gurugram;gurgaon;ghaziabad;kochi;thiruvananthapuram;visakhapatnam;guwahati;srinagar;amritsar;ludhiana;varanasi;agra;mysuru;mysore;coimbatore;madurai;ranchi;raipur;dehradun;shimla;imphal;kohima;itanagar;shillong;aizawl;agartala;gangtok;bhubaneswar;dhaka;karachi;lahore;islamabad;kabul;kathmandu;colombo;
london;paris;berlin;madrid;rome;lisbon;amsterdam;brussels;vienna;zurich;geneva;stockholm;oslo;copenhagen;helsinki;warsaw;prague;budapest;bucharest;sofia;athens;istanbul;ankara;moscow;kyiv;kiev;kharkiv;odesa;mariupol;minsk;tbilisi;baku;yerevan;tehran;baghdad;damascus;beirut;amman;jerusalem;tel aviv;riyadh;jeddah;doha;dubai;abu dhabi;cairo;tripoli;tunis;algiers;casablanca;lagos;abuja;nairobi;kampala;addis ababa;johannesburg;cape town;pretoria;kinshasa;khartoum;mogadishu;beijing;shanghai;shenzhen;guangzhou;tokyo;osaka;seoul;pyongyang;taipei;manila;jakarta;bangkok;hanoi;kuala lumpur;sydney;melbourne;brisbane;perth;auckland;wellington;toronto;vancouver;montreal;ottawa;calgary;new york city;los angeles;chicago;houston;phoenix;philadelphia;san antonio;san diego;dallas;san francisco;seattle;boston;miami;atlanta;detroit;denver;las vegas;portland;minneapolis;baltimore;nashville;memphis;new orleans;orlando;tampa;charlotte;pittsburgh;cleveland;cincinnati;st louis;kansas city;salt lake city;austin;el paso;sacramento;honolulu;anchorage;mexico city;guadalajara;monterrey;havana;caracas;bogota;lima;santiago;buenos aires;sao paulo;rio de janeiro;brasilia;quito;montevideo;kingston;panama city;guatemala city;manchester;birmingham;liverpool;leeds;glasgow;edinburgh;cardiff;belfast;dublin;bristol;sheffield;barcelona;valencia;seville;milan;naples;turin;venice;florence;munich;hamburg;frankfurt;cologne;lyon;marseille;nice;rotterdam;antwerp"""
GAZ={x.strip() for x in GAZ_TEXT.replace('\n','').split(';') if x.strip()}
SUFFIX=('pur','abad','nagar','garh','ganj','ville','burg','burgh','shire','stan','polis','grad')
NONPLACE={'day','festival','cup','party','act','bill','show','awards','award','league','series','university','college','bank','court','council','ministry','police','airlines','montana','trophy','tournament','championship','election','elections','committee','commission','corporation','company','group','news','times','post','express','tv','fc','united','city','county','district','airport'}
STOP={'the','and','for','from','with','said','news','city','state','court','police','government','minister','president','bank','new','group','company','national','international','university','united'}
def words(s):return re.findall(r"[a-z]+",s.lower())
def core(s):return {w for w in re.findall(r"[a-z]{3,}",s.lower()) if w not in STOP}
def is_place(surface,person_words):
    """-> (ok, reason)"""
    k=' '.join(words(surface));w=words(surface)
    if not w:return False,'not a place: not in the gazetteer and no place suffix'
    if w[-1] in NONPLACE and k not in GAZ:return False,'not a place: non-place word'
    if set(w)&person_words and k not in GAZ:return False,'not a place: person-like'
    if k in GAZ:return True,'gazetteer'
    if w[-1].endswith(SUFFIX):return True,'suffix'
    return False,'not a place: not in the gazetteer and no place suffix'
def same_place(a,b):
    x,y=' '.join(words(a)),' '.join(words(b));return x==y or x in y or y in x
def clean(attr,place,bank,events):
    main=[];hard=[];dropped=[];dropped_items=[];by_ev={}
    for b in bank:by_ev.setdefault(b['event_name'],{}).setdefault(b['subject_key'],b['subject_surface'])
    for nm,o in place['events'].items():
        a=attr.get(nm,{});pl=a.get('place',[]);pers=a.get('person',[])
        pw={w for p in pers for w in re.findall(r"[a-z]{3,}",p['surface'].lower())}|set().union(*[core(s) for s in by_ev.get(nm,{}).keys()]) if by_ev.get(nm) or pers else set()
        if not pl:dropped.append(dict(event=nm,reason='no place'));continue
        A=pl[0];s=A['share']
        if s<=0 or not o.get('c_SA'):dropped.append(dict(event=nm,reason='c_SA = 0',A=A['surface'],s=s,c_SA=o.get('c_SA')));continue
        ok,why=is_place(A['surface'],pw)
        if not ok:dropped.append(dict(event=nm,reason=why,A=A['surface'],s=s));continue
        A2=None
        for c in pl[1:]:
            if same_place(c['surface'],A['surface']):continue
            if is_place(c['surface'],pw)[0]:A2=c;break
        s2=A2['share'] if A2 else 0.0;ratio=(s/s2) if s2>0 else float('inf')
        items=[];ca=core(A['surface'])
        for sk,sf in sorted(by_ev.get(nm,{}).items()):
            if core(sk)&ca or set(words(sf))&set(words(A['surface'])):dropped_items.append(dict(event=nm,S=sf,A=A['surface'],reason='S overlaps A'))
            else:items.append(dict(S_key=sk,S=sf))
        if not items:dropped.append(dict(event=nm,reason='all items overlap A',A=A['surface'],s=s));continue
        rec=dict(event=nm,A=A['surface'],A_key=A['key'],s=s,A2=A2['surface'] if A2 else None,s2=s2,ratio=None if ratio==float('inf') else ratio,c_SA=o['c_SA'],items=items,A_via=why,
                 other_places=[dict(surface=c['surface'],share=c['share']) for c in pl[1:] if is_place(c['surface'],pw)[0]])
        (main if ratio>=RATIO else hard).append(rec)
    return main,hard,dropped,dropped_items
def report(main,hard,dropped,dropped_items,sample,attr):
    n_i=lambda L:sum(len(r['items']) for r in L)
    L=['# E57-lite PLACE clean-up (CPU, rules fixed before the results; see the docstring of e57_placeclean.py)','',f"- main events: **{len(main)}** with **{n_i(main)}** items (event, S); hard events (s / s2 < {RATIO}, 2nd place after removing persons and non-places): {len(hard)} with {n_i(hard)} items; dropped events: {len(dropped)}; dropped items inside kept events: {len(dropped_items)}.",'']
    import collections
    c=collections.Counter(d['reason'] for d in dropped);L+=['## reasons for dropped events','','| reason | events |','|---|---|']+[f'| {k} | {v} |' for k,v in c.most_common()]+['']
    L+=['## dropped events','','| event | reason | A | s |','|---|---|---|---|']+[f"| {d['event']} | {d['reason']} | {d.get('A','')} | {('%.2f'%d['s']) if d.get('s') is not None else ''} |" for d in dropped]+['']
    L+=['## dropped items (S overlaps A)','','| event | S | A |','|---|---|---|']+[f"| {d['event']} | {d['S']} | {d['A']} |" for d in dropped_items]+['']
    for title,lst in(('main events',main),('hard events (s / s2 below %.1f: separate hard-distractor indicator, not main)'%RATIO,hard)):
        L+=[f'## {title}','','| event | A | s | 2nd place (valid place only) | s2 | s / s2 | c_SA | items | A accepted by |','|---|---|---|---|---|---|---|---|---|']
        L+=[f"| {r['event']} | {r['A']} | {r['s']:.2f} | {r['A2'] or '-'} | {r['s2']:.2f} | {'inf' if r['ratio'] is None else '%.2f'%r['ratio']} | {r['c_SA']:.2f} | {len(r['items'])} | {r['A_via']} |" for r in lst]+['']
    kept={r['event']:'main' for r in main};kept.update({r['event']:'hard' for r in hard});dr={d['event']:d['reason'] for d in dropped}
    L+=['## the 20 seeded extractions of job 223 with the verdict of this clean-up','','Evidence sentences: lite_place_sample.json on the server (publisher text).','','| # | event | S phrase near A | answer A | article | sentence | status of the event |','|---|---|---|---|---|---|---|']
    L+=[f"| {i+1} | {x['event']} | {x['S_phrase']} | {x['answer']} | {x['rid']} | {x['sent_idx']} | {kept.get(x['event']) or 'dropped: '+dr.get(x['event'],'?')} |" for i,x in enumerate(sample)]
    L+=['','## places the gazetteer did not know (dropped as unknown; add by hand to reuse them)','']+[f"- {d['event']}: {d['A']}" for d in dropped if d['reason'].endswith('no place suffix')]
    return '\n'.join(L)+'\n'
def run(fd):
    fd=pathlib.Path(fd);need=['lite_attributes.json','lite_place.json','lite_bank.json','lite_events.json']
    miss=[x for x in need if not (fd/x).exists()]
    if miss:print('missing',miss,'(job 223 first) (exit 7)');sys.exit(7)
    attr=json.load(open(fd/'lite_attributes.json'));place=json.load(open(fd/'lite_place.json'));bank=json.load(open(fd/'lite_bank.json'));ev=json.load(open(fd/'lite_events.json'))
    main,hard,dropped,di=clean(attr,place,bank,ev);txt=report(main,hard,dropped,di,place.get('sample20',[]),attr)
    out=dict(rules=dict(ratio=RATIO,gazetteer_size=len(GAZ),suffix=SUFFIX,nonplace=sorted(NONPLACE),source='files of job 223'),main=main,hard=hard,dropped=dropped,dropped_items=di)
    s=json.dumps(out,indent=1,default=str);(fd/'place_clean.json').write_text(s);(fd/'place_clean_report.md').write_text(txt);(fd/'place_clean.sha').write_text(hashlib.sha256(s.encode()).hexdigest()+'\n');print(txt[:6000]);return out
def _selftest():
    import tempfile
    td=pathlib.Path(tempfile.mkdtemp());mk=lambda nm,pl,per=(),c=0.5:(nm,dict(place=[dict(surface=s,key=s.lower(),share=h,articles=1) for s,h in pl],person=[dict(surface=s,key=s.lower(),share=h,articles=1) for s,h in per]),c)
    evs=[mk('Karnataka flood',[('Bengaluru',.6),('Mysuru',.2)]),mk('Pune event',[('Pune',.5),('Mumbai',.45)]),mk('Film Rashmika',[('Mumbai',.5),('Vijay Deverakonda',.4),('Hyderabad',.1)],[('Vijay Deverakonda',.4)]),mk('El Paso',[('El Paso',.7)]),mk('Zero',[('Pune',.5)],(),0.0),mk('Day',[('Republic Day',.6)]),mk('Unknown',[('Zzzland',.6)]),mk('None',[])]
    attr={e[0]:e[1] for e in evs};place=dict(events={e[0]:dict(c_SA=e[2]) for e in evs},sample20=[dict(event='Pune event',S_phrase='X',answer='Pune',rid=1,sent_idx=0,conflict_2nd_place_in_lead=True)])
    bank=[dict(event_name=e[0],subject_key=k,subject_surface=k.title()) for e in evs for k in(['a b'] if e[0]!='El Paso' else ['el paso electric','john smith'])]
    main,hard,dr,di=clean(attr,place,bank,None);nm={r['event'] for r in main};hd={r['event'] for r in hard};dn={d['event']:d['reason'] for d in dr}
    assert nm=={'Karnataka flood','Film Rashmika','El Paso'} or nm=={'Karnataka flood','Film Rashmika'}|{'El Paso'},(nm,hd,dn)
    assert hd=={'Pune event'} and dn['Zero']=='c_SA = 0' and dn['Day'].startswith('not a place: non-place') and dn['Unknown'].endswith('no place suffix') and dn['None']=='no place',(hd,dn)
    r=[x for x in main if x['event']=='Film Rashmika'][0];assert r['A2']=='Hyderabad' and abs(r['ratio']-5)<1e-6,r      # the person ranked 2nd is skipped
    assert any(d['S']=='El Paso Electric' for d in di) and [x for x in main if x['event']=='El Paso'][0]['items']==[dict(S_key='john smith',S='John Smith')]
    fd=td/'e57_lite';fd.mkdir()
    for f,o in(('lite_attributes.json',attr),('lite_place.json',place),('lite_bank.json',bank),('lite_events.json',dict(events=[]))):(fd/f).write_text(json.dumps(o))
    out=run(fd);assert (fd/'place_clean_report.md').exists() and out['main'] and 'hard events' in (fd/'place_clean_report.md').read_text();print('e57_placeclean selftest ok:',len(main),'main',len(hard),'hard',len(dr),'dropped')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--selftest',action='store_true');ap.add_argument('--rerun',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    fd=pathlib.Path(a.out)/'e57_lite'
    if (fd/'place_clean_report.md').exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo)');sys.exit(8)
    run(fd)
if __name__=='__main__':main()
