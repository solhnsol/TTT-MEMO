"""Automatic leakage / quality checks for the natural-text worlds (DESIGN.md section 9 + v2 R9). Exit code 1 on any failure.
python3 -I checks.py OUTDIR PERSONS_PKL [squad_dev.json]"""
import json,re,sys,glob,pickle,os
from collections import Counter
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import nat_world as nw
from common import LT,QT,tl
OUT,PP=sys.argv[1],sys.argv[2];SQ=sys.argv[3] if len(sys.argv)>3 else None
priv=json.load(open(f'{OUT}/private/real_name_map.json'));persons={p['title']:p for p in pickle.load(open(PP,'rb'))}
ALLFULL=sorted({persons[v['title']]['forms']['full'] for v in priv.values()})
fails=[];report={}
def fail(m):fails.append(m)
def wb(s,x):return re.search(r'(?<![\w])'+re.escape(x)+r'(?![\w])',s) is not None
squad=None
if SQ:
    d=json.load(open(SQ));squad='\n\n'.join(' '.join(p['context'].split()) for a in d['data'] for p in a['paragraphs'][:400])
ALLOWED={'seed','v','kind','layout','triples','ctrl_persons','ctrl_places','pseudo_persons','pseudo_places','texts','pairs','frames','stats'}
TRI_KEYS={'id','pair','X','XF','XS','P','Q','E','o','gender','natg','region'}
for f in sorted(glob.glob(f'{OUT}/nat_*.json')):
    w=nw.load_world(f);sid=w['seed'];tag=os.path.basename(f);raw=open(f).read()
    if set(w)-ALLOWED:fail(f'{tag}: unexpected top-level keys {set(w)-ALLOWED}')
    for t in w['triples']:
        if set(t)-TRI_KEYS:fail(f'{tag}: unexpected triple keys {set(t)-TRI_KEYS}')
    real=[persons[priv[f'{sid}:{t["id"]}']['title']] for t in w['triples']]
    A,nA,sA=nw.realize(w,False);B,nB,sB=nw.realize(w,True)
    for x in ALLFULL:
        if wb(raw,x):fail(f'{tag}: real full name of a target (any stream) {x!r} in json')
    # 1. real names: full names / aliases / surnames anywhere (JSON, A, B, squad tail); given & middle names inside own units
    for p,t in zip(real,w['triples']):
        fo=p['forms']
        for x in [fo['full'],fo['given']+' '+fo['surname'],fo['surname']]+fo['aliases']:
            if len(x)<4:continue
            for nmx,txt in (('json',raw),('A',A),('B',B)):
                if wb(txt,x):fail(f'{tag}: real name {x!r} found in {nmx}')
            if squad and wb(squad,x):fail(f'{tag}: real name {x!r} in SQuAD tail')
        own=' '.join(s['text'] for s in sA if s['t']==t['id'])
        for x in [fo['given']]+fo['middle']:
            if len(x)>=4 and wb(own,x) and x!=t['XF']:fail(f'{tag}: real given/middle {x!r} in own units of t{t["id"]}')
    # 2. fake names/candidates absent from the SQuAD tail
    if squad:   # world_text.squad_text drops paragraphs containing any forbid string; check enough tail remains
        paras=squad.split('\n\n');fb=nw.forbid_strings(w);kept=[x for x in paras if not any(f in x for f in fb)]
        for name,T in (('llama3',LT),('qwen',QT)):
            n=len(T.encode('\n\n'.join(kept),add_special_tokens=False).ids)
            report.setdefault(tag,{})[f'squad_tail_tokens_after_forbid_{name}']=n
            if n<7168:fail(f'{tag}: SQuAD tail too short after forbid filter ({name} {n})')
    # 3. B differs from A only at name slots
    nameset=sorted({x for t in w['triples'] for x in (t['XF'],t['XS'])},key=len,reverse=True)
    pat=re.compile(r'\b(?:'+'|'.join(re.escape(x) for x in nameset)+r')\b')
    if pat.sub('#',A)!=pat.sub('#',B):fail(f'{tag}: A and B differ outside name slots')
    # 4. birthplace values only inside their own person's units
    for t in w['triples']:
        for x in w['texts']:
            if wb(x['text'],t['P']) and not (x['kind']=='expo' and x['t']==t['id']):fail(f'{tag}: birthplace {t["P"]!r} outside own units (kind={x["kind"]}, t={x.get("t")})');break
    # 5. token lengths (llama3, qwen) matched within pairs and for foils
    by={}
    for t in w['triples']:by.setdefault(t['pair'],[]).append(t)
    for a,b in by.values():
        for k in ('X','XF','XS','P'):
            if tl(a[k])!=tl(b[k]):fail(f'{tag}: token length mismatch {k} {a[k]!r}/{b[k]!r}')
        if (a['E'] or b['E']) and tl(a['E'])!=tl(b['E']):fail(f'{tag}: school length mismatch')
    for t in w['triples']:
        if t['Q'] and tl(t['Q'])!=tl(t['P']):fail(f'{tag}: Q length mismatch')
    for i,t in enumerate(w['triples']):
        if tl(w['ctrl_persons'][i])!=tl(t['X']):fail(f'{tag}: ctrl person length mismatch t{i}')
        if w['ctrl_places'][i] and tl(w['ctrl_places'][i])!=tl(t['P']):fail(f'{tag}: ctrl place length mismatch t{i}')
    # 6. stream length and exposure position (both tokenizers)
    for name,T in (('llama3',LT),('qwen',QT)):
        e=T.encode(A,add_special_tokens=False)
        if len(e.ids)<63*1024:fail(f'{tag}: {name} stream too short {len(e.ids)}')
        last=max(s['end'] for s in sA);tpos=next(i for i,(s0,s1) in enumerate(e.offsets) if s1>=last)
        report.setdefault(tag,{})[f'{name}_tokens']=len(e.ids);report[tag][f'{name}_last_expo_chunk']=tpos//1024
        if tpos>=53*1024:fail(f'{tag}: {name} last exposure in chunk {tpos//1024} (>52)')
    # 7. probe frames absent from all texts
    for fr in [v for k,v in w['frames'].items() if k!='em']+w['frames']['em']:
        stem=fr.split('{')[0][:30]
        if len(stem)>12 and stem in A:fail(f'{tag}: frame stem {stem!r} in stream')
    fam=nw.build_items(w)
    report[tag].update({'fam_sizes':{k:len(v) for k,v in fam.items()},'stats':{k:w['stats'][k] for k in ('fact_repeats','fact_repeats_in_8k','in_window_ratio','expo_tokens_llama')},
        'fact_units_per_person':Counter(w['stats']['fact_units'].get(str(t['id']),w['stats']['fact_units'].get(t['id'],0)) for t in w['triples']),
        'natg':Counter(t['natg'] for t in w['triples']),'gender':Counter(t['gender'] for t in w['triples']),
        'same_region_pairs':sum(1 for a,b in by.values() if a['region'] and a['region']==b['region']),'region_unknown_pairs':sum(1 for a,b in by.values() if not(a['region'] and b['region']))})
json.dump({'fails':fails,'report':report},open(f'{OUT}/checks_report.json','w'),indent=1,default=str)
print('FAILS',len(fails));[print(' ',x) for x in fails[:40]]
sys.exit(1 if fails else 0)
