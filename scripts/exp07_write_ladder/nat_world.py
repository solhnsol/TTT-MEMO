"""Exp10 natural-text worlds: real Wikipedia bundles with counterfactual renaming (design: data/natural_text/DESIGN.md).

Drop-in counterpart of scripts/exp04_what_ttt_retains/gen_world.py for the natural setting:
  load_world(path)            -> world dict (same core schema as gen_world.build_world: triples, texts[expo|bg], pairs, ...)
  realize(world, swap)        -> (text, name, spans) exactly like gen_world.realize (swap=True = world B)
  probes(world, name)         -> L2/L3/L3r/L1p/L1l-style items like gen_world.probes (L3 = name -> birthplace)
  build_items(world)          -> families in ladder order L3, XP, PS, FN, FP, FN0, FP0, L3r, L2, then the natural-only
                                 families L3b (in-bundle foil) and L3e (school); every item's candidates are fixed across
                                 worlds A/B (canonical world-A names), as in exp09 world_text.build_items.
Placeholders in texts: @X{id}@ full name, @XF{id}@ given name, @XS{id}@ surname. A pair swaps all three together in B.
Pure Python, no tokenizer needed. The world JSON carries no real names (checked by checks.py)."""
import json,re

def load_world(path):return json.load(open(path))

def names(world,swap):
    tri=world['triples'];nm={t['id']:(t['X'],t['XF'],t['XS']) for t in tri}
    if swap:
        by={}
        for t in tri:by.setdefault(t['pair'],[]).append(t['id'])
        for a,b in by.values():nm[a],nm[b]=nm[b],nm[a]
    return nm

PH=re.compile(r'@X([FS]?)(\d+)@')
def realize(world,swap):
    nm=names(world,swap);k={'':0,'F':1,'S':2}
    out=[];pos=0;spans=[]
    for t in world['texts']:
        txt=PH.sub(lambda m:nm[int(m.group(2))][k[m.group(1)]],t['text'])
        if t['kind']=='expo':spans.append({'t':t['t'],'round':t['round'],'label':t.get('label'),'start':pos,'end':pos+len(txt),'text':txt})
        out.append(txt);pos+=len(txt)+2
    return '\n\n'.join(out),{i:v[0] for i,v in nm.items()},spans

def probes(world,name):
    F=world['frames'];P=[];by={}
    for t in world['triples']:by.setdefault(t['pair'],[]).append(t)
    for pair,(t1,t2) in by.items():
        for a,b in ((t1,t2),(t2,t1)):
            P.append({'level':'L3','pair':pair,'tid':a['id'],'prefix':F['birth'].format(X=name[a['id']]),'gold':' '+a['P'],'foil':' '+b['P']})
            P.append({'level':'L3r','pair':pair,'tid':a['id'],'prefix':F['rev'].format(P=a['P']),'gold':' '+name[a['id']],'foil':' '+name[b['id']]})
            if a['Q']:P.append({'level':'L3b','pair':pair,'tid':a['id'],'prefix':F['birth'].format(X=name[a['id']]),'gold':' '+a['P'],'foil':' '+a['Q']})
            if a['E'] and b['E']:P.append({'level':'L3e','pair':pair,'tid':a['id'],'prefix':F['school'].format(X=name[a['id']]),'gold':' '+a['E'],'foil':' '+b['E']})
    for i,t in enumerate(world['triples']):
        P.append({'level':'L1p','pair':t['pair'],'tid':t['id'],'prefix':F['freq_p'],'gold':' '+name[t['id']],'foil':' '+world['ctrl_persons'][i]})
        if world['ctrl_places'][i]:P.append({'level':'L1l','pair':t['pair'],'tid':t['id'],'prefix':F['freq_l'],'gold':' '+t['P'],'foil':' '+world['ctrl_places'][i]})
    return P

def build_items(world):
    """families with candidates fixed across A/B (canonical A names). L2 is empty in the natural setting (no
    occupation->name relation); it is kept so code that indexes the ladder family list still works."""
    _,name,_=realize(world,False);tri=world['triples'];F=world['frames'];allp=probes(world,name)
    fam={'L3':[it for it in allp if it['level']=='L3']}
    npairs=world['pairs'];xp=[]
    for p in range(npairs):
        q=(p+1)%npairs
        for j in (0,1):
            a=2*p+j
            xp.append({'level':'XP','pair':f'x{p}','tid':a,'prefix':F['birth'].format(X=name[a]),'gold':' '+tri[2*q+j]['P'],'foil':' '+tri[2*q+1-j]['P']})
    fam['XP']=xp
    pp,pl=world['pseudo_persons'],world['pseudo_places'];ps=[]
    for p in range(npairs):
        for x,y in ((2*p,2*p+1),(2*p+1,2*p)):
            if pl[x] and pl[y]:ps.append({'level':'PS','pair':f's{p}','tid':200+x,'prefix':F['birth'].format(X=pp[x]),'gold':' '+pl[x],'foil':' '+pl[y]})
    fam['PS']=ps
    fam['FN']=[it for it in allp if it['level']=='L1p'];fam['FP']=[it for it in allp if it['level']=='L1l']
    cp,cl=world['ctrl_persons'],world['ctrl_places']
    fam['FN0']=[{'level':'FN0','pair':f's{x//2}','tid':300+x,'prefix':F['freq_p'],'gold':' '+pp[x],'foil':' '+cp[x]} for x in range(len(tri))]
    fam['FP0']=[{'level':'FP0','pair':f's{x//2}','tid':400+x,'prefix':F['freq_l'],'gold':' '+pl[x],'foil':' '+cl[x]} for x in range(len(tri)) if pl[x] and cl[x]]
    fam['L3r']=[it for it in allp if it['level']=='L3r'];fam['L2']=[]
    fam['L3b']=[it for it in allp if it['level']=='L3b'];fam['L3e']=[it for it in allp if it['level']=='L3e']
    return fam

def forbid_strings(world):
    """every name and candidate string of the world; readout tails must not contain any of them."""
    s=set()
    for t in world['triples']:s|={t['X'],t['XS'],t['P']}|({t['Q']} if t['Q'] else set())|({t['E']} if t['E'] else set())
    s|=set(world['ctrl_persons'])|set(x for x in world['ctrl_places'] if x)|set(world['pseudo_persons'])|set(x for x in world['pseudo_places'] if x)
    return sorted(s)

def l1_rows_text(world,swap,variant='doc'):
    """L1 positive-control rows as texts: 'doc' = every exposure unit as its own row (main), 'fact' = only the
    sentences of fact units that contain both the name and the birthplace (secondary, the synthetic L1 analogue)."""
    _,_,spans=realize(world,swap);tri={t['id']:t for t in world['triples']};nm=names(world,swap)
    if variant=='doc':return ['\n\n'+s['text'] for s in spans]
    rows=[]
    for s in spans:
        if s['label']!='fact':continue
        for x in re.split(r'(?<=[.!?])\s+(?=[A-Z"“(])',s['text']):
            n=nm[s['t']]
            if tri[s['t']]['P'] in x and (n[0] in x or n[1] in x or n[2] in x):rows.append('\n\n'+x)
    return rows
