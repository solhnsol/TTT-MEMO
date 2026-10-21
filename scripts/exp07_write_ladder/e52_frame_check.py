"""E52 automatic check of paraphrased sentence frames (the paraphrases are written by a sub-agent; this script only filters).
Input JSON: list of {"id": str, "orig": frame template, "para": paraphrase template, optional "pair": pair id}. Templates use the slots {X} person, {P} place, {o} occupation, {a_o} article + occupation (frame_bank conventions).
Checks per entry: (1) slot multiset equal to the original and str.format works with the four slots only; (2) no new fact material: after the slots are filled with placeholders, no digit and no capitalised word except the first word of the sentence;
(3) filled length 9-16 words (frame_bank._words rule); (4) word 5-gram overlap with the original <= 0.30 (placeholders count as words) and para != orig; (5) para not identical to any other entry's para (duplicates);
(6) pair rule: all entries with the same "pair" and the same "orig" must have the same "para" (the two persons of a pair share the variant frame) and entries of different pairs must not share a para.
Exit 0 = all pass, 3 = some entry failed (list printed, with the failing check). Manual step stays: a random sample of 30 is read for grammar and meaning.
Usage: e52_frame_check.py FILE.json [--min-words 9 --max-words 16 --max-overlap 0.30]   |   --selftest"""
import sys,re,json,argparse,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[0]/'exp04_what_ttt_retains'))
SLOT=re.compile(r'\{(X|P|o|a_o)\}')
def words(t):return t.replace('{a_o}','an occupation').replace('{X}','X').replace('{P}','P').replace('{o}','o').split()
def ngr(t,n=5):
    w=[x.strip('.,;:!?"\'').lower() for x in re.sub(r'\{(\w+)\}',r'slot_\1',t).split()];return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
def check_one(e,lo=9,hi=16,mx=0.30):
    o,p=e['orig'],e['para'];errs=[]
    if collections.Counter(SLOT.findall(o))!=collections.Counter(SLOT.findall(p)):errs.append('slots differ')
    try:p.format(X='x',P='p',o='o',a_o='an o')
    except Exception as ex:errs.append(f'format error {ex!r}')
    body=SLOT.sub('slot',p);ws=body.split()
    if re.search(r'\d',body):errs.append('digit')
    caps=[w for i,w in enumerate(ws) if i>0 and w.strip('.,;:!?"\'')[:1].isupper() and w.strip('.,;:!?"\'')!='I']
    if caps:errs.append(f'capitalised word(s) {caps}')
    n=len(words(p))
    if not lo<=n<=hi:errs.append(f'length {n} outside {lo}-{hi}')
    if p.strip()==o.strip():errs.append('identical to the original')
    g,h=ngr(p),ngr(o)
    if g and len(g&h)/len(g)>mx:errs.append(f'5-gram overlap {len(g&h)/len(g):.2f} > {mx}')
    return errs
def check_all(es,lo=9,hi=16,mx=0.30):
    bad={};cnt=collections.Counter(e['para'] for e in es)
    for e in es:
        er=check_one(e,lo,hi,mx)
        if cnt[e['para']]>1 and not e.get('pair'):er.append('para used by several entries')
        if er:bad[e['id']]=er
    by=collections.defaultdict(set);byp=collections.defaultdict(set)
    for e in es:
        if e.get('pair') is not None:by[(e['pair'],e['orig'])].add(e['para']);byp[e['para']].add(e['pair'])
    for (pair,orig),ps in by.items():
        if len(ps)>1:
            for e in es:
                if e.get('pair')==pair and e['orig']==orig:bad.setdefault(e['id'],[]).append('pair members use different paraphrases of one frame')
    for para,prs in byp.items():
        if len(prs)>1:
            for e in es:
                if e['para']==para:bad.setdefault(e['id'],[]).append('same paraphrase used by several pairs')
    return bad
def selftest():
    o='{X}, the {o} of {P}, had never once left the valley.'
    ok={'id':'a','orig':o,'para':'Nobody in {P} ever saw {X}, the local {o}, travel beyond the hills.'}
    assert check_one(ok)==[],check_one(ok)
    bads={'slot':'Nobody in {P} ever saw the local {o} travel beyond the hills.','digit':'{X}, the {o} of {P}, had never left in 1850 or any other year.','cap':'{X}, the {o} of {P}, had never visited London in all those long years.',
          'short':'{X} the {o} of {P} stayed.','long':'{X}, the {o} of {P}, had never in all of his long and uneventful years left the valley or the hills or the river or the town.','same':o,'overlap':'{X}, the {o} of {P}, had never once left the valley, as everyone knew.','fmt':'{X} the {o} of {P} {Q} stayed there for the whole of his quiet life.'}
    for k,v in bads.items():
        er=check_one({'id':k,'orig':o,'para':v});assert er,(k,v)
    es=[dict(id='1',orig=o,para=ok['para'],pair='p0'),dict(id='2',orig=o,para=ok['para'],pair='p0'),dict(id='3',orig=o,para='In {P} nobody had ever known {X} the {o} to wander far from home.',pair='p1')]
    assert check_all(es)=={},check_all(es)
    es[1]=dict(id='2',orig=o,para='In {P} nobody had ever known {X} the {o} to wander far from home.',pair='p0');b=check_all(es);assert '1' in b and '2' in b,b
    es=[dict(id='1',orig=o,para=ok['para'],pair='p0'),dict(id='2',orig=o,para=ok['para'],pair='p1')];assert 'same paraphrase used by several pairs' in check_all(es)['1']
    print('e52_frame_check SELFTEST OK')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('file',nargs='?');ap.add_argument('--min-words',type=int,default=9);ap.add_argument('--max-words',type=int,default=16);ap.add_argument('--max-overlap',type=float,default=0.30);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    es=json.load(open(a.file));bad=check_all(es,a.min_words,a.max_words,a.max_overlap)
    for k,v in bad.items():print('FAIL',k,v)
    print(f'{len(es)} entries, {len(bad)} failed');sys.exit(3 if bad else 0)
if __name__=='__main__':main()
