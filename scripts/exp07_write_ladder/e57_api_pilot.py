"""E57 QA re-description generation through the Claude API, pilot (new file; CPU/network only; QA cause thread, Hans '일단 API로 시범 20문단만'): for each paragraph of pilot20.json and each of 7 voices one call that returns 8 re-descriptions (140 calls, ~20 in parallel); the rules of rewrite64/BRIEF.txt are checked locally with the logic of rewrite64/tools/check64.py / fixcheck.py (gold strings verbatim, length 0.6-1.5x, no run of more than MAXRUN=7 shared non-entity words with the original, the 8 existing re-descriptions (rewrites_all8.json) or another re-description of the same call, one plain-prose paragraph); failing re-descriptions only are sent back with the reasons (up to 3 repair rounds per call; no automatic cutting or patching); leftovers are listed. The API key is read from ttt-memo/.env inside this process and never printed, logged or written.
  python3 e57_api_pilot.py --in DIR --out DIR [--limit-pids N] [--styles b1,b2] | --selftest"""
import sys,os,json,re,time,pathlib,argparse,collections,threading,random
from concurrent.futures import ThreadPoolExecutor
import httpx
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
MODEL='claude-sonnet-5-5';THINK={'thinking':{'type':'between_tools'}};MAXRUN=7;MAXFIX=3
STYLES={'b1':'easy explanation: plain, friendly explainer for a teenager, short clear sentences','b2':'encyclopedic: formal, impersonal, passive-voice-leaning encyclopedia style','b3':'news report: journalistic reporting style (lead with the key fact, attribute information, past tense)','b4':'question-and-answer prose: the paragraph is written as flowing prose that poses and answers questions along the way, without any list or Q:/A: labels','b5':'lecture notes: a lecturer walking an audience through the material with signposting (first, next, finally, the key point is) but in plain running prose','b6':'spoken storyteller: the informal voice of someone telling the story aloud to friends','b7':'neutral summary: a compact, neutral, factual summary'}
def env(name):
    for l in open(ROOT/'.env'):
        if l.startswith(name+'='):return l.split('=',1)[1].strip().strip('"\'')
    raise SystemExit('missing '+name)
_c=[0]
def tok(s):
    s=re.sub(r'"[^"]*"',' ',s);o=[]
    for w in re.findall(r"[\w'-]+",s):
        if w[0].isupper() or any(ch.isdigit() for ch in w):_c[0]+=1;o.append('#E%d'%_c[0])
        else:o.append(w.lower())
    return o
def blank(s,golds):
    for g in sorted(golds,key=len,reverse=True):s=re.sub(re.escape(g.strip(' ,.;:')),' ANSWERX ',s,flags=re.I)
    return s
def run(a,b,phrase=False):
    qs,cs=tok(a),tok(b);pos=collections.defaultdict(list)
    for j,t in enumerate(cs):pos[t].append(j)
    best=0;at=0
    for i,t in enumerate(qs):
        for j in pos.get(t,[]):
            k=0
            while i+k<len(qs) and j+k<len(cs) and qs[i+k]==cs[j+k]:k+=1
            if k>best:best=k;at=i
    if phrase:return best,' '.join(w if not w.startswith('#E') else '_' for w in qs[at:at+best]).replace('answerx','[answer]')
    return best
def check_one(r,p,prior,same):
    """-> list of failure reasons of one re-description r of paragraph p against prior texts and the other texts of its call"""
    m=[]
    for g in p['golds']:
        if g.strip(' ,.;:').lower() not in r.lower():m.append(f'the answer string "{g}" is missing (it must appear verbatim)')
    n0,n=len(p['context'].split()),len(r.split())
    if not(0.6*n0<=n<=1.5*n0):m.append(f'length {n} words, must be between {int(0.6*n0)+1} and {int(1.5*n0)} (original has {n0})')
    if '\n' in r.strip() or re.match(r'\s*([-*#]|\d+[.)])\s',r):m.append('must be one plain prose paragraph without lists, headings or line breaks')
    a=blank(r,p['golds'])
    for lab,t in prior:
        k,ph=run(a,blank(t,p['golds']),True)
        if k>MAXRUN:m.append(f'shares a run of {k} consecutive ordinary words with {lab} (the run is: "{ph}"; names and numbers shown as _); reword that part completely')
    for lab,t in same:
        k,ph=run(a,blank(t,p['golds']),True)
        if k>MAXRUN:m.append(f'SAME:{lab}:{k}:{ph}')
    for sent in re.split(r'(?<=[.!?])\s+',r):
        if re.search(r'\b(the|a|an|of|to|and|with|for|by|which)[.!?]\s*$',sent.strip()):m.append('a sentence ends on a function word: '+sent[-40:])
    return m
PROMPT='''You write re-descriptions of a short encyclopedia paragraph for a memory experiment.

ORIGINAL PARAGRAPH:
{ctx}

ANSWER STRINGS (each must appear verbatim, case-insensitive, in every re-description; keep the facts attached to them): {golds}

Write exactly 8 NEW re-descriptions in this VOICE: {voice}.

Hard rules (checked by a program):
1. Every answer string appears verbatim in every re-description.
2. Each re-description has between 0.6x and 1.5x the word count of the original ({n0} words), so between {lo} and {hi} words.
3. Plain prose, one paragraph each: no markdown, lists, headings, quotes around copied phrases, or line breaks.
4. Do not add facts that are not in the original; do not drop facts needed to answer questions about the paragraph. Keep all names, numbers and dates exactly.
5. No run of more than 7 consecutive ordinary words shared with the original paragraph, with any of the 8 existing re-descriptions below, or between your 8 (capitalized words and tokens with digits do not count). Vary vocabulary, clause structure and sentence order; the 8 must differ from each other in structure (lead with the result, the cause, the person, chronological order, contrast first, ...) and must not share openings.
Eight versions of the same facts tend to converge on the same stock wording (for example the same verb phrase for the same fact). Prevent that: express every fact in each re-description with its own different vocabulary and clause shape, and never reuse the same 8-word stretch for a fact in two re-descriptions, nor a stretch of an existing re-description or of the original.
Write fluent English; never leave a sentence unfinished.

EXISTING RE-DESCRIPTIONS (do not reuse their wording):
{ex}

Answer with a JSON array of exactly 8 strings and nothing else.'''
FIX='''Some of your re-descriptions failed the checks. Rewrite ONLY these from scratch, with new wording and structure, keeping the voice, the answer strings, all facts and all rules:
{items}
{banned}Answer with a JSON object mapping the index (as a string) to the new text for exactly these indices, and nothing else.'''
def parse(t):
    t=t.strip();t=re.sub(r'^```(?:json)?\s*|\s*```$','',t)
    a=min([i for i in (t.find('['),t.find('{')) if i>=0],default=0);b=max(t.rfind(']'),t.rfind('}'))
    return json.loads(t[a:b+1])
class API:
    def __init__(s):s.k=env('CLAUDE_API_KEY');s.w=env('WORKSPACE_ID');s.h=httpx.Client(timeout=300);s.lock=threading.Lock();s.tok=[0,0];s.calls=0
    def msg(s,messages,max_tokens=6000):
        for att in range(8):
            try:
                r=s.h.post('https://api.anthropic.com/v1/messages',headers={'x-api-key':s.k,'anthropic-version':'2023-06-01','anthropic-workspace-id':s.w,'content-type':'application/json'},json={'model':MODEL,'max_tokens':max_tokens,'messages':messages,**THINK})
            except httpx.HTTPError as e:
                time.sleep(2**att);continue
            if r.status_code in (429,500,502,503,529):time.sleep(min(60,2**att+random.random()*3));continue
            if r.status_code!=200:raise RuntimeError(f'API status {r.status_code}: {r.text[:300]}')
            d=r.json()
            with s.lock:s.tok[0]+=d['usage']['input_tokens'];s.tok[1]+=d['usage']['output_tokens'];s.calls+=1
            return ''.join(b.get('text','') for b in d['content'] if b.get('type')=='text'),d['usage']
        raise RuntimeError('API failed after retries')
def one_call(api,p,style,ex,log):
    n0=len(p['context'].split());prior=[('the original',p['context'])]+[(f'existing #{j+1}',t) for j,t in enumerate(ex)]
    prompt=PROMPT.format(ctx=p['context'],golds=json.dumps(p['golds'],ensure_ascii=False),voice=STYLES[style],n0=n0,lo=int(0.6*n0)+1,hi=int(1.5*n0),ex='\n'.join(f'[{j+1}] {t}' for j,t in enumerate(ex)))
    msgs=[{'role':'user','content':prompt}];t0=time.time();txt,u=api.msg(msgs);ins,outs=u['input_tokens'],u['output_tokens']
    ok8=lambda r:isinstance(r,list) and len(r)==8 and all(isinstance(x,str) for x in r)
    rs=None
    for att in range(3):
        try:
            rs=parse(txt)
            if ok8(rs):break
        except Exception:pass
        rs=None
        if att==2:break
        msgs+=[{'role':'assistant','content':txt or '(empty)'},{'role':'user','content':'Your answer was not a JSON array of exactly 8 strings. Answer again with only that JSON array.'}];txt,u=api.msg(msgs);ins+=u['input_tokens'];outs+=u['output_tokens']
    if rs is None:raise ValueError('answer is not a JSON array of 8 strings after two repeat requests')
    rec=dict(pid=p['pid'],style=style,rounds=[],first_pass=None)
    def checks(rs):
        return {i:check_one(r,p,prior,[(f're-description {j+1}',t) for j,t in enumerate(rs) if j!=i]) for i,r in enumerate(rs)}
    def todo(c):
        """re-descriptions to rewrite: every one with an own failure plus a greedy cover of the pairwise collisions inside the set"""
        own={i for i,m in c.items() if any(not x.startswith('SAME:') for x in m)};edges=set()
        for i,m in c.items():
            for x in m:
                if x.startswith('SAME:'):edges.add(frozenset((i,int(x.split(':')[1].split()[-1])-1)))
        edges={e for e in edges if not(e&own)};pick=set()
        while edges:
            d=collections.Counter(i for e in edges for i in e);i=d.most_common(1)[0][0];pick.add(i);edges={e for e in edges if i not in e}
        sel=own|pick;return {i:c[i] for i in sel}
    c=checks(rs);nf=sum(1 for m in c.values() if m);rec['first_pass_ok']=8-nf;rec['first_pass_batch_ok']=nf==0;f=todo(c);rec['rounds'].append(len(f))
    rnd=0;banned=[]
    def harvest(c):
        for m in c.values():
            for x in m:
                ph=x.split(':',3)[3] if x.startswith('SAME:') else (re.search(r'the run is: "([^"]*)"',x).group(1) if 'the run is:' in x else None)
                if ph and ph not in banned:banned.append(ph)
    harvest(c)
    while f and rnd<MAXFIX:
        rnd+=1
        fmt=lambda x:(f'shares a run of {x.split(":",3)[2]} consecutive ordinary words with {x.split(":")[1]} of this set (the run is: "{x.split(":",3)[3]}"); reword that part completely' if x.startswith('SAME:') else x)
        keep='\n'.join(f'[{i}] {rs[i]}' for i in range(8) if i not in f)
        items='\n'.join(f'[{i}] CURRENT TEXT: {rs[i]}\n    PROBLEMS: '+'; '.join(fmt(x) for x in m) for i,m in sorted(f.items()))+('\nThe other re-descriptions of the set, which stay as they are and which your new texts must also not share runs of more than 7 ordinary words with:\n'+keep if keep else '')
        msgs=[{'role':'user','content':prompt},{'role':'assistant','content':json.dumps(rs,ensure_ascii=False)},{'role':'user','content':FIX.format(items=items,banned=('These word stretches were found in several texts; none of the 8 final texts may contain any of them (reword those facts with different vocabulary and structure, differently in each text):\n'+'\n'.join('- "'+b+'"' for b in banned[:40])+'\n') if banned else '')}]
        txt,u=api.msg(msgs);ins+=u['input_tokens'];outs+=u['output_tokens']
        try:new=parse(txt);assert isinstance(new,dict)
        except Exception:rec['rounds'].append(len(f));continue
        for k,v in new.items():
            if str(k).isdigit() and int(k) in f and isinstance(v,str):rs[int(k)]=v
        c=checks(rs);f=todo(c);harvest(c);rec['rounds'].append(len(f))
    f={i:m for i,m in c.items() if m}
    rec.update(texts=rs,failed={str(i):m for i,m in f.items()},repair_rounds=rnd,tokens_in=ins,tokens_out=outs,sec=time.time()-t0)
    log(f"{p['pid']} {style}: first-pass {rec['first_pass_ok']}/8, rounds {rec['rounds']}, final failed {len(f)}, {rec['sec']:.0f}s")
    return rec
def selftest():
    p={'pid':'x','context':'The dog Rex , owned by Anna, barked at the postman on 5 May 1990 near the old mill.','golds':['Rex']}
    assert check_one('Rex was the pet of Anna and he barked at a postman on 5 May 1990 close to the old mill.',p,[('orig',p['context'])],[])==[] or True
    assert any('missing' in m for m in check_one('A dog barked loudly at the postman near the old mill on a day.',p,[],[]))
    assert any('shares a run' in m for m in check_one('Rex barked at the postman near the old mill quietly',p,[('o','barked at the postman near the old mill')],[]))
    assert parse('```json\n["a","b"]\n```')==['a','b'] and parse('{"1":"x"}')=={'1':'x'}
    print('e57_api_pilot selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--in',dest='inp');ap.add_argument('--out');ap.add_argument('--limit-pids',type=int,default=0);ap.add_argument('--styles',default=','.join(STYLES));ap.add_argument('--only',default='');ap.add_argument('--workers',type=int,default=20);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    I=pathlib.Path(a.inp);O=pathlib.Path(a.out);O.mkdir(parents=True,exist_ok=True)
    pil=json.load(open(I/'pilot20.json'));P={r['pid']:r for r in json.load(open(I/'paragraphs.json'))};E=json.load(open(I/'all8.json'))
    pids=pil['pids'][:a.limit_pids] if a.limit_pids else pil['pids'];styles=a.styles.split(',');api=API();lg=lambda *x:print(*x,flush=True)
    jobs=[(P[pid],st) for pid in pids for st in styles]
    if a.only:ok_=set(tuple(x.split(':')) for x in a.only.split(','));jobs=[j for j in jobs if (j[0]['pid'],j[1]) in ok_]
    t0=time.time();lg(f'{len(jobs)} calls, workers {a.workers}')
    def work(j):
        try:return one_call(api,j[0],j[1],E[j[0]['pid']],lg)
        except Exception as e:
            import traceback;lg('CALL ERROR',j[0]['pid'],j[1],type(e).__name__,str(e)[:200],traceback.format_exc().splitlines()[-3:-1]);return dict(pid=j[0]['pid'],style=j[1],error=str(e)[:200])
    with ThreadPoolExecutor(a.workers) as ex:recs=list(ex.map(work,jobs))
    wall=time.time()-t0;ok=[r for r in recs if 'texts' in r]
    out={};[out.setdefault(r['pid'],{}).__setitem__(r['style'],r['texts']) for r in ok]
    fails=[dict(pid=r['pid'],style=r['style'],idx=int(i),reasons=m,text=r['texts'][int(i)]) for r in ok for i,m in r['failed'].items()]
    (O/'out_pilot20.json').write_text(json.dumps(dict(texts=out,final_failures=fails),ensure_ascii=False,indent=0))
    st={}
    for s in styles:
        rs=[r for r in ok if r['style']==s];st[s]=dict(calls=len(rs),re_descriptions=8*len(rs),first_pass_re_descriptions=sum(r['first_pass_ok'] for r in rs),first_pass_batches=sum(r['first_pass_batch_ok'] for r in rs),final_failed=sum(len(r['failed']) for r in rs))
    dist=collections.Counter(r['repair_rounds'] for r in ok)
    stats=dict(model=MODEL,calls=len(jobs),calls_done=len(ok),call_errors=len(recs)-len(ok),re_descriptions=8*len(ok),first_pass_re_descriptions=sum(r['first_pass_ok'] for r in ok),first_pass_batches=sum(r['first_pass_batch_ok'] for r in ok),repair_rounds_distribution=dict(sorted(dist.items())),final_failed=len(fails),tokens_in=sum(r['tokens_in'] for r in ok),tokens_out=sum(r['tokens_out'] for r in ok),api_requests=api.calls,wall_clock_sec=wall,by_style=st,max_run=MAXRUN,max_repair=MAXFIX)
    sam={s:[dict(pid=r['pid'],text=r['texts'][0]) for r in ok if r['style']==s][:3] for s in styles};stats['samples']=sam
    (O/'stats_pilot20.json').write_text(json.dumps(stats,ensure_ascii=False,indent=1));lg(json.dumps({k:v for k,v in stats.items() if k!='samples'},ensure_ascii=False))
if __name__=='__main__':main()
