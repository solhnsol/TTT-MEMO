"""E57 QA re-descriptions through the Claude API, 8 formats x 8 instructions grid (new file; network only; QA cause thread, Hans '8개의 형식 x 8개의 지시, 개별 호출'): for one paragraph 64 independent calls (one re-description per call: a FORMAT voice and a structural INSTRUCTION), checked locally with the logic of e57_api_pilot.py (gold strings, length 0.6-1.5x, one plain paragraph, no run of more than 7 shared ordinary words with the original, the 8 existing re-descriptions or any other of the 64); failing ones (own failures plus a greedy cover of the pairwise collisions) are sent back with the concrete colliding stretches, up to 3 repair rounds, no automatic cutting. The key is read from ttt-memo/.env in this process and never printed.
  python3 e57_api_grid.py --in DIR --out DIR --pid PID --model claude-sonnet-5-5 | --selftest"""
import sys,os,json,re,time,pathlib,argparse,collections,threading,random
from concurrent.futures import ThreadPoolExecutor
import httpx
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_api_pilot as A
FORMATS=['novel prose: literary narrative prose as in a novel, with a narrator','encyclopedia wiki article style: neutral, impersonal, passive-leaning','screenplay style written as flowing prose describing scenes and spoken lines, no script formatting or character labels','news report: journalistic reporting style, attribution, past tense','spoken storyteller: the informal voice of someone telling it aloud to friends','lecture notes: a lecturer walking an audience through the material in plain running prose','question-and-answer prose: poses and answers questions along the way inside flowing prose, no Q:/A: labels','neutral compact summary']
INSTR=['begin with the result or outcome and then explain how it came about','begin with the cause or background and move toward the consequences','begin with the main person, group or subject and organize everything around them','tell it in strict chronological order','begin with a contrast, surprise or complication and then resolve it','begin with the key numbers and dates and build the explanation around them','begin with a short scene-setting or context sentence, then give the facts','begin with the last fact of the original and work backwards to the first']
PROMPT='''You write one re-description of a short encyclopedia paragraph for a memory experiment.

ORIGINAL PARAGRAPH:
{ctx}

ANSWER STRINGS (each must appear verbatim, case-insensitive; keep the facts attached to them): {golds}

FORMAT / VOICE: {fmt}.
STRUCTURE: {ins}.

Rules (checked by a program): every answer string appears verbatim; between {lo} and {hi} words ({n0} words in the original, 0.6x-1.5x); one plain prose paragraph, no markdown, lists, headings, quotes around copied phrases or line breaks; no new facts and no needed facts dropped, keep all names, numbers and dates exactly; do not reuse any stretch of more than 7 consecutive ordinary words from the original paragraph or from the existing re-descriptions below - use different vocabulary and clause structure; write fluent English and finish every sentence.

EXISTING RE-DESCRIPTIONS (do not reuse their wording):
{ex}

Answer with the re-description text only.'''
FIX='''That text failed the checks:
{probs}
Rewrite it from scratch with new wording and structure, keeping the format, the structure instruction, the answer strings and all facts. Do not reuse these stretches: {ban}
Answer with the re-description text only.'''
class API2(A.API):
    def __init__(s,model):
        super().__init__();s.model=model;s.think=True
    def msg(s,messages,max_tokens=2500):
        for att in range(8):
            body={'model':s.model,'max_tokens':max_tokens,'messages':messages}
            if s.think:body.update(A.THINK)
            try:r=s.h.post('https://api.anthropic.com/v1/messages',headers={'x-api-key':s.k,'anthropic-version':'2023-06-01','anthropic-workspace-id':s.w,'content-type':'application/json'},json=body)
            except httpx.HTTPError:time.sleep(2**att);continue
            if r.status_code in (429,500,502,503,529):time.sleep(min(60,2**att+random.random()*3));continue
            if r.status_code==400 and 'thinking' in r.text:s.think=False;continue
            if r.status_code!=200:raise RuntimeError(f'API status {r.status_code}: {r.text[:300]}')
            d=r.json()
            with s.lock:s.tok[0]+=d['usage']['input_tokens'];s.tok[1]+=d['usage']['output_tokens'];s.calls+=1
            return ''.join(b.get('text','') for b in d['content'] if b.get('type')=='text').strip(),d['usage']
        raise RuntimeError('API failed after retries')
def _tk(s):return re.findall(r"[a-z0-9']+",s.lower())
def _ng(t,n):return set(zip(*[t[i:] for i in range(n)]))
def _jac(a,b):return len(a&b)/max(1,len(a|b))
J2=0.35;J4=0.15
def jcheck(k,texts,p,prior):
    """similarity check of text k against the original, the existing re-descriptions and the other texts: Jaccard of word 2-grams > J2 or of 4-grams > J4 (answer strings blanked); -> list of (label, bigram, fourgram) and the own-rule failures"""
    m=[]
    for g in [p['golds']]:pass
    own=A.check_one(texts[k],p,[],[])      # gold / length / plain paragraph / sentence end only (no run rule)
    a2=_ng(_tk(A.blank(texts[k],p['golds'])),2);a4=_ng(_tk(A.blank(texts[k],p['golds'])),4);sims=[]
    for lab,t in prior+[(f're-description {j+1}',texts[j]) for j in range(len(texts)) if j!=k and texts[j]]:
        b=A.blank(t,p['golds']);x2=_jac(a2,_ng(_tk(b),2));x4=_jac(a4,_ng(_tk(b),4))
        if x2>J2 or x4>J4:sims.append((lab,x2,x4))
    return own,sims
def todo(c):
    own={i for i,m in c.items() if any(not x.startswith('SAME:') for x in m)};edges=set()
    for i,m in c.items():
        for x in m:
            if x.startswith('SAME:'):edges.add(frozenset((i,int(x.split(':')[1].split()[-1])-1)))
    edges={e for e in edges if not(e&own)};pick=set()
    while edges:
        d=collections.Counter(i for e in edges for i in e);i=d.most_common(1)[0][0];pick.add(i);edges={e for e in edges if i not in e}
    return own|pick
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--in',dest='inp');ap.add_argument('--out');ap.add_argument('--pid');ap.add_argument('--model');ap.add_argument('--workers',type=int,default=20);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:assert len(FORMATS)==8 and len(INSTR)==8 and todo({0:['SAME:re-description 2:9:x'],1:['SAME:re-description 1:9:x']}) in ({0},{1});print('e57_api_grid selftest ok');return
    I=pathlib.Path(a.inp);O=pathlib.Path(a.out);O.mkdir(parents=True,exist_ok=True)
    P={r['pid']:r for r in json.load(open(I/'paragraphs.json'))};E=json.load(open(I/'all8.json'));p=P[a.pid];ex=E[a.pid];n0=len(p['context'].split())
    api=API2(a.model);lg=lambda *x:print(*x,flush=True);t0=time.time();prior=[('the original',p['context'])]+[(f'existing #{j+1}',t) for j,t in enumerate(ex)]
    cells=[(f,g) for f in range(8) for g in range(8)];texts=[None]*64;hist=[None]*64
    base=lambda f,g:PROMPT.format(ctx=p['context'],golds=json.dumps(p['golds'],ensure_ascii=False),fmt=FORMATS[f],ins=INSTR[g],n0=n0,lo=int(0.6*n0)+1,hi=int(1.5*n0),ex='\n'.join(f'[{j+1}] {t}' for j,t in enumerate(ex)))
    def gen(k):
        f,g=cells[k];m=[{'role':'user','content':base(f,g)}];t,u=api.msg(m);hist[k]=m+[{'role':'assistant','content':t}];texts[k]=t
    with ThreadPoolExecutor(a.workers) as ex_:list(ex_.map(gen,range(64)))
    def checks():
        return {k:A.check_one(texts[k],p,prior,[(f're-description {j+1}',texts[j]) for j in range(64) if j!=k]) for k in range(64)}
    c=checks();first=sum(1 for m in c.values() if not m);rounds=[len(todo(c))];ban=collections.defaultdict(list)
    def fixone(k):
        probs='\n'.join('- '+(f"shares a run of {x.split(':',3)[2]} ordinary words with re-description {x.split(':')[1].split()[-1]} of this set" if x.startswith('SAME:') else x) for x in c[k])
        for x in c[k]:
            ph=x.split(':',3)[3] if x.startswith('SAME:') else (re.search(r'the run is: "([^"]*)"',x).group(1) if 'the run is:' in x else None)
            if ph and ph not in ban[k]:ban[k].append(ph)
        m=hist[k]+[{'role':'user','content':FIX.format(probs=probs,ban='; '.join('"'+b+'"' for b in ban[k][:10]) or '(see above)')}];t,u=api.msg(m);hist[k]=m+[{'role':'assistant','content':t}];texts[k]=t
    r=0
    while todo(c) and r<3:
        r+=1
        with ThreadPoolExecutor(a.workers) as ex_:list(ex_.map(fixone,sorted(todo(c))))
        c=checks();rounds.append(len(todo(c)))
    fails={k:c[k] for k in range(64) if c[k]}
    runs=[];
    for k in range(64):
        for j in range(k+1,64):runs.append(A.run(A.blank(texts[k],p['golds']),A.blank(texts[j],p['golds'])))
    pr=[A.run(A.blank(texts[k],p['golds']),A.blank(t,p['golds'])) for k in range(64) for _,t in prior]
    stats=dict(model=a.model,pid=a.pid,words_original=n0,first_pass_ok=first,rounds_need_rewrite=rounds,final_with_any_violation=len(fails),final_need_rewrite_min=len(todo(c)),tokens_in=api.tok[0],tokens_out=api.tok[1],requests=api.calls,sec=time.time()-t0,max_pairwise_run_among_64=max(runs),median_pairwise_run=sorted(runs)[len(runs)//2],max_run_vs_original_and_existing=max(pr),median_run_vs_original_and_existing=sorted(pr)[len(pr)//2],length_words=dict(min=min(len(t.split()) for t in texts),max=max(len(t.split()) for t in texts)),thinking_param_used=api.think)
    json.dump(dict(stats=stats,cells=[dict(format=FORMATS[f],instruction=INSTR[g],text=texts[k],violations=c[k]) for k,(f,g) in enumerate(cells)]),open(O/f'grid_{a.model}_{a.pid}.json','w'),ensure_ascii=False,indent=1);lg(json.dumps(stats))
if __name__=='__main__':main()
