"""E57 QA re-descriptions through the Claude API, 8 x 8 grid with a SIMILARITY check (new file; network only; Hans 11/05 '유사도/Jaccard 기준으로 겹침을 검사'): as e57_api_grid.py (64 independent calls per paragraph, one re-description each) but (1) the existing re-descriptions are not sent to the model, (2) a re-description fails when the Jaccard overlap of its word 2-grams exceeds 0.35 or of its word 4-grams 0.15 with the original, an existing re-description or another of the 64 (answer strings blanked), or when it breaks the own rules (answer strings, length 0.6-1.5x, one plain paragraph, finished sentences); (3) repairs are single-turn (original + the failing text + the problem + the text it resembles), the failing set being the own failures plus a greedy cover of the similarity pairs, up to 3 rounds. The key is read from ttt-memo/.env inside this process only.
  python3 e57_api_grid2.py --in DIR --out DIR --pid PID --model M | --selftest"""
import sys,json,re,time,pathlib,argparse,collections
from concurrent.futures import ThreadPoolExecutor
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_api_grid as G
A=G.A
GENERAL=['state the single most important point of the paragraph first in one sentence, then add the supporting details','begin with the broader background or context, then the specific facts, and put the main conclusion last','begin with whichever fact seems most striking or interesting, then bring in the rest in whatever order reads best','present the material in roughly reverse order of the original: start from its last point and work back to its first','group the facts by theme or topic and handle each group together, regardless of the original order','use several short, plain sentences, one idea per sentence','use a few long, well-connected sentences that tie related facts together','address the reader directly, for example by opening with a question or a phrase that speaks to the reader, and explain the material to them']
P2=G.PROMPT.split('EXISTING RE-DESCRIPTIONS')[0].replace(', or from the existing re-descriptions below','').replace(' from the original paragraph or from','').rstrip()+'\n\nAnswer with the re-description text only.'
P2=P2.replace('do not reuse any stretch of more than 7 consecutive ordinary words from the original paragraph','do not copy stretches of the original paragraph (use clearly different vocabulary and clause structure)')
FIXP='''ORIGINAL PARAGRAPH:
{ctx}

ANSWER STRINGS (each must appear verbatim): {golds}
The re-description below was written in this FORMAT / VOICE: {fmt}; STRUCTURE: {ins}.

TEXT:
{text}

PROBLEM: {prob}
{other}
Rewrite the text from scratch with clearly different vocabulary, clause structure and sentence order, keeping the format, the structure instruction, the answer strings, all facts and names/numbers/dates, between {lo} and {hi} words, one plain prose paragraph, finished sentences.
Answer with the re-description text only.'''
def evaluate(texts,p,prior):
    n=len(texts);own={k:A.check_one(texts[k],p,[],[]) for k in range(n)};ng={k:(G._ng(G._tk(A.blank(texts[k],p['golds'])),2),G._ng(G._tk(A.blank(texts[k],p['golds'])),4)) for k in range(n)}
    pri=[(lab,G._ng(G._tk(A.blank(t,p['golds'])),2),G._ng(G._tk(A.blank(t,p['golds'])),4)) for lab,t in prior]
    simprior={k:[] for k in range(n)};edges={}
    for k in range(n):
        for lab,a2,a4 in pri:
            x2=G._jac(ng[k][0],a2);x4=G._jac(ng[k][1],a4)
            if x2>G.J2 or x4>G.J4:simprior[k].append((lab,x2,x4))
    for i in range(n):
        for j in range(i+1,n):
            x2=G._jac(ng[i][0],ng[j][0]);x4=G._jac(ng[i][1],ng[j][1])
            if x2>G.J2 or x4>G.J4:edges[(i,j)]=(x2,x4)
    return own,simprior,edges
def need(own,simprior,edges,n):
    base={k for k in range(n) if own[k] or simprior[k]};E={frozenset(e) for e in edges if not(set(e)&base)};pick=set()
    while E:
        d=collections.Counter(i for e in E for i in e);i=d.most_common(1)[0][0];pick.add(i);E={e for e in E if i not in e}
    return base|pick
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--in',dest='inp');ap.add_argument('--out');ap.add_argument('--pid');ap.add_argument('--model');ap.add_argument('--workers',type=int,default=20);ap.add_argument('--instr',default='orig');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        p={'pid':'x','context':'The dog Rex barked at the postman near the old mill on 5 May 1990 and then slept.','golds':['Rex']}
        o,sp,ed=evaluate(['Rex barked at the postman near the old mill on 5 May 1990 and then slept.','Near the old mill in 1990 the postman met a barking Rex, who later rested for hours that day.'],p,[('orig',p['context'])])
        assert sp[0] and need(o,sp,ed,2)>={0};print('e57_api_grid2 selftest ok');return
    if a.instr=='general':G.INSTR=GENERAL
    I=pathlib.Path(a.inp);O=pathlib.Path(a.out);O.mkdir(parents=True,exist_ok=True)
    Pp={r['pid']:r for r in json.load(open(I/'paragraphs.json'))};E=json.load(open(I/'all8.json'));p=Pp[a.pid];ex=E[a.pid];n0=len(p['context'].split());lo,hi=int(0.6*n0)+1,int(1.5*n0)
    api=G.API2(a.model);t0=time.time();prior=[('the original',p['context'])]+[(f'existing #{j+1}',t) for j,t in enumerate(ex)]
    cells=[(f,g) for f in range(8) for g in range(8)];texts=['']*64
    def gen(k):
        f,g=cells[k];t,u=api.msg([{'role':'user','content':P2.format(ctx=p['context'],golds=json.dumps(p['golds'],ensure_ascii=False),fmt=G.FORMATS[f],ins=G.INSTR[g],n0=n0,lo=lo,hi=hi)}]);texts[k]=t
    with ThreadPoolExecutor(a.workers) as e_:list(e_.map(gen,range(64)))
    own,sp,ed=evaluate(texts,p,prior);first_ok=sum(1 for k in range(64) if not own[k] and not sp[k] and not any(k in e for e in ed));rounds=[len(need(own,sp,ed,64))]
    def fix(k):
        f,g=cells[k];probs=list(own[k]);other=''
        for lab,x2,x4 in sp[k]:probs.append(f'too similar to {lab} (word 2-gram overlap {x2:.2f}, 4-gram {x4:.2f})')
        for (i,j),(x2,x4) in ed.items():
            if k in (i,j):
                o=j if k==i else i;probs.append(f'too similar to another re-description (2-gram overlap {x2:.2f}, 4-gram {x4:.2f})');other+=f'\nTHE TEXT IT RESEMBLES (do not reuse its wording):\n{texts[o]}\n';break
        if not other and sp[k] and sp[k][0][0]!='the original':
            lab=sp[k][0][0]
            if lab.startswith('existing #'):other=f'\nTHE TEXT IT RESEMBLES (do not reuse its wording):\n{ex[int(lab.split("#")[1])-1]}\n'
        t,u=api.msg([{'role':'user','content':FIXP.format(ctx=p['context'],golds=json.dumps(p['golds'],ensure_ascii=False),fmt=G.FORMATS[f],ins=G.INSTR[g],text=texts[k],prob='; '.join(probs),other=other,lo=lo,hi=hi)}]);texts[k]=t
    r=0
    while need(own,sp,ed,64) and r<3:
        r+=1
        with ThreadPoolExecutor(a.workers) as e_:list(e_.map(fix,sorted(need(own,sp,ed,64))))
        own,sp,ed=evaluate(texts,p,prior);rounds.append(len(need(own,sp,ed,64)))
    allv=[v for k in range(64) for v in (list(own[k]) if own[k] else [])]
    b2=[G._jac(G._ng(G._tk(A.blank(texts[i],p['golds'])),2),G._ng(G._tk(A.blank(texts[j],p['golds'])),2)) for i in range(64) for j in range(i+1,64)]
    stats=dict(model=a.model,pid=a.pid,words_original=n0,thresholds=dict(bigram_jaccard=G.J2,fourgram_jaccard=G.J4),first_pass_clean=first_ok,rounds_need_rewrite=rounds,final_own_failures=sum(1 for k in range(64) if own[k]),final_similar_pairs=len(ed),final_similar_to_original_or_existing=sum(1 for k in range(64) if sp[k]),pairwise_bigram_jaccard_mean=sum(b2)/len(b2),pairwise_bigram_jaccard_max=max(b2),tokens_in=api.tok[0],tokens_out=api.tok[1],requests=api.calls,sec=time.time()-t0,length_words=dict(min=min(len(t.split()) for t in texts),max=max(len(t.split()) for t in texts)))
    json.dump(dict(stats=stats,cells=[dict(format=G.FORMATS[f],instruction=G.INSTR[g],text=texts[k]) for k,(f,g) in enumerate(cells)]),open(O/f'grid2{"g" if a.instr=="general" else ""}_{a.model}_{a.pid}.json','w'),ensure_ascii=False,indent=1);print(json.dumps(stats))
if __name__=='__main__':main()
