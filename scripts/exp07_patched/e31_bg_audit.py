"""E31 A5 background-paragraph overlap audit (CPU, read-only; needs data/narrative_gutenberg and data/tokenizer on the server).
Usage: e31_bg_audit.py [--seeds 101,102,103] [--passes 8] [--canary-chunks 7]
Reproduces gw.background(random.Random(7000+p)) for p=2..8 exactly as ladder.run_L0_fillerswap does (slot-wise replacement of the world's
'bg' texts by pool[bi % len(pool)]; the extra paragraphs appended when the stream is short are approximated by continuing the cycle until the
approximate token budget NCH*CH is met, flagged 'approx'). Pass 1 = the world's own bg texts from gw.build_world(seed).
Metrics per (seed, pass): source book, same-book-as-previous-pass, #bg slots, within-pass duplicate paragraphs, exact-paragraph overlap with
all earlier passes and with pass 1, exact overlap and character 8-gram overlap with the canary texts (pg11 first 8*CH tokens ~ chars; pg1342 tail)
and with the R1-alt context (SQuAD-dev, only if --squad PATH is given). Names are scrubbed the same way as in gen_world (applied by background()).
Output: OUT/e31_bg_audit.{md,json}; also records sha256 of the pg*.txt used."""
import os,sys,re,json,random,hashlib,argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101,102,103');ap.add_argument('--passes',type=int,default=8);ap.add_argument('--squad',default=None)
a=ap.parse_args()
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
import gen_world as gw
OUT=Path(os.environ.get('EXP07_OUT',ROOT/'experiments/exp07_write_ladder'));DATA=ROOT/'data/narrative_gutenberg'
norm=lambda s:' '.join(s.split())
def book(rng):   # same choice as gw.background (first rng call)
    files=[f for f in sorted(DATA.glob('pg*.txt')) if f.name in ('pg98.txt','pg1260.txt','pg768.txt','pg766.txt')];return rng.choice(files).name
sha={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(DATA.glob('pg*.txt'))}
def canary_text(name):
    t=(DATA/name).read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t;return norm(t.replace('\r',''))
can={'pg11':canary_text('pg11.txt'),'pg1342':canary_text('pg1342.txt')}
def grams(s,n=8):return {s[i:i+n] for i in range(len(s)-n+1)}
cg={k:grams(re.sub(r'\b[A-Z][a-z]+\b','someone',v)) for k,v in can.items()}   # name-scrubbed variant for the neutral tail
cg_raw={k:grams(v) for k,v in can.items()}
sq=None
if a.squad:
    d=json.load(open(a.squad));sq=norm(' '.join(p['context'] for x in d['data'] for p in x['paragraphs']));sqg=grams(sq)
def ov(paras,G):
    t=norm(' '.join(paras));g=grams(t);return len(g&G)/max(len(g),1)
rows=[];res={}
for seed in map(int,a.seeds.split(',')):
    w=gw.build_world(seed);bg1=[t['text'] for t in w['texts'] if t['kind']=='bg'];nslot=len(bg1);passes={1:bg1};books={1:'(world rng)'}
    for p in range(2,a.passes+1):
        pool=gw.background(random.Random(7000+p));books[p]=book(random.Random(7000+p));passes[p]=[pool[i%len(pool)] for i in range(nslot)]
        res.setdefault(f's{seed}',{})[f'pool_size_p{p}']=len(pool)
    seen=set();first=set(passes[1])
    for p in range(1,a.passes+1):
        L=passes[p];S=set(L);prev=books.get(p-1);row={'seed':seed,'pass':p,'book':books[p],'same_book_as_prev':bool(p>1 and books[p]==prev),'n_slots':nslot,'dup_within':len(L)-len(S),
          'overlap_earlier':len(S&seen)/len(S),'overlap_pass1':len(S&first)/len(S),
          'pg11_8gram':ov(L,cg_raw['pg11']),'pg1342_8gram_scrubbed':ov(L,cg['pg1342']),'pg1342_8gram_raw':ov(L,cg_raw['pg1342'])}
        row['pg11_exact_para']=sum(1 for x in S if x in can['pg11'])/len(S);row['pg1342_exact_para']=sum(1 for x in S if x in can['pg1342'])/len(S)
        if sq:row['squad_8gram']=ov(L,sqg)
        rows.append(row);seen|=S
cols=[k for k in rows[0]]
md=['# E31 A5 background overlap audit\n','Pool paragraphs after gen_world scrubbing; pass 1 = world seed stream. Extras appended for short streams are NOT reproduced (slot-wise only), so counts are lower bounds.\n',
    '| '+' | '.join(cols)+' |','|'+'---|'*len(cols)]
for r in rows:md.append('| '+' | '.join(f'{r[c]:.3f}' if isinstance(r[c],float) else str(r[c]) for c in cols)+' |')
md.append('\npg sha256: '+json.dumps({k:v for k,v in sha.items() if k in ('pg11.txt','pg1342.txt','pg98.txt','pg766.txt','pg768.txt','pg1260.txt')}))
json.dump({'rows':rows,'sha256':sha,'extra':res},open(OUT/'e31_bg_audit.json','w'),indent=1);(OUT/'e31_bg_audit.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
