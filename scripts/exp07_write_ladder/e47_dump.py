"""E47 mask dump (CPU only; server: gen_world, suppress, tokenizer): for chosen chunks with an oracle set, the sentences of the chunk's pairs (every round that lies in [window start, chunk end)) printed as text, with the tokens masked by each cell marked by [[ ]]: cells K3 (Sets.sentencecover), k3h (Sets.k3hmask, only the masked person of each pair; the rule of half_person_of(seed, pair, triples)), k3all (Sets.k3all_parts: every earlier-round token of both persons, the current round excluded). Each sentence line: sentence index, role (WRITE = a fact sentence of the chunk, earlier round / same round, other person / later), person, round, token range, whether it lies in the window or in the chunk, and the rendering per cell. Masked text is shown token by token (a token is one tokenizer piece incl. its leading space); adjacent masked tokens are merged.
Usage: e47_dump.py --seed 101 --world A --pass 2 [--chunks 12,30,41 | --auto 3] --out dump.txt   |   --selftest
--auto n picks n chunks with O in this order of preference: (1) a pair with both persons' fact sentences in the chunk and an earlier round of both inside the window, (2) a pair whose partner has no fact sentence in the chunk (split pair) if one exists, (3) the next chunk with the most earlier-round sentences."""
import sys,re,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
CH=SETS.CH;W=SETS.W
def render(text,offs,toks,masked,start,end):
    """text of the sentence char range [start,end) with the masked tokens (set of token indices) in [[ ]]"""
    out=[];pos=start
    for j in sorted(toks):
        a,b=offs[j]
        if a<pos:a=pos
        out.append(text[pos:a]);out.append(('[['+text[a:b]+']]') if j in masked else text[a:b]);pos=max(pos,b)
    out.append(text[pos:end]);r=''.join(out);return re.sub(r'\]\](\s*)\[\[',r'\1',r)
def pick(sets,n):
    byk={};chosen=[]
    for k in range(63):
        if not sets.oracle(k):continue
        cur=sets.cur_sentences(k);lo=max(0,k*CH-W);both=False;split=False;prior=0;by={}
        for s in cur:by.setdefault(s['pair'],set()).add(s['t'])
        for pr,ts in by.items():
            earlier=[u for u in sets.t if u['pair']==pr and u['place'][0]>=lo and any(u['round']<s['round'] for s in cur if s['pair']==pr)];prior+=len(earlier)
            if len(ts)==2 and {u['t'] for u in earlier}==ts:both=True
            if len(ts)==1:split=True
        byk[k]=(both,split,prior)
    for k,(b,s,p) in byk.items():
        if b:chosen.append(k);break
    for k,(b,s,p) in byk.items():
        if s and k not in chosen:chosen.append(k);break
    for k,(b,s,p) in sorted(byk.items(),key=lambda x:-x[1][2]):
        if len(chosen)>=n:break
        if k not in chosen:chosen.append(k)
    return chosen[:n]
def dump_chunk(sets,k,text,offs,spans,seed):
    lo=max(0,k*CH-W);k0=k*CH;cur=sets.cur_sentences(k);curidx={s['idx'] for s in cur};K3=set(sets.sentencecover(k));H=set(sets.k3hmask(k));A,st=sets.k3all_parts(k);A=set(A);L=[f'=== chunk {k}: positions {k0}..{k0+CH-1}, window start {lo}; fact (WRITE) sentences {sorted(curidx)}; |K3|={len(K3)} |k3h|={len(H)} |k3all|={len(A)}; k3all stats {st}']
    pairs=sorted({s['pair'] for s in cur})
    for pr in pairs:
        ts=sorted({u['t'] for u in sets.t if u['pair']==pr});L.append(f'--- pair {pr}: persons {ts}; masked person of k3h = {sets.half_person(pr)}')
        wr=[s for s in cur if s['pair']==pr];rcur=max(s['round'] for s in wr)
        for u in sorted([u for u in sets.t if u['pair']==pr and lo<=u['toks'][-1] and u['toks'][0]<k0+CH],key=lambda u:u['place'][0]):
            role='WRITE' if u['idx'] in curidx else ('earlier round' if u['round']<rcur else ('same round, other person' if u['round']==rcur else 'later round'))
            where='in chunk' if u['place'][0]>=k0 else 'in window'
            sp=spans[u['idx']];e=sp['end'] if 'end' in sp else sp['start']+len(sp['text'])
            L.append(f"  [{u['idx']}] {role}; person {u['t']} round {u['round']}; tokens {u['toks'][0]}..{u['toks'][-1]} ({where})")
            for nm,M in (('K3   ',K3),('k3h  ',H),('k3all',A)):L.append(f"      {nm}: {render(text,offs,u['toks'],M,sp['start'],e)}")
    return '\n'.join(L)+'\n'
def run(a):
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    w=K2.build_world_pass(gw,a.seed,a.passno);swap=a.world=='B';table=SETS.sentence_table(w,swap,gw.TOK,gw,SP);sets=SETS.Sets(table,a.seed,a.passno)
    text,_,spans=gw.realize(w,swap);e=gw.TOK.encode(text,add_special_tokens=False);offs=[(0,0)]+list(e.offsets)
    chunks=[int(x) for x in a.chunks.split(',')] if a.chunks else pick(sets,a.auto);out=[f'# E47 mask dump: seed {a.seed}, world {a.world}, pass {a.passno}, chunks {chunks} (K3, k3h, k3all; [[ ]] = masked)\n']
    for k in chunks:
        if not sets.oracle(k):out.append(f'=== chunk {k}: no oracle set, nothing is masked\n');continue
        out.append(dump_chunk(sets,k,text,offs,spans,a.seed))
    Path(a.out).write_text('\n'.join(out));print('\n'.join(out))
def selftest():
    text='Orsk worked as a smith in Velmar. Lenz, the smith of Velmar, said nothing.';offs=[(0,0)]+[(m.start(),m.end()) for m in re.finditer(r'\S+',text)]
    t=render(text,offs,[1,2,3,4,5,6,7],{1,7},0,33);assert t=='[[Orsk]] worked as a smith in [[Velmar.]]',t
    t2=render(text,offs,[1,2,3],{1,2,3},0,20);assert t2=='[[Orsk worked as]]'+text[14:20],t2
    import e47_k3all_test as T;tb=T.table();sets=SETS.Sets(tb,101,2);assert pick(sets,3)[0]==5
    words=[f'w{i}' for i in range(8300)];txt=' '.join(words);of=[(0,0)];p=0
    for w_ in words:of.append((p,p+len(w_)));p+=len(w_)+1
    sp=[{'start':of[u['toks'][0]][0],'end':of[u['toks'][-1]][1],'text':''} for u in sorted(tb,key=lambda u:u['idx'])]
    dc=dump_chunk(sets,5,txt,of,sp,101);assert 'WRITE' in dc and 'earlier round' in dc and '[[' in dc and dc.count('k3all')>=8,dc[:400]
    print('e47_dump SELFTEST OK: render marks and merges masked tokens; pick chooses a pair-in-chunk chunk first',pick(sets,3))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--world',default='A');ap.add_argument('--pass',dest='passno',type=int,default=2);ap.add_argument('--chunks',default=None);ap.add_argument('--auto',type=int,default=3);ap.add_argument('--out',default='/tmp/e47_dump.txt');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    run(a)
