"""E51 CPU audit of the synthetic name/place/occupation/frame pools: can an eval world be disjoint from the dev worlds (seeds 101-103)?
Reproduces gen_world.build_world's first draws exactly (person_names(rng,120), place_names(rng,120), token-length class filter with the real tokenizer, shuffle) without building the background text.
Reports: size of every token-length class of the full pools (SYL1 x SYL2 = 600 persons, PL_PRE x PL_SUF = 300 places), the class each seed uses, the names each dev seed uses (38 persons incl. 8 control, 38 places),
their union over the dev seeds, what is left in the same class for an eval world (38 + 38 needed), the same for halved pools, the 30 occupations (all used by every world), and the E48 frame bank facts
(W1 'pair' bank depends only on (R, npairs): seed-independent, whole component pools at 15 pairs).
No GPU; needs the real tokenizer (data/tokenizer/llama3/tokenizer.json via gen_world). Usage: e51_pool_audit.py --dev-seeds 101,102,103 [--eval-seeds 201] | --selftest"""
import sys,argparse,random,itertools
from pathlib import Path
HERE=Path(__file__).resolve().parent;GW=HERE.parents[0]/'exp04_what_ttt_retains';sys.path.insert(0,str(GW));sys.path.insert(0,str(HERE))
def import_gw(fake=False):
    if fake:
        import types
        class _T:
            def encode(self,s,add_special_tokens=False):
                w=s.strip();return types.SimpleNamespace(ids=list(range(1+(len(w)>5)+(len(w)>8))))
        m=types.ModuleType('tokenizers');m.Tokenizer=types.SimpleNamespace(from_file=lambda p:_T());sys.modules['tokenizers']=m
    import gen_world as G;return G
def draw(G,seed,pairs=15):
    """the first rng draws of build_world: returns (persons used incl. ctrl, places used incl. ctrl, person class length, place class length, all persons of the class, all places of the class)"""
    rng=random.Random(seed);nt=pairs*2
    pn=G.person_names(rng,nt*4);pl=G.place_names(rng,nt*4)
    byl={};[byl.setdefault(G.ntok(n),[]).append(n) for n in pn]
    L=max(byl,key=lambda k:len(byl[k]));persons=byl[L]
    byp={};[byp.setdefault(G.ntok(n),[]).append(n) for n in pl]
    Lp=max(byp,key=lambda k:len(byp[k]));places=byp[Lp]
    rng.shuffle(persons);rng.shuffle(places)
    return persons[:nt+8],places[:nt+8],L,Lp,len(persons),len(places)
def classes(G):
    pp=[a+b for a in G.SYL1 for b in G.SYL2];pl=[a+b for a in G.PL_PRE for b in G.PL_SUF]
    cp={};[cp.setdefault(G.ntok(n),[]).append(n) for n in pp];cl={};[cl.setdefault(G.ntok(n),[]).append(n) for n in pl];return cp,cl
def audit(G,dev,ev,pairs=15):
    nt=pairs*2;out=[];cp,cl=classes(G)
    out.append(f'full pools: persons {sum(map(len,cp.values()))} by token length { {k:len(v) for k,v in sorted(cp.items())} }; places {sum(map(len,cl.values()))} by token length { {k:len(v) for k,v in sorted(cl.items())} }; occupations {len(G.OCC)} (every world uses all {nt})')
    used_p=set();used_l=set();rows=[]
    for s in dev:
        P,L,lp,ll,nP,nL=draw(G,s,pairs);rows.append((s,lp,ll,len(P),len(L)));used_p|=set(P);used_l|=set(L)
    for s,lp,ll,a,b in rows:out.append(f'dev seed {s}: person class {lp} tokens, place class {ll} tokens, uses {a} persons / {b} places')
    out.append(f'dev union: {len(used_p)} distinct persons, {len(used_l)} distinct places (of the class pool)')
    for s in ev:
        P,L,lp,ll,nP,nL=draw(G,s,pairs);ov_p=len(set(P)&used_p);ov_l=len(set(L)&used_l)
        out.append(f'eval seed {s} as drawn today: person class {lp} ({nP} in class), overlap with dev persons {ov_p}/{len(P)}, places class {ll} ({nL} in class), overlap with dev places {ov_l}/{len(L)}')
    cpool={};[cpool.setdefault(0,[]).extend(v) for v in cp.values()]
    for (lp,names),(ll,pls) in itertools.product(cp.items(),cl.items()):pass
    # disjoint eval world from the same class as the dev worlds
    P0,L0,lp0,ll0,nP0,nL0=draw(G,dev[0],pairs);cls_p=[n for n in cp[lp0]];cls_l=[n for n in cl[ll0]]
    left_p=[n for n in cls_p if n not in used_p];left_l=[n for n in cls_l if n not in used_l];need=nt+8
    out.append(f'same-class names left after the dev union: persons {len(left_p)} (need {need}) -> {"OK" if len(left_p)>=need else "NOT ENOUGH"}; places {len(left_l)} (need {need}) -> {"OK" if len(left_l)>=need else "NOT ENOUGH"}')
    out.append(f'halved class pools (disjoint dev / eval, {need} each): persons {len(cls_p)//2} per half, places {len(cls_l)//2} per half')
    other_p={k:len(v) for k,v in cp.items() if k!=lp0};other_l={k:len(v) for k,v in cl.items() if k!=ll0}
    out.append(f'other token-length classes (different-length names would change the tokenization-matched design): persons {other_p}, places {other_l}')
    return out
def selftest():
    G=import_gw(True);P,L,lp,ll,nP,nL=draw(G,101);assert len(P)==38 and len(L)==38 and len(set(P))==38,(len(P),len(L))
    P2,*_=draw(G,101);assert P==P2
    lines=audit(G,[101,102,103],[201]);print('\n'.join(lines));print('e51_pool_audit SELFTEST OK (fake tokenizer: counts are NOT real)')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dev-seeds',default='101,102,103');ap.add_argument('--eval-seeds',default='');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    G=import_gw();print('\n'.join(audit(G,[int(x) for x in a.dev_seeds.split(',')],[int(x) for x in a.eval_seeds.split(',') if x])))
    import frame_bank as FB
    print('E48 frame pools:',{k:len(v) for k,v in FB.POOLS.items()},'; W1 needs per 15 pairs',{'subj':30,'verb':45,'adj':45,'noun':45,'adv':45,'lead':45},'-> the whole pools; bank seed-independent (make(mode,R,npairs) only)')
if __name__=='__main__':main()
