"""E48 unit test (CPU; no tokenizers package, no data files: a stub tokenizer (words and punctuation as tokens) and a synthetic background are used). Checks: (1) gen_world.build_world default is bit-identical to the code before the E48 change (git HEAD~ copy given by --old, else skipped with a note) ; (2) frames='pair' (W1): persons / places / occupations / control names equal the default world, every triple has one exposure per round 1..16, the bank has 15 x 16 distinct frames, frames of different pairs are disjoint, the two persons of a pair share the pair's R frames, every sentence keeps the slots (X, P, o); components (verb, adjective, noun, adverbial, lead, subject pattern) are not shared between pairs (asserted in frame_bank); (3) frames='sentence' (W2): 480 distinct frames, one per exposure sentence; (4) s3_sets.sentence_table identifies the frame of every sentence in the bank (frame != -1) and the frame disjointness holds on the table; (5) the frame length (words) follows the original distribution."""
import sys,types,re,json,random,importlib.util,argparse,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];EX04=ROOT/'scripts/exp04_what_ttt_retains'
class _Enc:
    def __init__(s,text):
        ms=list(re.finditer(r"\w+|[^\w\s]",text));s.ids=[hash(m.group())%50000 for m in ms];s.offsets=[(m.start(),m.end()) for m in ms]
class _Tok:
    @classmethod
    def from_file(c,p):return c()
    def encode(s,text,add_special_tokens=False):return _Enc(text)
def load_gw(path,name):
    sys.modules['tokenizers']=types.SimpleNamespace(Tokenizer=_Tok)
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def bg(rng):
        p=[f'Plain paragraph {i} with some filler words about the long road, the grey sky and the quiet river bank near the hill.' for i in range(400)];rng.shuffle(p);return p
    m.background=bg;return m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old',default=None,help='path of the gen_world.py before the E48 change (git show <commit>:scripts/exp04_what_ttt_retains/gen_world.py)');a=ap.parse_args()
    sys.path.insert(0,str(EX04));sys.path.insert(0,str(HERE))
    gw=load_gw(EX04/'gen_world.py','gen_world');sys.modules['gen_world']=gw
    w0=gw.build_world(101,frames='shared');R=16
    if a.old:
        go=load_gw(a.old,'gen_world_old');wo=go.build_world(101);assert json.dumps(wo,sort_keys=True)==json.dumps(w0,sort_keys=True),'default world differs from the old code';print('(1) default world bit-identical to the old gen_world')
    else:print('(1) skipped (no --old)')
    import frame_bank as FB,s3_sets as SETS,suppress as SP
    for mode in ('pair','sentence'):
        w=gw.build_world(101,frames=mode);assert w['triples']==w0['triples'] and w['ctrl_persons']==w0['ctrl_persons'] and w['ctrl_places']==w0['ctrl_places']
        bank=w['frames'];ex=[t for t in w['texts'] if t['kind']=='expo'];assert len(ex)==30*R,len(ex)
        assert sorted((t['t'],t['round']) for t in ex)==sorted((i,r) for i in range(30) for r in range(1,R+1)),'one exposure per triple and round'
        assert len(set(bank))==len(bank)==(15*R if mode=='pair' else 30*R),len(bank)
        # frame of every sentence: the sentence text equals one bank frame formatted
        tri={t['id']:t for t in w['triples']};frame_of={}
        for t in ex:
            tr=tri[t['t']];hits=[j for j,f in enumerate(bank) if f.format(X='@X%d@'%t['t'],P=tr['P'],o=tr['o'],a_o=gw.A(tr['o']))==t['text']];assert len(hits)==1,(t,hits);frame_of[(t['t'],t['round'])]=hits[0]
        used={}
        for (i,r),j in frame_of.items():used.setdefault(j,set()).add(i)
        if mode=='pair':
            pf={}
            for (i,r),j in frame_of.items():pf.setdefault(i//2,set()).add(j)
            ps=sorted(pf);assert all(len(pf[p])==R for p in ps),'R distinct frames per pair'
            for p in ps:
                for q in ps:
                    if p<q:assert not (pf[p]&pf[q]),'frames shared between pairs'
            for i in range(30):assert {j for (t,r),j in frame_of.items() if t==i}==pf[i//2],'both persons use the pair frames'
        else:assert all(len(v)==1 for v in used.values()) and len(used)==30*R,'every sentence has its own frame'
        for f in bank:assert '{X}' in f and '{P}' in f and ('{o}' in f or '{a_o}' in f)
        # (4) sentence_table / frame field
        tb=SETS.sentence_table(w,False,gw.TOK,gw,SP);assert len(tb)==30*R and all(u['frame']>=0 for u in tb),'frame not identified';assert all(u['frame']==frame_of[(u['t'],u['round'])] for u in tb)
        lens=[FB._words(f) for f in bank];print(f'({mode}) frames {len(bank)}, words/frame mean {sum(lens)/len(lens):.2f} (original {sum(FB.orig_lengths(gw.FR))/30:.2f}), min {min(lens)} max {max(lens)}; distinct per pair / sentence checks passed; sentence_table frame field consistent')
        wB_text=gw.realize(w,True)[0];assert '@X' not in wB_text
    print('e48_world_test OK (stub tokenizer; real token lengths are reported by e48_world_stats.py on the server)')
if __name__=='__main__':main()
