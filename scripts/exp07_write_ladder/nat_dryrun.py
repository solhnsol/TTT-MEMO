"""E33 R CPU dry run of the filler assembly (no GPU, no model, no jax): the same steps as NatLadder.fs_streams for L0fs, both worlds (A = native names, B = swapped), passes 2..P:
pool selection + audit (nat_filler), matched stream assembly (nat_pass.build_pass_stream_matched), exposure-unit positions equal to pass 1 (drift <= 2), strata equal to pass 1, no pool paragraph used twice,
audit on the used texts. Tokenizer: --tokenizer-file FILE (the real llama3 tokenizer.json: the lab runs this with it) or the baseline_config.TOKENIZER_PATH when importable; otherwise a regex PROXY tokenizer is used and
the report says so (proxy results test the logic and the pool, not the real drift).
Usage: nat_dryrun.py --nat-dir DIR --stream 9105 --passes 3 [--tokenizer-file tokenizer.json] [--worlds AB]   exit 3 on any failure"""
import os,sys,re,argparse,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_world as NW,nat_filler as NF,nat_pass as NP
NCH,CH=63,1024
class Proxy:
    """regex pre-tokenizer stand-in: ids = piece indices (not the real tokenizer)"""
    rx=re.compile(r"\s?\w+|\s?[^\w\s]+|\n+|\s+")
    def __init__(s):s.v={};s.inv=[]
    def encode(s,text):
        ids=[];ends=[]
        for m in s.rx.finditer(text):
            p=m.group();i=s.v.setdefault(p,len(s.v))
            if i==len(s.inv):s.inv.append(p)
            ids.append(i);ends.append(m.end())
        return ids,ends
    def decode(s,ids):return ''.join(s.inv[i] for i in ids)
def load_tok(path):
    if path is None:
        try:
            sys.path.insert(0,str(HERE.parents[0]/'exp01_gold_evidence_qa'));from baseline_config import TOKENIZER_PATH;path=str(TOKENIZER_PATH)
        except Exception:return None
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nat-dir',default=os.environ.get('NAT_DIR','/mnt/project-files/data/natural-text/worlds'));ap.add_argument('--stream',type=int,default=9105);ap.add_argument('--passes',type=int,default=3);ap.add_argument('--layout',default='spread')
    ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--worlds',default='AB');ap.add_argument('--require-real',action='store_true');a=ap.parse_args()
    tk=None
    try:tk=load_tok(a.tokenizer_file)
    except Exception as e:print('real tokenizer not usable:',repr(e)[:150])
    if tk is not None:
        def encode(t):
            e=tk.encode(t,add_special_tokens=False);return e.ids,[b for _,b in e.offsets]
        decode=lambda ids:tk.decode(list(ids));kind='REAL tokenizers'
    else:
        px=Proxy();encode=px.encode;decode=px.decode;kind='PROXY regex tokenizer (logic and pool only, drift is not the real one)'
    print('tokenizer:',kind,flush=True)
    assert tk is not None or not a.require_real,'--require-real: the real tokenizer could not be loaded (pip install tokenizers; set --tokenizer-file)'
    w0=NW.load_world(NF.world_path(a.nat_dir,a.stream,a.layout));canary=NF.canary_grams()
    pools,info=NF.select(a.nat_dir,a.stream,a.passes,a.layout,canary=canary or None);rep=NF.audit(pools,NF.bg_texts(w0),canary or None);hits=NF.forbid_hits(pools,w0)
    assert rep['pass'] and all(v==0 for v in hits.values()),('filler audit failed',rep,hits)
    print('filler audit PASS; pool/slot chars',{p:round(sum(len(t) for t in v)/sum(len(t) for t in NF.bg_texts(w0)),2) for p,v in pools.items()},flush=True)
    for wd in a.worlds:
        swap=wd=='B';t0=time.monotonic();text0,_,spans0=NW.realize(w0,swap);ids0,ends0=encode(text0);sp0=NP.tok_spans(spans0,ends0);st1=NP.strata_real(sp0);used_all={}
        for ps in range(2,a.passes+1):
            w,ids,sp,binfo,used=NP.build_pass_stream_matched(w0,pools[ps],swap,encode,decode,0,NCH*CH if tk is not None else len(ids0)-8)   # proxy streams are shorter than the real 63-chunk stream
            st=NP.strata_real(sp);assert all((st[t]['s_cross'],st[t]['s_same'],st[t]['n_out'])==(st1[t]['s_cross'],st1[t]['s_same'],st1[t]['n_out']) for t in st1),f'world {wd} pass {ps}: strata differ from pass 1'
            used_all[ps]=used;dr=[s['tok_start']-s0['tok_start'] for s,s0 in zip(sp,sp0)];print(f'world {wd} pass {ps}: OK, units {len(sp)}, max |drift| {max(abs(x) for x in dr)}, mean |drift| {sum(abs(x) for x in dr)/len(dr):.2f}, build {binfo}',flush=True)
        flat=[x for v in used_all.values() for x in v];assert len(flat)==len(set(flat)),'a used pool text appears twice'
        rep2=NF.audit(used_all,NF.bg_texts(w0),canary or None);h2=NF.forbid_hits(used_all,w0);assert rep2['pass'] and all(v==0 for v in h2.values()),('audit on used texts failed',rep2,h2)
        print(f'world {wd}: audit on used texts PASS ({time.monotonic()-t0:.0f}s)',flush=True)
    print('DRYRUN OK ('+kind+')')
if __name__=='__main__':
    try:main()
    except Exception as e:
        print('DRYRUN FAILED:',repr(e)[:1500]);sys.exit(3)
