"""Run in the JAX env (.venv) on CPU: dump the texts and llama3 token lists the Exp07 JAX pipeline uses, built by the
JAX pipeline's own functions (readouts.neutral_tail, stage2.Probe, ladder.Ladder.sentences / l1_rows / fs_streams /
_gen_setup, ladder.squad_tail) called on light stand-in objects (no model is loaded). Used by test_texts_vs_jax.py.
Usage: CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python ref_dump_jax.py OUT.json [seeds...]"""
import sys,json,hashlib,contextlib
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/exp05_write_dynamics'));sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
import readouts as RD
import ladder as LD
gw=RD.gw;stage2=RD.stage2
def h(x):return hashlib.sha256(json.dumps(x).encode()).hexdigest()
def main():
    out=Path(sys.argv[1]);seeds=[int(x) for x in sys.argv[2:]] or [101,102,103]
    tok=RD.reuse.load_tokenizer();bos=tok.token_to_id('<|begin_of_text|>');eos=tok.token_to_id('<|end_of_text|>')
    rt=NS(tok=tok,bos=bos,eos=eos,base=None,mesh=contextlib.nullcontext(),seq=lambda ids:list(ids),prefix=lambda b,s:None)
    tail_ids,tail_text,tail_sha=RD.neutral_tail(tok,bos)
    res={'bos':bos,'tail':{'hash':h(tail_ids),'len':len(tail_ids),'sha':tail_sha,'text_hash':h(tail_text)}}
    fake=NS(rt=rt,T=NS(tok=tok));LD.Ladder._gen_setup(fake);res['gen']={'hash':h(list(fake._gseq)),'len':len(fake._gseq)}
    for seed in seeds:
        w=gw.build_world(seed,v=2);P={s:stage2.Probe(rt,w,s) for s in (False,True)}
        T=NS(w=w,tok=tok,P=P);L=NS(T=T,rt=rt);d={}
        for s,tag in ((False,'A'),(True,'B')):
            text,_,spans=gw.realize(w,s)
            d[tag]={'text':text,'spans':[x['text'] for x in spans],'stream_hash':h(list(P[s].ids)),'stream_len':len(P[s].ids),
                    'sentences_hash':h(LD.Ladder.sentences(L,s)),
                    'rows':{v:h([[list(map(int,a)),list(b)] for a,b in LD.Ladder.l1_rows(L,s,v)]) for v in ('plain','ctx','irr','fill')},
                    'fs3':[h(list(map(int,x))) for x in LD.Ladder.fs_streams(L,s,3)]}
        forbid=[t['X'] for t in w['triples']]+[t['P'] for t in w['triples']]+list(w['ctrl_persons'])+list(w['ctrl_places'])
        alt=LD.squad_tail(tok,bos,forbid);d['alt']={'hash':h(alt),'len':len(alt)}
        streams=gw.realize(w,False)[0]+gw.realize(w,True)[0]
        pp,pl=RD.pseudo_names(w,2*RD.N_PSEUDO,tok,streams+'\n'+tail_text);d['pseudo']={'persons':pp,'places':pl}
        _,name,_=gw.realize(w,False);d['probes_gw']=gw.probes(w,name)
        res[str(seed)]=d;print('seed',seed,'stream lens',d['A']['stream_len'],d['B']['stream_len'],flush=True)
    json.dump(res,open(out,'w'));print('wrote',out)
if __name__=='__main__':main()
