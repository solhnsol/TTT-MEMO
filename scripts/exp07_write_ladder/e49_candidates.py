"""E49 full-candidate readout (re-reads SAVED states, no training). For the L3 items of the world (30 items: 'X lives in ...') every one of the 30 places is scored as the continuation
(summed span log-prob, same packed fp32 readouts as Ladder.evaluate: R1 with the neutral pg1342 tail and R1-alt with the SQuAD tail). Per cell and world A/B the raw 30 x 30 score matrices are
stored, and e49_stats computes: bound target log-prob, normalized probability of the target among the 30 candidates, rank, top-1/top-3 accuracy, mass on the swap partner's place, 2-candidate accuracy, with 15-pair bootstrap CIs and changes vs a reference cell.
Usage (GPU):  e49_candidates.py --seed 101 --epoch 4 --ref k3all --cell k3all=S3_k3all --cell native=L0fs_fskeep ... [--w0]
  --cell NAME=PREFIX : states OUT/states/<PREFIX>_s<seed>/{A,B}/, the log OUT/<PREFIX>_s<seed>_{A,B}.saved.json (the entry with epoch == --epoch, hash checked); cells whose states are missing are skipped with a message (exit 7 with --strict).
  --w0 : also read the untrained model (baseline). Per cell the result file OUT/e49_cand_s<seed>_<NAME>.json is written (an existing one is reused: restart-safe).
CPU (no model):  e49_candidates.py --analyze --seed 101 --ref k3all --cell k3all=S3_k3all ...   (re-computes the tables from the stored result files; no jax needed)"""
import os,sys,json,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,e49_stats as ES
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--epoch',type=int,default=4);ap.add_argument('--epochs',default=None,help='comma list of passes to read per cell (default: --epoch only); with several passes the cell names become <name>_p<epoch>');ap.add_argument('--cell',action='append',default=[]);ap.add_argument('--ref',default=None);ap.add_argument('--w0',action='store_true')
    ap.add_argument('--analyze',action='store_true');ap.add_argument('--w0-name',default='W0',help='result name of the untrained-model read (default W0; give another name, e.g. w1_W0, so an existing W0 result file of another world is neither reused nor overwritten)');ap.add_argument('--strict',action='store_true');ap.add_argument('--nboot',type=int,default=4000);ap.add_argument('--out',default=None,help='result directory (default ladder.OUT)');ap.add_argument('--tag',default='');a=ap.parse_args()
    w0n=a.w0_name;EPS=[int(x) for x in a.epochs.split(',')] if a.epochs else [a.epoch];nm=lambda n,ep:n if len(EPS)==1 else f'{n}_p{ep}'
    cells=[c.split('=',1) for c in a.cell];names=[nm(n,ep) for n,_ in cells for ep in EPS]+([w0n] if a.w0 else []);cellpre={nm(n,ep):(pre,ep) for n,pre in cells for ep in EPS}
    OUT=Path(a.out or os.environ.get('EXP07_OUT',Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder'))   # same default as ladder.OUT, without importing jax
    fn=lambda n:OUT/f'e49_cand_s{a.seed}_{n}.json'
    if not a.analyze:
        assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
        import ladder as LD
        jax,Runtime,packing,stage2,reuse=LD.jax,LD.Runtime,LD.packing,LD.stage2,LD.reuse
        miss=[]
        for n,pre in cells:
            for w in 'AB':
                sd=OUT/'states'/f'{pre}_s{a.seed}'/w;sj=OUT/f'{pre}_s{a.seed}_{w}.saved.json'
                if not (sd.is_dir() and sj.exists()):miss.append((n,w))
        if miss:
            print('MISSING saved states / logs for',miss,flush=True)
            if a.strict:sys.exit(7)
        todo=[nm(n,ep) for n,_ in cells for ep in EPS if (n,'A') not in miss and (n,'B') not in miss and not fn(nm(n,ep)).exists()]+([w0n] if a.w0 and not fn(w0n).exists() else [])
        print('to read:',todo,'| already done:',[n for n in names if fn(n).exists()],flush=True)
        if todo:
            rt=Runtime();L=LD.Ladder(rt,a.seed,readout=True);T=L.T;w_=T.w
            places,gi,fi,rows_t,pairs=ES.candidate_items(L.fam['L3'],w_['triples']);C=len(places);n=len(pairs)
            tl=[len(T.tok.encode(p,add_special_tokens=False).ids) for p in places]
            if len(set(tl))!=1:print('WARNING candidate places differ in token length',sorted(set(tl)),'(summed span log-probs are not length-normalized)',flush=True)
            raw=[]
            for pre_,c in rows_t:
                ids,span=stage2.item_ids(T.tok,stage2.make_slot(T.tok,None),pre_,c);raw.append((ids,(span[0]-1,span[1]-1)))
            print('items',n,'candidates',C,'rows',len(raw),flush=True)
            with L.rt32.mesh,jax.default_matmul_precision('highest'):
                Pr=packing.Packed(L.ctx,raw,GP=2);Pa=packing.Packed(L.ctx_alt,raw,GP=2)
            def read(m):
                m32=L.to_fp32(m)
                with jax.default_matmul_precision('highest'):r=Pr.r1(m32);ra=Pa.r1(m32)
                return {'R1':np.asarray(r).reshape(n,C),'alt':np.asarray(ra).reshape(n,C)}
            t0=time.monotonic()
            for name in todo:
                S={}
                if name==w0n:S={'A':read(rt.base),'B':None};S['B']=S['A']   # untrained model: the worlds differ only in the stream, so both worlds read the same scores
                else:
                    pre,epn=cellpre[name]
                    for w in 'AB':
                        log=json.load(open(OUT/f'{pre}_s{a.seed}_{w}.saved.json'))['log'];ent=[e for e in log if isinstance(e.get('margins'),dict) and '_file' in e['margins'] and e.get('epoch')==epn]
                        assert len(ent)==1,f'{name} world {w}: epoch {epn} state not found in the saved log ({[e.get("epoch") for e in log]})'
                        m=L.load_state(ent[0]['margins']['_file']);assert reuse._hash_pytree(m.inner_parameters())[0]==ent[0]['margins']['_hash'],'loaded weights differ from the streamed weights'
                        S[w]=read(m);print(name,w,'read',f'{time.monotonic()-t0:.0f}s',flush=True)
                json.dump(ES.cell_record(name,places,gi,fi,pairs,S,{'epoch':(cellpre[name][1] if name!=w0n else 0),'cand_tokens':tl,'sec_total':time.monotonic()-t0}),open(fn(name),'w'))
    recs=[json.load(open(fn(n))) for n in names if fn(n).exists()]
    if not recs:print('no result files');return
    ref=nm(a.ref,max(EPS)) if a.ref else recs[0]['cell']   # several passes: the reference is the reference cell at the last read pass
    out,text=ES.analyze(recs,ref,a.nboot);print(text)
    json.dump({'ref':ref,'summary':{ro:{'cell':o['cell'],'change':o['change']} for ro,o in out.items()},'rule':'CI excludes 0 and |delta| >= 0.2 nats (log-prob metrics) / 0.10 (accuracy-like)'},open(OUT/f'e49_candidates_s{a.seed}{a.tag}.json','w'))
if __name__=='__main__':main()
