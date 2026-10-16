"""Long-context NLL after a write (PROTOCOL_stage01.md, main-track must-haves, Q2 (formerly M2)). A held-out 32-chunk (32,768-token)
book excerpt is read by native TTT-E2E (inner update on, NLL measured before each chunk's own update) starting from the
written weights instead of W0. Reported: mean NLL over all tokens and over positions >= 8192 (beyond the 8K window;
primary). Usage: longctx_nll.py SEED LEVEL TAG  (states listed in OUT/LEVEL_TAG_sSEED_{A,B}.saved.json, kept files)."""
import os,sys,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
import ladder as LD
from ladder import *
OUT=LD.OUT;NDOC=32;CANDS=('pg2160','pg1399','pg1023')
def doc_ids(rt,tok,name):
    t=(ROOT/f'data/narrative_gutenberg/{name}.txt').read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t
    t=' '.join(t.replace('\r','').split());t=t[len(t)//3:];ids=([rt.bos]+stage2.enc(tok,t[:400000]).ids)[:NDOC*CH];assert len(ids)==NDOC*CH;return ids
def main():
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    seed,level,tag=int(sys.argv[1]),sys.argv[2],sys.argv[3]
    rt=Runtime();L=LD.Ladder(rt,seed,readout=False);tok=L.T.tok;orig=rt.base
    def nll(m,ids):
        rt.base=m
        try:lp,_=rt.stream(ids,True)
        finally:rt.base=orig
        x=-np.asarray(lp[:-1],np.float64);return {'all':float(x.mean()),'beyond':float(x[8192:].mean()),'chunk0':float(x[:CH-1].mean())}
    dj=OUT/'longnll_doc.json'
    if dj.exists():doc=json.load(open(dj))
    else:
        doc=None
        for nm in CANDS:     # registered: first candidate whose W0 beyond-window NLL >= 1.5 (not memorized, unlike pg11)
            r=nll(orig,doc_ids(rt,tok,nm));print('candidate',nm,r,flush=True)
            if r['beyond']>=1.5:doc={'name':nm,'W0':r};break
        assert doc,'no candidate passed the memorization check';json.dump(doc,open(dj,'w'),indent=1)
    ids=doc_ids(rt,tok,doc['name']);w0=nll(orig,ids);assert abs(w0['all']-doc['W0']['all'])<=1e-6,(w0,doc['W0'])
    res={'doc':doc['name'],'W0':w0,'checks':{}}
    for w in 'AB':
        d=json.load(open(OUT/f'{level}_{tag}_s{seed}_{w}.saved.json'));res[w]=[]
        for e in d['log']:
            if 'margins' not in e or '_file' not in e['margins']:continue
            m=L.load_state(e['margins']['_file']);assert reuse._hash_pytree(m.inner_parameters())[0]==e['margins']['_hash']
            r=nll(m,ids);assert rt.base is orig
            if not res['checks'] and e.get('epoch',e.get('state'))!=0:   # state 0 is W0 itself: check on the first written state
                # LC: the stream really starts from the written weights: its first-chunk NLL equals a direct fresh-KV
                # forward of m on chunk 0 and differs from W0's
                with rt.mesh:
                    sq=rt.seq(ids);pf=rt.prefix(orig,sq);_,lp0=rt.chunk_logp(m,clone_state(rt.state_suffix),rt.chunk(sq,0),pf[:CH])
                direct=float(-np.asarray(lp0,np.float64)[:CH-1].mean())
                res['checks']={'direct_chunk0':direct,'stream_chunk0':r['chunk0'],'W0_chunk0':w0['chunk0'],'pass':abs(direct-r['chunk0'])<=1e-4 and abs(r['chunk0']-w0['chunk0'])>1e-6}
                print('check',res['checks'],flush=True);assert res['checks']['pass'],res['checks']
            res[w].append({'epoch':e.get('epoch',e.get('state')),**r,'rise_beyond':r['beyond']-w0['beyond'],'rise_all':r['all']-w0['all']})
            print(w,res[w][-1],flush=True);json.dump(res,open(OUT/f'longnll_{level}_{tag}_s{seed}.json','w'),indent=1)
if __name__=='__main__':main()
