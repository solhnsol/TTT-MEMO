"""E37 training-distribution damage: per-sample first-chunk NLL of token-id samples under W0 and written states.
Input: uint32 little-endian token ids, consecutive samples of 8193 tokens (DCLM val format, llama3 tokenizer, no document boundaries).
For every sample ONE chunk (default: the first CH=1024 tokens, --chunk k for chunk k) is read with no context (fresh suffix state, prefix from W0 as in
ladder.gen_nll) and its mean NLL over CH-1 targets is stored. W0 is always included. No 8-chunk continuation (unrelated samples are not a continuation).
Usage: e37_firstchunk_nll.py --ids data/dclm_val_subset/val_c0.u32le.bin [--n N] [--chunk 0] --state L0fs_s101_A=OUT/.../state_004.npz ... [--tag e37]
Output: OUT/e37_firstchunk_{tag}.json and .csv (rows = samples, columns = states), plus a W0 repeat read (noise floor, 0 if deterministic)."""
import os,sys,json,argparse,time,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ladder as LD
from ladder import *
OUT=LD.OUT;SAMPLE=8193
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ids',required=True);ap.add_argument('--n',type=int,default=0,help='max samples (0 = all)')
    ap.add_argument('--chunk',type=int,default=0);ap.add_argument('--state',action='append',default=[],help='LABEL=PATH (npz inner-weight state); W0 always included')
    ap.add_argument('--seed',type=int,default=101);ap.add_argument('--tag',default='e37');ap.add_argument('--no-repeat',action='store_true',help='skip the W0 repeat read')
    a=ap.parse_args();assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    raw=np.fromfile(a.ids,dtype='<u4')
    if len(raw)<SAMPLE and len(raw)>=CH:print(f'short file ({len(raw)} tokens): treated as one sample, padded');raw=np.concatenate([raw,np.zeros(SAMPLE-len(raw),raw.dtype)])
    ns=len(raw)//SAMPLE;assert ns>=1,f'{a.ids}: {len(raw)} tokens < one chunk'
    if len(raw)%SAMPLE:print(f'WARNING: {len(raw)%SAMPLE} trailing tokens dropped')
    if a.n:ns=min(ns,a.n)
    S=raw[:ns*SAMPLE].reshape(ns,SAMPLE).astype(np.int64);assert 0<=a.chunk<=(SAMPLE-1)//CH-1;lo=a.chunk*CH
    print('samples',ns,'chunk',a.chunk,'token range',S.min(),S.max(),flush=True)
    rt=Runtime();L=LD.Ladder(rt,a.seed,readout=False);orig=rt.base;vocab=int(getattr(rt.cfg.model,'vocab_size',0) or 0)
    if vocab:assert S.max()<vocab,f'token id {S.max()} >= vocab {vocab}: wrong tokenizer/format?'
    def read(m,x):
        ids=[int(t) for t in x[lo:lo+CH]]
        with rt.mesh:
            sq=rt.seq(ids);pf=rt.prefix(orig,sq);_,lp=rt.chunk_logp(m,clone_state(rt.state_suffix),rt.chunk(sq,0),pf[:CH])
        return float(-np.asarray(lp,np.float64)[:CH-1].mean())
    states=[('W0',None)]+[(s.split('=',1)[0],s.split('=',1)[1]) for s in a.state];res={'ids':a.ids,'n':ns,'chunk':a.chunk,'states':{},'bos_first_token_frac':float((S[:,lo]==rt.bos).mean())};t0=time.monotonic()
    for lab,p in states:
        if p is None:m=orig
        else:
            m=L.load_state(p);hh=reuse._hash_pytree(m.inner_parameters())[0]
        v=[read(m,S[i]) for i in range(ns)];res['states'][lab]={'path':p,'nll':v,'mean':float(np.mean(v)),**({'hash':hh} if p else {})}
        print(lab,f'mean {np.mean(v):.5f}',f'{time.monotonic()-t0:.0f}s',flush=True);del m
        json.dump(res,open(OUT/f'e37_firstchunk_{a.tag}.json','w'))
    if not a.no_repeat:
        v2=[read(orig,S[i]) for i in range(ns)];d=np.abs(np.array(v2)-np.array(res['states']['W0']['nll']));res['W0_repeat']={'nll':v2,'max_abs_diff':float(d.max())};print('W0 repeat max |diff|',d.max())
    json.dump(res,open(OUT/f'e37_firstchunk_{a.tag}.json','w'),indent=0)
    with open(OUT/f'e37_firstchunk_{a.tag}.csv','w',newline='') as f:
        w=csv.writer(f);labs=list(res['states']);w.writerow(['sample']+labs+(['W0_repeat'] if 'W0_repeat' in res else []))
        for i in range(ns):w.writerow([i]+[res['states'][l]['nll'][i] for l in labs]+([res['W0_repeat']['nll'][i]] if 'W0_repeat' in res else []))
    print('wrote',OUT/f'e37_firstchunk_{a.tag}.json')
if __name__=='__main__':main()
