"""W0 error check for the "complement" cell definition (design 14:05 KST 10/18; Hans "Checks first"): W0 only, pass-1 original stream, no update, no D.
Place-token error (1 - p(y)) of every exposure's place-token pieces under four forwards of the same chunk (the natural chain state at W0):
  native        the natural read (carried KV)
  oracle        the S3 masked forward with S = O (the earlier same-fact exposure name / place tokens masked)
  Oonly         "O only visible": S = every window position before the chunk except the O tokens (the complement cell's mask)
  prefixall     S = every window position before the chunk
Each place-token piece is labelled with the interval of the previous same-triple exposure (definition T: same chunk / other chunk inside 8K / outside 8K / none) as in tokerr.
Rule registered before the results: in the "other chunk inside 8K" bin, the ratio mean(1-p)[Oonly] / mean(1-p)[native] within 1 +- 0.1 -> the complement cell definition is OK; close to prefixall -> replace it by the sentence-level split (spec line 74: keep the whole sentences that contain the O spans).
Usage: w0err.py --world-seed 101 [--worlds AB]  ->  OUT/w0err_s{seed}_{w}.json ; w0err.py --analyze"""
import os,sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_stage1 as ST
from s3_stage1 import *
class W0Err(ST.Stage1):
    def run_w0(self,swap,seed):
        rt=self.rt;m=rt.base;ids=np.asarray(self.T.P[swap].ids)[:NCH*CH];W=WIN;mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask)
        tab=SETS.sentence_table(self.T.w,swap,self.T.tok,gw,SP);sets=SETS.Sets(tab,seed,1);recs=[];t0=time.monotonic()
        interval={}
        for s in tab:
            k=s['place'][0]//CH;k0=k*CH;prev=[u for u in tab if u['t']==s['t'] and u['place'][0]<s['place'][0]]
            if not prev:interval[s['idx']]='none'
            else:
                p=max(u['place'][0] for u in prev);interval[s['idx']]='same' if p//CH==k else ('in8k' if p>=k0-W else 'out8k')
        bychunk={}
        for s in tab:
            for j in s['place']:
                k=j//CH;row=j-1-k*CH
                if row>=0:bychunk.setdefault(k,[]).append((s['idx'],s['pair'],j,row))
        with rt.mesh:
            pref_nat=rt.prefix(rt.base,seq);Kh,Vh,_=self.record(ids);print(f'  w0err setup {time.monotonic()-t0:.0f}s',flush=True);st=clone_state(rt.state_suffix)
            for k in range(NCH):
                c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH]
                if k in bychunk:
                    items=bychunk[k];k0=k*CH;rows=jnp.asarray([r for *_,r in items],dtype=jnp.int32);ys=np.asarray([ids[j] for _,_,j,_ in items]);Kc,Vc=self.ctx_kv(Kh,Vh,k)
                    O=set(sets.oracle(k));allpos=list(range(max(0,k0-W),k0));Oonly=[p for p in allpos if p not in O]
                    def pyv(S):
                        keep=sets.keep_vector(k,S);pw=self.prefix_masked(rt.base,jnp.asarray(ids[k0:k0+CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k0),jnp.asarray(keep))
                        p=np.asarray(self.logits_masked(rt.base,st,c,pw,jnp.asarray(keep),rows));return p[np.arange(len(ys)),ys]
                    pn=np.asarray(self.logits_nat(rt.base,st,c,pref_c,rows));pn=pn[np.arange(len(ys)),ys]
                    po=pyv(sorted(O)) if O else pn;pc=pyv(Oonly);pa=pyv(allpos)
                    for i,(idx,pair,j,row) in enumerate(items):
                        recs.append(dict(idx=int(idx),pair=pair,j=int(j),k=int(k),interval=interval[idx],has_O=bool(O),p_native=float(pn[i]),p_oracle=float(po[i]),p_Oonly=float(pc[i]),p_prefixall=float(pa[i])))
                st=self.read_state(m,st,c,pref_c)
                if k%8==0:print(f'  w0err chunk {k}/{NCH} {time.monotonic()-t0:.0f}s',flush=True)
        return {'seed':seed,'swap':bool(swap),'records':recs}
def analyze():
    rec=[]
    for sd in (101,102,103):
        for w in 'AB':
            f=OUT/f'w0err_s{sd}_{w}.json'
            if f.exists():
                for r in json.load(open(f))['records']:r['cl']=(sd,w,r['pair']);rec.append(r)
    conds=['native','oracle','Oonly','prefixall'];out={}
    rng=np.random.default_rng(0)
    def stat(rs,iv):
        sel=[r for r in rs if r['interval'] in iv];return {c:np.mean([1-r['p_'+('native' if c=='native' else c)] for r in sel]) for c in conds} if sel else None
    cls={};
    for r in rec:cls.setdefault(r['cl'],[]).append(r)
    keys=sorted(cls)
    for name,iv in (('in8k',['in8k']),('same',['same']),('out8k+none',['out8k','none'])):
        s=stat(rec,iv);rat={c:s[c]/s['native'] for c in conds}
        xs={c:[] for c in conds}
        for _ in range(4000):
            idx=rng.integers(0,len(keys),len(keys));rs=[r for i in idx for r in cls[keys[i]]];b=stat(rs,iv)
            if b:
                for c in conds:xs[c].append(b[c]/b['native'])
        n=len([r for r in rec if r['interval'] in iv]);out[name]=dict(n_pieces=n,mean_err=s,ratio_to_native=rat,ci={c:[float(np.percentile(xs[c],2.5)),float(np.percentile(xs[c],97.5))] for c in conds})
        print(f'bin {name} ({n} place-token pieces): mean (1-p): '+', '.join(f'{c} {s[c]:.4f}' for c in conds)+' | ratio to native: '+', '.join(f'{c} {rat[c]:.3f} [{out[name]["ci"][c][0]:.3f},{out[name]["ci"][c][1]:.3f}]' for c in conds))
    r_=out['in8k']['ratio_to_native']['Oonly'];print(f'rule: in8k ratio(Oonly/native) = {r_:.3f}: '+('within 1 +- 0.1 -> the complement definition is OK' if abs(r_-1)<=0.1 else ('closer to prefix-all-pos (%.3f)'%out['in8k']['ratio_to_native']['prefixall']+' -> use the sentence-level split' if abs(r_-out['in8k']['ratio_to_native']['prefixall'])<abs(r_-1) else 'outside 1 +- 0.1 and not closest to prefix-all-pos: report and ask')))
    json.dump(out,open(OUT/'w0err_result.json','w'),indent=1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--world-seed',type=int,default=101);ap.add_argument('--worlds',default='AB');ap.add_argument('--analyze',action='store_true');a=ap.parse_args()
    if a.analyze:analyze();return
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    rt=Runtime();L=W0Err(rt,a.world_seed,readout=False);L.build3();L.build1()
    for w in a.worlds:
        f=OUT/f'w0err_s{a.world_seed}_{w}.json'
        if f.exists():print('exists',f);continue
        t0=time.monotonic();res=L.run_w0(w=='B',a.world_seed);json.dump(res,open(f,'w'));print('saved',f,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
