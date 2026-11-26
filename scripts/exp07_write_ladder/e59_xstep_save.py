"""Cross one-step, STAGE 1 (job 1358a 'xstep_save'; GPU; NEW file, wrapper of e59_kvmulti.py which stays UNCHANGED; implementation thread 11/26; design: Hans 11/26 18:12 KST, E59 [S2617-XSTEP-OK]; EXPLORATORY). Re-runs the two 1348 streams P0s (SAMEVIS) and P1s (C0) with the SAME code path (e59_kvmulti.run, same args, same read points 8/16/32/64/96/128, --save-final) and additionally stores, without touching the computation:
  - the weights W_t after t = 32 / 64 / 96 exposures (state files) and the suffix-layer carried state S_t that enters chunk t (captured from the engine's read_state call of chunk t-1; the prefix-layer window K,V are the natural W0 K,V of the stream, recorded identically in every run, so they are not stored);
  - the weight hashes after 1 / 33 / 65 / 97 exposures (for the stage-2 check "one step from W_t with the stream's own mask = stream weights after t+1").
There is no optimizer state (plain SGD with a global clip; the ilr state is copied once per run and never changes).
After the run it compares with job 1348: R1 scores dS1 at every read point and R2 at 128 (max abs difference) and the final-weight hash; result in xstep_save_s<seed>.json ('same_as_1348').
  python3 e59_xstep_save.py --seed 231 --out O [--nch 128] [--base-dir experiments/exp07_write_ladder/kvmulti] | --selftest"""
import sys,os,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
SAVE_T=(32,64,96);HASH_T=(1,33,65,97)
def extra_chunks(nch):return sorted({t-1 for t in SAVE_T+HASH_T if t<=nch})
def save_suffix_state(st,fn):
    import jax;leaves=[np.asarray(x) for x in jax.tree_util.tree_leaves(st)];pathlib.Path(fn).parent.mkdir(parents=True,exist_ok=True);np.savez(fn,*leaves);return dict(_file=str(fn),n_leaves=len(leaves),shapes=[list(x.shape) for x in leaves],dtypes=[str(x.dtype) for x in leaves])
def load_suffix_state(fn,template):
    """template = s3.clone_state(rt.state_suffix): the tree structure"""
    import jax,jax.numpy as jnp;z=np.load(fn);leaves=[jnp.asarray(z[f'arr_{i}']) for i in range(len(z.files))];td=jax.tree_util.tree_structure(template);assert td.num_leaves==len(leaves),(td.num_leaves,len(leaves));return jax.tree_util.tree_unflatten(td,leaves)
def install(seed,out,SIDE,log=print):
    """wrap e55_iso.Iso55 so that every stream run (a) records the output of read_state and (b) gets extra hook chunks where it saves W_t / S_t / hashes; the original read hook is called only at its own chunks, and inline reads of the extra chunks are dropped"""
    import e55_iso as ISO,s3
    orig=ISO.Iso55.__init__
    def init(self,*a,**k):
        orig(self,*a,**k);rs=self.read_state;self._xs_last=None
        def rs_w(*aa,**kk):r=rs(*aa,**kk);self._xs_last=r;return r
        self.read_state=rs_w;orun=self.run_stream
        def run_w(*aa,**kk):
            hook=kk['read_hook'];hc=set(kk['hook_chunks']);arm=kk['tag'].split('_',1)[1];nch=kk['nch'];ex=extra_chunks(nch);rec=SIDE.setdefault(arm,{})
            def hk(k,m):
                t=k+1;d={}
                if t in SAVE_T:
                    d['W']=self.save_state(m,name=f'W{t}_{arm}_s{seed}.npz');d['S']=save_suffix_state(self._xs_last,out/'states'/f's{seed}_{arm}'/f'S{t}_{arm}_s{seed}.npz');log(f'  xstep: saved W_{t}, S_{t} of {arm}')
                if t in HASH_T or t in SAVE_T:d['hash']=s3.reuse._hash_pytree(m.inner_parameters())[0]
                if d:rec[str(t)]=d
                return hook(k,m) if k in hc else None
            kk['hook_chunks']=sorted(hc|set(ex));kk['read_hook']=hk;r=orun(*aa,**kk);r['inline']={k:v for k,v in r.get('inline',{}).items() if k in hc};return r
        self.run_stream=run_w
    ISO.Iso55.__init__=init
def compare(new,old):
    out={}
    for arm in ('P0s','P1s'):
        a,b=new['arms'][arm],old['arms'][arm];d={}
        for e in sorted(set(a['reads'])&set(b['reads']),key=int):
            d[f'dS1_{e}']=float(np.abs(np.asarray(a['reads'][e]['dS1'])-np.asarray(b['reads'][e]['dS1'])).max())
            if 'dS2' in a['reads'][e] and 'dS2' in b['reads'][e]:d[f'dS2_{e}']=float(np.abs(np.asarray(a['reads'][e]['dS2'])-np.asarray(b['reads'][e]['dS2'])).max())
        fa=a['reads'].get('128',{}).get('final_state',{}).get('_hash');fb=b['reads'].get('128',{}).get('final_state',{}).get('_hash');d['final_hash_equal']=(fa is not None and fa==fb)
        d['gnorm_maxabs']=float(np.abs(np.asarray([x['gnorm'] for x in a['steps']])-np.asarray([x['gnorm'] for x in b['steps']])).max()) if len(a['steps'])==len(b['steps']) else None
        out[arm]=d
    out['same']=all(out[x]['final_hash_equal'] and max(v for k,v in out[x].items() if k.startswith('dS'))==0.0 for x in ('P0s','P1s'));return out
def selftest():
    assert extra_chunks(128)==[0,31,32,63,64,95,96] and extra_chunks(16)==[0]
    import tempfile;d=tempfile.mkdtemp();st={'k':np.arange(6,dtype=np.float32).reshape(2,3),'v':np.ones((2,2),np.float16)};info=save_suffix_state(st,d+'/s.npz');r=load_suffix_state(d+'/s.npz',st);assert info['n_leaves']==2 and np.array_equal(np.asarray(r['k']),st['k']) and np.asarray(r['v']).dtype==np.float16
    A={'arms':{x:{'reads':{'8':{'dS1':[[1.0]]},'128':{'dS1':[[2.0]],'dS2':[[3.0]],'final_state':{'_hash':'h'+x}}},'steps':[{'gnorm':1.0}]} for x in ('P0s','P1s')}};assert compare(A,A)['same']
    B=json.loads(json.dumps(A));B['arms']['P1s']['reads']['8']['dS1']=[[1.5]];c=compare(B,A);assert not c['same'] and c['P1s']['dS1_8']==0.5
    print('e59_xstep_save selftest ok: save chunks, suffix-state save/load, 1348 comparison')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--nch',type=int,default=128);ap_.add_argument('--expo',default='8,16,32,64,96,128');ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--base-dir',default='experiments/exp07_write_ladder/kvmulti');a=ap_.parse_args()
    if a.selftest:selftest()
    else:
        import e59_kvmulti as KV
        out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);SIDE={};install(a.seed,out,SIDE,lambda *x:print(*x,flush=True))
        ka=argparse.Namespace(seed=a.seed,out=str(out),nch=a.nch,expo=a.expo,save_final=True,selftest=False);KV.run(ka)
        new=json.load(open(out/f'kvmulti_s{a.seed}.json'));res=dict(seed=a.seed,saved=SIDE,args=vars(a))
        ob=pathlib.Path(a.base_dir)/f'kvmulti_s{a.seed}.json'
        if a.nch==128 and ob.exists():res['same_as_1348']=compare(new,json.load(open(ob)));print('compare with 1348:',json.dumps(res['same_as_1348']),flush=True)
        json.dump(res,open(out/f'xstep_save_s{a.seed}.json','w'));print('wrote',out/f'xstep_save_s{a.seed}.json',flush=True)
