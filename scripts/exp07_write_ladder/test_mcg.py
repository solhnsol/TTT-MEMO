"""CPU checks for mcg.py helpers: selection count/ties/eligibility, random selection reproducibility, gold tokens decode."""
import os;os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np,math
import mcg as M
def test():
    g=np.zeros(1024);g[[5,10,700]]=3.0;g[[20,30]]=2.0
    s=M.select(g,'all',0.005);assert len(s)==math.ceil(0.005*1023)==6 and set(s)>={5,10,700,20,30},s
    s=M.select(np.zeros(1024),'all',0.05);assert list(s)==list(range(52)),'ties -> earlier'
    s=M.select(g,'64',0.05);assert min(s)>=64 and len(s)==math.ceil(0.05*959) and 700 in s
    a=M.rand_select('all',0.1,101,7);b=M.rand_select('all',0.1,101,7);c=M.rand_select('all',0.1,101,8);assert (a==b).all() and not (a==c).all() and len(a)==103
    assert M.prf([1,2,3],[2,3,4])==(2,1,1)
    w=M.gw.build_world(101);tok=M.gw.TOK
    for sw in (False,True):
        G=M.gold_tokens(w,sw,tok,0);text,_,_=M.gw.realize(w,sw);ids=[0]+tok.encode(text,add_special_tokens=False).ids
        info={s['idx']:s for s in M.SP._sent_info(w,sw)};pl=[i for i,v in G.items() if v['kind']=='place'];nm=[i for i,v in G.items() if v['kind']=='name']
        ps={t['P'] for t in w['triples']};dec=tok.decode([ids[i+1] for i in pl])
        assert sum(p in dec for p in ps)==len(ps),'every place appears among gold place targets'
        print('world','B' if sw else 'A','place targets',len(pl),'name targets',len(nm),'earlier-in-chunk frac',round(np.mean([v['earlier_in_chunk'] for v in G.values()]),3),'sentences',len(info))
    print('test_mcg OK')
if __name__=='__main__':test()
