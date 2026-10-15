"""Greedy exact match for the L3 probes (name -> place), without generation (rules: PROTOCOL_stage01.md, "Greedy EM").
The place name is the probe's continuation, so greedy decoding reproduces it exactly iff every gold place token is the
arg-max of the full-vocabulary next-token distribution at its position (teacher forcing with the gold prefix equals
greedy decoding while all earlier tokens match). Wrong answers are classified by the arg-max FIRST place token:
partner's place / another pair's place / other. Packed fp32 readouts R1 (neutral tail) and R1-alt (Wikipedia).
Usage: em_eval.py L1ck            -> W0 + saved L1 s101 'ck' epochs 1, 2, 4, 8, 32
       em_eval.py saved LEVEL TAG -> W0 + every state listed in OUT/LEVEL_TAG_s101_{A,B}.saved.json (kept files)"""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
import ladder as LD
from ladder import *
from ttt.model.data import Batch
def main():
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    mode=sys.argv[1];rt=Runtime();L=LD.Ladder(rt,101);OUT=LD.OUT;T=L.T;l3=L.fam['L3'];i0=0
    for f,its in L.fam.items():
        if f=='L3':break
        i0+=len(its)
    gold_rows=[2*(i0+k) for k in range(len(l3))]
    first={t['id']:stage2.enc(T.tok,' '+t['P']).ids[0] for t in T.w['triples']}     # first token of every world place
    tri={t['id']:t for t in T.w['triples']}
    def make_fn(P):
        @eqx.filter_jit
        def am(m,st,ids_,tg_,mk_,pref_,seg_,loc_):
            def one(i,t,k,p,s,l):
                packing.PACK_SUF['seg']=s;packing.PACK_SUF['loc']=l
                try:
                    r=m.language_model.suffix_call(prefix_outputs=p,state=st,seq=Batch(input_ids=i,target_tokens=t,loss_masks=k))
                    return jnp.argmax(r.logits.astype(jnp.float32),-1)
                finally:packing.PACK_SUF.clear()
            return jax.vmap(one)(ids_,tg_,mk_,pref_,seg_,loc_)
        def score(m32):
            rt32=L.rt32
            with rt32.mesh,jax.default_matmul_precision('highest'):
                st=clone_state(rt32.state_suffix)
                for j in range(len(P.T.tail_ids)//CH):st,_=rt32.chunk_logp(m32,st,rt32.chunk(P.T._tail_seq,j),P.T._tail_pref[j*CH:(j+1)*CH])
                out=[]
                for g in range(0,P.ids.shape[0],P.GP):out.append(np.asarray(am(m32,st,P.ids[g:g+P.GP],P.tg[g:g+P.GP],P.mk[g:g+P.GP],P.pref1[g:g+P.GP],P.seg[g:g+P.GP],P.loc[g:g+P.GP])))
            pred=np.concatenate(out);tg=np.asarray(P.tg);rows=[]
            for k,r in enumerate(gold_rows):
                c,a,b=P.where[r];cf,af,bf=P.where[r+1]
                em_gold=bool((pred[c,a:b]==tg[c,a:b]).all());em_partner=bool((pred[cf,af:bf]==tg[cf,af:bf]).all())
                ft=int(pred[c,a]);it=l3[k];me=it['tid'];partner=[t for t in T.w['triples'] if t['pair']==tri[me]['pair'] and t['id']!=me][0]['id']
                cls='self' if ft==first[me] else ('partner' if ft==first[partner] else ('other_pair' if ft in [first[x] for x in first if x not in (me,partner)] else 'other'))
                rows.append({'em_gold':em_gold,'em_partner':em_partner,'first_cls':cls})
            return rows
        return score
    sc={'R1':make_fn(L.P),'alt':make_fn(L.P_alt)};res={}
    if mode=='L1ck':
        SD=OUT/'states'/'L1_ck_s101';pts=[(0,{w:None for w in 'AB'})]+[(ep,{w:SD/w/f'epoch_{ep:03d}.npz' for w in 'AB'}) for ep in (1,2,4,8,32)];name='L1ck'
    else:
        level,tag=sys.argv[2],sys.argv[3];name=f'{level}_{tag}';logs={w:[e for e in json.load(open(OUT/f'{level}_{tag}_s101_{w}.saved.json'))['log'] if 'margins' in e] for w in 'AB'}
        pts=[(e.get('epoch',e.get('state')),{w:Path(logs[w][i]['margins']['_file']) for w in 'AB'}) for i,e in enumerate(logs['A'])]
    for key_,files in pts:
        for w in 'AB':
            m=rt.base if files[w] is None else L.load_state(files[w]);m32=L.to_fp32(m)
            for key,f in sc.items():res[f'{key_}|{w}|{key}']=f(m32)
        print(name,key_,'done',flush=True);json.dump(res,open(OUT/f'em_{name}_s101.json','w'),indent=1)
    md=[f'# Greedy EM, {name}, seed 101 (rules: PROTOCOL_stage01.md)\n','| point | readout | EM A | EM B | **flip-EM** | wrong first token: partner / other pair / other (A) |','|---|---|---|---|---|---|']
    for key_,_ in pts:
        for key in ('alt','R1'):
            A=res[f'{key_}|A|{key}'];B=res[f'{key_}|B|{key}'];ea=np.asarray([r['em_gold'] for r in A]);eb=np.asarray([r['em_partner'] for r in B])
            wr=[r['first_cls'] for r in A if r['first_cls']!='self'];n=max(len(wr),1)
            md.append(f"| {key_} | {key} | {ea.mean():.2f} | {eb.mean():.2f} | **{(ea&eb).mean():.2f}** | {wr.count('partner')/n:.2f} / {wr.count('other_pair')/n:.2f} / {wr.count('other')/n:.2f} (n={len(wr)}) |")
    md.append('\nBaselines: uniform over the 30 world places 1/30 = 0.033 per world; 2-way (pair) level 0.5. EM B = the partner place is generated in world B.')
    (OUT/f'em_{name}_s101.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
