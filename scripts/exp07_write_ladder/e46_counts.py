"""E46 masked-token counts per chunk (CPU only, no GPU; server: needs gen_world, suppress and the tokenizer, like k3_counts.py). For seed(s), worlds A / B, passes 1..P and every chunk with an oracle set: |O|, |OCC| (occupation tokens of the earlier same-triple sentences), |FRM| (the rest of K3), |K3|, the sizes of the cells occ (O+OCC), frm (O+FRM), unif_occ (O + |OCC| uniform), the mask fraction len(S) / (window + chunk key positions), and the check O + OCC + FRM == K3 (assert, per chunk).
Output: <out-dir>/e46_counts.{json,md}.   Usage: e46_counts.py --seeds 101 --passes 4 --worlds AB [--out-dir D]   |   --selftest (synthetic table, plus a stub-tokenizer check of the occupation token extraction in sentence_table)"""
import sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
CH=SETS.CH;W=SETS.W;NCH=63
def counts(sets):
    rows=[]
    for k in range(NCH):
        O,C,F=sets.kindsplit(k)
        if not O:continue
        K3=sets.sentencecover(k);assert sorted(set(O)|set(C)|set(F))==K3 and len(O)+len(C)+len(F)==len(K3),k   # the three kinds partition K3
        k0=k*CH;keys=k0-max(0,k0-W)+CH;oc,fr,un=sets.occmask(k),sets.frmmask(k),sets.unifoccmask(k)
        rows.append({'k':k,'n_O':len(O),'n_OCC':len(C),'n_FRM':len(F),'n_K3':len(K3),'n_occ':len(oc),'n_frm':len(fr),'n_unif_occ':len(un),'keys':keys,'frac_occ':len(oc)/keys,'frac_frm':len(fr)/keys,'frac_K3':len(K3)/keys,'frac_unif_occ':len(un)/keys,'occ_over_K3':len(oc)/len(K3),'frm_over_K3':len(fr)/len(K3)})
    return rows
def summ(rows):
    s={'n_O_chunks':len(rows)}
    for f in ('n_O','n_OCC','n_FRM','n_K3','n_occ','n_frm','n_unif_occ'):s['sum_'+f]=int(sum(r[f] for r in rows))
    for f in ('frac_occ','frac_frm','frac_K3','frac_unif_occ'):s['mean_'+f]=float(np.mean([r[f] for r in rows])) if rows else None
    s['sum_occ_over_K3']=s['sum_n_occ']/s['sum_n_K3'] if rows else None;s['sum_frm_over_K3']=s['sum_n_frm']/s['sum_n_K3'] if rows else None;s['sum_OCC_over_K3_minus_O']=s['sum_n_OCC']/max(1,s['sum_n_K3']-s['sum_n_O']);return s
def md_of(res):
    L=['# E46 masked tokens by kind: O / OCC / FRM (CPU)\n','Fractions = masked tokens / key positions the chunk sees (window before the chunk + the chunk), mean over the chunks with O.\n','| run | O chunks | sum O | sum OCC | sum FRM | sum K3 | sum occ | sum frm | sum unif_occ | mean frac occ | mean frac frm | mean frac K3 | occ / K3 | frm / K3 |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k,v in res['runs'].items():
        s=v['summary'];L.append(f"| {k} | {s['n_O_chunks']} | {s['sum_n_O']} | {s['sum_n_OCC']} | {s['sum_n_FRM']} | {s['sum_n_K3']} | {s['sum_n_occ']} | {s['sum_n_frm']} | {s['sum_n_unif_occ']} | {s['mean_frac_occ']:.3f} | {s['mean_frac_frm']:.3f} | {s['mean_frac_K3']:.3f} | {s['sum_occ_over_K3']:.3f} | {s['sum_frm_over_K3']:.3f} |")
    for k,v in res['runs'].items():
        L+=[f'\n## {k}: per chunk\n','| chunk | O | OCC | FRM | K3 | occ | frm | unif_occ | frac occ | frac K3 |','|---|---|---|---|---|---|---|---|---|---|']+[f"| {r['k']} | {r['n_O']} | {r['n_OCC']} | {r['n_FRM']} | {r['n_K3']} | {r['n_occ']} | {r['n_frm']} | {r['n_unif_occ']} | {r['frac_occ']:.3f} | {r['frac_K3']:.3f} |" for r in v['chunks']]
    return '\n'.join(L)+'\n'
def run(a):
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    res={'runs':{},'args':vars(a)}
    for seed in [int(x) for x in a.seeds.split(',')]:
        for wd in a.worlds:
            for ps in range(1,a.passes+1):
                w=K2.build_world_pass(gw,seed,ps);table=SETS.sentence_table(w,wd=='B',gw.TOK,gw,SP);rows=counts(SETS.Sets(table,seed,ps))
                res['runs'][f's{seed}_{wd}_p{ps}']={'chunks':rows,'summary':summ(rows),'n_sentences':len(table),'n_sent_without_occ':sum(1 for u in table if not u['occ']),'mean_occ_tokens_per_sentence':float(np.mean([len(u['occ']) for u in table]))}
    return res
def stub_sentence_table():
    """sentence_table with a stub tokenizer (whitespace tokens with offsets) and stub SP / gw: checks the occupation token extraction (whole-word match, a_o article excluded, name / place tokens removed, multi-word occupation, the occupation word inside another word not matched)."""
    import re
    class E:pass
    class Tok:
        def encode(self,text,add_special_tokens=False):
            e=E();offs=[(m.start(),m.end()) for m in re.finditer(r'\S+',text)];e.offsets=offs;e.ids=list(range(len(offs)));return e
    texts=['Orsk worked as a smith in Velmar, as his father had before him.','In Velmar the blacksmith was a quiet fellow named Orsk.','The smith of Velmar, Orsk, wiped his hands.','Orsk, a ship builder from Velmar, could mend almost anything.']
    full=' '.join(texts);spans=[];pos=0
    for i,t in enumerate(texts):spans.append({'t':'t0','round':i+1,'text':t,'start':full.index(t)})
    class GW:
        @staticmethod
        def realize(w,swap):return full,{},spans
    class SP_:
        @staticmethod
        def _sent_info(w,swap):
            o=[]
            for i,s in enumerate(spans):
                t=s['text'];X='Orsk';P='Velmar';xs=[m.start() for m in re.finditer(r'\bOrsk\b',t)];p0=t.index(P);o.append(dict(s,idx=i,X=X,P=P,xs=xs,p0=p0))
            return o
        @staticmethod
        def _ptoks(offs,c0,c1,base=0):return [j for j,(a,b) in enumerate(offs) if b>c0 and a<c1]
    for occupation,expect in (('smith',[1,0,1,0]),('ship builder',[0,0,0,1])):
        w={'triples':[{'id':'t0','pair':'p0','o':occupation}]}
        tb=SETS.sentence_table(w,False,Tok(),GW,SP_);assert [len(u['occ']) for u in tb]==[e*len(occupation.split()) for e in expect],(occupation,[len(u['occ']) for u in tb])   # 'blacksmith' is not matched by 'smith'
        for u in tb:assert not (set(u['occ'])&(set(u['names'])|set(u['place']))) and set(u['occ'])<=set(u['toks'])
    return True
def selftest():
    import e46_sets_test as T
    assert stub_sentence_table();rows=counts(SETS.Sets(T.table(),101,2));assert rows and all(r['n_occ']==r['n_O']+r['n_OCC'] and r['n_K3']==r['n_O']+r['n_OCC']+r['n_FRM'] for r in rows);print('e46_counts SELFTEST OK (stub-tokenizer occupation extraction + synthetic counts)',rows[0])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--worlds',default='AB');ap.add_argument('--out-dir',default='/tmp/e46_counts');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    res=run(a);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);json.dump(res,open(od/'e46_counts.json','w'),indent=1);md=md_of(res);(od/'e46_counts.md').write_text(md);print(md)
