"""E47 masked-token counts per chunk (CPU only, no GPU; server: needs gen_world, suppress and the tokenizer, like e46_counts.py). For seed(s), worlds A / B, passes 1..P and every chunk with an oracle set: |O|, |K3|, |P|, |X| (E44 type classification outside K3), the sizes of kp (K3+P), kx (K3+X), kp_u, kx_u (K3 + |P| / |X| uniform), the mask fractions len(S) / (window + chunk key positions), the ratios to K3, and the X sentence frame statistics (X tokens / sentences with the same frame id as a fact sentence of the chunk, share; mean number of shared words between an X sentence and the fact sentence(s)). Asserts per chunk: K3, P, X pairwise disjoint and SENT == K3 + P + X + later same-triple window tokens.
Output: <out-dir>/e47_counts.{json,md}.   Usage: e47_counts.py --seeds 101 --passes 4 --worlds AB [--out-dir D]   |   --selftest"""
import sys,json,argparse,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
CH=SETS.CH;W=SETS.W;NCH=63
def _words(t):return set(re.findall(r"[a-z']+",t.lower()))
def counts(sets):
    rows=[]
    for k in range(NCH):
        O=sets.oracle(k)
        if not O:continue
        K3=set(sets.sentencecover(k));P,X=sets.pxsplit(k);Xs,Xo=sets.xframe_split(k);k0=k*CH;lo=max(0,k0-W);keys=k0-lo+CH
        cur=sets.cur_sentences(k);ts={s['t'] for s in cur};Bw={p for u in sets.t if u['t'] in ts for p in u['toks'] if lo<=p<k0}
        assert not(set(P)&set(X)) and not(set(P)&K3) and not(set(X)&K3),k;assert set(sets.sentmask(k))==K3|set(P)|set(X)|Bw,k
        kp,kx,kpu,kxu=sets.kpmask(k),sets.kxmask(k),sets.kpumask(k),sets.kxumask(k);ps={s['pair'] for s in cur};fr={s.get('frame',-1) for s in cur}-{-1}
        xs_=[u for u in sets.t if u['t'] not in ts and u['pair'] not in ps and any(lo<=p<k0 for p in u['toks'])];same=[u for u in xs_ if u.get('frame',-1) in fr]
        cw=set().union(*[_words(s['text']) for s in cur]) if cur and all(s.get('text') for s in cur) else None
        ov=float(np.mean([len(_words(u['text'])&cw) for u in xs_])) if cw is not None and xs_ and all(u.get('text') for u in xs_) else None
        rows.append({'k':k,'n_O':len(O),'n_K3':len(K3),'n_P':len(P),'n_X':len(X),'n_X_same_frame':len(Xs),'n_X_other_frame':len(Xo),'n_X_sent':len(xs_),'n_X_sent_same_frame':len(same),'frac_X_sent_same_frame':len(same)/len(xs_) if xs_ else None,'x_mean_shared_words':ov,
                     'n_kp':len(kp),'n_kx':len(kx),'n_kp_u':len(kpu),'n_kx_u':len(kxu),'keys':keys,'frac_K3':len(K3)/keys,'frac_kp':len(kp)/keys,'frac_kx':len(kx)/keys,'kp_over_K3':len(kp)/len(K3),'kx_over_K3':len(kx)/len(K3)})
    return rows
def summ(rows):
    s={'n_O_chunks':len(rows)}
    for f in ('n_O','n_K3','n_P','n_X','n_X_same_frame','n_X_sent','n_X_sent_same_frame','n_kp','n_kx','n_kp_u','n_kx_u'):s['sum_'+f]=int(sum(r[f] for r in rows))
    for f in ('frac_K3','frac_kp','frac_kx'):s['mean_'+f]=float(np.mean([r[f] for r in rows])) if rows else None
    s['sum_kp_over_K3']=s['sum_n_kp']/max(1,s['sum_n_K3']);s['sum_kx_over_K3']=s['sum_n_kx']/max(1,s['sum_n_K3']);s['share_X_tokens_same_frame']=s['sum_n_X_same_frame']/max(1,s['sum_n_X']);s['share_X_sent_same_frame']=s['sum_n_X_sent_same_frame']/max(1,s['sum_n_X_sent']);return s
def md_of(res):
    L=['# E47 masked tokens beyond K3: P (same pair, other person) and X (other pairs) (CPU)\n','Fractions = masked tokens / key positions the chunk sees (window before the chunk + the chunk), mean over the chunks with O. Same frame = the X sentence uses the same sentence frame id as a fact sentence of the chunk (frame id -1 = unidentified counts as other).\n','| run | O chunks | sum O | sum K3 | sum P | sum X | sum kp | sum kx | sum kp_u | sum kx_u | mean frac K3 | mean frac kp | mean frac kx | kp / K3 | kx / K3 | X tokens same frame | X sentences same frame |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k,v in res['runs'].items():
        s=v['summary'];L.append(f"| {k} | {s['n_O_chunks']} | {s['sum_n_O']} | {s['sum_n_K3']} | {s['sum_n_P']} | {s['sum_n_X']} | {s['sum_n_kp']} | {s['sum_n_kx']} | {s['sum_n_kp_u']} | {s['sum_n_kx_u']} | {s['mean_frac_K3']:.3f} | {s['mean_frac_kp']:.3f} | {s['mean_frac_kx']:.3f} | {s['sum_kp_over_K3']:.2f} | {s['sum_kx_over_K3']:.2f} | {s['share_X_tokens_same_frame']:.3f} | {s['share_X_sent_same_frame']:.3f} |")
    for k,v in res['runs'].items():
        L+=[f'\n## {k}: per chunk\n','| chunk | O | K3 | P | X | X same frame | X sent (same frame) | kp | kx | kp_u | kx_u | mean shared words X vs fact |','|---|---|---|---|---|---|---|---|---|---|---|---|']+[f"| {r['k']} | {r['n_O']} | {r['n_K3']} | {r['n_P']} | {r['n_X']} | {r['n_X_same_frame']} | {r['n_X_sent']} ({r['n_X_sent_same_frame']}) | {r['n_kp']} | {r['n_kx']} | {r['n_kp_u']} | {r['n_kx_u']} | {'' if r['x_mean_shared_words'] is None else format(r['x_mean_shared_words'],'.1f')} |" for r in v['chunks']]
    return '\n'.join(L)+'\n'
def run(a):
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    res={'runs':{},'args':vars(a)}
    for seed in [int(x) for x in a.seeds.split(',')]:
        for wd in a.worlds:
            for ps in range(1,a.passes+1):
                w=K2.build_world_pass(gw,seed,ps);table=SETS.sentence_table(w,wd=='B',gw.TOK,gw,SP);rows=counts(SETS.Sets(table,seed,ps))
                res['runs'][f's{seed}_{wd}_p{ps}']={'chunks':rows,'summary':summ(rows),'n_sentences':len(table),'n_sent_frame_unknown':sum(1 for u in table if u['frame']<0)}
    return res
def stub_frame():
    """frame_of with a stub gen_world (FR templates, A) and stub tokenizer: the frame id of each exposure sentence is the index of the template that reproduces its text."""
    FR=['{X}, the {o} of {P}, had never left.','{X} worked as {a_o} in {P}.','The {o} {X} came down from {P}.'];A=lambda o:('an ' if o[0] in 'aeiou' else 'a ')+o
    tri={'id':0,'P':'Velmar','o':'smith','pair':'p0','X':'Orsk'};texts=[{'kind':'bg','text':'filler'}]+[{'kind':'expo','t':0,'round':r+1,'text':FR[j].format(X='@X0@',P='Velmar',o='smith',a_o=A('smith'))} for r,j in enumerate((2,0,1,1))]
    real=[x['text'].replace('@X0@','Orsk') for x in texts if x['kind']=='expo'];full=' '.join(real);spans=[];pos=0
    for i,t in enumerate(real):
        st=full.index(t,pos);spans.append({'t':0,'round':i+1,'text':t,'start':st});pos=st+len(t)
    class GWr:
        @staticmethod
        def realize(w,swap):return full,{},spans
    GWr.FR=FR;GWr.A=staticmethod(A)
    class Tok:
        def encode(self,text,add_special_tokens=False):
            class E:pass
            e=E();e.offsets=[(m.start(),m.end()) for m in re.finditer(r'\S+',text)];e.ids=list(range(len(e.offsets)));return e
    class SP_:
        @staticmethod
        def _sent_info(w,swap):
            o=[]
            for i,s in enumerate(spans):
                t=s['text'];o.append(dict(s,idx=i,X='Orsk',P='Velmar',xs=[m.start() for m in re.finditer(r'\bOrsk\b',t)],p0=t.index('Velmar')))
            return o
        @staticmethod
        def _ptoks(offs,c0,c1,base=0):return [j for j,(a,b) in enumerate(offs) if b>c0 and a<c1]
    tb=SETS.sentence_table({'triples':[tri],'texts':texts},False,Tok(),GWr,SP_);assert [u['frame'] for u in tb]==[2,0,1,1],[u['frame'] for u in tb]
    return True
def selftest():
    import e47_sets_test as T
    assert stub_frame();rows=counts(SETS.Sets(T.table(),101,2));assert rows and all(r['n_kp']==r['n_K3']+r['n_P'] and r['n_kx']==r['n_K3']+r['n_X'] and r['n_X']==r['n_X_same_frame']+r['n_X_other_frame'] for r in rows);print('e47_counts SELFTEST OK (stub frame identification + synthetic counts)',rows[0])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--worlds',default='AB');ap.add_argument('--out-dir',default='/tmp/e47_counts');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    res=run(a);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);json.dump(res,open(od/'e47_counts.json','w'),indent=1);md=md_of(res);(od/'e47_counts.md').write_text(md);print(md)
