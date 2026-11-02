"""E57 public QA stage 11 (E57 122-AA / 123; new file; read only, NO weight update): re-framed questions. The stored states of jobs 640 (NATQ, ALLQ128), 650 (GAINQ512 p.10) and 670 (RANDQ512 p.10) at n = 1 / 3 / 5 / 9 versions (33 paragraph-group episodes) and W0 are read on the items455 with two paraphrased questions per item (qvariants_all.json, sha256 899d62b9...; v1, v2); prompt frame, candidates, OUT context (stage-1 tail), scorer (FastOUTG, packing per group, e57_pqa6_run.py) as in jobs 640 / 650, only the question sentence changes.
  A'v   the item's own variant question v
  B'v   the SAME 16 candidates under the variant question v of ANOTHER item: the partner of job 620 (E57_pqa4_assignB.json: same answer type, different document) when that partner is itself an item of items455 (195 of the 455), otherwise a seeded redraw inside items455 with the same rule (same type, different title; np.random.default_rng(20261104 + item index in the group order)); the share of original / redrawn pairs is recorded and the 'pairs_original' subset is reported separately.
  C     not re-read: the stored C rows of jobs 640 / 650 / 670.
Metrics per arm and n: rise of the gold log-prob over W0 (state minus W0) for A' and B', gain A'-B' per variant and their mean, accuracy and rank ratio of A'; the same table for the original question (A-B of the stored rows) beside it; subsets items455, main373 and pairs_original.
  python3 e57_pqa11_run.py --stage read --out O --src640 D --src650 D --src670 D --qvariants Q --items-file F --bank B --assign A --stage1 S1 [--max-groups N] | --stage analyze --out O --src640 D --src650 D --src670 D | --selftest"""
import sys,json,time,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa4_run as P4
import e57_pqa6_run as P6
CH=1024;NREAD=P6.NREAD;SEEDP=20261104
ARMS=[('NATQ','640'),('ALLQ128','640'),('GAINQ512P10','650'),('RANDQ512P10','670')]
def partners(item_ids,types,titles,assign,seed=SEEDP):
    """-> {item id: (partner id inside items455, original pair flag)}"""
    inn=set(item_ids);out={}
    for k,i in enumerate(item_ids):
        p=assign[i]
        if p in inn and titles[p]!=titles[i] and types[p]==types[i]:out[i]=(p,True);continue
        el=[x for x in item_ids if x!=i and types[x]==types[i] and titles[x]!=titles[i]];assert el,i
        out[i]=(el[int(np.random.default_rng(seed+k).integers(len(el)))],False)
    return out
def write_paths(a):
    return {'640':pathlib.Path(a['src640'])/'e57_pqa6_writes.json','650':pathlib.Path(a['src650'])/'e57_pqa7_writes.json','670':pathlib.Path(a['src670'])/'e57_pqa7_writes.json'}
def read_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True);meta,_=P6.load_groups(a['src640']);wp=write_paths(a);W={j:json.load(open(p)) for j,p in wp.items() if p.exists()}
    arms=[(arm,job) for arm,job in ARMS if job in W];lg('arms read:',[x[0] for x in arms]);G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups']
    LD,rt,rt32,rd,items,h,raw0,ntok0,out_tail,tsha=P2.setup_read(a,lg);import e57_reads as RD
    qv=json.load(open(a['qvariants']));assert hashlib.sha256(open(a['qvariants'],'rb').read()).hexdigest()=='899d62b9b4619d16fcf21e73306ed86ebb8e102db56734f606899ebe9bf58543','qvariants sha256 differs'
    byid={it['id']:it for it in items};asg=json.load(open(a['assign']))['assignment'];ids455=[i for g in meta['groups'] for i in g['item_ids']];types={i:byid[i]['type'] for i in ids455};titles={i:byid[i]['title'] for i in ids455}
    typ_all={i:it['type'] for i,it in byid.items()};tit_all={i:it['title'] for i,it in byid.items()};pr=partners(ids455,typ_all,tit_all,asg);(out/'e57_pqa11_pairs.json').write_text(json.dumps({i:dict(partner=p,original_pair=o) for i,(p,o) in pr.items()}))
    lg('B pairs: original',sum(o for _,o in pr.values()),'redrawn',sum(not o for _,o in pr.values()))
    conds=('A1','B1','A2','B2');grows=[];gnt=[];gsizes=[]
    for g in G:
        its=[byid[i] for i in g['item_ids']];rows=[];nts=[];sz=[]
        for c in conds:
            v=int(c[1])-1;pre=[(qv[it['id']][v] if c[0]=='A' else qv[pr[it['id']][0]][v]).strip() for it in its];it2=[dict(it,prefix=p) for it,p in zip(its,pre)];r_,n_,bad=RD.build_rows(rt.tok,it2);assert not bad,bad;rows+=r_;nts.append(n_);sz.append(len(r_))
        grows.append(rows);gnt.append(nts);gsizes.append(sz)
    m32w0=rd.to_fp32(rt.base);fo=P6.FastOUTG(rt32,m32w0,out_tail,grows,log=lg);labels=['W0']+[f'{arm}@n{n}' for arm,_ in arms for n in NREAD];R={c:{l:[] for l in labels} for c in conds}
    def put(label,gi,m32):
        flat=fo.score_group(m32,gi);its=[byid[i] for i in G[gi]['item_ids']];o=0
        for ci,c in enumerate(conds):
            sub=flat[o:o+gsizes[gi][ci]];o+=gsizes[gi][ci];R[c][label]+=P2.rows_of(its,gnt[gi][ci],sub,f'{c}:{label}')
    t00=time.monotonic()
    for gi,g in enumerate(G):
        t0=time.monotonic();put('W0',gi,m32w0)
        for arm,job in arms:
            for n in NREAD:
                s=W[job][str(g['gid'])][arm]['states'][str(g['read_chunks'][str(n)])];m=rd.load_state(s['file'],expect_hash=s['hash']);put(f'{arm}@n{n}',gi,rd.to_fp32(m));del m
        lg(f'  group {g["gid"]} read: {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
    for c in conds:
        for l in labels:(out/f'E57_pqa11_rows_{c}_{l.replace("@","_")}.json').write_text(json.dumps(R[c][l],default=float))
    (out/'e57_pqa11_read_meta.json').write_text(json.dumps(dict(items_sha256=h,out_tail_sha=tsha,item_ids=[i for g in G for i in g['item_ids']],arms=[x[0] for x in arms],n_pairs_original=int(sum(o for _,o in pr.values())),n_pairs_redrawn=int(sum(not o for _,o in pr.values())),qvariants_sha256='899d62b9b4619d16fcf21e73306ed86ebb8e102db56734f606899ebe9bf58543')))
def rows_orig(a,arm,job,c,n):
    d={'640':(a['src640'],'E57_pqa6'),'650':(a['src650'],'E57_pqa7'),'670':(a['src670'],'E57_pqa7')}[job];return json.load(open(pathlib.Path(d[0])/f'{d[1]}_rows_{c}_{arm}_n{n}.json'))
def analyze(a):
    out=pathlib.Path(a['out']);rm=json.load(open(out/'e57_pqa11_read_meta.json'));ids=rm['item_ids'];meta,_=P6.load_groups(a['src640']);main={i for g in meta['groups'] for i in g['main_ids']};pr=json.load(open(out/'e57_pqa11_pairs.json'))
    gold=lambda R_:np.asarray([r['cand_lp'][0] for r in R_]);rr=lambda R_:np.asarray([r['rank_ratio_mean'] for r in R_]);acc=lambda R_:np.asarray([float(r['acc_mean']) for r in R_]);s6=pathlib.Path(a['src640'])
    masks={'items455':np.ones(len(ids),bool),'main373':np.asarray([i in main for i in ids]),'pairs_original':np.asarray([pr[i]['original_pair'] for i in ids])}
    ldn=lambda c,l:json.load(open(out/f'E57_pqa11_rows_{c}_{l.replace("@","_")}.json'));W0={c:gold(ldn(c,'W0')) for c in ('A1','B1','A2','B2')};W0r={c:rr(ldn(c,'W0')) for c in ('A1','B1','A2','B2')}
    O0={c:gold(json.load(open(s6/f'E57_pqa6_rows_{c}_W0.json'))) for c in 'AB'};O0r={c:rr(json.load(open(s6/f'E57_pqa6_rows_{c}_W0.json'))) for c in 'AB'};res={}
    for arm in rm['arms']:
        job=dict(ARMS)[arm]
        for n in NREAD:
            l=f'{arm}@n{n}';imp={c:gold(ldn(c,l))-W0[c] for c in ('A1','B1','A2','B2')};imr={c:W0r[c]-rr(ldn(c,l)) for c in ('A1','B1','A2','B2')}
            OA,OB=rows_orig(a,arm,job,'A',n),rows_orig(a,arm,job,'B',n);oimp={'A':gold(OA)-O0['A'],'B':gold(OB)-O0['B']};oimr={'A':O0r['A']-rr(OA),'B':O0r['B']-rr(OB)}
            gains={'orig':oimp['A']-oimp['B'],'v1':imp['A1']-imp['B1'],'v2':imp['A2']-imp['B2']};gains['mean_v']=(gains['v1']+gains['v2'])/2
            rgains={'orig':oimr['A']-oimr['B'],'v1':imr['A1']-imr['B1'],'v2':imr['A2']-imr['B2']};rgains['mean_v']=(rgains['v1']+rgains['v2'])/2
            e={}
            for sn,m in masks.items():
                e[sn]=dict(gain_lp_sum={k:P4.boot(v[m]) for k,v in gains.items()},gain_rank_ratio_improvement={k:P4.boot(v[m]) for k,v in rgains.items()},
                    A_rise={'orig':P4.boot(oimp['A'][m]),'v1':P4.boot(imp['A1'][m]),'v2':P4.boot(imp['A2'][m])},B_rise={'orig':P4.boot(oimp['B'][m]),'v1':P4.boot(imp['B1'][m]),'v2':P4.boot(imp['B2'][m])},
                    A_acc_mean={'orig':P4.boot(acc(OA)[m]),'v1':P4.boot(acc(ldn('A1',l))[m]),'v2':P4.boot(acc(ldn('A2',l))[m])},A_rank_ratio={'orig':P4.boot(rr(OA)[m]),'v1':P4.boot(rr(ldn('A1',l))[m]),'v2':P4.boot(rr(ldn('A2',l))[m])})
            res[l]=e
    (out/'E57_pqa11_keys.json').write_text(json.dumps(dict(meta=dict(arms=rm['arms'],n_items=len(ids),pairs_original=rm['n_pairs_original'],pairs_redrawn=rm['n_pairs_redrawn'],gain_definition="(A'(state)-A'(W0)) - (B'(state)-B'(W0)) per variant; orig = the same with the original question rows of jobs 640 / 650 / 670; rank-ratio gain as improvement (W0 minus state)",subsets={k:int(v.sum()) for k,v in masks.items()}),states=res),indent=1,default=float))
def selftest():
    P6.selftest()
    ids=['a','b','c','d','e'];ty={'a':'CAP','b':'CAP','c':'CAP','d':'LOW','e':'LOW'};ti={'a':'X','b':'Y','c':'Z','d':'X','e':'W'};asg={'a':'b','b':'zzz','c':'a','d':'a','e':'zzz'};p=partners(ids,ty,ti,asg)
    assert p['a']==('b',True) and p['b'][1] is False and ty[p['b'][0]]=='CAP' and ti[p['b'][0]]!='Y' and p['d']==('e',False) and p['e']==('d',False)
    assert p==partners(ids,ty,ti,asg)
    print('e57_pqa11_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k);a=dict(out=g('--out'),src640=g('--src640'),src650=g('--src650'),src670=g('--src670'),qvariants=g('--qvariants'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),max_groups=g('--max-groups'))
    {'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
