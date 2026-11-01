"""E57 public QA stage 4 (E57 122-S / 123; new file; GPU read only, NO weight update): the binding-gain control read of the states written by job 610 (e57_pqa3_run.py; NAT / NATQ / ALLQ128 at the end of the six sections of the accumulating re-description stream) and of W0, on the same 1,000 frozen stage-1 items x 16 candidates, OUT context = the stage-1 8192-token tail, FastOUT with the 640-row packing of stage 1 (e57_pqa2_run.FastOUT).
Conditions (the candidates of an item, gold first, are always its own 16):
  A  the item's own question (the rows of job 610; here only re-computed for W0 and the three final states as a check against the stored 610 rows)
  B  the same 16 candidates scored under the question of ANOTHER item: same answer type (CAP / NUM / DATE / LOW), a different document (title), one question per item drawn with np.random.default_rng(SEED) from the eligible items (a derangement-like assignment, recorded in the output)
  C  no question: the frame "Answer:" only ("\n\nAnswer: <candidate>")
Counts: how often the gold string and each of the 16 candidates occur in the stream text, in total and per section (filler / orig / rw1..rw4; case-sensitive substring count 'cnt' and count between non-alphanumeric characters 'cnt_wb'), written into the item rows.
Binding gain (analyze): per item and state, gain = (metric A(state) - metric A(W0)) - (metric B(state) - metric B(W0)) for the gold log-prob (sum and per-token mean) and for the rank-ratio improvement (rank ratio of W0 minus rank ratio of the state, positive = better); mean over the items of a subset (items455 / main373 / all1000) with the bootstrap 95% CI over items.
  python3 e57_pqa4_run.py --stage read --cond B|C|A --out O --manifest M --states610 DIR610 --items-file F --bank B [--stage1 S1] [--paragraphs P] | --stage analyze --out O --dir610 D --stream-dir SD --paragraphs P --items-file F | --selftest"""
import sys,json,time,hashlib,pathlib,re
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa3_run as P3
import e57_squad_w0_reads as Q
SEED=20261101;SECS=P3.SECS;ARMS=['NATQ','ALLQ128','NAT']
def assign_B(items,seed=SEED):
    """item id -> id of the item whose question is used: same type, different title, one per item (seeded)"""
    rng=np.random.default_rng(seed);by={}
    for it in items:by.setdefault(it['type'],[]).append(it)
    out={}
    for t,L in sorted(by.items()):
        for it in L:
            el=[x for x in L if x['title']!=it['title']];assert el,('no eligible question',t,it['id']);out[it['id']]=el[int(rng.integers(len(el)))]['id']
    return out
def prefix_of(items,cond,asg):
    q={it['id']:it['prefix'] for it in items}
    if cond=='A':return [it['prefix'] for it in items]
    if cond=='B':return [q[asg[it['id']]] for it in items]
    if cond=='C':return ['Answer:' for it in items]
    raise ValueError(cond)
def order_states(man):
    """final states first (end of the last section), then the others; W0 is read first"""
    names=list(man);fin=[n for n in names if man[n]['section']=='rw4'];return fin+[n for n in names if n not in fin]
def read_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True);cond=a['cond']
    LD,rt,rt32,rd,items,h,raw0,ntok0,out_tail,tsha=P2.setup_read(a,lg);import e57_reads as RD
    asg=assign_B(items);(out/'E57_pqa4_assignB.json').write_text(json.dumps(dict(rule='same answer type, different title, one question per item; np.random.default_rng(%d), first draw over the sorted types and the items in file order'%SEED,seed=SEED,assignment=asg)))
    pre=prefix_of(items,cond,asg);items2=[dict(it,prefix=p) for it,p in zip(items,pre)];raw,ntok,bad=RD.build_rows(rt.tok,items2);assert not bad,bad
    man=json.load(open(a['manifest']));names=order_states(man)
    if cond=='A':names=[n for n in names if man[n]['section']=='rw4']
    m32w0=rd.to_fp32(rt.base);t0=time.monotonic();fo=P2.FastOUT(rt32,m32w0,out_tail,raw,log=lg);tim={'prep':time.monotonic()-t0};lg(f'cond {cond}: first frames {pre[:2]}')
    def save(lab,rows):(out/f'E57_pqa4_rows_{cond}_{lab.replace("@","_")}.json').write_text(json.dumps(rows,default=float))
    t0=time.monotonic();rw0=P2.rows_of(items2,ntok,fo.score(m32w0),f'{cond}:W0');tim['W0']=time.monotonic()-t0;save('W0',rw0);lg(f'  W0 read: {tim["W0"]:.0f}s gold lp mean {np.mean([r["cand_lp"][0] for r in rw0]):.3f}')
    chk={}
    if cond=='A':
        d610=pathlib.Path(a['dir610']);w=json.load(open(d610/'E57_pqa3_rows_W0.json'));chk['W0']=float(max(np.abs(np.asarray(x['cand_lp'])-np.asarray(y['cand_lp'])).max() for x,y in zip(rw0,w)))
    for n in names:
        s=man[n];t0=time.monotonic();m=rd.load_state(s['file'],expect_hash=s['hash']);rows=P2.rows_of(items2,ntok,fo.score(rd.to_fp32(m)),f'{cond}:{n}');del m;tim[n]=time.monotonic()-t0;save(n,rows)
        if cond=='A':
            w=json.load(open(pathlib.Path(a['dir610'])/f'E57_pqa3_rows_{n.replace("@","_")}.json'));chk[n]=float(max(np.abs(np.asarray(x['cand_lp'])-np.asarray(y['cand_lp'])).max() for x,y in zip(rows,w)))
        lg(f'  {n} read: {tim[n]:.0f}s gold lp mean {np.mean([r["cand_lp"][0] for r in rows]):.3f}'+(f' max|diff to 610| {chk[n]:.2e}' if cond=='A' else ''))
    (out/f'E57_pqa4_meta_{cond}.json').write_text(json.dumps(dict(cond=cond,items_sha256=h,out_tail_sha=tsha,timings=tim,check_vs_610=chk,states=man,frame_rule={'A':'own question','B':'question of another item (assignB)','C':'"Answer:"'}[cond]),default=float))
def count_strings(stream_dir):
    """-> (counter(s) -> {'total':..,'sec':[..6]}, wb) for the strings asked later; texts per section from the jsonl"""
    txt={s:[] for s in SECS+['tail']}
    for l in open(pathlib.Path(stream_dir)/'streamP3_stream_sentences.jsonl'):
        r=json.loads(l)
        if r['type']=='frag':txt[r['section']].append(r['text'])
    return {k:''.join(v) for k,v in txt.items()}
def counts_of(texts,strings):
    out={}
    for s in sorted(set(strings)):
        rx=re.compile(r'(?<![A-Za-z0-9])'+re.escape(s)+r'(?![A-Za-z0-9])') if s else None
        out[s]=dict(cnt=[texts[k].count(s) for k in SECS],cnt_wb=[len(rx.findall(texts[k])) for k in SECS])
    return out
def boot(x,seed=3,B=2000):
    x=np.asarray(x,np.float64);ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))])
def metric_vec(rows,key):
    if key=='lp_sum':return np.asarray([r['cand_lp'][0] for r in rows])
    if key=='lp_mean':return np.asarray([r['cand_mean'][0] for r in rows])
    if key=='rr_sum':return np.asarray([r['rank_ratio_sum'] for r in rows])
    if key=='rr_mean':return np.asarray([r['rank_ratio_mean'] for r in rows])
    raise KeyError(key)
def gain_table(A,B,A0,B0,sel):
    """A, B, A0, B0: row lists in item order; sel: bool mask -> {metric: gain stats}; rank-ratio metrics are 'improvement' (W0 minus state)"""
    out={}
    for k in ('lp_sum','lp_mean','rr_sum','rr_mean'):
        sgn=1.0 if k.startswith('lp') else -1.0
        impA=sgn*(metric_vec(A,k)-metric_vec(A0,k));impB=sgn*(metric_vec(B,k)-metric_vec(B0,k));out[k]=dict(A=boot(impA[sel]),B=boot(impB[sel]),gain=boot((impA-impB)[sel]))
    return out
def analyze(a):
    out=pathlib.Path(a['out']);d610=pathlib.Path(a['dir610']);items,h=Q.load_items_file(a['items']);paras=json.load(open(a['paragraphs']));sub=P3.subsets(paras);man=json.load(open(a['manifest']))
    ld=lambda cond,lab:json.load(open((d610 if cond=='A' else out)/(f'E57_pqa3_rows_{lab.replace("@","_")}.json' if cond=='A' else f'E57_pqa4_rows_{cond}_{lab.replace("@","_")}.json')))
    ids=[it['id'] for it in items];masks={'all1000':np.ones(len(ids),bool),'items455':np.asarray([i in sub['items455'] for i in ids]),'main373':np.asarray([i in sub['main373'] for i in ids])}
    A0,B0,C0=ld('A','W0'),ld('B','W0'),ld('C','W0');res={}
    for n in man:
        if not (out/f'E57_pqa4_rows_B_{n.replace("@","_")}.json').exists():continue
        A=ld('A',n);B=ld('B',n);C=ld('C',n) if (out/f'E57_pqa4_rows_C_{n.replace("@","_")}.json').exists() else None
        res[n]={sn:dict(gain=gain_table(A,B,A0,B0,m),C=None if C is None else {k:boot(metric_vec(C,k)[m]) for k in ('lp_sum','lp_mean','rr_sum','rr_mean')},C_minus_W0=None if C is None else {k:boot((metric_vec(C,k)-metric_vec(C0,k))[m]) for k in ('lp_sum','rr_mean')},
            A_abs={k:boot(metric_vec(A,k)[m]) for k in ('lp_sum','lp_mean','rr_mean')},B_abs={k:boot(metric_vec(B,k)[m]) for k in ('lp_sum','lp_mean','rr_mean')}) for sn,m in masks.items()}
    base={sn:dict(A=dict(lp_sum=boot(metric_vec(A0,'lp_sum')[m]),rr_mean=boot(metric_vec(A0,'rr_mean')[m])),B=dict(lp_sum=boot(metric_vec(B0,'lp_sum')[m]),rr_mean=boot(metric_vec(B0,'rr_mean')[m])),C=dict(lp_sum=boot(metric_vec(C0,'lp_sum')[m]),rr_mean=boot(metric_vec(C0,'rr_mean')[m]))) for sn,m in masks.items()}
    # item rows with the stream counts
    texts=count_strings(a['stream_dir']);strs=[s for it in items for s in it['cands']];cnt=counts_of(texts,strs);asg=json.load(open(out/'E57_pqa4_assignB.json'))['assignment'];irows=[]
    for k,it in enumerate(items):
        r=dict(id=it['id'],title=it['title'],type=it['type'],assigned_question_id=asg[it['id']],in_items455=bool(masks['items455'][k]),in_main373=bool(masks['main373'][k]),gold=it['cands'][0],cands=it['cands'],counts_total=[sum(cnt[c]['cnt']) for c in it['cands']],counts_wb_total=[sum(cnt[c]['cnt_wb']) for c in it['cands']],counts_by_section=[cnt[c]['cnt'] for c in it['cands']],counts_wb_by_section=[cnt[c]['cnt_wb'] for c in it['cands']])
        r['W0']=dict(A=A0[k]['cand_lp'][0],B=B0[k]['cand_lp'][0],C=C0[k]['cand_lp'][0])
        for n in res:r[n]=dict(A=ld_cache(ld,'A',n)[k]['cand_lp'][0],B=ld_cache(ld,'B',n)[k]['cand_lp'][0])
        irows.append(r)
    (out/'E57_pqa4_keys.json').write_text(json.dumps(dict(meta=dict(items_sha256=h,gain_definition='(metric A(state)-metric A(W0)) - (metric B(state)-metric B(W0)); rank-ratio metrics as improvement (W0 minus state), positive = better',B_rule='same answer type, different title, one question per item, np.random.default_rng(%d)'%SEED,C_rule='"Answer:" only',subsets={k:int(v.sum()) for k,v in masks.items()}),W0=base,states=res),indent=1,default=float))
    (out/'E57_pqa4_item_rows.json').write_text(json.dumps(irows,default=float))
_CACHE={}
def ld_cache(ld,cond,n):
    if (cond,n) not in _CACHE:_CACHE[(cond,n)]=ld(cond,n)
    return _CACHE[(cond,n)]
def selftest():
    P3.selftest()
    its=[dict(id=f'{t}{k}',type=t,title=f'T{k%3}',prefix=f'q{t}{k}') for t in ('CAP','LOW') for k in range(6)];asg=assign_B(its)
    by={it['id']:it for it in its};assert all(by[asg[i]]['type']==by[i]['type'] and by[asg[i]]['title']!=by[i]['title'] for i in asg) and asg==assign_B(its)
    assert prefix_of(its,'C',asg)==['Answer:']*len(its) and prefix_of(its,'A',asg)[0]=='qCAP0'
    man={'NATQ@orig':dict(section='orig'),'NATQ@rw4':dict(section='rw4'),'NAT@rw4':dict(section='rw4')};assert order_states(man)[:2]==['NATQ@rw4','NAT@rw4']
    rows=lambda v:[dict(cand_lp=[x],cand_mean=[x],rank_ratio_sum=1-x*0.1,rank_ratio_mean=1-x*0.1) for x in v]
    g=gain_table(rows([1,2]),rows([0,1]),rows([0,0]),rows([0,0]),np.ones(2,bool));assert abs(g['lp_sum']['gain']['mean']-1.0)<1e-9
    tx={k:'' for k in SECS+['tail']};tx['orig']='Paris is in France. paris. Paris2 Paris';c=counts_of(tx,['Paris','France']);assert c['Paris']['cnt'][1]==3 and c['Paris']['cnt_wb'][1]==2
    print('e57_pqa4_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');a=dict(cond=P2._pop('--cond'),out=P2._pop('--out'),manifest=P2._pop('--manifest'),items=P2._pop('--items-file'),bank=P2._pop('--bank'),stage1=P2._pop('--stage1'),dir610=P2._pop('--dir610'),stream_dir=P2._pop('--stream-dir'),paragraphs=P2._pop('--paragraphs'))
    {'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
