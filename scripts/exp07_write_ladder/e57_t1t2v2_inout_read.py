"""E57 T1/T2 v2 OUT + IN reads (job 595; GPU, read only; new file; post-hoc, exploratory): the frozen v2 items of jobs 570 / 575 / 585 (e57_t1t2v2_run.py items, candidates, per-token mean primary score, T1.rows_for_state) read in two contexts for every state of the manifest and W0:
  OUT = the neutral 8192-token SQuAD tail of jobs 276 / 410 / 570 (e57_t1t2v2_run.read_stage);
  IN  = the natural window of the stream at the end of the cycle: the last 8192 tokens of the stream of the item (stream A: chunks 272..279, stream B20: chunks 278..285; ids from e57_t1t2v2_run.stream_ids).
IN rows are split by whether the ANSWER is inside the window: present = the gold surface or a frozen variant of the item (strings of 3+ characters, word-bounded, case-insensitive) occurs in the decoded window text; absent otherwise (a string-match proxy for 'the gold article is in the window'; the counts per stream are printed by --cpu-check).
Output rows carry state_label = <label>|OUT, <label>|IN|all, <label>|IN|present, <label>|IN|absent and go through e57_t1t2v2_run.aggregate (per stream, per form group, and combined with equal weight per event).
  python3 e57_t1t2v2_inout_read.py --frozen F --cands C --bank B --manifest M.json --sentences-a SA --sentences-b SB --out O [--no-out] | --cpu-check (same args) | --selftest"""
import sys,json,time,gc,hashlib,pathlib,re
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_t1t2v2_run as V
import e57_t1t2_run as T1
import e57_reads as RD
WIN=8192
def present_flags(items,text):
    """{item id: bool} answer string inside the window text"""
    low=text.lower();out={}
    for it in items:
        ss=[s for s in [it['cands'][0]]+list(it['variants']) if len(s)>=3]
        out[it['id']]=any(re.search(r'(?<!\w)'+re.escape(s.lower())+r'(?!\w)',low) for s in ss)
    return out
def stream_windows(sa,sb):
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(str(V.HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json'));W={}
    for sm,path,key in (('A',sa,'A'),('B',sb,'B20')):
        ids=V.stream_ids(path,key);n=len(ids)//V.CH;w=[int(x) for x in ids[n*V.CH-WIN:n*V.CH]];W[sm]=dict(ids=w,text=tk.decode(w),n_chunks=n)
    return W
def cpu_check(a):
    frozen=V.load_frozen(a['frozen']);cands=json.load(open(a['cands']));items=V.make_items(frozen,cands,a['out']);W=stream_windows(a['sentences_a'],a['sentences_b'])
    for sm in ('A','B'):
        its=[it for it in items if it['stream']==sm];f=present_flags(its,W[sm]['text']);ev={it['event'] for it in its if f[it['id']]}
        print(f'stream {sm}: window {len(W[sm]["ids"])} tokens (last of {W[sm]["n_chunks"]} chunks); frames {len(its)}; answer string inside the window: {sum(f.values())} present / {len(f)-sum(f.values())} absent; events with a present frame {len(ev)}')
    if a.get('manifest'):
        for k,s in json.load(open(a['manifest'])).items():p=pathlib.Path(s['file']);print(f'state {k}: stream {s["stream"]} exists={p.exists()} hash={s["hash"][:12]}')
def read_stage(a):
    frozen=V.load_frozen(a['frozen']);cands=json.load(open(a['cands']));out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True)
    new=V.make_items(frozen,cands,out);man=json.load(open(a['manifest']));W=stream_windows(a['sentences_a'],a['sentences_b'])
    import e55_swap as E
    E.ADAM['NATADAMH']=('NAT',0.5,False)
    if 'NATADAMH' not in E.ARMS:E.ARMS.append('NATADAMH')
    import e55_run as R,e55_gpu as RG,e57_main_run as M
    LD,s3,C=R.lab();rt=LD.Runtime();lg=lambda *x:print(*x,flush=True);can=R.Canary55(LD,rt);lg('canary W0',can(rt.base))
    bank=RD.load_items(a['bank']);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,new);assert not bad,bad
    forbid=sorted({s for it in bank for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)});out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid)
    off=[0]
    for it in new:off.append(off[-1]+len(it['cands']))
    ctxs=[('OUT',lambda sm:out_tail)]*0
    conds=([] if a.get('no_out') else [('OUT',None)])+[('IN','win')]
    packs={}
    for sm in ('A','B'):
        idx=[k for k,it in enumerate(new) if it['stream']==sm];pres=present_flags([new[k] for k in idx],W[sm]['text'])
        for cn,_ in conds:
            ctx=out_tail if cn=='OUT' else W[sm]['ids'];groups=[]
            for g0 in range(0,len(idx),40):
                sub=idx[g0:g0+40];rr=[r for k in sub for r in raw[off[k]:off[k+1]]];groups.append(([new[k] for k in sub],rd.packed(rr,ctx)))
            packs[(sm,cn)]=dict(groups=groups,items=[new[k] for k in idx],ntok=[ntok[k] for k in idx],pres=pres)
        lg(f'  stream {sm}: {len(idx)} frames; IN window answer present {sum(pres.values())} / absent {len(pres)-sum(pres)}')
    def score(m,sm,cn):
        m32=rd.to_fp32(m);o=[]
        with LD.jax.default_matmul_precision('highest'):
            for sub,P in packs[(sm,cn)]['groups']:o+=RD.split_scores(np.asarray(P.r1(m32)),sub)
        return o
    allrows=[];w0r={};tim={};part=out/'E57_t1t2v2_inout_scores_partial.json'
    def emit(label,m,name,w0mode):
        for sm in ('A','B'):
            for cn,_ in conds:
                P=packs[(sm,cn)];rows=T1.rows_for_state(P['items'],score(m,sm,cn),P['ntok'],name,w0=None if w0mode else w0r[(sm,cn)])
                if w0mode:w0r[(sm,cn)]=rows
                for r in rows:
                    r['stream']=sm;r['cond']=cn
                    if cn=='OUT':r['state_label']=f'{label}|OUT';allrows.append(r)
                    else:
                        r['present']=bool(P['pres'][r['id']]);r['state_label']=f'{label}|IN|all';allrows.append(r)
                        for tag,flag in (('present',True),('absent',False)):
                            if r['present']==flag:r2=dict(r);r2['state_label']=f'{label}|IN|{tag}';allrows.append(r2)
    t0=time.monotonic();emit('W0',rt.base,'W0',True);tim['W0']=time.monotonic()-t0;lg(f'  W0 read: {tim["W0"]:.0f}s')
    live=M.inner_dtypes(LD,rd,rt.base)
    for name,s in man.items():
        t0=time.monotonic();m=rd.load_state(s['file'],expect_hash=s['hash']);dt=M.inner_dtypes(LD,rd,m);assert dt==live,('saved weights have another precision',dt,live)
        # a state belongs to one stream: only that stream's items are read
        sm0=s['stream']
        for sm in ('A','B'):
            if sm!=sm0:continue
            for cn,_ in conds:
                P=packs[(sm,cn)];rows=T1.rows_for_state(P['items'],score(m,sm,cn),P['ntok'],name,w0=w0r[(sm,cn)])
                for r in rows:
                    r['stream']=sm;r['cond']=cn
                    if cn=='OUT':r['state_label']=f"{s['label']}|OUT";allrows.append(r)
                    else:
                        r['present']=bool(P['pres'][r['id']]);r['state_label']=f"{s['label']}|IN|all";allrows.append(r)
                        for tag,flag in (('present',True),('absent',False)):
                            if r['present']==flag:r2=dict(r);r2['state_label']=f"{s['label']}|IN|{tag}";allrows.append(r2)
        del m;tim[name]=time.monotonic()-t0;lg(f'  {name} (stream {sm0}) read: {tim[name]:.0f}s');part.write_text(json.dumps(dict(done=list(tim),rows=allrows),default=float))
    meta=dict(frozen_sha256=V.FROZEN_SHA,cands_sha256=hashlib.sha256(open(a['cands'],'rb').read()).hexdigest(),states={k:dict(v) for k,v in man.items()},contexts='OUT = 8192-token SQuAD tail of jobs 276/410/570; IN = last 8192 tokens of the stream of the item',present_rule='gold surface or frozen variant (3+ chars, word-bounded, case-insensitive) inside the decoded IN window',window_present_counts={sm:int(sum(packs[(sm,conds[-1][0])]['pres'].values())) for sm in ('A','B')},timing=tim)
    (out/'E57_t1t2v2_inout_scores.json').write_text(json.dumps(dict(meta=meta,rows=allrows),default=float));kn=dict(meta=meta,**V.aggregate(allrows));(out/'E57_t1t2v2_inout_score_keys.json').write_text(json.dumps(kn,indent=1,default=float))
    lg(json.dumps({st:{g:(v['n_events'],round(v['acc_all_mean'],3)) for g,v in d.items() if g=='ALL'} for st,d in kn['combined_by_event'].items()}))
def selftest():
    its=[{'id':'a','cands':['March 24'],'variants':['24 March']},{'id':'b','cands':['Iran'],'variants':[]},{'id':'c','cands':['Ab'],'variants':[]}]
    f=present_flags(its,'The meeting was on march 24, in Tehran. Ab');assert f=={'a':True,'b':False,'c':False},f
    f=present_flags(its,'He said Iran.');assert f['b'] and not f['a']
    V.selftest();print('e57_t1t2v2_inout_read selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    cpu='--cpu-check' in sys.argv
    if cpu:sys.argv.remove('--cpu-check')
    no_out='--no-out' in sys.argv
    if no_out:sys.argv.remove('--no-out')
    a=dict(frozen=V._pop('--frozen'),cands=V._pop('--cands'),bank=V._pop('--bank'),manifest=V._pop('--manifest'),sentences_a=V._pop('--sentences-a'),sentences_b=V._pop('--sentences-b'),out=V._pop('--out'),no_out=no_out)
    cpu_check(a) if cpu else read_stage(a)
if __name__=='__main__':main()
