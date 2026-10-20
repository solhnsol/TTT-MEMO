"""E33 R pure-Python helpers (no jax, no tokenizer object needed; the caller passes `encode`). Used by nat_ladder.py; selftest on CPU.
  tok_spans            char spans of the exposure units -> token spans (BOS offset 1), same rule as stage2.Probe
  strata_real          per person: fact-unit token starts and s_cross / s_same / n_out recomputed from the REAL tokenization (strata.py rule: previous fact unit >= 8192 tokens
                       before = n_out, in an earlier 1024-token chunk = s_cross, same chunk = s_same)
  strata_diff          comparison with data/natural-text/strata.json (approximate positions): persons whose (s_cross, s_same, n_out) changed
  build_pass_stream    pass p >= 2 stream: the stream's world with every background text replaced, in order, by the next paragraph of `pool` (no cycling); if the stream is shorter
                       than `target` tokens, further pool paragraphs are appended at the end; asserts that every exposure unit stays inside the first `target` tokens
  tail_check           world strings found in a readout tail text (persons of the targets must be absent, the rest is recorded)
Usage: python3 nat_pass.py --selftest"""
import re,bisect,copy,sys
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nat_world as NW
CH=1024;WIN=8192

def tok_spans(spans,ends,off=1):
    return [dict(s,tok_start=off+bisect.bisect_right(ends,s['start']),tok_end=off+bisect.bisect_right(ends,s['end']-1)) for s in spans]

def strata_real(spans):
    ex={}
    for s in spans:
        if s['label']=='fact':ex.setdefault(s['t'],[]).append(s['tok_start'])
    out={}
    for tid,ps in ex.items():
        ps=sorted(ps);sc=ss=no=0
        for a,b in zip(ps,ps[1:]):
            if b-a>=WIN:no+=1
            elif b//CH!=a//CH:sc+=1
            else:ss+=1
        out[tid]={'n_fact':len(ps),'s_cross':sc,'s_same':ss,'n_out':no,'pos':ps}
    return out

def strata_diff(real,rows,stream,layout):
    """rows: strata.json list. returns {'persons':n,'changed':[tid..],'sum_s_cross_json':..,'sum_s_cross_real':..}"""
    ref={r['tid']:r for r in rows if r['stream']==stream and r['layout']==layout}
    ch=[t for t,v in real.items() if t in ref and (v['s_cross'],v['s_same'],v['n_out'])!=(ref[t]['s_cross'],ref[t]['s_same'],ref[t]['n_out'])]
    return {'persons':len(real),'in_json':len(ref),'changed':sorted(ch),'sum_s_cross_json':sum(r['s_cross'] for r in ref.values()),'sum_s_cross_real':sum(v['s_cross'] for v in real.values()),
            'sum_s_same_json':sum(r['s_same'] for r in ref.values()),'sum_s_same_real':sum(v['s_same'] for v in real.values()),'sum_n_out_json':sum(r['n_out'] for r in ref.values()),'sum_n_out_real':sum(v['n_out'] for v in real.values())}

def build_pass_stream(w0,pool,swap,encode,bos,target):
    """encode(text) -> (ids, ends) with ends[i] = char end of token i. returns (world, ids (with BOS, NOT truncated), spans with tok_start/tok_end, n_pool_used)."""
    w=copy.deepcopy(w0);bi=0;new=[]
    for t in w['texts']:
        if t['kind']=='bg':
            assert bi<len(pool),f'filler pool exhausted ({len(pool)} paragraphs)';new.append({'kind':'bg','text':pool[bi]});bi+=1
        else:new.append(t)
    w['texts']=new
    def enc_():
        text,_,spans=NW.realize(w,swap);ids,ends=encode(text);return [bos]+list(ids),ends,spans
    ids,ends,spans=enc_()
    while len(ids)<target:
        for _ in range(max(1,(target-len(ids))//400)):
            assert bi<len(pool),f'filler pool exhausted while padding ({len(pool)} paragraphs)';w['texts'].append({'kind':'bg','text':pool[bi]});bi+=1
        ids,ends,spans=enc_()
    sp=tok_spans(spans,ends)
    late=[s for s in sp if s['tok_end']>=target]
    assert not late,f'{len(late)} exposure units fall beyond token {target}'
    return w,ids,sp,bi


def realized_units(w,swap):
    """per text unit the realized string, same substitution as NW.realize (asserted equal when joined)."""
    nm=NW.names(w,swap);k={'':0,'F':1,'S':2}
    out=[NW.PH.sub(lambda m:nm[int(m.group(2))][k[m.group(1)]],t['text']) for t in w['texts']]
    assert '\n\n'.join(out)==NW.realize(w,swap)[0];return out

def build_pass_stream_matched(w0,pool,swap,encode,decode,bos,target,slack=8,min_tok=8,max_rounds=12,max_drift=2,pool_factor=1.5):
    """pass p >= 2 stream with the SAME exposure-unit token positions as the stream w0 itself (pass 1): every background slot gets a pool paragraph cut at a token boundary to the slot's token
    length (best fit: the shortest unused paragraph with >= slot length + slack tokens, longest slots first, no pool paragraph used twice), then the joint tokenization is compared with pass 1 and
    each slot just before a drifting exposure unit is lengthened / shortened by the drift (tokenization at the '\\n\\n' joints is not exactly additive) until every exposure unit starts at the pass-1 token.
    encode(text) -> (ids, ends); decode(ids) -> text. Returns (world, ids (BOS + tokens, not truncated), spans with tok_start/tok_end, info{...}, used_bg texts)."""
    text0,_,spans0=NW.realize(w0,swap);ids0,ends0=encode(text0);sp0=tok_spans(spans0,ends0)
    base={(s['t'],s['round'],s['text']):s['tok_start'] for s in sp0};assert len(base)==len(sp0)
    texts=w0['texts'];slots=[i for i,t in enumerate(texts) if t['kind']=='bg']
    n={i:len(encode(texts[i]['text'])[0]) for i in slots}
    pids=[encode(p)[0] for p in pool];plen=[len(x) for x in pids];need_tok=sum(n.values());pool_tok=sum(plen)
    assert pool_tok>=pool_factor*need_tok,f'pool too small: {pool_tok} tokens < {pool_factor} x {need_tok} (slots total); {len(pool)} paragraphs'
    # slots in stream order take consecutive pool paragraphs (shuffled pool order, each paragraph used once, joined by blank lines) until n + slack tokens are covered; the last paragraph is cut at a token boundary
    assign={};join_ids={};cursor=0
    for i in slots:
        got=[];acc=0
        while True:
            assert cursor<len(pool),f'pool exhausted at slot {i} ({n[i]} tokens) after {cursor} paragraphs'
            got.append(cursor);acc+=plen[cursor]+1;cursor+=1
            if acc>=n[i]+slack:
                jids=encode('\n\n'.join(pool[x] for x in got))[0]
                if len(jids)>=n[i]+slack:break
        assign[i]=got;join_ids[i]=jids
    prev_bg={}
    last=None
    for i,t in enumerate(texts):
        prev_bg[i]=last
        if t['kind']=='bg':last=i
    def cut(i):
        ji=join_ids[i];m=min(n[i],len(ji))
        while m>min_tok:
            tx=decode(ji[:m])
            if '\ufffd' not in tx:return tx.strip()
            m-=1
        raise AssertionError(f'cannot cut the pool text of slot {i} cleanly')
    def assemble():
        w=copy.deepcopy(w0)
        for i in slots:w['texts'][i]={'kind':'bg','text':cut(i)}
        ids,ends=encode(NW.realize(w,swap)[0]);return w,[bos]+list(ids),ends
    rounds=0;resid=None
    while True:
        w,ids,ends=assemble();sp=tok_spans(NW.realize(w,swap)[2],ends)
        assert {(s['t'],s['round'],s['text']) for s in sp}==set(base),'exposure units changed'
        d=[s['tok_start']-base[(s['t'],s['round'],s['text'])] for s in sp];resid=max(abs(x) for x in d)
        if resid==0 or rounds>=max_rounds:break
        rounds+=1;prev=0;idx_of={(s['t'],s['round'],s['text']):None for s in sp}
        # sp is in stream order = order of expo units in texts; map each to its text index
        eidx=[i for i,t in enumerate(texts) if t['kind']=='expo'];assert len(eidx)==len(sp)
        for ti,dj in zip(eidx,d):
            e=dj-prev;prev=dj
            if e!=0:
                s_=prev_bg[ti]
                if s_ is not None:n[s_]=max(min_tok,min(n[s_]-e,len(join_ids[s_])))
    assert resid<=max_drift,f'exposure positions drift by up to {resid} tokens after {rounds} rounds (allowed {max_drift})'
    late=[s for s in sp if s['tok_end']>=target]
    assert not late,f'{len(late)} exposure units fall beyond token {target}'
    used=[x for i in slots for x in w['texts'][i]['text'].split('\n\n') if x]   # paragraphs actually used (the last one of a slot is the cut piece)
    if len(ids)<target:raise AssertionError(f'matched stream is {len(ids)} < {target} tokens')
    return w,ids,sp,{'rounds':rounds,'max_abs_drift':int(resid),'slots':len(slots),'tokens_cut_mean':float(np.mean([len(join_ids[i])-n[i] for i in slots])),'pool_tokens':pool_tok,'need_tokens':need_tok,'pool_ratio':pool_tok/need_tok,'paragraphs_consumed':cursor,'paragraphs_unused':len(pool)-cursor},used

def tail_check(w,tail_text):
    """returns {'hits': strings found, 'target_person_hits': [...]}; target persons (fake full names) must be absent."""
    fs=NW.forbid_strings(w);hit=[s for s in fs if re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)',tail_text)]
    tp={t['X'] for t in w['triples']};bad=[s for s in hit if s in tp]
    assert not bad,f'target person names inside the readout tail: {bad}'
    return {'hits':hit,'n_forbid':len(fs)}

def selftest():
    # fake tokenizer: one token per word, ends = char end of each word
    def encode(text):
        ids=[];ends=[]
        for m in re.finditer(r'\S+',text):ids.append(hash(m.group())%1000+10);ends.append(m.end())
        return ids,ends
    def mk(n,tag):return ' '.join(f'{tag}{i}' for i in range(n))
    tri=[{'id':i,'pair':f'p{i//2}','X':f'Fa{i} Zo{i}','XF':f'Fa{i}','XS':f'Zo{i}','P':f'Town{i}','Q':None,'E':None} for i in range(4)]
    texts=[]
    for k in range(20):
        texts.append({'kind':'bg','text':mk(300,f'b{k}x')})
        if k%2==0:
            t=k//2%4;texts.append({'kind':'expo','t':t,'round':k//8+1,'label':'fact','text':f'@X{t}@ was born in {tri[t]["P"]} '+mk(20,f'e{k}x')})
    w={'seed':1,'triples':tri,'texts':texts,'pairs':2,'ctrl_persons':['Cp'],'ctrl_places':['Cl'],'pseudo_persons':['Pp'],'pseudo_places':['Pl'],'frames':{}}
    pool=[mk(300,f"q{k}x") for k in range(60)]
    target=6200
    wp,ids,sp,used=build_pass_stream(w,pool,False,encode,0,target)
    assert len(ids)>=target and used>=20 and len(sp)==10 and all(s['tok_end']<target for s in sp)
    assert [s['text'] for s in sp]==[s['text'] for s in NW.realize(w,False)[2]],'exposure texts changed'
    # exposure text kept, background different
    assert all(t['text'] not in {x['text'] for x in w['texts'] if x['kind']=='bg'} for t in wp['texts'] if t['kind']=='bg')
    # B world swaps the names only
    wb,idb,spb,_=build_pass_stream(w,pool,True,encode,0,target);assert len(spb)==10 and [s['t'] for s in spb]==[s['t'] for s in sp]
    r=strata_real(sp);assert set(r)=={0,1,2,3} and all(v['n_fact']>=1 for v in r.values())
    rows=[{'stream':1,'layout':'spread','tid':t,'s_cross':v['s_cross'],'s_same':v['s_same'],'n_out':v['n_out']} for t,v in r.items()]
    d=strata_diff(r,rows,1,'spread');assert d['changed']==[] and d['persons']==4
    rows[0]['s_cross']+=1;assert strata_diff(r,rows,1,'spread')['changed']==[rows[0]['tid']]
    # pool exhaustion must raise
    try:build_pass_stream(w,pool[:5],False,encode,0,target);raise SystemExit('exhausted pool not detected')
    except AssertionError:pass
    # exposure beyond target must raise
    try:build_pass_stream(w,pool,False,encode,0,1500);raise SystemExit('late exposure not detected')
    except AssertionError:pass
    assert tail_check(w,'a tale of Town1 and nothing else')['hits']==['Town1']
    try:tail_check(w,'then Fa2 Zo2 came');raise SystemExit('person in tail not detected')
    except AssertionError:pass

    # matched builder: fake tokenizer where a unit starting with 'q' gets one EXTRA token (context effect), decode = join
    def encode2(text):
        ids=[];ends=[];off=0
        for u in text.split('\n\n'):
            first=True;nw=len(u.split())
            for m in re.finditer(r'\S+',u):
                if first and m.group().startswith('q') and nw%2==0:ids.append(7);ends.append(off+m.start()+1)
                first=False;ids.append(hash(m.group())%1000+10);ends.append(off+m.end())
            off+=len(u)+2
        return ids,ends
    # decode of a prefix of word-ids needs the words: keep a registry
    reg={}
    def encode3(text):
        ids,ends=encode2(text)
        for m in re.finditer(r'\S+',text):reg[hash(m.group())%1000+10]=m.group()
        return ids,ends
    def decode3(ids):return ' '.join(reg[i] for i in ids if i!=7)
    pool2=[mk(int(200+17*k%250),f'q{k}x') for k in range(60)]
    wm,idm,spm,infom,usedm=build_pass_stream_matched(w,pool2,False,encode3,decode3,0,6200)
    assert infom['max_abs_drift']<=2,infom
    assert len(set(usedm))==len(usedm) and not (set(usedm)&{t['text'] for t in w['texts'] if t['kind']=='bg'})
    assert [s['text'] for s in spm]==[s['text'] for s in NW.realize(w,False)[2]]
    assert infom['paragraphs_consumed']+infom['paragraphs_unused']==60 and infom['pool_ratio']>=1.5
    try:build_pass_stream_matched(w,pool2[:20],False,encode3,decode3,0,6200);raise SystemExit('small pool not detected')
    except AssertionError as e:assert 'pool too small' in str(e) or 'exhausted' in str(e),e
    w_long=copy.deepcopy(w);w_long['texts'][0]['text']=mk(700,'lg')   # a slot longer than any single pool paragraph (the job-58 failure) is filled from several paragraphs
    wl,idl,spl,infol,usedl=build_pass_stream_matched(w_long,pool2,False,encode3,decode3,0,6600)
    assert infol['max_abs_drift']<=2 and len(wl['texts'][0]['text'].split())>=690,infol
    print('matched builder OK',infom)
    print('nat_pass SELFTEST OK',d['sum_s_cross_real'],'s_cross,',used,'pool paragraphs used')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
