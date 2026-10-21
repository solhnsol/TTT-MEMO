"""E51 CPU table of the natural-text streams: per person s_cross / s_same / n_out (real tokenization), name-only exposures, one-time persons, and per stream/layout the O-mask size (E33 nat_signal rule).
Per (stream, layout): realized world-A text -> tokenizer offsets -> nat_signal.build_table (O sets) + nat_pass.tok_spans/strata_real (strata).
Per person: n_fact, s_cross, s_same, n_out, n_name_only (units with label != 'fact', whole stream), name_only_in_win (name-only units that lie inside the 8192 tokens before at least one of the person's fact units),
one_time (n_fact == 1).  Per stream: persons, one-time persons, sum s_cross, chunks with a fact unit, chunks with O (= eligible, answers non-empty), sum |O| over chunks, sum |O| per chunk with O, answer tokens.
No GPU, no model. Tokenizer: --tokenizer-file / $TOKENIZER_JSON / <repo>/data/tokenizer/llama3/tokenizer.json, else the regex proxy (geometry only; the table says which).
Usage: e51_nat_table.py --nat-dir DIR --nat 9103:clustered,9103:spread,9101:spread,... [--tokenizer-file F] [--require-real] [--persons] [--out-dir D]   |   --selftest"""
import os,sys,re,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_world as NW,nat_signal as NS,nat_pass as NP
CH,WIN,NCH=NS.CH,NS.WIN,NS.NCH
def table_of(w,offs,ntok=None):
    """returns (persons rows, stream summary) for world w (swap=False) and encoding offsets offs (without BOS)"""
    text,_,spans=NW.realize(w,False);ends=[b for a,b in offs]
    sp=NP.tok_spans(spans,ends);st=NP.strata_real(sp)
    tab,owner=NS.build_table(w,False,offs,ntok=len(offs)+1 if ntok is None else ntok);sets=NS.NatSets(tab,0)
    names=NW.names(w,False);ids={}
    for s in sp:ids.setdefault(s['t'],[]).append(s)
    rows=[]
    for t,us in sorted(ids.items()):
        facts=[u for u in us if u['label']=='fact'];no=[u for u in us if u['label']!='fact']
        fs=[u['tok_start'] for u in facts]
        inwin=sum(1 for u in no if any(0<=f-u['tok_start']<WIN for f in fs))
        r=st.get(t,{'n_fact':0,'s_cross':0,'s_same':0,'n_out':0})
        rows.append(dict(t=t,name=names[t][0] if isinstance(names[t],(list,tuple)) else str(names[t]),n_fact=r['n_fact'],s_cross=r['s_cross'],s_same=r['s_same'],n_out=r['n_out'],n_name_only=len(no),name_only_in_win=inwin,one_time=r['n_fact']==1))
    chunks=[];
    for k in range(NCH):
        cs=sets.cur_sentences(k)
        if cs:O=sets.oracle(k);an=sets.answers(k);chunks.append(dict(k=k,n_fact=len(cs),n_O=len(O),n_ans=len(an),elig=bool(an)))
    withO=[c for c in chunks if c['n_O']]
    summ=dict(persons=len(rows),one_time=sum(r['one_time'] for r in rows),sum_s_cross=sum(r['s_cross'] for r in rows),sum_s_same=sum(r['s_same'] for r in rows),sum_n_out=sum(r['n_out'] for r in rows),
              name_only_units=sum(r['n_name_only'] for r in rows),persons_with_name_only_in_win=sum(r['name_only_in_win']>0 for r in rows),stream_tokens=len(offs)+1,
              chunks_with_fact=len(chunks),chunks_with_O=len(withO),eligible_chunks=sum(c['elig'] for c in chunks),sum_O=sum(c['n_O'] for c in chunks),mean_O_per_chunk_with_O=float(np.mean([c['n_O'] for c in withO])) if withO else 0.0,answer_tokens=sum(c['n_ans'] for c in chunks))
    return rows,summ,chunks
def fmt_row(label,s):
    return f"| {label} | {s['persons']} | {s['one_time']} | {s['sum_s_cross']} | {s['sum_s_same']} | {s['sum_n_out']} | {s['name_only_units']} | {s['persons_with_name_only_in_win']} | {s['chunks_with_fact']} | {s['chunks_with_O']} | {s['eligible_chunks']} | {s['sum_O']} | {s['mean_O_per_chunk_with_O']:.0f} | {s['answer_tokens']} |"
HDR="| stream:layout | persons | one-time persons | sum s_cross | sum s_same | sum n_out | name-only units | persons w/ name-only in 8K before a fact | chunks w/ fact | chunks w/ O | eligible | sum O tokens | O per chunk | answer tokens |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
def encode_offs(tk,text):
    if tk is not None:return list(tk.encode(text,add_special_tokens=False).offsets)
    import nat_dryrun as D
    px=D.Proxy();_,ends=px.encode(text);starts=[0]+ends[:-1];return list(zip(starts,ends))
def selftest():
    import nat_dryrun as D
    w={'triples':[{'id':0,'pair':'p0','X':'Ann Bell','XF':'Ann','XS':'Bell','P':'Oslo'},{'id':1,'pair':'p0','X':'Cy Dunn','XF':'Cy','XS':'Dunn','P':'Rome'}],'texts':[],'pairs':1}
    def bg(n):return {'kind':'bg','text':' '.join(f'bg{i}' for i in range(n))}
    for i in range(14):
        w['texts'].append(bg(900))
        if i in (1,5,12):w['texts'].append({'kind':'expo','t':0,'round':i,'label':'fact' if i!=1 else 'name_only','text':'@X0@ lived. @XF0@ was born in Oslo today. Then Bell left.'})
        if i==7:w['texts'].append({'kind':'expo','t':1,'round':1,'label':'fact','text':'@X1@ was born in Rome.'})
    text=NW.realize(w,False)[0];offs=[(m.start(),m.end()) for m in re.finditer(r'\S+',text)]
    rows,s,ch=table_of(w,offs)
    r0=[r for r in rows if r['t']==0][0];r1=[r for r in rows if r['t']==1][0]
    assert r0['n_fact']==2 and r0['n_name_only']==1 and r1['n_fact']==1 and r1['one_time'] and not r0['one_time'],(r0,r1)
    assert r0['s_cross']+r0['s_same']+r0['n_out']==1
    assert s['persons']==2 and s['one_time']==1 and s['name_only_units']==1 and s['answer_tokens']>=1 and s['chunks_with_O']>=1,s
    # name-only unit of person 0 sits ~900+ tokens before the first fact unit -> inside the 8K window
    assert r0['name_only_in_win']==1,r0
    print(HDR);print(fmt_row('selftest',s));print('e51_nat_table SELFTEST OK')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nat',default='9103:clustered,9103:spread');ap.add_argument('--nat-dir',default=os.environ.get('NAT_DIR','/mnt/project-files/data/natural-text/worlds'));ap.add_argument('--tokenizer-file',default=None)
    ap.add_argument('--require-real',action='store_true');ap.add_argument('--persons',action='store_true');ap.add_argument('--out-dir',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import nat_dryrun as D
    try:tk=D.load_tok(a.tokenizer_file)
    except Exception as e:print('real tokenizer not usable:',repr(e)[:100]);tk=None
    assert tk is not None or not a.require_real,'--require-real: no real tokenizer'
    kind='REAL tokenizer' if tk is not None else 'PROXY tokenizer (geometry only, numbers are NOT final)';print('tokenizer:',kind);out=[];allrows={}
    for item in a.nat.split(','):
        stream,layout=item.split(':')
        import nat_filler as NF
        p=NF.world_path(a.nat_dir,int(stream),layout)
        if not Path(p).exists():print('missing',p);continue
        w=NW.load_world(p);offs=encode_offs(tk,NW.realize(w,False)[0]);rows,s,ch=table_of(w,offs);allrows[item]=dict(summary=s,persons=rows,chunks=ch);out.append(fmt_row(item,s))
        if a.persons:
            print(f'## {item} per person');print('| t | n_fact | s_cross | s_same | n_out | name-only units | name-only in 8K | one-time |\n|---|---|---|---|---|---|---|---|')
            for r in rows:print(f"| {r['t']} | {r['n_fact']} | {r['s_cross']} | {r['s_same']} | {r['n_out']} | {r['n_name_only']} | {r['name_only_in_win']} | {int(r['one_time'])} |")
    print(HDR);print('\n'.join(out))
    if a.out_dir:
        Path(a.out_dir).mkdir(parents=True,exist_ok=True);(Path(a.out_dir)/'e51_nat_table.json').write_text(json.dumps(dict(tokenizer=kind,streams=allrows),indent=1));print('wrote',a.out_dir)
if __name__=='__main__':main()
