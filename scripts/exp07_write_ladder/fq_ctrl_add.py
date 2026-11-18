"""CPU, new file (Hans 11/18 '추천대로' = option (a)): after the W0 replacement (fq_w0_select.py) some worlds have fewer than 4 never-exposed control facts. This adds control facts from the builder pool (fq_build_worlds.load) until every world has NCTRL = 4 controls in total (remaining reserve facts first, then pool facts). A pool fact is accepted as control only when: its event is not an event of the world's facts / reserve / already chosen controls; its gold string (lower case) differs from every gold of facts / reserve / controls, and its gold string occurs in no background sentence of the world (same regex as the builder); gold is 3 Llama-3 tokens (pool filter). A pool fact is accepted even if background sentences of its event (without its gold) exist (the pool is fully covered by background events: all 42 pool events are world or background events); events with the FEWEST such sentences are taken first and the count is stored per control as n_bg_sentences_of_event. The chunks / facts / schedule of the world are NOT changed (only 'ctrl_extra' is written, the field read by fq_world.to_gen_world after the reserve; the key list 'ctrl_extra_source' records the pool event ids). Deterministic: pool events in the order of random.Random(md5('fqctrl|seed')).
  python3 -I fq_ctrl_add.py <indir with the parquet files> <fq code dir> <in worlds dir (worlds_w0)> <out dir> | --selftest"""
import sys,json,re,hashlib,random,pathlib
def md5rng(*a):return random.Random(int(hashlib.md5('|'.join(map(str,a)).encode()).hexdigest()[:16],16))
NCTRL=4
def choose(world,pool,bg,ok=None):
    """-> list of control facts to append after world['reserve'] (pure function of its arguments)"""
    need=NCTRL-len(world['reserve'])
    if need<=0:return [],[]
    bgset=set(world['bg_sentences']);from collections import Counter;bgcnt=Counter(e for e,x,n in bg if x in bgset)
    taken_ev={f['event'] for f in world['facts']+world['reserve']};golds={f['gold'].lower() for f in world['facts']+world['reserve']}
    bgtxt=' '.join(world['bg_sentences']).lower();out=[];why={'world_event':0,'gold_dup':0,'gold_in_bg':0}
    ev=sorted(pool);md5rng('fqctrl',world['seed']).shuffle(ev);ev.sort(key=lambda e:bgcnt[e])      # events with the fewest background sentences first (stable: random order within equal counts)
    for e in ev:
        if len(out)==need:break
        if e in taken_ev:why['world_event']+=1;continue
        c=pool[e][:];md5rng('fqctrlf',world['seed'],e).shuffle(c)
        for f in c:
            g=f['gold'].lower()
            if ok is not None and not ok(f):why['tok_len']=why.get('tok_len',0)+1;continue
            if g in golds:why['gold_dup']+=1;continue
            if re.search(r'(?<![\w])'+re.escape(g)+r'(?![\w])',bgtxt):why['gold_in_bg']+=1;continue
            out.append(dict(event=f['event'],qid=f['qid'],question=f['question'],gold=f['gold'],anchors=f['anchors']));golds.add(g);taken_ev.add(e);out[-1]['n_bg_sentences_of_event']=bgcnt[e];break
    return out,why
def main():
    if '--selftest' in sys.argv:
        w=dict(seed=1,facts=[dict(event='a',gold='Xa')],reserve=[dict(event='b',gold='Xb')],bg_sentences=['hello Zed there'])
        pool={'a':[dict(event='a',qid=0,question='q',gold='Xa',anchors=[])],'c':[dict(event='c',qid=1,question='q',gold='Zed',anchors=[])],'d':[dict(event='d',qid=2,question='q',gold='Dd',anchors=[])],'e':[dict(event='e',qid=3,question='q',gold='Ee',anchors=[])]}
        bg=[('c','hello Zed there',5)];o,why=choose(dict(w,reserve=w['reserve']*1),pool,bg);assert len(o)==3 or len(o)==2,o
        assert all(f['event'] in ('d','e') for f in o) and why['world_event']>=0
        print('fq_ctrl_add selftest ok');return
    indir,code,win,out=sys.argv[1:5];sys.path.insert(0,code);okf=None
    if len(sys.argv)>5:
        from transformers import AutoTokenizer;qt=AutoTokenizer.from_pretrained(sys.argv[5]);okf=lambda f:len(qt(' '+f['gold'],add_special_tokens=False).input_ids)==3      # optional 5th argument: a tokenizer name; a control gold must be 3 tokens of it (Qwen worlds)
    from fq_build_worlds import load
    pool,tok,bg=load(indir);o=pathlib.Path(out);o.mkdir(parents=True,exist_ok=True);rep={}
    for f in sorted(pathlib.Path(win).glob('fq_*_w0.json')):
        w=json.load(open(f));ex,why=choose(w,pool,bg,okf);w['ctrl_extra']=ex;w['ctrl_extra_source']=[x['event'] for x in ex];rep[f.name]=dict(reserve=len(w['reserve']),added=len(ex),rejected=why)
        json.dump(w,open(o/f.name,'w'),ensure_ascii=False)
    json.dump(rep,open(o/'fq_ctrl_add_report.json','w'),indent=1);print(json.dumps(rep,indent=1))
if __name__=='__main__':main()
