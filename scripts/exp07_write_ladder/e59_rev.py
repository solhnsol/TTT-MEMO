"""E59 REVERSE-question readout (Hans 11/19 '역방향까지 붙여서 준비해보자'; new file; secondary readout, the main metric stays the forward exposure-128 out-of-window margin change): the synthetic v3_16_1 worlds (E2E and Qwen) ask place -> person instead of person -> place. Item of fact f: prefix = one of 3 REVERSE question frames with {P} = the place of f (frame = fact id mod 3, like the forward frames); candidates = the 16 NAMES of the 16 facts of the world (' ' + name; own name FIRST = gold_idx 0, then the other 15 in fact-id order); same summed span log-prob margin / top-1 / rank as the forward readout; reads in the same two contexts (out-of-window neutral 8192 tail, and in-window = the real chunks before the group's next appearance, only the 16 fact items, no controls; plan = the forward plan with its item index lists cut to ids < 16). The names of one world have equal token length (world name_len, asserted with the tokenizer), the reverse frames are checked against the exposure frame bank with the same 5-gram overlap count as the forward question frames (must not exceed it). No training; weights are only read.
  rev_items(world) -> 16 items in gen_facts.build_items format; rev_plan(plan) -> plan with the item lists cut to the fact ids; overlap_counts(world); checks(world, enc) ; --selftest (CPU, synthetic world seed 71)."""
import sys,re,copy,pathlib
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
RQFRAMES=['Nobody could say exactly who had grown up in {P}, though most guessed the person named',
          'If you asked anyone who called {P} home, they would name the person called',
          'The registry listed the person born in {P} as the individual named']
NQ=len(RQFRAMES)
def rev_items(world):
    """16 reverse items (kind 'fact'); gold = the name of the fact whose place is in the question"""
    facts=world['facts'];k=len(facts);items=[]
    for f in facts:
        cl=[f]+[g for g in facts if g['id']!=f['id']];cl.sort(key=lambda g:(g['id']!=f['id'],g['id']))
        items.append({'kind':'fact','fact':f['id'],'group':f['group'],'name_id':f['id'],'place_id':f['id'],'name':f['P'],'qframe':f['id']%NQ,'prefix':RQFRAMES[f['id']%NQ].format(P=f['P']),'cands':[' '+x['X'] for x in cl],'gold_idx':0,'cand_fact':[x['id'] for x in cl],'cand_group':[x['group'] for x in cl],'cand_same':[x['group']==f['group'] for x in cl],'cand_exposed':[True]*len(cl),'n_rank':len(cl)})
    return items
def rev_plan(plan):
    """forward plan (e55_read.plan(world)) -> plan with item lists cut to the 16 fact items (ids < 16); the forward plan lists control items (ids >= 16) in group 0 / 'out' which the reverse readout does not have"""
    p=copy.deepcopy(plan);n=16
    for a,v in p['read'].items():
        for name,c in v['ctx'].items():c['items']=[i for i in c['items'] if i<n]
    p['items']=None;return p
def overlap_counts(world):
    import gen_world_ctl as W
    fr=[*world['bank']] if isinstance(world.get('bank'),list) else list(world['bank'])
    fs=set().union(*[W._slotless_ngrams(f) for f in fr]);fw=len(set().union(*[W._slotless_ngrams(q) for q in W.QFRAMES])&fs);rv=len(set().union(*[W._slotless_ngrams(q) for q in RQFRAMES])&fs);return dict(forward=fw,reverse=rv)
def checks(world,enc_len):
    """enc_len(text)->token count; returns dict of booleans / counts"""
    its=rev_items(world);r={}
    r['16_items']=len(its)==16 and all(len(i['cands'])==16 for i in its)
    r['gold_first_and_own_name']=all(i['cands'][0]==' '+world['facts'][i['fact']]['X'] for i in its)
    r['names_distinct']=len({x['X'] for x in world['facts']})==16
    lens={enc_len(c) for i in its for c in i['cands']};r['candidate_token_lengths']=sorted(lens);r['equal_length']=len(lens)==1
    r['prefix_has_place_not_name']=all(world['facts'][i['fact']]['P'] in i['prefix'] and not any(x['X'] in i['prefix'] for x in world['facts']) for i in its)
    r['overlap']=overlap_counts(world);r['overlap_ok']=r['overlap']['reverse']<=r['overlap']['forward']
    return r
def selftest():
    import e55_run as R55,e55_read as R,e57_syn_pass as SP,gen_facts as G
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));enc,w,tf=R55.build(SP.synth_args(71,'v3_16_1'),71,read_apps={16,32,64,128},arms=['NAT'])
    c=checks(w,lambda t:len(tk.encode(t,add_special_tokens=False).ids));print(c);assert c['16_items'] and c['gold_first_and_own_name'] and c['names_distinct'] and c['equal_length'] and c['prefix_has_place_not_name'] and c['overlap_ok'],c
    its=rev_items(w);m=G.item_meta(its);assert len(m['kind'])==16 and all(g==0 for g in m['gold_idx'])
    # prefix + candidate tokenisation is prefix-stable and the candidate span has the same length for every candidate
    for it in its:
        p=tk.encode(it['prefix'],add_special_tokens=False).ids;ls=set()
        for cnd in it['cands']:
            f=tk.encode(it['prefix']+cnd,add_special_tokens=False).ids;assert f[:len(p)]==p and len(f)>len(p),(it['prefix'],cnd);ls.add(len(f)-len(p))
        assert len(ls)==1,ls
    P=R.plan(w,enc);rp=rev_plan(P);assert all(max(c['items'])<16 for v in rp['read'].values() for c in v['ctx'].values() if c['items']) and set(rp['read'])==set(P['read'])
    fw=set(P['items'][0]['cands']);print('reverse item 0:',its[0]['prefix'],'| gold',its[0]['cands'][0],'| 16 cands',len(its[0]['cands']));print('e59_rev selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
