"""E42 K2 (CPU only, no GPU): token-type composition of the masked sets S stored by the attention cells (S3_{attnsel,randattn,attnfill...}_s{seed}_{A,B}.saved.json: log[epoch].chunks[i] = {k, S (absolute stream positions, BOS = 0), n_masked, n_O, n_O_cand, tp}).
Every position of the pass's stream is classified, per chunk k (the classes depend on the chunk's own fact sentences):
  O                 Sets.oracle(k) positions (name/place tokens of earlier same-triple exposures of the chunk's fact sentences)
  B_rel/B_punct/B_other/B_entity   rest of those earlier same-triple exposure sentences (not O): relation words (alphabetic tokens between the first and the last entity token of the sentence), punctuation/whitespace, other words, name/place tokens that are not in O
  C_entity / C_rest the chunk's own fact sentences (place token in chunk k), not O: name+place tokens / the rest
  D_pair_entity/D_pair_rest         other exposure sentences of the SAME pair (other triple), name+place tokens / rest
  D_other_entity/D_other_rest       exposure sentences of OTHER pairs, name+place tokens / rest
  E_bg              background filler (everything that is not an exposure sentence)
  Z_pad             position 0 (BOS) and positions after the encoded text (padding)
Output per cell x group (chunks WITH O / chunks WITHOUT O): token counts and shares (sum 1) pooled over seed, world, pass, the mean of the per-(seed,world,pass) shares, and the EXPECTED share if the same number of positions were drawn uniformly from the candidate set of the chunk (positions >= 4, < chunk start - 128, inside the 8192 window; the attention cells' candidate rule), so over/under-selection = observed / expected.
Runs on the server (needs gen_world, suppress, tokenizer; no GPU): e42_k2_tokens.py --seeds 101,102,103 --cells attnsel=attnsel_p4 randattn=randattn_p4 [attnfill35=attnfill35] --in-dir $OUT --out-dir /tmp/e42_k2
Checks (per chunk, reported and counted, not aborting): sum of classes = n_masked; |O| = n_O; |S and O and candidates| = tp; the world/stream reproduces the run (a mismatch of n_O means the wrong world). `--selftest` runs the pure functions on a synthetic table."""
import os,sys,json,re,random,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import s3_sets as SETS
CH=SETS.CH;W=SETS.W;NCH=63;NTOK=NCH*CH
CLASSES=['O','B_rel','B_punct','B_other','B_entity','C_entity','C_rest','D_pair_entity','D_pair_rest','D_other_entity','D_other_rest','E_bg','Z_pad']
CI={c:i for i,c in enumerate(CLASSES)}
ROLES=['name','place','rel','punct','other']
def sentence_tokens(text,offs,s_start,s_end,name_pos,place_pos,ntok):
    """roles of the tokens of one sentence: {pos: role}. pos = index into the stream (BOS 0); offs[pos] = (char0, char1) in the realized text. Entity tokens come from the sentence table;
    relation = alphabetic token strictly between the first and the last entity token; punct = no alphanumeric character; other = the rest."""
    pos=[j for j in range(1,min(len(offs),ntok)) if offs[j][0]>=s_start and offs[j][1]<=s_end and offs[j][1]>offs[j][0]]
    ent=set(name_pos)|set(place_pos);out={}
    lo=min(ent) if ent else None;hi=max(ent) if ent else None
    for j in pos:
        tx=text[offs[j][0]:offs[j][1]]
        if j in place_pos:r='place'
        elif j in name_pos:r='name'
        elif not re.search(r'\w',tx):r='punct'
        elif ent and lo<j<hi:r='rel'
        else:r='other'
        out[j]=r
    for j in ent:out.setdefault(j,'place' if j in place_pos else 'name')
    return out
def pos_table(sents,n):
    """sents = [{'idx','t','pair','round','place':[..],'tokens':{pos:role}}] -> arrays sid[n] (-1 none) and role[n] (index into ROLES, -1 none); the pure input of classify_chunk."""
    sid=np.full(n,-1,np.int32);role=np.full(n,-1,np.int8)
    for i,s in enumerate(sents):
        for p,r in s['tokens'].items():
            if p<n:sid[p]=i;role[p]=ROLES.index(r)
    return sid,role
def classify_chunk(sents,sid,role,n,sets,k,n_text):
    """class code per stream position (CLASSES index) for chunk k. n_text = number of positions covered by the encoded text (BOS + tokens), the rest is padding."""
    cls=np.full(n,CI['E_bg'],np.int8);cls[0]=CI['Z_pad'];cls[n_text:]=CI['Z_pad']
    cur=[i for i,s in enumerate(sents) if k*CH<=s['place'][0]<(k+1)*CH];cur_t={sents[i]['t'] for i in cur};cur_pair={sents[i]['pair'] for i in cur}
    prior=[i for i,u in enumerate(sents) if any(u['t']==sents[c]['t'] and u['round']<sents[c]['round'] for c in cur)]
    ent=(role==ROLES.index('name'))|(role==ROLES.index('place'));has=sid>=0
    for i,u in enumerate(sents):
        m=sid==i
        if not m.any():continue
        e=ent&m;same=u['pair'] in cur_pair
        cls[m&~e]=CI['D_pair_rest'] if same else CI['D_other_rest'];cls[e]=CI['D_pair_entity'] if same else CI['D_other_entity']
    for i in prior:
        m=sid==i;r=role
        cls[m&(r==ROLES.index('rel'))]=CI['B_rel'];cls[m&(r==ROLES.index('punct'))]=CI['B_punct'];cls[m&(r==ROLES.index('other'))]=CI['B_other'];cls[m&ent]=CI['B_entity']
    for i in cur:
        m=sid==i;e=ent&m;cls[m&~e]=CI['C_rest'];cls[e]=CI['C_entity']
    O=sets.oracle(k)
    if O:cls[np.asarray(O,int)]=CI['O']
    return cls
def candidates(k,n):
    """the attention cells' candidate rule (s3_attn.cand_vec): positions >= 4, < chunk start - 128, inside the window before the chunk."""
    k0=k*CH;lo=max(0,k0-W);c=np.zeros(n,bool);a=max(lo,4);b=max(a,k0-128);c[a:b]=True;return c
def tally(cls,S,k,n):
    cnt=np.bincount(cls[np.asarray(S,int)],minlength=len(CLASSES)) if len(S) else np.zeros(len(CLASSES),int)
    cand=candidates(k,n);ex=np.bincount(cls[cand],minlength=len(CLASSES)).astype(float)
    return cnt,ex
def build_world_pass(gw,seed,ps):
    """as ladder.fs_streams: pass 1 = the world, pass p >= 2 = same texts with the background replaced by gw.background(Random(7000+p)) (cycled), padded with further pool paragraphs until the stream has NTOK tokens."""
    import copy
    w0=gw.build_world(seed)
    if ps==1:return w0
    w=copy.deepcopy(w0);pool=gw.background(random.Random(7000+ps));bi=0;new=[]
    for t in w['texts']:
        if t['kind']=='bg':new.append({'kind':'bg','text':pool[bi%len(pool)]});bi+=1
        else:new.append(t)
    w['texts']=new
    while len(gw.TOK.encode(gw.realize(w,False)[0],add_special_tokens=False).ids)+1<NTOK:
        w['texts'].append({'kind':'bg','text':pool[bi%len(pool)]});bi+=1
    return w
def real_tables(gw,SP,seed,ps,swap):
    w=build_world_pass(gw,seed,ps);tok=gw.TOK;text,_,rsp=gw.realize(w,swap);span_of={(x['t'],x['round']):(x['start'],x['end']) for x in rsp};e=tok.encode(text,add_special_tokens=False);offs=[(0,0)]+list(e.offsets);n_text=min(len(offs),NTOK)
    table=SETS.sentence_table(w,swap,tok,gw,SP);info={s['idx']:s for s in SP._sent_info(w,swap)};sents=[]
    for s in table:
        i=info[s['idx']];name_pos=set(s['names']);place_pos=set(s['place'])
        sents.append(dict(idx=s['idx'],t=s['t'],pair=s['pair'],round=s['round'],place=s['place'],names=s['names'],spans=s['spans'],text=s['text'],tokens=sentence_tokens(text,offs,*span_of.get((s['t'],s['round']),(i['start'],i['start']+len(i['text']))),name_pos,place_pos,NTOK)))
    return table,sents,n_text
def run(a):
    sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    inD=Path(a.in_dir);res={'cells':{},'checks':{},'args':vars(a)};cache={}
    agg={}   # (cell, group) -> {'obs':counts, 'exp':counts, 'units':[shares...], 'units_exp':[...]}
    chk={'chunks':0,'sum_mismatch':0,'nO_mismatch':0,'tp_mismatch':0}
    for lab,tag in (c.split('=') for c in a.cells):
        for seed in [int(x) for x in a.seeds.split(',')]:
            for wld in 'AB':
                f=inD/f'S3_{tag}_s{seed}_{wld}.saved.json'
                if not f.exists():res['checks'].setdefault('missing',[]).append(str(f));continue
                log=json.load(open(f))['log']
                for e in log:
                    if 'chunks' not in e:continue
                    ps=e['epoch'];key=(seed,ps,wld)
                    if key not in cache:
                        table,sents,n_text=real_tables(gw,SP,seed,ps,wld=='B');sid,role=pos_table(sents,NTOK);sets=SETS.Sets(table,seed,ps);cache[key]=(sents,sid,role,sets,n_text)
                    sents,sid,role,sets,n_text=cache[key];unit={}
                    for c in e['chunks']:
                        k=c['k'];S=np.asarray(c['S'],int);cls=classify_chunk(sents,sid,role,NTOK,sets,k,n_text);cnt,ex=tally(cls,S,k,NTOK)
                        O=sets.oracle(k);cand=candidates(k,NTOK);tp=len(set(S.tolist())&{p for p in O if p<NTOK and cand[p]})
                        chk['chunks']+=1;chk['sum_mismatch']+=int(cnt.sum()!=c['n_masked']);chk['nO_mismatch']+=int(len(O)!=c.get('n_O',len(O)));chk['tp_mismatch']+=int(tp!=c.get('tp',tp))
                        g='withO' if O else 'noO';d=agg.setdefault((lab,g),{'obs':np.zeros(len(CLASSES)),'exp':np.zeros(len(CLASSES)),'units':{},'units_exp':{}})
                        d['obs']+=cnt;u=unit.setdefault(g,[np.zeros(len(CLASSES)),np.zeros(len(CLASSES))]);u[0]+=cnt;u[1]+=ex*(cnt.sum()/max(ex.sum(),1))   # expected counts for the same number of positions drawn uniformly from the candidates
                        d['exp']+=ex*(cnt.sum()/max(ex.sum(),1))
                    for g,(o_,e_) in unit.items():
                        d=agg[(lab,g)];d['units'][(seed,wld,ps)]=o_/max(o_.sum(),1);d['units_exp'][(seed,wld,ps)]=e_/max(e_.sum(),1)
    for (lab,g),d in agg.items():
        sh=lambda v:(v/max(v.sum(),1)).tolist()
        U=np.mean(list(d['units'].values()),axis=0) if d['units'] else None;UE=np.mean(list(d['units_exp'].values()),axis=0) if d['units_exp'] else None
        res['cells'].setdefault(lab,{})[g]={'classes':CLASSES,'tokens_pooled':d['obs'].tolist(),'share_pooled':sh(d['obs']),'share_mean_over_units':None if U is None else U.tolist(),'expected_share_uniform_in_candidates':sh(d['exp']),
            'expected_share_mean_over_units':None if UE is None else UE.tolist(),'over_under_selection':[(o/e if e>0 else None) for o,e in zip(sh(d['obs']),sh(d['exp']))],'n_units':len(d['units'])}
    res['checks'].update(chk);return res
def md_of(res):
    L=['# E42 K2: token types of the masked sets (CPU)\n',f"checks: {json.dumps(res['checks'])}\n"]
    for lab,gs in res['cells'].items():
        for g,v in gs.items():
            L+=[f"## {lab}, chunks {'with' if g=='withO' else 'without'} an oracle set (units: {v['n_units']} seed x world x pass)\n",'| class | tokens (pooled) | share pooled | share mean over units | expected share (uniform in candidates) | observed / expected |','|---|---|---|---|---|---|']
            for i,c in enumerate(CLASSES):
                L.append(f"| {c} | {v['tokens_pooled'][i]:.0f} | {v['share_pooled'][i]:.4f} | {v['share_mean_over_units'][i]:.4f} | {v['expected_share_uniform_in_candidates'][i]:.4f} | {'-' if v['over_under_selection'][i] is None else format(v['over_under_selection'][i],'.2f')} |")
            L.append('')
    return '\n'.join(L)+'\n'
def selftest():
    """synthetic table (as e42_k6_test): each sentence = tokens [pos-5 .. pos+2] = other, other, name, name, rel, place(pos), punct, punct."""
    t=[]
    for tr in range(30):
        for r in range(1,5):
            pos=(r-1)*6144+tr*150+15;t.append(dict(idx=len(t),t=f't{tr}',pair=f'p{tr//2}',round=r,place=[pos],names=[pos-3,pos-2],spans=[[pos],[pos-3,pos-2]],text=''))
    t=sorted(t,key=lambda s:s['place'][0]);roles=['other','other','name','name','rel','place','punct','punct']
    sents=[dict(s,tokens={s['place'][0]-5+j:r for j,r in enumerate(roles)}) for s in t];n=NTOK;sid,role=pos_table(sents,n);sets=SETS.Sets(t,101,2);rng=np.random.default_rng(0);ok=True
    k=[k for k in range(8,NCH) if sets.oracle(k)][0];cls=classify_chunk(sents,sid,role,n,sets,k,n);O=sets.oracle(k);k0=k*CH
    ok&=all(cls[p]==CI['O'] for p in O)
    S=np.sort(rng.choice(np.arange(max(4,k0-W),k0-128),2000,replace=False));cnt,ex=tally(cls,S,k,n)
    ok&=cnt.sum()==len(S) and abs(ex.sum()-(k0-128-max(4,k0-W)))<1e-9                        # partition: the classes sum to the number of positions
    cur=[s for s in sents if k0<=s['place'][0]<k0+CH];ok&=all(cls[s['place'][0]-5+j] in (CI['C_rest'],CI['C_entity'],CI['O']) for s in cur for j in range(8))
    c0=cur[0];pri=[u for u in sents if u['t']==c0['t'] and u['round']<c0['round']]
    for u in pri:                                                                              # earlier same-triple sentence: rel/punct/other -> B_*, name/place -> O or B_entity
        p0=u['place'][0];ok&=cls[p0-1]==CI['B_rel'] and cls[p0+1]==CI['B_punct'] and cls[p0-5]==CI['B_other'] and cls[p0] in (CI['O'],CI['B_entity'])
    oth=[u for u in sents if u['pair']!=c0['pair'] and u['round']==1][0];ok&=cls[oth['place'][0]]==CI['D_other_entity'] and cls[oth['place'][0]-1]==CI['D_other_rest']
    ok&=cls[0]==CI['Z_pad'] and (cls==CI['E_bg']).sum()>0
    cnt2,_=tally(cls,np.asarray(O),k,n);ok&=cnt2[CI['O']]==len(O)
    r=sentence_tokens('aa bb cc dd ee.',[(0,0),(0,2),(3,5),(6,8),(9,11),(12,14),(14,15)],0,15,{2},{5},7);ok&=[r[j] for j in range(1,7)]==['other','name','rel','rel','place','punct']
    print("SELFTEST OK" if ok else "SELFTEST FAIL");sys.exit(0 if ok else 1)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101,102,103');ap.add_argument('--cells',nargs='+',default=['attnsel=attnsel_p4','randattn=randattn_p4'],help='LABEL=TAG, file S3_<TAG>_s<seed>_<world>.saved.json')
    ap.add_argument('--in-dir',default=os.environ.get('EXP07_OUT',str(ROOT/'experiments/exp07_write_ladder')));ap.add_argument('--out-dir',default='/tmp/e42_k2');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    res=run(a);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);json.dump(res,open(od/'e42_k2_tokens.json','w'),indent=1,default=float);md=md_of(res);(od/'e42_k2_tokens.md').write_text(md);print(md)
