"""Test 6: cached-context batched scoring == full-concatenation scoring (fp32, tiny model, real Qwen tokenizer and real
probe rows), margin sign convention (gold - foil) on a constructed example, canary NLL == explicit CE."""
from _testutil import *
import torch.nn.functional as F
import world_text as WT,readout as RO,inner as IN

mdir=QR.qwen_snapshot_dir();tok=WT.load_tokenizer(mdir/'tokenizer.json');bos=tok.token_to_id('<|endoftext|>')
W=WT.World(101,tok,bos,tail_len=7168,gen_len=300);rows=W.probe_rows()
pm=tiny(seed=0);randomize_lora_(pm,std=0.02)
sel=rows[:6]+rows[60:64]+rows[-6:]          # L3, XP/PS and L2 rows, different lengths
kv,L=RO.encode_context(pm,W.tail_ids,DEV)
for B in (1,4,16):
    c=RO.score_rows(pm,kv,L,sel,DEV,batch=B,pad_id=bos)
    f=[RO.score_full(pm,W.tail_ids,r,DEV) for r in sel]
    md=max(abs(x-y) for x,y in zip(c,f))
    check(f'6 cached-context scoring == full concatenation (batch {B}, {len(sel)} probe rows, context {L} tok)',md<=1e-4,f'max abs diff {md:.2e}')
# without context (ctx_kv None): equals scoring the row alone
c0=RO.score_rows(pm,None,0,sel[:4],DEV,batch=4);f0=[RO.score_full(pm,[],r,DEV) for r in sel[:4]]
check('6 no-context scoring == row alone',max(abs(x-y) for x,y in zip(c0,f0))<=1e-4)
# span covers exactly the candidate tokens
ids,(a,b)=rows[0];check('6 candidate span decodes to the gold continuation',tok.decode(ids[a:b]).strip()==W.items[0]['gold'].strip(),repr(tok.decode(ids[a:b])))
# sign convention on a constructed example: train the LoRA to prefer the gold candidate, margin must turn positive
it={'level':'L3','pair':'p0','tid':0,'prefix':'Nobody could say exactly where Tolek had grown up, though most guessed the town of','gold':' Ashmoor','foil':' Elmdale'}
slot=WT.make_slot(tok,None);rg=WT.item_ids(tok,slot,it['prefix'],it['gold']);rf=WT.item_ids(tok,slot,it['prefix'],it['foil'])
class _W:fam={'L3':[it]};tail_ids=W.tail_ids[:64];alt_ids=W.alt_ids[:64];gen_ids=W.gen_ids[:64]
E=RO.Evaluator(_W,[rg,rf],DEV,batch=2,pad_id=bos)
pm2=tiny(seed=0);m0=E.margins(pm2,_W.tail_ids)['L3'][0]
ctxrow=(_W.tail_ids+rg[0],(len(_W.tail_ids)+rg[1][0]-1,len(_W.tail_ids)+rg[1][1]-1))   # loss on the gold tokens after the context
for _ in range(30):IN.sgd_step(pm2,IN.packed_batch(IN.pack_rows([ctxrow],1024)[0],DEV),lr=0.5)
m1=E.margins(pm2,_W.tail_ids)['L3'][0]
gl=RO.score_rows(pm2,*RO.encode_context(pm2,_W.tail_ids,DEV),[rg,rf],DEV)
check('6 margin = logp(gold) - logp(foil): training toward gold raises it above 0',m1>m0 and m1>0 and abs(m1-(gl[0]-gl[1]))<1e-9,f'margin {m0:+.3f} -> {m1:+.3f}')
pm3=tiny(seed=0);ctxf=(_W.tail_ids+rf[0],(len(_W.tail_ids)+rf[1][0]-1,len(_W.tail_ids)+rf[1][1]-1))
for _ in range(30):IN.sgd_step(pm3,IN.packed_batch(IN.pack_rows([ctxf],1024)[0],DEV),lr=0.5)
m2=E.margins(pm3,_W.tail_ids)['L3'][0];check('6 training toward the foil drives the margin below 0',m2<0,f'margin {m0:+.3f} -> {m2:+.3f}')
# canary
x=torch.tensor(W.gen_ids)[None]
with torch.no_grad():ref=float(F.cross_entropy(pm(input_ids=x).logits[0,:-1].float(),x[0,1:]))
got=RO.nll(pm,W.gen_ids,DEV,piece=64);check('6 canary NLL == mean CE over len-1 predictions (pieced lm_head)',abs(got-ref)<=1e-5,f'{got:.6f} vs {ref:.6f}')
done()
