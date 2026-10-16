"""Test 11: Qwen long-context NLL (qwen_ladder.Run.long_nll, the exp07 Q2 analogue) on the tiny random model (CPU).
(a) the LoRA state and the frozen weights are unchanged after long_nll; (b) chunk 0's NLL equals a plain forward of chunk 0
at the current weights (no cache); (c) the beyond-window mean is the token-weighted
mean of chunks >= 8192/CH; (d) a repeat gives identical values; (e) the document text matches the 3B Q2 preparation (BOS first)."""
from _testutil import *
import argparse,copy,torch.nn.functional as F
import qwen_ladder as QL,world_text as WT,inner as IN
a=QL.parser().parse_args(['--level','L0mp','--tiny','--device','cpu','--chunk','128','--longnll','--longnll-chunks','67','--lr','1e-2',
                         '--tail-len','256','--canary-len','256','--readout-batch','16','--world-seed','101'])
R=QL.Run(a);randomize_lora_(R.model,seed=3)
h0=QR.lora_hash(R.model);fh=QR.frozen_hash(R.model,R.watch)
r=R.long_nll()
check('11a LoRA state and frozen weights unchanged after long_nll',QR.lora_hash(R.model)==h0 and QR.frozen_hash(R.model,R.watch)==fh)
doc=R.doc;CH=a.chunk
with torch.no_grad():
    dec,head=IN.parts(R.model);ids=torch.tensor(doc[:CH])[None];hs=dec(input_ids=ids,position_ids=torch.arange(CH)[None]).last_hidden_state[0]
    nll0=float(F.cross_entropy(head(hs).float(),torch.tensor(doc[1:CH+1])))
check('11b chunk-0 NLL equals a plain forward at the current weights',abs(nll0-r['ce'][0])<1e-5,f"{r['ce'][0]:.6f} vs {nll0:.6f}")
w=8192//CH;n=[CH]*(len(r['ce'])-1)+[CH-1]
bey=sum(c*k for c,k in zip(r['ce'][w:],n[w:]))/sum(n[w:]) if len(r['ce'])>w else None
check('11c beyond = token-weighted mean of chunks >= 8192/CH (3 chunks here), all over every chunk',bey is not None and abs(bey-r['beyond'])<1e-9 and len(r['ce'])-w==3,f"w={w}, n_chunks={len(r['ce'])}")
r2=R.long_nll()
check('11d repeat gives the same values (CPU, restored state)',max(abs(x-y) for x,y in zip(r['ce'],r2['ce']))<1e-6)
t=(ROOT/'data/narrative_gutenberg/pg2160.txt').read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t;t=' '.join(t.replace('\r','').split());t=t[len(t)//3:]
check('11e document = the 3B Q2 text preparation, BOS first',doc[0]==R.bos and R.tok.decode(doc[1:40]).strip()[:20]==t[:60].strip()[:20],repr(R.tok.decode(doc[1:12])))
done()
