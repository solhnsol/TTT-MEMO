"""Identity screen (DESIGN v2 R7), HF causal LM version (Qwen3-4B-Base). GPU, short.
For every target: redact its first exposure unit (all name placeholders -> "[name]"), append a question, and rank the real
surname among ~1,000 common surnames + the real one by total log-prob. Exclude the target's pair if rank <= 10 here OR in
the 3B W0 screen (union; the JAX version is the lab server's to write with the same inputs/score).
Needs the PRIVATE map (real names). Never commit it; copy it to the GPU box by hand and delete it afterwards.
python identity_screen.py --model Qwen/Qwen3-4B-Base --worlds data/natural_text --private PATH/real_name_map.json --out screen_qwen.json"""
import argparse,json,glob,os,re
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ap=argparse.ArgumentParser();ap.add_argument('--model',default='Qwen/Qwen3-4B-Base');ap.add_argument('--worlds',required=True)
ap.add_argument('--private',required=True);ap.add_argument('--out',required=True);ap.add_argument('--batch',type=int,default=64);a=ap.parse_args()
priv=json.load(open(a.private));common=json.load(open(os.path.join(a.worlds,'identity_screen_surnames.json')))['common_surnames']
tok=AutoTokenizer.from_pretrained(a.model);m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16).cuda().eval()
Q="\n\nQuestion: what is the real family name of the person called [name] above?\nAnswer: The family name is"
@torch.no_grad()
def score(ctx,cands):
    """total log-prob of ' '+cand after ctx; full forward per batch (no cache tricks), right padding."""
    c_ids=tok(ctx,add_special_tokens=False).input_ids;res=[]
    for i in range(0,len(cands),a.batch):
        cb=[tok(' '+c,add_special_tokens=False).input_ids for c in cands[i:i+a.batch]];L=max(map(len,cb))
        x=torch.tensor([c_ids+c+[0]*(L-len(c)) for c in cb]).cuda();att=torch.tensor([[1]*(len(c_ids)+len(c))+[0]*(L-len(c)) for c in cb]).cuda()
        lp=m(x,attention_mask=att).logits.float().log_softmax(-1);n=len(c_ids)
        for j,c in enumerate(cb):res.append(sum(lp[j,n+t-1,c[t]].item() for t in range(len(c))))
    return res
seen={};out=[]
for f in sorted(glob.glob(os.path.join(a.worlds,'nat_*_spread.json'))):
    w=json.load(open(f))
    for t in w['triples']:
        key=f"{w['seed']}:{t['id']}";title=priv[key]['title'];real=re.sub(r'\s*\(.*\)$','',title).split()[-1]
        if title in seen:continue
        unit=next(x['text'] for x in w['texts'] if x['kind']=='expo' and x['t']==t['id'])
        ctx=re.sub(r'@X[FS]?\d+@','[name]',unit)+Q
        cands=[c for c in common if c!=real]+[real];sc=score(ctx,cands);r=1+sum(s>sc[-1] for s in sc[:-1])
        seen[title]=r;out.append({'key':key,'pair':t['pair'],'stream':w['seed'],'rank':r,'exclude':r<=10})
json.dump(out,open(a.out,'w'),indent=0)
print('targets',len(out),'excluded (rank<=10)',sum(x['exclude'] for x in out))
