"""L1-ctx/irr manipulation check (PROTOCOL "Qwen L1-ctx / L1-irr v1"; forward only, W0): the current sentence's place-token NLL
under plain (sentence alone), ctx (previous round's two pair sentences before it) and irr (same-length other-pair context), for
exposure sentences of rounds >= 2 (round 1 has no context), worlds A and B. fp32. Writes OUT/manip_s{seed}.json with per-sentence
NLLs. Usage: python manip_ctxirr.py --seed 101 --out DIR"""
import sys,json,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import torch,torch.nn.functional as F
import world_text as WT,qwen_runtime as QR
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    mdir=QR.resolve_model(QR.BASE_ID);tok=WT.load_tokenizer(QR.tokenizer_file(mdir));bos=tok.token_to_id('<|endoftext|>')
    m=QR.load_base(mdir,torch.device('cuda'),torch.float32).eval();W=WT.World(a.seed,tok,bos);res={}
    import gen_world as gw
    for wn,sw in (('A',False),('B',True)):
        _,_,spans=gw.realize(W.w,sw);tri={t['id']:t for t in W.w['triples']}
        rows={v:W.l1_rows(sw,v) for v in ('plain','ctx','irr')};out=[]
        for i,s_ in enumerate(spans):
            if s_['round']<2:continue
            P=tri[s_['t']]['P'];txt='\n\n'+s_['text'];e=tok.encode(txt,add_special_tokens=False);c0=txt.index(P);pt=[j for j,(x,y) in enumerate(e.offsets) if y>c0 and x<c0+len(P)]
            r={'i':i,'round':s_['round']}
            for v in ('plain','ctx','irr'):
                ids,(lo,hi)=rows[v][i];off=len(ids)-len(e.ids);assert ids[off:]==e.ids
                x=torch.tensor([ids],device='cuda')
                with torch.no_grad():lp=F.log_softmax(m(input_ids=x).logits[0].float(),-1)
                r[v]=float(-sum(lp[off+j-1,ids[off+j]] for j in pt)/len(pt))
            out.append(r)
        res[wn]=out;print(wn,len(out),'sentences',flush=True)
    Path(a.out).mkdir(parents=True,exist_ok=True);json.dump(res,open(Path(a.out)/f'manip_s{a.seed}.json','w'),indent=1)
if __name__=='__main__':main()
