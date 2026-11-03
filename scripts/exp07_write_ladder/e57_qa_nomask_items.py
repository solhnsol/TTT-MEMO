"""E57 per-item table for the thread 'QA에서 가림이 안 되는 이유 규명' (new file, CPU only; stored rows of job 674 = e57_public_qa_stage13): one row per item of the items455 with the first / second token of the gold string and the n = 9 first-token changes of NATQ, ALLQ128 and GAINQ512 p.10.
Columns: id, type, gold (string), first_token_id, first_token_str, n_gold_tokens, W0_first_token_logprob (condition A, own question), dA_first / dB_first (state minus W0 of the gold first-token log-prob under conditions A / B) for NATQ, ALLQ128, GAINQ512P10, W0_first_token_rank (1 + number of the other 15 candidates whose first-token log-prob under W0 / A exceeds the gold's; 1 = best of 16), second_token_id / second_token_str / W0_second_token_logprob / dA_second per arm (null if the gold has a single token).
  python3 e57_qa_nomask_items.py --dir experiments/exp07_write_ladder/e57_public_qa_stage13 --out OUT.json"""
import sys,json,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_stream as S
STATES=(('NATQ','NATQ_n9'),('ALLQ128','ALLQ128_n9'),('GAINQ512P10','GAINQ512P10_n9'))
def run(dirp,out):
    d=pathlib.Path(dirp);tk=S.tokenizer();rm=json.load(open(d/'e57_pqa13_read_meta.json'));ids=rm['item_ids'];tok=json.load(open(d/'e57_pqa13_tokens.json'))
    ld=lambda c,l:json.load(open(d/f'E57_pqa13_rows_{c}_{l}.json'));R={(c,l):ld(c,l) for c in 'AB' for l in ('W0',)+tuple(s for _,s in STATES)}
    for (c,l),rows in R.items():assert [r['id'] for r in rows]==ids
    items=[]
    for k,i in enumerate(ids):
        w0=R[('A','W0')][k];t_ids=tok['A'][i][0];lp0=w0['cand_tok_lp'];g=lp0[0];firsts=[x[0] for x in lp0];row=dict(id=i,type=w0['type'],gold=w0['cand_surface'][0],first_token_id=int(t_ids[0]),first_token_str=tk.decode([int(t_ids[0])]),n_gold_tokens=len(t_ids),W0_first_token_logprob=float(g[0]),W0_first_token_rank=int(1+sum(1 for c in range(1,len(firsts)) if firsts[c]>g[0])))
        for name,st in STATES:
            a=R[('A',st)][k]['cand_tok_lp'][0];b=R[('B',st)][k]['cand_tok_lp'][0];b0=R[('B','W0')][k]['cand_tok_lp'][0];row[f'dA_first_{name}']=float(a[0]-g[0]);row[f'dB_first_{name}']=float(b[0]-b0[0])
        if len(t_ids)>1:
            row['second_token_id']=int(t_ids[1]);row['second_token_str']=tk.decode([int(t_ids[1])]);row['W0_second_token_logprob']=float(g[1])
            for name,st in STATES:row[f'dA_second_{name}']=float(R[('A',st)][k]['cand_tok_lp'][0][1]-g[1])
        else:
            row['second_token_id']=None;row['second_token_str']=None;row['W0_second_token_logprob']=None
            for name,st in STATES:row[f'dA_second_{name}']=None
        items.append(row)
    pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(out).write_text(json.dumps(dict(n_items=len(items),columns=list(items[0]),note='items455, job 674 rows (n = 9 states), original 16 candidates; dA / dB = state minus W0 of the gold token log-prob under condition A (own question) / B (question of another item); W0 rank of the gold first token among the 16 candidates under A',items=items),indent=0,default=float));return items
if __name__=='__main__':
    g=lambda k:sys.argv[sys.argv.index(k)+1];run(g('--dir'),g('--out'))
