"""Gate: two independent processes (deterministic ops) must agree on gradient/update, weight hash and margins for the first chunks."""
import json,sys,os
from pathlib import Path
import numpy as np
OUT=Path(os.environ['EXP05_OUT'])
a=json.load(open(OUT/'traj_seed101_chk1_A.json'));b=json.load(open(OUT/'traj_seed101_chk2_A.json'))
na=np.load(OUT/'traj_seed101_chk1_A_tok_nll.npz')['tok_nll'].astype(np.float32);nb=np.load(OUT/'traj_seed101_chk2_A_tok_nll.npz')['tok_nll'].astype(np.float32)
res={}
res['hashes_identical']=a['hashes']==b['hashes'];res['n_states']=len(a['hashes'])
res['max_grad_norm_diff']=max(abs(x['grad_norm_preclip']-y['grad_norm_preclip']) for x,y in zip(a['rows'],b['rows']))
res['max_update_norm_diff']=max(abs(x['update_norm']-y['update_norm']) for x,y in zip(a['rows'],b['rows']))
res['max_token_nll_diff']=float(np.abs(na-nb).max())
res['max_margin_diff']=max(abs(a['margins_state'][s][p]-b['margins_state'][s][p]) for s in a['margins_state'] for p in a['margins_state'][s])
res['max_ctrl_margin_diff']=max(abs(a['ctrl_state'][s][p]-b['ctrl_state'][s][p]) for s in a['ctrl_state'] for p in a['ctrl_state'][s])
res['V2_max_rel_err']=max(max(x['v2_rel_err'] for x in a['rows']),max(x['v2_rel_err'] for x in b['rows']))
res['xla_flags']=[a.get('xla_flags'),b.get('xla_flags')]
ok=res['hashes_identical'] and res['max_margin_diff']<=1e-6 and res['max_ctrl_margin_diff']<=1e-6 and res['max_grad_norm_diff']==0 and res['max_token_nll_diff']==0 and res['V2_max_rel_err']<=1e-6 and 'deterministic_ops=true' in (a.get('xla_flags') or '')
res['PASS']=bool(ok);json.dump(res,open(OUT/'short_gate.json','w'),indent=1);print(json.dumps(res,indent=1))
sys.exit(0 if ok else 1)
