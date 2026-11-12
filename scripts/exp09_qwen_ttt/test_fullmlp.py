"""Test 12: full-MLP fast weights (qwen_runtime.wrap_fullmlp; exploratory run registered 2026-10-16 23:48 KST), CPU, tiny Qwen3.
(a) exactly the gate/up/down weights of the last N layers are trainable, fp32; (b) W0 = the base model (identical logits);
(c) one clipped SGD step moves only the fast weights and equals the hand-computed step; (d) a bf16 base runs with the fp32
fast modules (activation dtype preserved); (e) qwen_ladder.py --fast fullmlp L0mp and L1 smoke runs."""
from _testutil import *
import copy,subprocess,shutil,json
import inner as IN
torch.manual_seed(0)
base=QR.tiny_qwen(seed=0);ref=copy.deepcopy(base)
ids=torch.randint(0,1000,(1,64))
with torch.no_grad():l0=ref(input_ids=ids).logits
m,cfg=QR.wrap_fullmlp(base,('gate_proj','up_proj','down_proj'),2)
tr=[n for n,p in m.named_parameters() if p.requires_grad]
check('12a trainable = gate/up/down weights of the last 2 layers, fp32',sorted(tr)==sorted(f'model.layers.{i}.mlp.{t}.weight' for i in (2,3) for t in ('gate_proj','up_proj','down_proj')) and all(p.dtype==torch.float32 for n,p in m.named_parameters() if p.requires_grad),str(len(tr)))
with torch.no_grad():l1=m(input_ids=ids).logits
check('12b W0 = base (identical logits)',torch.equal(l0,l1))
fz={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad};fa={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
m.zero_grad();out=m(input_ids=ids).logits[0,:-1];loss=torch.nn.functional.cross_entropy(out.float(),ids[0,1:]);loss.backward()
g={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.requires_grad};gn=torch.sqrt(sum((x.double()**2).sum() for x in g.values())).item();coef=min(1.0,1.0/gn)
with torch.no_grad():
    for n,p in m.named_parameters():
        if p.requires_grad:p.add_(-0.1*coef*g[n]);p.grad=None
mv=all(torch.equal(p,fz[n]) for n,p in m.named_parameters() if not p.requires_grad)
st=max((p-(fa[n]-0.1*coef*g[n])).abs().max().item() for n,p in m.named_parameters() if p.requires_grad)
check('12c step moves only the fast weights, equals the clipped SGD step',mv and st<1e-7,f'gnorm {gn:.3f}, coef {coef:.3f}, maxdiff {st:.1e}')
QR.FAST_NAMES=None
b16=QR.tiny_qwen(seed=1).to(torch.bfloat16);m16,_=QR.wrap_fullmlp(b16,('gate_proj','up_proj','down_proj'),2)
with torch.no_grad():o=m16(input_ids=ids).logits
check('12d bf16 base with fp32 fast MLP runs, activations stay bf16',o.dtype==torch.bfloat16 and torch.isfinite(o.float()).all().item())
QR.FAST_NAMES=None
SM=ROOT/'experiments/exp09_qwen_ttt/smoke_fullmlp'
if SM.exists():shutil.rmtree(SM)
common=['--tiny','--device','cpu','--max-chunks','3','--chunk','128','--passes','1','--epochs','2','--tail-len','256','--canary-len','256','--readout-batch','16','--out',str(SM),'--world-seed','101','--lr','1e-2','--threads','8','--fast','fullmlp','--lora-layers','2']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
for lev in ('L0mp','L1'):
    p=subprocess.run([sys.executable,str(HERE/'qwen_ladder.py'),'--level',lev]+common,env=env,capture_output=True,text=True)
    ok=p.returncode==0
    if ok:
        d=json.load(open(SM/f'{lev}_main_s101_A.json'))['log'];ev=[e for e in d if 'margins' in e];ok=ev[0]['lora_hash']!=ev[-1]['lora_hash']
    check(f'12e qwen_ladder --fast fullmlp {lev} runs and moves the fast weights',ok,p.stderr[-600:] if p.returncode else '')
done()
