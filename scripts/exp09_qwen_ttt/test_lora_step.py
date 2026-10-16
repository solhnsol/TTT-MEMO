"""Tests 1-3: W0 identity (B=0 LoRA == unwrapped model), one inner step == hand-computed clipped SGD step, only LoRA
params change (frozen hash unchanged)."""
from _testutil import *
import inner as IN

ids=torch.randint(0,151000,(1,300),generator=torch.Generator().manual_seed(3))
# ---- 1. W0 identity ----
base=QR.tiny_qwen(seed=0)
with torch.no_grad():ref=base(input_ids=ids).logits.clone()
pm=tiny(seed=0)
with torch.no_grad():got=pm(input_ids=ids).logits
d=float((ref-got).abs().max());check('1 W0 identity: LoRA(B=0) logits == base logits',d==0.0,f'max abs diff {d:.3e}')
nb=[n for n,p in QR.lora_params(pm) if 'lora_B' in n];check('1 LoRA B initialised to zero',all(float(p.detach().abs().max())==0 for n,p in QR.lora_params(pm) if 'lora_B' in n),f'{len(nb)} B tensors')
lay=sorted({int(n.split('.layers.')[1].split('.')[0]) for n,_ in QR.lora_params(pm)})
check('1 LoRA only on the last N layers (tiny: last 2 of 4), MLP projections only',lay==[2,3] and all(any(t in n for t in ('gate_proj','up_proj','down_proj')) for n,_ in QR.lora_params(pm)),str(lay))
check('1 LoRA params fp32',all(p.dtype==torch.float32 for _,p in QR.lora_params(pm)))

# ---- 2. one inner step == hand-computed ----
def hand(pm,batch,lr,scale,clip):
    ps=[p for _,p in QR.lora_params(pm)];old=[p.detach().clone() for p in ps]
    loss=IN.loss_of(pm,batch);g=torch.autograd.grad(loss,ps)
    gn=torch.sqrt(sum((x**2).sum() for x in g));coef=1.0 if gn<clip else clip/gn
    return [o-lr*scale*coef*x for o,x in zip(old,g)],float(gn)
stream=torch.randint(0,151000,(5*64,),generator=torch.Generator().manual_seed(4)).tolist()
for tag,lr,scale,clip in (('clipped',1e-2,2.0,1e-3),('unclipped',1e-2,1.0,1e6)):
    pm=tiny(seed=0);randomize_lora_(pm)
    b=IN.native_batch(stream,3,64,2,DEV)
    exp,gn=hand(pm,b,lr,scale,clip)
    st=IN.sgd_step(pm,b,lr,scale,clip)
    md=max(float((p-e).abs().max()) for (_,p),e in zip(QR.lora_params(pm),exp))
    rel=max(float((p-e).abs().max()/(e.abs().max()+1e-30)) for (_,p),e in zip(QR.lora_params(pm),exp))
    cc=(st['clip_coef']<1.0) if tag=='clipped' else (st['clip_coef']==1.0)
    check(f'2 inner step == param - lr*scale*clipcoef*grad ({tag}, grad norm {gn:.3g}, coef {st["clip_coef"]:.3g})',md<=1e-7 and cc,f'max abs diff {md:.2e}, rel {rel:.1e}')

# first step from B=0: only B moves (dL/dA = 0 when B = 0)
pm=tiny(seed=0);s0=QR.lora_state(pm);IN.sgd_step(pm,IN.masked_batch(stream,0,64,DEV),1e-2)
mvA=max(float((p.detach()-s0[n]).abs().max()) for n,p in QR.lora_params(pm) if 'lora_A' in n)
mvB=max(float((p.detach()-s0[n]).abs().max()) for n,p in QR.lora_params(pm) if 'lora_B' in n)
check('2 first step from W0 moves B only (standard LoRA)',mvA==0.0 and mvB>0,f'|dA| {mvA:.1e} |dB| {mvB:.1e}')

# ---- 3. only LoRA params change ----
pm=tiny(seed=0);names=QR.watch_names(pm,all_frozen=True);h0=QR.frozen_hash(pm,names);l0=QR.lora_hash(pm)
for k in range(4):IN.sgd_step(pm,IN.native_batch(stream,k,64,2,DEV),5e-2)
for s_ in IN.pack_rows([(stream[i:i+20],(0,19)) for i in range(0,200,20)],64):IN.sgd_step(pm,IN.packed_batch(s_,DEV),5e-2)
check('3 frozen hash unchanged after native + packed steps',QR.frozen_hash(pm,names)==h0,f'{len(names)} frozen tensors')
check('3 LoRA params did change',QR.lora_hash(pm)!=l0)
check('3 no frozen param requires grad',all(not p.requires_grad for n,p in pm.named_parameters() if not QR.is_lora(n)))
# fp32 readout cast round trip is exact (bf16 base)
pmb=tiny(seed=0);QR.cast_base_(pmb,torch.bfloat16);hb=QR.frozen_hash(pmb,QR.watch_names(pmb,True))
QR.cast_base_(pmb,torch.float32);QR.cast_base_(pmb,torch.bfloat16)
check('3 bf16 -> fp32 -> bf16 base cast (fp32 readout option) is exact; LoRA stays fp32',QR.frozen_hash(pmb,QR.watch_names(pmb,True))==hb and all(p.dtype==torch.float32 for _,p in QR.lora_params(pmb)))
done()
