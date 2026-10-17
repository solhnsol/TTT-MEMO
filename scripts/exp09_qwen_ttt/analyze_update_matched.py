"""Update-matched R for L1 variants (design request 10/17 12:49 KST; one method for Qwen and the 3B), CPU, stored logs only.
Method: cumulative update norm C (sum of per-step ||Delta W||, as logged). For a variant at readout epoch e_v with C_v, the plain
D_alt at the same C is linearly interpolated between the two plain readouts whose C bracket C_v (per world, then pooled per pair);
if C_v exceeds the largest plain readout C (extrapolation), the variant point moves to its largest in-range readout epoch.
R_um = mean D_alt(variant, e_v) / mean D_alt(plain at C_v), paired 15-pair bootstrap 95% CI.
Variant point: Qwen e* = 16 (registered), 3B the last variant epoch (8); both move down only if extrapolation requires.
Also: loss tokens per step and C per loss token (cumulative), and the familiarity reduction for L1-ctx (3B s102/s103).
Usage: analyze_update_matched.py"""
import os,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
def mods(out,seed,l1tag):
    os.environ['EXP07_OUT']=str(out);saved=sys.argv;sys.argv=['x',str(seed),l1tag]
    for m in ('analyze_ladder','analyze_stage01'):sys.modules.pop(m,None)
    try:import analyze_ladder as AL,analyze_stage01 as S
    finally:sys.argv=saved
    return AL,S
def cumC(d,tag,seed):
    """per world: {epoch: cumulative update norm at that readout}, steps per epoch, loss tokens per step (if logged)."""
    out={}
    for w in 'AB':
        L=json.load(open(Path(d)/f'L1_{tag}_s{seed}_{w}.json'))['log'];c={0:0.0};sp={};lt=[]
        for e in L:
            if 'margins' in e:continue
            c[e['epoch']]=e['cum_update_norm'];sp[e['epoch']]=sp.get(e['epoch'],0)+1
            if 'loss_tokens' in e:lt.append(e['loss_tokens'])
        out[w]=(c,sp,lt)
    return out
def ratio_ci(num,den,n=8000,seed=0):
    r=np.random.default_rng(seed);b=[];k=len(num)
    for _ in range(n):i=r.integers(0,k,k);b.append(num[i].mean()/den[i].mean())
    return float(num.mean()/den.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def family(name,out_of,seeds,plain_tag,variants,vpoint,loss_tokens_fn=None,plain_C_tag=None):
    res={}
    for sd in seeds:
        d=out_of(sd);AL,S=mods(d,sd,plain_tag);P=S.load('L1',plain_tag);cP=S.curve(*P);Pd={ep:sa['D'] for ep,s,sa,dd in cP};Pc=cumC(d,plain_C_tag or plain_tag,sd);Pf={ep:s['F_name'].mean()-s['F_null'].mean() for ep,s,sa,dd in cP}
        pe=sorted(Pd);res[sd]={}
        for v in variants:
            try:X=S.load('L1',v)
            except Exception:X=None
            if not X:continue
            cV=S.curve(*X);Vd={ep:sa['D'] for ep,s,sa,dd in cV};Vf={ep:s['F_name'].mean()-s['F_null'].mean() for ep,s,sa,dd in cV};Vc=cumC(d,v,sd)
            ev=vpoint if vpoint in Vd else max(Vd);moved=False
            while True:
                CvA=Vc['A'][0][ev];CvB=Vc['B'][0][ev];Cv=(CvA+CvB)/2;Cmax=min(Pc[w][0][max(pe)] for w in 'AB')
                if Cv<=Cmax or ev==min(e for e in Vd if e>0):break
                ev=max(e for e in Vd if e<ev);moved=True
            # interpolate plain D (per pair) at Cv using the A/B-mean plain cumulative norm at each readout
            pc=[(ep,(Pc['A'][0][ep]+Pc['B'][0][ep])/2) for ep in pe]
            lo=max([p for p in pc if p[1]<=Cv],key=lambda p:p[1]);hi=min([p for p in pc if p[1]>=Cv],key=lambda p:p[1]) if any(p[1]>=Cv for p in pc) else lo
            t=0.0 if hi[1]==lo[1] else (Cv-lo[1])/(hi[1]-lo[1]);Dp=(1-t)*Pd[lo[0]]+t*Pd[hi[0]];Fp=(1-t)*Pf[lo[0]]+t*Pf[hi[0]]
            r=ratio_ci(Vd[ev],Dp)
            lt=Vc['A'][2] or (loss_tokens_fn(sd,v) if loss_tokens_fn else None);ltp=Pc['A'][2] or (loss_tokens_fn(sd,plain_C_tag or plain_tag) if loss_tokens_fn else None)
            def per_tok(cc,ltok,ep_,steps):
                if not ltok:return None
                if isinstance(ltok,list):tot=sum(ltok[:sum(steps[e] for e in range(1,ep_+1))])
                else:tot=ltok*ep_
                return cc[ep_]/tot
            res[sd][v]={'variant_epoch':ev,'moved_for_extrapolation':moved,'C_variant':Cv,'plain_bracket_epochs':(lo[0],hi[0]),'t':t,'R_um':r,
                        'D_variant':float(Vd[ev].mean()),'D_plain_matched':float(Dp.mean()),'F_variant':float(Vf[ev]),'F_plain_matched':float(Fp),
                        'bind_red':1-r[0],'fam_red':1-float(Vf[ev])/float(Fp) if Fp else None,
                        'steps_per_epoch':Vc['A'][1].get(1),'plain_steps_per_epoch':Pc['A'][1].get(1),
                        'loss_tokens_per_step':(float(np.mean(lt)) if isinstance(lt,list) and lt else (lt/Vc['A'][1].get(1) if lt else None)),
                        'C_per_loss_token':per_tok(Vc['A'][0],lt,ev,Vc['A'][1]),'plain_C_per_loss_token_at_bracket':per_tok(Pc['A'][0],ltp,lo[0],Pc['A'][1]) if lo[0]>0 else None}
            x=res[sd][v];print(f"{name} s{sd} {v}: ep{ev}{' (moved: extrapolation)' if moved else ''} C {Cv:.2f} -> plain ep{lo[0]}-{hi[0]} t={t:.2f}; R_um {x['R_um'][0]:.3f} [{x['R_um'][1]:.3f},{x['R_um'][2]:.3f}]; D var {x['D_variant']:+.3f} vs plain-matched {x['D_plain_matched']:+.3f}; binding red {x['bind_red']:+.2f} familiarity red {x['fam_red'] if x['fam_red'] is None else round(x['fam_red'],2)}; steps/epoch {x['steps_per_epoch']} (plain {x['plain_steps_per_epoch']}); loss tok/step {x['loss_tokens_per_step']}; C/loss tok {x['C_per_loss_token']}",flush=True)
    return res
def threeb_loss_tokens(sd,variant):
    """3B loss tokens per epoch (rows' loss spans, ladder.l1_rows replicated with the llama tokenizer); per step = / steps."""
    import gen_world as gw,stage2
    w=gw.build_world(sd);tok=gw.TOK;tot=0
    for swap in (False,):
        _,_,spans=gw.realize(w,swap);enc=lambda t:stage2.enc(tok,t).ids;tri={t['id']:t for t in w['triples']}
        if variant in ('main','plain','fill'):tot+=sum(len(enc('\n\n'+s_['text']))-1 for s_ in spans)
        if variant=='fill':tot+=sum(min(len(enc('\n\n'+t['text'])),1024)-1 for t in w['texts'] if t['kind']=='bg')
        if variant in ('ctx','irr'):tot+=sum(len(enc('\n\n'+s_['text'])) for s_ in spans)-sum(1 for s_ in spans if s_['round']<2)
    return tot
def main():
    E=ROOT/'experiments';res={}
    res['qwen']=family('Qwen',lambda sd:E/f'exp09_qwen_ttt/l1ci_pod_s{sd}',(101,102,103),'plain',('ctx','irr'),16)
    res['3b']=family('3B',lambda sd:E/'exp07_write_ladder',(101,102,103),'ck',('ctx','irr','fill'),8,threeb_loss_tokens,plain_C_tag='main')   # 3B plain: D from L1_ck (R1-alt re-eval of the main run's checkpoints), C from L1_main's step log
    json.dump(res,open(E/'exp09_qwen_ttt/update_matched_R.json','w'),indent=1,default=str)
    irr3=[res['3b'][sd]['irr']['R_um'][0] for sd in res['3b'] if 'irr' in res['3b'][sd]]
    print('3B update-matched irr R >= 0.70 on',sum(r>=0.70 for r in irr3),'/',len(irr3),'seeds ->','family difference' if sum(r>=0.70 for r in irr3)>=2 else 'same picture in both families')
if __name__=='__main__':main()
