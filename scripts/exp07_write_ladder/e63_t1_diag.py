"""Diagnostic table of the stored runs of jobs 1344t0 (P0s, P1s) and 1344t1 (S2As-u, E2As) (CPU; new file; thread-server 11/25; reads only the saved json files, NOTHING is recomputed; design thread (A of the follow-up bundle)). Per arm: interval means over chunks 1-16, 17-32, 33-64, 65-96, 97-128 of the clip coefficient, the pre-clip gradient norm and the original chunk CE (mean over seeds, per-seed min..max in brackets); every 8 chunks the place-token (2 tokens of each read fact) NLL under the C0 mask and under the SAMEVIS mask, ||e_C0||, and the correct logit under SAMEVIS (stored only for the 1344t1 arms; the 1344t0 arms did not store it), median over the tokens of a seed, mean over seeds; also by the sentence order of the fact (place first / name first). The place-token order of each stored token is recovered from the world of the seed (deterministic: e59_u12 + e59_leak_probe.answer_positions).
  python e63_t1_diag.py --kv KVMULTI_DIR --kv2 KVMULTI2_DIR --seeds 231,232,233,234 [--out FILE]"""
import sys,os,json,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
a=sys.argv;g=lambda k,d=None:a[a.index(k)+1] if k in a else d
KV=g('--kv');KV2=g('--kv2');SEEDS=[int(x) for x in g('--seeds','231,232,233,234').split(',')];OUT=g('--out')
import e59_u12 as U,e55_run as R55,e57_syn_pass as SP,e59_leak_probe as LP
U.install();tk=LP.tk_load();CH=1024
def token_orders(seed):
    cache=pathlib.Path(KV2)/f'token_orders_s{seed}.json'
    if cache.exists():return {int(k):v for k,v in json.load(open(cache)).items()}
    enc,w,tf=R55.build(SP.synth_args(seed,U.CELL),seed,read_apps={16,32,64,128},arms=['NAT']);ap,miss,tot=LP.answer_positions(w,tk,128);assert miss==0
    res={}
    for k in range(7,128,8):
        d={}
        for x in ap[k]:
            if x['fact']<16:
                for t in x['place']:d[t-1]=0 if x['order']=='place_first' else 1
        res[k+1]=d
    json.dump({str(k):{str(i):o for i,o in d.items()} for k,d in res.items()},open(cache,'w'));return res
ARMS=[('P0s',KV,'kvmulti_s{s}.json','SAMEVIS','C0'),('S2As-u',KV2,'kvmulti2_s{s}.json','SAMEVIS','C0'),('E2As',KV2,'kvmulti2_s{s}.json','SAMEVIS','C0'),('P1s',KV,'kvmulti_s{s}.json','C0','SAMEVIS')]
R={s:{'kv':json.load(open(f'{KV}/kvmulti_s{s}.json')),'kv2':json.load(open(f'{KV2}/kvmulti2_s{s}.json'))} for s in SEEDS};TO={s:token_orders(s) for s in SEEDS}
out=[];P=lambda *x:out.append(' '.join(str(y) for y in x))
P(f'1344t1 diagnostics from the stored runs; seeds {SEEDS}; mean over seeds [per-seed min..max]; "-" = not stored')
INT=[(1,16),(17,32),(33,64),(65,96),(97,128)]
def rec(arm,s):return R[s]['kv']['arms'][arm] if arm in ('P0s','P1s') else R[s]['kv2']['arms'][arm]
P('== per-chunk values, interval means over chunks: clip coefficient | pre-clip gradient norm | original chunk CE')
for key,nm in (('coef','clip coefficient'),('gnorm','pre-clip gradient norm'),('ce','original chunk CE')):
    P(f'  [{nm}]')
    for arm,*_ in ARMS:
        row=[]
        for lo,hi in INT:
            v=np.array([np.mean([x[key] for x in rec(arm,s)['steps'][lo-1:hi]]) for s in SEEDS]);row.append(f'c{lo}-{hi}: {v.mean():.3f} [{v.min():.3f}..{v.max():.3f}]')
        P(f'    {arm:7s}: '+' | '.join(row))
P('== every 8 chunks, place tokens (before the write step of that chunk, weights of the arm): NLL under C0 mask | NLL under SAMEVIS mask | ||e_C0|| | correct logit (SAMEVIS); median over tokens of a seed, mean over seeds')
def vals(arm,ck,s,own_mask,key_c0,key_sv,token_filter=None):
    d=rec(arm,s)['diag'].get(str(ck))
    if d is None:return None
    own,oth=d['cond_own'],d['cond_other'];m={own:'own',oth:'other'}
    ix=np.array(d['idx']);sel=np.ones(len(ix),bool) if token_filter is None else np.array([TO[s][ck].get(int(i),-1)==token_filter for i in ix])
    if not sel.any():return None
    def pick(keys):
        k=keys[0]+'_'+('own' if own==own_mask else 'other');return np.array(d[k])[sel]
    out_={'nll_c0':np.median(np.array(d['nll_own' if own=='C0' else 'nll_other'])[sel]),'nll_sv':np.median(np.array(d['nll_own' if own=='SAMEVIS' else 'nll_other'])[sel]),'en_c0':np.median(np.array(d['en_own' if own=='C0' else 'en_other'])[sel]),'logit':(np.median(np.array(d['gold_logit_sv'])[sel]) if 'gold_logit_sv' in d else np.nan)}
    return out_
for tf,tn in ((None,'all place tokens'),(0,'fact sentence place first'),(1,'fact sentence name first')):
    P(f'  -- {tn}')
    for arm,*_ in ARMS:
        P(f'    {arm}:')
        for ck in range(8,129,8):
            vs=[vals(arm,ck,s,None,None,None,tf) for s in SEEDS];vs=[v for v in vs if v]
            if not vs:continue
            mm=lambda k:np.mean([v[k] for v in vs]);lg='-' if not np.isfinite(mm('logit')) else f"{mm('logit'):6.2f}"
            P(f'      chunk {ck:3d}: NLL C0 {mm("nll_c0"):6.2f} | NLL SAMEVIS {mm("nll_sv"):7.4f} | ||e_C0|| {mm("en_c0"):6.3f} | correct logit {lg}'+('' if tf is not None else f'   (n seeds {len(vs)})'))
txt='\n'.join(out);print(txt)
if OUT:open(OUT,'w').write(txt+'\n')
