"""Test 8: end-to-end smoke of qwen_ladder.py on the tiny random model (CPU): truncated stream (3 chunks of 128 tokens),
2 passes / 2 epochs, both worlds, levels L0mp, L0mw, L1 (+ L0mwfs, L0 and the L1 ctx variant); JSON in the exp07 log
format; the exp07 analysis code (analyze_ladder.stats via analyze_stage01.load/curve) consumes it."""
from _testutil import *
import json,shutil,subprocess,os
SM=ROOT/'experiments/exp09_qwen_ttt/smoke'
if SM.exists():shutil.rmtree(SM)
common=['--tiny','--device','cpu','--max-chunks','3','--chunk','128','--passes','2','--epochs','2','--tail-len','256','--canary-len','256',
        '--readout-batch','16','--out',str(SM),'--world-seed','101','--lr','1e-2','--eval-every','2','--threads','8']
runs=[['--level','L1'],['--level','L0mp'],['--level','L0mw'],['--level','L0mwfs'],['--level','L0'],['--level','L1','--variant','ctx','--tag','ctx'],['--level','L0mp','--native-ctx','recompute','--tag','rc']]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
for r in runs:
    p=subprocess.run([sys.executable,str(HERE/'qwen_ladder.py')]+r+common,env=env,capture_output=True,text=True)
    tail='\n'.join(p.stdout.strip().splitlines()[-3:])
    check(f'8 qwen_ladder {" ".join(r)} ran',p.returncode==0,(p.stderr[-800:] if p.returncode else tail.replace('\n',' | ')))
FAMS=['L3','XP','PS','FN','FP','FN0','FP0','L3r','L2']
meta=json.load(open(SM/'meta_L1_main_s101.json'))
mp=json.load(open(SM/'meta_L0mp_main_s101.json'))
check('8 defaults: Base model id, detached native write, fp32 readout, 63-chunk stream',mp['model_id']=='Qwen/Qwen3-4B-Base' and mp['args']['native_ctx']=='detached' and mp['args']['readout_dtype']=='fp32' and mp['args']['stream_chunks']=='63')
rc=json.load(open(SM/'L0mp_rc_s101_A.json'))['log'];dt=json.load(open(SM/'L0mp_main_s101_A.json'))['log']
check('8 detached and recompute native modes give different weights after chunk 1 (different write context)',rc[-1]['lora_hash']!=dt[-1]['lora_hash'] and abs(rc[1]['ce'][0]-dt[1]['ce'][0])<1e-5,f"chunk-0 ce {dt[1]['ce'][0]:.5f} vs {rc[1]['ce'][0]:.5f} (identical: no context yet)")
check('8 meta has items and fam_sizes as ladder.py writes',list(meta['fam_sizes'])==FAMS and len(meta['items'])==sum(meta['fam_sizes'].values())==276)
for lev,tag,npts,key in (('L1','main',3,'epoch'),('L0mp','main',3,'epoch'),('L0mw','main',3,'epoch'),('L0mwfs','main',3,'epoch'),('L0','main',3,'state'),('L1','ctx',3,'epoch'),('L0mp','rc',3,'epoch')):
    ok=True;why=[]
    for w in 'AB':
        d=json.load(open(SM/f'{lev}_{tag}_s101_{w}.json'));ev=[e for e in d['log'] if 'margins' in e]
        if len(ev)!=npts:ok=False;why.append(f'{w}: {len(ev)} eval points')
        for e in ev:
            m=e['margins']
            if key not in e or any(len(m[f])!=meta['fam_sizes'][f] for f in FAMS) or set(m['_alt'])!=set(FAMS) or not isinstance(m['_gen_nll'],float) or not isinstance(m['_tail_nll'],float):ok=False;why.append('format')
        if ev[-1]['lora_hash']==ev[0]['lora_hash']:ok=False;why.append('weights did not move')
    check(f'8 {lev}_{tag} A/B logs: {npts} eval points, every family + _alt + canaries, weights moved',ok,', '.join(why))
# the A/B worlds differ in the stream -> different weights
import world_text as WT
tok=WT.load_tokenizer(QR.qwen_snapshot_dir()/'tokenizer.json');Wd=WT.World(101,tok,tok.token_to_id('<|endoftext|>'),tail_len=256,gen_len=256)
same=Wd.stream_ids(False)[:384]==Wd.stream_ids(True)[:384]
L=lambda lev,w:json.load(open(SM/f'{lev}_main_s101_{w}.json'))['log']
mpA,mpB,l1A,l1B=L('L0mp','A'),L('L0mp','B'),L('L1','A'),L('L1','B')
check('8 L1: worlds A and B give different final weights from identical W0 margins',l1A[-1]['lora_hash']!=l1B[-1]['lora_hash'] and l1A[0]['margins']==l1B[0]['margins'])
check('8 L0mp: truncated A/B streams (384 tok, before any exposure) are identical, so are the final weights (determinism)',same and mpA[-1]['lora_hash']==mpB[-1]['lora_hash'])
# analysis with the exp07 code
sys.argv=[sys.argv[0]];import analyze_qwen
res=analyze_qwen.main(['--seed','101','--out',str(SM),'L1:main','L0mp:main','L0mw:main','L0mwfs:main','L0:main','L1:ctx','L0mp:rc'])
check('8 exp07 analysis (AL.stats via analyze_stage01.load/curve) runs on the exp09 logs',all(len(res[k])==3 for k in res) and len(res)==7,str({k:len(v) for k,v in res.items()}))
done()
