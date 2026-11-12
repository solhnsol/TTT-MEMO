"""Test 7: world texts, exposure sentences, probe items and readout/canary texts are IDENTICAL to the JAX pipeline's,
seeds 101-103. Reference: ref_dump_jax.py run in the JAX env (.venv, CPU, JAX_PLATFORMS=cpu) calls the JAX pipeline's
own functions; here world_text.py is run with the same llama3 tokenizer + BOS and every string / token list must match
(token identity under the same tokenizer proves the underlying strings are identical). Items are also compared with
the meta_L1_main_s{seed}.json files written by the actual JAX runs. Finally the Qwen tokenization is summarised."""
from _testutil import *
import json,subprocess,os
import world_text as WT

REF=OUT/'jax_ref.json';SEEDS=[101,102,103];NCH,CH=WT.JAX_NCH,WT.JAX_CH
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
r=subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'ref_dump_jax.py'),str(REF)]+[str(s) for s in SEEDS],env=env,capture_output=True,text=True)
check('7 JAX reference dump ran (.venv, CPU)',r.returncode==0,r.stderr[-300:] if r.returncode else 'ok')
ref=json.load(open(REF));tok=WT.llama_tokenizer();bos=ref['bos'];h=WT.h
for seed in SEEDS:
    W=WT.World(seed,tok,bos);d=ref[str(seed)];bad=[]
    meta=json.load(open(ROOT/f'experiments/exp07_write_ladder/meta_L1_main_s{seed}.json'))
    if W.items!=meta['items']:bad.append('items vs JAX meta')
    if {k:len(v) for k,v in W.fam.items()}!=meta['fam_sizes'] or list(W.fam)!=list(meta['fam_sizes']):bad.append('fam order/sizes')
    if W.pseudo!=d['pseudo']:bad.append('pseudo names')
    import gen_world as gw
    _,name,_=gw.realize(W.w,False)
    if gw.probes(W.w,name)!=d['probes_gw']:bad.append('gw.probes')
    for s,tag in ((False,'A'),(True,'B')):
        D=d[tag];text,_,spans=gw.realize(W.w,s)
        if text!=D['text'] or W.text(s)!=D['text']:bad.append(f'{tag} text')
        if [x['text'] for x in spans]!=D['spans']:bad.append(f'{tag} exposure sentences')
        if h(W.stream_ids(s)[:NCH*CH])!=D['stream_hash']:bad.append(f'{tag} stream ids')
        if h(W.sentences(s))!=D['sentences_hash']:bad.append(f'{tag} sentences')
        for v in ('plain','ctx','irr','fill'):
            if h([[list(a),list(b)] for a,b in W.l1_rows(s,v)])!=D['rows'][v]:bad.append(f'{tag} l1_rows {v}')
        if [h(list(x)) for x in W.fs_streams(s,3,NCH*CH)]!=D['fs3']:bad.append(f'{tag} fs_streams')
    if h(W.alt_ids)!=d['alt']['hash']:bad.append('squad tail')
    if h(W.tail_ids)!=ref['tail']['hash'] or h(W.tail_text)!=ref['tail']['text_hash'] or W.tail_sha!=ref['tail']['sha']:bad.append('neutral tail')
    if h(W.gen_ids)!=ref['gen']['hash']:bad.append('pg11 canary')
    ns=len(d['A']['spans'])
    check(f'7 seed {seed}: texts A/B, {ns} exposure sentences, {len(W.items)} items, L1 rows (4 variants), fs streams, tails, canary == JAX',not bad,', '.join(bad) or 'identical')
# Qwen tokenization summary (informational)
mdir=QR.qwen_snapshot_dir();qt=WT.load_tokenizer(mdir/'tokenizer.json');qb=qt.token_to_id('<|endoftext|>')
for seed in SEEDS:
    W=WT.World(seed,qt,qb);nA,nB=len(W.stream_ids(False)),len(W.stream_ids(True));L=WT.World(seed,tok,bos)
    lA=len(L.stream_ids(False))
    tl=lambda s:len(WT.enc(qt,' '+s).ids)
    nm=sorted({tl(t['X']) for t in W.w['triples']});pl=sorted({tl(t['P']) for t in W.w['triples']})
    rows=W.probe_rows();sent=W.sentences(False)
    print(f'  qwen seed {seed}: stream A {nA} tok ({-(-nA//1024)} chunks of 1024), B {nB}; llama full text {lA} tok (JAX used the first {NCH*CH}); '
          f'exposure sentences {len(sent)} ({sum(map(len,sent))} tok); probe rows {len(rows)} (max {max(len(x) for x,_ in rows)} tok); '
          f'name tok-lengths {nm}, place {pl}; tail {len(W.tail_ids)}, alt {len(W.alt_ids)}, gen {len(W.gen_ids)}',flush=True)
done()
