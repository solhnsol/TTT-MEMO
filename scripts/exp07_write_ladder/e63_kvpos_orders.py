"""Recreate the sentence order (place first / name first) of the 16 read facts at the measured chunks of the U12 world (CPU; new file; thread-server 11/25; deterministic from the seed: the world build of e59_u12 + e59_leak_probe.answer_positions) and write kvpos/sent_order_s<seed>.json = {chunk: [order of fact 0..15]} with 0 = place first, 1 = name first. Usage: python e63_kvpos_orders.py --out D [--seeds 231,...,238] [--chunks 16,80]"""
import sys,json,pathlib,collections
a=sys.argv;out=pathlib.Path(a[a.index('--out')+1]);seeds=[int(x) for x in (a[a.index('--seeds')+1] if '--seeds' in a else '231,232,233,234,235,236,237,238').split(',')];chunks=[int(x) for x in (a[a.index('--chunks')+1] if '--chunks' in a else '16,80').split(',')]
import e59_u12 as U,e55_run as R55,e57_syn_pass as SP,e59_leak_probe as LP
U.install();tk=LP.tk_load();out.mkdir(parents=True,exist_ok=True)
for seed in seeds:
    f=out/f'sent_order_s{seed}.json'
    if f.exists():continue
    enc,w,tf=R55.build(SP.synth_args(seed,U.CELL),seed,read_apps={16,32,64,128},arms=['NAT']);ap,miss,tot=LP.answer_positions(w,tk,max(chunks)+1);assert miss==0
    res={str(k):[None]*16 for k in chunks}
    for k in chunks:
        for d in ap[k]:
            if d['fact']<16:res[str(k)][d['fact']]=0 if d['order']=='place_first' else 1
        assert all(v is not None for v in res[str(k)]),(seed,k)
    json.dump(res,open(f,'w'));print(seed,{k:collections.Counter(v) for k,v in res.items()},flush=True)
