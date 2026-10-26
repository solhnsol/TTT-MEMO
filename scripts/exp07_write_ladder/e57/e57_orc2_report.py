"""E57 ORC v2 mask draft report (CPU only): masked token ratios, over-masking, leak recount per variant, private per-question examples. See e57_orc2.py for the variants.
  python3 e57_orc2_report.py --bundle B --w0mask E57_w0mask.json --out PREFIX    (writes PREFIX.json, PREFIX_examples_private.json; leak recounts PREFIX_leak_<variant>.json via e57_delorc_leak)"""
import sys,json,argparse,pathlib,subprocess,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
for p in (HERE,HERE.parent,ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
VARIANTS=('orc','v1','v2','v2g','v2p')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--w0mask');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_bank as B,e57_orc2 as O2
    from tokenizers import Tokenizer
    tf=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);tk=Tokenizer.from_file(tf)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//1024
    out={};bds={}
    for v in VARIANTS:
        bds[v]=O2.build(v,ids,tk,tf,items,iv,UVm,B);out[v]=O2.summarize_variant(bds[v],n);s=out[v];print(f"{v:4s} hidden tokens {s['hidden_tokens']} ({s['hidden_token_share_of_stream']:.3f} of stream), window share mean {s['window_hidden_share']['mean']:.3f} (min {s['window_hidden_share']['min']:.3f} max {s['window_hidden_share']['max']:.3f}), chunk share mean {s['chunk_hidden_share']['mean']:.3f}; partial-only share {s['share_partial_only']:.3f}, cross-event-only share {s['share_strict_cross_event_only']:.3f}",flush=True)
    out['reference_261_ORC_ORCA']={'ORC_window_hidden_share_mean':0.264,'ORCA_window_hidden_share_mean':0.265,'ORCA_chunk_hidden_share_mean':0.264,'note':'job 261 ORC hides the window only (about 2160 of 8192 tokens, 26%); the pod ORCA window + chunk masks (2145 window / 270 chunk tokens per chunk); the orc row of this report recomputes the same rule'}
    # per-question examples (private): fragments masked by v2 / v2p but not by the strict rule (new masks), up to 3 per question, from the question's own event
    ex={};b2=bds['v2'];b1=bds['v1'];bp=bds['v2p']
    for qi,it in enumerate(items):
        q=it['q'];prx=O2.partial_rxs(q);new=[j for j in range(len(b2['fr'])) if b2['partial_only'][j] and b2['event'][j]==q['event'] and any(r.search(b2['txt'][j]) for r in prx)]
        ex[it['id']]=dict(question=it['id'],event=it['event'],type=it['type'],gold=q['canonical'],n_new_fragments_v2=len(new),n_new_fragments_v2p=int(sum(1 for j in new if bp['hidden'][j])),examples_v2_only_new=[b2['txt'][j][:260] for j in new[:3]],examples_removed_by_v2p=[b2['txt'][j][:260] for j in new if not bp['hidden'][j]][:3])
    pathlib.Path(str(a.out)+'.json').write_text(json.dumps(out,indent=1,default=float));pathlib.Path(str(a.out)+'_examples_private.json').write_text(json.dumps(dict(note='private, internal research only; stream sentences',examples=ex),indent=1,ensure_ascii=False))
    for v in ('orc','v1','v2','v2p'):
        subprocess.run([sys.executable,str(HERE/'e57_delorc_leak.py'),'--bundle',a.bundle,'--w0mask',a.w0mask,'--out',f'{a.out}_leak_{v}','--variant',v],check=True,stdout=subprocess.DEVNULL)
    print('wrote',a.out)
if __name__=='__main__':main()
