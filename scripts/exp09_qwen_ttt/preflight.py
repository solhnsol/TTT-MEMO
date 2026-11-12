"""Exp09 preflight (run once on the GPU pod after the model is available; no weights are loaded, CPU is enough):
prints the model config values the design depends on and asserts them, and checks that the model's tokenizer
tokenizes the Exp07 texts exactly like the Qwen3 tokenizer the CPU tests used (Instruct-2507, cached here).
Usage: CUDA_VISIBLE_DEVICES= python preflight.py [--model-id base|instruct|<hub id>] [--model DIR] [--no-download]
Expected values for Qwen/Qwen3-4B-Base are ASSUMED from the public model card (not verified on this machine; the
weights/config are not cached here): see EXPECT. Exit code 1 on any mismatch."""
import sys,argparse,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import qwen_runtime as QR,world_text as WT

EXPECT={
 'Qwen/Qwen3-4B-Base':{'model_type':'qwen3','num_hidden_layers':36,'hidden_size':2560,'intermediate_size':9728,'num_attention_heads':32,
     'num_key_value_heads':8,'head_dim':128,'max_position_embeddings':32768,'rope_theta':1000000,'rope_scaling':None,
     'tie_word_embeddings':True,'vocab_size':151936,'use_sliding_window':False},
 'Qwen/Qwen3-4B-Instruct-2507':{'model_type':'qwen3','num_hidden_layers':36,'hidden_size':2560,'intermediate_size':9728,'num_attention_heads':32,
     'num_key_value_heads':8,'head_dim':128,'max_position_embeddings':262144,'rope_theta':5000000,'rope_scaling':None,
     'tie_word_embeddings':True,'vocab_size':151936,'use_sliding_window':False},
}
WINDOW=8*1024   # largest position used anywhere: native window (7 cached chunks + chunk), readout (7168 tail + probe), canary
def get(cfg,k):
    if k=='rope_theta':   # transformers 5 keeps it in rope_parameters
        rp=getattr(cfg,'rope_parameters',None) or {}
        return rp.get('rope_theta',getattr(cfg,'rope_theta',None))
    if k=='rope_scaling':
        rp=getattr(cfg,'rope_parameters',None) or {}
        t=rp.get('rope_type','default') if rp else None
        sc=getattr(cfg,'rope_scaling',None)
        return None if (t in (None,'default') and not (sc and sc.get('rope_type','default')!='default')) else (sc or rp)
    return getattr(cfg,k,None)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-id',default='base');ap.add_argument('--model',default='');ap.add_argument('--no-download',action='store_true')
    a=ap.parse_args();mid=QR.MODEL_IDS.get(a.model_id,a.model_id)
    src=Path(a.model) if a.model else (QR.cached_snapshot(mid) if a.no_download else QR.resolve_model(mid))
    if src is None:print(f'{mid}: not in the HF cache and --no-download given; nothing to check');return 2
    from transformers import AutoConfig
    cfg=AutoConfig.from_pretrained(str(src));exp=EXPECT.get(mid,EXPECT['Qwen/Qwen3-4B-Base']);bad=[]
    print(f'model {mid} from {src}')
    for k,v in exp.items():
        got=get(cfg,k);ok=(got==v) or (isinstance(v,(int,float)) and got is not None and float(got)==float(v))
        print(f'  {k:26s} {got!s:>12}  expected {v!s:>12}  {"ok" if ok else "MISMATCH"}')
        if not ok:bad.append(k)
    if (get(cfg,'max_position_embeddings') or 0)<WINDOW:bad.append('max_position_embeddings < 8192')
    # tokenizer: identical token ids on the experiment texts as the tokenizer the tests used
    tok=WT.load_tokenizer(QR.tokenizer_file(src));bos=tok.token_to_id('<|endoftext|>')
    print(f'  <|endoftext|> id {bos} (expected 151643)');bad+=[] if bos==151643 else ['endoftext id']
    ref_dir=QR.cached_snapshot(QR.INSTRUCT_ID)
    if ref_dir is not None:
        ref=WT.load_tokenizer(ref_dir/'tokenizer.json');diff=[]
        for seed in (101,102,103):
            Wm=WT.World(seed,tok,bos);Wr=WT.World(seed,ref,ref.token_to_id('<|endoftext|>'))
            for nm,f in (('stream A',lambda W:W.stream_ids(False)),('stream B',lambda W:W.stream_ids(True)),('probe rows',lambda W:W.probe_rows()),
                         ('tail',lambda W:W.tail_ids),('alt',lambda W:W.alt_ids),('gen',lambda W:W.gen_ids),('L1 rows',lambda W:W.l1_rows(False,'plain'))):
                if WT.h(f(Wm))!=WT.h(f(Wr)):diff.append(f'{seed} {nm}')
        print('  tokenization of all experiment texts vs the tested Qwen3 tokenizer:','identical' if not diff else f'DIFFERS: {diff}')
        bad+=diff
    else:print('  (Instruct tokenizer not cached: tokenizer identity not checked)')
    print('PREFLIGHT', 'OK' if not bad else f'FAILED: {bad}');return 1 if bad else 0
if __name__=='__main__':sys.exit(main())
