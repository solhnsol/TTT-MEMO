# Local data (not in git)

| Path | Purpose |
|---|---|
| `_probe_val_head.bin` | First 8192 uint32 LE tokens of official DCLM val (`gs://llama3-dclm-filter-8k/data.zarr/val/c/0`). Uncompressed zarr; a byte range is enough. Official 8K NTP samples are `seq_len+1=8193` tokens; this probe is one short, so STEP 4 masks the last target. |
| `tokenizer/llama3/tokenizer.json` | Llama-3 tokenizer. Download + sha256 via `scripts/baseline_125m/sanity/verify_tokenizer.py`. Pin: `upstream/llama3-tokenizer.toml`. |
| `ruler_niah/` | Downloaded RULER NIAH validation splits (gitignored). `_raw/{8192,16384,32768,65536,131072}/{task}/validation.jsonl` plus a copy organized as `{S,MK,MQ,MV}-NIAH/{task}/{8k,16k,32k,64k,128k}/*.jsonl`. 8 tasks × 5 lengths × 200 samples. Manifest: `ruler_niah/manifest.json`. |
