# Local data (not in git)

| Path | Purpose |
|---|---|
| `_probe_val_head.bin` | First 8192 uint32 LE tokens of official DCLM val (`gs://llama3-dclm-filter-8k/data.zarr/val/c/0`). Uncompressed zarr; a byte range is enough. Scripts need `T+1` tokens. |
| `tokenizer/llama3/tokenizer.json` | Llama-3 tokenizer. Download + sha256 via `scripts/baseline_125m/verify_tokenizer.py`. Pin: `upstream/llama3-tokenizer.toml`. |
