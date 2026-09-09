# RULER NIAH token labels (local)

`examples.parquet` and `spans.parquet` are **not** in git (~842MB). Rebuild from `data/ruler_niah/` JSONL:

```bash
python scripts/baseline_125m/sanity/test_ruler_niah_labels.py
python scripts/baseline_125m/experiments/prepare_ruler_niah_labels.py
```

`summary.json` is tracked (8000 OK / 0 fail, per-split counts). Indexing: `loss_t = raw_j - 1`.

STEP 7A `matches.parquet` / `exact.parquet` are also local (regenerable from the pilot JSON/NPZ).
