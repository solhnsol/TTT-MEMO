# 125M TTT-E2E baseline — context for the next agent

This directory is the **verified official baseline**, not TTT-MEMO yet. Scripts live outside `e2e/` and import `ttt` via `PYTHONPATH=/workspace/e2e`. They do not patch the submodule.

Read this before adding gradient logging, a router, or a new eval. If a change would alter `e2e/`, architecture, `mini_batch_size`, or RoPE `seq_len`, stop and discuss it.

## What the official model actually does

Checkpoint: `checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc` (Orbax step 4799, params only).

Hydra experiment: `+experiment=125m/pretrain/pretrain-125m-e2e` (composed in `common/baseline_config.py`).

| Setting | Value | Why it matters |
|---|---|---|
| `train_mode` | `meta` | `loss_for_sequence` **always** runs the inner loop, including eval. There is no separate inference mode. |
| `mini_batch_size` | 1024 | Inner-loop chunk size. `T` must be a multiple of 1024. |
| `sliding_window_size` | 8192 | Equals pretrain `seq_length`. Local attention can cover a full 8K context by itself. |
| `cfg.model.seq_len` | **keep 8192** | Sizes the RoPE table (`2 * seq_len`). Suffix indexes up to `window + mini_batch - 1`. Setting this to `T` produces NaN RoPE. |
| `cfg.training.seq_length` | **keep 8192** | Used only in `seqlen % mini_batch == 0`. Actual length is the Batch. |
| `spec_inner` | `language_model.**.suffix_blocks.feed_forward_prime.**` | Fast weights. Checkpoint stores them as `h.prime_storage.feed_forward_prime`, shape `(suffix_len=3, ...)`. `suffix_blocks` is assembled at forward time. |
| `suffix_len` | 3 | Last 3 layers get the prime FFN; first 9 are prefix (no inner update). |
| `force_flash` | False | Suffix attention already uses `implementation=None` (XLA). Prefix SWA **hardcodes** `implementation="cudnn"`. |
| Inner SGD | `lr=1`, clip 1.0, **no momentum** | Chunk NTP is computed **before** that chunk's update. Optimizer state does not affect the reported number. |
| Eval `step_index` | `int32.max - 100` | Matches `train.py` eval. Without this, inner LR is still in warmup (`ilr_warmup_steps=480`). |

Forward (meta): embed → prefix 9 layers on the **full** sequence → split into chunks of 1024 → `inner_loop_step` per chunk (suffix 3 layers + CE + SGD on prime FFN). Prefix activations are **not** TTT-updated. Across chunks, information moves through **fast weights** and **SWA KV cache**. Do not attribute a chunk-1 vs chunk-0 loss drop to TTT alone.

`loss_for_sequence` returns:

- scalar `loss` = mean of per-chunk masked CE
- `metrics["loss"]` shape `[n_chunks]` (1-D arrays are not flattened)
- `metrics["token_nll_loss"]` shape `[T]` (flattened)

It never returns logits. Per-chunk logits inside `suffix_call` are `[1024, 128256]`.

Official code has **no tokenizer**. `config.tokenizer_name` is dead and wrongly set to Llama-2. Data is Llama-3 token IDs (`vocab 128256`, BOS `128000`, EOS `128001`).

## Files here

| Path | Role |
|---|---|
| `common/baseline_config.py` | Shared Hydra compose. Import this; do not duplicate overrides. |
| `common/restore_checkpoint.py` | Orbax restore; both-way pytree match; init vs restored leaf stats. |
| `common/measurement_utils.py` | Shared `l_t^TTT` / pytree / ranking helpers. No eval-point setup. |
| `common/runtime_helpers.py` | GPU info, `AttentionOverride`, `tokens_to_batch`. |
| `sanity/inspect_config.py` | Config only. No JAX. |
| `sanity/verify_tokenizer.py` | Download `tokenizer.json`, sha256, DCLM probe round-trip. |
| `sanity/forward_sanity.py` | Official `loss_for_sequence`. `--seq-len` (default 1024). Restored vs random. |
| `sanity/fastweight_kv_ablation.py` | Orchestrates official `inner_loop_step`. Chunk 1: same KV, fast weights updated vs reset. |
| `sanity/jvp_vjp_numerical_diagnostic.py` | JVP vs VJP directional-identity diagnostic (not an estimator study). |
| `experiments/token_grad_norm_jvp_validation.py` | STEP 2: 32-token exact vs JVP on chunk 0. |
| `experiments/token_grad_norm_full_validation.py` | STEP 3: full 1024-token exact vs JVP on chunk 0. |
| `experiments/token_grad_align_validation.py` | STEP 6C.1: DCLM T=1024 chunk-0 `Ahat`/`Ihat` vs 32-token exact (cuDNN). |
| `experiments/ruler_niah_measurement_pilot.py` | STEP 6D: one RULER 8K S-NIAH sample × official TTT trajectory (pipeline sanity). |
| `experiments/ruler_niah_mechanism_pilot_8k.py` | STEP 7A: 20+20 RULER 8K single_2 / MK1 mechanism pilot (in-SWA). |
| `experiments/ruler_predictive_ranking_8k.py` | STEP 8A: within-sample L=32 ranking, M0/M1/M2 linear scorers (in-SWA). |
| `common/ranking_windows.py` | STEP 8A window placement + pairwise logistic helpers. CPU. |
| `sanity/test_ranking_windows.py` | Unit tests for L=32 placement, splits, shuffle. |
| `experiments/token_grad_norm_trajectory_validation.py` | STEP 4: 8 sequential TTT chunks × 4 JVP seeds (A100 run exists; JSON has a flagged comparison — inspect before treating as clean). |
| `common/ruler_niah_labels.py` | STEP 6B: RULER NIAH needle span / loss-index helpers. CPU, no TTT. |
| `sanity/test_ruler_niah_labels.py` | Unit tests for `t=j-1`, padding mask, overlap, one real S1 row. |
| `experiments/prepare_ruler_niah_labels.py` | Label all 8000 JSONL rows → `results/raw/baseline_125m/ruler_niah_labels/`. |
| `README.md` | This file. |

Reuse from `common/runtime_helpers.py`: `AttentionOverride`, `describe_gpu`, `tokens_to_batch`. Dependency direction: `common` → nothing; `sanity` → `common`; `experiments` → `common`.

`sanity/fastweight_kv_ablation.py` is the pattern for “run official inner steps without editing `e2e/`”: `BlockCollectionSplit` **after** `state.substate(h.blocks)`, fp32 cast, `prefix_call` once, `tree_rearrange`, then `inner_loop_step`. Outer params are recombined from the pre-scan model every chunk, as in `process_suffix_chunk`.

## How to run

From repo root:

```bash
git submodule update --init --recursive
docker compose build
docker compose run --rm ttt python scripts/baseline_125m/sanity/inspect_config.py
docker compose run --rm ttt python scripts/baseline_125m/common/restore_checkpoint.py
docker compose run --rm ttt python scripts/baseline_125m/sanity/verify_tokenizer.py
docker compose run --rm ttt python scripts/baseline_125m/sanity/forward_sanity.py
docker compose run --rm ttt python scripts/baseline_125m/sanity/forward_sanity.py --seq-len 2048
docker compose run --rm ttt python scripts/baseline_125m/sanity/fastweight_kv_ablation.py
docker compose run --rm -e PYTHONUNBUFFERED=1 ttt python scripts/baseline_125m/experiments/token_grad_norm_jvp_validation.py
docker compose run --rm -e PYTHONUNBUFFERED=1 ttt python scripts/baseline_125m/experiments/token_grad_norm_full_validation.py
```

RULER NIAH labels (CPU, no TTT):

```bash
python scripts/baseline_125m/sanity/test_ruler_niah_labels.py
python scripts/baseline_125m/experiments/prepare_ruler_niah_labels.py
```

STEP 6C.1 / 6D / 7A (A100 cuDNN; NVRTC `LD_LIBRARY_PATH` as above; 6C.1 does not overwrite STEP 3):

```bash
export LD_LIBRARY_PATH="/home/solhnsol/workspace/TTT-MEMO/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHONUNBUFFERED=1 python scripts/baseline_125m/experiments/token_grad_align_validation.py
PYTHONUNBUFFERED=1 python scripts/baseline_125m/experiments/ruler_niah_measurement_pilot.py
PYTHONUNBUFFERED=1 python scripts/baseline_125m/experiments/ruler_niah_mechanism_pilot_8k.py
```

STEP 4 (8K trajectory, expensive; run on a GPU that can hold official 8K prefix SWA):

```bash
docker compose run --rm -e PYTHONUNBUFFERED=1 ttt python scripts/baseline_125m/experiments/token_grad_norm_trajectory_validation.py
```

Disable the attention workaround only if cuDNN prefix SDPA works on that GPU:

```bash
docker compose run --rm ttt python scripts/baseline_125m/sanity/forward_sanity.py --no-attention-cudnn-override
```

### Local files (gitignored)

| Path | How to get it |
|---|---|
| `checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc/` | Official GCS, Requester Pays. |
| `data/tokenizer/llama3/tokenizer.json` | `verify_tokenizer.py`; pin `upstream/llama3-tokenizer.toml` (`sha256 e134af98…`). |
| `data/_probe_val_head.bin` | First 8192 little-endian uint32 tokens of `gs://llama3-dclm-filter-8k/data.zarr/val/c/0` (uncompressed). Need `T+1` tokens. |
| `results/raw/baseline_125m/ruler_niah_labels/{examples,spans}.parquet` | ~842MB; rebuild with `prepare_ruler_niah_labels.py`. Not in git. |
| `data/ruler_niah/` | Official RULER NIAH JSONL (gitignored). |

## What we already know (one DCLM val prefix, RTX 3060)

Not paper numbers. One probe, XLA prefix attention.

**Restore.** 20 parameter leaves, all overwritten. 184,363,776 params. No missing paths.

**Tokenizer.** Vocab/BOS/EOS match config. 512-token encode(decode(ids)) round-trip on the probe (BOS stripped) was exact.

**T=1024** (`forward_sanity.py`): restored masked mean NTP **3.1315** vs random **11.9817**. Random ≈ `ln(128256)`. No NaN/Inf. At one chunk the reported loss does **not** include that chunk's inner update.

**T=2048** (`--seq-len 2048`): `metrics["loss"]` is `(2,)`. Restored chunk 0 **3.1301**, chunk 1 **3.1155**, overall **3.1228**. Random overall **11.810**. Chunk 0 vs T=1024 overall: CE Δ ≈ **0.0014**. Token-NLL is not bitwise identical (prefix SDPA compiled at different `T`); mean abs diff ≈ 0.01, max 0.125. Inputs for the first 1024 tokens are identical. Treat as consistent, not bit-exact.

**KV-matched fast-weight ablation** (`fastweight_kv_ablation.py`): harness vs that scan, chunk 0 Δ **+0.00025**, updated chunk 1 Δ **−0.00069**. Then same KV:

| Chunk 1 condition | NTP |
|---|---|
| updated (after chunk 0 inner step) | 3.1148 |
| reset (initial `feed_forward_prime`) | 3.1459 |
| reset − updated | **+0.0312** |

On this sequence, fast weights change chunk-1 NTP with KV held fixed. The effect is two orders of magnitude smaller than restored vs random. Do **not** generalize; do **not** treat chunk1 < chunk0 as a TTT win (text difficulty + KV both move).

## Main experiment backend (A100)

**MAIN EXPERIMENT BACKEND = cuDNN** (`--no-attention-cudnn-override`).

On A100, exact `G_t` ranking vs the XLA override is effectively identical (top 1/5/10% exact sets identical). Same-backend JVP-vs-exact quality is unchanged. cuDNN matches the official TTT-E2E prefix SWA path (`implementation="cudnn"`).

cuDNN prefix SDPA requires NVRTC at runtime. This is **not** in `e2e/uv.lock` and must not be added there.

- Package (host venv only): `nvidia-cuda-nvrtc-cu12==12.8.93`
- Library: `.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12`
- Loader: `libnvrtc.so.12` must be visible by **bare name**. The cuDNN plugin does not RPATH this directory, so set

```bash
export LD_LIBRARY_PATH="/home/solhnsol/workspace/TTT-MEMO/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Without this, prefix SWA raises `No valid engine configs`.

## RTX 3060 attention override

Prefix SWA: `e2e/ttt/model/attention.py` `SWA.full_sw_attention`, `implementation="cudnn"`. On an RTX 3060 (and on A100 if NVRTC is not loader-visible) that call raises `No valid engine configs`. Same shapes work with `"xla"`.

Implemented only in `AttentionOverride` (`common/runtime_helpers.py`).

- Not an `e2e/` patch, not a change to mask/window/loss.
- Rewrites **only** `implementation="cudnn"` → `"xla"`.
- Suffix is already `None` when `force_flash=False`. JIT trace: 1 redirect (prefix scan body), 1 passthrough (suffix).
- If someone sets `force_flash=True`, suffix would be redirected too.
- This override is a **fallback**, not the main A100 experiment path.

## Environment

Image `ttt-memo:py312-jax0.5.3-cu128`: `python:3.12-slim-bookworm` + `e2e/uv.lock` (`jax==0.5.3`) + `docker/requirements-research.txt`.

On the lab A100 the working environment is the host venv at `.venv` (same lockfile). Do not modify `e2e/uv.lock`. NVRTC is an extra runtime library, documented above.

Research extras must stay additive. Do not let `huggingface-hub` 1.x upgrade `click` (locked at 8.1.8). Tokenizer stack is `tokenizers==0.23.1` and `huggingface-hub==0.36.2`.

`XLA_PYTHON_CLIENT_PREALLOCATE=false`. JAX cache: `/home/solhnsol/.cache/jax` on the A100 host (docker volume `jax-cache` → `/tmp/jax_cache` in compose).

## Rules of interpretation (carry these forward)

1. Do not edit `e2e/` unless the user explicitly accepts an upstream patch.
2. Do not set `cfg.model.seq_len` to the batch length `T`.
3. Do not change `mini_batch_size` just to run T=512; that changes the TTT chunk definition.
4. In-window (≤8192) quality is not evidence of long-range TTT memory.
5. Inner-loop gradients ≠ outer-loop gradients. Say which loss, which params, when in the update, and the grain (token / chunk / layer).
6. Fast weights in the checkpoint are **meta-learned initializers** of the prime FFN, not a stored KV.
7. Prefer official functions (`loss_for_sequence`, `inner_loop_step`) over re-deriving CE. If you must orchestrate, compare against `forward_sanity --seq-len 2048` as a harness.
8. Log raw values; do not overwrite them with normalized copies. Small analyzable outputs under `results/raw/baseline_125m/` may be committed; huge regenerated labels (`ruler_niah_labels/*.parquet`) stay local.

## Current status (after STEP 8A)

Stopped after the 8K **in-SWA** within-sample ranking test. Not a long-range memory result. Do not start a router, window-length sweep, or 16K run from this.

**Done (A100 cuDNN, 125M 8K checkpoint):** official restore/forward; STEP 2/3 `G_t` estimator; STEP 6B labels; STEP 6C.1; STEP 6D; STEP 7A; STEP 8A `niah_single_2` + `niah_multikey_2` (200+200, L=32, M0/M1/M2).

**8A takeaway (not H1, not a router):** On single_2, mean NLL already ranks the needle window well (test MRR 0.19 vs chance ~0.025). Adding log(Ghat) does **not** stably improve ΔMRR (95% CI includes 0). M1 < M0. Shuffle control ≈ M0. On MK2, all scorers are near chance (relevant vs format-matched distractors; query still in the future). Artifacts: `results/raw/baseline_125m/experiments/ruler_predictive_ranking_8k/`.

**7A descriptive takeaway (not H1/H2):** Spearman(Ghat, NLL) ≈ 0.80 at token grain. Do not use `Ihat` as exact `I`.

**Local-only:** `data/ruler_niah/`, `checkpoints/`, `ruler_niah_labels/{examples,spans}.parquet`, STEP 8A `window_*.parquet` / `token_features_*.npz`.

**Parked (discuss before running):** 16K seq_len/RoPE-table decision. Out-of-SWA memory. Teacher saliency only with a prospective student.
