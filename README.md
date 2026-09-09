# TTT-MEMO

Research on a hybrid memory system that extends TTT-E2E: compressed fast-weight memory plus explicit high-fidelity memory for tokens predicted to have high future utility.

The hypothesis is tested, not assumed. Official TTT-E2E is the baseline. **Do not modify `e2e/`.**

A new agent continuing this work should start at [`scripts/baseline_125m/README.md`](scripts/baseline_125m/README.md). That file records what has already been verified, how to rerun it, and which helpers to reuse.

## Layout

| Path | Role |
|---|---|
| `e2e/` | Official [TTT-E2E](https://github.com/test-time-training/e2e) submodule. Pin: `upstream/ttt-e2e.toml`. |
| `scripts/baseline_125m/` | Baseline research scripts **outside** the submodule: `common/`, `sanity/`, `experiments/`. |
| `docker/` | Image on top of `e2e/uv.lock` plus a short research extras list. |
| `checkpoints/` | Official weights (gitignored). |
| `data/` | Tokenizer, probe tokens, datasets (gitignored except README / `.gitkeep` / the 32 KB DCLM probe). |
| `results/raw/baseline_125m/` | Completed baseline measurements (JSON/NPZ). Other raw dumps stay gitignored. |
| `upstream/` | Pins for code and tokenizer files not authored here. |

```bash
git submodule update --init --recursive
docker compose build
docker compose run --rm ttt python scripts/baseline_125m/sanity/inspect_config.py
```

## Status (2026-09)

The 125M official checkpoint `125m_ttt_e2e_pretrain_dclm_8k_1x_cc` loads and runs on A100 cuDNN. Token-wise `G_t` is measured with exact reverse-mode and a Rademacher JVP estimator (STEP 2/3). Alignment metrics `Ahat`/`Ihat` were checked on DCLM chunk 0 (STEP 6C.1). RULER NIAH labels exist for 8000 rows (parquet local). An 8K **in-SWA** mechanism pilot (STEP 7A, 20+20) is recorded; it is not a long-range memory result. Details: [`scripts/baseline_125m/README.md`](scripts/baseline_125m/README.md).

**Main experiment backend on A100 is cuDNN** (`--no-attention-cudnn-override`). Runtime extra, not in `e2e/uv.lock`: `nvidia-cuda-nvrtc-cu12==12.8.93` with `libnvrtc.so.12` on `LD_LIBRARY_PATH`. Details: [`scripts/baseline_125m/README.md`](scripts/baseline_125m/README.md). The XLA `AttentionOverride` remains a fallback (RTX 3060, or A100 without NVRTC).

## What this repo is not yet

No TTT-MEMO router, no prospective student, no out-of-window long-context eval. Sliding window is 8192, matching the pretrain length, so 8K-in-window scores must not be attributed to TTT memory alone. `G_t` is a gradient-signal magnitude, not importance. RULER needle spans are token labels, not teacher saliency.
