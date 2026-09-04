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

The 125M official checkpoint `125m_ttt_e2e_pretrain_dclm_8k_1x_cc` loads and runs. Token-wise `G_t = ||∇_{W_fast} l_t^TTT||_2` is measured with exact reverse-mode and a Rademacher JVP estimator (STEP 2/3). STEP 4 (8-chunk trajectory × 4 projection seeds) is implemented but not yet a completed result.

Host GPU during development was an RTX 3060. Prefix SWA hardcodes `implementation="cudnn"`; scripts may rewrite that keyword to `"xla"` locally. On a machine where the official cuDNN path works, run with `--no-attention-cudnn-override`.

## What this repo is not yet

No TTT-MEMO router, no saliency labels, no future-utility experiment, no out-of-window long-context eval. Sliding window is 8192, matching the pretrain length, so 8K-in-window scores must not be attributed to TTT memory alone. `G_t` is a gradient-signal magnitude, not importance.
