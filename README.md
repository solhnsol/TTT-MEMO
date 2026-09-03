# TTT-MEMO

Research on a hybrid memory system that extends TTT-E2E: compressed fast-weight memory plus explicit high-fidelity memory for tokens predicted to have high future utility.

The hypothesis is tested, not assumed. Official TTT-E2E is the baseline. **Do not modify `e2e/`.**

A new agent continuing this work should start at [`scripts/baseline_125m/README.md`](scripts/baseline_125m/README.md). That file records what has already been verified, how to rerun it, and which helpers to reuse.

## Layout

| Path | Role |
|---|---|
| `e2e/` | Official [TTT-E2E](https://github.com/test-time-training/e2e) submodule. Pin: `upstream/ttt-e2e.toml`. |
| `scripts/baseline_125m/` | Sanity checks and ablations **outside** the submodule. |
| `docker/` | Image on top of `e2e/uv.lock` plus a short research extras list. |
| `checkpoints/` | Official weights (gitignored). |
| `data/` | Tokenizer, probe tokens, datasets (gitignored except README / `.gitkeep`). |
| `results/raw/` | Raw measurements (gitignored). Copy numbers you care about into notes. |
| `upstream/` | Pins for code and tokenizer files not authored here. |

```bash
git submodule update --init --recursive
docker compose build
docker compose run --rm ttt python scripts/baseline_125m/<script>.py
```

## Status (2026-09)

The 125M official checkpoint `125m_ttt_e2e_pretrain_dclm_8k_1x_cc` loads and runs on this machine.

- Restored NTP on a real DCLM val prefix is ~3.13 vs ~11.98 for random init (T=1024).
- T=2048 is two inner-loop chunks of 1024. Official `metrics["loss"]` is shape `[2]`.
- Holding SWA KV fixed, resetting vs keeping chunk-0 fast weights changes chunk-1 NTP by ~0.03 nat on **one** probe sequence. That is a functional signal, not a claim that TTT “works well.”

Host GPU is an RTX 3060 (sm_86). Prefix SWA hardcodes `implementation="cudnn"`; that kernel currently fails here. Script-local override rewrites only that keyword to `"xla"`. See the baseline README.

## What this repo is not yet

No TTT-MEMO router, no saliency labels, no gradient-norm logging, no out-of-window long-context eval. Sliding window is 8192, matching the pretrain length, so 8K-in-window scores must not be attributed to TTT memory alone.
