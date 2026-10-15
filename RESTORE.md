# Restoring TTT-MEMO on a fresh machine

Use this when the A100 (`/home/solhnsol/workspace/TTT-MEMO`) is gone, wiped, or unavailable.

## Where everything lives

| What | Primary copy | Backup | How it gets there |
|---|---|---|---|
| Code, protocols, `env/` freezes | GitHub `solhnsol/TTT-MEMO` | branch `autosnap/a100` (hourly, includes uncommitted edits) | cron on A100 → `scripts/backup/code_autosnap.sh` |
| `experiments/`, `results/`, `data/`, `checkpoints/`, `.git` | A100 | lab PC `D:\Project\KFAI\2026\TTT_ACL\TTT-MEMO\` (WSL: `/mnt/d/Project/KFAI/2026/TTT_ACL/TTT-MEMO/`) | Windows Task Scheduler (WSL rsync) pulls every 15 min → `scripts/backup/pull_from_a100.sh`, set up by `scripts/backup/windows/setup_pc.ps1` |
| Files overwritten/deleted on the A100 | — | lab PC `/mnt/d/Project/KFAI/2026/TTT_ACL/_previous/<date>/` (kept 14 days) | same pull |
| Claude project memory | A100 `~/.claude/projects/...` | lab PC `/mnt/d/Project/KFAI/2026/TTT_ACL/_claude_memory/` | same pull |

Not backed up (rebuild instead): `.venv`, `.venv-exp02a` (≈12 GB, see step 3), HF model cache (≈38 GB, step 5), JAX compile cache.

Check the backup is fresh: `cat /mnt/d/Project/KFAI/2026/TTT_ACL/LAST_OK` on the PC; logs in `/mnt/d/Project/KFAI/2026/TTT_ACL/_logs/`.

## 1. Code

```bash
git clone https://github.com/solhnsol/TTT-MEMO.git && cd TTT-MEMO
git submodule update --init --recursive        # e2e pinned in upstream/ttt-e2e.toml; never modify e2e/
git checkout autosnap/a100                     # newest snapshot, incl. uncommitted A100 work
# (or `main` / a feature branch if you only want committed history)
```

The PC backup also holds the full repo with `.git`, so `/mnt/d/Project/KFAI/2026/TTT_ACL/TTT-MEMO` is itself a working checkout if GitHub is unreachable.

## 2. Data, results, checkpoints

From the lab PC to the new machine (`NEW` = `user@host`, run in WSL on the PC):

```bash
rsync -a --info=progress2 /mnt/d/Project/KFAI/2026/TTT_ACL/TTT-MEMO/{data,checkpoints,experiments,results} NEW:~/workspace/TTT-MEMO/
```

Sizes at setup (2026-10-15): data 9.9 GB, checkpoints 11 GB, experiments 21 GB (exp07 13 GB, exp04 4.1 GB, exp08 3.6 GB), results 0.9 GB.

Checkpoints are also downloadable, but the bucket is **Requester Pays** (needs a GCP billing project), so prefer the PC copy:

- 125M: `checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc` (step 4799)
- 3B: `checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc` (step 1949) — `scripts/baseline_3b/download_checkpoint.py --billing-project <proj>`

Data rebuild paths (if needed): tokenizer via `scripts/baseline_125m/sanity/verify_tokenizer.py`; RULER per `data/README.md`.

## 3. Python environments

Exact package lists from the A100 are in `env/venv.freeze.txt`, `env/venv-exp02a.freeze.txt`, `env/system.txt` (Python 3.12.3, A100 driver 580, CUDA 12 wheels).

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r env/venv.freeze.txt          # JAX 0.5.3 + cuda12 plugin, nvrtc 12.8.93

uv venv .venv-exp02a --python 3.12                                        # torch stack for exp02a* / HF models
uv pip install --python .venv-exp02a/bin/python -r env/venv-exp02a.freeze.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

cuDNN attention needs NVRTC visible by bare name (see `scripts/baseline_125m/README.md`):

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

On a non-A100 GPU (e.g. RTX 3060) cuDNN prefix SDPA may fail with `No valid engine configs`; use the XLA fallback (`--attention-cudnn-override`, see baseline README). Docker alternative: `docker compose build`.

## 4. Hard-coded paths

23 run scripts (mostly `scripts/exp07_write_ladder/run_*.sh`, `scripts/exp04_what_ttt_retains/*.sh`) hard-code `/home/solhnsol/workspace/TTT-MEMO`. Easiest: put the repo at that same path, or symlink it:

```bash
sudo mkdir -p /home/solhnsol/workspace && sudo ln -s "$PWD" /home/solhnsol/workspace/TTT-MEMO
```

List them with `grep -rl /home/solhnsol scripts`.

## 5. Hugging Face models (exp02a*)

Re-download into the HF cache: `Qwen/Qwen3-4B-Instruct-2507`, `meta-llama/Llama-3.1-8B-Instruct` (gated: `huggingface-cli login` with an approved token).

## 6. Sanity check before resuming

```bash
.venv/bin/python scripts/baseline_125m/sanity/forward_sanity.py      # 125M checkpoint loads and runs
```

Then resume from the latest state in `experiments/<exp>/` (each exp's `scripts/<exp>/PROTOCOL*.md` and `experiments/research_log/`). The project research log lives in the project files (`paper/research-log.md`).

## Backup maintenance

- A100 cron (`crontab -l`): `7 * * * *` autosnap; log `~/.local/state/ttt-backup/autosnap.log`.
- A100 has no system rsync; a user-space one is at `~/.local/bin/rsync` (micromamba env `~/.local/share/rsync-env`). The PC script calls it by absolute path.
- Round-trip test: on the PC `pull_from_a100.sh --restore-test`, then on the A100 `scripts/backup/verify_restore_test.sh`.
- The PC script reaches the A100 through an ssh alias `a100` in the PC's `~/.ssh/config` (host, user, key live there, not in this public repo).
- The repo is **public**: autosnap only picks up untracked files under `scripts/ env/ docker/ upstream/` and top-level `*.md`, < 5 MB, and respects `.gitignore`. Keep secrets and raw data out of those paths.
