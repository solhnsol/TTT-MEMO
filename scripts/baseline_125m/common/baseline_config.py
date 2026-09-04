"""Shared entry point for reconstructing the official 125M TTT-E2E baseline config.

Both the stage-1 config inspection and the stage-2 checkpoint restore need the
exact same config tree, so it lives here instead of being duplicated. Nothing in
`e2e/` is modified: we only add it to `sys.path` and call its own
`register_configs()` / Hydra config directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file lives at scripts/baseline_125m/common/baseline_config.py
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
E2E_ROOT = REPO_ROOT / "e2e"
CONFIG_DIR = E2E_ROOT / "configs"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "125m_ttt_e2e_pretrain_dclm_8k_1x_cc"

EXPERIMENT = "125m/pretrain/pretrain-125m-e2e"

TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer" / "llama3" / "tokenizer.json"
PROBE_PATH = REPO_ROOT / "data" / "_probe_val_head.bin"

RAW_RESULT_DIR = REPO_ROOT / "results" / "raw" / "baseline_125m"
SANITY_RESULT_DIR = RAW_RESULT_DIR / "sanity"
EXPERIMENT_RESULT_DIR = RAW_RESULT_DIR / "experiments"


def add_e2e_to_path() -> None:
    """Make the official `ttt` package importable without installing it."""
    path = str(E2E_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def compose_baseline_config():
    """Build the config tree that `train +experiment=125m/pretrain/pretrain-125m-e2e` would see.

    Hydra compose needs two things first:
    1. `register_configs()` -- registers the dataclass defaults such as `base_model`.
    2. `initialize_config_dir` -- points at `e2e/configs`, not this repo's root.

    The overrides below are the only deviation from a real training launch, and
    each one is deliberate:
      - `checkpoint.resume_checkpoint_dir`: the downloaded checkpoint, absolute,
        because `interactive.yaml` leaves `deploy_paths.checkpoint` unset for us.
      - `training.load_part=params`: restore weights only, no optimizer/loop state.
      - `training.dummy_dataset=true`: we have no DCLM shards locally, and none of
        the stages up to the forward pass read real tokens.
      - `deploy_paths.data.*`: `interactive.yaml` leaves these as OmegaConf MISSING
        (`???`), which would raise on resolve. They are never dereferenced under
        `dummy_dataset=true`.
    """
    if not CONFIG_DIR.is_dir():
        raise FileNotFoundError(f"Official config directory not found: {CONFIG_DIR}")

    add_e2e_to_path()

    from hydra import compose, initialize_config_dir

    from ttt.config import register_configs

    register_configs()

    overrides = [
        "+deploy=interactive",
        f"+experiment={EXPERIMENT}",
        f"checkpoint.resume_checkpoint_dir={CHECKPOINT_DIR.resolve().as_posix()}",
        "training.load_part=params",
        "training.dummy_dataset=true",
        "deploy_paths.data.dclm_filter_8k=unused",
        "deploy_paths.data.books3=unused",
    ]

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(config_name="config", overrides=overrides)
