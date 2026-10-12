"""Shared Hydra compose for the official 3B TTT-E2E 128K Books checkpoint.

Does not modify `e2e/` or the 125M baseline scripts. The experiment id matches
the released long-context checkpoint in e2e/README.md:

    gs://ttt-e2e-checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc
    +experiment=3b/extension/ext-3b-e2e-128K
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
E2E_ROOT = REPO_ROOT / "e2e"
CONFIG_DIR = E2E_ROOT / "configs"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "3b_ttt_e2e_finetune_books_128k_3x_cc"
CHECKPOINT_GCS = "gs://ttt-e2e-checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc"

EXPERIMENT = "3b/extension/ext-3b-e2e-128K"

TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer" / "llama3" / "tokenizer.json"
PROBE_PATH = REPO_ROOT / "data" / "_probe_val_head.bin"

RAW_RESULT_DIR = REPO_ROOT / "results" / "raw" / "baseline_3b"
SANITY_RESULT_DIR = RAW_RESULT_DIR / "sanity"


def add_e2e_to_path() -> None:
    path = str(E2E_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def compose_3b_128k_config():
    """Config tree for `train +experiment=3b/extension/ext-3b-e2e-128K`.

    Same override style as the 125M compose: params-only restore, dummy data
    paths so interactive.yaml MISSING fields resolve.
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
