"""Stage 1: compose the official 125M TTT-E2E Hydra config, without touching weights.

This only checks that we can reconstruct the baseline configuration from outside
`e2e/`. It does not create a model, initialize JAX devices, or read checkpoint
tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINE_ROOT / "common"))
sys.path.insert(0, str(_BASELINE_ROOT / "sanity"))

from baseline_config import CHECKPOINT_DIR, CONFIG_DIR, compose_baseline_config  # noqa: E402


def main() -> None:
    cfg = compose_baseline_config()

    from omegaconf import OmegaConf

    fields = {
        "model.name": cfg.model.name,
        "model.vocab_size": cfg.model.vocab_size,
        "model.bos_token_id": cfg.model.bos_token_id,
        "model.eos_token_id": cfg.model.eos_token_id,
        "model.num_hidden_layers": cfg.model.num_hidden_layers,
        "model.hidden_size": cfg.model.hidden_size,
        "model.num_attention_heads": cfg.model.num_attention_heads,
        "model.intermediate_size": cfg.model.intermediate_size,
        "model.seq_modeling_block": cfg.model.seq_modeling_block,
        "model.sliding_window_size": cfg.model.sliding_window_size,
        "model.mini_batch_size": cfg.model.mini_batch_size,
        "model.prime": cfg.model.prime,
        "model.suffix_len": cfg.model.suffix_len,
        "model.tie_word_embeddings": cfg.model.tie_word_embeddings,
        "model.param_dtype": cfg.model.param_dtype,
        "model.compute_dtype": cfg.model.compute_dtype,
        "model.state_dtype": cfg.model.state_dtype,
        "training.exp_name": cfg.training.exp_name,
        "training.train_mode": cfg.training.train_mode,
        "training.seq_length": cfg.training.seq_length,
        "training.load_part": cfg.training.load_part,
        "training.dummy_dataset": cfg.training.dummy_dataset,
        "training.model_seed": cfg.training.model_seed,
        "training.n_state_parallel": cfg.training.n_state_parallel,
        "training.n_data_parallel": cfg.training.n_data_parallel,
        "training.spec_inner": OmegaConf.to_container(cfg.training.spec_inner),
        "checkpoint.resume_checkpoint_dir": cfg.checkpoint.resume_checkpoint_dir,
    }

    print("Resolved 125M TTT-E2E baseline config")
    print(f"e2e config dir: {CONFIG_DIR}")
    print(f"checkpoint exists: {CHECKPOINT_DIR.is_dir()} ({CHECKPOINT_DIR})")
    print()
    for key, value in fields.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
