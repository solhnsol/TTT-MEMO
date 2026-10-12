"""Compose the official 3B 128K TTT-E2E Hydra config. No JAX, no weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINE_ROOT / "common"))

from baseline_config import CHECKPOINT_DIR, CHECKPOINT_GCS, CONFIG_DIR, EXPERIMENT, SANITY_RESULT_DIR, TOKENIZER_PATH, compose_3b_128k_config  # noqa: E402


def main() -> None:
    cfg = compose_3b_128k_config()
    from omegaconf import OmegaConf

    fields = {
        "experiment": EXPERIMENT,
        "checkpoint_gcs": CHECKPOINT_GCS,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "checkpoint_exists": CHECKPOINT_DIR.is_dir(),
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_exists": TOKENIZER_PATH.is_file(),
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
        "model.prefix_len_implied": int(cfg.model.num_hidden_layers) - int(cfg.model.suffix_len),
        "model.tie_word_embeddings": cfg.model.tie_word_embeddings,
        "model.param_dtype": cfg.model.param_dtype,
        "model.compute_dtype": cfg.model.compute_dtype,
        "model.state_dtype": cfg.model.state_dtype,
        "model.rope_theta": cfg.model.rope_theta,
        "model.seq_len_before_train_py_assign": cfg.model.seq_len,
        "model.force_flash": cfg.model.force_flash,
        "training.exp_name": cfg.training.exp_name,
        "training.train_mode": cfg.training.train_mode,
        "training.seq_length": cfg.training.seq_length,
        "training.load_part": cfg.training.load_part,
        "training.spec_inner": OmegaConf.to_container(cfg.training.spec_inner),
        "training.ilr_warmup_steps": cfg.training.ilr_warmup_steps,
        "training.ilr_init": cfg.training.ilr_init,
        "training.optimizer_inner.optimizer_type": str(cfg.training.optimizer_inner.optimizer_type),
        "training.optimizer_inner.lr": cfg.training.optimizer_inner.lr,
        "checkpoint.resume_checkpoint_dir": cfg.checkpoint.resume_checkpoint_dir,
    }

    print("Resolved 3B TTT-E2E 128K config")
    print(f"e2e config dir: {CONFIG_DIR}")
    print(f"checkpoint exists: {CHECKPOINT_DIR.is_dir()} ({CHECKPOINT_DIR})")
    print()
    for key, value in fields.items():
        print(f"{key}: {value}")

    SANITY_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = SANITY_RESULT_DIR / "inspect_config.json"
    out.write_text(json.dumps(fields, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
