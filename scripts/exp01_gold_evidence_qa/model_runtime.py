"""Load the official 125M TTT-E2E checkpoint for Exp 01 generation.

Reuses `scripts/baseline_125m/common/` restore + Hydra compose. Does not
modify `e2e/`. After `BlockCollectionSplit`, prefix/suffix calls are the same
primitives `loss_for_sequence` uses; we keep the logits that the official
path discards after `lm_loss`.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_COMMON = _REPO_ROOT / "scripts" / "baseline_125m" / "common"
sys.path.insert(0, str(_BASELINE_COMMON))

from baseline_config import CHECKPOINT_DIR, TOKENIZER_PATH, add_e2e_to_path, compose_baseline_config  # noqa: E402
from runtime_helpers import AttentionOverride, describe_gpu  # noqa: E402


@dataclass
class LoadedModel:
    cfg: Any
    model: Any
    state_prefix: Any
    state_suffix: Any
    mesh: Any
    gpu: Any
    attention_cfg: dict
    checkpoint_dir: Path
    tokenizer_path: Path
    swa_window: int
    mini_batch_size: int
    bos_id: int
    eos_id: int
    vocab_size: int
    fast_weight_snapshot: dict
    fast_weight_hash: str
    spec_inner: list[str]
    suffix_len: int
    prefix_len: int
    step_index_value: int
    backend_note: str


def _hash_pytree(tree) -> tuple[str, dict]:
    import equinox as eqx
    import jax
    import numpy as np

    digest = hashlib.sha256()
    leaf_info = []
    for path, leaf in jax.tree.flatten_with_path(tree)[0]:
        if not eqx.is_inexact_array(leaf):
            continue
        arr = np.asarray(leaf)
        digest.update(str(path).encode())
        digest.update(str(arr.shape).encode())
        digest.update(str(arr.dtype).encode())
        digest.update(arr.tobytes())
        leaf_info.append(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": int(arr.nbytes),
            }
        )
    return digest.hexdigest(), {"n_leaves": len(leaf_info), "leaves": leaf_info}


def snapshot_fast_weights(model) -> tuple[Any, str, dict]:
    """CPU copy + sha256 of inner-loop parameters (`feed_forward_prime`)."""
    import jax

    inner = model.inner_parameters()
    cpu = jax.device_get(inner)
    digest, meta = _hash_pytree(cpu)
    return cpu, digest, meta


def fast_weight_max_abs_diff(current_model, snapshot) -> float:
    from measurement_utils import pytree_max_abs_diff

    return pytree_max_abs_diff(current_model.inner_parameters(), snapshot)


def load_tokenizer():
    from tokenizers import Tokenizer

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"tokenizer missing: {TOKENIZER_PATH}")
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def load_official_from_cfg(
    cfg,
    checkpoint_dir: Path,
    *,
    attention_cudnn_override: bool = False,
) -> LoadedModel:
    """Restore official TTT-E2E weights and split prefix/suffix for generation.

    `cfg.model.seq_len` is set to `training.seq_length` (RoPE table), never to
    the padded generation length. Generation does not call `inner_loop_step`.
    """
    add_e2e_to_path()

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from jax.sharding import PartitionSpec as P

    from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import BlockCollectionSplit, MetaModel
    from ttt.utils.jax_utils import eval_shape_and_sharding, initialize_distibuted, set_random_seed

    initialize_distibuted(cfg.backend)
    gpu = describe_gpu()
    override = AttentionOverride(enabled=attention_cudnn_override)
    override.install()

    # train.py:123 — RoPE table is 2 * seq_len. Keep the checkpoint's training length.
    cfg.model.seq_len = cfg.training.seq_length

    key = set_random_seed(cfg.training.model_seed)
    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh

    @eqx.filter_jit
    def create_sharded_model_and_state():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = jax.device_put(state, jax.NamedSharding(mesh, P()))
        model = model_sharding.shard_params(model)
        return model, state

    abstract_model_weights = eval_shape_and_sharding(lambda: create_sharded_model_and_state()[0].weights())
    loader = Checkpointer(config=cfg, for_saving=False)
    if not loader.checkpoint_exists():
        raise FileNotFoundError(f"No Orbax checkpoint under {checkpoint_dir}")
    out_state = loader.load_checkpoint(
        step=cfg.training.resume_step,
        targets={"model_weights": abstract_model_weights},
        restore=cfg.training.load_part,
    )
    model_fresh, state = create_sharded_model_and_state()
    model, missing = unify_dict_with_eqx_module(out_state["model_weights"], model_fresh)
    if missing:
        raise RuntimeError(f"checkpoint did not cover {len(missing)} model paths: {missing[:8]}")
    del out_state, model_fresh

    # Official eval sets a fully warmed-up inner LR. Generation does not call
    # inner_loop_step, so this does not change logits; it documents the eval point.
    step_index_value = int(jnp.iinfo(jnp.int32).max - 100)
    state = state.set(model.step_index, jnp.array(step_index_value, dtype=jnp.int32))

    block_collection = model.language_model.model.h.blocks
    prime_storage = model.language_model.model.h.prime_storage if cfg.model.prime else None
    if prime_storage is None:
        raise RuntimeError("checkpoint has prime=False; Exp 01 needs feed_forward_prime")

    state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
    state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)

    new_collection = BlockCollectionSplit(
        cfg.model,
        block_collection=block_collection,
        prime_storage=prime_storage,
        key=jrandom.PRNGKey(0),
    )
    model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)
    # Official meta branch casts the module to state_dtype (fp32) before TTT.
    model = jax.tree.map(lambda p: p.astype(model.state_dtype), model)

    suffix_len = int(cfg.model.suffix_len)
    prefix_len = int(cfg.model.num_hidden_layers) - suffix_len

    snapshot, digest, _snap_meta = snapshot_fast_weights(model)
    attention_cfg = override.report_for_config(
        force_flash=bool(cfg.model.force_flash),
        seq_modeling_block=str(cfg.model.seq_modeling_block),
    )
    backend_note = (
        "cuDNN prefix SWA (official path)"
        if not attention_cudnn_override
        else "XLA AttentionOverride fallback (implementation=cudnn rewritten to xla)"
    )
    return LoadedModel(
        cfg=cfg,
        model=model,
        state_prefix=state_prefix,
        state_suffix=state_suffix,
        mesh=mesh,
        gpu=gpu,
        attention_cfg=attention_cfg,
        checkpoint_dir=Path(checkpoint_dir),
        tokenizer_path=TOKENIZER_PATH,
        swa_window=int(cfg.model.sliding_window_size),
        mini_batch_size=int(cfg.model.mini_batch_size),
        bos_id=int(cfg.model.bos_token_id),
        eos_id=int(cfg.model.eos_token_id),
        vocab_size=int(cfg.model.vocab_size),
        fast_weight_snapshot=snapshot,
        fast_weight_hash=digest,
        spec_inner=list(cfg.training.spec_inner),
        suffix_len=suffix_len,
        prefix_len=prefix_len,
        step_index_value=step_index_value,
        backend_note=backend_note,
    )


def load_official_125m(*, attention_cudnn_override: bool = False) -> LoadedModel:
    """Restore the same 125M checkpoint used by previous TTT-MEMO experiments."""
    cfg = compose_baseline_config()
    return load_official_from_cfg(
        cfg,
        CHECKPOINT_DIR,
        attention_cudnn_override=attention_cudnn_override,
    )


def load_official_3b(*, attention_cudnn_override: bool = False) -> LoadedModel:
    """Restore the official 3B 128K Books checkpoint via the 3B compose helper.

    Loads `scripts/baseline_3b/common/baseline_config.py` by file path so this
    module can keep using the 125M `baseline_config` name on sys.path.
    """
    import importlib.util

    path = _REPO_ROOT / "scripts" / "baseline_3b" / "common" / "baseline_config.py"
    spec = importlib.util.spec_from_file_location("ttt_memo_baseline_3b_config", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"3B baseline_config missing: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.compose_3b_128k_config()
    return load_official_from_cfg(
        cfg,
        Path(mod.CHECKPOINT_DIR),
        attention_cudnn_override=attention_cudnn_override,
    )


def loaded_config_record(loaded: LoadedModel) -> dict:
    cfg = loaded.cfg
    gpu = loaded.gpu
    return {
        "checkpoint_path": str(loaded.checkpoint_dir),
        "checkpoint_name": loaded.checkpoint_dir.name,
        "tokenizer": str(loaded.tokenizer_path),
        "tokenizer_kind": "llama3 tokenizer.json (official TTT-E2E data tokenizer; config.tokenizer_name is unused)",
        "model_name": str(cfg.model.name),
        "model_seq_len_rope_table": int(cfg.model.seq_len),
        "training_seq_length": int(cfg.training.seq_length),
        "sliding_window_size": loaded.swa_window,
        "mini_batch_size": loaded.mini_batch_size,
        "suffix_len": loaded.suffix_len,
        "prefix_len": loaded.prefix_len,
        "prime": bool(cfg.model.prime),
        "spec_inner": loaded.spec_inner,
        "train_mode": str(cfg.training.train_mode),
        "seq_modeling_block": str(cfg.model.seq_modeling_block),
        "force_flash": bool(cfg.model.force_flash),
        "param_dtype": str(cfg.model.param_dtype),
        "compute_dtype": str(cfg.model.compute_dtype),
        "state_dtype": str(cfg.model.state_dtype),
        "vocab_size": loaded.vocab_size,
        "bos_token_id": loaded.bos_id,
        "eos_token_id": loaded.eos_id,
        "ilr_warmup_steps": int(cfg.training.ilr_warmup_steps),
        "inner_optimizer": {
            "optimizer_type": str(cfg.training.optimizer_inner.optimizer_type),
            "lr": float(cfg.training.optimizer_inner.lr),
            "clip_gradient": float(cfg.training.optimizer_inner.clip_gradient),
        },
        "eval_step_index": loaded.step_index_value,
        "fast_weight_hash_at_load": loaded.fast_weight_hash,
        "backend": {
            "note": loaded.backend_note,
            "gpu_name": gpu.name,
            "compute_capability": gpu.compute_capability,
            "jax_device": gpu.jax_device,
            "jax_backend": gpu.jax_backend,
            "attention": loaded.attention_cfg,
        },
    }
