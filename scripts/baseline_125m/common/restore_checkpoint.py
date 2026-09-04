"""Stage 2: restore the official 125M weights into a freshly built MetaModel.

No forward pass here. The question this stage answers is narrow but load-bearing
for every later experiment:

    Does the model we build from the composed config have *exactly* the same
    parameter tree as the downloaded checkpoint, and do the restored arrays
    actually replace the random initialization?

It mirrors `e2e/ttt/train.py` lines ~120-207, dropping everything the params-only
restore path does not need (wandb, data iterator, outer optimizer state). The
upstream code is not modified.

Two failure modes this script is designed to expose, because `train.py` only
emits a `warnings.warn` for them and then discards the result:

1. A path exists in the checkpoint but not in our model tree.
   Detected inside `Checkpointer.load_checkpoint` -> `fetch_from_eqx_module`,
   which also asserts shape equality per leaf. Captured here via warnings.
2. A path exists in our model tree but not in the checkpoint.
   Returned as `not_found_paths` by `unify_dict_with_eqx_module`. Such a leaf
   silently keeps its random initialization, which would be a silent correctness
   bug in any downstream measurement.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_config import CHECKPOINT_DIR, add_e2e_to_path, compose_baseline_config  # noqa: E402

# Curated leaves to report statistics for. Substring match on the flattened
# keypath, chosen to cover embeddings, a norm, attention/TTT projections, the
# FFN, and the prime storage FFN (the TTT-specific part).
PROBE_PATTERNS = (
    "wte",
    "ln_f",
    "seq_modeling_block.wq",
    "seq_modeling_block.wo",
    "feed_forward.w1",
    "feed_forward_prime.w1",
    "lm_head",
)


def leaves_by_path(tree) -> dict[str, "object"]:
    """Flatten an Equinox module / pytree into {printable keypath: array}."""
    import jax

    return {jax.tree_util.keystr(path): value for path, value in jax.tree.flatten_with_path(tree)[0]}


def describe(array) -> str:
    import jax.numpy as jnp

    return (
        f"shape={tuple(array.shape)} dtype={array.dtype} "
        f"mean={float(jnp.mean(array.astype(jnp.float32))):+.5f} "
        f"std={float(jnp.std(array.astype(jnp.float32))):.5f} "
        f"absmax={float(jnp.max(jnp.abs(array.astype(jnp.float32)))):.5f}"
    )


def main() -> None:
    cfg = compose_baseline_config()
    add_e2e_to_path()

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import MetaModel
    from ttt.utils.jax_utils import eval_shape_and_sharding, initialize_distibuted, set_random_seed

    # Sets JAX_PLATFORM_NAME from cfg.backend.backend before any device is touched.
    # cfg.backend.distributed is false under `deploy=interactive`, so this does not
    # call jax.distributed.initialize.
    initialize_distibuted(cfg.backend)

    print(f"devices: {jax.devices()}  backend: {jax.default_backend()}")

    # train.py:88 -- the model init key. Fixed by cfg.training.model_seed, so the
    # random initialization we compare against is reproducible.
    key = set_random_seed(cfg.training.model_seed)

    # train.py:123 -- the model config carries no seq_len of its own; training does.
    # Skipping this leaves seq_len unset and changes the built parameter tree.
    cfg.model.seq_len = cfg.training.seq_length

    # train.py:120. With 1 device and n_state_parallel=1, ModelSharding derives
    # n_data_parallel=1 internally and builds a (1, 1) ("data", "state") mesh.
    # We do not call train.py's _prepare_data_parallelism because nothing here
    # reads cfg.training.n_data_parallel; ModelSharding runs the same assertion.
    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh
    print(f"mesh: {mesh}")

    @eqx.filter_jit
    def create_sharded_model_and_state():
        """train.py:127-132, verbatim.

        `make_with_state` splits MetaModel into (model, state): `state` holds the
        `nn.StateIndex` values, i.e. the TTT step counter, not parameters. Only
        `model` is checkpointed. Returns arrays already committed to `mesh`.
        """
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = jax.device_put(state, jax.NamedSharding(mesh, P()))
        model = model_sharding.shard_params(model)
        return model, state

    # train.py:161. Compiles the initializer to learn output shapes *and*
    # shardings without allocating any parameter. Orbax needs the sharding to
    # place each restored array directly onto the right device.
    abstract_model_weights = eval_shape_and_sharding(lambda: create_sharded_model_and_state()[0].weights())
    abstract_leaves = leaves_by_path(abstract_model_weights)
    print(f"abstract model_weights leaves: {len(abstract_leaves)}")

    loader = Checkpointer(config=cfg, for_saving=False)
    print(f"checkpoint dir: {loader.manager.directory}")
    print(f"available steps: {loader.manager.all_steps()}  latest: {loader.manager.latest_step()}")
    if not loader.checkpoint_exists():
        raise FileNotFoundError(f"No Orbax checkpoint under {CHECKPOINT_DIR}")

    print("\n--- restoring model_weights ---")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_state = loader.load_checkpoint(
            step=cfg.training.resume_step,  # None -> latest_step()
            targets={"model_weights": abstract_model_weights},
            restore=cfg.training.load_part,  # "params"
        )
    ckpt_only = [str(w.message) for w in caught]
    print(f"paths in checkpoint but missing from our model: {len(ckpt_only)}")
    for message in ckpt_only:
        print(f"  WARN {message}")

    restored_leaves = leaves_by_path(out_state["model_weights"])
    print(f"restored leaves: {len(restored_leaves)}")

    # train.py:175-176. Build the randomly initialized model, then overwrite the
    # leaves present in the checkpoint. We keep the fresh model around so we can
    # prove the overwrite actually happened.
    model_fresh, _state = create_sharded_model_and_state()
    model_restored, model_only = unify_dict_with_eqx_module(out_state["model_weights"], model_fresh)

    print(f"\npaths in our model but missing from checkpoint: {len(model_only)}")
    for path in model_only:
        print(f"  WARN {path}")

    fresh = leaves_by_path(model_fresh.weights())
    final = leaves_by_path(model_restored.weights())
    assert fresh.keys() == final.keys(), "restore changed the model pytree structure"

    unchanged = [k for k in fresh if jnp.array_equal(fresh[k], final[k])]
    print(f"\nleaves overwritten by checkpoint: {len(fresh) - len(unchanged)} / {len(fresh)}")
    if unchanged:
        print("leaves identical to random init (suspicious unless intentionally frozen):")
        for path in unchanged:
            print(f"  {path}")

    n_all = sum(x.size for x in jax.tree_util.tree_leaves(model_restored.weights()))
    n_trainable = sum(x.size for x in jax.tree_util.tree_leaves(model_restored.trainable_parameters()))
    n_non_embedding = n_trainable - model_restored.language_model.model.wte.weight.size
    if model_restored.language_model.lm_head is not None:
        n_non_embedding -= model_restored.language_model.lm_head.weight.size
    print(f"\n#all params:        {n_all:,}")
    print(f"#trainable params:  {n_trainable:,}")
    print(f"#non-embed params:  {n_non_embedding:,}")

    print("\n--- per-leaf statistics (random init vs restored) ---")
    for path in fresh:
        if not any(pattern in path for pattern in PROBE_PATTERNS):
            continue
        print(path)
        print(f"  init:     {describe(fresh[path])}")
        print(f"  restored: {describe(final[path])}")


if __name__ == "__main__":
    main()
