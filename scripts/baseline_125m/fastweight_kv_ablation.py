"""Chunk-1 ablation: same KV state, fast weights updated vs reset.

Does not edit e2e/. Orchestrates the official meta-mode primitives that
`loss_for_sequence` already calls (BlockCollectionSplit, prefix_call,
inner_loop_step). The algorithm is not reimplemented.

After chunk 0's official inner_loop_step, chunk 1 is run twice:

* updated — carry fast weights from chunk 0 (same as the official scan)
* reset   — meta-learned inner params, but the cloned suffix/KV state from chunk 0

Reported NTP is the inner-loop value computed *before* that chunk's update.
Inner SGD has no momentum, so optimizer state does not enter this number.

Harness check: `updated` chunk-1 NTP should match
`results/raw/baseline_125m/forward_sanity_t2048.json` restored chunk 1 within
bf16 noise. If it does not, the ablation is not the official path.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_config import CHECKPOINT_DIR, REPO_ROOT, add_e2e_to_path, compose_baseline_config  # noqa: E402
from forward_sanity import (  # noqa: E402
    RESULT_DIR,
    AttentionOverride,
    describe_gpu,
    tokens_to_batch,
)

PROBE_PATH = REPO_ROOT / "data" / "_probe_val_head.bin"
SEQ_LEN = 2048
REF_JSON = RESULT_DIR / "forward_sanity_t2048.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same RTX 3060 compatibility flag as forward_sanity.py.",
    )
    return parser.parse_args()


def ntp_stats(loss, nll) -> dict:
    import jax.numpy as jnp

    nll_f = nll.astype(jnp.float32)
    return {
        "mean_ntp": float(loss),
        "token_nll_mean": float(jnp.mean(nll_f)),
        "any_nan": bool(jnp.any(jnp.isnan(loss)) or jnp.any(jnp.isnan(nll_f))),
        "any_inf": bool(jnp.any(jnp.isinf(loss)) or jnp.any(jnp.isinf(nll_f))),
    }


def main(attention_cudnn_override: bool = True) -> None:
    cfg = compose_baseline_config()
    add_e2e_to_path()

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
    from ttt.model.data import tree_slice
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import BlockCollectionSplit, MetaModel
    from ttt.utils.filter_utils import get_filter_spec
    from ttt.utils.jax_utils import clone_pytree, eval_shape_and_sharding, initialize_distibuted, set_random_seed, tree_rearrange

    initialize_distibuted(cfg.backend)
    gpu = describe_gpu()
    override = AttentionOverride(enabled=attention_cudnn_override)
    override.install()

    cfg.model.seq_len = cfg.training.seq_length
    mini_batch = int(cfg.model.mini_batch_size)
    if SEQ_LEN % mini_batch != 0:
        raise SystemExit(f"SEQ_LEN={SEQ_LEN} is not a multiple of mini_batch_size={mini_batch}")
    n_chunks = SEQ_LEN // mini_batch
    if n_chunks != 2:
        raise SystemExit("this ablation is defined for exactly two chunks")

    probe = np.fromfile(PROBE_PATH, dtype="<u4")
    if probe.size < SEQ_LEN + 1:
        raise SystemExit(f"{PROBE_PATH} has {probe.size} tokens, need {SEQ_LEN + 1}")
    seq = tokens_to_batch(jnp.asarray(probe[: SEQ_LEN + 1], dtype=jnp.int32), bos_token_id=int(cfg.model.bos_token_id))

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
        raise FileNotFoundError(f"No Orbax checkpoint under {CHECKPOINT_DIR}")
    out_state = loader.load_checkpoint(
        step=cfg.training.resume_step,
        targets={"model_weights": abstract_model_weights},
        restore=cfg.training.load_part,
    )
    model_fresh, state = create_sharded_model_and_state()
    model, missing = unify_dict_with_eqx_module(out_state["model_weights"], model_fresh)
    if missing:
        raise SystemExit(f"checkpoint did not cover {len(missing)} model paths: {missing[:8]}")
    del out_state, model_fresh

    state = state.set(model.step_index, jnp.array(jnp.iinfo(jnp.int32).max - 100, dtype=jnp.int32))
    M = MetaModel.MetricType

    def recombine_inner(model_inner, model_outer):
        """Official process_suffix_chunk: keep inner from carry, outer from the pre-scan model."""
        spec_inner = get_filter_spec(model_inner, model_outer.config.training.spec_inner, "inner parameters")
        inner_params, _ = eqx.partition(model_inner, spec_inner)
        _, outer_params = eqx.partition(model_outer, spec_inner)
        return eqx.combine(inner_params, outer_params)

    @eqx.filter_jit
    def official_chunk_step(model_inner, opt_state, state_tuple, suffix_chunk, prefix_chunk, model_outer):
        model_inner = recombine_inner(model_inner, model_outer)
        return MetaModel.inner_loop_step(model_inner, opt_state, state_tuple, suffix_chunk, prefix_chunk)

    attention_cfg = override.report_for_config(
        force_flash=bool(cfg.model.force_flash),
        seq_modeling_block=str(cfg.model.seq_modeling_block),
    )
    print("--- setup ---")
    print(f"GPU name:            {gpu.name}")
    print(f"compute capability:  {gpu.compute_capability}")
    print(f"jax device:          {gpu.jax_device}")
    print(f"override enabled:    {attention_cfg['override_enabled']}")
    print(f"prefix requested:    {attention_cfg['prefix_requested_backend']}")
    print(f"prefix effective:    {attention_cfg['prefix_effective_backend']}")
    print(f"suffix requested:    {attention_cfg['suffix_requested_backend']}")
    print(f"suffix effective:    {attention_cfg['suffix_effective_backend']}")
    print(f"sequence length T:   {SEQ_LEN}")
    print(f"chunk size:          {mini_batch}")
    print(f"n chunks:            {n_chunks}")
    print(f"checkpoint:          {CHECKPOINT_DIR.name}")
    print(f"param_dtype:         {cfg.model.param_dtype}")
    print(f"compute_dtype:       {cfg.model.compute_dtype}")

    with mesh:
        # Official loss_for_sequence: split State off `h.blocks` *before* replacing
        # `h` with BlockCollectionSplit (that module has prefix/suffix, not `.blocks`).
        block_collection = model.language_model.model.h.blocks
        prime_storage = model.language_model.model.h.prime_storage if cfg.model.prime else None
        state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
        state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
        state_all = clone_pytree(state)

        new_collection = BlockCollectionSplit(
            cfg.model,
            block_collection=block_collection,
            prime_storage=prime_storage,
            key=jrandom.PRNGKey(0),
        )
        model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)

        # Official meta branch casts the module to state_dtype (fp32) before the inner scan.
        model_outer = jax.tree.map(lambda p: p.astype(model.state_dtype), model)
        inner_opt_state = model_outer.inner_optimizer(state_all).init(model_outer.inner_parameters())

        xt_embed = model_outer.language_model.wte_call(seq.input_ids)
        prefix_output = eqx.filter_checkpoint(model_outer.language_model.prefix_call)(
            model_outer.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq
        ).last_hidden_state

        seq_chunks = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=mini_batch)
        prefix_chunks = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=mini_batch)
        seq0, seq1 = tree_slice(seq_chunks, 0), tree_slice(seq_chunks, 1)
        pref0, pref1 = prefix_chunks[0], prefix_chunks[1]

        print("\n--- chunk 0 (official inner_loop_step) ---")
        out0 = official_chunk_step(
            model_outer, inner_opt_state, (state_all, state_suffix), seq0, pref0, model_outer
        )
        kv_after_0 = clone_pytree(out0.new_state)

        print("--- chunk 1 updated (fast weights from chunk 0, same KV) ---")
        out_updated = official_chunk_step(
            out0.new_model, out0.new_optimizer_state, kv_after_0, seq1, pref1, model_outer
        )

        print("--- chunk 1 reset (original fast weights, same KV) ---")
        opt_reset = model_outer.inner_optimizer(state_all).init(model_outer.inner_parameters())
        out_reset = official_chunk_step(
            model_outer, opt_reset, clone_pytree(kv_after_0), seq1, pref1, model_outer
        )

    chunk0 = ntp_stats(out0.metrics[M.loss], out0.metrics[M.token_nll_loss])
    updated = ntp_stats(out_updated.metrics[M.loss], out_updated.metrics[M.token_nll_loss])
    reset = ntp_stats(out_reset.metrics[M.loss], out_reset.metrics[M.token_nll_loss])
    delta = reset["mean_ntp"] - updated["mean_ntp"]

    print("\n--- NTP ---")
    for label, row in ("chunk 0", chunk0), ("chunk 1 updated", updated), ("chunk 1 reset", reset):
        print(
            f"{label}: mean NTP={row['mean_ntp']:.6f}  "
            f"NaN={row['any_nan']}  Inf={row['any_inf']}"
        )
    print(f"chunk 1 reset - updated: {delta:.6f}")
    print("positive => updated fast weights lower chunk-1 NTP than reset, holding KV fixed.")
    print("not a pass/fail threshold; KV-matched first look at whether TTT changes prediction.")

    harness = None
    if REF_JSON.exists():
        ref = json.loads(REF_JSON.read_text())
        ref0 = ref["restored_chunks"][0]["mean_ntp"]
        ref1 = ref["restored_chunks"][1]["mean_ntp"]
        harness = {
            "ref_json": str(REF_JSON),
            "chunk0_minus_ref": chunk0["mean_ntp"] - ref0,
            "updated_chunk1_minus_ref": updated["mean_ntp"] - ref1,
            "ref_chunk0": ref0,
            "ref_chunk1": ref1,
        }
        print("\n--- harness vs official loss_for_sequence scan (forward_sanity --seq-len 2048) ---")
        print(f"chunk 0 minus ref:           {harness['chunk0_minus_ref']:+.6f}")
        print(f"chunk 1 updated minus ref:   {harness['updated_chunk1_minus_ref']:+.6f}")
        print("updated should track the scan's chunk 1; large disagreement means the harness drifted.")
    else:
        print(f"\nno {REF_JSON.name}; skip scan harness comparison.")

    attention_cfg = override.report_for_config(
        force_flash=bool(cfg.model.force_flash),
        seq_modeling_block=str(cfg.model.seq_modeling_block),
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULT_DIR / "fastweight_kv_ablation_t2048"
    np.savez(
        stem.with_suffix(".npz"),
        chunk0_token_nll=np.asarray(out0.metrics[M.token_nll_loss]),
        updated_token_nll=np.asarray(out_updated.metrics[M.token_nll_loss]),
        reset_token_nll=np.asarray(out_reset.metrics[M.token_nll_loss]),
    )
    meta = {
        "seq_len": SEQ_LEN,
        "mini_batch_size": mini_batch,
        "n_chunks": n_chunks,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "probe_path": str(PROBE_PATH),
        "param_dtype": str(cfg.model.param_dtype),
        "compute_dtype": str(cfg.model.compute_dtype),
        "gpu": asdict(gpu),
        "attention": attention_cfg,
        "chunk0": chunk0,
        "chunk1_updated": updated,
        "chunk1_reset": reset,
        "reset_minus_updated": delta,
        "harness_vs_scan": harness,
    }
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nraw: {stem.with_suffix('.npz')}")
    print(f"meta: {stem.with_suffix('.json')}")


if __name__ == "__main__":
    args = parse_args()
    main(attention_cudnn_override=args.attention_cudnn_override)
