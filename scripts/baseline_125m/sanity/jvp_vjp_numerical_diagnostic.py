"""JVP vs VJP numerical-gap diagnostic at the frozen T=1024 chunk-0 eval point.

Not an estimator study and not a saliency experiment. Does not edit e2e/.
Does not change the official baseline compute dtype used by other scripts.

STEP 2 found, for one (token, direction):

    <g_t, v>  !=  (Jv)[t]
    standalone f(W0)  !=  JVP-returned primal

with relative directional gap ~1.4% and token-CE max abs ~0.068.
This script asks where that discrepancy appears:

1. JIT / transform compilation boundary (standalone f vs jvp primal, jit vs not)
2. Several tokens x directions (not only t=15, v=0)
3. Official-like bf16 compute vs a script-level fp32 compute_dtype control

The fp32 condition is a diagnostic override of `cfg.model.compute_dtype` before
MetaModel construction. Linear/attention `promote_dtype` reads that field.
It is not an upstream patch and is not written back into the estimator experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINE_ROOT / "common"))
sys.path.insert(0, str(_BASELINE_ROOT / "sanity"))

from baseline_config import CHECKPOINT_DIR, REPO_ROOT, SANITY_RESULT_DIR, add_e2e_to_path, compose_baseline_config  # noqa: E402
from forward_sanity import AttentionOverride, describe_gpu, tokens_to_batch  # noqa: E402
from measurement_utils import (  # noqa: E402
    pytree_max_abs_diff,
    pytree_vdot,
    rademacher_like,
    select_even_valid_indices,
    token_ttt_ce_from_logits,
)

PROBE_PATH = REPO_ROOT / "data" / "_probe_val_head.bin"
SEQ_LEN = 1024
CHUNK_INDEX = 0
JVP_SEED = 0
N_DIRECTIONS = 4
# Subset of the estimator's 32 even-valid midpoints: front / ~1/3 / ~2/3 / back.
TOKEN_PICKS_IN_EVEN32 = (0, 10, 21, 31)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same RTX 3060 compatibility flag as forward_sanity.py.",
    )
    parser.add_argument(
        "--skip-unjitted",
        action="store_true",
        help="Skip unjitted f / filter_jvp (can be slow). Still runs jit paths.",
    )
    return parser.parse_args()


def vector_diff(a, b) -> dict:
    import numpy as np

    a_np = np.asarray(a)
    b_np = np.asarray(b)
    absd = np.abs(a_np.astype(np.float64) - b_np.astype(np.float64))
    i = int(np.argmax(absd))
    return {
        "max_abs": float(absd.max()),
        "mean_abs": float(absd.mean()),
        "argmax_token": i,
        "dtype_a": str(a_np.dtype),
        "dtype_b": str(b_np.dtype),
        "shape": list(a_np.shape),
    }


def summarize_gaps(rows: list[dict]) -> dict:
    import numpy as np

    abs_ = np.array([r["abs_diff"] for r in rows], dtype=np.float64)
    rel_ = np.array([r["relative_diff"] for r in rows], dtype=np.float64)
    return {
        "n_pairs": len(rows),
        "median_abs": float(np.median(abs_)),
        "max_abs": float(np.max(abs_)),
        "median_relative": float(np.median(rel_)),
        "max_relative": float(np.max(rel_)),
        "argmax_pair": rows[int(np.argmax(abs_))]["pair"] if rows else None,
    }


def timed(label: str, fn, *, warmup: bool = False):
    import jax

    t0 = time.perf_counter()
    result = fn()
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)
    elapsed = time.perf_counter() - t0
    kind = "compile+run" if warmup else "run"
    print(f"  time[{kind}] {label}: {elapsed:.3f}s", flush=True)
    return result, elapsed


def run_one_condition(
    *,
    condition_name: str,
    compute_dtype_override: str | None,
    attention_cudnn_override: bool,
    skip_unjitted: bool,
) -> dict:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from omegaconf import OmegaConf

    from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
    from ttt.model.data import tree_slice
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import BlockCollectionSplit, MetaModel
    from ttt.utils.filter_utils import get_filter_spec
    from ttt.utils.jax_utils import clone_pytree, eval_shape_and_sharding, set_random_seed, tree_rearrange

    cfg = compose_baseline_config()
    compute_dtype_config_before = str(cfg.model.compute_dtype)
    if compute_dtype_override is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.model.compute_dtype = compute_dtype_override
    compute_dtype_used = str(cfg.model.compute_dtype)

    add_e2e_to_path()
    gpu = describe_gpu()
    cfg.model.seq_len = cfg.training.seq_length
    mini_batch = int(cfg.model.mini_batch_size)

    probe = np.fromfile(PROBE_PATH, dtype="<u4")
    if probe.size < SEQ_LEN + 1:
        raise SystemExit(f"{PROBE_PATH} has {probe.size} tokens, need {SEQ_LEN + 1}")
    seq = tokens_to_batch(
        jnp.asarray(probe[: SEQ_LEN + 1], dtype=jnp.int32),
        bos_token_id=int(cfg.model.bos_token_id),
    )

    key = set_random_seed(cfg.training.model_seed)
    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh

    @eqx.filter_jit
    def create_sharded_model_and_state():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = jax.device_put(state, jax.NamedSharding(mesh, P()))
        model = model_sharding.shard_params(model)
        return model, state

    print(f"\n======== condition: {condition_name}  compute_dtype={compute_dtype_used} ========", flush=True)
    print(f"config compute_dtype before override: {compute_dtype_config_before}", flush=True)

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

    out: dict = {
        "condition_name": condition_name,
        "compute_dtype_config_before": compute_dtype_config_before,
        "compute_dtype_used": compute_dtype_used,
        "compute_dtype_override": compute_dtype_override,
        "gpu": asdict(gpu),
    }

    with mesh:
        block_collection = model.language_model.model.h.blocks
        prime_storage = model.language_model.model.h.prime_storage if cfg.model.prime else None
        state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
        state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
        new_collection = BlockCollectionSplit(
            cfg.model, block_collection=block_collection, prime_storage=prime_storage, key=jrandom.PRNGKey(0)
        )
        model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)
        model_outer = jax.tree.map(lambda p: p.astype(model.state_dtype), model)

        spec_inner = get_filter_spec(model_outer, cfg.training.spec_inner, "inner parameters")
        inner_w0, _ = eqx.partition(model_outer, spec_inner)
        _, outer_params = eqx.partition(model_outer, spec_inner)

        xt_embed = model_outer.language_model.wte_call(seq.input_ids)
        prefix_output = eqx.filter_checkpoint(model_outer.language_model.prefix_call)(
            model_outer.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq
        ).last_hidden_state
        prefix_output.block_until_ready()

        seq_chunks = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=mini_batch)
        prefix_chunks = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=mini_batch)
        seq_chunk = tree_slice(seq_chunks, CHUNK_INDEX)
        prefix_chunk = prefix_chunks[CHUNK_INDEX]
        suffix_state_canonical = clone_pytree(state_suffix)

        def fresh_state():
            return clone_pytree(suffix_state_canonical)

        def token_ttt_ce_vector(inner, state):
            model_i = eqx.combine(inner, outer_params)
            lm_out = model_i.language_model.suffix_call(
                prefix_outputs=prefix_chunk, state=state, seq=seq_chunk
            )
            return token_ttt_ce_from_logits(lm_out.logits, seq_chunk.target_tokens), lm_out.logits.dtype

        def token_ttt_ce_only(inner, state):
            ce, _dtype = token_ttt_ce_vector(inner, state)
            return ce

        def token_ttt_ce_at_index(inner, idx, state):
            return token_ttt_ce_only(inner, state)[idx]

        def jvp_token_ttt_ce(inner, tangent, state):
            def f(w):
                return token_ttt_ce_only(w, state)

            return eqx.filter_jvp(f, (inner,), (tangent,))

        f_jit = eqx.filter_jit(token_ttt_ce_only)
        jvp_jit = eqx.filter_jit(jvp_token_ttt_ce)
        grad_token_jit = eqx.filter_jit(eqx.filter_grad(token_ttt_ce_at_index))

        selected32, selection_meta = select_even_valid_indices(np.asarray(seq_chunk.loss_masks), 32)
        tokens = [selected32[i] for i in TOKEN_PICKS_IN_EVEN32]
        print(f"  selected tokens (front/mid/back of even-32): {tokens}", flush=True)

        inner_before = clone_pytree(inner_w0)
        state_before = clone_pytree(suffix_state_canonical)

        # --- purity ---
        y_jit_1, t_f_compile = timed("jit(f) #1", lambda: f_jit(inner_w0, fresh_state()), warmup=True)
        y_jit_2, t_f_run = timed("jit(f) #2", lambda: f_jit(inner_w0, fresh_state()))
        y1 = np.asarray(y_jit_1)
        y2 = np.asarray(y_jit_2)
        purity = {
            "jit_f_repeat_max_abs_diff": float(np.max(np.abs(y1 - y2))),
            "inner_w_max_abs_diff": pytree_max_abs_diff(inner_before, inner_w0),
            "canonical_s0_max_abs_diff": pytree_max_abs_diff(state_before, suffix_state_canonical),
            "token_ttt_ce_shape": list(y1.shape),
            "token_ttt_ce_dtype": str(y_jit_1.dtype),
        }
        print(
            f"  purity jit(f) repeat={purity['jit_f_repeat_max_abs_diff']:.3e}  "
            f"W={purity['inner_w_max_abs_diff']:.3e}  S0={purity['canonical_s0_max_abs_diff']:.3e}",
            flush=True,
        )

        _, logits_dtype = token_ttt_ce_vector(inner_w0, fresh_state())
        logits_dtype = str(logits_dtype)

        jvp_keys = jrandom.split(jrandom.PRNGKey(JVP_SEED), N_DIRECTIONS)
        v0 = rademacher_like(inner_w0, jvp_keys[0])

        # --- Diagnostic 1: primal paths ---
        print("  --- diagnostic 1: primal paths ---", flush=True)
        primals = {"jit_f": y1}
        primal_times = {"jit_f_compile_s": t_f_compile, "jit_f_run_s": t_f_run}

        (y_jvp_jit, jv_jit), t_jvp_compile = timed(
            "jit(jvp) #1", lambda: jvp_jit(inner_w0, v0, fresh_state()), warmup=True
        )
        primals["jit_jvp_primal"] = np.asarray(y_jvp_jit)
        primal_times["jit_jvp_compile_s"] = t_jvp_compile
        jv0 = np.asarray(jv_jit)

        if skip_unjitted:
            primal_times["unjitted_skipped"] = True
        else:
            y_raw, t_raw = timed("unjitted f", lambda: token_ttt_ce_only(inner_w0, fresh_state()))
            primals["unjitted_f"] = np.asarray(y_raw)
            primal_times["unjitted_f_s"] = t_raw
            (y_jvp_raw, _), t_jvp_raw = timed(
                "unjitted filter_jvp primal", lambda: jvp_token_ttt_ce(inner_w0, v0, fresh_state())
            )
            primals["unjitted_jvp_primal"] = np.asarray(y_jvp_raw)
            primal_times["unjitted_jvp_s"] = t_jvp_raw

        names = list(primals.keys())
        pairwise = {}
        for i, na in enumerate(names):
            for nb in names[i + 1 :]:
                pairwise[f"{na}__vs__{nb}"] = vector_diff(primals[na], primals[nb])
                d = pairwise[f"{na}__vs__{nb}"]
                print(
                    f"    {na} vs {nb}: max_abs={d['max_abs']:.6g} mean_abs={d['mean_abs']:.6g} "
                    f"argmax_t={d['argmax_token']} dtype={d['dtype_a']}/{d['dtype_b']}",
                    flush=True,
                )

        # --- Diagnostic 2: 4 tokens x 4 directions ---
        print("  --- diagnostic 2: tokens x directions ---", flush=True)
        grads = {}
        for i, t_idx in enumerate(tokens):
            g_t, dt = timed(
                f"exact grad t={t_idx}",
                lambda t=t_idx: grad_token_jit(inner_w0, jnp.int32(t), fresh_state()),
                warmup=(i == 0),
            )
            grads[t_idx] = g_t
            if i == 0:
                primal_times["exact_grad_compile_s"] = dt

        jvs = [jv0]
        for k_i in range(1, N_DIRECTIONS):
            v_k = rademacher_like(inner_w0, jvp_keys[k_i])
            (_, jv_k), _dt = timed(
                f"jvp direction {k_i}",
                lambda v=v_k: jvp_jit(inner_w0, v, fresh_state()),
            )
            jvs.append(np.asarray(jv_k))

        pair_rows = []
        for k_i in range(N_DIRECTIONS):
            v_k = rademacher_like(inner_w0, jvp_keys[k_i])
            jv_k = jvs[k_i]
            for t_idx in tokens:
                d_vjp = float(pytree_vdot(grads[t_idx], v_k))
                d_jvp = float(np.asarray(jv_k[t_idx], dtype=np.float64))
                abs_diff = abs(d_vjp - d_jvp)
                rel = abs_diff / max(abs(d_vjp), 1e-12)
                row = {
                    "pair": f"t={t_idx},k={k_i}",
                    "token_index": int(t_idx),
                    "direction_index": int(k_i),
                    "d_vjp": d_vjp,
                    "d_jvp": d_jvp,
                    "abs_diff": abs_diff,
                    "relative_diff": rel,
                }
                pair_rows.append(row)
                print(
                    f"    t={t_idx:4d} k={k_i}  d_vjp={d_vjp:.8f}  d_jvp={d_jvp:.8f}  "
                    f"abs={abs_diff:.3e}  rel={rel:.3e}",
                    flush=True,
                )

        gap_summary = summarize_gaps(pair_rows)
        print(
            f"  gap summary: median_abs={gap_summary['median_abs']:.3e} max_abs={gap_summary['max_abs']:.3e} "
            f"median_rel={gap_summary['median_relative']:.3e} max_rel={gap_summary['max_relative']:.3e}",
            flush=True,
        )

        purity_after = {
            "inner_w_after_max_abs_diff": pytree_max_abs_diff(inner_before, inner_w0),
            "canonical_s0_after_max_abs_diff": pytree_max_abs_diff(state_before, suffix_state_canonical),
        }

    out.update(
        {
            "logits_dtype": logits_dtype,
            "param_dtype_config": str(cfg.model.param_dtype),
            "state_dtype_config": str(cfg.model.state_dtype),
            "inner_eval_cast": "state_dtype fp32 (official meta branch)",
            "tokens": tokens,
            "token_picks_in_even32": list(TOKEN_PICKS_IN_EVEN32),
            "selection_meta": selection_meta,
            "jvp_seed": JVP_SEED,
            "n_directions": N_DIRECTIONS,
            "purity": purity,
            "purity_after": purity_after,
            "primal_path_diffs": pairwise,
            "primal_times_s": primal_times,
            "directional_pairs": pair_rows,
            "directional_gap_summary": gap_summary,
        }
    )
    return out


def main(attention_cudnn_override: bool = True, skip_unjitted: bool = False) -> None:
    add_e2e_to_path()
    cfg0 = compose_baseline_config()
    from ttt.utils.jax_utils import initialize_distibuted

    initialize_distibuted(cfg0.backend)
    override = AttentionOverride(enabled=attention_cudnn_override)
    override.install()

    conditions = [
        {"condition_name": "official_bf16", "compute_dtype_override": None},
        {"condition_name": "diagnostic_fp32", "compute_dtype_override": "fp32"},
    ]
    results = []
    for spec in conditions:
        results.append(
            run_one_condition(
                condition_name=spec["condition_name"],
                compute_dtype_override=spec["compute_dtype_override"],
                attention_cudnn_override=attention_cudnn_override,
                skip_unjitted=skip_unjitted,
            )
        )

    bf16 = next(r for r in results if r["condition_name"] == "official_bf16")
    fp32 = next(r for r in results if r["condition_name"] == "diagnostic_fp32")
    comparison = {
        "matched_pairs": [],
        "note": (
            "Same token indices, same JVP seed/directions, same W0/S0 construction. "
            "fp32 is a diagnostic compute_dtype override, not the official baseline."
        ),
    }
    bf16_by_pair = {r["pair"]: r for r in bf16["directional_pairs"]}
    for row in fp32["directional_pairs"]:
        b = bf16_by_pair[row["pair"]]
        comparison["matched_pairs"].append(
            {
                "pair": row["pair"],
                "bf16_abs": b["abs_diff"],
                "bf16_rel": b["relative_diff"],
                "fp32_abs": row["abs_diff"],
                "fp32_rel": row["relative_diff"],
                "abs_reduced": b["abs_diff"] - row["abs_diff"],
            }
        )
    comparison["bf16_gap_summary"] = bf16["directional_gap_summary"]
    comparison["fp32_gap_summary"] = fp32["directional_gap_summary"]

    print("\n======== bf16 vs fp32 directional gap ========", flush=True)
    print(
        f"  bf16 median_rel={bf16['directional_gap_summary']['median_relative']:.3e} "
        f"max_rel={bf16['directional_gap_summary']['max_relative']:.3e}",
        flush=True,
    )
    print(
        f"  fp32 median_rel={fp32['directional_gap_summary']['median_relative']:.3e} "
        f"max_rel={fp32['directional_gap_summary']['max_relative']:.3e}",
        flush=True,
    )

    SANITY_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = SANITY_RESULT_DIR / "jvp_vjp_numerical_diagnostic_t1024"
    import numpy as np

    np.savez(
        stem.with_suffix(".npz"),
        tokens=np.asarray(bf16["tokens"], dtype=np.int32),
        bf16_d_vjp=np.asarray([r["d_vjp"] for r in bf16["directional_pairs"]]),
        bf16_d_jvp=np.asarray([r["d_jvp"] for r in bf16["directional_pairs"]]),
        fp32_d_vjp=np.asarray([r["d_vjp"] for r in fp32["directional_pairs"]]),
        fp32_d_jvp=np.asarray([r["d_jvp"] for r in fp32["directional_pairs"]]),
    )
    meta = {
        "question": (
            "Where does <g_t,v> vs (Jv)[t] disagree at the frozen T=1024 chunk-0 point?"
        ),
        "not_an_estimator_study": True,
        "seq_len": SEQ_LEN,
        "chunk_index": CHUNK_INDEX,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "conditions": results,
        "bf16_vs_fp32": comparison,
        "finite_difference": "not run (reserved if 1-3 do not locate the cause)",
    }
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nraw:  {stem.with_suffix('.npz')}", flush=True)
    print(f"meta: {stem.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(
        attention_cudnn_override=args.attention_cudnn_override,
        skip_unjitted=args.skip_unjitted,
    )
