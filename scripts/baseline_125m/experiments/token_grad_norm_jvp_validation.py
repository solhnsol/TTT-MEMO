"""Validate a token-wise fast-weight gradient-norm estimator (exact vs JVP).

Does not edit e2e/. Does not change official TTT loss semantics.

Question for this script (not saliency, not long-range memory):

    At one frozen TTT evaluation point (T=1024, chunk 0, W0, S0),
    does the Rademacher JVP estimator recover the exact token-wise
    ||∇_{W_fast} l_t^TTT||_2 well enough to use later?

Primary token loss l_t^TTT is the unnormalized per-token NLL from the same
fp32 log-softmax as official inner-loop CE. It is *not* metrics['token_nll_loss'].

Official `cross_entropy_loss_and_accuracy` returns two *scalars* (`loss` and
`loss_pure_ce`), and in the current upstream they are identical masked means.
It does not return a per-token vector, so this script reproduces only the
unmasked fp32 take-along-axis NLL (the term inside that helper, before mask
and reduction). Mask is applied for valid-token selection and for the official
chunk reduction

    L = sum_t mask_t l_t^TTT / sum_t mask_t

not absorbed into l_t itself.

Every Exact/JVP call clones the canonical pre-chunk suffix state with official
`clone_pytree`. `new_suffix_state` is never carried forward. Inner SGD is not
run; gradients are pre-clip raw ∇_{W_fast}.
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

from baseline_config import CHECKPOINT_DIR, EXPERIMENT_RESULT_DIR, REPO_ROOT, add_e2e_to_path, compose_baseline_config  # noqa: E402
from forward_sanity import (  # noqa: E402
    AttentionOverride,
    describe_gpu,
    tokens_to_batch,
)
from measurement_utils import (  # noqa: E402
    inexact_leaves,
    official_masked_reduction,
    pytree_l2_norm,
    pytree_max_abs_diff,
    pytree_vdot,
    rademacher_like,
    select_even_valid_indices,
    token_ttt_ce_from_logits,
)
from verify_tokenizer import TOKENIZER_PATH  # noqa: E402

PROBE_PATH = REPO_ROOT / "data" / "_probe_val_head.bin"
RESULT_DIR = EXPERIMENT_RESULT_DIR

SEQ_LEN = 1024
CHUNK_INDEX = 0
K_LIST_DEFAULT = (1, 2, 4, 8, 16, 32)
NORM_ACCUM_DTYPE = "float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-exact-tokens",
        type=int,
        default=32,
        help="Exact backward count. First run uses 32; 64 is the same code path.",
    )
    parser.add_argument(
        "--jvp-seed",
        type=int,
        default=0,
        help="PRNG seed for the 32 Rademacher directions. Independent of token selection.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=32,
        help="Generate this many directions; K in {1,2,4,...} uses the prefix.",
    )
    parser.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same RTX 3060 compatibility flag as forward_sanity.py.",
    )
    return parser.parse_args()


def _path_str(path) -> str:
    import jax

    return jax.tree_util.keystr(path)


def inspect_inexact_paths(tree) -> list[dict]:
    import equinox as eqx
    import jax

    rows = []
    for path, value in jax.tree.flatten_with_path(tree)[0]:
        if eqx.is_inexact_array(value):
            rows.append(
                {
                    "path": _path_str(path),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "size": int(value.size),
                }
            )
    return rows


def leading_axis_size(tree) -> int:
    leaves = inexact_leaves(tree)
    if not leaves:
        return 0
    sizes = {int(leaf.shape[0]) for leaf in leaves if leaf.ndim >= 1}
    if len(sizes) != 1:
        raise RuntimeError(f"expected a shared leading axis, got {sizes}")
    return next(iter(sizes))


def slice_prime_layer(feed_forward_prime, layer_i: int):
    import equinox as eqx
    import jax

    return jax.tree.map(lambda x: x[layer_i] if eqx.is_inexact_array(x) else x, feed_forward_prime)


def decode_token_id(tokenizer, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    piece = tokenizer.id_to_token(int(token_id))
    return piece


def try_load_tokenizer():
    if not TOKENIZER_PATH.exists():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER_PATH))


def as_float(x) -> float:
    return float(x)


def finite_report(name: str, *arrays) -> dict:
    import jax.numpy as jnp

    any_nan = False
    any_inf = False
    for a in arrays:
        x = jnp.asarray(a)
        any_nan = bool(any_nan or jnp.any(jnp.isnan(x)))
        any_inf = bool(any_inf or jnp.any(jnp.isinf(x)))
    row = {"name": name, "any_nan": any_nan, "any_inf": any_inf, "all_finite": not any_nan and not any_inf}
    print(f"  finite[{name}]: nan={any_nan} inf={any_inf}", flush=True)
    return row


def timed(label: str, fn, *, warmup: bool = False):
    """Run fn(), block, return (result, seconds). warmup=True still times that call."""
    import jax

    t0 = time.perf_counter()
    result = fn()
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)
    elapsed = time.perf_counter() - t0
    kind = "compile+run" if warmup else "run"
    print(f"  time[{kind}] {label}: {elapsed:.3f}s", flush=True)
    return result, elapsed


def try_memory_stats():
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
        if not stats:
            return None
        return {str(k): (int(v) if isinstance(v, (int,)) else v) for k, v in stats.items()}
    except Exception:
        return None


def correlations(exact, estimated) -> dict:
    import numpy as np

    exact = np.asarray(exact, dtype=np.float64)
    estimated = np.asarray(estimated, dtype=np.float64)
    rel = np.abs(estimated - exact) / np.maximum(np.abs(exact), 1e-12)
    row = {
        "n": int(exact.size),
        "pearson": None,
        "spearman": None,
        "relative_error_mean": float(np.mean(rel)),
        "relative_error_median": float(np.median(rel)),
        "relative_error_max": float(np.max(rel)),
        "exact_mean": float(np.mean(exact)),
        "estimated_mean": float(np.mean(estimated)),
    }
    try:
        from scipy.stats import pearsonr, spearmanr

        if exact.size >= 2 and np.std(exact) > 0 and np.std(estimated) > 0:
            row["pearson"] = float(pearsonr(exact, estimated).statistic)
            row["spearman"] = float(spearmanr(exact, estimated).statistic)
        else:
            row["pearson"] = None
            row["spearman"] = None
            row["correlation_note"] = "undefined: fewer than 2 points or zero variance"
    except Exception as exc:
        row["correlation_error"] = repr(exc)
    return row


def leafwise_max_diffs(a, b) -> list[dict]:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    b_leaves = {jax.tree_util.keystr(p): v for p, v in jax.tree.flatten_with_path(b)[0]}
    rows = []
    for path, x in jax.tree.flatten_with_path(a)[0]:
        if not eqx.is_inexact_array(x):
            continue
        key = jax.tree_util.keystr(path)
        y = b_leaves[key]
        diff = jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)))
        rows.append({"path": key, "max_abs_diff": float(diff)})
    rows.sort(key=lambda r: r["max_abs_diff"], reverse=True)
    return rows


def main(
    n_exact_tokens: int = 32,
    jvp_seed: int = 0,
    k_max: int = 32,
    attention_cudnn_override: bool = True,
) -> None:
    if n_exact_tokens <= 0:
        raise SystemExit("--n-exact-tokens must be positive")
    if k_max <= 0:
        raise SystemExit("--k-max must be positive")
    k_list = [k for k in K_LIST_DEFAULT if k <= k_max]
    if k_max not in k_list:
        k_list.append(k_max)
        k_list = sorted(set(k_list))

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
    from ttt.model.loss import cross_entropy_loss_and_accuracy, token_log_probs
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
    if SEQ_LEN != mini_batch:
        raise SystemExit(f"this validation is defined for T=chunk=mini_batch ({mini_batch}), got T={SEQ_LEN}")
    if SEQ_LEN % mini_batch != 0:
        raise SystemExit(f"T={SEQ_LEN} is not a multiple of mini_batch_size={mini_batch}")

    probe = np.fromfile(PROBE_PATH, dtype="<u4")
    if probe.size < SEQ_LEN + 1:
        raise SystemExit(f"{PROBE_PATH} has {probe.size} tokens, need {SEQ_LEN + 1}")
    tokens = jnp.asarray(probe[: SEQ_LEN + 1], dtype=jnp.int32)
    seq = tokens_to_batch(tokens, bos_token_id=int(cfg.model.bos_token_id))

    key = set_random_seed(cfg.training.model_seed)
    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh
    tokenizer = try_load_tokenizer()

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

    # step_index is not part of ∇_{W_fast} l_t. We do not call inner_loop_step /
    # the inner optimizer, so we leave the restored step_index untouched and
    # record that fact. (Official eval would set int32.max-100 for inner LR.)

    attention_cfg = override.report_for_config(
        force_flash=bool(cfg.model.force_flash),
        seq_modeling_block=str(cfg.model.seq_modeling_block),
    )
    print("--- setup ---", flush=True)
    print(f"GPU name:            {gpu.name}", flush=True)
    print(f"compute capability:  {gpu.compute_capability}", flush=True)
    print(f"jax device:          {gpu.jax_device}", flush=True)
    print(f"T / chunk:           {SEQ_LEN} / {CHUNK_INDEX}", flush=True)
    print(f"n exact tokens:      {n_exact_tokens}", flush=True)
    print(f"jvp seed:            {jvp_seed}", flush=True)
    print(f"K list:              {k_list}", flush=True)
    print(f"param_dtype:         {cfg.model.param_dtype}", flush=True)
    print(f"compute_dtype:       {cfg.model.compute_dtype}", flush=True)
    print(f"state_dtype:         {cfg.model.state_dtype}", flush=True)
    print(f"override enabled:    {attention_cfg['override_enabled']}", flush=True)
    print(f"step_index used:     no (pure f / no inner_loop_step)", flush=True)

    runtime = {}
    memory = {"before_compute": try_memory_stats()}
    sanity = {}

    with mesh:
        block_collection = model.language_model.model.h.blocks
        prime_storage = model.language_model.model.h.prime_storage if cfg.model.prime else None
        if prime_storage is None:
            raise SystemExit("checkpoint has prime=False; this measurement needs feed_forward_prime")

        inspect_ckpt_prime = inspect_inexact_paths(prime_storage.feed_forward_prime)
        print("\n--- W_fast inspect (checkpoint tree: h.prime_storage.feed_forward_prime) ---", flush=True)
        for row in inspect_ckpt_prime:
            print(f"  {row['path']}  shape={tuple(row['shape'])}  dtype={row['dtype']}", flush=True)

        state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
        state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)

        new_collection = BlockCollectionSplit(
            cfg.model,
            block_collection=block_collection,
            prime_storage=prime_storage,
            key=jrandom.PRNGKey(0),
        )
        model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)
        # Official meta branch: cast the whole module to state_dtype (fp32) before TTT.
        model_outer = jax.tree.map(lambda p: p.astype(model.state_dtype), model)

        prefix_len = leading_axis_size(model_outer.language_model.model.h.prefix_blocks)
        suffix_len = leading_axis_size(model_outer.language_model.model.h.suffix_blocks)
        global_layer_map = [int(prefix_len + i) for i in range(suffix_len)]
        print("\n--- layer map (runtime, not hardcoded) ---", flush=True)
        print(f"  prefix_len P={prefix_len}  suffix_len S={suffix_len}", flush=True)
        print(f"  global_layer(i)=P+i -> {global_layer_map}", flush=True)

        spec_inner = get_filter_spec(model_outer, cfg.training.spec_inner, "inner parameters")
        inner_w0, _ = eqx.partition(model_outer, spec_inner)
        _, outer_params = eqx.partition(model_outer, spec_inner)
        inspect_inner = inspect_inexact_paths(inner_w0)
        inspect_suffix_prime = inspect_inexact_paths(
            model_outer.language_model.model.h.suffix_blocks.feed_forward_prime
        )
        print("\n--- W_fast inspect (runtime spec_inner after BlockCollectionSplit) ---", flush=True)
        print(f"  spec_inner: {list(cfg.training.spec_inner)}", flush=True)
        for row in inspect_inner:
            print(f"  {row['path']}  shape={tuple(row['shape'])}  dtype={row['dtype']}", flush=True)
        n_fast = sum(r["size"] for r in inspect_inner)
        print(f"  #W_fast scalars: {n_fast:,}", flush=True)
        inner_paths = {r["path"] for r in inspect_inner}
        if not any("suffix_blocks.feed_forward_prime" in p for p in inner_paths):
            raise SystemExit("spec_inner did not select suffix_blocks.feed_forward_prime")
        if any("ffn_prime_norm" in p for p in inner_paths):
            raise SystemExit("spec_inner unexpectedly selected prime norms")

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

        def combine_inner(inner):
            return eqx.combine(inner, outer_params)

        def suffix_call_from_inner(inner, state):
            model_i = combine_inner(inner)
            return model_i.language_model.suffix_call(
                prefix_outputs=prefix_chunk, state=state, seq=seq_chunk
            )

        def token_ttt_ce_vector(inner, state):
            out = suffix_call_from_inner(inner, state)
            return token_ttt_ce_from_logits(out.logits, seq_chunk.target_tokens)

        def official_lm_loss_scalar(inner, state):
            model_i = combine_inner(inner)
            loss, _aux = model_i.lm_loss(seq_chunk, state, prefix_outputs=prefix_chunk)
            return loss

        def reconstructed_chunk_ce(inner, state):
            return official_masked_reduction(token_ttt_ce_vector(inner, state), seq_chunk.loss_masks)

        def token_ttt_ce_at_index(inner, idx, state):
            return token_ttt_ce_vector(inner, state)[idx]

        token_ttt_ce_vector_jit = eqx.filter_jit(token_ttt_ce_vector)
        official_lm_loss_scalar_jit = eqx.filter_jit(official_lm_loss_scalar)
        reconstructed_chunk_ce_jit = eqx.filter_jit(reconstructed_chunk_ce)
        suffix_call_jit = eqx.filter_jit(suffix_call_from_inner)
        grad_reconstructed_jit = eqx.filter_jit(eqx.filter_grad(reconstructed_chunk_ce))
        grad_official_jit = eqx.filter_jit(eqx.filter_grad(official_lm_loss_scalar))
        grad_token_jit = eqx.filter_jit(eqx.filter_grad(token_ttt_ce_at_index))

        def jvp_token_ttt_ce(inner, tangent, state):
            def f(w):
                return token_ttt_ce_vector(w, state)

            primals_out, tangents_out = eqx.filter_jvp(f, (inner,), (tangent,))
            return primals_out, tangents_out

        jvp_token_ttt_ce_jit = eqx.filter_jit(jvp_token_ttt_ce)

        selected, selection_meta = select_even_valid_indices(np.asarray(seq_chunk.loss_masks), n_exact_tokens)
        print("\n--- selected exact tokens ---", flush=True)
        print(f"  rule: {selection_meta['rule']}", flush=True)
        print(f"  n_valid={selection_meta['n_valid']}  n_selected={selection_meta['n_selected']}", flush=True)
        print(f"  indices: {selected}", flush=True)

        # ---- state purity ----
        print("\n--- state purity: f(W0) twice ---", flush=True)
        inner_before = clone_pytree(inner_w0)
        state_before = clone_pytree(suffix_state_canonical)
        y1, t_ce_compile = timed("token_ttt_ce_vector #1", lambda: token_ttt_ce_vector_jit(inner_w0, fresh_state()), warmup=True)
        y2, t_ce_run = timed("token_ttt_ce_vector #2", lambda: token_ttt_ce_vector_jit(inner_w0, fresh_state()))
        y1_np = np.asarray(y1)
        y2_np = np.asarray(y2)
        purity = {
            "token_ttt_ce_max_abs_diff": float(np.max(np.abs(y1_np - y2_np))),
            "inner_w_max_abs_diff": pytree_max_abs_diff(inner_before, inner_w0),
            "canonical_suffix_state_max_abs_diff": pytree_max_abs_diff(state_before, suffix_state_canonical),
            "token_ttt_ce_shape": list(y1_np.shape),
            "token_ttt_ce_dtype": str(y1.dtype),
        }
        sanity["purity"] = purity
        runtime["token_ttt_ce_compile_s"] = t_ce_compile
        runtime["token_ttt_ce_run_s"] = t_ce_run
        print(f"  ce vector shape={y1_np.shape} dtype={y1.dtype}", flush=True)
        print(f"  ce #1 vs #2 max abs diff: {purity['token_ttt_ce_max_abs_diff']:.3e}", flush=True)
        print(f"  inner W before/after max abs: {purity['inner_w_max_abs_diff']:.3e}", flush=True)
        print(f"  canonical S0 before/after max abs: {purity['canonical_suffix_state_max_abs_diff']:.3e}", flush=True)
        sanity["finite_token_ttt_ce"] = finite_report("token_ttt_ce", y1)
        purity_ok = (
            np.isfinite(y1_np).all()
            and purity["token_ttt_ce_max_abs_diff"] == 0.0
            and purity["inner_w_max_abs_diff"] == 0.0
            and purity["canonical_suffix_state_max_abs_diff"] == 0.0
        )
        if y1_np.shape != (mini_batch,):
            raise SystemExit(f"token_ttt_ce_vector shape {y1_np.shape} != ({mini_batch},)")
        if not purity_ok:
            print("PURITY FAILED. Exact/JVP results must not be interpreted.", flush=True)

        # ---- value-level reconstruction ----
        print("\n--- scalar reconstruction identity ---", flush=True)
        out0 = suffix_call_jit(inner_w0, fresh_state())
        logits0 = out0.logits
        token_ttt_ce = token_ttt_ce_from_logits(logits0, seq_chunk.target_tokens)
        token_nll_legacy = -token_log_probs(logits0, seq_chunk.target_tokens)
        sanity["ce_from_f_vs_direct_suffix_call_max_abs_diff"] = float(
            np.max(np.abs(y1_np - np.asarray(token_ttt_ce)))
        )
        L_from_same_logits_helper = cross_entropy_loss_and_accuracy(
            logits0, seq_chunk.target_tokens, seq_chunk.loss_masks
        )[0]
        L_reconstructed = official_masked_reduction(token_ttt_ce, seq_chunk.loss_masks)
        L_official, t_off_compile = timed(
            "official lm_loss scalar #1",
            lambda: official_lm_loss_scalar_jit(inner_w0, fresh_state()),
            warmup=True,
        )
        L_official_2, t_off_run = timed(
            "official lm_loss scalar #2",
            lambda: official_lm_loss_scalar_jit(inner_w0, fresh_state()),
        )
        L_rec_jit, _ = timed(
            "reconstructed_chunk_ce",
            lambda: reconstructed_chunk_ce_jit(inner_w0, fresh_state()),
        )
        runtime["official_lm_loss_compile_s"] = t_off_compile
        runtime["official_lm_loss_run_s"] = t_off_run
        scalar_id = {
            "L_reconstructed": as_float(L_reconstructed),
            "L_reconstructed_from_f": as_float(L_rec_jit),
            "L_official_lm_loss": as_float(L_official),
            "L_official_repeat": as_float(L_official_2),
            "L_official_helper_on_same_logits": as_float(L_from_same_logits_helper),
            "abs_diff_reconstructed_vs_lm_loss": abs(as_float(L_reconstructed) - as_float(L_official)),
            "abs_diff_reconstructed_vs_helper": abs(as_float(L_reconstructed) - as_float(L_from_same_logits_helper)),
            "abs_diff_lm_loss_repeat": abs(as_float(L_official) - as_float(L_official_2)),
            "loss_pure_ce_note": (
                "cross_entropy_loss_and_accuracy returns two scalars (loss, loss_pure_ce); "
                "they are identical masked means, not a per-token vector. "
                "Primary l_t is the unmasked fp32 NLL term inside that helper."
            ),
        }
        sanity["scalar_identity"] = scalar_id
        sanity["finite_legacy_nll"] = finite_report("token_nll_legacy", token_nll_legacy)
        print(f"  L_reconstructed:              {scalar_id['L_reconstructed']:.8f}", flush=True)
        print(f"  L_official lm_loss:           {scalar_id['L_official_lm_loss']:.8f}", flush=True)
        print(f"  L_official helper same logits:{scalar_id['L_official_helper_on_same_logits']:.8f}", flush=True)
        print(f"  abs diff reconstructed vs lm_loss: {scalar_id['abs_diff_reconstructed_vs_lm_loss']:.3e}", flush=True)

        # ---- gradient identity ----
        print("\n--- gradient identity (reconstructed vs official scalar, inner only) ---", flush=True)
        g_rec, t_grec_compile = timed(
            "grad reconstructed #1",
            lambda: grad_reconstructed_jit(inner_w0, fresh_state()),
            warmup=True,
        )
        g_off, t_goff_compile = timed(
            "grad official lm_loss #1",
            lambda: grad_official_jit(inner_w0, fresh_state()),
            warmup=True,
        )
        g_rec2, t_grec_run = timed(
            "grad reconstructed #2",
            lambda: grad_reconstructed_jit(inner_w0, fresh_state()),
        )
        g_off2, t_goff_run = timed(
            "grad official lm_loss #2",
            lambda: grad_official_jit(inner_w0, fresh_state()),
        )
        runtime["grad_reconstructed_compile_s"] = t_grec_compile
        runtime["grad_reconstructed_run_s"] = t_grec_run
        runtime["grad_official_compile_s"] = t_goff_compile
        runtime["grad_official_run_s"] = t_goff_run

        diff_tree = jax.tree.map(
            lambda x, y: (x.astype(jnp.float32) - y.astype(jnp.float32)) if eqx.is_inexact_array(x) else x,
            g_rec,
            g_off,
        )
        g_off_norm = float(pytree_l2_norm(g_off))
        g_rec_norm = float(pytree_l2_norm(g_rec))
        g_diff_norm = float(pytree_l2_norm(diff_tree))
        grad_id = {
            "official_inner_grad_l2": g_off_norm,
            "reconstructed_inner_grad_l2": g_rec_norm,
            "l2_diff": g_diff_norm,
            "relative_l2_error": g_diff_norm / max(g_off_norm, 1e-12),
            "max_abs_diff": pytree_max_abs_diff(g_rec, g_off),
            "repeat_official_max_abs_diff": pytree_max_abs_diff(g_off, g_off2),
            "repeat_reconstructed_max_abs_diff": pytree_max_abs_diff(g_rec, g_rec2),
            "leafwise_max_abs_diff": leafwise_max_diffs(g_rec, g_off),
            "note": (
                "eqx.filter_grad on frozen inner->combine->lm_loss scalar, "
                "not inner_loop_step / optimizer / clip"
            ),
        }
        sanity["gradient_identity"] = grad_id
        print(f"  ||g_official||_2:      {g_off_norm:.8f}", flush=True)
        print(f"  ||g_reconstructed||_2: {g_rec_norm:.8f}", flush=True)
        print(f"  L2 diff:               {g_diff_norm:.3e}", flush=True)
        print(f"  relative L2:           {grad_id['relative_l2_error']:.3e}", flush=True)
        print(f"  max abs diff:          {grad_id['max_abs_diff']:.3e}", flush=True)
        if grad_id["leafwise_max_abs_diff"]:
            top = grad_id["leafwise_max_abs_diff"][0]
            print(f"  worst leaf: {top['path']}  max_abs={top['max_abs_diff']:.3e}", flush=True)

        identity_nan = not np.isfinite([g_off_norm, g_rec_norm, g_diff_norm]).all()
        if identity_nan:
            print("GRADIENT IDENTITY has NaN/Inf. Stopping before Exact/JVP estimator.", flush=True)
            proceed_estimator = False
        elif not purity_ok:
            print("Purity failed. Stopping before Exact/JVP estimator.", flush=True)
            proceed_estimator = False
        else:
            proceed_estimator = True
            print("Gradient identity recorded with no NaN/Inf; no pass/fail threshold applied.", flush=True)

        exact_norms = None
        exact_layer_norms = None
        estimated_by_k = None
        identity_gv = None
        corr_by_k = None
        exact_meta_rows = None

        if proceed_estimator:
            # ---- exact grads ----
            print("\n--- exact token grads ---", flush=True)
            idx0 = jnp.int32(selected[0])
            g0, t_exact_compile = timed(
                f"exact grad token {int(selected[0])} #1",
                lambda: grad_token_jit(inner_w0, idx0, fresh_state()),
                warmup=True,
            )
            runtime["exact_grad_compile_s"] = t_exact_compile
            exact_norms = []
            exact_layer_norms = []
            exact_meta_rows = []
            t_exact_loop = 0.0
            g_first = g0
            for i, t_idx in enumerate(selected):
                if i == 0:
                    g_t = g0
                    dt = 0.0
                else:
                    g_t, dt = timed(
                        f"exact grad token {t_idx}",
                        lambda t=t_idx: grad_token_jit(inner_w0, jnp.int32(t), fresh_state()),
                    )
                    t_exact_loop += dt
                total = float(pytree_l2_norm(g_t))
                per_layer = []
                prime_g = g_t.language_model.model.h.suffix_blocks.feed_forward_prime
                for layer_i in range(suffix_len):
                    per_layer.append(float(pytree_l2_norm(slice_prime_layer(prime_g, layer_i))))
                exact_norms.append(total)
                exact_layer_norms.append(per_layer)
                in_id = int(seq_chunk.input_ids[t_idx])
                tgt_id = int(seq_chunk.target_tokens[t_idx])
                exact_meta_rows.append(
                    {
                        "loss_index_t": int(t_idx),
                        "input_token_id": in_id,
                        "target_token_id": tgt_id,
                        "decoded_target": decode_token_id(tokenizer, tgt_id),
                        "valid_mask": bool(seq_chunk.loss_masks[t_idx]),
                        "token_ttt_ce": float(y1_np[t_idx]),
                        "token_nll_legacy": float(np.asarray(token_nll_legacy)[t_idx]),
                        "exact_grad_norm_total": total,
                        "exact_grad_norm_per_suffix_layer": per_layer,
                        "global_layer_per_suffix_layer": global_layer_map,
                    }
                )
                if i == 0:
                    g_first = g_t
                print(
                    f"  t={t_idx:4d} target={tgt_id}  G={total:.6f}  layers={['%.4f' % z for z in per_layer]}",
                    flush=True,
                )
            runtime["exact_grad_remaining_s"] = t_exact_loop
            runtime["exact_grad_total_after_compile_s"] = t_exact_loop
            sanity["finite_exact_norms"] = finite_report("exact_norms", jnp.asarray(exact_norms))
            sanity["exact_inner_w_after_loop_max_abs_diff"] = pytree_max_abs_diff(inner_before, inner_w0)
            sanity["exact_canonical_state_after_loop_max_abs_diff"] = pytree_max_abs_diff(
                state_before, suffix_state_canonical
            )

            # ---- JVP ----
            print("\n--- JVP estimator (sequential, unnormalized Rademacher) ---", flush=True)
            jvp_keys = jrandom.split(jrandom.PRNGKey(jvp_seed), k_max)
            v0 = rademacher_like(inner_w0, jvp_keys[0])
            (y_jvp, jv0), t_jvp_compile = timed(
                "jvp direction 0",
                lambda: jvp_token_ttt_ce_jit(inner_w0, v0, fresh_state()),
                warmup=True,
            )
            runtime["jvp_compile_s"] = t_jvp_compile
            jv0_f = np.asarray(jv0, dtype=np.float32)
            y_jvp_np = np.asarray(y_jvp)
            sanity["jvp_vs_f_ce_max_abs_diff"] = float(np.max(np.abs(y_jvp_np - y1_np)))

            # Exact ↔ JVP identity on selected[0], direction 0
            direct = float(pytree_vdot(g_first, v0))
            jvp_t = float(jv0_f[selected[0]])
            identity_gv = {
                "token_index": int(selected[0]),
                "direction_index": 0,
                "direct_g_dot_v": direct,
                "jvp_at_t": jvp_t,
                "abs_diff": abs(direct - jvp_t),
                "relative_diff": abs(direct - jvp_t) / max(abs(direct), 1e-12),
                "accumulation_dtype": NORM_ACCUM_DTYPE,
                "note": "no pre-set pass/fail threshold; bf16 compute is in the suffix",
            }
            sanity["exact_jvp_directional_identity"] = identity_gv
            print(f"  <g_t, v> = {direct:.8f}", flush=True)
            print(f"  (Jv)[t]  = {jvp_t:.8f}", flush=True)
            print(f"  abs diff = {identity_gv['abs_diff']:.3e}  rel = {identity_gv['relative_diff']:.3e}", flush=True)

            sumsq = jv0_f.astype(np.float64) ** 2
            estimated_by_k = {}
            t_jvp_loop = 0.0
            if 1 in k_list:
                estimated_by_k[1] = np.sqrt(sumsq / 1.0)
            for k_i in range(1, k_max):
                v_k = rademacher_like(inner_w0, jvp_keys[k_i])
                (_, jv_k), dt = timed(
                    f"jvp direction {k_i}",
                    lambda v=v_k: jvp_token_ttt_ce_jit(inner_w0, v, fresh_state()),
                )
                t_jvp_loop += dt
                jv_np = np.asarray(jv_k, dtype=np.float32)
                sumsq = sumsq + jv_np.astype(np.float64) ** 2
                k_now = k_i + 1
                if k_now in k_list:
                    estimated_by_k[k_now] = np.sqrt(sumsq / float(k_now))
                    print(f"  stored cumulative K={k_now}", flush=True)
            runtime["jvp_remaining_s"] = t_jvp_loop
            sanity["finite_jvp"] = finite_report(
                "jvp_estimates",
                *[jnp.asarray(estimated_by_k[k]) for k in k_list],
            )
            sanity["jvp_inner_w_after_loop_max_abs_diff"] = pytree_max_abs_diff(inner_before, inner_w0)
            sanity["jvp_canonical_state_after_loop_max_abs_diff"] = pytree_max_abs_diff(
                state_before, suffix_state_canonical
            )

            print("\n--- exact vs Ghat on selected tokens ---", flush=True)
            corr_by_k = {}
            exact_arr = np.asarray(exact_norms, dtype=np.float64)
            for k in k_list:
                ghat_sel = np.asarray(estimated_by_k[k], dtype=np.float64)[np.asarray(selected)]
                corr_by_k[str(k)] = correlations(exact_arr, ghat_sel)
                c = corr_by_k[str(k)]
                print(
                    f"  K={k:2d}  spearman={c['spearman']}  pearson={c['pearson']}  "
                    f"rel_med={c['relative_error_median']:.4f}  rel_max={c['relative_error_max']:.4f}",
                    flush=True,
                )

        memory["after_compute"] = try_memory_stats()
        attention_cfg = override.report_for_config(
            force_flash=bool(cfg.model.force_flash),
            seq_modeling_block=str(cfg.model.seq_modeling_block),
        )

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        stem = RESULT_DIR / f"token_grad_norm_jvp_validation_t{SEQ_LEN}"
        npz_kwargs = {
            "input_ids": np.asarray(seq_chunk.input_ids),
            "target_ids": np.asarray(seq_chunk.target_tokens),
            "valid_masks": np.asarray(seq_chunk.loss_masks),
            "token_ttt_ce": np.asarray(token_ttt_ce),
            "token_nll_legacy": np.asarray(token_nll_legacy),
            "exact_token_indices": np.asarray(selected, dtype=np.int32),
            "global_layer_index": np.asarray(global_layer_map, dtype=np.int32),
        }
        if exact_norms is not None:
            npz_kwargs["exact_grad_norm_total"] = np.asarray(exact_norms, dtype=np.float32)
            npz_kwargs["exact_grad_norm_per_layer"] = np.asarray(exact_layer_norms, dtype=np.float32)
        if estimated_by_k is not None:
            npz_kwargs["k_values"] = np.asarray(k_list, dtype=np.int32)
            npz_kwargs["estimated_grad_norm"] = np.stack([estimated_by_k[k] for k in k_list]).astype(np.float32)
        np.savez(stem.with_suffix(".npz"), **npz_kwargs)

        meta = {
            "hypothesis": (
                "At frozen chunk-0 W0/S0, Rademacher JVP estimates of "
                "||∇_{W_fast} l_t^TTT||_2 should track exact norms as K grows. "
                "G_t is a gradient-signal magnitude, not token importance."
            ),
            "seq_len": SEQ_LEN,
            "chunk_index": CHUNK_INDEX,
            "mini_batch_size": mini_batch,
            "checkpoint_dir": str(CHECKPOINT_DIR),
            "probe_path": str(PROBE_PATH),
            "spec_inner": list(cfg.training.spec_inner),
            "n_fastweight_scalars": n_fast,
            "wfast_checkpoint_prime_storage": inspect_ckpt_prime,
            "wfast_runtime_spec_inner": inspect_inner,
            "wfast_runtime_suffix_blocks_prime": inspect_suffix_prime,
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "global_layer_map": global_layer_map,
            "global_layer_rule": "global_layer(i) = prefix_len + i, prefix/suffix lens from vmap leading axis",
            "exact_token_selection": selection_meta,
            "jvp_seed": jvp_seed,
            "k_list": k_list,
            "k_max": k_max,
            "rademacher": {
                "distribution": "independent ±1 per inner parameter element",
                "normalized": False,
                "dtype_rule": "match inner leaf dtype",
            },
            "loss_definitions": {
                "primary": "l_t^TTT = -log_softmax(logits.astype(fp32))[target_t], unmasked",
                "legacy_diagnostic": "official -token_log_probs(logits, target) (no fp32 cast, no mask)",
                "chunk_reduction": "sum(mask * l_t^TTT) / sum(mask)",
            },
            "dtypes": {
                "param_dtype_config": str(cfg.model.param_dtype),
                "compute_dtype_config": str(cfg.model.compute_dtype),
                "state_dtype_config": str(cfg.model.state_dtype),
                "inner_eval_cast": str(model_outer.state_dtype),
                "inner_leaf_dtype": inspect_inner[0]["dtype"] if inspect_inner else None,
                "norm_accumulation_dtype": NORM_ACCUM_DTYPE,
                "token_ttt_ce_dtype": str(y1.dtype),
                "token_nll_legacy_dtype": str(token_nll_legacy.dtype),
            },
            "step_index": {
                "set_to_eval_warmup_value": False,
                "reason": "pure f / exact / JVP do not call inner_loop_step; step_index is not in the gradient definition",
            },
            "gpu": asdict(gpu),
            "attention": attention_cfg,
            "tokenizer_path": str(TOKENIZER_PATH) if tokenizer is not None else None,
            "proceed_to_estimator": proceed_estimator,
            "sanity": sanity,
            "correlations_exact_vs_ghat_selected_tokens": corr_by_k,
            "exact_token_rows": exact_meta_rows,
            "runtime_s": runtime,
            "memory": memory,
        }
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"\nraw:  {stem.with_suffix('.npz')}", flush=True)
        print(f"meta: {stem.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(
        n_exact_tokens=args.n_exact_tokens,
        jvp_seed=args.jvp_seed,
        k_max=args.k_max,
        attention_cudnn_override=args.attention_cudnn_override,
    )
