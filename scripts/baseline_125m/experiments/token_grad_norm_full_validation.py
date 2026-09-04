"""Full-chunk exact vs JVP token-wise fast-weight gradient-norm validation.

Not a saliency study. Does not edit e2e/. Does not overwrite
token_grad_norm_jvp_validation_t1024.*.

Question:

    At the official bf16 frozen T=1024 chunk-0 evaluation point, how well does
    the Rademacher JVP estimator recover exact reverse-mode ||∇_{W_fast} l_t^TTT||_2
    on every valid token?

JVP is *not* assumed to be a mathematically identical substitute for reverse-mode
norms under bf16 mixed precision (see STEP 2.5). This run measures empirical
agreement on the full chunk.
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

from baseline_config import (  # noqa: E402
    CHECKPOINT_DIR,
    EXPERIMENT_RESULT_DIR,
    PROBE_PATH,
    TOKENIZER_PATH,
    add_e2e_to_path,
    compose_baseline_config,
)
from measurement_utils import (  # noqa: E402
    leading_axis_size,
    pytree_l2_norm,
    pytree_max_abs_diff,
    pytree_vdot,
    rademacher_like,
    slice_prime_layer,
    token_ttt_ce_from_logits,
)
from runtime_helpers import AttentionOverride, describe_gpu, tokens_to_batch  # noqa: E402

PREV_STEM = EXPERIMENT_RESULT_DIR / "token_grad_norm_jvp_validation_t1024"
SEQ_LEN = 1024
CHUNK_INDEX = 0
JVP_SEED = 0
K_LIST = (1, 2, 4, 8, 16, 32)
K_MAX = 32
REL_EPS = 1e-12
TOP_FRACTIONS = (0.01, 0.05, 0.10)
TIE_RULE = "descending G_t, ties broken by smaller position index (np.lexsort)"
DIRECTIONAL_REGRESSION_REL = 0.05  # flag only; not a pass/fail for the estimator
PREV_JVP_REGRESSION_ABS = 1e-3
PREV_EXACT_OVERLAP_ABS = 1e-3
EVAL_POINT_CE_ABS = 1e-3
EXACT_PROGRESS_EVERY = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same RTX 3060 compatibility flag as forward_sanity.py.",
    )
    return parser.parse_args()


def block_until_ready(tree):
    import jax

    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)
    return tree


def timed(label: str, fn, *, warmup: bool = False, quiet: bool = False):
    t0 = time.perf_counter()
    result = fn()
    block_until_ready(result)
    elapsed = time.perf_counter() - t0
    if not quiet:
        kind = "compile+run" if warmup else "run"
        print(f"  time[{kind}] {label}: {elapsed:.3f}s", flush=True)
    return result, elapsed


def inspect_inexact_paths(tree) -> list[dict]:
    import equinox as eqx
    import jax

    rows = []
    for path, value in jax.tree.flatten_with_path(tree)[0]:
        if eqx.is_inexact_array(value):
            rows.append(
                {
                    "path": jax.tree_util.keystr(path),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "size": int(value.size),
                }
            )
    return rows


def to_jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def try_memory_stats():
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
        if not stats:
            return None
        return {str(k): (int(v) if isinstance(v, (int,)) else v) for k, v in stats.items()}
    except Exception:
        return None


def try_load_tokenizer():
    if not TOKENIZER_PATH.exists():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER_PATH))


def decode_token_id(tokenizer, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    return tokenizer.id_to_token(int(token_id))


def ranking_order(scores):
    """Return indices that sort scores descending; ties -> smaller index."""
    import numpy as np

    scores = np.asarray(scores)
    n = scores.size
    return np.lexsort((np.arange(n), -scores.astype(np.float64)))


def topk_overlap(exact, estimated, valid, frac: float) -> dict:
    import numpy as np

    idx = np.flatnonzero(np.asarray(valid).astype(bool))
    e = np.asarray(exact, dtype=np.float64)[idx]
    h = np.asarray(estimated, dtype=np.float64)[idx]
    n = int(idx.size)
    k = max(1, int(round(n * frac)))
    order_e = idx[ranking_order(e)]
    order_h = idx[ranking_order(h)]
    set_e = set(order_e[:k].tolist())
    set_h = set(order_h[:k].tolist())
    inter = len(set_e & set_h)
    return {
        "fraction": frac,
        "k": k,
        "n_valid": n,
        "overlap": inter,
        "topk_recall": inter / k,
        "tie_rule": TIE_RULE,
    }


def error_metrics(exact, estimated, valid, *, eps: float) -> dict:
    import numpy as np

    idx = np.flatnonzero(np.asarray(valid).astype(bool))
    e = np.asarray(exact, dtype=np.float64)[idx]
    h = np.asarray(estimated, dtype=np.float64)[idx]
    rel = np.abs(h - e) / (e + eps)
    abserr = np.abs(h - e)
    row = {
        "n": int(idx.size),
        "spearman": None,
        "pearson": None,
        "relative_error_mean": float(np.mean(rel)),
        "relative_error_median": float(np.median(rel)),
        "relative_error_p90": float(np.percentile(rel, 90)),
        "relative_error_p95": float(np.percentile(rel, 95)),
        "relative_error_max": float(np.max(rel)),
        "absolute_error_median": float(np.median(abserr)),
        "absolute_error_mean": float(np.mean(abserr)),
        "exact_mean": float(np.mean(e)),
        "estimated_mean": float(np.mean(h)),
        "eps": eps,
    }
    q25, q75 = np.percentile(e, [25, 75])
    bins = {
        "bottom_25": e <= q25,
        "middle_50": (e > q25) & (e <= q75),
        "top_25": e > q75,
    }
    by_q = {}
    for name, mask in bins.items():
        if not np.any(mask):
            by_q[name] = None
            continue
        by_q[name] = {
            "n": int(np.sum(mask)),
            "exact_g_lo": float(e[mask].min()),
            "exact_g_hi": float(e[mask].max()),
            "relative_error_median": float(np.median(rel[mask])),
            "relative_error_mean": float(np.mean(rel[mask])),
            "absolute_error_median": float(np.median(abserr[mask])),
        }
    row["relative_error_by_exact_g_quantile"] = by_q
    try:
        from scipy.stats import pearsonr, spearmanr

        if e.size >= 2 and np.std(e) > 0 and np.std(h) > 0:
            row["pearson"] = float(pearsonr(e, h).statistic)
            row["spearman"] = float(spearmanr(e, h).statistic)
        else:
            row["correlation_note"] = "undefined: fewer than 2 points or zero variance"
    except Exception as exc:
        row["correlation_error"] = repr(exc)
    return row


def checkpoint_id(path: str) -> str:
    return Path(path).name


def load_previous():
    prev_json = PREV_STEM.with_suffix(".json")
    prev_npz = PREV_STEM.with_suffix(".npz")
    meta = json.loads(prev_json.read_text()) if prev_json.exists() else None
    arrays = None
    if prev_npz.exists():
        import numpy as np

        arrays = dict(np.load(prev_npz, allow_pickle=False))
    return meta, arrays


def finite_flags(name: str, arr) -> dict:
    import numpy as np

    x = np.asarray(arr)
    finite = np.isfinite(x)
    return {
        "name": name,
        "any_nan": bool(np.any(np.isnan(x))),
        "any_inf": bool(np.any(np.isinf(x))),
        "all_finite": bool(np.all(finite)),
    }


def main(attention_cudnn_override: bool = True) -> None:
    wall0 = time.perf_counter()
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
    from ttt.model.loss import token_log_probs
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import BlockCollectionSplit, MetaModel
    from ttt.utils.filter_utils import get_filter_spec
    from ttt.utils.jax_utils import clone_pytree, eval_shape_and_sharding, initialize_distibuted, set_random_seed, tree_rearrange

    prev_meta, prev_arrays = load_previous()
    if prev_meta is None or prev_arrays is None:
        raise SystemExit(f"previous validation missing under {PREV_STEM}; refuse to run without eval-point check")

    cfg.model.seq_len = cfg.training.seq_length
    mini_batch = int(cfg.model.mini_batch_size)
    resume_step = cfg.training.resume_step
    if SEQ_LEN != mini_batch:
        raise SystemExit(f"this validation is defined for T=chunk=mini_batch ({mini_batch}), got T={SEQ_LEN}")

    eval_point_checks = {
        "seq_len": prev_meta.get("seq_len") == SEQ_LEN,
        "chunk_index": prev_meta.get("chunk_index") == CHUNK_INDEX,
        "mini_batch_size": prev_meta.get("mini_batch_size") == mini_batch,
        "checkpoint_id": checkpoint_id(prev_meta.get("checkpoint_dir", "")) == checkpoint_id(str(CHECKPOINT_DIR)),
        "probe_id": checkpoint_id(prev_meta.get("probe_path", "")) == checkpoint_id(str(PROBE_PATH)),
        "spec_inner": list(prev_meta.get("spec_inner", [])) == list(cfg.training.spec_inner),
        "jvp_seed": prev_meta.get("jvp_seed") == JVP_SEED,
        "k_list": list(prev_meta.get("k_list", [])) == list(K_LIST),
        "compute_dtype_config": prev_meta.get("dtypes", {}).get("compute_dtype_config") == str(cfg.model.compute_dtype),
        "param_dtype_config": prev_meta.get("dtypes", {}).get("param_dtype_config") == str(cfg.model.param_dtype),
        "state_dtype_config": prev_meta.get("dtypes", {}).get("state_dtype_config") == str(cfg.model.state_dtype),
        "primary_loss_def_mentions_fp32": "fp32" in str(prev_meta.get("loss_definitions", {}).get("primary", "")),
        "compute_dtype_is_official_bf16": str(cfg.model.compute_dtype) in ("bf16", "bfloat16"),
    }
    print("--- evaluation-point checks vs token_grad_norm_jvp_validation_t1024 ---", flush=True)
    for k, ok in eval_point_checks.items():
        print(f"  {k}: {ok}", flush=True)
    print(f"  resume_step (this run): {resume_step}", flush=True)
    if not all(eval_point_checks.values()):
        raise SystemExit(f"evaluation point metadata mismatch: {eval_point_checks}")

    initialize_distibuted(cfg.backend)
    gpu = describe_gpu()
    override = AttentionOverride(enabled=attention_cudnn_override)
    override.install()

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

    print("--- setup ---", flush=True)
    print(f"GPU: {gpu.name}  jax={gpu.jax_device}", flush=True)
    print(f"T/chunk={SEQ_LEN}/{CHUNK_INDEX}  compute={cfg.model.compute_dtype}  seed={JVP_SEED}", flush=True)
    print(f"resume_step: {resume_step}", flush=True)
    print("step_index: not set (no inner_loop_step)", flush=True)

    memory = {"before_compute": try_memory_stats()}
    runtime: dict = {}
    sanity: dict = {}

    with mesh:
        block_collection = model.language_model.model.h.blocks
        prime_storage = model.language_model.model.h.prime_storage if cfg.model.prime else None
        if prime_storage is None:
            raise SystemExit("checkpoint has prime=False; this measurement needs feed_forward_prime")
        state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
        state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
        new_collection = BlockCollectionSplit(
            cfg.model, block_collection=block_collection, prime_storage=prime_storage, key=jrandom.PRNGKey(0)
        )
        model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)
        model_outer = jax.tree.map(lambda p: p.astype(model.state_dtype), model)

        prefix_len = leading_axis_size(model_outer.language_model.model.h.prefix_blocks)
        suffix_len = leading_axis_size(model_outer.language_model.model.h.suffix_blocks)
        global_layer_map = [int(prefix_len + i) for i in range(suffix_len)]

        spec_inner = get_filter_spec(model_outer, cfg.training.spec_inner, "inner parameters")
        inner_w0, _ = eqx.partition(model_outer, spec_inner)
        _, outer_params = eqx.partition(model_outer, spec_inner)
        inspect_inner = inspect_inexact_paths(inner_w0)
        n_fast = sum(r["size"] for r in inspect_inner)
        inner_paths = {r["path"] for r in inspect_inner}
        if not any("suffix_blocks.feed_forward_prime" in p for p in inner_paths):
            raise SystemExit("spec_inner did not select suffix_blocks.feed_forward_prime")
        if any("ffn_prime_norm" in p for p in inner_paths):
            raise SystemExit("spec_inner unexpectedly selected prime norms")
        eval_point_checks["prefix_len"] = prev_meta.get("prefix_len") == prefix_len
        eval_point_checks["suffix_len"] = prev_meta.get("suffix_len") == suffix_len
        eval_point_checks["global_layer_map"] = list(prev_meta.get("global_layer_map", [])) == global_layer_map
        eval_point_checks["n_fastweight_scalars"] = prev_meta.get("n_fastweight_scalars") == n_fast
        print(f"  P={prefix_len} S={suffix_len} layers={global_layer_map}  #W_fast={n_fast:,}", flush=True)
        for k in ("prefix_len", "suffix_len", "global_layer_map", "n_fastweight_scalars"):
            print(f"  {k}: {eval_point_checks[k]}", flush=True)
        if not all(eval_point_checks[k] for k in ("prefix_len", "suffix_len", "global_layer_map", "n_fastweight_scalars")):
            raise SystemExit(f"runtime W_fast / layer map mismatch: {eval_point_checks}")

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
            out = model_i.language_model.suffix_call(
                prefix_outputs=prefix_chunk, state=state, seq=seq_chunk
            )
            return token_ttt_ce_from_logits(out.logits, seq_chunk.target_tokens), out.logits

        def token_ttt_ce_only(inner, state):
            ce, _logits = token_ttt_ce_vector(inner, state)
            return ce

        def token_ttt_ce_at_index(inner, idx, state):
            return token_ttt_ce_only(inner, state)[idx]

        def jvp_token_ttt_ce(inner, tangent, state):
            def f(w):
                return token_ttt_ce_only(w, state)

            return eqx.filter_jvp(f, (inner,), (tangent,))

        f_jit = eqx.filter_jit(token_ttt_ce_only)
        suffix_jit = eqx.filter_jit(token_ttt_ce_vector)
        grad_token_jit = eqx.filter_jit(eqx.filter_grad(token_ttt_ce_at_index))
        jvp_jit = eqx.filter_jit(jvp_token_ttt_ce)

        inner_before = clone_pytree(inner_w0)
        state_before = clone_pytree(suffix_state_canonical)

        (token_ttt_ce, logits), t_f_compile = timed(
            "f(W0) compile", lambda: suffix_jit(inner_w0, fresh_state()), warmup=True
        )
        y2, t_f_run = timed("f(W0) repeat", lambda: f_jit(inner_w0, fresh_state()))
        token_ttt_ce = np.asarray(token_ttt_ce)
        y2 = np.asarray(y2)
        legacy_nll = np.asarray(-token_log_probs(logits, seq_chunk.target_tokens))
        valid = np.asarray(seq_chunk.loss_masks).astype(bool)
        n_valid = int(valid.sum())
        runtime["f_compile_s"] = t_f_compile
        runtime["f_run_s"] = t_f_run

        prev_ce = np.asarray(prev_arrays["token_ttt_ce"])
        prev_ids = np.asarray(prev_arrays["input_ids"])
        prev_tgt = np.asarray(prev_arrays["target_ids"])
        prev_valid = np.asarray(prev_arrays["valid_masks"]).astype(bool)
        ce_diff = float(np.max(np.abs(token_ttt_ce.astype(np.float64) - prev_ce.astype(np.float64))))
        ids_diff = float(np.max(np.abs(np.asarray(seq_chunk.input_ids) - prev_ids)))
        tgt_diff = float(np.max(np.abs(np.asarray(seq_chunk.target_tokens) - prev_tgt)))
        mask_equal = bool(np.array_equal(valid, prev_valid))
        eval_point_checks["token_ttt_ce_max_abs_vs_prev"] = ce_diff
        eval_point_checks["input_ids_max_abs_vs_prev"] = ids_diff
        eval_point_checks["target_ids_max_abs_vs_prev"] = tgt_diff
        eval_point_checks["token_ttt_ce_matches_prev"] = ce_diff < EVAL_POINT_CE_ABS
        eval_point_checks["input_ids_match_prev"] = ids_diff == 0.0
        eval_point_checks["target_ids_match_prev"] = tgt_diff == 0.0
        eval_point_checks["valid_masks_match_prev"] = mask_equal
        print(f"  token_ttt_ce vs prev max abs: {ce_diff:.3e}", flush=True)
        print(f"  input_ids vs prev max abs:    {ids_diff:.3e}", flush=True)
        print(f"  target_ids vs prev max abs:   {tgt_diff:.3e}", flush=True)
        print(f"  valid_masks match:            {mask_equal}", flush=True)
        print(f"  valid tokens: {n_valid}/{SEQ_LEN}", flush=True)
        if not (
            eval_point_checks["token_ttt_ce_matches_prev"]
            and eval_point_checks["input_ids_match_prev"]
            and eval_point_checks["target_ids_match_prev"]
            and eval_point_checks["valid_masks_match_prev"]
        ):
            raise SystemExit("evaluation point arrays do not match previous validation; refuse to continue")

        sanity["purity_f_repeat"] = float(np.max(np.abs(token_ttt_ce - y2)))
        sanity["token_ttt_ce_shape"] = list(token_ttt_ce.shape)
        sanity["token_ttt_ce_dtype"] = str(token_ttt_ce.dtype)
        sanity["legacy_nll_dtype"] = str(legacy_nll.dtype)
        sanity["n_valid"] = n_valid
        sanity["finite_token_ttt_ce"] = finite_flags("token_ttt_ce", token_ttt_ce)
        sanity["finite_legacy_nll"] = finite_flags("legacy_nll", legacy_nll)
        if token_ttt_ce.shape != (SEQ_LEN,) or str(token_ttt_ce.dtype) != "float32":
            raise SystemExit(f"unexpected token_ttt_ce shape/dtype {token_ttt_ce.shape} {token_ttt_ce.dtype}")

        valid_positions = np.flatnonzero(valid).astype(np.int32)
        exact_norm = np.full((SEQ_LEN,), np.nan, dtype=np.float32)
        exact_norm_sq = np.full((SEQ_LEN,), np.nan, dtype=np.float32)
        exact_layer = np.full((SEQ_LEN, suffix_len), np.nan, dtype=np.float32)
        exact_times = []

        print("\n--- exact reverse-mode G_t on all valid tokens ---", flush=True)
        t0 = valid_positions[0]
        g0, t_exact_compile = timed(
            f"exact t={int(t0)}",
            lambda: grad_token_jit(inner_w0, jnp.int32(t0), fresh_state()),
            warmup=True,
        )
        runtime["exact_compile_s"] = t_exact_compile
        g_first = g0
        remaining_s = 0.0
        for i, t_idx in enumerate(valid_positions):
            if i == 0:
                g_t = g0
            else:
                g_t, dt = timed(
                    f"exact t={int(t_idx)}",
                    lambda t=int(t_idx): grad_token_jit(inner_w0, jnp.int32(t), fresh_state()),
                    quiet=True,
                )
                remaining_s += dt
                exact_times.append(dt)
            total = float(pytree_l2_norm(g_t))
            exact_norm[t_idx] = total
            exact_norm_sq[t_idx] = total * total
            prime_g = g_t.language_model.model.h.suffix_blocks.feed_forward_prime
            for layer_i in range(suffix_len):
                exact_layer[t_idx, layer_i] = float(pytree_l2_norm(slice_prime_layer(prime_g, layer_i)))
            if i == 0 or (i + 1) % EXACT_PROGRESS_EVERY == 0 or i + 1 == n_valid:
                mean_so_far = float(np.mean(exact_times)) if exact_times else t_exact_compile
                print(
                    f"  exact progress {i + 1}/{n_valid}  last G={total:.4f}  "
                    f"steady_mean_s={mean_so_far:.3f}",
                    flush=True,
                )
            if i == 0:
                g_first = g_t

        runtime["exact_remaining_s"] = remaining_s
        runtime["exact_n_after_compile"] = n_valid - 1
        runtime["exact_per_token_mean_s"] = float(np.mean(exact_times)) if exact_times else None
        runtime["exact_per_token_median_s"] = float(np.median(exact_times)) if exact_times else None
        runtime["exact_total_after_compile_s"] = remaining_s
        runtime["exact_total_s"] = t_exact_compile + remaining_s
        print(
            f"  exact compile+first={t_exact_compile:.3f}s  remaining_{n_valid - 1}={remaining_s:.3f}s  "
            f"steady mean/median={runtime['exact_per_token_mean_s']}/{runtime['exact_per_token_median_s']}",
            flush=True,
        )

        exact_overlap = None
        if "exact_token_indices" in prev_arrays and "exact_grad_norm_total" in prev_arrays:
            prev_idx = np.asarray(prev_arrays["exact_token_indices"])
            prev_g = np.asarray(prev_arrays["exact_grad_norm_total"], dtype=np.float64)
            cur_g = exact_norm[prev_idx].astype(np.float64)
            if np.any(~np.isfinite(cur_g)):
                raise SystemExit("previous exact-token indices are not valid in this run")
            max_abs = float(np.max(np.abs(cur_g - prev_g)))
            mean_abs = float(np.mean(np.abs(cur_g - prev_g)))
            exact_overlap = {
                "n": int(prev_idx.size),
                "indices": prev_idx.tolist(),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "flagged": max_abs > PREV_EXACT_OVERLAP_ABS,
            }
            print(
                f"  exact overlap vs prev 32 tokens: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
                f"flagged={exact_overlap['flagged']}",
                flush=True,
            )
            if exact_overlap["flagged"]:
                print(
                    "REGRESSION: exact G_t on the previous 32-token subset differs more than expected. "
                    "Do not treat this full validation as confirmed until explained.",
                    flush=True,
                )

        print("\n--- JVP estimator (seed=0, sequential, unnormalized Rademacher) ---", flush=True)
        jvp_keys = jrandom.split(jrandom.PRNGKey(JVP_SEED), K_MAX)
        v0 = rademacher_like(inner_w0, jvp_keys[0])
        (y_jvp, jv0), t_jvp_compile = timed(
            "jvp direction 0",
            lambda: jvp_jit(inner_w0, v0, fresh_state()),
            warmup=True,
        )
        runtime["jvp_compile_s"] = t_jvp_compile
        jv_projections = np.zeros((K_MAX, SEQ_LEN), dtype=np.float32)
        jv_projections[0] = np.asarray(jv0, dtype=np.float32)
        jvp_dir_times = []
        remaining_jvp_s = 0.0
        for k_i in range(1, K_MAX):
            v_k = rademacher_like(inner_w0, jvp_keys[k_i])
            (_, jv_k), dt = timed(
                f"jvp direction {k_i}",
                lambda v=v_k: jvp_jit(inner_w0, v, fresh_state()),
                quiet=True,
            )
            jv_projections[k_i] = np.asarray(jv_k, dtype=np.float32)
            jvp_dir_times.append(dt)
            remaining_jvp_s += dt
            if (k_i + 1) % 8 == 0 or k_i + 1 == K_MAX:
                print(
                    f"  jvp progress {k_i + 1}/{K_MAX}  last_dir_s={dt:.3f}",
                    flush=True,
                )
        runtime["jvp_remaining_s"] = remaining_jvp_s
        runtime["jvp_per_direction_mean_s"] = float(np.mean(jvp_dir_times)) if jvp_dir_times else None
        runtime["jvp_per_direction_median_s"] = float(np.median(jvp_dir_times)) if jvp_dir_times else None
        # Cumulative estimated runtime: compile once, then K sequential directions.
        # Direction 0 is inside compile+run; remaining directions use steady-state times.
        # For K=1 we report compile as the first-direction cost, not mixed into K>1 extras.
        runtime["jvp_cumulative_estimated_s"] = {}
        for k in K_LIST:
            extra = float(np.sum(jvp_dir_times[: max(k - 1, 0)])) if jvp_dir_times else 0.0
            runtime["jvp_cumulative_estimated_s"][str(k)] = t_jvp_compile + extra
        print(
            f"  jvp compile+first={t_jvp_compile:.3f}s  remaining_{K_MAX - 1}={remaining_jvp_s:.3f}s  "
            f"steady mean/median={runtime['jvp_per_direction_mean_s']}/{runtime['jvp_per_direction_median_s']}",
            flush=True,
        )

        sumsq = np.zeros((SEQ_LEN,), dtype=np.float64)
        estimated_norm = {}
        estimated_sq = {}
        for k_i in range(K_MAX):
            jv_f = jv_projections[k_i].astype(np.float64)
            sumsq = sumsq + jv_f * jv_f
            k_now = k_i + 1
            if k_now in K_LIST:
                ghat_sq = sumsq / float(k_now)
                estimated_sq[k_now] = ghat_sq.astype(np.float32)
                estimated_norm[k_now] = np.sqrt(ghat_sq).astype(np.float32)

        # Directional identity on first valid token, direction 0 — diagnostic, not pass/fail.
        t_dir = int(valid_positions[0])
        direct = float(pytree_vdot(g_first, v0))
        jvp_t = float(jv_projections[0, t_dir])
        dir_rel = abs(direct - jvp_t) / max(abs(direct), 1e-12)
        directional = {
            "token_index": t_dir,
            "direction_index": 0,
            "d_vjp": direct,
            "d_jvp": jvp_t,
            "abs_diff": abs(direct - jvp_t),
            "relative_diff": dir_rel,
            "used_as_pass_fail": False,
            "regression_flag_if_rel_gt": DIRECTIONAL_REGRESSION_REL,
            "regression_flag": dir_rel > DIRECTIONAL_REGRESSION_REL,
        }
        sanity["directional_identity"] = directional
        print(
            f"  directional t={t_dir}: <g,v>={direct:.6f} (Jv)[t]={jvp_t:.6f} "
            f"rel={dir_rel:.3e}  regression_flag={directional['regression_flag']}",
            flush=True,
        )

        sanity["inner_w_after_max_abs"] = pytree_max_abs_diff(inner_before, inner_w0)
        sanity["canonical_s0_after_max_abs"] = pytree_max_abs_diff(state_before, suffix_state_canonical)
        sanity["exact_norm_nonnegative"] = bool(np.all(exact_norm[valid] >= 0))
        sanity["jvp_estimates_nonnegative"] = bool(np.all(estimated_norm[K_MAX] >= 0))
        sanity["finite_exact"] = finite_flags("exact_norm_valid", exact_norm[valid])
        sanity["finite_jvp"] = finite_flags("estimated_k32", estimated_norm[K_MAX])
        sanity["jvp_vs_f_primal_max_abs"] = float(np.max(np.abs(np.asarray(y_jvp) - token_ttt_ce)))

        print("\n--- K-wise accuracy on all valid tokens ---", flush=True)
        metrics_by_k = {}
        topk_by_k = {}
        for k in K_LIST:
            m = error_metrics(exact_norm, estimated_norm[k], valid, eps=REL_EPS)
            top = {str(frac): topk_overlap(exact_norm, estimated_norm[k], valid, frac) for frac in TOP_FRACTIONS}
            metrics_by_k[str(k)] = m
            topk_by_k[str(k)] = top
            print(
                f"  K={k:2d}  spearman={m['spearman']}  pearson={m['pearson']}  "
                f"rel_med={m['relative_error_median']:.4f}  rel_p90={m['relative_error_p90']:.4f}  "
                f"rel_max={m['relative_error_max']:.4f}  abs_med={m['absolute_error_median']:.4f}",
                flush=True,
            )
            print(
                f"       top1%={top['0.01']['topk_recall']:.3f}  "
                f"top5%={top['0.05']['topk_recall']:.3f}  "
                f"top10%={top['0.1']['topk_recall']:.3f}",
                flush=True,
            )

        order_exact = valid_positions[ranking_order(exact_norm[valid_positions])]
        order_k16 = valid_positions[ranking_order(estimated_norm[16][valid_positions])]
        order_k32 = valid_positions[ranking_order(estimated_norm[32][valid_positions])]

        def top_rows(order, n=20):
            rows = []
            for t in order[:n]:
                t = int(t)
                tgt = int(seq_chunk.target_tokens[t])
                rows.append(
                    {
                        "position": t,
                        "input_token_id": int(seq_chunk.input_ids[t]),
                        "target_token_id": tgt,
                        "decoded_target": decode_token_id(tokenizer, tgt),
                        "token_ttt_ce": float(token_ttt_ce[t]),
                        "legacy_nll": float(legacy_nll[t]),
                        "exact_grad_norm": float(exact_norm[t]),
                        "valid_mask": bool(valid[t]),
                    }
                )
            return rows

        ranking_diag = {
            "tie_rule": TIE_RULE,
            "exact_top20": top_rows(order_exact),
            "k16_estimated_top20": top_rows(order_k16),
            "k32_estimated_top20": top_rows(order_k32),
        }

        g_valid = exact_norm[valid]
        exact_dist = {
            "n": n_valid,
            "min": float(np.min(g_valid)),
            "p25": float(np.percentile(g_valid, 25)),
            "median": float(np.median(g_valid)),
            "p75": float(np.percentile(g_valid, 75)),
            "p90": float(np.percentile(g_valid, 90)),
            "p95": float(np.percentile(g_valid, 95)),
            "max": float(np.max(g_valid)),
            "mean": float(np.mean(g_valid)),
        }
        print(
            f"\nexact G_t valid: min={exact_dist['min']:.4f} med={exact_dist['median']:.4f} "
            f"p90={exact_dist['p90']:.4f} max={exact_dist['max']:.4f}",
            flush=True,
        )

        regression = None
        if "estimated_grad_norm" in prev_arrays:
            prev_est = np.asarray(prev_arrays["estimated_grad_norm"])
            prev_k = np.asarray(prev_arrays["k_values"]) if "k_values" in prev_arrays else np.asarray(K_LIST)
            regression = {"compared": "previous estimated_grad_norm [n_k, T]", "per_k": {}}
            print("\n--- regression vs previous JVP estimates ---", flush=True)
            worst = 0.0
            for i, k in enumerate(prev_k.tolist()):
                k = int(k)
                if k not in estimated_norm:
                    continue
                a = prev_est[i].astype(np.float64)
                b = estimated_norm[k].astype(np.float64)
                max_abs = float(np.max(np.abs(a - b)))
                mean_abs = float(np.mean(np.abs(a - b)))
                regression["per_k"][str(k)] = {"max_abs": max_abs, "mean_abs": mean_abs}
                worst = max(worst, max_abs)
                print(f"  K={k:2d} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}", flush=True)
            regression["max_abs_any_k"] = worst
            regression["flagged"] = worst > PREV_JVP_REGRESSION_ABS
            if regression["flagged"]:
                print(
                    f"REGRESSION: JVP estimates differ from previous run by max abs {worst:.3e} "
                    f"> {PREV_JVP_REGRESSION_ABS}. Do not treat this full validation as confirmed.",
                    flush=True,
                )
        else:
            regression = {"compared": None, "note": "previous NPZ had no estimated_grad_norm"}

        memory["after_compute"] = try_memory_stats()
        attention_cfg = override.report_for_config(
            force_flash=bool(cfg.model.force_flash),
            seq_modeling_block=str(cfg.model.seq_modeling_block),
        )

        EXPERIMENT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        stem = EXPERIMENT_RESULT_DIR / f"token_grad_norm_full_validation_t{SEQ_LEN}"
        npz_kwargs = {
            "positions": np.arange(SEQ_LEN, dtype=np.int32),
            "input_ids": np.asarray(seq_chunk.input_ids),
            "target_ids": np.asarray(seq_chunk.target_tokens),
            "valid_masks": valid.astype(np.int8),
            "token_ttt_ce": token_ttt_ce.astype(np.float32),
            "legacy_token_nll": legacy_nll,
            "exact_grad_norm": exact_norm,
            "exact_grad_norm_sq": exact_norm_sq,
            "exact_layerwise_grad_norm": exact_layer,
            "jv_projections": jv_projections,
            "k_values": np.asarray(K_LIST, dtype=np.int32),
            "global_layer_index": np.asarray(global_layer_map, dtype=np.int32),
        }
        for k in K_LIST:
            npz_kwargs[f"estimated_grad_norm_k{k}"] = estimated_norm[k]
            npz_kwargs[f"estimated_grad_norm_sq_k{k}"] = estimated_sq[k]
        np.savez(stem.with_suffix(".npz"), **npz_kwargs)

        runtime["wall_s"] = time.perf_counter() - wall0
        meta = {
            "hypothesis": (
                "Under official bf16 TTT, the Rademacher JVP estimator empirically "
                "tracks exact reverse-mode token-wise ||∇_{W_fast} l_t^TTT||_2 on a full "
                "1024-token chunk. G_t is a gradient-signal magnitude, not importance."
            ),
            "seq_len": SEQ_LEN,
            "chunk_index": CHUNK_INDEX,
            "mini_batch_size": mini_batch,
            "checkpoint_dir": str(CHECKPOINT_DIR),
            "checkpoint_resume_step": resume_step,
            "probe_path": str(PROBE_PATH),
            "spec_inner": list(cfg.training.spec_inner),
            "n_fastweight_scalars": n_fast,
            "wfast_runtime_spec_inner": inspect_inner,
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "global_layer_map": global_layer_map,
            "jvp_seed": JVP_SEED,
            "k_list": list(K_LIST),
            "rademacher": {
                "distribution": "independent ±1 per inner parameter element",
                "normalized": False,
                "dtype_rule": "match inner leaf dtype",
            },
            "loss_definitions": {
                "primary": "l_t^TTT = -log_softmax(logits.astype(fp32))[target_t], unmasked",
                "legacy_diagnostic": "official -token_log_probs(logits, target)",
            },
            "estimator_identity_note": (
                "sqrt of the Rademacher second-moment estimate is not assumed unbiased; "
                "squared estimates are stored separately. bf16 JVP/VJP directional gap is "
                "known (STEP 2.5) and is not a pass/fail for this run."
            ),
            "topk": {
                "fractions": list(TOP_FRACTIONS),
                "metric": "topk_recall = |TopExact ∩ TopEstimated| / k, same cardinality",
                "tie_rule": TIE_RULE,
            },
            "relative_error": "|Ghat-G| / (G + eps)",
            "rel_eps": REL_EPS,
            "dtypes": {
                "param_dtype_config": str(cfg.model.param_dtype),
                "compute_dtype_config": str(cfg.model.compute_dtype),
                "state_dtype_config": str(cfg.model.state_dtype),
                "inner_eval_cast": str(model_outer.state_dtype),
                "norm_accumulation_dtype": "float32",
                "token_ttt_ce_dtype": str(token_ttt_ce.dtype),
                "legacy_nll_dtype": str(legacy_nll.dtype),
            },
            "step_index": {
                "set_to_eval_warmup_value": False,
                "reason": "pure f / exact / JVP do not call inner_loop_step",
            },
            "gpu": asdict(gpu),
            "attention": attention_cfg,
            "tokenizer_path": str(TOKENIZER_PATH) if tokenizer is not None else None,
            "evaluation_point_vs_previous": eval_point_checks,
            "previous_result": str(PREV_STEM),
            "exact_g_distribution": exact_dist,
            "metrics_by_k": metrics_by_k,
            "topk_by_k": topk_by_k,
            "ranking_diagnostic": ranking_diag,
            "jvp_regression_vs_previous": regression,
            "exact_overlap_vs_previous_32": exact_overlap,
            "invalid_positions": "exact_grad_norm is NaN; JVP estimates still stored for all T",
            "sanity": sanity,
            "runtime_s": runtime,
            "memory": memory,
        }
        stem.with_suffix(".json").write_text(json.dumps(to_jsonable(meta), indent=2) + "\n")
        print(f"\nraw:  {stem.with_suffix('.npz')}", flush=True)
        print(f"meta: {stem.with_suffix('.json')}", flush=True)
        print(f"wall: {runtime['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(attention_cudnn_override=args.attention_cudnn_override)
