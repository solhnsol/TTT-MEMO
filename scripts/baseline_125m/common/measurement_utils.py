"""Shared measurement helpers for 125M TTT-E2E scripts.

Pure functions only: no restore, no frozen eval-point setup, no official loss
semantics changes. Extracted so sanity diagnostics can reuse the same l_t^TTT
and pytree ops as the estimator experiment without importing that experiment.
"""

from __future__ import annotations


def token_ttt_ce_from_logits(logits, target_tokens):
    """Unmasked per-token NLL from official CE's fp32 log-softmax.

    Matches e2e/ttt/model/loss.py `cross_entropy_loss_and_accuracy` lines that
    cast logits to fp32, take log_softmax, and gather the target log-prob.
    Does *not* apply the helper's `where(valid, log_prob, 0)` or the
    sum/valid_length reduction. Those are applied separately.

    `loss_pure_ce` is not reused: it is a scalar, equal to `loss`, not a vector.

    Args:
        logits: [T, V], typically bf16 compute dtype from suffix_call
        target_tokens: [T] int

    Returns:
        l_t^TTT, shape [T], dtype float32
    """
    import jax
    import jax.numpy as jnp

    logits_f = logits.astype(jnp.float32)
    log_prob = jax.nn.log_softmax(logits_f, axis=-1)
    token_log_prob = jnp.take_along_axis(log_prob, jnp.expand_dims(target_tokens, -1), axis=-1)
    token_log_prob = jnp.squeeze(token_log_prob, -1)
    return -token_log_prob


def official_masked_reduction(token_ce, mask):
    """L = sum(mask * l) / sum(mask), same 1e-10 floor as the official helper."""
    import jax.numpy as jnp

    valid = mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(valid), jnp.asarray(1e-10, dtype=jnp.float32))
    return jnp.sum(valid * token_ce.astype(jnp.float32)) / denom


def inexact_leaves(tree):
    import equinox as eqx
    import jax

    return [leaf for leaf in jax.tree.leaves(tree) if eqx.is_inexact_array(leaf)]


def pytree_l2_norm(tree):
    """||tree||_2 over inexact leaves. Squares and sum are float32."""
    import jax.numpy as jnp

    acc = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in inexact_leaves(tree):
        x = leaf.astype(jnp.float32)
        acc = acc + jnp.sum(x * x)
    return jnp.sqrt(acc)


def pytree_vdot(a, b):
    """<a, b> over inexact leaves, float32 accumulation."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    acc = jnp.asarray(0.0, dtype=jnp.float32)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        if eqx.is_inexact_array(x):
            acc = acc + jnp.vdot(x.astype(jnp.float32).ravel(), y.astype(jnp.float32).ravel())
    return acc


def pytree_max_abs_diff(a, b) -> float:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    max_diff = jnp.asarray(0.0, dtype=jnp.float32)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        if eqx.is_inexact_array(x):
            max_diff = jnp.maximum(max_diff, jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32))))
        elif hasattr(x, "dtype"):
            max_diff = jnp.maximum(max_diff, jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32))))
    return float(max_diff)


def rademacher_like(tree, key):
    """Independent ±1 tangent, same pytree / shapes / dtypes as `tree`. No scaling."""
    import equinox as eqx
    import jax
    import jax.random as jrandom

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jrandom.split(key, max(len(leaves), 1))
    out = []
    for k, leaf in zip(keys, leaves):
        if eqx.is_inexact_array(leaf):
            out.append(jrandom.rademacher(k, shape=leaf.shape, dtype=leaf.dtype))
        else:
            out.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, out)


def select_even_valid_indices(mask, n: int) -> tuple[list[int], dict]:
    """Split valid positions into n segments; take the midpoint of each.

    Selection is over the valid-index list, not raw [0, T), so invalid (BOS-target)
    slots are never chosen. Deterministic; no RNG.
    """
    import numpy as np

    valid = np.flatnonzero(np.asarray(mask).astype(bool))
    n_valid = int(valid.size)
    n_take = min(int(n), n_valid)
    if n_take <= 0:
        raise SystemExit("no valid tokens under loss_masks")
    starts = (np.arange(n_take) * n_valid) // n_take
    ends = ((np.arange(n_take) + 1) * n_valid) // n_take
    if np.any(ends <= starts):
        raise SystemExit("empty valid-index segment; refuse a biased fallback")
    mids = starts + (ends - starts - 1) // 2
    selected = [int(valid[i]) for i in mids]
    meta = {
        "rule": "even_segments_midpoint_of_valid_positions",
        "n_requested": int(n),
        "n_valid": n_valid,
        "n_selected": n_take,
        "segment_start_in_valid_list": starts.tolist(),
        "segment_end_in_valid_list": ends.tolist(),
        "midpoint_in_valid_list": mids.tolist(),
        "selected_indices": selected,
        "rng_used": False,
    }
    return selected, meta


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


TIE_RULE = "descending score, ties broken by smaller position index (np.lexsort)"


def ranking_order(scores):
    """Indices that sort scores descending; ties -> smaller index."""
    import numpy as np

    scores = np.asarray(scores)
    n = scores.size
    return np.lexsort((np.arange(n), -scores.astype(np.float64)))


def topk_sets(exact, estimated, valid, frac: float):
    """Same-cardinality top-k index sets over valid positions."""
    import numpy as np

    idx = np.flatnonzero(np.asarray(valid).astype(bool))
    e = np.asarray(exact, dtype=np.float64)[idx]
    h = np.asarray(estimated, dtype=np.float64)[idx]
    n = int(idx.size)
    k = max(1, int(round(n * frac)))
    order_e = idx[ranking_order(e)]
    order_h = idx[ranking_order(h)]
    set_e = order_e[:k]
    set_h = order_h[:k]
    return {
        "fraction": frac,
        "k": k,
        "n_valid": n,
        "set_exact": set_e,
        "set_estimated": set_h,
        "tie_rule": TIE_RULE,
    }


def topk_overlap(exact, estimated, valid, frac: float) -> dict:
    """|TopExact ∩ TopEstimated| / k. Same cardinality => precision = recall."""
    info = topk_sets(exact, estimated, valid, frac)
    set_e = set(info["set_exact"].tolist())
    set_h = set(info["set_estimated"].tolist())
    inter = len(set_e & set_h)
    k = info["k"]
    return {
        "fraction": frac,
        "k": k,
        "n_valid": info["n_valid"],
        "overlap": inter,
        "topk_overlap": inter / k,
        "tie_rule": TIE_RULE,
    }


def score_capture(exact, estimated, valid, frac: float, *, eps: float = 1e-12) -> dict:
    """Fraction of exact top-k G_t mass recovered by the estimated top-k set.

    Numerator uses exact G_t on Shat, not estimated scores.
    """
    import numpy as np

    info = topk_sets(exact, estimated, valid, frac)
    g = np.asarray(exact, dtype=np.float64)
    num = float(np.sum(g[info["set_estimated"]]))
    den = float(np.sum(g[info["set_exact"]]))
    capture = num / (den + eps)
    return {
        "fraction": frac,
        "k": info["k"],
        "n_valid": info["n_valid"],
        "score_capture": capture,
        "numerator_exact_mass_on_estimated_set": num,
        "denominator_exact_mass_on_exact_set": den,
        "tie_rule": TIE_RULE,
    }


def cutoff_margin(exact, valid, frac: float, *, eps: float = 1e-12) -> dict:
    """Exact ranking cutoff sharpness: G_(k) - G_(k+1) over valid tokens."""
    import numpy as np

    idx = np.flatnonzero(np.asarray(valid).astype(bool))
    e = np.asarray(exact, dtype=np.float64)[idx]
    n = int(idx.size)
    k = max(1, int(round(n * frac)))
    order = ranking_order(e)
    g_sorted = e[order]
    if k >= n:
        abs_m = None
        rel_m = None
        g_next = None
    else:
        abs_m = float(g_sorted[k - 1] - g_sorted[k])
        rel_m = abs_m / (float(g_sorted[k - 1]) + eps)
        g_next = float(g_sorted[k])
    return {
        "fraction": frac,
        "k": k,
        "n_valid": n,
        "g_at_k": float(g_sorted[k - 1]) if n else None,
        "g_at_k_plus_1": g_next,
        "absolute_margin": abs_m,
        "relative_margin": rel_m,
        "tie_rule": TIE_RULE,
    }


def error_metrics(exact, estimated, valid, *, eps: float) -> dict:
    """Spearman/Pearson plus relative and absolute error on valid tokens."""
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


def summarize_values(values) -> dict:
    """Distribution summary over a 1-D list of measurements."""
    import numpy as np

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=0)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
