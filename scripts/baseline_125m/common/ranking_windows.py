"""STEP 8A helpers: fixed-length token windows and linear pairwise ranking.

Windows live on NTP loss index t (where NLL and Ghat are defined), not raw j.
G_t is attached to the predicted target at raw j = t+1.

No JAX, no checkpoint. Labels are used only to place windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

L_PRIMARY = 32
LOG_G_EPS = 1e-12
MINI_BATCH = 1024
SPLIT_SEED = 20260909
SHUFFLE_SEED = 20260909
BOOTSTRAP_SEED = 20260909
N_BOOTSTRAP = 10_000
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def kv_loss_span(span) -> tuple[int, int]:
    """Inclusive loss-t union of key and value (the information to retain)."""
    t0 = min(int(span["key_loss_t_start"]), int(span["value_loss_t_start"]))
    t1 = max(int(span["key_loss_t_end"]), int(span["value_loss_t_end"]))
    if t1 < t0:
        raise ValueError(f"empty key/value loss span {t0}:{t1}")
    return t0, t1


def sentence_loss_span(span) -> tuple[int, int]:
    return int(span["sentence_loss_t_start"]), int(span["sentence_loss_t_end"])


def span_center_t(span) -> int:
    t0, t1 = kv_loss_span(span)
    return (t0 + t1) // 2


def valid_loss_mask(T: int, T_pad: int) -> np.ndarray:
    """mask[t]=True iff target raw j=t+1 is a real token. Length T_pad (last slot unused)."""
    mask = np.zeros((T_pad,), dtype=bool)
    if T < 2:
        return mask
    mask[: T - 1] = True
    return mask


def query_mask(T_pad: int, query_t0: int, query_t1: int) -> np.ndarray:
    m = np.zeros((T_pad,), dtype=bool)
    m[int(query_t0) : int(query_t1) + 1] = True
    return m


def cover_mask(T_pad: int, spans_inclusive: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros((T_pad,), dtype=bool)
    for a, b in spans_inclusive:
        m[int(a) : int(b) + 1] = True
    return m


def place_centered_window(
    center: int,
    L: int,
    allowed: np.ndarray,
) -> tuple[int, int, bool]:
    """Length-L window, shifted to stay inside allowed tokens.

    Returns (start, end_inclusive, failed_overlap).
    failed_overlap=True means no fully-allowed placement of length L exists.
    """
    n = int(allowed.size)
    if L <= 0 or L > n:
        raise ValueError(f"L={L} n={n}")
    start = int(center) - L // 2
    if start < 0:
        start = 0
    if start + L > n:
        start = n - L

    def ok(s: int) -> bool:
        if s < 0 or s + L > n:
            return False
        return bool(np.all(allowed[s : s + L]))

    if ok(start):
        return start, start + L - 1, False
    for delta in range(1, n):
        if ok(start - delta):
            return start - delta, start - delta + L - 1, False
        if ok(start + delta):
            return start + delta, start + delta + L - 1, False
    return start, start + L - 1, True


def nonoverlapping_background_windows(
    allowed: np.ndarray,
    L: int,
) -> list[tuple[int, int]]:
    """Greedy left-to-right length-L windows on contiguous allowed runs."""
    n = int(allowed.size)
    out: list[tuple[int, int]] = []
    t = 0
    while t <= n - L:
        if bool(np.all(allowed[t : t + L])):
            out.append((t, t + L - 1))
            t += L
        else:
            t += 1
    return out


def window_contains_span(w0: int, w1: int, s0: int, s1: int) -> bool:
    return int(w0) <= int(s0) and int(s1) <= int(w1)


def split_sample_ids(n: int, *, seed: int, salt: int) -> dict[str, list[int]]:
    """60/20/20 over sample_id 0..n-1. Same sample never crosses splits."""
    if n < 3:
        raise ValueError(f"need at least 3 samples, got {n}")
    rng = np.random.default_rng(seed + salt)
    perm = rng.permutation(n)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"empty split for n={n}")
    train = sorted(int(x) for x in perm[:n_train])
    val = sorted(int(x) for x in perm[n_train : n_train + n_val])
    test = sorted(int(x) for x in perm[n_train + n_val :])
    if sorted(train + val + test) != list(range(n)):
        raise ValueError("split is not a partition of 0..n-1")
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("split leak")
    return {"train": train, "val": val, "test": test}


def mean_finite(values: np.ndarray, w0: int, w1: int) -> float:
    sl = values[int(w0) : int(w1) + 1]
    sl = sl[np.isfinite(sl)]
    if sl.size == 0:
        return float("nan")
    return float(np.mean(sl))


def max_finite(values: np.ndarray, w0: int, w1: int) -> float:
    sl = values[int(w0) : int(w1) + 1]
    sl = sl[np.isfinite(sl)]
    if sl.size == 0:
        return float("nan")
    return float(np.max(sl))


def window_features_from_tokens(
    *,
    nll: np.ndarray,
    ghat: np.ndarray,
    w0: int,
    w1: int,
    eps: float = LOG_G_EPS,
) -> dict[str, float]:
    g = ghat[int(w0) : int(w1) + 1].astype(np.float64)
    logg = np.log(np.maximum(g, 0.0) + eps)
    nll_w = nll[int(w0) : int(w1) + 1].astype(np.float64)
    return {
        "mean_nll": mean_finite(nll, w0, w1),
        "mean_ghat": mean_finite(ghat, w0, w1),
        "mean_log_ghat": float(np.mean(logg[np.isfinite(logg)])) if np.any(np.isfinite(logg)) else float("nan"),
        "max_ghat": max_finite(ghat, w0, w1),
        "n_finite_nll": int(np.sum(np.isfinite(nll_w))),
        "n_finite_ghat": int(np.sum(np.isfinite(g))),
    }


def ranking_order(scores: np.ndarray, tie_index: np.ndarray) -> np.ndarray:
    """Descending score; ties -> smaller tie_index."""
    scores = np.asarray(scores, dtype=np.float64)
    idx = np.asarray(tie_index)
    return np.lexsort((idx, -scores))


def rank_of_positive(scores: np.ndarray, is_positive: np.ndarray, tie_index: np.ndarray) -> int:
    """1-based rank of the unique positive. Ties broken by smaller window start."""
    order = ranking_order(scores, tie_index)
    pos = np.flatnonzero(np.asarray(is_positive).astype(bool))
    if pos.size != 1:
        raise ValueError(f"expected exactly 1 positive, got {pos.size}")
    target = int(pos[0])
    rank = int(np.flatnonzero(order == target)[0]) + 1
    return rank


def ranking_metrics(rank: int, n: int) -> dict[str, float]:
    if n < 2:
        raise ValueError(f"need n>=2 candidates, got {n}")
    k1pct = max(1, int(np.ceil(0.01 * n)))
    return {
        "n_candidates": int(n),
        "rank": int(rank),
        "mrr": 1.0 / float(rank),
        "recall_at_1": float(rank <= 1),
        "recall_at_5": float(rank <= 5),
        "recall_at_1pct": float(rank <= k1pct),
        "k_1pct": int(k1pct),
        "normalized_rank": (float(rank) - 1.0) / float(n - 1),
        "percentile_rank": (float(n - rank) / float(n - 1)),
    }


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu.astype(np.float64), sd.astype(np.float64)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=np.float64) - mu) / sd


def pairwise_diffs(X: np.ndarray, y: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    """Within-sample (x_pos - x_neg) rows. One positive per sample."""
    diffs = []
    for sid in np.unique(sample_ids):
        m = sample_ids == sid
        Xp = X[m]
        yp = y[m]
        pos = np.flatnonzero(yp == 1)
        neg = np.flatnonzero(yp == 0)
        if pos.size != 1 or neg.size < 1:
            raise ValueError(f"sample {sid}: n_pos={pos.size} n_neg={neg.size}")
        diffs.append(Xp[pos[0]] - Xp[neg])
    return np.concatenate(diffs, axis=0)


def pairwise_logistic_loss(w: np.ndarray, diffs: np.ndarray, lam: float) -> float:
    s = diffs @ w
    data = float(np.mean(np.logaddexp(0.0, -s)))
    return data + 0.5 * float(lam) * float(np.dot(w, w))


def pairwise_logistic_grad(w: np.ndarray, diffs: np.ndarray, lam: float) -> np.ndarray:
    s = diffs @ w
    # d/ds log(1+e^{-s}) = -sigmoid(-s) = -(1 / (1 + e^{s}))
    weight = -1.0 / (1.0 + np.exp(np.clip(s, -40.0, 40.0)))
    data = diffs.T @ (weight / diffs.shape[0])
    return data + float(lam) * w


def fit_linear_ranker(diffs: np.ndarray, lam: float) -> np.ndarray:
    """s = w^T x. Intercept omitted: it cancels in within-sample score differences."""
    from scipy.optimize import minimize

    d = int(diffs.shape[1])
    w0 = np.zeros((d,), dtype=np.float64)
    res = minimize(
        lambda w: pairwise_logistic_loss(w, diffs, lam),
        w0,
        jac=lambda w: pairwise_logistic_grad(w, diffs, lam),
        method="L-BFGS-B",
    )
    if not res.success:
        raise RuntimeError(f"ranker fit failed: {res.message}")
    return np.asarray(res.x, dtype=np.float64)


def score_linear(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64)


def select_lambda(
    diffs_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sid_val: np.ndarray,
    starts_val: np.ndarray,
    grid: tuple[float, ...] = LAMBDA_GRID,
) -> tuple[float, np.ndarray, dict]:
    """Pick λ by validation MRR. Returns (lambda, w, val metrics at that λ)."""
    best = None
    rows = []
    for lam in grid:
        w = fit_linear_ranker(diffs_train, lam)
        m = mean_metrics_for_scorer(X_val, y_val, sid_val, starts_val, w)
        rows.append({"lambda": float(lam), **m})
        key = (m["mrr"], -m["normalized_rank"])
        if best is None or key > best[0]:
            best = (key, float(lam), w, m)
    assert best is not None
    return best[1], best[2], {"grid": rows, "chosen": {"lambda": best[1], **best[3]}}


def mean_metrics_for_scorer(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    starts: np.ndarray,
    w: np.ndarray,
) -> dict[str, float]:
    scores = score_linear(X, w)
    return mean_metrics_from_scores(scores, y, sample_ids, starts)


def mean_metrics_from_scores(
    scores: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    starts: np.ndarray,
) -> dict[str, float]:
    acc: dict[str, list[float]] = {
        "mrr": [],
        "recall_at_1": [],
        "recall_at_5": [],
        "recall_at_1pct": [],
        "normalized_rank": [],
        "percentile_rank": [],
        "rank": [],
        "n_candidates": [],
    }
    for sid in np.unique(sample_ids):
        m = sample_ids == sid
        mets = ranking_metrics(
            rank_of_positive(scores[m], y[m], starts[m]),
            int(np.sum(m)),
        )
        for k in acc:
            acc[k].append(mets[k] if k != "n_candidates" else float(mets["n_candidates"]))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["n_samples"] = float(len(np.unique(sample_ids)))
    return out


def per_sample_rank_table(
    scores: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    starts: np.ndarray,
) -> list[dict]:
    rows = []
    for sid in np.unique(sample_ids):
        m = sample_ids == sid
        n = int(np.sum(m))
        rank = rank_of_positive(scores[m], y[m], starts[m])
        mets = ranking_metrics(rank, n)
        rows.append({"sample_id": int(sid), **mets})
    return rows


def bootstrap_delta(
    rows_a: list[dict],
    rows_b: list[dict],
    keys: tuple[str, ...],
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Sample-level bootstrap of mean(metric_b) - mean(metric_a). Unit = sample."""
    ids_a = [r["sample_id"] for r in rows_a]
    ids_b = [r["sample_id"] for r in rows_b]
    if ids_a != ids_b:
        raise ValueError("bootstrap rows must share sample order")
    n = len(rows_a)
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for key in keys:
        a = np.asarray([r[key] for r in rows_a], dtype=np.float64)
        b = np.asarray([r[key] for r in rows_b], dtype=np.float64)
        delta = b - a
        point = float(np.mean(delta))
        boots = np.empty((n_boot,), dtype=np.float64)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots[i] = float(np.mean(delta[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[key] = {
            "point": point,
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
            "n_samples": n,
            "n_bootstrap": n_boot,
            "unit": "sample",
        }
    return out


def shuffle_logg_within_sample(
    logg: np.ndarray,
    sample_ids: np.ndarray,
    *,
    seed: int,
    salt: int,
) -> np.ndarray:
    """Permute log(G) among windows of the same sample. NLL / labels unchanged."""
    out = np.asarray(logg, dtype=np.float64).copy()
    rng = np.random.default_rng(seed + salt)
    for sid in np.unique(sample_ids):
        m = np.flatnonzero(sample_ids == sid)
        perm = rng.permutation(m.size)
        out[m] = out[m][perm]
    return out


@dataclass
class WindowRow:
    task: str
    sample_id: int
    window_kind: str  # positive | background | relevant | distractor
    is_positive: int
    start_t: int
    end_t: int
    center_t: int
    span_index: int
    span_role: str
    kv_t0: int
    kv_t1: int
    kv_fully_inside: int
    query_or_pad_overlap: int
    placement_failed: int
    chunk_start: int
    chunk_end: int
    n_tokens: int


def _window_row(
    *,
    task: str,
    sample_id: int,
    window_kind: str,
    is_positive: int,
    start_t: int,
    end_t: int,
    center_t: int,
    span_index: int,
    span_role: str,
    kv_t0: int,
    kv_t1: int,
    query_m: np.ndarray,
    valid_m: np.ndarray,
    L: int,
    placement_failed: bool,
) -> WindowRow:
    sl = slice(start_t, end_t + 1)
    qpad = bool(np.any(query_m[sl]) or np.any(~valid_m[sl]))
    return WindowRow(
        task=task,
        sample_id=int(sample_id),
        window_kind=window_kind,
        is_positive=int(is_positive),
        start_t=int(start_t),
        end_t=int(end_t),
        center_t=int(center_t),
        span_index=int(span_index),
        span_role=str(span_role),
        kv_t0=int(kv_t0),
        kv_t1=int(kv_t1),
        kv_fully_inside=int(window_contains_span(start_t, end_t, kv_t0, kv_t1)),
        query_or_pad_overlap=int(qpad),
        placement_failed=int(placement_failed),
        chunk_start=int(start_t) // MINI_BATCH,
        chunk_end=int(end_t) // MINI_BATCH,
        n_tokens=int(end_t - start_t + 1),
    )


def build_single2_windows(
    *,
    task: str,
    sample_id: int,
    T: int,
    T_pad: int,
    query_t0: int,
    query_t1: int,
    spans: list,
    L: int = L_PRIMARY,
) -> list[WindowRow]:
    """1 positive needle window + non-overlapping haystack background windows."""
    relevant = [s for s in spans if str(s["role"]) == "relevant"]
    if len(relevant) != 1:
        raise ValueError(f"{task} sid={sample_id}: expected 1 relevant span, got {len(relevant)}")
    valid = valid_loss_mask(T, T_pad)
    query = query_mask(T_pad, query_t0, query_t1)
    needle_sents = [sentence_loss_span(s) for s in spans]
    needle_m = cover_mask(T_pad, needle_sents)
    allowed_pos = valid & ~query
    rel = relevant[0]
    kv0, kv1 = kv_loss_span(rel)
    center = span_center_t(rel)
    w0, w1, failed = place_centered_window(center, L, allowed_pos)
    pos = _window_row(
        task=task,
        sample_id=sample_id,
        window_kind="positive",
        is_positive=1,
        start_t=w0,
        end_t=w1,
        center_t=center,
        span_index=int(rel["span_index"]),
        span_role="relevant",
        kv_t0=kv0,
        kv_t1=kv1,
        query_m=query,
        valid_m=valid,
        L=L,
        placement_failed=failed,
    )
    pos_m = np.zeros((T_pad,), dtype=bool)
    pos_m[w0 : w1 + 1] = True
    allowed_bg = valid & ~query & ~needle_m & ~pos_m
    rows = [pos]
    for b0, b1 in nonoverlapping_background_windows(allowed_bg, L):
        rows.append(
            _window_row(
                task=task,
                sample_id=sample_id,
                window_kind="background",
                is_positive=0,
                start_t=b0,
                end_t=b1,
                center_t=(b0 + b1) // 2,
                span_index=-1,
                span_role="background",
                kv_t0=-1,
                kv_t1=-1,
                query_m=query,
                valid_m=valid,
                L=L,
                placement_failed=False,
            )
        )
    return rows


def build_mk2_windows(
    *,
    task: str,
    sample_id: int,
    T: int,
    T_pad: int,
    query_t0: int,
    query_t1: int,
    spans: list,
    L: int = L_PRIMARY,
) -> list[WindowRow]:
    """Relevant needle window vs all distractor needle windows. No haystack."""
    relevant = [s for s in spans if str(s["role"]) == "relevant"]
    distractors = [s for s in spans if str(s["role"]) == "distractor"]
    if len(relevant) != 1:
        raise ValueError(f"{task} sid={sample_id}: expected 1 relevant, got {len(relevant)}")
    if not distractors:
        raise ValueError(f"{task} sid={sample_id}: no distractors")
    valid = valid_loss_mask(T, T_pad)
    query = query_mask(T_pad, query_t0, query_t1)
    allowed = valid & ~query
    rows: list[WindowRow] = []
    for sp in relevant + distractors:
        kv0, kv1 = kv_loss_span(sp)
        center = span_center_t(sp)
        w0, w1, failed = place_centered_window(center, L, allowed)
        role = str(sp["role"])
        rows.append(
            _window_row(
                task=task,
                sample_id=sample_id,
                window_kind="relevant" if role == "relevant" else "distractor",
                is_positive=int(role == "relevant"),
                start_t=w0,
                end_t=w1,
                center_t=center,
                span_index=int(sp["span_index"]),
                span_role=role,
                kv_t0=kv0,
                kv_t1=kv1,
                query_m=query,
                valid_m=valid,
                L=L,
                placement_failed=failed,
            )
        )
    n_pos = sum(r.is_positive for r in rows)
    if n_pos != 1:
        raise ValueError(f"{task} sid={sample_id}: n_positive={n_pos}")
    return rows
