"""CPU tests for STEP 8A window placement and ranking helpers. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINE_ROOT / "common"))

from ranking_windows import (  # noqa: E402
    L_PRIMARY,
    bootstrap_delta,
    build_mk2_windows,
    build_single2_windows,
    fit_linear_ranker,
    pairwise_diffs,
    place_centered_window,
    rank_of_positive,
    ranking_metrics,
    shuffle_logg_within_sample,
    split_sample_ids,
    valid_loss_mask,
)


def assert_eq(got, expected, msg: str) -> None:
    if got != expected:
        raise SystemExit(f"FAIL {msg}: got {got!r}, expected {expected!r}")


def assert_true(cond, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL {msg}")


def test_place_centered_and_boundary() -> None:
    n = 100
    allowed = np.ones((n,), dtype=bool)
    s, e, fail = place_centered_window(50, 32, allowed)
    assert_eq((e - s + 1, fail), (32, False), "centered length")
    assert_eq(s, 50 - 16, "center-16 start")
    s0, e0, f0 = place_centered_window(2, 32, allowed)
    assert_eq((s0, e0, f0), (0, 31, False), "left boundary shift")
    s1, e1, f1 = place_centered_window(98, 32, allowed)
    assert_eq((s1, e1, f1), (68, 99, False), "right boundary shift")


def test_shift_avoids_query() -> None:
    n = 80
    allowed = np.ones((n,), dtype=bool)
    allowed[60:] = False
    s, e, fail = place_centered_window(55, 32, allowed)
    assert_eq(fail, False, "should find a left shift")
    assert_true(e < 60, "no query tokens")
    assert_eq(e - s + 1, 32, "length preserved")


def test_valid_mask_indexing() -> None:
    m = valid_loss_mask(T=8040, T_pad=8192)
    assert_eq(int(m.sum()), 8039, "T-1 valid losses")
    assert_eq(bool(m[8038]), True, "last real target")
    assert_eq(bool(m[8039]), False, "first pad target")


def test_single2_and_mk2_toy() -> None:
    T, T_pad = 200, 1024
    spans_s2 = [
        {
            "role": "relevant",
            "span_index": 0,
            "key_loss_t_start": 80,
            "key_loss_t_end": 85,
            "value_loss_t_start": 90,
            "value_loss_t_end": 95,
            "sentence_loss_t_start": 70,
            "sentence_loss_t_end": 96,
        }
    ]
    rows = build_single2_windows(
        task="niah_single_2",
        sample_id=0,
        T=T,
        T_pad=T_pad,
        query_t0=150,
        query_t1=170,
        spans=spans_s2,
        L=32,
    )
    pos = [r for r in rows if r.is_positive]
    bg = [r for r in rows if not r.is_positive]
    assert_eq(len(pos), 1, "one positive")
    assert_true(len(bg) >= 1, "some background")
    assert_eq(pos[0].kv_fully_inside, 1, "kv inside L=32")
    assert_eq(pos[0].query_or_pad_overlap, 0, "pos no query")
    assert_eq(pos[0].n_tokens, 32, "L")
    starts = sorted(r.start_t for r in bg)
    for a, b in zip(starts, starts[1:]):
        assert_true(b >= a + 32, "non-overlapping bg")
    for r in bg:
        assert_true(r.end_t < 70 or r.start_t > 96, "bg misses needle sentence")
        assert_true(r.end_t < 150 or r.start_t > 170, "bg misses query")

    spans_mk = spans_s2 + [
        {
            "role": "distractor",
            "span_index": 0,
            "key_loss_t_start": 20,
            "key_loss_t_end": 22,
            "value_loss_t_start": 24,
            "value_loss_t_end": 26,
            "sentence_loss_t_start": 18,
            "sentence_loss_t_end": 27,
        }
    ]
    mk = build_mk2_windows(
        task="niah_multikey_2",
        sample_id=1,
        T=T,
        T_pad=T_pad,
        query_t0=150,
        query_t1=170,
        spans=spans_mk,
        L=32,
    )
    assert_eq(sum(r.is_positive for r in mk), 1, "one relevant")
    assert_eq(len(mk), 2, "relevant + 1 distractor, no haystack")


def test_ranking_and_split() -> None:
    scores = np.array([0.1, 0.9, 0.5])
    y = np.array([0, 1, 0])
    starts = np.array([10, 20, 30])
    rank = rank_of_positive(scores, y, starts)
    assert_eq(rank, 1, "highest score is rank 1")
    m = ranking_metrics(1, 100)
    assert_eq(m["mrr"], 1.0, "MRR@rank1")
    assert_eq(m["recall_at_1pct"], 1.0, "k_1pct=1 for n=100")
    splits = split_sample_ids(200, seed=20260909, salt=0)
    n = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert_eq(n, 200, "partition")
    assert_eq(len(splits["train"]), 120, "60%")
    assert_eq(len(splits["val"]), 40, "20%")
    assert_eq(len(splits["test"]), 40, "20%")


def test_pairwise_fit_and_shuffle() -> None:
    rng = np.random.default_rng(0)
    # 8 samples, 5 windows: positive has larger feature
    X, y, sid, starts = [], [], [], []
    for s in range(8):
        for i in range(5):
            pos = i == 0
            X.append([10.0 if pos else 0.0])
            y.append(1 if pos else 0)
            sid.append(s)
            starts.append(i)
    X = np.asarray(X)
    y = np.asarray(y)
    sid = np.asarray(sid)
    diffs = pairwise_diffs(X, y, sid)
    w = fit_linear_ranker(diffs, 1e-3)
    assert_true(w[0] > 0, "learns positive weight on the informative feature")
    logg = rng.normal(size=y.size)
    sh = shuffle_logg_within_sample(logg, sid, seed=1, salt=0)
    assert_eq(sorted(sh[sid == 0].tolist()), sorted(logg[sid == 0].tolist()), "within-sample permute")
    rows_a = [{"sample_id": i, "mrr": 0.5} for i in range(10)]
    rows_b = [{"sample_id": i, "mrr": 0.6} for i in range(10)]
    boot = bootstrap_delta(rows_a, rows_b, ("mrr",), n_boot=200, seed=0)
    assert_true(abs(boot["mrr"]["point"] - 0.1) < 1e-12, "delta MRR")


def main() -> None:
    test_place_centered_and_boundary()
    test_shift_avoids_query()
    test_valid_mask_indexing()
    test_single2_and_mk2_toy()
    test_ranking_and_split()
    test_pairwise_fit_and_shuffle()
    print(f"OK  L_PRIMARY={L_PRIMARY}  ranking window tests passed")


if __name__ == "__main__":
    main()
