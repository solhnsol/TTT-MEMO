"""CPU tests for binding metrics. No tokenizer, no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import b_swap, binding_score_from_matrix, flip_diagnostics, rank_candidates


def test_b_swap_true_binding() -> None:
    mat_a = {
        "e1": {"v1": 0.0, "v2": -2.0},
        "e2": {"v1": -2.0, "v2": 0.0},
    }
    mat_b = {
        "e1": {"v1": -2.0, "v2": 0.0},
        "e2": {"v1": 0.0, "v2": -2.0},
    }
    out = b_swap(mat_a, mat_b)
    assert abs(out["B_A"] - 2.0) < 1e-9
    assert abs(out["B_B"] - 2.0) < 1e-9
    assert abs(out["B_swap"] - 2.0) < 1e-9
    flips = flip_diagnostics(mat_a, mat_b)
    assert flips["both_flipped"] == 1


def test_b_swap_cancels_global_value_boost() -> None:
    # v1 is always more probable, independent of entity and of mapping.
    mat = {
        "e1": {"v1": 0.0, "v2": -2.0},
        "e2": {"v1": 0.0, "v2": -2.0},
    }
    assert abs(binding_score_from_matrix(mat, "A")) < 1e-9
    assert abs(binding_score_from_matrix(mat, "B")) < 1e-9
    out = b_swap(mat, mat)
    assert abs(out["B_swap"]) < 1e-9
    flips = flip_diagnostics(mat, mat)
    assert flips["e1_flipped"] == 0
    assert flips["both_flipped"] == 0


def test_two_way_rank_uses_mean() -> None:
    rows = [
        {"candidate": "gold", "logp_sum": -2.0, "logp_mean": -0.5, "n_tokens": 4},
        {"candidate": "other", "logp_sum": -1.0, "logp_mean": -1.0, "n_tokens": 1},
    ]
    out = rank_candidates(rows, gold="gold")
    assert out["accuracy"] == 1
    assert out["accuracy_sum_logp"] == 0


if __name__ == "__main__":
    test_b_swap_true_binding()
    test_b_swap_cancels_global_value_boost()
    test_two_way_rank_uses_mean()
    print("test_binding_metrics: PASS")
