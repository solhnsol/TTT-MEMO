"""Evaluation metrics for Exp 02A-1.

Copied ranking/EM/F1/bootstrap helpers from Exp 02A. Binding-specificity
scores (B_A, B_B, B_swap, entity flips) are added below. Primary candidate
score remains mean per-token log-likelihood; sum-logp is diagnostic only.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(ground_truth).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2.0 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def paired_bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap CI for mean(a) - mean(b) with paired resampling."""
    import numpy as np

    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired bootstrap requires nonempty equal-length arrays")
    n = a.shape[0]
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        deltas[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(deltas, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "mean_delta": float(a.mean() - b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "seed": seed,
        "n": n,
    }


def generation_scores(prediction: str, gold: str) -> dict[str, Any]:
    return {
        "em": int(exact_match_score(prediction, gold)),
        "f1": float(f1_score(prediction, gold)),
        "normalized_prediction": normalize_answer(prediction),
        "normalized_gold": normalize_answer(gold),
    }


def rank_candidates(rows: list[dict[str, Any]], *, gold: str) -> dict[str, Any]:
    """Primary rank uses mean token log-likelihood. Sum logp is diagnostic.

    score(a) = sum_j log p(a_j | q, a_<j) / n_answer_tokens
    Higher is better. Ties break by candidate string for determinism.
    """
    if not rows:
        raise ValueError("empty candidate rows")
    gold_rows = [r for r in rows if r["candidate"] == gold]
    if len(gold_rows) != 1:
        raise ValueError(f"expected one gold row, got {len(gold_rows)}")
    gold_row = gold_rows[0]
    decoys = [r for r in rows if r["candidate"] != gold]
    best_decoy_mean = max(decoys, key=lambda r: (r["logp_mean"], r["candidate"]))
    best_decoy_sum = max(decoys, key=lambda r: (r["logp_sum"], r["candidate"]))
    ordered_mean = sorted(rows, key=lambda r: (-r["logp_mean"], r["candidate"]))
    ordered_sum = sorted(rows, key=lambda r: (-r["logp_sum"], r["candidate"]))
    rank_mean = next(i for i, r in enumerate(ordered_mean, start=1) if r["candidate"] == gold)
    rank_sum = next(i for i, r in enumerate(ordered_sum, start=1) if r["candidate"] == gold)
    return {
        "primary_score": "mean_token_log_likelihood",
        "gold_rank": rank_mean,
        "accuracy": int(rank_mean == 1),
        "margin_logp_mean": gold_row["logp_mean"] - best_decoy_mean["logp_mean"],
        "best_decoy": best_decoy_mean["candidate"],
        "best_decoy_logp_mean": best_decoy_mean["logp_mean"],
        "gold_logp_mean": gold_row["logp_mean"],
        "gold_nll_mean": -gold_row["logp_mean"],
        "gold_n_tokens": gold_row["n_tokens"],
        "gold_rank_sum_logp": rank_sum,
        "accuracy_sum_logp": int(rank_sum == 1),
        "gold_logp_sum": gold_row["logp_sum"],
        "best_decoy_sum_logp": best_decoy_sum["candidate"],
        "best_decoy_logp_sum": best_decoy_sum["logp_sum"],
        "margin_logp_sum": gold_row["logp_sum"] - best_decoy_sum["logp_sum"],
        "gold_rank_mean_logp": rank_mean,
    }


def binding_score_from_matrix(matrix: dict[str, dict[str, float]], mapping: str) -> float:
    """B_A or B_B from a 2x2 mean-logp matrix.

    matrix[entity_key][value_key] = s(e, v) = mean_j log p(v_j | query(e), v_<j)

    Mapping A gold pairing is (e1,v1) and (e2,v2).
    Mapping B gold pairing is (e1,v2) and (e2,v1).

    B_A = [ s(e1,v1)+s(e2,v2) - s(e1,v2)-s(e2,v1) ] / 2
    B_B = [ s(e1,v2)+s(e2,v1) - s(e1,v1)-s(e2,v2) ] / 2

    A global boost of one value on both queries cancels. A positive score
    means the preference follows the entity-value mapping.
    """
    s11 = matrix["e1"]["v1"]
    s12 = matrix["e1"]["v2"]
    s21 = matrix["e2"]["v1"]
    s22 = matrix["e2"]["v2"]
    if mapping == "A":
        return 0.5 * (s11 + s22 - s12 - s21)
    if mapping == "B":
        return 0.5 * (s12 + s21 - s11 - s22)
    raise ValueError(f"mapping must be A or B, got {mapping!r}")


def entity_prefers(matrix: dict[str, dict[str, float]], entity: str, value: str) -> bool:
    """Strict mean-logp preference of `value` over the other value for `entity`."""
    other = "v2" if value == "v1" else "v1"
    return matrix[entity][value] > matrix[entity][other]


def flip_diagnostics(matrix_a: dict[str, dict[str, float]], matrix_b: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Whether each entity's preferred value flips with the swapped mapping.

    e1 flips correctly if A prefers v1 over v2 AND B prefers v2 over v1.
    e2 flips correctly if A prefers v2 over v1 AND B prefers v1 over v2.
    """
    e1_a = entity_prefers(matrix_a, "e1", "v1")
    e1_b = entity_prefers(matrix_b, "e1", "v2")
    e2_a = entity_prefers(matrix_a, "e2", "v2")
    e2_b = entity_prefers(matrix_b, "e2", "v1")
    e1_flip = bool(e1_a and e1_b)
    e2_flip = bool(e2_a and e2_b)
    return {
        "e1_prefers_gold_A": int(e1_a),
        "e1_prefers_gold_B": int(e1_b),
        "e2_prefers_gold_A": int(e2_a),
        "e2_prefers_gold_B": int(e2_b),
        "e1_flipped": int(e1_flip),
        "e2_flipped": int(e2_flip),
        "both_flipped": int(e1_flip and e2_flip),
    }


def b_swap(matrix_a: dict[str, dict[str, float]], matrix_b: dict[str, dict[str, float]]) -> dict[str, float]:
    b_a = binding_score_from_matrix(matrix_a, "A")
    b_b = binding_score_from_matrix(matrix_b, "B")
    return {
        "B_A": float(b_a),
        "B_B": float(b_b),
        "B_swap": float(0.5 * (b_a + b_b)),
    }
