"""Official SQuAD v1.1 normalization, Exact Match, and token F1.

Copied from the SQuAD 1.1 evaluation script (`evaluate-v1.1.py`): lowercase,
strip punctuation, strip articles, collapse whitespace, then max over all
acceptable gold answers. This is the scoring implementation for Exp 01; it is
frozen before the main run.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]):
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def squad_em_f1(prediction: str, gold_answers: list[str]) -> tuple[int, float]:
    """Return (EM in {0,1}, F1 in [0,1]) using max over gold answers."""
    em = metric_max_over_ground_truths(exact_match_score, prediction, gold_answers)
    f1 = metric_max_over_ground_truths(f1_score, prediction, gold_answers)
    return int(bool(em)), float(f1)
