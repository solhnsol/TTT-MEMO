"""CPU tests for metrics and prompt string construction. Tokenizer tests wait for HF."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_v1 import build_dataset_v1
from metrics import exact_match_score, f1_score, generation_scores, rank_candidates
from prompts import recall_user_content, write_user_content


def test_write_contains_fact_once_and_not_question() -> None:
    for ex in build_dataset_v1()[:4]:
        text = write_user_content(ex["fact_sentence"])
        assert text.startswith("Fact:\n")
        assert text.count(ex["fact_sentence"]) == 1
        assert text.count(ex["write_payload"]) == 1
        assert ex["question"] not in text
        assert "?" not in text


def test_source_free_recall_omits_fact() -> None:
    ex = build_dataset_v1()[0]
    text = recall_user_content(ex["question"], fact_sentence=None)
    assert ex["fact_sentence"] not in text
    assert ex["write_payload"] not in text
    assert ex["question"] in text
    assert "Reply with only the answer." in text


def test_source_present_recall_contains_fact() -> None:
    ex = build_dataset_v1()[0]
    text = recall_user_content(ex["question"], fact_sentence=ex["fact_sentence"])
    assert ex["fact_sentence"] in text
    assert ex["question"] in text


def test_rank_candidates_margin() -> None:
    rows = [
        {"candidate": "gold", "logp_sum": -1.0, "logp_mean": -0.5, "n_tokens": 2},
        {"candidate": "d1", "logp_sum": -2.0, "logp_mean": -1.0, "n_tokens": 2},
        {"candidate": "d2", "logp_sum": -1.5, "logp_mean": -0.75, "n_tokens": 2},
    ]
    out = rank_candidates(rows, gold="gold")
    assert out["primary_score"] == "mean_token_log_likelihood"
    assert out["accuracy"] == 1
    assert out["gold_rank"] == 1
    assert abs(out["margin_logp_mean"] - 0.25) < 1e-9
    assert abs(out["margin_logp_sum"] - 0.5) < 1e-9
    assert out["best_decoy"] == "d2"


def test_primary_is_mean_not_sum() -> None:
    # Decoy wins on sum (one short token), gold wins on mean (two tokens).
    rows = [
        {"candidate": "gold", "logp_sum": -2.0, "logp_mean": -1.0, "n_tokens": 2},
        {"candidate": "short", "logp_sum": -1.5, "logp_mean": -1.5, "n_tokens": 1},
    ]
    out = rank_candidates(rows, gold="gold")
    assert out["accuracy"] == 1
    assert out["gold_rank"] == 1
    assert out["accuracy_sum_logp"] == 0
    assert out["gold_rank_sum_logp"] == 2


def test_generation_em_f1() -> None:
    scores = generation_scores("The electrician.", "electrician")
    assert scores["em"] == 1
    assert scores["f1"] == 1.0
    assert exact_match_score("Cedar Violet Harbor", "cedar violet harbor")
    assert f1_score("cedar violet", "cedar violet harbor") > 0.0


if __name__ == "__main__":
    test_write_contains_fact_once_and_not_question()
    test_source_free_recall_omits_fact()
    test_source_present_recall_contains_fact()
    test_rank_candidates_margin()
    test_primary_is_mean_not_sum()
    test_generation_em_f1()
    print("test_metrics_prompts: PASS")
