"""Unit tests for official SQuAD v1.1 scoring. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from squad_metrics import exact_match_score, f1_score, normalize_answer, squad_em_f1


def test_normalize_articles_punct_case() -> None:
    assert normalize_answer("The Notre Dame.") == "notre dame"
    assert normalize_answer("  A  CAT  ") == "cat"
    assert normalize_answer("An answer!") == "answer"


def test_em_basic() -> None:
    assert exact_match_score("Notre Dame", "the Notre Dame")
    assert exact_match_score("Saint Bernadette Soubirous", "Saint Bernadette Soubirous")
    assert not exact_match_score("Indiana", "Italy")


def test_f1_partial() -> None:
    # Official token F1 after normalization.
    f1 = f1_score("Notre Dame", "the University of Notre Dame")
    # pred={notre,dame} gold={university,of,notre,dame} overlap=2
    # P=1 R=0.5 F1=2/3
    assert abs(f1 - (2.0 / 3.0)) < 1e-9
    assert f1_score("Indiana", "Indiana") == 1.0
    assert f1_score("foo", "bar") == 0.0


def test_max_over_golds() -> None:
    em, f1 = squad_em_f1("Denver Broncos", ["Broncos", "Denver Broncos", "the Denver Broncos"])
    assert em == 1
    assert f1 == 1.0
    em, f1 = squad_em_f1("Broncos", ["Denver Broncos", "the Denver Broncos"])
    assert em == 0
    assert f1 > 0.0


def test_empty_prediction() -> None:
    em, f1 = squad_em_f1("", ["Indiana"])
    assert em == 0
    assert f1 == 0.0


if __name__ == "__main__":
    test_normalize_articles_punct_case()
    test_em_basic()
    test_f1_partial()
    test_max_over_golds()
    test_empty_prediction()
    print("test_squad_metrics: PASS")
