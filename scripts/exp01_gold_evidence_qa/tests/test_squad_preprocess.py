"""CPU unit tests for gold-sentence extraction and no-evidence assignment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from squad_metrics import normalize_answer
from squad_preprocess import (
    SquadExample,
    assign_no_evidence,
    gold_sentence_for_span,
    load_squad_examples,
    select_disjoint_splits,
    sentence_spans,
)


def test_sentence_spans_basic() -> None:
    text = "The Notre Dame is in Indiana. It was founded in 1842."
    spans = sentence_spans(text)
    sents = [text[a:b] for a, b in spans]
    assert sents == ["The Notre Dame is in Indiana.", "It was founded in 1842."]


def test_gold_sentence_contains_answer() -> None:
    para = "The Notre Dame is in Indiana. It was founded in 1842."
    answer = "Indiana"
    start = para.index(answer)
    sent, span = gold_sentence_for_span(para, start, answer)
    assert sent == "The Notre Dame is in Indiana."
    assert answer in sent
    assert para[span[0] : span[1]] == sent


def test_cross_sentence_span_fails() -> None:
    para = "First sentence. Second sentence."
    # Span covering the period between sentences.
    start = para.index("sentence. S")
    text = para[start : start + len("sentence. S")]
    try:
        gold_sentence_for_span(para, start, text)
    except ValueError as exc:
        assert "crosses" in str(exc)
    else:
        raise AssertionError("expected cross-sentence failure")


def test_load_and_split_fixture(tmp_path: Path | None = None) -> None:
    payload = {
        "data": [
            {
                "title": "ArticleA",
                "paragraphs": [
                    {
                        "context": "Alice lives in Paris. She likes bread.",
                        "qas": [
                            {
                                "id": "a1",
                                "question": "Where does Alice live?",
                                "answers": [{"text": "Paris", "answer_start": 15}],
                            }
                        ],
                    }
                ],
            },
            {
                "title": "ArticleB",
                "paragraphs": [
                    {
                        "context": "Bob lives in Tokyo. He likes tea.",
                        "qas": [
                            {
                                "id": "b1",
                                "question": "Where does Bob live?",
                                "answers": [{"text": "Tokyo", "answer_start": 13}],
                            }
                        ],
                    }
                ],
            },
            {
                "title": "ArticleC",
                "paragraphs": [
                    {
                        "context": "Carol lives in Rome. She likes pasta.",
                        "qas": [
                            {
                                "id": "c1",
                                "question": "Where does Carol live?",
                                "answers": [{"text": "Rome", "answer_start": 15}],
                            }
                        ],
                    }
                ],
            },
        ]
    }
    if tmp_path is None:
        examples = load_squad_examples_from_obj(payload)
    else:
        path = tmp_path / "mini.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        examples = load_squad_examples(path)
    assert all(ex.status == "ok" for ex in examples)
    assert examples[0].gold_sentence == "Alice lives in Paris."
    for ex in examples:
        ex.paragraph_n_tokens = len(ex.gold_paragraph.split())
    assign_no_evidence(examples, seed=42)
    for ex in examples:
        assert ex.no_evidence_source_id is not None
        assert ex.no_evidence_source_id != ex.id
        src = next(s for s in examples if s.id == ex.no_evidence_source_id)
        assert src.title != ex.title
        for gold in ex.gold_answers:
            assert normalize_answer(gold) not in normalize_answer(ex.no_evidence_context or "")


def load_squad_examples_from_obj(payload: dict) -> list[SquadExample]:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        path = Path(fh.name)
    try:
        return load_squad_examples(path)
    finally:
        path.unlink(missing_ok=True)


def test_disjoint_splits() -> None:
    examples = []
    for i in range(40):
        examples.append(
            SquadExample(
                id=f"id{i}",
                title=f"t{i}",
                question="q",
                gold_paragraph=f"para {i}.",
                gold_answers=["x"],
                answer_starts=[0],
                gold_sentence=f"para {i}.",
                gold_sentence_char_span=(0, 8),
                status="ok",
            )
        )
    splits = select_disjoint_splits(examples, seed=42, prompt_dev_n=8, main_eval_n=16)
    dev_ids = {ex.id for ex in splits["prompt_dev"]}
    main_ids = {ex.id for ex in splits["main_eval"]}
    assert len(dev_ids) == 8
    assert len(main_ids) == 16
    assert dev_ids.isdisjoint(main_ids)
    smoke_ids = {ex.id for ex in splits["smoke"]}
    assert smoke_ids <= dev_ids
    # Deterministic: same seed -> same ids
    splits2 = select_disjoint_splits(examples, seed=42, prompt_dev_n=8, main_eval_n=16)
    assert [ex.id for ex in splits2["prompt_dev"]] == [ex.id for ex in splits["prompt_dev"]]


if __name__ == "__main__":
    test_sentence_spans_basic()
    test_gold_sentence_contains_answer()
    test_cross_sentence_span_fails()
    test_load_and_split_fixture()
    test_disjoint_splits()
    print("test_squad_preprocess: PASS")
