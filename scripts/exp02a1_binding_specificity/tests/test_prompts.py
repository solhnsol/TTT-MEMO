"""CPU tests for prompt strings. Tokenizer tests run at freeze/eval."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts import recall_user_content, write_user_content


def test_write_wrapper() -> None:
    block = "The passphrase assigned to Areth-001 is cedar violet harbor.\nThe passphrase assigned to Belmar-002 is maple amber quartz."
    text = write_user_content(block)
    assert text.startswith("Fact:\n")
    assert text.count(block) == 1
    assert "?" not in text


def test_source_free_omits_facts() -> None:
    q = "What is the passphrase assigned to Areth-001?"
    text = recall_user_content(q, fact_block=None)
    assert q in text
    assert "Reply with only the answer." in text
    assert "Fact:" not in text
    assert "cedar" not in text


def test_source_present_contains_both_facts() -> None:
    q = "What is the passphrase assigned to Areth-001?"
    block = "sent A\nsent B"
    text = recall_user_content(q, fact_block=block)
    assert text.startswith("Fact:\n")
    assert block in text
    assert q in text


if __name__ == "__main__":
    test_write_wrapper()
    test_source_free_omits_facts()
    test_source_present_contains_both_facts()
    print("test_prompts: PASS")
