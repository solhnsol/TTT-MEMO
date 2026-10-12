"""CPU tests that condition prompts differ only in the injected context."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts import CANDIDATE_TEMPLATES, render_prompt


def test_conditions_differ_only_by_context() -> None:
    question = "Where is Notre Dame?"
    sent = "The Notre Dame is in Indiana."
    para = "The Notre Dame is in Indiana. It was founded in 1842."
    other = "Bob lives in Tokyo. He likes tea."
    for tmpl in CANDIDATE_TEMPLATES:
        p_sent = render_prompt(tmpl, context=sent, question=question)
        p_para = render_prompt(tmpl, context=para, question=question)
        p_none = render_prompt(tmpl, context=other, question=question)
        assert p_sent.replace(sent, "CTX") == p_para.replace(para, "CTX") == p_none.replace(other, "CTX")
        assert p_sent.count(question) == 1
        assert p_sent.index(sent) < p_sent.index(question)


if __name__ == "__main__":
    test_conditions_differ_only_by_context()
    print("test_prompts: PASS")
