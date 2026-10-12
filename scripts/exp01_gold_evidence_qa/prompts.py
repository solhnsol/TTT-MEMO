"""Frozen candidate prompt templates for Exp 01.

The three strings are the experimental specification. Prompt selection uses
only the gold_sentence condition on the 64-example prompt-dev split.
The winner is frozen before the 256-example main evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    template: str


# Exact strings from the Exp 01 spec. Trailing "Answer:" / "A:" has no extra space.
T1 = PromptTemplate(
    template_id="T1",
    template="Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
)
T2 = PromptTemplate(
    template_id="T2",
    template="{context}\n\nQ: {question}\nA:",
)
T3 = PromptTemplate(
    template_id="T3",
    template=(
        "Read the context and answer the question with a short answer.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    ),
)

CANDIDATE_TEMPLATES: tuple[PromptTemplate, ...] = (T1, T2, T3)
TEMPLATE_BY_ID = {t.template_id: t for t in CANDIDATE_TEMPLATES}


def render_prompt(template: PromptTemplate, *, context: str, question: str) -> str:
    return template.template.format(context=context, question=question)


def context_char_span(template: PromptTemplate, *, context: str, question: str) -> tuple[int, int]:
    """Character span of `{context}` inside the rendered prompt."""
    rendered = render_prompt(template, context=context, question=question)
    marker = "{context}"
    prefix = template.template.split(marker, 1)[0].format(question=question)
    start = len(prefix)
    end = start + len(context)
    if rendered[start:end] != context:
        idx = rendered.find(context)
        if idx < 0:
            raise ValueError("could not locate context span in rendered prompt")
        start, end = idx, idx + len(context)
    return start, end


def question_char_span(template: PromptTemplate, *, context: str, question: str) -> tuple[int, int]:
    rendered = render_prompt(template, context=context, question=question)
    idx = rendered.find(question)
    if idx < 0:
        raise ValueError("could not locate question span in rendered prompt")
    return idx, idx + len(question)
