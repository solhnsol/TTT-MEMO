"""SQuAD 1.1 validation preprocessing for Exp 01.

Downloads the official `dev-v1.1.json` if needed, extracts a gold sentence
from the annotated character span, builds disjoint prompt-dev / main-eval
subsets, and assigns a deterministic no-evidence replacement paragraph.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from squad_metrics import normalize_answer

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
DEFAULT_SEED = 42
PROMPT_DEV_N = 64
MAIN_EVAL_N = 256

# Sentence boundary: terminal punctuation plus optional closing quotes/brackets,
# then whitespace, or a newline. Deterministic; no extra NLP dependency.
_TERMINAL = re.compile(r'[.!?]+["\'”’)\]]*')


@dataclass
class SquadExample:
    id: str
    title: str
    question: str
    gold_paragraph: str
    gold_answers: list[str]
    answer_starts: list[int]
    gold_sentence: str | None
    gold_sentence_char_span: tuple[int, int] | None
    paragraph_n_tokens: int | None = None
    gold_sentence_n_tokens: int | None = None
    no_evidence_source_id: str | None = None
    no_evidence_context: str | None = None
    no_evidence_token_length: int | None = None
    status: str = "ok"
    error: str | None = None
    split: str | None = None

    def to_record(self) -> dict[str, Any]:
        d = asdict(self)
        span = d.get("gold_sentence_char_span")
        if span is not None:
            d["gold_sentence_char_span"] = list(span)
        return d


def download_squad_dev(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(SQUAD_DEV_URL, timeout=120) as response, tmp.open("wb") as fh:
        fh.write(response.read())
    tmp.rename(dest)
    return dest


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return half-open character spans of sentences in `text`."""
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ".!?":
            match = _TERMINAL.match(text, i)
            if match is None:
                i += 1
                continue
            j = match.end()
            if j >= n or text[j].isspace():
                k = j
                while k < n and text[k].isspace() and text[k] != "\n":
                    k += 1
                next_start = k
                if k < n and text[k] == "\n":
                    next_start = k + 1
                    while next_start < n and text[next_start] == "\n":
                        next_start += 1
                if j > start:
                    spans.append((start, j))
                start = next_start
                i = next_start
                continue
        elif ch == "\n":
            end = i
            next_start = i + 1
            while next_start < n and text[next_start] == "\n":
                next_start += 1
            if end > start and text[start:end].strip():
                spans.append((start, end))
            start = next_start
            i = next_start
            continue
        i += 1
    if start < n and text[start:].strip():
        spans.append((start, n))
    return [(a, b) for a, b in spans if a < b]


def gold_sentence_for_span(paragraph: str, answer_start: int, answer_text: str) -> tuple[str, tuple[int, int]]:
    answer_end = answer_start + len(answer_text)
    if answer_start < 0 or answer_end > len(paragraph):
        raise ValueError("answer offset outside paragraph")
    if paragraph[answer_start:answer_end] != answer_text:
        raise ValueError("answer_start does not match answer text")
    spans = sentence_spans(paragraph)
    if not spans:
        raise ValueError("sentence splitter produced no spans")
    overlapping = [(a, b) for a, b in spans if a < answer_end and b > answer_start]
    if not overlapping:
        raise ValueError("no sentence overlaps the gold answer span")
    if len(overlapping) > 1:
        raise ValueError("gold answer span crosses sentence boundaries")
    a, b = overlapping[0]
    sent = paragraph[a:b].strip()
    # Recompute span of the stripped sentence inside the original paragraph.
    stripped_start = paragraph.find(sent, a)
    if stripped_start < 0:
        stripped_start = a
    stripped_end = stripped_start + len(sent)
    return sent, (stripped_start, stripped_end)


def load_squad_examples(path: Path) -> list[SquadExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples: list[SquadExample] = []
    for article in payload["data"]:
        title = article.get("title") or ""
        for para in article.get("paragraphs", []):
            context = para["context"]
            for qa in para.get("qas", []):
                answers = qa.get("answers") or []
                gold_texts = [a["text"] for a in answers if a.get("text")]
                starts = [int(a["answer_start"]) for a in answers if "answer_start" in a]
                ex = SquadExample(
                    id=str(qa["id"]),
                    title=title,
                    question=qa["question"],
                    gold_paragraph=context,
                    gold_answers=gold_texts,
                    answer_starts=starts,
                    gold_sentence=None,
                    gold_sentence_char_span=None,
                )
                if not gold_texts or not starts:
                    ex.status = "preprocess_error"
                    ex.error = "no contiguous gold answer span"
                    examples.append(ex)
                    continue
                # Locate the sentence from the first annotated span that matches its text.
                located = False
                last_err = None
                n_answers = min(len(gold_texts), len(starts))
                for text, start in zip(gold_texts[:n_answers], starts[:n_answers]):
                    try:
                        sent, span = gold_sentence_for_span(context, start, text)
                    except ValueError as exc:
                        last_err = str(exc)
                        continue
                    if not any(ans in sent for ans in gold_texts):
                        last_err = "gold sentence does not contain any annotated gold answer string"
                        continue
                    ex.gold_sentence = sent
                    ex.gold_sentence_char_span = span
                    located = True
                    break
                if not located:
                    ex.status = "preprocess_error"
                    ex.error = last_err or "could not locate gold sentence"
                examples.append(ex)
    return examples


def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def attach_token_lengths(examples: list[SquadExample], tokenizer) -> None:
    for ex in examples:
        ex.paragraph_n_tokens = _n_tokens(tokenizer, ex.gold_paragraph)
        if ex.gold_sentence is not None:
            ex.gold_sentence_n_tokens = _n_tokens(tokenizer, ex.gold_sentence)


def assign_no_evidence(
    examples: list[SquadExample],
    *,
    seed: int = DEFAULT_SEED,
    targets: list[SquadExample] | None = None,
) -> None:
    """Pick an unrelated paragraph per target example.

    Candidates must come from a different article title, must not equal the
    target paragraph, and must not contain any non-empty normalized gold answer
    string. Among remaining candidates, prefer closest paragraph token length;
    ties follow a single seed-42 shuffle order.

    Only `targets` are assigned (default: every ok example). The candidate pool
    is always all successfully preprocessed examples.
    """
    rng = random.Random(seed)
    pool = [ex for ex in examples if ex.status == "ok"]
    shuffled = list(pool)
    rng.shuffle(shuffled)
    shuffle_rank = {ex.id: i for i, ex in enumerate(shuffled)}
    cand_norm = {ex.id: normalize_answer(ex.gold_paragraph) for ex in pool}
    assign_to = targets if targets is not None else pool

    for ex in assign_to:
        if ex.status != "ok":
            continue
        gold_norms = [g for g in (normalize_answer(a) for a in ex.gold_answers) if g]
        best = None
        best_key = None
        target_len = ex.paragraph_n_tokens or 0
        for cand in shuffled:
            if cand.id == ex.id:
                continue
            if cand.title == ex.title:
                continue
            if cand.gold_paragraph == ex.gold_paragraph:
                continue
            if any(g in cand_norm[cand.id] for g in gold_norms):
                continue
            length_delta = abs((cand.paragraph_n_tokens or 0) - target_len)
            key = (length_delta, shuffle_rank[cand.id])
            if best_key is None or key < best_key:
                best = cand
                best_key = key
        if best is None:
            ex.status = "preprocess_error"
            ex.error = "no valid no_evidence replacement"
            continue
        ex.no_evidence_source_id = best.id
        ex.no_evidence_context = best.gold_paragraph
        ex.no_evidence_token_length = best.paragraph_n_tokens


def select_disjoint_splits(
    examples: list[SquadExample],
    *,
    seed: int = DEFAULT_SEED,
    prompt_dev_n: int = PROMPT_DEV_N,
    main_eval_n: int = MAIN_EVAL_N,
) -> dict[str, list[SquadExample]]:
    ok = [ex for ex in examples if ex.status == "ok"]
    rng = random.Random(seed)
    order = list(ok)
    rng.shuffle(order)
    need = prompt_dev_n + main_eval_n
    if len(order) < need:
        raise ValueError(f"only {len(order)} usable examples, need {need}")
    prompt_dev = order[:prompt_dev_n]
    main_eval = order[prompt_dev_n : prompt_dev_n + main_eval_n]
    smoke = prompt_dev[:16]
    assigned = {ex.id for ex in prompt_dev + main_eval}
    for ex in prompt_dev:
        ex.split = "prompt_dev"
    for ex in main_eval:
        ex.split = "main_eval"
    leftover = [ex for ex in ok if ex.id not in assigned]
    return {
        "prompt_dev": prompt_dev,
        "main_eval": main_eval,
        "smoke": smoke,
        "all_ok": ok,
        "leftover": leftover,
    }


def prepare_squad(
    *,
    squad_path: Path,
    tokenizer,
    seed: int = DEFAULT_SEED,
    prompt_dev_n: int = PROMPT_DEV_N,
    main_eval_n: int = MAIN_EVAL_N,
) -> dict[str, Any]:
    download_squad_dev(squad_path)
    examples = load_squad_examples(squad_path)
    attach_token_lengths(examples, tokenizer)
    splits = select_disjoint_splits(
        examples, seed=seed, prompt_dev_n=prompt_dev_n, main_eval_n=main_eval_n
    )
    needed = splits["prompt_dev"] + splits["main_eval"]
    assign_no_evidence(examples, seed=seed, targets=needed)

    def backfill(current: list[SquadExample], n: int, split_name: str) -> list[SquadExample]:
        kept = [ex for ex in current if ex.status == "ok"]
        used = {ex.id for ex in kept}
        leftover = [ex for ex in splits["leftover"] if ex.id not in used]
        i = 0
        while len(kept) < n and i < len(leftover):
            cand = leftover[i]
            i += 1
            assign_no_evidence(examples, seed=seed, targets=[cand])
            if cand.status == "ok":
                cand.split = split_name
                kept.append(cand)
                used.add(cand.id)
        if len(kept) < n:
            raise ValueError(f"could not backfill {split_name} to {n} examples")
        return kept[:n]

    splits["prompt_dev"] = backfill(splits["prompt_dev"], prompt_dev_n, "prompt_dev")
    splits["main_eval"] = backfill(splits["main_eval"], main_eval_n, "main_eval")
    splits["smoke"] = splits["prompt_dev"][:16]
    splits["all_ok"] = [ex for ex in examples if ex.status == "ok"]
    n_fail = sum(1 for ex in examples if ex.status != "ok")
    return {
        "examples": examples,
        "splits": splits,
        "n_total": len(examples),
        "n_ok": sum(1 for ex in examples if ex.status == "ok"),
        "n_preprocess_error": n_fail,
        "seed": seed,
        "prompt_dev_ids": [ex.id for ex in splits["prompt_dev"]],
        "main_eval_ids": [ex.id for ex in splits["main_eval"]],
        "smoke_ids": [ex.id for ex in splits["smoke"]],
    }
