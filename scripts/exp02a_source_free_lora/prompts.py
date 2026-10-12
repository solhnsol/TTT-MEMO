"""Write/recall rendering and oracle-span loss masks for Exp 02A.

Write is a single user turn containing the fact once. Recall is a fresh chat
with no write tensors. Token labels use character-offset overlap so the mask
does not depend on a second, possibly inconsistent, tokenization of the payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WRITE_WRAPPER_PREFIX = "Fact:\n"
REPLY_INSTRUCTION = "Reply with only the answer."


@dataclass(frozen=True)
class EncodedSequence:
    text: str
    input_ids: list[int]
    labels: list[int]
    oracle_token_indices: list[int]
    oracle_span_char: tuple[int, int]
    payload_text: str
    n_labeled: int


def write_user_content(fact_sentence: str) -> str:
    if not fact_sentence:
        raise ValueError("empty fact_sentence")
    return WRITE_WRAPPER_PREFIX + fact_sentence


def recall_user_content(question: str, *, fact_sentence: str | None) -> str:
    if not question:
        raise ValueError("empty question")
    tail = f"{question}\n{REPLY_INSTRUCTION}"
    if fact_sentence is None:
        return tail
    return f"{WRITE_WRAPPER_PREFIX}{fact_sentence}\n\n{tail}"


def render_chat(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    # Instruct-2507 is non-thinking. Do not pass enable_thinking=True.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text)}")
    if "<think>" in text:
        raise RuntimeError("chat template inserted a thinking block")
    return text


def _unique_span(haystack: str, needle: str, *, kind: str) -> tuple[int, int]:
    start = haystack.find(needle)
    if start < 0:
        raise ValueError(f"{kind} not found in rendered text")
    if haystack.find(needle, start + 1) >= 0:
        raise ValueError(f"{kind} is not unique in rendered text")
    return start, start + len(needle)


def _tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    if len(ids) != len(offsets):
        raise RuntimeError("offset mapping length mismatch")
    if not ids:
        raise RuntimeError("empty tokenization")
    return ids, offsets


def mask_oracle_payload(
    tokenizer,
    text: str,
    *,
    payload: str,
    fact_sentence: str,
) -> EncodedSequence:
    """Label only tokens that overlap the payload character span.

    Causal CE is applied by the HF model via the usual shift: labels[t] is the
    target for logits[t-1]. Wrappers, chat headers, and fact scaffolding stay
    at -100 and receive no gradient.
    """
    fact_start, _fact_end = _unique_span(text, fact_sentence, kind="fact_sentence")
    rel_start, rel_end = _unique_span(fact_sentence, payload, kind="payload")
    span = (fact_start + rel_start, fact_start + rel_end)
    ids, offsets = _tokenize_with_offsets(tokenizer, text)
    labels = [-100] * len(ids)
    oracle_idx: list[int] = []
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if s < span[1] and e > span[0]:
            labels[i] = ids[i]
            oracle_idx.append(i)
    if not oracle_idx:
        raise RuntimeError("oracle payload overlapped no tokens")
    if labels.count(-100) == len(labels):
        raise RuntimeError("all labels masked")
    return EncodedSequence(
        text=text,
        input_ids=ids,
        labels=labels,
        oracle_token_indices=oracle_idx,
        oracle_span_char=span,
        payload_text=payload,
        n_labeled=len(oracle_idx),
    )


def encode_write(tokenizer, example: dict[str, Any]) -> EncodedSequence:
    content = write_user_content(example["fact_sentence"])
    text = render_chat(
        tokenizer,
        [{"role": "user", "content": content}],
        add_generation_prompt=False,
    )
    if example["question"] in text:
        raise RuntimeError(f"{example['example_id']}: question leaked into write text")
    if text.count(example["write_payload"]) != 1:
        raise RuntimeError(f"{example['example_id']}: payload not unique in write text")
    if text.count(example["fact_sentence"]) != 1:
        raise RuntimeError(f"{example['example_id']}: fact sentence not unique in write text")
    encoded = mask_oracle_payload(
        tokenizer,
        text,
        payload=example["write_payload"],
        fact_sentence=example["fact_sentence"],
    )
    return encoded


def encode_recall_prompt(
    tokenizer,
    example: dict[str, Any],
    *,
    source_present: bool,
) -> dict[str, Any]:
    fact = example["fact_sentence"] if source_present else None
    content = recall_user_content(example["question"], fact_sentence=fact)
    text = render_chat(
        tokenizer,
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
    )
    ids, _offsets = _tokenize_with_offsets(tokenizer, text)
    if not source_present:
        if example["fact_sentence"] in text:
            raise RuntimeError(f"{example['example_id']}: fact leaked into source-free recall")
        if example["write_payload"] in text:
            raise RuntimeError(f"{example['example_id']}: payload leaked into source-free recall")
        if example["info_type"] == "exact" and example["answer"] in text:
            raise RuntimeError(f"{example['example_id']}: exact answer leaked into recall prompt")
    else:
        if example["fact_sentence"] not in text:
            raise RuntimeError(f"{example['example_id']}: source-present recall missing fact")
    if "<think>" in text:
        raise RuntimeError("thinking block in recall prompt")
    return {"text": text, "input_ids": ids, "source_present": source_present}


def encode_candidate(
    tokenizer,
    prompt_text: str,
    candidate: str,
) -> dict[str, Any]:
    """Full chat = recall prompt + assistant candidate. Score candidate tokens only.

    Tokenize the concatenated string once so merge rules at the prompt/answer
    boundary cannot desynchronize two separate encodings.
    """
    if not prompt_text.endswith("<|im_start|>assistant\n"):
        raise RuntimeError(
            "unexpected generation prompt tail; refuse to guess candidate tokenization: "
            f"{prompt_text[-40]!r}"
        )
    full_text = prompt_text + candidate + "<|im_end|>\n"
    ids, offsets = _tokenize_with_offsets(tokenizer, full_text)
    # Restrict the candidate span to the assistant suffix. Exact source-present
    # prompts also contain the gold string inside the fact sentence.
    suffix_start = len(prompt_text)
    if full_text[suffix_start : suffix_start + len(candidate)] != candidate:
        raise RuntimeError("candidate is not a prefix of the assistant suffix")
    cand_start = suffix_start
    cand_end = suffix_start + len(candidate)
    labels = [-100] * len(ids)
    cand_idx: list[int] = []
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if s < cand_end and e > cand_start:
            labels[i] = ids[i]
            cand_idx.append(i)
    if not cand_idx:
        raise RuntimeError("candidate overlapped no tokens")
    if cand_idx[0] == 0:
        raise RuntimeError("candidate tokens start at position 0")
    return {
        "text": full_text,
        "input_ids": ids,
        "labels": labels,
        "candidate_token_indices": cand_idx,
        "n_tokens": len(cand_idx),
        "candidate": candidate,
        "prompt_n_tokens": cand_idx[0],
    }


def to_tensors(encoded: EncodedSequence | dict[str, Any], device, *, include_labels: bool):
    import torch

    ids = encoded.input_ids if isinstance(encoded, EncodedSequence) else encoded["input_ids"]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if include_labels:
        labels = encoded.labels if isinstance(encoded, EncodedSequence) else encoded["labels"]
        out["labels"] = torch.tensor([labels], dtype=torch.long, device=device)
    return out


def assert_no_cache_reuse(*, past_key_values: Any, where: str) -> None:
    if past_key_values is not None:
        raise RuntimeError(f"{where}: past_key_values must be None (source-free invariant)")
