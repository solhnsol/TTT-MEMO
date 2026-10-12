"""Write/recall rendering and oracle-span loss masks for Exp 02A-1.

Write is a single user turn containing both facts of a pair once. Recall is a
fresh chat with no write tensors. Token labels use character-offset overlap so
the mask does not depend on a second tokenization of the payload.
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
    oracle_span_char: list[tuple[int, int]]
    payload_texts: list[str]
    n_labeled: int


def write_user_content(write_block: str) -> str:
    if not write_block:
        raise ValueError("empty write_block")
    return WRITE_WRAPPER_PREFIX + write_block


def recall_user_content(question: str, *, fact_block: str | None) -> str:
    if not question:
        raise ValueError("empty question")
    tail = f"{question}\n{REPLY_INSTRUCTION}"
    if fact_block is None:
        return tail
    return f"{WRITE_WRAPPER_PREFIX}{fact_block}\n\n{tail}"


def render_chat(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
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


def mask_oracle_payloads(
    tokenizer,
    text: str,
    *,
    items: list[tuple[str, str]],
) -> EncodedSequence:
    """Label tokens that overlap any payload character span.

    Causal CE is applied by the HF model via the usual shift: labels[t] is the
    target for logits[t-1]. Wrappers, chat headers, and fact scaffolding stay
    at -100 and receive no gradient.

    items: list of (fact_sentence, payload) in write order. Both values of the
    pair are supervised; this is one exposure, not two sequences.
    """
    if len(items) != 2:
        raise ValueError(f"expected two (sentence, payload) items, got {len(items)}")
    ids, offsets = _tokenize_with_offsets(tokenizer, text)
    labels = [-100] * len(ids)
    oracle_idx: list[int] = []
    spans: list[tuple[int, int]] = []
    payloads: list[str] = []
    for fact_sentence, payload in items:
        fact_start, _fact_end = _unique_span(text, fact_sentence, kind="fact_sentence")
        rel_start, rel_end = _unique_span(fact_sentence, payload, kind="payload")
        span = (fact_start + rel_start, fact_start + rel_end)
        spans.append(span)
        payloads.append(payload)
        for i, (s, e) in enumerate(offsets):
            if e <= s:
                continue
            if s < span[1] and e > span[0]:
                labels[i] = ids[i]
                if not oracle_idx or oracle_idx[-1] != i:
                    if i not in oracle_idx:
                        oracle_idx.append(i)
    oracle_idx = sorted(set(oracle_idx))
    if not oracle_idx:
        raise RuntimeError("oracle payloads overlapped no tokens")
    if labels.count(-100) == len(labels):
        raise RuntimeError("all labels masked")
    return EncodedSequence(
        text=text,
        input_ids=ids,
        labels=labels,
        oracle_token_indices=oracle_idx,
        oracle_span_char=spans,
        payload_texts=payloads,
        n_labeled=len(oracle_idx),
    )


def encode_write_pair(tokenizer, pair: dict[str, Any], *, mapping: str) -> EncodedSequence:
    if mapping not in ("A", "B"):
        raise ValueError(f"mapping must be A or B, got {mapping!r}")
    block = pair[f"write_block_{mapping}"]
    sentences = pair[f"sentences_{mapping}"]
    payloads = pair[f"payloads_in_write_order_{mapping}"]
    content = write_user_content(block)
    text = render_chat(
        tokenizer,
        [{"role": "user", "content": content}],
        add_generation_prompt=False,
    )
    for ent_key in ("e1", "e2"):
        for style, question in pair["questions"][ent_key].items():
            if question in text:
                raise RuntimeError(
                    f"{pair['pair_id']} mapping {mapping}: {style} question leaked into write"
                )
    if text.count(block) != 1:
        raise RuntimeError(f"{pair['pair_id']}: write block not unique in rendered text")
    for payload in payloads:
        if text.count(payload) != 1:
            raise RuntimeError(f"{pair['pair_id']}: payload {payload!r} not unique in write")
    encoded = mask_oracle_payloads(
        tokenizer,
        text,
        items=list(zip(sentences, payloads)),
    )
    if encoded.n_labeled < 2:
        raise RuntimeError(f"{pair['pair_id']}: expected labels on both values")
    return encoded


def encode_recall_prompt(
    tokenizer,
    pair: dict[str, Any],
    *,
    question: str,
    source_present: bool,
    mapping: str | None,
) -> dict[str, Any]:
    fact = None
    if source_present:
        if mapping not in ("A", "B"):
            raise ValueError("source-present recall requires mapping A or B")
        fact = pair[f"write_block_{mapping}"]
    content = recall_user_content(question, fact_block=fact)
    text = render_chat(
        tokenizer,
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
    )
    ids, _offsets = _tokenize_with_offsets(tokenizer, text)
    if not source_present:
        if pair["write_block_A"] in text or pair["write_block_B"] in text:
            raise RuntimeError(f"{pair['pair_id']}: fact leaked into source-free recall")
        for val in (pair["v1"], pair["v2"]):
            if pair["info_type"] == "exact" and val in text:
                raise RuntimeError(f"{pair['pair_id']}: exact value leaked into recall prompt")
        for sent in pair["sentences_A"] + pair["sentences_B"]:
            if sent in text:
                raise RuntimeError(f"{pair['pair_id']}: write sentence leaked into recall")
    else:
        if fact not in text:
            raise RuntimeError(f"{pair['pair_id']}: source-present recall missing fact block")
    if "<think>" in text:
        raise RuntimeError("thinking block in recall prompt")
    return {"text": text, "input_ids": ids, "source_present": source_present, "question": question}


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
