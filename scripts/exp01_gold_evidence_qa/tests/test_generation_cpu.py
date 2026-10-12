"""CPU tests for frozen generation postprocessing and prompt encoding."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "baseline_125m" / "common"))

from baseline_config import TOKENIZER_PATH
from generation import encode_prompt, postprocess_generation
from prompts import T1, T2, T3


def test_postprocess_first_line_and_eos() -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos = tok.token_to_id("<|end_of_text|>")
    ids = tok.encode("Indiana\nShe was born", add_special_tokens=False).ids + [eos, eos]
    raw, scored = postprocess_generation(ids, tok, eos)
    assert "Indiana" in raw
    assert scored == "Indiana"


def test_encode_prompt_swa_and_question_after_context() -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    bos = tok.token_to_id("<|begin_of_text|>")
    eos = tok.token_to_id("<|end_of_text|>")
    ctx = "The Notre Dame is in Indiana."
    q = "Where is Notre Dame?"
    for tmpl in (T1, T2, T3):
        enc = encode_prompt(
            tok,
            tmpl,
            context=ctx,
            question=q,
            bos_id=bos,
            eos_id=eos,
            chunk_size=1024,
            swa_window=8192,
        )
        assert enc.prompt_len < 1024
        assert enc.evidence_token_length > 0
        assert enc.question_start > enc.evidence_start
        assert enc.evidence_question_distance < 8192
        assert len(enc.padded_ids) == 1024
        assert enc.padded_ids[: enc.prompt_len] == enc.input_ids_unpadded
        assert all(t == eos for t in enc.padded_ids[enc.prompt_len :])
        assert enc.input_ids_unpadded[0] == bos


if __name__ == "__main__":
    test_postprocess_first_line_and_eos()
    test_encode_prompt_swa_and_question_after_context()
    print("test_generation_cpu: PASS")
