"""Stage 3a: check the Llama-3 tokenizer against this checkpoint and against real DCLM tokens.

No model, no checkpoint restore, no JAX device work. This answers one question:

    Can we reproduce the exact token IDs the 125M checkpoint was trained on?

Matching config numbers (vocab 128256, BOS 128000, EOS 128001) is only a necessary
condition -- every Llama-3 derivative shares them. The decisive test is the round
trip on real data: take token IDs straight out of the official DCLM zarr array,
decode them to text, re-encode that text, and require the IDs to come back
identical. A wrong tokenizer cannot survive that.

The probe file is a 32 KB byte range from the head of the official val split:

    gcloud storage cat -r 0-32767 \
      gs://llama3-dclm-filter-8k/data.zarr/val/c/0 \
      --billing-project=<your-project> > data/_probe_val_head.bin

That array is uint32, little-endian and uncompressed (see its zarr.json), so a raw
byte range decodes straight to token IDs without any zarr machinery.

Provenance of the tokenizer itself is pinned in upstream/llama3-tokenizer.toml.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_config import REPO_ROOT, compose_baseline_config  # noqa: E402

# Keep these in sync with upstream/llama3-tokenizer.toml.
TOKENIZER_URL = "https://huggingface.co/NousResearch/Meta-Llama-3-8B/resolve/main/tokenizer.json"
TOKENIZER_SHA256 = "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7"
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer" / "llama3" / "tokenizer.json"

PROBE_PATH = REPO_ROOT / "data" / "_probe_val_head.bin"

BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"

N_ROUNDTRIP = 512  # text tokens to round trip, excluding the leading BOS
N_SHOW = 16  # how many IDs to print


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_tokenizer() -> Path:
    """Download tokenizer.json once, and refuse to proceed on a sha256 mismatch.

    Downloading to a temporary name and renaming only after the hash checks out
    means an interrupted download can never leave a corrupt file that later runs
    would silently accept.
    """
    if not TOKENIZER_PATH.exists():
        TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = TOKENIZER_PATH.with_suffix(".json.part")
        print(f"downloading {TOKENIZER_URL}")
        with urllib.request.urlopen(TOKENIZER_URL, timeout=300) as response, tmp.open("wb") as fh:
            fh.write(response.read())
        digest = sha256_of(tmp)
        if digest != TOKENIZER_SHA256:
            tmp.unlink()
            raise SystemExit(f"sha256 mismatch on download: got {digest}, expected {TOKENIZER_SHA256}")
        tmp.rename(TOKENIZER_PATH)
    return TOKENIZER_PATH


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> None:
    path = ensure_tokenizer()
    digest = sha256_of(path)

    print("--- tokenizer file ---")
    print(f"path:   {path}")
    print(f"bytes:  {path.stat().st_size}")
    print(f"sha256: {digest}")
    print(f"pinned sha256 matches: {digest == TOKENIZER_SHA256}")
    if digest != TOKENIZER_SHA256:
        raise SystemExit("refusing to continue with an unpinned tokenizer file")

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    vocab_all = tok.get_vocab_size(with_added_tokens=True)
    vocab_base = tok.get_vocab_size(with_added_tokens=False)
    bos = tok.token_to_id(BOS_TOKEN)
    eos = tok.token_to_id(EOS_TOKEN)

    print("\n--- tokenizer vs model config ---")
    cfg = compose_baseline_config()
    rows = [
        ("vocab size", vocab_all, cfg.model.vocab_size),
        ("BOS id", bos, cfg.model.bos_token_id),
        ("EOS id", eos, cfg.model.eos_token_id),
    ]
    for name, from_tokenizer, from_config in rows:
        print(f"{name:<11} tokenizer={from_tokenizer:<8} config={from_config:<8} match={from_tokenizer == from_config}")
    print(f"vocab size without added tokens: {vocab_base} (+{vocab_all - vocab_base} added = {vocab_all})")

    if not PROBE_PATH.exists():
        raise SystemExit(f"probe file missing: {PROBE_PATH}\nSee this script's docstring for the command that produces it.")

    import numpy as np

    probe = np.fromfile(PROBE_PATH, dtype="<u4")

    print("\n--- probe (real DCLM val tokens) ---")
    print(f"path:   {PROBE_PATH}")
    print(f"tokens: {probe.size}")
    print(f"first {N_SHOW} IDs: {probe[:N_SHOW].tolist()}")
    print(f"probe[0] is BOS: {int(probe[0]) == bos}")
    print(f"IDs >= 128000 in probe: {sorted(set(probe[probe >= 128000].tolist()))}")

    # Skip the leading BOS: we round trip text only, so that a stray special token
    # cannot make the comparison pass for the wrong reason.
    text_ids = probe[1 : 1 + N_ROUNDTRIP].tolist()
    decoded = tok.decode(text_ids, skip_special_tokens=False)
    reencoded = tok.encode(decoded, add_special_tokens=False).ids

    print("\n--- round trip ---")
    print(f"original IDs  [{len(text_ids)}]: {text_ids[:N_SHOW]}")
    print(f"decoded text  [{len(decoded)} chars]: {decoded[:300]!r}")
    print(f"re-encoded    [{len(reencoded)}]: {reencoded[:N_SHOW]}")

    exact = reencoded == text_ids
    prefix = common_prefix_len(text_ids, reencoded)
    print(f"\nexact match: {exact}")
    if not exact:
        print(f"common prefix: {prefix}/{len(text_ids)} tokens")
        # Our slice ends mid-word, so the decoded tail is not on a token boundary.
        # Disagreement confined to the last token or two is that artifact, not a
        # different tokenizer; anything earlier means the tokenizer is wrong.
        print(f"mismatch confined to slice boundary: {prefix >= len(text_ids) - 2}")

    # The construction stage 3b will actually use: explicit BOS + text tokens,
    # no EOS, matching how lm_dataset._to_batch consumes the pretokenized stream.
    rebuilt = [bos, *reencoded]
    print(f"[BOS] + re-encoded reproduces probe prefix: {rebuilt == probe[: 1 + len(reencoded)].tolist()}")


if __name__ == "__main__":
    main()
