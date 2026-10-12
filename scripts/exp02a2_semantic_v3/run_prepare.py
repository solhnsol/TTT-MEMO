"""Prepare and optionally freeze semantic_inferential_v3.

  python scripts/exp02a2_semantic_v3/run_prepare.py

Does not run LoRA TTT, source-free memory tests, or retention.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
sys.path.insert(0, str(_SCRIPT_DIR))

from calibrate import render_calibration_md, run_calibration, summarize_calibration  # noqa: E402
from dataset import (  # noqa: E402
    DATASET_PATH,
    MANIFEST_PATH,
    attach_token_metadata,
    build_pair_records,
    dataset_bytes,
    expand_pair_records,
    freeze_dataset_v3,
    sha256_bytes,
)
from encode_write import load_tokenizer  # noqa: E402
from lexical_audit import render_audit_md, run_lexical_audit  # noqa: E402
from protocol import CONCEPTS, DESIGN_CHOICES, PAIRS, SPEC  # noqa: E402
from tokenizer_stats import collect_tokenizer_stats  # noqa: E402

OUT_DIR = _REPO_ROOT / "experiments" / "exp02a2_semantic_v3" / "dataset_preparation"


def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(ch in s for ch in ":#\n") or s.startswith(" "):
        return json.dumps(s, ensure_ascii=False)
    return s


def to_yaml(obj, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                nested = to_yaml(v, indent + 1)
                if nested:
                    lines.append(nested)
            else:
                lines.append(f"{sp}{k}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return f"{sp}[]"
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}-")
                nested = to_yaml(item, indent + 1)
                if nested:
                    lines.append(nested)
            else:
                lines.append(f"{sp}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{sp}{_yaml_scalar(obj)}"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_revision() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def log_print(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a", encoding="utf-8")

    def _log(msg: str = "") -> None:
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    _log._fh = fh  # type: ignore[attr-defined]
    return _log


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = log_print(OUT_DIR / "logs" / f"prepare_{stamp}.log")
    try:
        _main(log, stamp)
    finally:
        log._fh.close()  # type: ignore[attr-defined]


def _main(log, stamp: str) -> None:
    log(f"Exp 02A-2 dataset preparation utc={stamp}")
    log("NO LoRA, NO K=32, NO source-free TTT, NO retention.")
    preserved = OUT_DIR / "source_present_summary.json"
    if preserved.exists():
        log(
            "REFUSE: first-pass dataset_preparation artifacts exist and must be preserved. "
            "Use scripts/exp02a2_semantic_v3/run_recalibrate_v2.py which writes to "
            "experiments/exp02a2_semantic_v3/dataset_preparation/recalibration_v2/."
        )
        raise SystemExit(4)
    v1 = _REPO_ROOT / "scripts/exp02a_source_free_lora/frozen/dataset_v1.jsonl"
    v2 = _REPO_ROOT / "scripts/exp02a_source_free_lora/frozen/dataset_v2.jsonl"
    v1_sha = sha256_bytes(v1.read_bytes())
    v2_sha = sha256_bytes(v2.read_bytes())
    if v1_sha != "431d14327bef0a48fa86da5d7148c982368d730ba8265730bf057561c96a3a32":
        raise RuntimeError("dataset_v1 hash changed; refuse to continue")
    if v2_sha != "5a9c483e31e1856479369b1b5853972dc822b2a44e622e8156abdb445b776a01":
        raise RuntimeError("dataset_v2 hash changed; refuse to continue")
    log(f"prior dataset_v1 sha256 unchanged: {v1_sha}")
    log(f"prior dataset_v2 sha256 unchanged: {v2_sha}")

    pairs = build_pair_records()
    expanded = expand_pair_records(pairs)
    log(f"built {len(pairs)} pairs / {len(expanded)} independent sequences")

    config = {
        "spec": SPEC,
        "design_choices": DESIGN_CHOICES,
        "concepts": CONCEPTS,
        "pairs": PAIRS,
        "utc": stamp,
        "did_not_run": [
            "K=32 LoRA",
            "source-free binding",
            "retention/interference",
            "TTT-E2E",
            "architecture change",
        ],
    }
    (OUT_DIR / "config.yaml").write_text(to_yaml(config) + "\n", encoding="utf-8")
    write_jsonl(OUT_DIR / "dataset_candidate.jsonl", pairs)
    write_jsonl(OUT_DIR / "dataset_expanded.jsonl", expanded)

    audit = run_lexical_audit(pairs)
    (OUT_DIR / "lexical_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    (OUT_DIR / "lexical_audit.md").write_text(render_audit_md(audit) + "\n")
    log(render_audit_md(audit))
    if not audit["passed"]:
        log("LEXICAL AUDIT FAILED. Stopping. Descriptions were not rewritten.")
        raise SystemExit(2)

    log("loading tokenizer for token-span metadata...")
    tokenizer = load_tokenizer()
    pairs_tok = attach_token_metadata(pairs, tokenizer)
    expanded_tok = expand_pair_records(pairs_tok)
    write_jsonl(OUT_DIR / "dataset_candidate.jsonl", pairs_tok)
    write_jsonl(OUT_DIR / "dataset_expanded.jsonl", expanded_tok)
    tok_stats = collect_tokenizer_stats(pairs_tok)
    (OUT_DIR / "tokenizer_stats.json").write_text(
        json.dumps(tok_stats, indent=2, ensure_ascii=False) + "\n"
    )
    log("tokenizer stats:")
    for row in tok_stats["per_pair"]:
        log(
            f"  {row['pair_id']}: labels {row['label_c1']}={row['label_n_tokens_c1']} "
            f"{row['label_c2']}={row['label_n_tokens_c2']} match={row['label_lengths_match']}; "
            f"desc {row['description_n_tokens_c1']} vs {row['description_n_tokens_c2']} "
            f"imbalance={row['description_highly_imbalanced']}"
        )

    pred = run_calibration(pairs_tok, include_static_source_free=True, log_fn=log)
    write_jsonl(OUT_DIR / "source_present_predictions.jsonl", pred)
    summary = summarize_calibration(pred)
    (OUT_DIR / "source_present_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    md = render_calibration_md(summary)
    (OUT_DIR / "source_present_summary.md").write_text(md + "\n")
    log(md)

    git_rev = git_revision()
    freeze_extra = {
        "lexical_audit_passed": True,
        "lexical_audit_n_failures": 0,
        "tokenizer_stats_per_pair": tok_stats["per_pair"],
        "source_present_calibration": {
            "passed": summary["passed"],
            "Q1_two_way_accuracy": summary["Q1"]["two_way_accuracy"],
            "Q2_two_way_accuracy": summary["Q2"]["two_way_accuracy"],
            "Q1_failures": summary["Q1"]["failures"],
            "Q2_failures": summary["Q2"]["failures"],
        },
        "code_git_revision": git_rev,
        "prior_dataset_v1_sha256": v1_sha,
        "prior_dataset_v2_sha256": v2_sha,
        "utc": stamp,
    }

    if not summary["passed"]:
        (OUT_DIR / "dataset_v3_manifest.json").write_text(
            json.dumps(
                {
                    "frozen": False,
                    "reason": "source-present calibration failed; dataset_v3 not frozen",
                    **freeze_extra,
                },
                indent=2,
            )
            + "\n"
        )
        log("SOURCE-PRESENT CALIBRATION FAILED. dataset_v3 was NOT frozen.")
        log("Do not run source-free TTT. Do not silently edit descriptions.")
        raise SystemExit(3)

    manifest = freeze_dataset_v3(pairs_tok, manifest_extra=freeze_extra, force=False)
    (OUT_DIR / "dataset_v3_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    log(f"FROZEN dataset_v3 sha256={manifest['sha256']}")
    log(f"frozen path={DATASET_PATH}")
    log(f"manifest path={MANIFEST_PATH}")
    log("STOP. Do not run K=32 LoRA or source-free binding.")


if __name__ == "__main__":
    main()
