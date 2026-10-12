"""Recalibrate semantic_inferential_v3 after cashier replacement and single-query change.

  python scripts/exp02a2_semantic_v3/run_recalibrate_v2.py

Writes to experiments/exp02a2_semantic_v3/dataset_preparation/recalibration_v2/.
Does not overwrite the first failed-calibration artifacts in dataset_preparation/.
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
    expand_pair_records,
    freeze_dataset_v3,
    sha256_bytes,
)
from encode_write import load_tokenizer  # noqa: E402
from lexical_audit import render_audit_md, run_lexical_audit  # noqa: E402
from protocol import CONCEPTS, DESIGN_CHOICES, PAIRS, SPEC  # noqa: E402
from tokenizer_stats import collect_tokenizer_stats  # noqa: E402

FIRST_PASS_DIR = _REPO_ROOT / "experiments" / "exp02a2_semantic_v3" / "dataset_preparation"
OUT_DIR = FIRST_PASS_DIR / "recalibration_v2"


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
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL
            )
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


def assert_first_pass_preserved() -> None:
    required = [
        FIRST_PASS_DIR / "source_present_summary.json",
        FIRST_PASS_DIR / "source_present_predictions.jsonl",
        FIRST_PASS_DIR / "dataset_candidate.jsonl",
        FIRST_PASS_DIR / "lexical_audit.json",
        FIRST_PASS_DIR / "dataset_v3_manifest.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"first-pass calibration artifacts missing: {missing}")
    labels = {CONCEPTS[cid]["label"] for cid in CONCEPTS}
    if "carpenter" in labels:
        raise RuntimeError("active CONCEPTS still contains carpenter; refuse to continue")
    if "cashier" not in labels:
        raise RuntimeError("active CONCEPTS missing cashier")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = log_print(OUT_DIR / "logs" / f"recalibrate_v2_{stamp}.log")
    try:
        _main(log, stamp)
    finally:
        log._fh.close()  # type: ignore[attr-defined]


def _main(log, stamp: str) -> None:
    log(f"Exp 02A-2 semantic v3 recalibration_v2 utc={stamp}")
    log("NO LoRA, NO K=32, NO source-free TTT, NO retention.")
    log("S4 carpenter -> cashier. Query set reduced to occupation_query only.")
    assert_first_pass_preserved()
    log(f"preserved first-pass dir={FIRST_PASS_DIR}")

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
    labels = [CONCEPTS[cid]["label"] for cid in CONCEPTS]
    log(f"built {len(pairs)} pairs / {len(expanded)} independent sequences")
    log(f"active labels: {labels}")
    p3 = next(p for p in pairs if p["pair_id"] == "P3")
    log(
        f"P3 labels={p3['candidates']} entities={p3['entities']} "
        f"write_e1A={p3['sequences']['mapping_A'][0]['write_sentence']}"
    )
    if "carpenter" in json.dumps(pairs):
        raise RuntimeError("carpenter leaked into candidate dataset")
    if p3["concepts"]["c1"]["label"] != "cashier":
        raise RuntimeError("P3 c1 is not cashier")

    config = {
        "spec": SPEC,
        "design_choices": DESIGN_CHOICES,
        "concepts": CONCEPTS,
        "pairs": PAIRS,
        "utc": stamp,
        "recalibration": {
            "version": "v2",
            "replaced": {"from": "carpenter", "to": "cashier", "concept_id": "S4"},
            "canonical_query": "occupation_query",
            "dropped_query": "What kind of work does {entity} do?",
            "preserves_first_pass_dir": str(FIRST_PASS_DIR.relative_to(_REPO_ROOT)),
        },
        "did_not_run": [
            "K=32 LoRA",
            "source-free semantic recall",
            "description-binding probe after LoRA",
            "swapped source-free binding",
            "retention/interference",
            "query paraphrase robustness",
            "new occupation search",
            "new hyperparameter configuration",
            "TTT-E2E",
        ],
    }
    (OUT_DIR / "config.yaml").write_text(to_yaml(config) + "\n", encoding="utf-8")
    write_jsonl(OUT_DIR / "dataset_candidate_v2.jsonl", pairs)
    write_jsonl(OUT_DIR / "dataset_expanded_v2.jsonl", expanded)

    audit = run_lexical_audit(pairs)
    (OUT_DIR / "lexical_audit_v2.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    (OUT_DIR / "lexical_audit_v2.md").write_text(render_audit_md(audit) + "\n")
    log(render_audit_md(audit))
    if not audit["passed"]:
        log("LEXICAL AUDIT FAILED. Stopping. Descriptions were not rewritten.")
        raise SystemExit(2)

    log("loading tokenizer for token-span metadata...")
    tokenizer = load_tokenizer()
    pairs_tok = attach_token_metadata(pairs, tokenizer)
    expanded_tok = expand_pair_records(pairs_tok)
    write_jsonl(OUT_DIR / "dataset_candidate_v2.jsonl", pairs_tok)
    write_jsonl(OUT_DIR / "dataset_expanded_v2.jsonl", expanded_tok)
    tok_stats = collect_tokenizer_stats(pairs_tok)
    (OUT_DIR / "tokenizer_stats_v2.json").write_text(
        json.dumps(tok_stats, indent=2, ensure_ascii=False) + "\n"
    )
    log("tokenizer stats:")
    for row in tok_stats["per_pair"]:
        log(
            f"  {row['pair_id']}: labels {row['label_c1']}={row['label_n_tokens_c1']} "
            f"{row['label_c2']}={row['label_n_tokens_c2']} match={row['label_lengths_match']}; "
            f"desc {row['description_n_tokens_c1']} vs {row['description_n_tokens_c2']} "
            f"diff={row['description_length_abs_diff']} "
            f"imbalance={row['description_highly_imbalanced']}"
        )

    pred = run_calibration(pairs_tok, include_static_source_free=True, log_fn=log)
    write_jsonl(OUT_DIR / "source_present_predictions_v2.jsonl", pred)
    summary = summarize_calibration(pred)
    (OUT_DIR / "source_present_summary_v2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    md = render_calibration_md(summary)
    (OUT_DIR / "source_present_summary_v2.md").write_text(md + "\n")
    log(md)

    git_rev = git_revision()
    freeze_extra = {
        "lexical_audit_passed": True,
        "lexical_audit_n_failures": 0,
        "tokenizer_stats_per_pair": tok_stats["per_pair"],
        "source_present_calibration": {
            "passed": summary["passed"],
            "occupation_query_n_correct": summary["occupation_query"]["n_correct"],
            "occupation_query_n": summary["occupation_query"]["n"],
            "occupation_query_two_way_accuracy": summary["occupation_query"]["two_way_accuracy"],
            "occupation_query_failures": summary["occupation_query"]["failures"],
            "generation_em": summary["occupation_query"]["generation_em"],
            "greedy_em_is_not_the_gate": True,
            "source_induced_uplift": summary["source_induced_uplift"],
        },
        "recalibration_version": "v2",
        "replaced_concept": {"from": "carpenter", "to": "cashier", "concept_id": "S4"},
        "canonical_query": SPEC["query_templates"]["occupation_query"],
        "code_git_revision": git_rev,
        "prior_dataset_v1_sha256": v1_sha,
        "prior_dataset_v2_sha256": v2_sha,
        "utc": stamp,
    }

    candidate_manifest = {
        "frozen": bool(summary["passed"]),
        "reason": (
            "source-present 16/16 passed; dataset_v3 frozen"
            if summary["passed"]
            else "source-present calibration failed; dataset_v3 not frozen"
        ),
        **freeze_extra,
    }
    (OUT_DIR / "dataset_v3_manifest_candidate.json").write_text(
        json.dumps(candidate_manifest, indent=2, ensure_ascii=False) + "\n"
    )

    if not summary["passed"]:
        log("SOURCE-PRESENT CALIBRATION FAILED. dataset_v3 was NOT frozen.")
        log("Do not run source-free TTT. Do not silently edit descriptions.")
        for f in summary["occupation_query"]["failures"]:
            up = f.get("source_induced_uplift") or {}
            log(
                f"FAIL {f['pair_id']} {f['mapping']} {f['entity']} gold={f['gold']} "
                f"decoy={f['paired_decoy']} gold_logp_mean={f['gold_logp_mean']:.6f} "
                f"margin={f['margin_logp_mean']:.6f} gold_uplift={up.get('gold_uplift')} "
                f"decoy_uplift={up.get('decoy_uplift')} uplift_margin={up.get('uplift_margin')}"
            )
        raise SystemExit(3)

    if DATASET_PATH.exists():
        raise RuntimeError(
            f"frozen dataset_v3 already exists at {DATASET_PATH}; refuse in-place overwrite"
        )
    manifest = freeze_dataset_v3(pairs_tok, manifest_extra=freeze_extra, force=False)
    log(f"FROZEN dataset_v3 sha256={manifest['sha256']}")
    log(f"frozen path={DATASET_PATH}")
    log(f"manifest path={MANIFEST_PATH}")
    log("STOP. Do not run K=32 LoRA or source-free binding.")


if __name__ == "__main__":
    main()
