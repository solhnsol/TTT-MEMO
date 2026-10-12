"""Exp 02A-2R: entity replication of frozen dataset_v3 concepts.

  python scripts/exp02a2_semantic_v3/run_exp02a2r.py --stage freeze-schedule
  python scripts/exp02a2_semantic_v3/run_exp02a2r.py --stage test
  python scripts/exp02a2_semantic_v3/run_exp02a2r.py --stage k32

Does not modify occupations, descriptions, scaffold, query, or LoRA recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import attach_token_metadata  # noqa: E402
from encode_write import load_tokenizer  # noqa: E402
from evaluate_memory import (  # noqa: E402
    flatten_matrices,
    flatten_predictions,
    pair_binding_table,
    run_invariants,
    run_pairs,
    write_jsonl,
)
from memory_protocol import DESIGN_CHOICES, EXPECTED_DATASET_V3_SHA256, SPEC  # noqa: E402
from metrics import bootstrap_mean_ci  # noqa: E402
from replication_schedule import (  # noqa: E402
    build_replicate_pairs,
    freeze_schedule,
    load_schedule,
    schedule_bytes,
    sha256_bytes,
)

RESULT_DIR = _REPO_ROOT / "experiments" / "exp02a2_semantic_v3" / "entity_replication_k32"
SCHEDULE_PATH = RESULT_DIR / "entity_schedule.jsonl"

REPLICATION_CONDITIONS = (
    "static_source_free",
    "ttt_mapping_A_source_free",
    "ttt_mapping_B_source_free",
    "static_mapping_A_source_present",
    "static_mapping_B_source_present",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("freeze-schedule", "test", "k32"), default="k32")
    p.add_argument("--k-steps", type=int, default=32)
    return p.parse_args()


def log_print(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a", encoding="utf-8")

    def _log(msg: str = "") -> None:
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    _log._fh = fh  # type: ignore[attr-defined]
    return _log


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


def run_cpu_tests(log) -> None:
    tests = sorted((_SCRIPT_DIR / "tests").glob("test_*.py"))
    for path in tests:
        log(f"running {path.name}")
        subprocess.check_call([sys.executable, str(path)], cwd=str(_SCRIPT_DIR))


def summarize_probe_rows(rows: list[dict]) -> dict:
    bswaps = [float(r["B_swap"]) for r in rows]
    return {
        "n": len(rows),
        "mean_B_swap": float(sum(bswaps) / len(bswaps)),
        "median_B_swap": float(statistics.median(bswaps)),
        "fraction_B_swap_gt_0": float(sum(x > 0 for x in bswaps) / len(bswaps)),
        "fraction_B_A_and_B_B_gt_0": float(
            sum(float(r["B_A"]) > 0 and float(r["B_B"]) > 0 for r in rows) / len(rows)
        ),
        "fraction_both_entity_flip": float(sum(int(r["both_flipped"]) for r in rows) / len(rows)),
        "fraction_e1_flip": float(sum(int(r["e1_flipped"]) for r in rows) / len(rows)),
        "fraction_e2_flip": float(sum(int(r["e2_flipped"]) for r in rows) / len(rows)),
        "mean_two_way_top1": float(sum(float(r["two_way_acc_ttt"]) for r in rows) / len(rows)),
        "bootstrap_mean_B_swap_95ci": bootstrap_mean_ci(bswaps),
    }


def replication_report(table: dict) -> dict:
    per_probe = {}
    for probe in ("description_binding", "semantic_label"):
        by_src: dict[str, list] = defaultdict(list)
        all_rows = []
        for prow in table["per_pair"]:
            row = dict(prow["probes"][probe])
            row["source_pair_id"] = prow["source_pair_id"]
            row["pair_id"] = prow["pair_id"]
            by_src[prow["source_pair_id"]].append(row)
            all_rows.append(row)
        per_source = {src: summarize_probe_rows(rs) for src, rs in sorted(by_src.items())}
        overall = summarize_probe_rows(all_rows)
        overall["note"] = (
            "32 entity replicates are not 32 independent semantic concepts. "
            "Concept-level generalization remains limited to the same 4 occupation pairs."
        )
        sp_rows = [r for r in all_rows if r.get("source_present_B_swap") is not None]
        source_present = None
        failures = []
        if sp_rows:
            source_present = {
                "n": len(sp_rows),
                "mean_B_swap": float(sum(float(r["source_present_B_swap"]) for r in sp_rows) / len(sp_rows)),
                "fraction_B_swap_gt_0": float(
                    sum(float(r["source_present_B_swap"]) > 0 for r in sp_rows) / len(sp_rows)
                ),
                "mean_two_way": float(
                    sum(float(r["source_present_two_way_accuracy"]) for r in sp_rows) / len(sp_rows)
                ),
            }
            for r in sp_rows:
                fail_b = float(r["source_present_B_swap"]) <= 0
                fail_acc = float(r["source_present_two_way_accuracy"]) < 1.0
                if fail_b or fail_acc:
                    failures.append(
                        {
                            "pair_id": r["pair_id"],
                            "source_pair_id": r["source_pair_id"],
                            "source_present_B_swap": r["source_present_B_swap"],
                            "source_present_two_way_accuracy": r["source_present_two_way_accuracy"],
                            "reason": (
                                ("B_swap<=0 " if fail_b else "")
                                + ("two_way<1 " if fail_acc else "")
                            ).strip(),
                        }
                    )
        per_probe[probe] = {
            "per_concept_pair": per_source,
            "overall": overall,
            "source_present_diagnostic": source_present,
            "source_present_failures": failures if probe == "semantic_label" else [],
        }
    return per_probe


def render_md(report: dict, *, schedule_sha: str, n: int) -> str:
    lines = [
        "# Exp 02A-2R entity replication (K=32)",
        "",
        f"dataset_v3 sha256: `{EXPECTED_DATASET_V3_SHA256}`",
        f"entity schedule sha256: `{schedule_sha}`",
        f"n replicates: {n} (4 concept pairs × 8 nonce-entity pairs).",
        "32 entity replicates are **not** 32 independent semantic concepts.",
        "",
    ]
    for probe in ("description_binding", "semantic_label"):
        block = report[probe]
        ov = block["overall"]
        ci = ov["bootstrap_mean_B_swap_95ci"]
        lines.extend(
            [
                f"## {probe}",
                "",
                f"- overall mean B_swap: **{ov['mean_B_swap']:.3f}**  "
                f"(descriptive 95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}])",
                f"- median B_swap: {ov['median_B_swap']:.3f}",
                f"- fraction B_swap > 0: {ov['fraction_B_swap_gt_0']:.3f}",
                f"- fraction B_A>0 and B_B>0: {ov['fraction_B_A_and_B_B_gt_0']:.3f}",
                f"- both-entity flip fraction: {ov['fraction_both_entity_flip']:.3f}",
                f"- mean TTT 2-way top-1: {ov['mean_two_way_top1']:.3f}",
                "",
                "| concept pair | mean B_swap | median | B_swap>0 | B_A>0 & B_B>0 | both flip |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for src, row in block["per_concept_pair"].items():
            lines.append(
                f"| {src} | {row['mean_B_swap']:.3f} | {row['median_B_swap']:.3f} | "
                f"{row['fraction_B_swap_gt_0']:.3f} | {row['fraction_B_A_and_B_B_gt_0']:.3f} | "
                f"{row['fraction_both_entity_flip']:.3f} |"
            )
        if probe == "semantic_label":
            sp = block["source_present_diagnostic"]
            fails = block["source_present_failures"]
            lines.extend(
                [
                    "",
                    "### Source-present semantic-label diagnostic (LoRA off)",
                    "",
                    f"- mean B_swap: {sp['mean_B_swap']:.3f}" if sp else "- no source-present rows",
                    f"- fraction B_swap>0: {sp['fraction_B_swap_gt_0']:.3f}" if sp else "",
                    f"- mean 2-way: {sp['mean_two_way']:.3f}" if sp else "",
                    f"- failures reported (not dropped): {len(fails)}",
                    "",
                ]
            )
            for f in fails:
                lines.append(
                    f"- {f['pair_id']} {f['reason']} "
                    f"B_swap={f['source_present_B_swap']:.3f} "
                    f"acc={f['source_present_two_way_accuracy']:.3f}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = log_print(RESULT_DIR / "logs" / f"{args.stage}_{stamp}.log")
    try:
        _main(args, log, stamp)
    finally:
        log._fh.close()  # type: ignore[attr-defined]


def _ensure_schedule(log) -> dict:
    log("loading tokenizer to freeze entity schedule (no model forward pass)...")
    tokenizer = load_tokenizer()
    meta = freeze_schedule(SCHEDULE_PATH, tokenizer, force=False)
    log(f"entity schedule sha256={meta['sha256']} n_matched={meta['n_token_length_matched_pairs']}/32")
    return meta


def _main(args, log, stamp: str) -> None:
    log(f"Exp 02A-2R stage={args.stage} utc={stamp}")
    log(f"dataset_v3 sha256={EXPECTED_DATASET_V3_SHA256}")
    if args.stage == "freeze-schedule":
        meta = _ensure_schedule(log)
        log("schedule frozen. Not loading the full model.")
        log(json.dumps(meta, indent=2))
        return
    run_cpu_tests(log)
    if args.stage == "test":
        log("CPU tests complete.")
        return
    if args.k_steps != 32:
        raise RuntimeError(f"K must be 32, got {args.k_steps}")
    meta = _ensure_schedule(log)
    schedule = load_schedule(SCHEDULE_PATH)
    payload_sha = sha256_bytes(schedule_bytes(schedule))
    if payload_sha != meta["sha256"]:
        raise RuntimeError("schedule jsonl sha256 diverged from manifest")
    tokenizer = load_tokenizer()
    pairs = attach_token_metadata(build_replicate_pairs(schedule), tokenizer)
    log(f"built {len(pairs)} replicate pairs")

    config = {
        "experiment_id": "exp02a2r_entity_replication",
        "dataset_v3_sha256": EXPECTED_DATASET_V3_SHA256,
        "entity_schedule_sha256": meta["sha256"],
        "k_steps": 32,
        "n_replicates": 32,
        "spec": SPEC,
        "design_choices": DESIGN_CHOICES,
        "utc": stamp,
        "did_not_run": [
            "retention/interference",
            "new occupations/descriptions",
            "query paraphrases",
            "K=8",
            "TTT-E2E",
            "hyperparameter search",
        ],
    }
    (RESULT_DIR / "config.yaml").write_text(to_yaml(config) + "\n", encoding="utf-8")
    write_jsonl(RESULT_DIR / "dataset_replicates.jsonl", pairs)

    from model_runtime import load_runtime

    log("loading Qwen3-4B-Instruct-2507 BF16 + LoRA for Exp 02A-2R...")
    runtime = load_runtime()
    log(
        f"loaded revision={runtime.revision} gpu={runtime.gpu_name} "
        f"trainable={runtime.n_trainable}"
    )
    records = run_pairs(
        runtime,
        pairs,
        k_steps=32,
        log_fn=log,
        conditions=REPLICATION_CONDITIONS,
    )
    failures = run_invariants(runtime, pairs, records, log_fn=log)
    table = pair_binding_table(records, pairs)
    report = replication_report(table)
    write_jsonl(RESULT_DIR / "predictions.jsonl", flatten_predictions(records))
    write_jsonl(RESULT_DIR / "score_matrices.jsonl", flatten_matrices(records))
    write_traj = []
    for rec in records:
        if rec.get("write"):
            write_traj.append(
                {
                    "pair_id": rec["pair_id"],
                    "source_pair_id": rec.get("source_pair_id"),
                    "condition": rec["condition"],
                    "mapping": rec["mapping"],
                    "loss_trajectory": rec["write"]["loss_trajectory"],
                    "loss_before": rec["write"]["loss_before"],
                    "loss_after": rec["write"]["loss_after"],
                    "lora_l2_delta_from_init": rec["write"]["lora_l2_delta_from_init"],
                    "lora_max_abs_diff_from_init": rec["write"]["lora_max_abs_diff_from_init"],
                }
            )
    write_jsonl(RESULT_DIR / "write_loss_trajectories.jsonl", write_traj)
    (RESULT_DIR / "invariants.json").write_text(
        json.dumps({"passed": not failures, "failures": failures}, indent=2) + "\n"
    )
    summary = {
        "utc": stamp,
        "dataset_v3_sha256": EXPECTED_DATASET_V3_SHA256,
        "entity_schedule_sha256": meta["sha256"],
        "k_steps": 32,
        "n_replicates": 32,
        "sanity_passed": not failures,
        "sanity_failures": failures,
        "binding_table": table,
        "replication_report": report,
        "did_not_run": config["did_not_run"],
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    md = render_md(report, schedule_sha=meta["sha256"], n=32)
    (RESULT_DIR / "summary.md").write_text(md + "\n")
    log(md)
    if failures:
        raise RuntimeError("exp02a2r sanity failed:\n" + "\n".join(failures))
    log("Exp 02A-2R complete. Stop here.")


if __name__ == "__main__":
    main()
