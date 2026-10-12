"""TTT-MEMO Exp 02A-1: source-free binding specificity.

Stages:
  python scripts/exp02a1_binding_specificity/run_exp02a1.py --stage freeze-dataset
  python scripts/exp02a1_binding_specificity/run_exp02a1.py --stage test
  python scripts/exp02a1_binding_specificity/run_exp02a1.py --stage smoke

Smoke is 4 exact + 4 semantic_direct pairs, K=32, Mapping A and B.
Do not run the full 32+32 from this entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import (  # noqa: E402
    DATASET_PATH,
    MANIFEST_PATH,
    freeze_dataset_pairs,
    load_dataset_pairs,
)
from evaluate import (  # noqa: E402
    flatten_matrices,
    flatten_predictions,
    pair_binding_table,
    render_summary_md,
    run_pairs,
    smoke_invariants,
    write_jsonl,
)
from protocol import DESIGN_CHOICES, SPEC  # noqa: E402

RESULT_DIR = _REPO_ROOT / "experiments" / "exp02a1_binding_specificity"
SMOKE_DIR = RESULT_DIR / "smoke"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("freeze-dataset", "test", "smoke"), default="smoke")
    p.add_argument("--k-steps", type=int, default=int(SPEC["lora"]["k_steps"]))
    p.add_argument("--force-dataset", action="store_true")
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


def run_cpu_tests(log) -> None:
    tests = sorted((_SCRIPT_DIR / "tests").glob("test_*.py"))
    if not tests:
        raise RuntimeError("no CPU tests found")
    for path in tests:
        log(f"running {path.name}")
        subprocess.check_call([sys.executable, str(path)], cwd=str(_SCRIPT_DIR))


def _to_yaml(obj, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            key = str(k)
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{key}:")
                nested = _to_yaml(v, indent + 1)
                if nested:
                    lines.append(nested)
            else:
                lines.append(f"{sp}{key}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return f"{sp}[]"
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}-")
                nested = _to_yaml(item, indent + 1)
                if nested:
                    lines.append(nested)
            else:
                lines.append(f"{sp}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{sp}{_yaml_scalar(obj)}"


def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if ":" in s or s.startswith(" ") or "#" in s or "\n" in s:
        return json.dumps(s, ensure_ascii=False)
    return s


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = log_print(SMOKE_DIR / "logs" / f"{args.stage}_{stamp}.log")
    try:
        _main(args, log, stamp)
    finally:
        log._fh.close()  # type: ignore[attr-defined]


def _main(args: argparse.Namespace, log, stamp: str) -> None:
    log(f"Exp 02A-1 stage={args.stage} utc={stamp}")
    log(f"spec.backbone.revision_sha={SPEC['backbone']['revision_sha']}")
    meta = freeze_dataset_pairs(force=args.force_dataset if args.stage == "freeze-dataset" else False)
    log(f"dataset sha256={meta['sha256']} path={meta['path']}")
    if args.stage == "freeze-dataset":
        log("freeze-dataset complete. Not loading the full model.")
        return
    run_cpu_tests(log)
    if args.stage == "test":
        log("CPU tests complete.")
        return
    if args.stage != "smoke":
        raise ValueError(f"unhandled stage {args.stage}")
    if args.k_steps != 32:
        raise RuntimeError(f"smoke must use K=32, got {args.k_steps}")
    _run_smoke(log, stamp, meta)


def _run_smoke(log, stamp: str, meta: dict) -> None:
    from model_runtime import load_runtime

    shutil.copy2(DATASET_PATH, SMOKE_DIR / "dataset_pairs.jsonl")
    shutil.copy2(MANIFEST_PATH, SMOKE_DIR / "dataset_manifest.json")
    config = {
        "spec": SPEC,
        "design_choices": DESIGN_CHOICES,
        "dataset_sha256": meta["sha256"],
        "k_steps": 32,
        "stage": "smoke",
        "n_exact_pairs": 4,
        "n_semantic_direct_pairs": 4,
        "utc": stamp,
        "did_not_run": [
            "full 32+32",
            "retention/interference",
            "semantic_inferential",
            "K=8",
            "new LoRA parameters",
            "TTT-E2E",
            "routing or explicit memory",
        ],
    }
    (SMOKE_DIR / "config.yaml").write_text(_to_yaml(config) + "\n", encoding="utf-8")

    log("loading Qwen3-4B-Instruct-2507 BF16 + LoRA for Exp 02A-1 smoke...")
    runtime = load_runtime()
    log(
        f"loaded model_id={runtime.model_id} revision={runtime.revision} "
        f"gpu={runtime.gpu_name} dtype={runtime.torch_dtype} "
        f"trainable={runtime.n_trainable} frozen={runtime.n_frozen}"
    )
    log(f"lora_config={json.dumps(runtime.lora_config)}")
    pairs = load_dataset_pairs(smoke=True)
    log(f"smoke n_pairs={len(pairs)} conditions={list(SPEC['conditions'])} k=32")
    records = run_pairs(runtime, pairs, k_steps=32, log_fn=log)
    failures = smoke_invariants(runtime, pairs, records, tokenizer=runtime.tokenizer, log_fn=log)
    table = pair_binding_table(records, pairs)

    pred_rows = flatten_predictions(records)
    mat_rows = flatten_matrices(records)
    write_jsonl(SMOKE_DIR / "predictions.jsonl", pred_rows)
    write_jsonl(SMOKE_DIR / "score_matrices.jsonl", mat_rows)
    invariants = {"passed": not failures, "failures": failures}
    (SMOKE_DIR / "invariants.json").write_text(
        json.dumps(invariants, indent=2) + "\n", encoding="utf-8"
    )

    import torch

    write_traj = []
    for rec in records:
        if rec.get("write"):
            write_traj.append(
                {
                    "pair_id": rec["pair_id"],
                    "condition": rec["condition"],
                    "mapping": rec["mapping"],
                    "loss_trajectory": rec["write"]["loss_trajectory"],
                    "loss_before": rec["write"]["loss_before"],
                    "loss_after": rec["write"]["loss_after"],
                    "lora_l2_delta_from_init": rec["write"]["lora_l2_delta_from_init"],
                    "lora_max_abs_diff_from_init": rec["write"]["lora_max_abs_diff_from_init"],
                }
            )
    summary = {
        "stage": "smoke",
        "utc": stamp,
        "spec": SPEC,
        "design_choices": DESIGN_CHOICES,
        "dataset_sha256": meta["sha256"],
        "k_steps": 32,
        "n_pairs": len(pairs),
        "pair_ids": [p["pair_id"] for p in pairs],
        "model": {
            "model_id": runtime.model_id,
            "revision": runtime.revision,
            "torch_dtype": runtime.torch_dtype,
            "lora": runtime.lora_config,
            "n_trainable": runtime.n_trainable,
            "n_frozen": runtime.n_frozen,
            "frozen_watch_hash": runtime.frozen_watch_hash,
        },
        "hardware": {
            "gpu_name": runtime.gpu_name,
            "cuda": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else None,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        },
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "binding_table": table,
        "write_trajectories": write_traj,
        "sanity_passed": not failures,
        "sanity_failures": failures,
        "did_not_run": config["did_not_run"],
        "interpretation_rule": DESIGN_CHOICES["interpretation"],
        "bootstrap_note": DESIGN_CHOICES["smoke_bootstrap"],
    }
    (SMOKE_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = render_summary_md(table, dataset_sha=meta["sha256"], n_pairs=len(pairs))
    (SMOKE_DIR / "summary.md").write_text(md, encoding="utf-8")
    log(md)
    if failures:
        raise RuntimeError("exp02a1 smoke sanity failed:\n" + "\n".join(failures))
    log("Exp 02A-1 smoke complete. Stop here. Do not run 32+32.")


if __name__ == "__main__":
    main()
