"""TTT-MEMO Exp 02A: source-free persistent-memory diagnostic.

Stages:
  python scripts/exp02a_source_free_lora/run_exp02a.py --stage freeze-dataset
  python scripts/exp02a_source_free_lora/run_exp02a.py --stage freeze-dataset-v2
  python scripts/exp02a_source_free_lora/run_exp02a.py --stage test
  python scripts/exp02a_source_free_lora/run_exp02a.py --stage smoke
  python scripts/exp02a_source_free_lora/run_exp02a.py --stage pilot-v2
  python scripts/exp02a_source_free_lora/run_exp02a.py --stage pilot-v2-k32

Smoke uses frozen dataset_v1 (8+8). The formal K=8 pilot uses dataset_v2
(32 exact + 32 semantic_direct). K=32 is the preregistered write-budget
escalation: same v2 examples, TTT conditions only.
"""

from __future__ import annotations

import argparse
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

from dataset_v1 import (  # noqa: E402
    DATASET_PATH,
    freeze_dataset_v1,
    load_dataset_v1,
)
from dataset_v2 import (  # noqa: E402
    DATASET_PATH as DATASET_V2_PATH,
    freeze_dataset_v2,
    load_dataset_v2,
)
from evaluate import (  # noqa: E402
    condition_metrics,
    cross_k_memory_table,
    paired_ttt_effect,
    run_conditions,
    smoke_invariants,
    write_jsonl,
)
from protocol import DESIGN_CHOICES, SPEC, TTT_CONDITIONS  # noqa: E402

RESULT_DIR = _REPO_ROOT / "experiments" / "exp02a_source_free_lora"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=("freeze-dataset", "freeze-dataset-v2", "test", "smoke", "pilot-v2", "pilot-v2-k32"),
        default="smoke",
    )
    p.add_argument("--k-steps", type=int, default=int(SPEC["lora"]["k_steps_primary"]))
    p.add_argument(
        "--force-dataset",
        action="store_true",
        help="Overwrite a frozen dataset. Do not use after scoring that version.",
    )
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


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "smoke").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = log_print(RESULT_DIR / "logs" / f"{args.stage}_{stamp}.log")
    try:
        _main(args, log, stamp)
    finally:
        log._fh.close()  # type: ignore[attr-defined]


def _main(args: argparse.Namespace, log, stamp: str) -> None:
    log(f"Exp 02A stage={args.stage} utc={stamp}")
    log(f"spec.backbone.revision_sha={SPEC['backbone']['revision_sha']}")
    meta_v1 = freeze_dataset_v1(force=False if args.stage != "freeze-dataset" else args.force_dataset)
    log(f"dataset_v1 sha256={meta_v1['sha256']} path={meta_v1['path']}")
    if args.stage == "freeze-dataset":
        log("freeze-dataset complete. Not loading the model. v1 was not rewritten unless --force-dataset.")
        return

    meta_v2 = None
    if args.stage in ("freeze-dataset-v2", "test", "pilot-v2", "pilot-v2-k32"):
        meta_v2 = freeze_dataset_v2(force=args.force_dataset if args.stage == "freeze-dataset-v2" else False)
        log(f"dataset_v2 sha256={meta_v2['sha256']} path={meta_v2['path']}")
        log(f"dataset_v2 source_v1_sha256={meta_v2['source_v1_sha256']}")
    if args.stage == "freeze-dataset-v2":
        log("freeze-dataset-v2 complete. Not loading the model.")
        return

    run_cpu_tests(log)
    if args.stage == "test":
        log("CPU tests complete.")
        return

    if args.stage == "smoke":
        _run_smoke(args, log, stamp, meta_v1)
        return
    if args.stage == "pilot-v2":
        if meta_v2 is None:
            raise RuntimeError("dataset_v2 was not frozen")
        _run_pilot_v2(args, log, stamp, meta_v1, meta_v2)
        return
    if args.stage == "pilot-v2-k32":
        if meta_v2 is None:
            raise RuntimeError("dataset_v2 was not frozen")
        _run_pilot_v2_k32(args, log, stamp, meta_v1, meta_v2)
        return
    raise ValueError(f"unhandled stage {args.stage}")


def _runtime_banner(runtime, log) -> None:
    log(
        f"loaded model_id={runtime.model_id} revision={runtime.revision} "
        f"gpu={runtime.gpu_name} dtype={runtime.torch_dtype} "
        f"trainable={runtime.n_trainable} frozen={runtime.n_frozen}"
    )
    log(f"lora_config={json.dumps(runtime.lora_config)}")


def _env_block(runtime) -> dict:
    import torch

    return {
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
    }


def _run_smoke(args, log, stamp: str, meta_v1: dict) -> None:
    if args.k_steps != 8:
        log(f"WARNING: smoke k_steps={args.k_steps} (protocol primary is 8)")
    from model_runtime import load_runtime

    log("loading Qwen3-4B-Instruct-2507 BF16 + LoRA...")
    runtime = load_runtime()
    _runtime_banner(runtime, log)
    full = load_dataset_v1(smoke=False)
    by_id = {ex["example_id"]: ex for ex in full}
    smoke_examples = [ex for ex in full if ex["smoke"]]
    log(f"smoke n={len(smoke_examples)} (8 exact + 8 semantic) dataset_v1")
    records = run_conditions(
        runtime,
        smoke_examples,
        by_id=by_id,
        k_steps=args.k_steps,
        log_fn=log,
    )
    metrics = condition_metrics(records)
    failures = smoke_invariants(
        runtime,
        smoke_examples,
        records,
        by_id=by_id,
        log_fn=log,
    )
    config_out = {
        "stage": "smoke",
        "utc": stamp,
        "spec": SPEC,
        "design_choices": DESIGN_CHOICES,
        "dataset_sha256": meta_v1["sha256"],
        "dataset_path": str(DATASET_PATH),
        "k_steps": args.k_steps,
        **_env_block(runtime),
        "metrics": metrics,
        "sanity_failures": failures,
        "sanity_passed": len(failures) == 0,
        "next_planned_run": SPEC["pilot_v2_k8"],
        "did_not_run": [
            "dataset_v1 32-per-type semantic_inferential pilot",
            "K=32 escalation",
            "saliency",
            "learned routing",
            "explicit memory",
            "8K/16K/32K retention",
            "continual unrelated-context TTT",
            "TTT-E2E comparison",
        ],
    }
    smoke_dir = RESULT_DIR / "smoke"
    write_jsonl(smoke_dir / "predictions.jsonl", records)
    (smoke_dir / "summary.json").write_text(
        json.dumps(config_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (smoke_dir / "invariants.json").write_text(
        json.dumps({"passed": len(failures) == 0, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"wrote {smoke_dir / 'predictions.jsonl'}")
    log(f"wrote {smoke_dir / 'summary.json'}")
    log("condition metrics:")
    log(json.dumps(metrics, indent=2))
    if failures:
        raise SystemExit(f"smoke sanity failed: {failures}")
    log("smoke complete.")


def _run_pilot_v2(args, log, stamp: str, meta_v1: dict, meta_v2: dict) -> None:
    if args.k_steps != 8:
        raise SystemExit("pilot-v2 is K=8 only; do not pass another K")
    from model_runtime import load_runtime

    log("loading Qwen3-4B-Instruct-2507 BF16 + LoRA for dataset_v2 K=8 pilot...")
    runtime = load_runtime()
    _runtime_banner(runtime, log)
    full = load_dataset_v2(pilot=False)
    by_id = {ex["example_id"]: ex for ex in full}
    examples = [ex for ex in full if ex["info_type"] in ("exact", "semantic_direct")]
    held = [ex for ex in full if ex["info_type"] == "semantic_inferential"]
    log(f"pilot n={len(examples)} (32 exact + 32 semantic_direct); held-out inferential={len(held)}")
    records = run_conditions(
        runtime,
        examples,
        by_id=by_id,
        k_steps=8,
        log_fn=log,
    )
    metrics = condition_metrics(records)
    paired = paired_ttt_effect(records)
    failures = smoke_invariants(
        runtime,
        examples,
        records,
        by_id=by_id,
        log_fn=log,
    )
    config_out = {
        "stage": "pilot_v2_k8",
        "utc": stamp,
        "spec": SPEC,
        "design_choices": DESIGN_CHOICES,
        "dataset_v1_sha256": meta_v1["sha256"],
        "dataset_v2_sha256": meta_v2["sha256"],
        "dataset_v2_path": str(DATASET_V2_PATH),
        "primary_candidate_score": "mean_token_log_likelihood",
        "k_steps": 8,
        **_env_block(runtime),
        "metrics": metrics,
        "paired_ttt_correct_vs_shuffled": paired,
        "source_present_upper_bound": metrics.get("static_source_present"),
        "sanity_failures": failures,
        "sanity_passed": len(failures) == 0,
        "did_not_run": [
            "semantic_inferential",
            "K=32",
            "retention distance / filler",
            "continual unrelated-context TTT",
            "TTT-E2E comparison",
            "architecture changes",
        ],
    }
    out_dir = RESULT_DIR / "pilot_v2_k8"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", records)
    (out_dir / "summary.json").write_text(
        json.dumps(config_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "paired.json").write_text(
        json.dumps(paired, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "invariants.json").write_text(
        json.dumps({"passed": len(failures) == 0, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"wrote {out_dir / 'predictions.jsonl'}")
    log(f"wrote {out_dir / 'summary.json'}")
    log("condition metrics:")
    log(json.dumps(metrics, indent=2))
    log("paired TTT-correct vs TTT-shuffled:")
    slim = {
        t: {k: v for k, v in payload.items() if k != "per_example"}
        for t, payload in paired["by_type"].items()
    }
    log(json.dumps(slim, indent=2))
    if failures:
        raise SystemExit(f"pilot-v2 sanity failed: {failures}")
    log("pilot-v2 K=8 complete. Stop here; do not run K=32.")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_pilot_v2_k32(args, log, stamp: str, meta_v1: dict, meta_v2: dict) -> None:
    if args.k_steps != 32 and args.k_steps != int(SPEC["lora"]["k_steps_primary"]):
        # Default CLI k-steps is 8; this stage ignores it and uses the preregistered 32.
        log(f"pilot-v2-k32 ignoring --k-steps={args.k_steps}; using preregistered K=32")
    k_steps = int(SPEC["lora"]["k_steps_preregistered_escalation"])
    if k_steps != 32:
        raise SystemExit("preregistered escalation K must be 32")
    expected_v2 = "5a9c483e31e1856479369b1b5853972dc822b2a44e622e8156abdb445b776a01"
    if meta_v2["sha256"] != expected_v2:
        raise SystemExit(f"dataset_v2 hash moved: {meta_v2['sha256']} != {expected_v2}")

    k8_dir = RESULT_DIR / "pilot_v2_k8"
    k8_pred = k8_dir / "predictions.jsonl"
    if not k8_pred.exists():
        raise FileNotFoundError(f"need K=8 predictions at {k8_pred}")
    records_k8 = _load_jsonl(k8_pred)

    from model_runtime import load_runtime

    log("loading Qwen3-4B-Instruct-2507 BF16 + LoRA for dataset_v2 K=32 escalation...")
    log("only experimental variable: K=8 -> K=32. TTT conditions only.")
    runtime = load_runtime()
    _runtime_banner(runtime, log)
    full = load_dataset_v2(pilot=False)
    by_id = {ex["example_id"]: ex for ex in full}
    examples = [ex for ex in full if ex["info_type"] in ("exact", "semantic_direct")]
    log(f"pilot-v2-k32 n={len(examples)} conditions={list(TTT_CONDITIONS)} k={k_steps}")
    records = run_conditions(
        runtime,
        examples,
        by_id=by_id,
        k_steps=k_steps,
        log_fn=log,
        conditions=TTT_CONDITIONS,
    )
    metrics = condition_metrics(records, conditions=TTT_CONDITIONS)
    paired = paired_ttt_effect(records)
    failures = smoke_invariants(
        runtime,
        examples,
        records,
        by_id=by_id,
        log_fn=log,
        require_source_present=False,
    )
    cross = cross_k_memory_table(records_k8, records)
    k8_summary = json.loads((k8_dir / "summary.json").read_text(encoding="utf-8"))
    config_out = {
        "stage": "pilot_v2_k32",
        "utc": stamp,
        "spec": SPEC,
        "design_choices": DESIGN_CHOICES,
        "dataset_v1_sha256": meta_v1["sha256"],
        "dataset_v2_sha256": meta_v2["sha256"],
        "dataset_v2_path": str(DATASET_V2_PATH),
        "primary_cross_k_metric": "per-token gold-answer conditional NLL; memory_nll_gain = shuffled - correct",
        "only_changed_variable": "k_steps",
        "k_steps": k_steps,
        "k8_reference": {
            "path": str(k8_pred),
            "dataset_v2_sha256": k8_summary.get("dataset_v2_sha256"),
            "reused_conditions": ["static_source_free", "static_source_present"],
        },
        **_env_block(runtime),
        "metrics": metrics,
        "paired_ttt_correct_vs_shuffled": paired,
        "cross_k": cross,
        "source_present_upper_bound_from_k8": k8_summary.get("source_present_upper_bound"),
        "sanity_failures": failures,
        "sanity_passed": len(failures) == 0,
        "did_not_run": [
            "static_source_free (reused k8)",
            "static_source_present (reused k8)",
            "semantic_inferential",
            "retention distance / filler",
            "continual unrelated-context TTT",
            "TTT-E2E comparison",
            "new hyperparameters",
        ],
    }
    out_dir = RESULT_DIR / "pilot_v2_k32"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", records)
    (out_dir / "summary.json").write_text(
        json.dumps(config_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "paired.json").write_text(
        json.dumps(paired, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "k8_vs_k32_table.json").write_text(
        json.dumps(cross, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "invariants.json").write_text(
        json.dumps({"passed": len(failures) == 0, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"wrote {out_dir / 'predictions.jsonl'}")
    log(f"wrote {out_dir / 'k8_vs_k32_table.json'}")
    log("K=8 vs K=32 table:")
    log(json.dumps(cross["table"], indent=2))
    log("bootstrap:")
    boot = {
        t: {
            "memory_nll_gain_K32": payload["bootstrap_memory_nll_gain_k32"],
            "memory_nll_gain_K32_minus_K8": payload["bootstrap_memory_nll_gain_k32_minus_k8"],
        }
        for t, payload in cross["by_type"].items()
    }
    log(json.dumps(boot, indent=2))
    if failures:
        raise SystemExit(f"pilot-v2-k32 sanity failed: {failures}")
    log("pilot-v2 K=32 complete. Stop here.")


if __name__ == "__main__":
    main()
