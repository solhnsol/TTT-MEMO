"""TTT-MEMO Exp 02A-2: source-free semantic binding on frozen dataset_v3.

  python scripts/exp02a2_semantic_v3/run_exp02a2.py --stage test
  python scripts/exp02a2_semantic_v3/run_exp02a2.py --stage k32

K=32 NTP-LoRA. Description-binding and semantic-label probes. No retention,
paraphrase robustness, K=8, or TTT-E2E.
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

from dataset import DATASET_PATH, MANIFEST_PATH, sha256_bytes  # noqa: E402
from evaluate_memory import (  # noqa: E402
    flatten_matrices,
    flatten_predictions,
    pair_binding_table,
    render_summary_md,
    run_invariants,
    run_pairs,
    write_jsonl,
)
from memory_protocol import DESIGN_CHOICES, EXPECTED_DATASET_V3_SHA256, SPEC  # noqa: E402

RESULT_DIR = _REPO_ROOT / "experiments" / "exp02a2_semantic_v3" / "source_free_k32"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("test", "k32"), default="k32")
    p.add_argument("--k-steps", type=int, default=int(SPEC["lora"]["k_steps"]))
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


def load_frozen_v3() -> tuple[list[dict], str]:
    if not DATASET_PATH.exists():
        raise RuntimeError(f"frozen dataset_v3 missing: {DATASET_PATH}")
    digest = sha256_bytes(DATASET_PATH.read_bytes())
    if digest != EXPECTED_DATASET_V3_SHA256:
        raise RuntimeError(
            f"dataset_v3 sha256 mismatch: got {digest}, expected {EXPECTED_DATASET_V3_SHA256}"
        )
    pairs = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [p["pair_id"] for p in pairs] != ["P1", "P2", "P3", "P4"]:
        raise RuntimeError(f"unexpected pair ids: {[p['pair_id'] for p in pairs]}")
    return pairs, digest


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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = log_print(RESULT_DIR / "logs" / f"{args.stage}_{stamp}.log")
    try:
        _main(args, log, stamp)
    finally:
        log._fh.close()  # type: ignore[attr-defined]


def _main(args: argparse.Namespace, log, stamp: str) -> None:
    log(f"Exp 02A-2 source-free stage={args.stage} utc={stamp}")
    log(f"spec.backbone.revision_sha={SPEC['backbone']['revision_sha']}")
    pairs, digest = load_frozen_v3()
    log(f"frozen dataset_v3 sha256={digest} n_pairs={len(pairs)}")
    run_cpu_tests(log)
    if args.stage == "test":
        log("CPU tests complete.")
        return
    if args.stage != "k32":
        raise ValueError(f"unhandled stage {args.stage}")
    if args.k_steps != 32:
        raise RuntimeError(f"this stage must use K=32, got {args.k_steps}")
    _run_k32(log, stamp, pairs, digest)


def _run_k32(log, stamp: str, pairs: list[dict], digest: str) -> None:
    from model_runtime import load_runtime

    shutil.copy2(DATASET_PATH, RESULT_DIR / "dataset_v3.jsonl")
    shutil.copy2(MANIFEST_PATH, RESULT_DIR / "dataset_v3_manifest.json")
    config = {
        "spec": SPEC,
        "design_choices": DESIGN_CHOICES,
        "dataset_sha256": digest,
        "k_steps": 32,
        "stage": "k32",
        "n_pairs": 4,
        "utc": stamp,
        "did_not_run": SPEC["not_this_experiment"],
    }
    (RESULT_DIR / "config.yaml").write_text(to_yaml(config) + "\n", encoding="utf-8")

    log("loading Qwen3-4B-Instruct-2507 BF16 + LoRA for Exp 02A-2 K=32...")
    runtime = load_runtime()
    log(
        f"loaded model_id={runtime.model_id} revision={runtime.revision} "
        f"gpu={runtime.gpu_name} dtype={runtime.torch_dtype} "
        f"trainable={runtime.n_trainable} frozen={runtime.n_frozen}"
    )
    log(f"lora_config={json.dumps(runtime.lora_config)}")
    records = run_pairs(runtime, pairs, k_steps=32, log_fn=log)
    failures = run_invariants(runtime, pairs, records, log_fn=log)
    table = pair_binding_table(records, pairs)

    write_jsonl(RESULT_DIR / "predictions.jsonl", flatten_predictions(records))
    write_jsonl(RESULT_DIR / "score_matrices.jsonl", flatten_matrices(records))
    (RESULT_DIR / "invariants.json").write_text(
        json.dumps({"passed": not failures, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )

    write_traj = []
    for rec in records:
        if rec.get("write"):
            write_traj.append(
                {
                    "pair_id": rec["pair_id"],
                    "condition": rec["condition"],
                    "mapping": rec["mapping"],
                    "loss_trajectory": rec["write"]["loss_trajectory"],
                    "token_weighted_ce_trajectory": rec["write"]["token_weighted_ce_trajectory"],
                    "loss_before": rec["write"]["loss_before"],
                    "loss_after": rec["write"]["loss_after"],
                    "lora_l2_delta_from_init": rec["write"]["lora_l2_delta_from_init"],
                    "lora_max_abs_diff_from_init": rec["write"]["lora_max_abs_diff_from_init"],
                    "write_n_labeled_tokens_per_sequence": rec["write"]["write_n_labeled_tokens_per_sequence"],
                }
            )
    write_jsonl(RESULT_DIR / "write_loss_trajectories.jsonl", write_traj)

    import torch

    summary = {
        "stage": "k32",
        "utc": stamp,
        "spec": SPEC,
        "design_choices": DESIGN_CHOICES,
        "dataset_sha256": digest,
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
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = render_summary_md(table, dataset_sha=digest, n_pairs=len(pairs))
    (RESULT_DIR / "summary.md").write_text(md, encoding="utf-8")
    log(md)
    if failures:
        raise RuntimeError("exp02a2 source-free sanity failed:\n" + "\n".join(failures))
    log("Exp 02A-2 K=32 source-free complete. Stop here.")


if __name__ == "__main__":
    main()
