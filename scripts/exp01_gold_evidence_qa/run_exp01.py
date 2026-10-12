"""TTT-MEMO Exp 01: gold evidence inside the TTT-E2E local window.

Stages (run in order):
  python scripts/exp01_gold_evidence_qa/run_exp01.py --stage smoke
  python scripts/exp01_gold_evidence_qa/run_exp01.py --stage prompt-dev
  python scripts/exp01_gold_evidence_qa/run_exp01.py --stage main

`--stage all` runs smoke, then prompt-dev, then main, and stops if smoke fails.

Does not modify `e2e/`. Reuses `scripts/baseline_125m/common/` for Hydra
compose, checkpoint restore, GPU/attention helpers, and pytree diffs.
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
_NVRTC_DIR = _REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cuda_nvrtc" / "lib"
if _NVRTC_DIR.is_dir():
    current = os.environ.get("LD_LIBRARY_PATH", "")
    nvrtc = str(_NVRTC_DIR)
    if nvrtc not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = nvrtc + (f":{current}" if current else "")

sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "baseline_125m" / "common"))

from baseline_config import TOKENIZER_PATH  # noqa: E402
from evaluate import (  # noqa: E402
    CONDITIONS,
    condition_metrics,
    length_breakdowns,
    paired_bootstrap_ci,
    run_conditions,
    sample_failures,
    select_prompt_template,
    smoke_invariants,
    write_failures_md,
    write_jsonl,
    write_summary_md,
)
from generation import MAX_NEW_TOKENS, make_greedy_generate_fn  # noqa: E402
from model_runtime import load_official_125m, load_tokenizer, loaded_config_record  # noqa: E402
from prompts import CANDIDATE_TEMPLATES, TEMPLATE_BY_ID  # noqa: E402
from squad_preprocess import DEFAULT_SEED, MAIN_EVAL_N, PROMPT_DEV_N, prepare_squad  # noqa: E402

RESULT_DIR = _REPO_ROOT / "experiments" / "exp01_gold_evidence_qa"
SQUAD_PATH = _REPO_ROOT / "data" / "squad" / "dev-v1.1.json"


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPO_ROOT,
                text=True,
                timeout=10,
            )
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("smoke", "prompt-dev", "main", "all"), default="all")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--prompt-dev-n", type=int, default=PROMPT_DEV_N)
    p.add_argument("--main-eval-n", type=int, default=MAIN_EVAL_N)
    p.add_argument("--smoke-n", type=int, default=16)
    p.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="XLA fallback. Default is official cuDNN prefix SWA (A100 main backend).",
    )
    p.add_argument("--force", action="store_true", help="Allow overwriting main predictions.jsonl")
    return p.parse_args()


def ensure_dirs() -> None:
    (RESULT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "smoke").mkdir(parents=True, exist_ok=True)


def log_print(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a", encoding="utf-8")

    def _log(msg: str = "") -> None:
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    _log._fh = fh  # type: ignore[attr-defined]
    return _log


def save_config(path: Path, payload: dict) -> None:
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.create(payload), path)


def aligned_scores(records: list[dict], cond_a: str, cond_b: str, field: str) -> tuple[list[float], list[float]]:
    by_id = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        by_id.setdefault(r["id"], {})[r["condition"]] = r
    a, b = [], []
    for _id, row in sorted(by_id.items()):
        if cond_a in row and cond_b in row:
            a.append(float(row[cond_a][field]))
            b.append(float(row[cond_b][field]))
    return a, b


def main() -> None:
    args = parse_args()
    ensure_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = log_print(RESULT_DIR / "logs" / f"{args.stage}_{stamp}.log")
    try:
        _main(args, log, stamp)
    finally:
        log._fh.close()  # type: ignore[attr-defined]


def _main(args: argparse.Namespace, log, stamp: str) -> None:
    log(f"Exp 01 stage={args.stage} utc={stamp}")
    tokenizer = load_tokenizer()
    log(f"tokenizer: {TOKENIZER_PATH}")
    log("preparing SQuAD 1.1 validation subsets...")
    prepared = prepare_squad(
        squad_path=SQUAD_PATH,
        tokenizer=tokenizer,
        seed=args.seed,
        prompt_dev_n=args.prompt_dev_n,
        main_eval_n=args.main_eval_n,
    )
    splits = prepared["splits"]
    log(
        f"SQuAD val: total={prepared['n_total']} ok={prepared['n_ok']} "
        f"preprocess_error={prepared['n_preprocess_error']}"
    )
    log(f"prompt-dev={len(splits['prompt_dev'])} main-eval={len(splits['main_eval'])} smoke={args.smoke_n}")

    preprocess_meta = {
        "squad_path": str(SQUAD_PATH),
        "seed": args.seed,
        "n_total": prepared["n_total"],
        "n_ok": prepared["n_ok"],
        "n_preprocess_error": prepared["n_preprocess_error"],
        "n_prompt_dev": len(splits["prompt_dev"]),
        "n_main_eval": len(splits["main_eval"]),
        "prompt_dev_ids": prepared["prompt_dev_ids"],
        "main_eval_ids": prepared["main_eval_ids"],
        "smoke_ids": [ex.id for ex in splits["prompt_dev"][: args.smoke_n]],
        "filtering": "answerable SQuAD 1.1 with locatable contiguous gold span; no model-output filter",
    }
    (RESULT_DIR / "preprocess_meta.json").write_text(
        json.dumps(preprocess_meta, indent=2), encoding="utf-8"
    )

    log("loading official 125M checkpoint...")
    loaded = load_official_125m(attention_cudnn_override=args.attention_cudnn_override)
    cfg_rec = loaded_config_record(loaded)
    log(f"checkpoint: {cfg_rec['checkpoint_path']}")
    log(f"SWA={cfg_rec['sliding_window_size']} mini_batch={cfg_rec['mini_batch_size']}")
    log(f"spec_inner={cfg_rec['spec_inner']}")
    log(f"backend={cfg_rec['backend']['note']} gpu={cfg_rec['backend']['gpu_name']}")
    log(f"fast_weight_hash={cfg_rec['fast_weight_hash_at_load']}")

    greedy_fn = make_greedy_generate_fn(loaded, max_new_tokens=MAX_NEW_TOKENS)

    stages = []
    if args.stage == "all":
        stages = ["smoke", "prompt-dev", "main"]
    else:
        stages = [args.stage]

    selected_id = None
    prompt_sel = None
        if (RESULT_DIR / "prompt_dev_results.json").exists() and "prompt-dev" not in stages:
            prev = json.loads((RESULT_DIR / "prompt_dev_results.json").read_text(encoding="utf-8"))
            selected_id = prev.get("selected")
            prompt_sel = {
                "selected": prev.get("selected"),
                "table": prev.get("table"),
                "rule": prev.get("rule"),
            }
            log(f"reusing frozen prompt template {selected_id}")

    if "smoke" in stages:
        smoke_examples = splits["prompt_dev"][: args.smoke_n]
        smoke_template = TEMPLATE_BY_ID["T1"]
        log(f"\n--- smoke: {len(smoke_examples)} examples, template T1, all 3 conditions ---")
        smoke_records = run_conditions(
            smoke_examples,
            conditions=CONDITIONS,
            template=smoke_template,
            tokenizer=tokenizer,
            loaded=loaded,
            greedy_fn=greedy_fn,
            log_fn=log,
        )
        write_jsonl(RESULT_DIR / "smoke" / "predictions.jsonl", smoke_records)
        failures = smoke_invariants(
            smoke_examples,
            smoke_records,
            tokenizer=tokenizer,
            template=smoke_template,
            loaded=loaded,
            greedy_fn=greedy_fn,
            log_fn=log,
        )
        (RESULT_DIR / "smoke" / "sanity.json").write_text(
            json.dumps({"passed": not failures, "failures": failures}, indent=2),
            encoding="utf-8",
        )
        if failures:
            raise SystemExit("smoke sanity checks failed; not continuing")

    if "prompt-dev" in stages:
        log("\n--- prompt-dev: 3 templates × gold_sentence only ---")
        by_template = {}
        for tmpl in CANDIDATE_TEMPLATES:
            log(f"template {tmpl.template_id}")
            recs = run_conditions(
                splits["prompt_dev"],
                conditions=("gold_sentence",),
                template=tmpl,
                tokenizer=tokenizer,
                loaded=loaded,
                greedy_fn=greedy_fn,
                log_fn=log,
            )
            by_template[tmpl.template_id] = recs
        prompt_sel = select_prompt_template(by_template)
        selected_id = prompt_sel["selected"]
        log(f"FROZEN prompt template: {selected_id}")
        payload = {
            **prompt_sel,
            "records": {tid: recs for tid, recs in by_template.items()},
        }
        (RESULT_DIR / "prompt_dev_results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if "main" in stages:
        if selected_id is None:
            raise SystemExit("main requires a frozen prompt template; run prompt-dev first")
        pred_path = RESULT_DIR / "predictions.jsonl"
        if pred_path.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {pred_path} (pass --force)")
        template = TEMPLATE_BY_ID[selected_id]
        log(f"\n--- main eval: {len(splits['main_eval'])} examples × 3 conditions, template {selected_id} ---")
        records = run_conditions(
            splits["main_eval"],
            conditions=CONDITIONS,
            template=template,
            tokenizer=tokenizer,
            loaded=loaded,
            greedy_fn=greedy_fn,
            log_fn=log,
        )
        write_jsonl(pred_path, records)
        metrics = condition_metrics(records)
        sent_em, none_em = aligned_scores(records, "gold_sentence", "no_evidence", "em")
        sent_f1, none_f1 = aligned_scores(records, "gold_sentence", "no_evidence", "f1")
        para_em, none_em_p = aligned_scores(records, "gold_paragraph", "no_evidence", "em")
        para_f1, none_f1_p = aligned_scores(records, "gold_paragraph", "no_evidence", "f1")
        deltas = {
            "sentence_em": paired_bootstrap_ci(sent_em, none_em),
            "sentence_f1": paired_bootstrap_ci(sent_f1, none_f1),
            "paragraph_em": paired_bootstrap_ci(para_em, none_em_p),
            "paragraph_f1": paired_bootstrap_ci(para_f1, none_f1_p),
        }
        breakdowns = length_breakdowns(records)
        fail_sample = sample_failures(records, n=30, seed=args.seed)
        write_failures_md(RESULT_DIR / "failures.md", fail_sample)
        n_eval = metrics["gold_sentence"]["n"]
        config_payload = {
            **cfg_rec,
            "dataset": "squad_v1.1",
            "split": "validation",
            "seed": args.seed,
            "prompt_dev_n": args.prompt_dev_n,
            "main_eval_n": args.main_eval_n,
            "prompt_dev_ids": prepared["prompt_dev_ids"],
            "main_eval_ids": prepared["main_eval_ids"],
            "filtering_rules": preprocess_meta["filtering"],
            "prompt_templates": {t.template_id: t.template for t in CANDIDATE_TEMPLATES},
            "selected_template": selected_id,
            "generation": {
                "decoding": "greedy_argmax",
                "temperature": 0,
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "pad_token": "eos",
                "pad_to": loaded.mini_batch_size,
                "scored_prediction_rule": "decode until EOS; take text before first newline; strip",
            },
            "git_commit": git_commit(),
            "e2e_submodule_commit": "a4fc4788ace38e29b5067916d4f4be33da894085",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ttt_memo_note": (
                "One-chunk (T=1024) official logits are pre-inner-update. "
                "This experiment tests in-window backbone use of oracle evidence, not TTT memory."
            ),
        }
        save_config(RESULT_DIR / "config.yaml", config_payload)
        summary = {
            "n_evaluated": n_eval,
            "metrics": metrics,
            "deltas": deltas,
            "prompt_selection": prompt_sel,
            "breakdowns": breakdowns,
            "n_errors": sum(1 for r in records if r.get("status") != "ok"),
        }
        (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_summary_md(
            RESULT_DIR / "summary.md",
            n_eval=n_eval,
            metrics=metrics,
            deltas=deltas,
            prompt_sel=prompt_sel,
            breakdowns=breakdowns,
            failures=fail_sample,
            config=config_payload,
            preprocess=preprocess_meta,
        )
        log("\n" + (RESULT_DIR / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
