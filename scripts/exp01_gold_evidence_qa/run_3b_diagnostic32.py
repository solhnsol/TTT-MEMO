"""32-example 3B diagnostic for Exp 01 gold-sentence vs no-evidence.

Uses the first 32 IDs of the saved 125M main-eval split and the already-stored
gold_sentence / no_evidence injected contexts. Frozen prompt T3. Does not
modify e2e/, does not overwrite 125M Exp 01 artifacts, does not run 256.

Measures local SWA + initialized prime FFN (one TTT chunk, logits before the
intra-chunk fast-weight update). No second post-update forward.
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

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "baseline_125m" / "common"))

from evaluate import (  # noqa: E402
    evaluate_example,
    mean,
    paired_bootstrap_ci,
    verify_fast_weights,
    write_jsonl,
)
from generation import MAX_NEW_TOKENS, encode_prompt, make_greedy_generate_fn  # noqa: E402
from model_runtime import (  # noqa: E402
    load_official_3b,
    load_tokenizer,
    loaded_config_record,
    snapshot_fast_weights,
)
from prompts import TEMPLATE_BY_ID, render_prompt  # noqa: E402
from squad_metrics import normalize_answer  # noqa: E402
from squad_preprocess import SquadExample  # noqa: E402

EXP01_125M_DIR = _REPO_ROOT / "experiments" / "exp01_gold_evidence_qa"
RESULT_DIR = EXP01_125M_DIR / "3b_diagnostic_32"
DIAG_N = 32
DIAG_CONDITIONS = ("gold_sentence", "no_evidence")
TEMPLATE_ID = "T3"


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            timeout=10,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def nvidia_smi_used_mib() -> int | None:
    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        return int(line.strip())
    except Exception:
        return None


def jax_memory_stats():
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
        if not stats:
            return None
        return {str(k): int(v) if isinstance(v, int) else v for k, v in stats.items()}
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=DIAG_N)
    p.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=False,
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


def save_config(path: Path, payload: dict) -> None:
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.create(payload), path)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_n_main_ids(n: int) -> list[str]:
    meta = json.loads((EXP01_125M_DIR / "preprocess_meta.json").read_text(encoding="utf-8"))
    ids = list(meta["main_eval_ids"])
    if len(ids) < n:
        raise SystemExit(f"saved main_eval_ids has {len(ids)} < {n}")
    selected = ids[:n]
    pred_ids = []
    seen = set()
    for row in load_jsonl(EXP01_125M_DIR / "predictions.jsonl"):
        if row["id"] not in seen:
            pred_ids.append(row["id"])
            seen.add(row["id"])
    if pred_ids[:n] != selected:
        raise SystemExit("predictions.jsonl ID order does not match preprocess_meta.json main_eval_ids")
    prompt_dev = set(meta["prompt_dev_ids"])
    overlap = [i for i in selected if i in prompt_dev]
    if overlap:
        raise SystemExit(f"diagnostic IDs overlap prompt-dev: {overlap}")
    return selected


def examples_from_stored_125m(ids: list[str]) -> tuple[list[SquadExample], dict[str, dict[str, dict]]]:
    by_id: dict[str, dict[str, dict]] = {}
    for row in load_jsonl(EXP01_125M_DIR / "predictions.jsonl"):
        if row["id"] in ids:
            by_id.setdefault(row["id"], {})[row["condition"]] = row
    examples = []
    for eid in ids:
        rowset = by_id.get(eid)
        if rowset is None:
            raise SystemExit(f"id {eid} missing from 125M predictions.jsonl")
        for cond in DIAG_CONDITIONS:
            if cond not in rowset:
                raise SystemExit(f"id {eid} missing stored condition {cond}")
            if rowset[cond].get("status") != "ok":
                raise SystemExit(f"id {eid} {cond} stored status={rowset[cond].get('status')}")
        gs = rowset["gold_sentence"]
        ne = rowset["no_evidence"]
        examples.append(
            SquadExample(
                id=eid,
                title="",
                question=gs["question"],
                gold_paragraph=gs.get("gold_paragraph") or "",
                gold_answers=list(gs["gold_answers"]),
                answer_starts=[],
                gold_sentence=gs["injected_context"],
                gold_sentence_char_span=None,
                no_evidence_source_id=ne.get("no_evidence_source_id"),
                no_evidence_context=ne["injected_context"],
                status="ok",
                split="main_eval",
            )
        )
    return examples, by_id


def aligned_scores(records: list[dict], cond_a: str, cond_b: str, field: str) -> tuple[list[float], list[float]]:
    by_id: dict[str, dict] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        by_id.setdefault(r["id"], {})[r["condition"]] = r
    a, b = [], []
    for _id in sorted(by_id):
        row = by_id[_id]
        if cond_a in row and cond_b in row:
            a.append(float(row[cond_a][field]))
            b.append(float(row[cond_b][field]))
    return a, b


def condition_metrics_two(records: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for condition in DIAG_CONDITIONS:
        rows = [r for r in records if r.get("condition") == condition and r.get("status") == "ok"]
        out[condition] = {
            "n": len(rows),
            "em": mean([r["em"] for r in rows]),
            "f1": mean([r["f1"] for r in rows]),
        }
    return out


def metrics_from_125m_subset(by_id: dict[str, dict[str, dict]]) -> dict[str, dict[str, float]]:
    out = {}
    for condition in DIAG_CONDITIONS:
        rows = [by_id[eid][condition] for eid in by_id]
        out[condition] = {
            "n": len(rows),
            "em": mean([r["em"] for r in rows]),
            "f1": mean([r["f1"] for r in rows]),
        }
    return out


def hit_max_new_tokens(record: dict, eos_id: int) -> bool:
    ids = record.get("generated_token_ids") or []
    return len(ids) == MAX_NEW_TOKENS and eos_id not in ids


def gold_substring_hit(prediction: str, gold_answers: list[str]) -> bool:
    return any(g and g in (prediction or "") for g in gold_answers)


def preflight_encodings(examples: list[SquadExample], stored: dict, tokenizer, template, loaded_like) -> dict:
    """Tokenize all diagnostic prompts before GPU work. Must stay inside one TTT chunk."""
    rows = []
    prompt_text_mismatches = []
    for ex in examples:
        for condition in DIAG_CONDITIONS:
            context = ex.gold_sentence if condition == "gold_sentence" else ex.no_evidence_context
            rendered = render_prompt(template, context=context, question=ex.question)
            stored_text = stored[ex.id][condition].get("prompt_text")
            if stored_text != rendered:
                prompt_text_mismatches.append(
                    {"id": ex.id, "condition": condition, "stored_len": len(stored_text or ""), "rendered_len": len(rendered)}
                )
            encoded = encode_prompt(
                tokenizer,
                template,
                context=context,
                question=ex.question,
                bos_id=loaded_like.bos_id,
                eos_id=loaded_like.eos_id,
                chunk_size=loaded_like.mini_batch_size,
                swa_window=loaded_like.swa_window,
            )
            budget = encoded.prompt_len + MAX_NEW_TOKENS
            rows.append(
                {
                    "id": ex.id,
                    "condition": condition,
                    "prompt_len": encoded.prompt_len,
                    "prompt_plus_max_new": budget,
                    "evidence_token_length": encoded.evidence_token_length,
                    "evidence_question_distance": encoded.evidence_question_distance,
                    "padded_to": len(encoded.padded_ids),
                    "fits_one_chunk": budget <= loaded_like.mini_batch_size,
                    "in_swa": encoded.evidence_question_distance < loaded_like.swa_window,
                }
            )
            if budget > loaded_like.mini_batch_size:
                raise SystemExit(
                    f"{ex.id} {condition}: prompt_len {encoded.prompt_len} + {MAX_NEW_TOKENS} "
                    f"exceeds mini_batch_size {loaded_like.mini_batch_size}"
                )
    prompt_lens = [r["prompt_len"] for r in rows]
    return {
        "n_prompts": len(rows),
        "prompt_len_min": min(prompt_lens),
        "prompt_len_max": max(prompt_lens),
        "prompt_plus_max_new_max": max(r["prompt_plus_max_new"] for r in rows),
        "all_fit_one_chunk": all(r["fits_one_chunk"] for r in rows),
        "all_in_swa": all(r["in_swa"] for r in rows),
        "prompt_text_mismatches_vs_125m": prompt_text_mismatches,
        "rows": rows,
    }


def inspect_gold_sentence_failures(records: list[dict], eos_id: int) -> dict:
    fails = [
        r
        for r in records
        if r.get("condition") == "gold_sentence" and r.get("status") == "ok" and int(r.get("em") or 0) == 0
    ]
    details = []
    n_substr = 0
    n_empty = 0
    n_max = 0
    for r in fails:
        scored = r.get("scored_prediction") or ""
        raw = r.get("raw_prediction") or ""
        golds = list(r.get("gold_answers") or [])
        substr = gold_substring_hit(scored, golds)
        empty = scored.strip() == ""
        maxed = hit_max_new_tokens(r, eos_id)
        if substr:
            n_substr += 1
        if empty:
            n_empty += 1
        if maxed:
            n_max += 1
        details.append(
            {
                "id": r["id"],
                "question": r.get("question"),
                "gold_sentence": r.get("gold_sentence") or r.get("injected_context"),
                "gold_answers": golds,
                "raw_prediction": raw,
                "scored_prediction": scored,
                "em": r.get("em"),
                "f1": r.get("f1"),
                "exact_gold_substring_in_scored": substr,
                "empty_scored": empty,
                "hit_max_new_tokens": maxed,
                "generated_token_ids": r.get("generated_token_ids"),
            }
        )
    return {
        "n_gold_sentence_em0": len(fails),
        "n_exact_gold_substring_in_scored_when_em0": n_substr,
        "n_empty_scored_when_em0": n_empty,
        "n_hit_max_new_tokens_when_em0": n_max,
        "failures": details,
    }


def representative_rows(records: list[dict], *, k: int = 4) -> dict[str, list[dict]]:
    gs_ok = [r for r in records if r.get("condition") == "gold_sentence" and r.get("status") == "ok"]
    hits = [r for r in gs_ok if int(r.get("em") or 0) == 1]
    misses = sorted(
        [r for r in gs_ok if int(r.get("em") or 0) == 0],
        key=lambda r: (r.get("f1") or 0.0, r["id"]),
    )
    slim = lambda rows: [  # noqa: E731
        {
            "id": r["id"],
            "question": r.get("question"),
            "gold_answers": r.get("gold_answers"),
            "scored_prediction": r.get("scored_prediction"),
            "raw_prediction": r.get("raw_prediction"),
            "em": r.get("em"),
            "f1": r.get("f1"),
        }
        for r in rows
    ]
    return {"em1": slim(hits[:k]), "em0_lowest_f1": slim(misses[:k])}


def write_summary_md(
    path: Path,
    *,
    n: int,
    ids: list[str],
    metrics_3b: dict,
    metrics_125m: dict,
    deltas_3b: dict,
    preflight: dict,
    failure_inspect: dict,
    examples_rep: dict,
    config: dict,
    memory: dict,
    state: dict,
) -> None:
    gs = metrics_3b["gold_sentence"]
    ne = metrics_3b["no_evidence"]
    g125 = metrics_125m["gold_sentence"]
    n125 = metrics_125m["no_evidence"]
    pct = lambda x: 100.0 * x  # noqa: E731
    lines = [
        "# TTT-MEMO Exp 01 — 3B diagnostic (N=32)",
        "",
        "Apples-to-apples with the completed 125M Exp 01 main split.",
        "First 32 saved main-eval IDs. Frozen template T3. Stored 125M contexts.",
        "Not a full Exp 01 rerun. No GO/NO-GO.",
        "",
        "Checkpoints differ in scale **and** training/extension setup.",
        "Do not attribute 125M vs 3B differences solely to parameter count.",
        "",
        "## Setup",
        "",
        f"- 3B checkpoint: `{config.get('checkpoint_path')}`",
        f"- Hydra: `{config.get('hydra_experiment')}`",
        f"- RoPE seq_len: {config.get('model_seq_len_rope_table')} (not prompt length)",
        f"- SWA: {config.get('sliding_window_size')}  TTT chunk: {config.get('mini_batch_size')}",
        f"- prefix/suffix: {config.get('prefix_len')} / {config.get('suffix_len')}",
        f"- template: {TEMPLATE_ID}",
        f"- generation: greedy argmax, max_new_tokens={MAX_NEW_TOKENS}, pad to one chunk with EOS",
        f"- N={n} IDs from saved main_eval ordering; prompt-dev unused",
        "",
        "One-chunk official logits are computed before that chunk's fast-weight update.",
        "This diagnostic measures local SWA + initialized prime FFN, not intra-prompt TTT memory.",
        "",
        "## Preflight token lengths",
        "",
        f"- prompt_len min/max: {preflight['prompt_len_min']} / {preflight['prompt_len_max']}",
        f"- prompt_len + max_new_tokens max: {preflight['prompt_plus_max_new_max']} (chunk={config.get('mini_batch_size')})",
        f"- all fit one TTT chunk: {preflight['all_fit_one_chunk']}",
        f"- all evidence-to-query distances inside SWA: {preflight['all_in_swa']}",
        f"- T3+stored-context prompt_text mismatches vs 125M: {len(preflight['prompt_text_mismatches_vs_125m'])}",
        "",
        "## 3B metrics",
        "",
        "| condition | N | EM | F1 |",
        "|---|---:|---:|---:|",
        f"| gold_sentence | {gs['n']} | {pct(gs['em']):.2f} | {pct(gs['f1']):.2f} |",
        f"| no_evidence | {ne['n']} | {pct(ne['em']):.2f} | {pct(ne['f1']):.2f} |",
        "",
        f"- Delta_sentence_EM = {pct(deltas_3b['sentence_em']['mean_delta']):.2f}",
        f"- Delta_sentence_F1 = {pct(deltas_3b['sentence_f1']['mean_delta']):.2f}",
        "",
        "## Paired comparison, same 32 IDs",
        "",
        "| model | condition | EM | F1 |",
        "|---|---|---:|---:|",
        f"| 125M | gold_sentence | {pct(g125['em']):.2f} | {pct(g125['f1']):.2f} |",
        f"| 125M | no_evidence | {pct(n125['em']):.2f} | {pct(n125['f1']):.2f} |",
        f"| 3B | gold_sentence | {pct(gs['em']):.2f} | {pct(gs['f1']):.2f} |",
        f"| 3B | no_evidence | {pct(ne['em']):.2f} | {pct(ne['f1']):.2f} |",
        "",
        "## Gold-sentence failure diagnostics (3B, scoring unchanged)",
        "",
        f"- EM=0: {failure_inspect['n_gold_sentence_em0']} / {gs['n']}",
        f"- exact gold answer substring in scored_prediction (among EM=0): {failure_inspect['n_exact_gold_substring_in_scored_when_em0']}",
        f"- empty scored_prediction (among EM=0): {failure_inspect['n_empty_scored_when_em0']}",
        f"- hit max_new_tokens with no EOS (among EM=0): {failure_inspect['n_hit_max_new_tokens_when_em0']}",
        "",
        "## Fast-weight state",
        "",
        f"- hash at load: `{state.get('hash_at_load')}`",
        f"- hash at end: `{state.get('hash_at_end')}`",
        f"- max_abs_diff after all examples: {state.get('max_abs_diff_end')}",
        "",
        "## GPU memory",
        "",
        f"- nvidia-smi after restore: {memory.get('nvidia_smi_used_mib_after_restore')} MiB",
        f"- nvidia-smi after first generation: {memory.get('nvidia_smi_used_mib_after_first_gen')} MiB",
        f"- nvidia-smi after all: {memory.get('nvidia_smi_used_mib_after_all')} MiB",
        f"- JAX peak_bytes_in_use: {memory.get('jax_peak_bytes_in_use')}",
        "",
        "## Representative 3B gold_sentence predictions",
        "",
        "### EM = 1",
        "",
    ]
    for r in examples_rep["em1"]:
        lines.append(
            f"- `{r['id']}` scored={r['scored_prediction']!r} gold={r['gold_answers']!r} Q={r['question']!r}"
        )
    if not examples_rep["em1"]:
        lines.append("- (none)")
    lines.extend(["", "### EM = 0 (lowest F1)", ""])
    for r in examples_rep["em0_lowest_f1"]:
        lines.append(
            f"- `{r['id']}` F1={r['f1']:.3f} scored={r['scored_prediction']!r} gold={r['gold_answers']!r} Q={r['question']!r}"
        )
    lines.extend(
        [
            "",
            "## IDs",
            "",
            ", ".join(f"`{i}`" for i in ids),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


class _TokCfg:
    def __init__(self, bos_id, eos_id, mini_batch_size, swa_window):
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.mini_batch_size = mini_batch_size
        self.swa_window = swa_window


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = log_print(RESULT_DIR / "logs" / f"diagnostic_{stamp}.log")
    try:
        _main(args, log, stamp)
    finally:
        log._fh.close()  # type: ignore[attr-defined]


def _main(args: argparse.Namespace, log, stamp: str) -> None:
    log(f"3B Exp 01 diagnostic N={args.n} utc={stamp}")
    ids = first_n_main_ids(args.n)
    log(f"selected {len(ids)} IDs from saved main_eval ordering")
    (RESULT_DIR / "example_ids.json").write_text(
        json.dumps({"n": len(ids), "source": "preprocess_meta.json main_eval_ids[:N]", "ids": ids}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    examples, stored = examples_from_stored_125m(ids)
    template = TEMPLATE_BY_ID[TEMPLATE_ID]
    tokenizer = load_tokenizer()
    tok_cfg = _TokCfg(bos_id=128000, eos_id=128001, mini_batch_size=1024, swa_window=8192)
    log("preflight encodings (tokenizer only, before 3B restore)...")
    preflight = preflight_encodings(examples, stored, tokenizer, template, tok_cfg)
    log(
        f"prompt_len min/max={preflight['prompt_len_min']}/{preflight['prompt_len_max']} "
        f"prompt+32 max={preflight['prompt_plus_max_new_max']} "
        f"mismatches_vs_125m={len(preflight['prompt_text_mismatches_vs_125m'])}"
    )
    if not preflight["all_fit_one_chunk"]:
        raise SystemExit("preflight failed: a prompt does not fit in one TTT chunk")
    (RESULT_DIR / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")

    metrics_125m = metrics_from_125m_subset(stored)
    log(
        "125M same-32 gold_sentence "
        f"EM={100*metrics_125m['gold_sentence']['em']:.2f} F1={100*metrics_125m['gold_sentence']['f1']:.2f} "
        "no_evidence "
        f"EM={100*metrics_125m['no_evidence']['em']:.2f} F1={100*metrics_125m['no_evidence']['f1']:.2f}"
    )

    log("loading official 3B 128K checkpoint...")
    try:
        loaded = load_official_3b(attention_cudnn_override=args.attention_cudnn_override)
    except Exception as exc:
        log(f"RESTORE FAILED nvidia-smi={nvidia_smi_used_mib()} MiB: {type(exc).__name__}: {exc}")
        raise
    cfg_rec = loaded_config_record(loaded)
    cfg_rec["hydra_experiment"] = "3b/extension/ext-3b-e2e-128K"
    cfg_rec["hidden_size"] = int(loaded.cfg.model.hidden_size)
    cfg_rec["num_hidden_layers"] = int(loaded.cfg.model.num_hidden_layers)
    cfg_rec["num_attention_heads"] = int(loaded.cfg.model.num_attention_heads)
    cfg_rec["intermediate_size"] = int(loaded.cfg.model.intermediate_size)
    log(f"checkpoint: {cfg_rec['checkpoint_path']}")
    log(
        f"RoPE={cfg_rec['model_seq_len_rope_table']} SWA={cfg_rec['sliding_window_size']} "
        f"chunk={cfg_rec['mini_batch_size']} prefix={cfg_rec['prefix_len']} suffix={cfg_rec['suffix_len']}"
    )
    if int(cfg_rec["model_seq_len_rope_table"]) != 131072:
        raise SystemExit(f"expected RoPE seq_len 131072, got {cfg_rec['model_seq_len_rope_table']}")
    if int(cfg_rec["sliding_window_size"]) != 8192 or int(cfg_rec["mini_batch_size"]) != 1024:
        raise SystemExit("unexpected SWA/chunk on loaded 3B config")
    if int(cfg_rec["prefix_len"]) != 24 or int(cfg_rec["suffix_len"]) != 8:
        raise SystemExit(f"expected prefix/suffix 24/8, got {cfg_rec['prefix_len']}/{cfg_rec['suffix_len']}")
    if loaded.mini_batch_size != 1024:
        raise SystemExit("refusing to pad to anything other than one TTT chunk of 1024")

    mem_after_restore = {
        "nvidia_smi_used_mib_after_restore": nvidia_smi_used_mib(),
        "jax_after_restore": jax_memory_stats(),
    }
    log(f"nvidia-smi after restore: {mem_after_restore['nvidia_smi_used_mib_after_restore']} MiB")
    log(f"fast_weight_hash={cfg_rec['fast_weight_hash_at_load']}")
    verify_fast_weights(loaded, where="post-restore")

    greedy_fn = make_greedy_generate_fn(loaded, max_new_tokens=MAX_NEW_TOKENS)
    n_total = len(examples) * len(DIAG_CONDITIONS)
    log(f"generating {n_total} prompts (batch size 1, greedy, T3)...")

    records = []
    mem_after_first = None
    k = 0
    try:
        for ex in examples:
            for condition in DIAG_CONDITIONS:
                k += 1
                rec = evaluate_example(
                    ex,
                    condition=condition,
                    template=template,
                    tokenizer=tokenizer,
                    loaded=loaded,
                    greedy_fn=greedy_fn,
                )
                records.append(rec)
                log(
                    f"[{k}/{n_total}] {ex.id} {condition} {rec['status']} "
                    f"EM={rec.get('em')} F1={rec.get('f1'):.3f}"
                    if rec.get("f1") is not None
                    else f"[{k}/{n_total}] {ex.id} {condition} {rec['status']}"
                )
                if k == 1:
                    mem_after_first = {
                        "nvidia_smi_used_mib_after_first_gen": nvidia_smi_used_mib(),
                        "jax_after_first_gen": jax_memory_stats(),
                    }
                    peak = (mem_after_first["jax_after_first_gen"] or {}).get("peak_bytes_in_use")
                    log(
                        f"nvidia-smi after first generation: "
                        f"{mem_after_first['nvidia_smi_used_mib_after_first_gen']} MiB "
                        f"jax_peak={peak}"
                    )
    except Exception as exc:
        smi = nvidia_smi_used_mib()
        jax_stats = jax_memory_stats()
        stage = f"generation k={k}/{n_total}"
        if records:
            last = records[-1]
            stage += f" last_ok={last.get('id')}:{last.get('condition')}"
        payload = {
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "nvidia_smi_used_mib": smi,
            "jax_memory": jax_stats,
            "n_completed_records": len(records),
        }
        (RESULT_DIR / "oom_or_crash.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        log(f"STOP at {stage} nvidia-smi={smi} MiB {type(exc).__name__}: {exc}")
        raise SystemExit(f"{stage}: {type(exc).__name__}: {exc}")

    _, digest_end, _ = snapshot_fast_weights(loaded.model)
    diff_end = verify_fast_weights(loaded, where="post-all")
    state = {
        "hash_at_load": loaded.fast_weight_hash,
        "hash_at_end": digest_end,
        "max_abs_diff_end": diff_end,
        "hash_unchanged": digest_end == loaded.fast_weight_hash,
        "inner_loop_step_called": False,
        "second_post_update_forward": False,
    }
    log(f"fast-weight hash unchanged={state['hash_unchanged']} max_abs_diff={diff_end}")

    jax_after_all = jax_memory_stats()
    memory = {
        **mem_after_restore,
        **(mem_after_first or {}),
        "nvidia_smi_used_mib_after_all": nvidia_smi_used_mib(),
        "jax_after_all": jax_after_all,
        "jax_peak_bytes_in_use": (jax_after_all or {}).get("peak_bytes_in_use"),
    }
    log(f"nvidia-smi after all: {memory['nvidia_smi_used_mib_after_all']} MiB")

    write_jsonl(RESULT_DIR / "predictions.jsonl", records)
    metrics_3b = condition_metrics_two(records)
    sent_em, none_em = aligned_scores(records, "gold_sentence", "no_evidence", "em")
    sent_f1, none_f1 = aligned_scores(records, "gold_sentence", "no_evidence", "f1")
    deltas_3b = {
        "sentence_em": paired_bootstrap_ci(sent_em, none_em),
        "sentence_f1": paired_bootstrap_ci(sent_f1, none_f1),
    }
    failure_inspect = inspect_gold_sentence_failures(records, loaded.eos_id)
    write_jsonl(RESULT_DIR / "gold_sentence_failures.jsonl", failure_inspect["failures"])
    examples_rep = representative_rows(records)

    n_errors = sum(1 for r in records if r.get("status") != "ok")
    config_payload = {
        **cfg_rec,
        "dataset": "squad_v1.1",
        "split": "validation",
        "diagnostic": True,
        "n": len(ids),
        "example_ids": ids,
        "id_source": "experiments/exp01_gold_evidence_qa/preprocess_meta.json main_eval_ids[:32]",
        "contexts_source": "experiments/exp01_gold_evidence_qa/predictions.jsonl injected_context",
        "selected_template": TEMPLATE_ID,
        "conditions": list(DIAG_CONDITIONS),
        "generation": {
            "decoding": "greedy_argmax",
            "temperature": 0,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token": "eos",
            "pad_to": loaded.mini_batch_size,
            "scored_prediction_rule": "decode until EOS; take text before first newline; strip",
            "batch_size": 1,
        },
        "git_commit": git_commit(),
        "e2e_submodule_commit": "a4fc4788ace38e29b5067916d4f4be33da894085",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ttt_memo_note": (
            "One-chunk (T=1024) official logits are pre-inner-update. "
            "This diagnostic tests in-window backbone use of oracle evidence, not TTT memory."
        ),
        "do_not_attribute_scale_alone": (
            "125M vs 3B checkpoints differ in model scale and training/extension setup."
        ),
    }
    save_config(RESULT_DIR / "config.yaml", config_payload)

    summary = {
        "n_evaluated": len(ids),
        "example_ids": ids,
        "metrics": metrics_3b,
        "deltas": {
            "Delta_sentence_EM": deltas_3b["sentence_em"]["mean_delta"],
            "Delta_sentence_F1": deltas_3b["sentence_f1"]["mean_delta"],
            "sentence_em": deltas_3b["sentence_em"],
            "sentence_f1": deltas_3b["sentence_f1"],
        },
        "paired_125m_same_ids": metrics_125m,
        "paired_table": {
            "125M": metrics_125m,
            "3B": metrics_3b,
        },
        "failure_inspection": {
            "n_gold_sentence_em0": failure_inspect["n_gold_sentence_em0"],
            "n_exact_gold_substring_in_scored_when_em0": failure_inspect["n_exact_gold_substring_in_scored_when_em0"],
            "n_empty_scored_when_em0": failure_inspect["n_empty_scored_when_em0"],
            "n_hit_max_new_tokens_when_em0": failure_inspect["n_hit_max_new_tokens_when_em0"],
        },
        "preflight": {k: v for k, v in preflight.items() if k != "rows"},
        "state": state,
        "memory": memory,
        "n_errors": n_errors,
        "representative": examples_rep,
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_md(
        RESULT_DIR / "summary.md",
        n=len(ids),
        ids=ids,
        metrics_3b=metrics_3b,
        metrics_125m=metrics_125m,
        deltas_3b=deltas_3b,
        preflight=preflight,
        failure_inspect=failure_inspect,
        examples_rep=examples_rep,
        config=config_payload,
        memory=memory,
        state=state,
    )
    log("\n" + (RESULT_DIR / "summary.md").read_text(encoding="utf-8"))
    log(f"wrote {RESULT_DIR}")


if __name__ == "__main__":
    main()
