"""Exp 01 evaluation loop: smoke, prompt-dev, main.

Every (example, condition) call starts from the same closed-over checkpoint
fast weights and suffix KV state. Inner parameters are hashed at load and
re-checked after each generation.
"""

from __future__ import annotations

import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from generation import (
    MAX_NEW_TOKENS,
    SWA_SAFETY_MARGIN,
    encode_prompt,
    generate_one,
    make_greedy_generate_fn,
)
from model_runtime import fast_weight_max_abs_diff, loaded_config_record, snapshot_fast_weights
from prompts import CANDIDATE_TEMPLATES, TEMPLATE_BY_ID, render_prompt
from squad_metrics import normalize_answer, squad_em_f1
from squad_preprocess import SquadExample

CONDITIONS = ("gold_sentence", "gold_paragraph", "no_evidence")


def injected_context(example: SquadExample, condition: str) -> str:
    if condition == "gold_sentence":
        if example.gold_sentence is None:
            raise ValueError("missing gold_sentence")
        return example.gold_sentence
    if condition == "gold_paragraph":
        return example.gold_paragraph
    if condition == "no_evidence":
        if example.no_evidence_context is None:
            raise ValueError("missing no_evidence_context")
        return example.no_evidence_context
    raise ValueError(f"unknown condition {condition}")


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def paired_bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    import numpy as np

    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    n = a.shape[0]
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        deltas[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(deltas, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "mean_delta": float(a.mean() - b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "seed": seed,
    }


def bin_label(value: int, edges: list[int]) -> str:
    for i, edge in enumerate(edges):
        lo = 0 if i == 0 else edges[i - 1]
        if value <= edge:
            return f"{lo + 1 if i else 0}-{edge}"
    return f"{edges[-1] + 1}+"


def length_breakdowns(records: list[dict]) -> dict[str, Any]:
    answer_edges = [5, 15, 30]
    evidence_edges = [32, 64, 128, 256]
    input_edges = [64, 128, 256, 512]
    out: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r.get("condition") == condition and r.get("status") == "ok"]
        buckets: dict[str, dict[str, list]] = {
            "answer_char_len": defaultdict(lambda: {"em": [], "f1": []}),
            "evidence_token_len": defaultdict(lambda: {"em": [], "f1": []}),
            "input_token_len": defaultdict(lambda: {"em": [], "f1": []}),
        }
        for r in rows:
            golds = r.get("gold_answers") or [""]
            ans_len = min(len(g) for g in golds)
            a_bin = bin_label(ans_len, answer_edges)
            e_bin = bin_label(int(r.get("evidence_token_length") or 0), evidence_edges)
            i_bin = bin_label(int(r.get("input_token_length") or 0), input_edges)
            buckets["answer_char_len"][a_bin]["em"].append(r["em"])
            buckets["answer_char_len"][a_bin]["f1"].append(r["f1"])
            buckets["evidence_token_len"][e_bin]["em"].append(r["em"])
            buckets["evidence_token_len"][e_bin]["f1"].append(r["f1"])
            buckets["input_token_len"][i_bin]["em"].append(r["em"])
            buckets["input_token_len"][i_bin]["f1"].append(r["f1"])
        summarized = {}
        for kind, table in buckets.items():
            summarized[kind] = {
                k: {"n": len(v["em"]), "em": mean(v["em"]), "f1": mean(v["f1"])}
                for k, v in sorted(table.items(), key=lambda kv: kv[0])
            }
        out[condition] = summarized
    return out


def condition_metrics(records: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r.get("condition") == condition and r.get("status") == "ok"]
        out[condition] = {
            "n": len(rows),
            "em": mean([r["em"] for r in rows]),
            "f1": mean([r["f1"] for r in rows]),
        }
    return out


def verify_fast_weights(loaded, *, where: str) -> float:
    diff = fast_weight_max_abs_diff(loaded.model, loaded.fast_weight_snapshot)
    if diff != 0.0:
        raise RuntimeError(f"fast-weight drift at {where}: max_abs_diff={diff}")
    return diff


def evaluate_example(
    example: SquadExample,
    *,
    condition: str,
    template,
    tokenizer,
    loaded,
    greedy_fn,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": example.id,
        "question": example.question,
        "gold_answers": list(example.gold_answers),
        "gold_sentence": example.gold_sentence,
        "gold_paragraph": example.gold_paragraph,
        "condition": condition,
        "prompt_template_id": template.template_id,
        "no_evidence_source_id": example.no_evidence_source_id if condition == "no_evidence" else None,
        "status": "ok",
        "error": None,
        "swa_window": loaded.swa_window,
    }
    try:
        if example.status != "ok":
            raise RuntimeError(example.error or "preprocess_error")
        context = injected_context(example, condition)
        record["injected_context"] = context
        encoded = encode_prompt(
            tokenizer,
            template,
            context=context,
            question=example.question,
            bos_id=loaded.bos_id,
            eos_id=loaded.eos_id,
            chunk_size=loaded.mini_batch_size,
            swa_window=loaded.swa_window,
        )
        gen = generate_one(greedy_fn, encoded, tokenizer, loaded.eos_id)
        record.update(
            {
                "input_token_length": gen["input_token_length"],
                "evidence_token_length": gen["evidence_token_length"],
                "evidence_question_distance": gen["evidence_question_distance"],
                "raw_prediction": gen["raw_prediction"],
                "scored_prediction": gen["scored_prediction"],
                "generated_token_ids": gen["generated_token_ids"],
                "prompt_text": gen["prompt_text"],
            }
        )
        em, f1 = squad_em_f1(gen["scored_prediction"], example.gold_answers)
        record["em"] = em
        record["f1"] = f1
        verify_fast_weights(loaded, where=f"{example.id}:{condition}")
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record.setdefault("injected_context", None)
        record.setdefault("raw_prediction", None)
        record.setdefault("scored_prediction", None)
        record.setdefault("em", 0)
        record.setdefault("f1", 0.0)
        record["traceback"] = traceback.format_exc()
    return record


def run_conditions(
    examples: list[SquadExample],
    *,
    conditions: tuple[str, ...],
    template,
    tokenizer,
    loaded,
    greedy_fn,
    log_fn=print,
) -> list[dict[str, Any]]:
    records = []
    n = len(examples) * len(conditions)
    k = 0
    for ex in examples:
        for condition in conditions:
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
            log_fn(
                f"[{k}/{n}] {ex.id} {condition} {rec['status']} "
                f"EM={rec.get('em')} F1={rec.get('f1'):.3f}"
                if rec.get("f1") is not None
                else f"[{k}/{n}] {ex.id} {condition} {rec['status']}"
            )
    return records


def smoke_invariants(
    examples: list[SquadExample],
    records: list[dict],
    *,
    tokenizer,
    template,
    loaded,
    greedy_fn,
    log_fn=print,
) -> list[str]:
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)
        log_fn(f"SANITY FAIL: {msg}")

    log_fn("\n=== smoke sanity checks ===")

    # 1. Gold sentence contains a gold answer.
    for ex in examples:
        if ex.gold_sentence is None or not any(a in ex.gold_sentence for a in ex.gold_answers):
            fail(f"{ex.id}: gold sentence does not contain a gold answer")
        else:
            log_fn(f"OK gold-sentence-contains-answer {ex.id}")

    # 2. No-evidence context does not contain normalized gold answers.
    for ex in examples:
        ctx_norm = normalize_answer(ex.no_evidence_context or "")
        leaked = [g for g in (normalize_answer(a) for a in ex.gold_answers) if g and g in ctx_norm]
        if leaked:
            fail(f"{ex.id}: no_evidence context contains {leaked}")
        else:
            log_fn(f"OK no-evidence-no-leak {ex.id}")

    # 3. Evidence within SWA.
    for rec in records:
        if rec["status"] != "ok":
            fail(f"{rec['id']} {rec['condition']}: {rec.get('error')}")
            continue
        dist = rec["evidence_question_distance"]
        if dist >= rec["swa_window"]:
            fail(f"{rec['id']} {rec['condition']}: distance {dist} >= SWA {rec['swa_window']}")
        else:
            log_fn(
                f"OK swa {rec['id']} {rec['condition']} dist={dist} "
                f"input={rec['input_token_length']} ev={rec['evidence_token_length']} swa={rec['swa_window']}"
            )

    # 4. Fast weights unchanged.
    diff = fast_weight_max_abs_diff(loaded.model, loaded.fast_weight_snapshot)
    _, digest, _ = snapshot_fast_weights(loaded.model)
    if diff != 0.0 or digest != loaded.fast_weight_hash:
        fail(f"fast-weight hash/diff mismatch diff={diff} hash={digest}")
    else:
        log_fn(f"OK state-reset hash={digest[:16]}... max_abs_diff={diff}")

    # 5. Deterministic generation: same input twice -> same output.
    probe = examples[0]
    enc = encode_prompt(
        tokenizer,
        template,
        context=probe.gold_sentence or "",
        question=probe.question,
        bos_id=loaded.bos_id,
        eos_id=loaded.eos_id,
        chunk_size=loaded.mini_batch_size,
        swa_window=loaded.swa_window,
    )
    a = generate_one(greedy_fn, enc, tokenizer, loaded.eos_id)
    b = generate_one(greedy_fn, enc, tokenizer, loaded.eos_id)
    if a["generated_token_ids"] != b["generated_token_ids"]:
        fail("generation not deterministic")
    else:
        log_fn(f"OK deterministic ids={a['generated_token_ids'][:8]}...")

    # 6. Condition prompts differ only in injected context.
    for ex in examples[:3]:
        texts = {}
        for cond in CONDITIONS:
            texts[cond] = render_prompt(template, context=injected_context(ex, cond), question=ex.question)
        marker_s = texts["gold_sentence"].replace(ex.gold_sentence, "CTX")
        marker_p = texts["gold_paragraph"].replace(ex.gold_paragraph, "CTX")
        marker_n = texts["no_evidence"].replace(ex.no_evidence_context, "CTX")
        if not (marker_s == marker_p == marker_n):
            fail(f"{ex.id}: prompts differ by more than context")
        else:
            log_fn(f"OK prompt-only-context {ex.id}")

    # 7. Scoring unit tests are run separately; spot-check one identity here.
    em, f1 = squad_em_f1("the Notre Dame", ["Notre Dame"])
    if em != 1 or f1 != 1.0:
        fail("SQuAD metric identity check failed")
    else:
        log_fn("OK squad-metric identity")

    log_fn("\n=== sample prompts (first 2 examples × 3 conditions) ===")
    for ex in examples[:2]:
        for cond in CONDITIONS:
            text = render_prompt(template, context=injected_context(ex, cond), question=ex.question)
            log_fn(f"\n--- {ex.id} / {cond} / {template.template_id} ---\n{text}\n")

    if failures:
        log_fn(f"\n{len(failures)} sanity failure(s). Stopping before prompt-dev/main.")
    else:
        log_fn("\nAll smoke sanity checks passed.")
    return failures


def select_prompt_template(dev_by_template: dict[str, list[dict]]) -> dict[str, Any]:
    rows = []
    for tid, recs in dev_by_template.items():
        ok = [r for r in recs if r.get("status") == "ok"]
        rows.append(
            {
                "prompt_template_id": tid,
                "n": len(ok),
                "em": mean([r["em"] for r in ok]),
                "f1": mean([r["f1"] for r in ok]),
            }
        )
    rows.sort(key=lambda r: (r["f1"], r["em"], r["prompt_template_id"]), reverse=True)
    winner = rows[0]["prompt_template_id"]
    return {"selected": winner, "table": rows, "rule": "max normalized F1, EM tie-break"}


def sample_failures(records: list[dict], *, n: int = 30, seed: int = 42) -> list[dict]:
    import random

    gs = [r for r in records if r.get("condition") == "gold_sentence" and r.get("status") == "ok" and r.get("em") == 0]
    low = sorted(gs, key=lambda r: (r["f1"], r["id"]))
    high = sorted(gs, key=lambda r: (-r["f1"], r["id"]))
    verbose = sorted(gs, key=lambda r: (-len(r.get("raw_prediction") or ""), r["id"]))
    picked: list[dict] = []
    seen: set[str] = set()
    for group in (low[:10], high[:10], verbose[:10]):
        for r in group:
            if r["id"] in seen:
                continue
            picked.append(r)
            seen.add(r["id"])
            if len(picked) >= n:
                return picked
    rng = random.Random(seed)
    rest = [r for r in gs if r["id"] not in seen]
    rng.shuffle(rest)
    for r in rest:
        picked.append(r)
        if len(picked) >= n:
            break
    return picked


def write_failures_md(path: Path, failures: list[dict]) -> None:
    lines = [
        "# Exp 01 Gold Sentence failures (sample)",
        "",
        "Manual taxonomy is left blank. Categories were not assigned automatically.",
        "",
    ]
    for i, r in enumerate(failures, 1):
        lines.extend(
            [
                f"## {i}. `{r['id']}`",
                "",
                f"- EM: {r.get('em')}  F1: {r.get('f1')}",
                f"- condition: {r.get('condition')}",
                f"- prompt_template_id: {r.get('prompt_template_id')}",
                "",
                "**Question**",
                "",
                r.get("question") or "",
                "",
                "**Gold evidence (injected context)**",
                "",
                r.get("injected_context") or "",
                "",
                "**Gold answers**",
                "",
                json.dumps(r.get("gold_answers"), ensure_ascii=False),
                "",
                "**Raw prediction**",
                "",
                "```",
                r.get("raw_prediction") or "",
                "```",
                "",
                "**Scored prediction**",
                "",
                "```",
                r.get("scored_prediction") or "",
                "```",
                "",
                "**Manual taxonomy:**",
                "",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            slim = {k: v for k, v in rec.items() if k != "traceback"}
            fh.write(json.dumps(slim, ensure_ascii=False) + "\n")


def write_summary_md(
    path: Path,
    *,
    n_eval: int,
    metrics: dict,
    deltas: dict,
    prompt_sel: dict | None,
    breakdowns: dict,
    failures: list[dict],
    config: dict,
    preprocess: dict,
) -> None:
    gs = metrics["gold_sentence"]
    gp = metrics["gold_paragraph"]
    ne = metrics["no_evidence"]
    lines = [
        "# TTT-MEMO Exp 01 — Gold evidence in the local window",
        "",
        "Feasibility check only. No GO/NO-GO decision is encoded here.",
        "",
        "## 1. Dataset / preprocessing",
        "",
        f"- dataset: SQuAD 1.1 validation (`{preprocess.get('squad_path')}`)",
        f"- seed: {preprocess.get('seed')}",
        f"- prompt-dev N: {preprocess.get('n_prompt_dev')}  main-eval N: {preprocess.get('n_main_eval')}",
        f"- usable after preprocessing: {preprocess.get('n_ok')} / {preprocess.get('n_total')}",
        f"- preprocess errors: {preprocess.get('n_preprocess_error')}",
        f"- filtering: answerable SQuAD 1.1 with a locatable contiguous gold span; no filtering on model output",
        "",
        "## 2. Checkpoint / config",
        "",
        f"- checkpoint: `{config.get('checkpoint_path')}`",
        f"- tokenizer: `{config.get('tokenizer')}`",
        f"- SWA window (loaded): {config.get('sliding_window_size')}",
        f"- TTT mini_batch_size: {config.get('mini_batch_size')}",
        f"- suffix_len / prefix_len: {config.get('suffix_len')} / {config.get('prefix_len')}",
        f"- spec_inner: {config.get('spec_inner')}",
        f"- train_mode: {config.get('train_mode')}",
        f"- backend: {config.get('backend', {}).get('note')} / {config.get('backend', {}).get('gpu_name')}",
        f"- RoPE seq_len: {config.get('model_seq_len_rope_table')} (not the padded generate length)",
        "",
        "Generation uses official `prefix_call` + `suffix_call` on one 1024-token chunk.",
        "Inner-loop logits for that chunk are computed before the chunk's fast-weight update,",
        "so Exp 01 measures the backbone (SWA + initial prime FFN), not intra-prompt TTT memory.",
        "",
        "## 3. Selected prompt + prompt-dev scores",
        "",
    ]
    if prompt_sel:
        lines.append(f"- selected: **{prompt_sel['selected']}** ({prompt_sel['rule']})")
        lines.append("")
        lines.append("| template | N | EM | F1 |")
        lines.append("|---|---:|---:|---:|")
        for row in prompt_sel["table"]:
            lines.append(
                f"| {row['prompt_template_id']} | {row['n']} | {100*row['em']:.2f} | {100*row['f1']:.2f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 4. Generation setup",
            "",
            f"- greedy argmax, no sampling, max_new_tokens={MAX_NEW_TOKENS}",
            "- right-pad to mini_batch_size with EOS; scored_prediction = first line before newline, cut at EOS",
            f"- SWA safety margin: {SWA_SAFETY_MARGIN} tokens",
            "",
            "## 5. Condition-level EM / F1 (main eval)",
            "",
            "| condition | N | EM | F1 |",
            "|---|---:|---:|---:|",
            f"| gold_sentence | {gs['n']} | {100*gs['em']:.2f} | {100*gs['f1']:.2f} |",
            f"| gold_paragraph | {gp['n']} | {100*gp['em']:.2f} | {100*gp['f1']:.2f} |",
            f"| no_evidence | {ne['n']} | {100*ne['em']:.2f} | {100*ne['f1']:.2f} |",
            "",
            "## 6. Gold vs no-evidence deltas",
            "",
            f"- Delta_sentence_EM = {100*deltas['sentence_em']['mean_delta']:.2f} "
            f"(95% paired bootstrap CI {100*deltas['sentence_em']['ci_low']:.2f}, {100*deltas['sentence_em']['ci_high']:.2f})",
            f"- Delta_sentence_F1 = {100*deltas['sentence_f1']['mean_delta']:.2f} "
            f"(95% CI {100*deltas['sentence_f1']['ci_low']:.2f}, {100*deltas['sentence_f1']['ci_high']:.2f})",
            f"- Delta_paragraph_EM = {100*deltas['paragraph_em']['mean_delta']:.2f} "
            f"(95% CI {100*deltas['paragraph_em']['ci_low']:.2f}, {100*deltas['paragraph_em']['ci_high']:.2f})",
            f"- Delta_paragraph_F1 = {100*deltas['paragraph_f1']['mean_delta']:.2f} "
            f"(95% CI {100*deltas['paragraph_f1']['ci_low']:.2f}, {100*deltas['paragraph_f1']['ci_high']:.2f})",
            "",
            "## 7. Lightweight length breakdown",
            "",
            "```json",
            json.dumps(breakdowns, indent=2),
            "```",
            "",
            "## 8. Representative Gold Sentence failures",
            "",
            f"Sample of {len(failures)} EM=0 examples. Manual taxonomy is empty.",
            "",
        ]
    )
    for r in failures[:8]:
        lines.append(
            f"- `{r['id']}` EM={r['em']} F1={r['f1']:.2f} scored={r.get('scored_prediction')!r} "
            f"gold={r.get('gold_answers')!r}"
        )
    lines.extend(
        [
            "",
            "## 9. Interpretation",
            "",
            "No automatic optimistic interpretation. Inspect raw metrics and failures.md.",
            "",
            f"On {n_eval} evaluated SQuAD examples, Gold Sentence achieved "
            f"{100*gs['em']:.2f} EM / {100*gs['f1']:.2f} F1, "
            f"Gold Paragraph achieved {100*gp['em']:.2f} EM / {100*gp['f1']:.2f} F1, "
            f"and No Evidence achieved {100*ne['em']:.2f} EM / {100*ne['f1']:.2f} F1.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
