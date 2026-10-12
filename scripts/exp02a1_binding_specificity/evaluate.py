"""Exp 02A-1 evaluation: swapped-binding write, source-free/source-present recall."""

from __future__ import annotations

import json
import traceback
from collections import defaultdict
from typing import Any

import torch

from metrics import (
    b_swap,
    binding_score_from_matrix,
    flip_diagnostics,
    generation_scores,
    paired_bootstrap_ci,
    rank_candidates,
)
from model_runtime import (
    LoadedRuntime,
    frozen_watch_max_abs_diff,
    lora_max_abs_diff,
    lora_state_cpu,
    reset_example_state,
    set_adapter_enabled,
)
from prompts import encode_candidate, encode_recall_prompt, encode_write_pair, to_tensors
from protocol import (
    QUERY_STYLES,
    SOURCE_FREE_CONDITIONS,
    SOURCE_PRESENT_CONDITIONS,
    SPEC,
    TTT_CONDITIONS,
)
from ttt_write import write_fact


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


@torch.no_grad()
def score_candidate_logprobs(runtime: LoadedRuntime, encoded: dict[str, Any]) -> dict[str, Any]:
    model = runtime.model
    model.eval()
    batch = to_tensors(encoded, runtime.device, include_labels=True)
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        past_key_values=None,
    )
    logits = out.logits[0].float()
    labels = batch["labels"][0]
    logp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = labels[1:]
    mask = tgt != -100
    if int(mask.sum().item()) == 0:
        raise RuntimeError("no labeled candidate tokens")
    token_logp = logp[mask, tgt[mask]]
    logp_sum = float(token_logp.sum().cpu())
    logp_mean = float(token_logp.mean().cpu())
    n_tokens = int(mask.sum().item())
    if n_tokens != encoded["n_tokens"]:
        raise RuntimeError(f"labeled {n_tokens} != encoded {encoded['n_tokens']}")
    del out, batch
    return {
        "candidate": encoded["candidate"],
        "logp_sum": logp_sum,
        "logp_mean": logp_mean,
        "nll_mean": -logp_mean,
        "n_tokens": n_tokens,
        "candidate_token_ids": [encoded["input_ids"][i] for i in encoded["candidate_token_indices"]],
    }


@torch.no_grad()
def greedy_generate(runtime: LoadedRuntime, prompt_ids: list[int]) -> dict[str, Any]:
    model = runtime.model
    tokenizer = runtime.tokenizer
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=runtime.device)
    attention_mask = torch.ones_like(input_ids)
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(SPEC["generation"]["max_new_tokens"]),
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = gen[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    del gen, input_ids, attention_mask
    return {"generated_token_ids": new_ids, "raw_prediction": text}


def _mapping_for_condition(condition: str) -> str | None:
    if "mapping_A" in condition:
        return "A"
    if "mapping_B" in condition:
        return "B"
    return None


def gold_value(pair: dict[str, Any], *, mapping: str, entity_key: str) -> str:
    return pair[f"mapping_{mapping}"][entity_key]


def evaluate_condition(
    runtime: LoadedRuntime,
    pair: dict[str, Any],
    *,
    condition: str,
    k_steps: int,
) -> dict[str, Any]:
    mapping = _mapping_for_condition(condition)
    source_present = condition in SOURCE_PRESENT_CONDITIONS
    do_write = condition in TTT_CONDITIONS
    record: dict[str, Any] = {
        "pair_id": pair["pair_id"],
        "info_type": pair["info_type"],
        "condition": condition,
        "mapping": mapping,
        "source_present": source_present,
        "adapter_enabled": do_write,
        "status": "ok",
        "error": None,
        "recall_used_past_key_values": False,
        "recall_updated_lora": False,
        "e1": pair["e1"],
        "e2": pair["e2"],
        "v1": pair["v1"],
        "v2": pair["v2"],
    }
    opt = None
    post_write_lora = None
    try:
        opt = reset_example_state(runtime)
        if do_write:
            if mapping not in ("A", "B"):
                raise RuntimeError("TTT condition requires mapping")
            encoded_write = encode_write_pair(runtime.tokenizer, pair, mapping=mapping)
            write_log = write_fact(runtime, encoded_write, opt, k_steps=k_steps)
            record["write"] = write_log
            record["write_text"] = encoded_write.text
            record["write_n_labeled_tokens"] = encoded_write.n_labeled
            record["decoded_oracle_tokens"] = runtime.tokenizer.decode(
                [encoded_write.input_ids[i] for i in encoded_write.oracle_token_indices],
                skip_special_tokens=False,
            )
            if write_log["lora_max_abs_diff_from_init"] <= 0.0:
                raise RuntimeError("LoRA did not change after TTT write")
            base_diff = frozen_watch_max_abs_diff(runtime.model, runtime.frozen_watch_cpu)
            if base_diff != 0.0:
                raise RuntimeError(f"base weights changed during write: {base_diff}")
            post_write_lora = lora_state_cpu(runtime.model)
            del encoded_write
            set_adapter_enabled(runtime.model, True)
        else:
            record["write"] = None
            set_adapter_enabled(runtime.model, False)

        queries: dict[str, Any] = {}
        matrices: dict[str, Any] = {}
        prediction_rows: list[dict[str, Any]] = []
        for style in QUERY_STYLES:
            matrix_mean = {"e1": {}, "e2": {}}
            matrix_sum = {"e1": {}, "e2": {}}
            style_block: dict[str, Any] = {}
            for entity_key in ("e1", "e2"):
                question = pair["questions"][entity_key][style]
                prompt = encode_recall_prompt(
                    runtime.tokenizer,
                    pair,
                    question=question,
                    source_present=source_present,
                    mapping=mapping,
                )
                if condition in SOURCE_FREE_CONDITIONS:
                    if pair["write_block_A"] in prompt["text"] or pair["write_block_B"] in prompt["text"]:
                        raise RuntimeError("source fact present in source-free recall prompt")
                cand_rows = []
                for cand in pair["candidates"]:
                    encoded_c = encode_candidate(runtime.tokenizer, prompt["text"], cand)
                    scored = score_candidate_logprobs(runtime, encoded_c)
                    scored["is_v1"] = cand == pair["v1"]
                    scored["is_v2"] = cand == pair["v2"]
                    cand_rows.append(scored)
                    del encoded_c
                by_text = {r["candidate"]: r for r in cand_rows}
                matrix_mean[entity_key]["v1"] = by_text[pair["v1"]]["logp_mean"]
                matrix_mean[entity_key]["v2"] = by_text[pair["v2"]]["logp_mean"]
                matrix_sum[entity_key]["v1"] = by_text[pair["v1"]]["logp_sum"]
                matrix_sum[entity_key]["v2"] = by_text[pair["v2"]]["logp_sum"]
                gold = gold_value(pair, mapping=mapping or "A", entity_key=entity_key)
                if mapping is None:
                    # Static has no mapping; still rank against Mapping A golds as a bias diagnostic.
                    gold = pair["mapping_A"][entity_key]
                ranked = rank_candidates(cand_rows, gold=gold)
                gen = greedy_generate(runtime, prompt["input_ids"])
                gen_scores = generation_scores(gen["raw_prediction"], gold)
                entity_out = {
                    "entity_key": entity_key,
                    "entity": pair[entity_key],
                    "question": question,
                    "gold": gold,
                    "candidates": cand_rows,
                    "recall_prompt": prompt["text"],
                    "recall_n_tokens": len(prompt["input_ids"]),
                    **ranked,
                    **gen,
                    **gen_scores,
                }
                style_block[entity_key] = entity_out
                prediction_rows.append(
                    {
                        "pair_id": pair["pair_id"],
                        "info_type": pair["info_type"],
                        "condition": condition,
                        "mapping": mapping,
                        "query_style": style,
                        "entity_key": entity_key,
                        "entity": pair[entity_key],
                        "gold": gold,
                        "v1": pair["v1"],
                        "v2": pair["v2"],
                        "s_v1_mean": by_text[pair["v1"]]["logp_mean"],
                        "s_v2_mean": by_text[pair["v2"]]["logp_mean"],
                        "s_v1_sum": by_text[pair["v1"]]["logp_sum"],
                        "s_v2_sum": by_text[pair["v2"]]["logp_sum"],
                        "gold_nll_mean": ranked["gold_nll_mean"],
                        "gold_rank": ranked["gold_rank"],
                        "accuracy": ranked["accuracy"],
                        "accuracy_sum_logp": ranked["accuracy_sum_logp"],
                        "margin_logp_mean": ranked["margin_logp_mean"],
                        "raw_prediction": gen["raw_prediction"],
                        "em": gen_scores["em"],
                        "f1": gen_scores["f1"],
                    }
                )
            matrices[style] = {
                "mean_logp": matrix_mean,
                "sum_logp": matrix_sum,
            }
            queries[style] = style_block

        if do_write:
            recall_lora_diff = lora_max_abs_diff(runtime.model, post_write_lora)
            record["recall_lora_max_abs_diff_from_post_write"] = recall_lora_diff
            if recall_lora_diff != 0.0:
                record["recall_updated_lora"] = True
                raise RuntimeError(f"recall updated LoRA: max_abs_diff={recall_lora_diff}")
        else:
            record["recall_lora_max_abs_diff_from_post_write"] = 0.0

        base_diff = frozen_watch_max_abs_diff(runtime.model, runtime.frozen_watch_cpu)
        record["base_weight_max_abs_diff"] = base_diff
        if base_diff != 0.0:
            raise RuntimeError(f"base weights changed at recall: {base_diff}")
        record["queries"] = queries
        record["matrices"] = matrices
        record["prediction_rows"] = prediction_rows
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        reset_example_state(runtime)
        del opt
        del post_write_lora
    return record


def run_pairs(
    runtime: LoadedRuntime,
    pairs: list[dict[str, Any]],
    *,
    k_steps: int,
    log_fn=print,
) -> list[dict[str, Any]]:
    records = []
    conditions = tuple(SPEC["conditions"])
    n = len(pairs) * len(conditions)
    k = 0
    for pair in pairs:
        for condition in conditions:
            k += 1
            rec = evaluate_condition(runtime, pair, condition=condition, k_steps=k_steps)
            records.append(rec)
            extra = rec["status"]
            if rec["status"] == "ok" and rec.get("write"):
                extra += (
                    f" loss={rec['write']['loss_before']:.3f}->{rec['write']['loss_after']:.3f} "
                    f"lora_l2={rec['write']['lora_l2_delta_from_init']:.3f}"
                )
            log_fn(f"[{k}/{n}] {pair['pair_id']} {condition} {extra}")
    return records


def flatten_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        if rec.get("status") != "ok":
            rows.append(
                {
                    "pair_id": rec["pair_id"],
                    "condition": rec["condition"],
                    "status": rec["status"],
                    "error": rec.get("error"),
                }
            )
            continue
        for row in rec.get("prediction_rows") or []:
            out = dict(row)
            out["status"] = "ok"
            if rec.get("write"):
                out["write_loss_before"] = rec["write"]["loss_before"]
                out["write_loss_after"] = rec["write"]["loss_after"]
                out["lora_l2_delta"] = rec["write"]["lora_l2_delta_from_init"]
                out["lora_max_abs_diff"] = rec["write"]["lora_max_abs_diff_from_init"]
            rows.append(out)
    return rows


def flatten_matrices(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        for style in QUERY_STYLES:
            mat = rec["matrices"][style]
            rows.append(
                {
                    "pair_id": rec["pair_id"],
                    "info_type": rec["info_type"],
                    "condition": rec["condition"],
                    "mapping": rec["mapping"],
                    "query_style": style,
                    "matrix_mean_logp": mat["mean_logp"],
                    "matrix_sum_logp": mat["sum_logp"],
                    "e1": rec["e1"],
                    "e2": rec["e2"],
                    "v1": rec["v1"],
                    "v2": rec["v2"],
                }
            )
    return rows


def pair_binding_table(records: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_pid = {p["pair_id"]: p for p in pairs}
    out: dict[str, Any] = {"by_type_style": {}, "per_pair": []}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        pid = pair["pair_id"]
        rec_a = next(r for r in records if r["pair_id"] == pid and r["condition"] == "ttt_mapping_A_source_free")
        rec_b = next(r for r in records if r["pair_id"] == pid and r["condition"] == "ttt_mapping_B_source_free")
        rec_sp_a = next(r for r in records if r["pair_id"] == pid and r["condition"] == "static_mapping_A_source_present")
        rec_sp_b = next(r for r in records if r["pair_id"] == pid and r["condition"] == "static_mapping_B_source_present")
        rec_st = next(r for r in records if r["pair_id"] == pid and r["condition"] == "static_source_free")
        pair_row: dict[str, Any] = {"pair_id": pid, "info_type": pair["info_type"], "styles": {}}
        for style in QUERY_STYLES:
            if rec_a.get("status") != "ok" or rec_b.get("status") != "ok":
                continue
            mat_a = rec_a["matrices"][style]["mean_logp"]
            mat_b = rec_b["matrices"][style]["mean_logp"]
            mat_sp_a = rec_sp_a["matrices"][style]["mean_logp"] if rec_sp_a.get("status") == "ok" else None
            mat_sp_b = rec_sp_b["matrices"][style]["mean_logp"] if rec_sp_b.get("status") == "ok" else None
            mat_st = rec_st["matrices"][style]["mean_logp"] if rec_st.get("status") == "ok" else None
            bind = b_swap(mat_a, mat_b)
            flips = flip_diagnostics(mat_a, mat_b)
            sp = b_swap(mat_sp_a, mat_sp_b) if mat_sp_a and mat_sp_b else None
            sp_flips = flip_diagnostics(mat_sp_a, mat_sp_b) if mat_sp_a and mat_sp_b else None
            static_b_a = None
            if mat_st is not None:
                static_b_a = binding_score_from_matrix(mat_st, "A")
            two_way_a = mean(
                [
                    float(rec_a["queries"][style]["e1"]["accuracy"]),
                    float(rec_a["queries"][style]["e2"]["accuracy"]),
                ]
            )
            two_way_b = mean(
                [
                    float(rec_b["queries"][style]["e1"]["accuracy"]),
                    float(rec_b["queries"][style]["e2"]["accuracy"]),
                ]
            )
            two_way_sp = None
            if rec_sp_a.get("status") == "ok" and rec_sp_b.get("status") == "ok":
                two_way_sp = mean(
                    [
                        float(rec_sp_a["queries"][style]["e1"]["accuracy"]),
                        float(rec_sp_a["queries"][style]["e2"]["accuracy"]),
                        float(rec_sp_b["queries"][style]["e1"]["accuracy"]),
                        float(rec_sp_b["queries"][style]["e2"]["accuracy"]),
                    ]
                )
            two_way_st = mean(
                [
                    float(rec_st["queries"][style]["e1"]["accuracy"]),
                    float(rec_st["queries"][style]["e2"]["accuracy"]),
                ]
            ) if rec_st.get("status") == "ok" else None
            em_ttt = mean(
                [
                    float(rec_a["queries"][style]["e1"]["em"]),
                    float(rec_a["queries"][style]["e2"]["em"]),
                    float(rec_b["queries"][style]["e1"]["em"]),
                    float(rec_b["queries"][style]["e2"]["em"]),
                ]
            )
            style_row = {
                "B_A": bind["B_A"],
                "B_B": bind["B_B"],
                "B_swap": bind["B_swap"],
                "e1_flipped": flips["e1_flipped"],
                "e2_flipped": flips["e2_flipped"],
                "both_flipped": flips["both_flipped"],
                "two_way_acc_A": two_way_a,
                "two_way_acc_B": two_way_b,
                "two_way_acc_ttt": mean([two_way_a, two_way_b]),
                "two_way_acc_source_present": two_way_sp,
                "two_way_acc_static": two_way_st,
                "source_present_B_swap": None if sp is None else sp["B_swap"],
                "source_present_B_A": None if sp is None else sp["B_A"],
                "source_present_B_B": None if sp is None else sp["B_B"],
                "source_present_both_flipped": None if sp_flips is None else sp_flips["both_flipped"],
                "static_B_A": static_b_a,
                "generation_em_ttt": em_ttt,
                "gold_nll_A_e1": rec_a["queries"][style]["e1"]["gold_nll_mean"],
                "gold_nll_A_e2": rec_a["queries"][style]["e2"]["gold_nll_mean"],
                "gold_nll_B_e1": rec_b["queries"][style]["e1"]["gold_nll_mean"],
                "gold_nll_B_e2": rec_b["queries"][style]["e2"]["gold_nll_mean"],
                "write_loss_pre_A": rec_a.get("write", {}).get("loss_before") if rec_a.get("write") else None,
                "write_loss_post_A": rec_a.get("write", {}).get("loss_after") if rec_a.get("write") else None,
                "write_loss_pre_B": rec_b.get("write", {}).get("loss_before") if rec_b.get("write") else None,
                "write_loss_post_B": rec_b.get("write", {}).get("loss_after") if rec_b.get("write") else None,
                "lora_l2_A": rec_a.get("write", {}).get("lora_l2_delta_from_init") if rec_a.get("write") else None,
                "lora_l2_B": rec_b.get("write", {}).get("lora_l2_delta_from_init") if rec_b.get("write") else None,
                "lora_max_abs_A": rec_a.get("write", {}).get("lora_max_abs_diff_from_init") if rec_a.get("write") else None,
                "lora_max_abs_B": rec_b.get("write", {}).get("lora_max_abs_diff_from_init") if rec_b.get("write") else None,
            }
            pair_row["styles"][style] = style_row
            grouped[(pair["info_type"], style)].append(style_row)
        out["per_pair"].append(pair_row)

    min_acc = float(SPEC["source_present_calibrated_min_two_way_acc"])
    for (info_type, style), rows in sorted(grouped.items()):
        bswaps = [r["B_swap"] for r in rows]
        zeros = [0.0] * len(bswaps)
        sp_acc = mean([r["two_way_acc_source_present"] for r in rows if r["two_way_acc_source_present"] is not None])
        sp_b = mean([r["source_present_B_swap"] for r in rows if r["source_present_B_swap"] is not None])
        calibrated = bool(sp_acc >= min_acc)
        block = {
            "n_pairs": len(rows),
            "source_present_B_swap": sp_b,
            "source_present_two_way_accuracy": sp_acc,
            "source_present_calibrated": calibrated,
            "static_B_A": mean([r["static_B_A"] for r in rows if r["static_B_A"] is not None]),
            "static_two_way_accuracy_vs_mapping_A": mean(
                [r["two_way_acc_static"] for r in rows if r["two_way_acc_static"] is not None]
            ),
            "Mapping_A_B_A": mean([r["B_A"] for r in rows]),
            "Mapping_B_B_B": mean([r["B_B"] for r in rows]),
            "B_swap": mean(bswaps),
            "fraction_e1_flipped": mean([float(r["e1_flipped"]) for r in rows]),
            "fraction_e2_flipped": mean([float(r["e2_flipped"]) for r in rows]),
            "fraction_both_flipped": mean([float(r["both_flipped"]) for r in rows]),
            "ttt_two_way_accuracy": mean([r["two_way_acc_ttt"] for r in rows]),
            "generation_em_ttt": mean([r["generation_em_ttt"] for r in rows]),
            "write_loss_pre": mean(
                [x for r in rows for x in (r["write_loss_pre_A"], r["write_loss_pre_B"]) if x is not None]
            ),
            "write_loss_post": mean(
                [x for r in rows for x in (r["write_loss_post_A"], r["write_loss_post_B"]) if x is not None]
            ),
            "lora_l2_delta": mean(
                [x for r in rows for x in (r["lora_l2_A"], r["lora_l2_B"]) if x is not None]
            ),
            "bootstrap_B_swap_descriptive_only": paired_bootstrap_ci(bswaps, zeros),
            "do_not_interpret_source_free": (not calibrated),
            "interpretation_note": (
                None
                if calibrated
                else (
                    f"source-present two-way accuracy {sp_acc:.3f} < {min_acc:.2f}; "
                    "do not interpret this style's source-free binding result."
                )
            ),
        }
        out["by_type_style"][f"{info_type}/{style}"] = block
    out["n_pairs_in_table"] = len(by_pid)
    return out


def smoke_invariants(
    runtime: LoadedRuntime,
    pairs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    tokenizer,
    log_fn=print,
) -> list[str]:
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)
        log_fn(f"SANITY FAIL: {msg}")

    def ok(msg: str) -> None:
        log_fn(f"OK {msg}")

    log_fn("\n=== smoke sanity checks ===")
    from dataset import token_id_multiset

    errors = [r for r in records if r["status"] != "ok"]
    if errors:
        fail(f"{len(errors)} evaluation errors, first={errors[0]['error']}")
    else:
        ok(f"all {len(records)} records status=ok")

    for p in pairs:
        ms_a = token_id_multiset(tokenizer, p["write_block_A"])
        ms_b = token_id_multiset(tokenizer, p["write_block_B"])
        if ms_a != ms_b:
            fail(f"{p['pair_id']}: A/B token multisets differ")
        if p["write_block_A"].count(p["v1"]) != 1 or p["write_block_B"].count(p["v1"]) != 1:
            fail(f"{p['pair_id']}: v1 not once in A and B")
        if p["write_block_A"].count(p["v2"]) != 1 or p["write_block_B"].count(p["v2"]) != 1:
            fail(f"{p['pair_id']}: v2 not once in A and B")
        if p["entities_in_write_order"] != [p[k] for k in p["write_order"]]:
            fail(f"{p['pair_id']}: entity order stored inconsistently")
        if p["entities_in_write_order"] != [
            p["e1"] if p["e1"] in sent else p["e2"] for sent in p["sentences_B"]
        ]:
            fail(f"{p['pair_id']}: entity order A vs B")
        if p["candidates"] != [p["v1"], p["v2"]]:
            fail(f"{p['pair_id']}: candidate set drifted")
    ok("A/B identical token multisets, values once, entity order identical, candidates matched")

    for rec in records:
        if rec["status"] != "ok":
            continue
        if rec["condition"] in TTT_CONDITIONS:
            w = rec.get("write") or {}
            write_text = rec.get("write_text") or ""
            pair = next(p for p in pairs if p["pair_id"] == rec["pair_id"])
            for ent_key in ("e1", "e2"):
                for q in pair["questions"][ent_key].values():
                    if q in write_text:
                        fail(f"{rec['pair_id']} {rec['condition']}: question in write")
            if not w.get("loss_decreased"):
                fail(
                    f"{rec['pair_id']} {rec['condition']}: write loss did not decrease "
                    f"{w.get('loss_before')} -> {w.get('loss_after')}"
                )
            if not (w.get("lora_max_abs_diff_from_init") or 0) > 0:
                fail(f"{rec['pair_id']} {rec['condition']}: LoRA delta is 0")
            if w.get("returned_past_key_values") or w.get("write_used_cache"):
                fail(f"{rec['pair_id']} {rec['condition']}: write used or returned cache")
            if rec.get("recall_updated_lora"):
                fail(f"{rec['pair_id']} {rec['condition']}: recall updated LoRA")
            if rec.get("recall_used_past_key_values"):
                fail(f"{rec['pair_id']} {rec['condition']}: recall used KV")
        if rec["condition"] in SOURCE_FREE_CONDITIONS:
            for style in QUERY_STYLES:
                for ent_key in ("e1", "e2"):
                    prompt = rec["queries"][style][ent_key]["recall_prompt"]
                    pair = next(p for p in pairs if p["pair_id"] == rec["pair_id"])
                    if pair["write_block_A"] in prompt or pair["write_block_B"] in prompt:
                        fail(f"{rec['pair_id']} {rec['condition']} {style}: source in recall")
    ok("write/recall invariants (question absent, KV discarded, LoRA freeze at recall)")

    base_diff = frozen_watch_max_abs_diff(runtime.model, runtime.frozen_watch_cpu)
    if base_diff != 0.0:
        fail(f"base weights changed at end: {base_diff}")
    else:
        ok("base weights unchanged")
    reset_diff = lora_max_abs_diff(runtime.model, runtime.lora_init_cpu)
    if reset_diff != 0.0:
        fail(f"LoRA not restored to init at end: {reset_diff}")
    else:
        ok("LoRA restored to identical init after each mapping")

    table = pair_binding_table(records, pairs)
    for key, block in table["by_type_style"].items():
        acc = block["source_present_two_way_accuracy"]
        if acc < 0.5:
            fail(f"{key}: source-present two-way acc {acc:.3f} < 0.5 (scoring/protocol likely broken)")
        elif not block["source_present_calibrated"]:
            log_fn(
                f"WARN {key}: source-present two-way acc {acc:.3f} < 0.75; "
                "do not interpret source-free binding for this style"
            )
        else:
            ok(f"{key}: source-present two-way acc={acc:.3f} B_swap={block['source_present_B_swap']:.3f}")

    probe = pairs[0]
    set_adapter_enabled(runtime.model, False)
    prompt = encode_recall_prompt(
        runtime.tokenizer,
        probe,
        question=probe["questions"]["e1"]["lexical_near"],
        source_present=True,
        mapping="A",
    )
    a = greedy_generate(runtime, prompt["input_ids"])
    b = greedy_generate(runtime, prompt["input_ids"])
    if a["generated_token_ids"] != b["generated_token_ids"]:
        fail("greedy generation is not deterministic")
    else:
        ok("deterministic greedy generation")
    set_adapter_enabled(runtime.model, True)
    reset_example_state(runtime)

    for rec in records:
        if rec.get("status") != "ok":
            continue
        for style in QUERY_STYLES:
            for ent_key in ("e1", "e2"):
                q = rec["queries"][style][ent_key]
                if "<think>" in (q.get("recall_prompt") or "") or "<think>" in (q.get("raw_prediction") or ""):
                    fail(f"{rec['pair_id']} thinking block")
    ok("no thinking blocks")

    if failures:
        log_fn(f"\n{len(failures)} sanity failure(s).")
    else:
        log_fn("\nAll smoke sanity checks passed.")
    return failures


def write_jsonl(path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def render_summary_md(table: dict[str, Any], *, dataset_sha: str, n_pairs: int) -> str:
    lines = [
        "# Exp 02A-1 smoke — source-free binding specificity",
        "",
        f"Frozen dataset sha256: `{dataset_sha}`",
        f"Smoke pairs: {n_pairs} (4 exact + 4 semantic_direct).",
        "K=32. Bootstrap CIs are **descriptive only**; do not make a statistical claim from n=4.",
        "",
        "Positive `B_swap` supports entity-value relational binding beyond a global value-token imprint.",
        "Do **not** call this semantic memory. Do not interpret a style with poor source-present calibration.",
        "",
        "| type | query | SP B_swap | SP 2-way | calibrated | static B_A | B_A | B_B | B_swap | e1 flip | e2 flip | both flip | TTT 2-way | EM |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, b in table["by_type_style"].items():
        info_type, style = key.split("/", 1)
        lines.append(
            "| {type} | {style} | {sp_b:.3f} | {sp_acc:.3f} | {cal} | {st:.3f} | {ba:.3f} | {bb:.3f} | {bs:.3f} | {e1:.3f} | {e2:.3f} | {both:.3f} | {ttt:.3f} | {em:.3f} |".format(
                type=info_type,
                style=style,
                sp_b=b["source_present_B_swap"],
                sp_acc=b["source_present_two_way_accuracy"],
                cal="yes" if b["source_present_calibrated"] else "NO",
                st=b["static_B_A"],
                ba=b["Mapping_A_B_A"],
                bb=b["Mapping_B_B_B"],
                bs=b["B_swap"],
                e1=b["fraction_e1_flipped"],
                e2=b["fraction_e2_flipped"],
                both=b["fraction_both_flipped"],
                ttt=b["ttt_two_way_accuracy"],
                em=b["generation_em_ttt"],
            )
        )
    lines.extend(
        [
            "",
            "## Descriptive bootstrap 95% CI on B_swap (n=4, not a claim)",
            "",
        ]
    )
    for key, b in table["by_type_style"].items():
        ci = b["bootstrap_B_swap_descriptive_only"]
        note = b["interpretation_note"] or ""
        lines.append(
            f"- `{key}`: mean={ci['mean_delta']:.3f}  CI=[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  {note}"
        )
    lines.append("")
    return "\n".join(lines)
