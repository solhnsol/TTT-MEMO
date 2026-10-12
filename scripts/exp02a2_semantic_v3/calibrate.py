"""Source-present semantic calibration. No LoRA, no source-free TTT."""

from __future__ import annotations

from typing import Any

import torch

from encode_write import encode_candidate, encode_recall_prompt
from metrics import generation_scores, rank_candidates
from protocol import SPEC

QUERY_CODE = "occupation_query"
QUERY_STYLE = "occupation_query"


def load_base_model(*, device: str | None = None):
    from transformers import AutoModelForCausalLM

    from encode_write import load_tokenizer

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("source-present calibration requires CUDA")
        device_obj = torch.device("cuda")
    else:
        device_obj = torch.device(device)
    tokenizer = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        SPEC["backbone"]["model_id"],
        revision=SPEC["backbone"]["revision_sha"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    dtypes = {str(p.dtype) for p in model.parameters()}
    if "torch.float8_e4m3fn" in dtypes or "torch.float8_e5m2" in dtypes:
        raise RuntimeError(f"FP8 weights loaded; forbidden. dtypes={dtypes}")
    model.to(device_obj)
    model.eval()
    model.config.use_cache = False
    return tokenizer, model, device_obj


def _to_batch(encoded: dict[str, Any], device, *, include_labels: bool):
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if include_labels:
        out["labels"] = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
    return out


@torch.no_grad()
def score_candidate_logprobs(model, device, encoded: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    batch = _to_batch(encoded, device, include_labels=True)
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
    n_tokens = int(mask.sum().item())
    if n_tokens != encoded["n_tokens"]:
        raise RuntimeError(f"labeled {n_tokens} != encoded {encoded['n_tokens']}")
    result = {
        "candidate": encoded["candidate"],
        "logp_sum": float(token_logp.sum().cpu()),
        "logp_mean": float(token_logp.mean().cpu()),
        "nll_mean": float(-token_logp.mean().cpu()),
        "n_tokens": n_tokens,
    }
    del out, batch
    return result


@torch.no_grad()
def greedy_generate(model, tokenizer, device, prompt_ids: list[int]) -> dict[str, Any]:
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = gen[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    del gen, input_ids, attention_mask
    return {"generated_token_ids": new_ids, "raw_prediction": text}


def _fact_block(pair: dict[str, Any], mapping: str) -> str:
    seqs = pair["sequences"][mapping]
    return "\n".join(s["write_sentence"] for s in seqs)


def _gold_label(pair: dict[str, Any], mapping: str, entity_key: str) -> str:
    concept_key = pair[mapping][entity_key]
    return pair["concepts"][concept_key]["label"]


def _paired_decoy(candidates: list[str], gold: str) -> str:
    others = [c for c in candidates if c != gold]
    if len(others) != 1:
        raise RuntimeError(f"expected one paired decoy, got {others} gold={gold}")
    return others[0]


def _cand_score(rec: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [c for c in rec["candidate_scores"] if c["candidate"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one score for {name}, got {len(matches)}")
    return matches[0]


def evaluate_query(
    tokenizer,
    model,
    device,
    pair: dict[str, Any],
    *,
    mapping: str,
    entity_key: str,
    source_present: bool,
) -> dict[str, Any]:
    entity = pair["entities"][entity_key]
    question = pair["query_templates"][QUERY_STYLE].format(entity=entity)
    gold = _gold_label(pair, mapping, entity_key)
    fact = _fact_block(pair, mapping) if source_present else None
    prompt = encode_recall_prompt(tokenizer, question=question, fact_block=fact)
    if source_present:
        if pair["sequences"][mapping][0]["write_sentence"] not in prompt["text"]:
            raise RuntimeError("source-present prompt missing first write sentence")
    else:
        for seq in pair["sequences"][mapping]:
            if seq["write_sentence"] in prompt["text"]:
                raise RuntimeError("static source-free prompt contains a write sentence")
            if seq["description"] in prompt["text"]:
                raise RuntimeError("static source-free prompt contains a description")
    cand_rows = []
    for cand in pair["candidates"]:
        encoded = encode_candidate(tokenizer, prompt["text"], cand)
        scored = score_candidate_logprobs(model, device, encoded)
        scored["is_gold"] = cand == gold
        cand_rows.append(scored)
        del encoded
    ranked = rank_candidates(cand_rows, gold=gold)
    gen = greedy_generate(model, tokenizer, device, prompt["input_ids"])
    gen_scores = generation_scores(gen["raw_prediction"], gold)
    return {
        "pair_id": pair["pair_id"],
        "mapping": mapping,
        "entity_key": entity_key,
        "entity": entity,
        "query_code": QUERY_CODE,
        "query_style": QUERY_STYLE,
        "question": question,
        "gold": gold,
        "paired_decoy": _paired_decoy(pair["candidates"], gold),
        "candidates": pair["candidates"],
        "source_present": source_present,
        "lora_used": False,
        "condition": "static_source_present" if source_present else "static_source_free",
        **ranked,
        **gen,
        **gen_scores,
        "candidate_scores": cand_rows,
    }


def attach_source_induced_uplift(records: list[dict[str, Any]]) -> None:
    """Attach uplift diagnostics. Does not change the calibration gate."""
    by_key: dict[tuple, dict[bool, dict[str, Any]]] = {}
    for rec in records:
        key = (rec["pair_id"], rec["mapping"], rec["entity_key"], rec["query_code"])
        by_key.setdefault(key, {})[bool(rec["source_present"])] = rec
    for key, pair_recs in by_key.items():
        if True not in pair_recs or False not in pair_recs:
            raise RuntimeError(f"missing source-present/source-free pair for {key}")
        sp = pair_recs[True]
        sf = pair_recs[False]
        gold = sp["gold"]
        decoy = _paired_decoy(sp["candidates"], gold)
        g_sp = _cand_score(sp, gold)
        d_sp = _cand_score(sp, decoy)
        g_sf = _cand_score(sf, gold)
        d_sf = _cand_score(sf, decoy)
        gold_uplift = g_sp["logp_mean"] - g_sf["logp_mean"]
        decoy_uplift = d_sp["logp_mean"] - d_sf["logp_mean"]
        diagnostic = {
            "gold": gold,
            "decoy": decoy,
            "gold_score_with_source": g_sp["logp_mean"],
            "decoy_score_with_source": d_sp["logp_mean"],
            "source_present_margin": g_sp["logp_mean"] - d_sp["logp_mean"],
            "gold_score_without_source": g_sf["logp_mean"],
            "decoy_score_without_source": d_sf["logp_mean"],
            "gold_uplift": gold_uplift,
            "decoy_uplift": decoy_uplift,
            "uplift_margin": gold_uplift - decoy_uplift,
            "gold_logp_sum_with_source": g_sp["logp_sum"],
            "decoy_logp_sum_with_source": d_sp["logp_sum"],
            "gold_logp_sum_without_source": g_sf["logp_sum"],
            "decoy_logp_sum_without_source": d_sf["logp_sum"],
        }
        sp["source_induced_uplift"] = diagnostic
        sf["source_induced_uplift"] = diagnostic


def run_calibration(
    pairs: list[dict[str, Any]],
    *,
    include_static_source_free: bool = True,
    log_fn=print,
) -> list[dict[str, Any]]:
    log_fn("loading base Qwen3-4B-Instruct-2507 BF16 (no LoRA) for source-present calibration...")
    tokenizer, model, device = load_base_model()
    log_fn(
        f"loaded revision={SPEC['backbone']['revision_sha']} gpu={torch.cuda.get_device_name(device)}"
    )
    records = []
    n = 4 * 2 * 2  # pairs, mappings, entities
    if include_static_source_free:
        n *= 2
    k = 0
    for pair in pairs:
        for mapping in ("mapping_A", "mapping_B"):
            for entity_key in ("e1", "e2"):
                for source_present in (True, False) if include_static_source_free else (True,):
                    k += 1
                    rec = evaluate_query(
                        tokenizer,
                        model,
                        device,
                        pair,
                        mapping=mapping,
                        entity_key=entity_key,
                        source_present=source_present,
                    )
                    records.append(rec)
                    log_fn(
                        f"[{k}/{n}] {rec['pair_id']} {mapping} {rec['entity']} "
                        f"{QUERY_CODE} src={int(source_present)} acc={rec['accuracy']} "
                        f"rank={rec['gold_rank']} em={rec['em']}"
                    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if include_static_source_free:
        attach_source_induced_uplift(records)
    return records


def summarize_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    sp = [r for r in records if r["source_present"]]
    sf = [r for r in records if not r["source_present"]]
    n_required = int(SPEC["calibration_gates"]["occupation_query_n_cells"])
    n_correct_required = int(SPEC["calibration_gates"]["occupation_query_required_correct"])
    n_correct = int(sum(int(r["accuracy"]) for r in sp))
    failures = []
    per_cell = []
    for r in sp:
        uplift = r.get("source_induced_uplift")
        cell = {
            "pair_id": r["pair_id"],
            "mapping": r["mapping"],
            "entity_key": r["entity_key"],
            "entity": r["entity"],
            "query_code": r["query_code"],
            "gold": r["gold"],
            "paired_decoy": r["paired_decoy"],
            "accuracy": r["accuracy"],
            "gold_rank": r["gold_rank"],
            "gold_logp_mean": r["gold_logp_mean"],
            "margin_logp_mean": r["margin_logp_mean"],
            "gold_logp_sum": r["gold_logp_sum"],
            "margin_logp_sum": r["margin_logp_sum"],
            "em": r["em"],
            "raw_prediction": r["raw_prediction"],
            "source_induced_uplift": uplift,
        }
        per_cell.append(cell)
        if r["accuracy"] != 1:
            failures.append(cell)
    passed = len(sp) == n_required and n_correct == n_correct_required
    uplifts = [r["source_induced_uplift"] for r in sp if r.get("source_induced_uplift")]
    uplift_summary = {
        "n": len(uplifts),
        "mean_gold_uplift": mean([u["gold_uplift"] for u in uplifts]),
        "mean_decoy_uplift": mean([u["decoy_uplift"] for u in uplifts]),
        "mean_uplift_margin": mean([u["uplift_margin"] for u in uplifts]),
        "n_positive_gold_uplift": int(sum(u["gold_uplift"] > 0 for u in uplifts)),
        "n_positive_uplift_margin": int(sum(u["uplift_margin"] > 0 for u in uplifts)),
        "note": (
            "Diagnostic only. Calibration pass/fail is source-present 2-way "
            "mean-logp ranking, not uplift."
        ),
    }
    return {
        "n_source_present": len(sp),
        "n_static_source_free": len(sf),
        "occupation_query": {
            "query_style": QUERY_STYLE,
            "n": len(sp),
            "n_correct": n_correct,
            "two_way_accuracy": mean([float(r["accuracy"]) for r in sp]),
            "mean_gold_rank": mean([float(r["gold_rank"]) for r in sp]),
            "mean_margin_logp_mean": mean([float(r["margin_logp_mean"]) for r in sp]),
            "generation_em": mean([float(r["em"]) for r in sp]),
            "gate": f"{n_correct_required}/{n_required}",
            "passed_gate": passed,
            "failures": failures,
        },
        "passed": passed,
        "static_source_free": {
            "occupation_query_two_way_accuracy": mean([float(r["accuracy"]) for r in sf]),
            "note": "Diagnostic only. Not used to select or rewrite the dataset.",
        },
        "source_induced_uplift": uplift_summary,
        "per_query": per_cell,
        "gates": SPEC["calibration_gates"],
        "lora_used": False,
        "source_removed": False,
        "greedy_em_is_not_the_gate": True,
    }


def render_calibration_md(summary: dict[str, Any]) -> str:
    b = summary["occupation_query"]
    u = summary["source_induced_uplift"]
    lines = [
        "# semantic_inferential_v3 source-present recalibration v2",
        "",
        "No LoRA. No test-time update. Descriptions are in the prompt.",
        "Single canonical query: occupation_query.",
        "",
        f"Passed 16/16 gate: **{summary['passed']}**",
        f"Source-present 2-way accuracy: **{b['n_correct']}/{b['n']}** "
        f"({b['two_way_accuracy']:.3f})",
        "",
        "| query | n | correct | 2-way acc | gate | mean rank | mean margin | greedy EM | passed |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
        f"| occupation_query | {b['n']} | {b['n_correct']} | {b['two_way_accuracy']:.3f} | "
        f"{b['gate']} | {b['mean_gold_rank']:.3f} | {b['mean_margin_logp_mean']:.3f} | "
        f"{b['generation_em']:.3f} | {'yes' if b['passed_gate'] else 'NO'} |",
        "",
        "Greedy EM is recorded, not the freeze gate.",
        "",
        f"Static source-free occupation_query acc (diagnostic): "
        f"{summary['static_source_free']['occupation_query_two_way_accuracy']:.3f}",
        "",
        "## Source-induced uplift (diagnostic)",
        "",
        f"- mean gold uplift: {u['mean_gold_uplift']:.4f}",
        f"- mean decoy uplift: {u['mean_decoy_uplift']:.4f}",
        f"- mean uplift margin: {u['mean_uplift_margin']:.4f}",
        f"- cells with gold_uplift > 0: {u['n_positive_gold_uplift']}/{u['n']}",
        f"- cells with uplift_margin > 0: {u['n_positive_uplift_margin']}/{u['n']}",
        "",
    ]
    if b["failures"]:
        lines.extend(["## Source-present failures", ""])
        for f in b["failures"]:
            up = f.get("source_induced_uplift") or {}
            lines.append(
                f"- {f['pair_id']} {f['mapping']} {f['entity']} gold=`{f['gold']}` "
                f"decoy=`{f['paired_decoy']}` rank={f['gold_rank']} "
                f"gold_logp_mean={f['gold_logp_mean']:.4f} "
                f"margin={f['margin_logp_mean']:.4f} gen={f['raw_prediction']!r} "
                f"gold_uplift={up.get('gold_uplift')} "
                f"decoy_uplift={up.get('decoy_uplift')} "
                f"uplift_margin={up.get('uplift_margin')}"
            )
    else:
        lines.append("No source-present two-way ranking failures.")
    lines.append("")
    return "\n".join(lines)
