"""Lexical leakage audit for semantic_inferential_v3. No model, no rewriting."""

from __future__ import annotations

import json
import re
from typing import Any

from protocol import (
    CONCEPTS,
    LEXICAL_FORBIDDEN,
    LEXICAL_FORBIDDEN_WORD_BOUNDARY,
    SCAFFOLD_FORBIDDEN,
    SYNONYM_FORBIDDEN,
)


def _find_ci(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def _find_word(haystack: str, needle: str) -> bool:
    return re.search(rf"\b{re.escape(needle)}\b", haystack, flags=re.IGNORECASE) is not None


def audit_text(text: str, concept_id: str, *, where: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    label = CONCEPTS[concept_id]["label"]
    if label in text:
        hits.append({"check": "A_exact_label_substring", "where": where, "term": label, "passed": False})
    else:
        hits.append({"check": "A_exact_label_substring", "where": where, "term": label, "passed": True})
    if _find_ci(text, label):
        hits.append({"check": "B_case_insensitive_label", "where": where, "term": label, "passed": False})
    else:
        hits.append({"check": "B_case_insensitive_label", "where": where, "term": label, "passed": True})
    for term in LEXICAL_FORBIDDEN.get(concept_id, []):
        ok = not _find_ci(text, term)
        hits.append(
            {
                "check": "C_D_forbidden_root",
                "where": where,
                "term": term,
                "passed": ok,
            }
        )
    for term in LEXICAL_FORBIDDEN_WORD_BOUNDARY.get(concept_id, []):
        ok = not _find_word(text, term)
        hits.append(
            {
                "check": "C_D_forbidden_word_boundary",
                "where": where,
                "term": term,
                "passed": ok,
            }
        )
    for term in SYNONYM_FORBIDDEN.get(concept_id, []):
        ok = not _find_ci(text, term)
        hits.append({"check": "E_synonym", "where": where, "term": term, "passed": ok})
    return hits


def audit_scaffold(sentence: str, *, where: str) -> list[dict[str, Any]]:
    hits = []
    for term in SCAFFOLD_FORBIDDEN:
        ok = not _find_ci(sentence, term)
        hits.append({"check": "F_scaffold_job_terminology", "where": where, "term": term, "passed": ok})
    return hits


def run_lexical_audit(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for pair in pairs:
        for mapping in ("mapping_A", "mapping_B"):
            for seq in pair["sequences"][mapping]:
                concept_id = pair["concepts"][seq["concept_key"]]["concept_id"]
                texts = {
                    "write_sentence": seq["write_sentence"],
                    "target": seq["target"],
                    "prefix": seq["prefix"],
                    "description": seq["description"],
                }
                seq_hits: list[dict[str, Any]] = []
                for where, text in texts.items():
                    seq_hits.extend(audit_text(text, concept_id, where=where))
                seq_hits.extend(audit_scaffold(seq["write_sentence"], where="write_sentence"))
                # Gold label of the OTHER concept in the pair must also be absent
                # from this sequence (inventory is shared, but each sequence is
                # only one description).
                other_key = "c2" if seq["concept_key"] == "c1" else "c1"
                other_id = pair["concepts"][other_key]["concept_id"]
                for where, text in texts.items():
                    seq_hits.extend(audit_text(text, other_id, where=f"{where}:other_label"))
                failed = [h for h in seq_hits if not h["passed"]]
                row = {
                    "pair_id": pair["pair_id"],
                    "sequence_id": seq["sequence_id"],
                    "mapping": mapping,
                    "entity": seq["entity"],
                    "label": seq["label"],
                    "write_sentence": seq["write_sentence"],
                    "n_checks": len(seq_hits),
                    "n_failures": len(failed),
                    "failures": failed,
                    "passed": not failed,
                }
                rows.append(row)
                failures.extend(
                    [{**f, "pair_id": pair["pair_id"], "sequence_id": seq["sequence_id"]} for f in failed]
                )
    passed = not failures
    return {
        "passed": passed,
        "n_sequences_audited": len(rows),
        "n_failures": len(failures),
        "failures": failures,
        "per_sequence": rows,
        "rule": (
            "If any check fails: STOP and report. Do not silently rewrite descriptions."
        ),
    }


def render_audit_md(audit: dict[str, Any]) -> str:
    lines = [
        "# semantic_inferential_v3 lexical audit",
        "",
        f"Passed: **{audit['passed']}**",
        f"Sequences audited: {audit['n_sequences_audited']}",
        f"Failures: {audit['n_failures']}",
        "",
        "| pair | sequence | entity | gold label | passed | n_fail |",
        "|---|---|---|---|---|---:|",
    ]
    for row in audit["per_sequence"]:
        lines.append(
            f"| {row['pair_id']} | `{row['sequence_id']}` | {row['entity']} | {row['label']} | "
            f"{'yes' if row['passed'] else 'NO'} | {row['n_failures']} |"
        )
    if audit["failures"]:
        lines.extend(["", "## Failures", ""])
        for f in audit["failures"]:
            lines.append(
                f"- `{f.get('sequence_id')}` {f['check']} term=`{f['term']}` where={f['where']}"
            )
    else:
        lines.extend(
            [
                "",
                "All exact-label, case-insensitive, plural/root, synonym, and scaffold checks passed.",
                "Gold occupation labels do not occur in write sentences or description targets.",
                "",
            ]
        )
    return "\n".join(lines)
