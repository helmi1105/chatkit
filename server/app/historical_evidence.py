"""Historical learner errors exposed to content-generation agents."""
from __future__ import annotations

import hashlib
import re
from typing import Any


def _history_id(kc_id: str, target_id: str, question: str) -> str:
    if target_id:
        return f"{kc_id}.{target_id}"
    digest = hashlib.sha256(str(question).strip().casefold().encode("utf-8")).hexdigest()[:10]
    return f"{kc_id}.Q{digest}"


def historical_error_candidates(sess: Any, graph: Any, current_kc_id: str, *, limit: int = 12) -> list[dict]:
    """Return deduplicated errors from KCs preceding the current KC.

    The underlying learner history is identical across graph modes. Relation
    metadata is the experimental treatment: ECG relations, flat order, or no
    relational guidance.
    """
    current_index = graph.index_by_kc.get(current_kc_id)
    if current_index is None:
        return []
    previous_ids = list(graph.ordered_kc_ids[:current_index])
    no_ecg = bool(getattr(graph, "implicit_prior_integration", False))
    full_ecg = bool(getattr(graph, "parent_by_id", {})) and not no_ecg
    graph_related = set(graph.previous_kcs(current_kc_id, limit=2)) if full_ecg else set()
    candidates: dict[str, dict] = {}

    for kc_id in previous_ids:
        targets = list(getattr(sess, "kc_essential_targets", {}).get(kc_id, []))
        objective_evidence = getattr(sess, "objective_evidence_by_kc", {}).get(kc_id, {})
        misconceptions = list(getattr(sess, "misconceptions", {}).get(kc_id, []))
        for target_id, evidence in objective_evidence.items():
            if not isinstance(evidence, dict) or int(evidence.get("failures", 0) or 0) <= 0:
                continue
            target = str(evidence.get("target") or "")
            if not target and re.fullmatch(r"E\d+", str(target_id)):
                index = int(str(target_id)[1:]) - 1
                target = targets[index] if 0 <= index < len(targets) else ""
            history_id = _history_id(kc_id, str(target_id), target)
            candidates[history_id] = {
                "history_id": history_id,
                "kc_id": kc_id,
                "kc_title": graph.nodes[kc_id].title if kc_id in graph.nodes else kc_id,
                "target_id": str(target_id),
                "objective": target,
                "failures": int(evidence.get("failures", 0) or 0),
                "total": int(evidence.get("total", 0) or 0),
                "resolved": int(evidence.get("consecutive_failures", 0) or 0) == 0,
                "misconceptions": misconceptions,
                "source_page": None,
            }

        for attempt in getattr(sess, "practice_evidence_by_kc", {}).get(kc_id, []):
            for item in attempt.get("items", []):
                if item.get("correct") is not False:
                    continue
                target_id = str(item.get("target_id") or "")
                question = str(item.get("question") or "")
                history_id = _history_id(kc_id, target_id, question)
                if history_id in candidates:
                    continue
                objective = ""
                if re.fullmatch(r"E\d+", target_id):
                    index = int(target_id[1:]) - 1
                    objective = targets[index] if 0 <= index < len(targets) else ""
                candidates[history_id] = {
                    "history_id": history_id,
                    "kc_id": kc_id,
                    "kc_title": graph.nodes[kc_id].title if kc_id in graph.nodes else kc_id,
                    "target_id": target_id,
                    "objective": objective,
                    "question": question,
                    "learner_answer": item.get("learner_choice") or item.get("learner_letter"),
                    "correct_answer": item.get("correct_choice") or item.get("correct_letter"),
                    "explanation": str(item.get("explanation") or ""),
                    "failures": 1,
                    "total": 1,
                    "resolved": kc_id in getattr(sess, "validated_kc_ids", []),
                    "misconceptions": misconceptions,
                    "source_page": item.get("page"),
                }

    rows = list(candidates.values())[-limit:]
    for row in rows:
        distance = current_index - graph.index_by_kc.get(row["kc_id"], current_index)
        if no_ecg:
            row["relation"] = {"basis": "none"}
        elif full_ecg:
            row["relation"] = {
                "basis": "ecg",
                "graph_related": row["kc_id"] in graph_related,
                "same_module": graph.module_of(row["kc_id"]) == graph.module_of(current_kc_id),
            }
        else:
            row["relation"] = {"basis": "curriculum_order", "distance": distance}
    return rows


def prompt_history(candidates: list[dict]) -> list[dict]:
    """Compact, mode-aware history supplied to a generator."""
    output = []
    for index, row in enumerate(candidates, 1):
        relation = row.get("relation") or {"basis": "none"}
        item = {
            "history_id": row["history_id"] if relation.get("basis") != "none" else f"H{index:02d}",
            "objective": row.get("objective"),
            "question": row.get("question"),
            "learner_answer": row.get("learner_answer"),
            "correct_answer": row.get("correct_answer"),
            "resolved": row.get("resolved"),
            "failures": row.get("failures"),
        }
        if relation.get("basis") != "none":
            item.update(kc_id=row.get("kc_id"), kc_title=row.get("kc_title"), relation=relation)
        output.append(item)
    return output


def raw_chat_history(
    sess: Any,
    graph: Any,
    current_kc_id: str,
    *,
    limit_turns: int = 24,
    max_chars: int = 9000,
) -> str:
    """Render prior practice as dialogue without KC/objective/state metadata."""
    current_index = graph.index_by_kc.get(current_kc_id)
    if current_index is None:
        return ""
    turns: list[str] = []
    for kc_id in graph.ordered_kc_ids[:current_index]:
        for attempt in getattr(sess, "practice_evidence_by_kc", {}).get(kc_id, []):
            for item in attempt.get("items", []):
                question = str(item.get("question") or "").strip()
                if not question:
                    continue
                learner = str(
                    item.get("learner_choice") or item.get("learner_letter") or "sans reponse"
                ).strip()
                correct = str(
                    item.get("correct_choice") or item.get("correct_letter") or ""
                ).strip()
                explanation = str(item.get("explanation") or "").strip()
                correction = "Reponse correcte."
                if item.get("correct") is False:
                    correction = "Correction"
                    if correct:
                        correction += f" : {correct}."
                    if explanation:
                        correction += f" {explanation}"
                turns.append(
                    f"Tuteur : {question}\nApprenant : {learner}\nTuteur : {correction}"
                )
    text = "\n\n".join(turns[-max(1, int(limit_turns)):])
    return text[-max_chars:] if len(text) > max_chars else text
