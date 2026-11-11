"""Objective-level practice evidence recorded when answers are scored."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any


def target_fingerprint(target: str) -> str:
    normalized = re.sub(r"\s+", " ", str(target or "").strip().casefold())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def expected_objectives(targets: list[str]) -> dict[str, dict[str, str]]:
    return {
        f"E{index}": {"target": target, "target_hash": target_fingerprint(target)}
        for index, target in enumerate(targets, start=1)
    }


def update_objective_evidence(
    evidence_by_kc: dict[str, dict[str, dict[str, Any]]],
    kc_id: str,
    targets: list[str],
    items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply current-KC scored items without merging changed positional targets."""
    expected = expected_objectives(targets)
    kc_evidence = evidence_by_kc.setdefault(kc_id, {})
    for item in items:
        objective_id = str(item.get("target_id") or "")
        definition = expected.get(objective_id)
        if definition is None or kc_id not in item.get("credited_kc_ids", []):
            continue
        entry = kc_evidence.get(objective_id)
        if not isinstance(entry, dict) or entry.get("target_hash") != definition["target_hash"]:
            entry = {
                **definition,
                "correct": 0,
                "total": 0,
                "failures": 0,
                "consecutive_failures": 0,
            }
            kc_evidence[objective_id] = entry
        correct = bool(item.get("correct"))
        entry["total"] += 1
        entry["correct"] += int(correct)
        entry["failures"] += int(not correct)
        entry["consecutive_failures"] = 0 if correct else entry["consecutive_failures"] + 1
    return deepcopy(kc_evidence)


def summarize_objective_evidence(
    targets: list[str], evidence: dict[str, dict[str, Any]], *, repeated_failure_count: int = 2
) -> dict[str, Any]:
    expected = expected_objectives(targets)
    valid = {
        objective_id: entry
        for objective_id, definition in expected.items()
        if isinstance((entry := evidence.get(objective_id)), dict)
        and entry.get("target_hash") == definition["target_hash"]
    }
    missing = [objective_id for objective_id in expected if valid.get(objective_id, {}).get("total", 0) < 1]
    # Historical failures remain in ``failures`` for analysis. Routing must use
    # unresolved evidence, otherwise one old mistake can force CONFIRM forever
    # even after the learner answers a targeted follow-up correctly.
    failed = [
        objective_id for objective_id, entry in valid.items()
        if entry.get("consecutive_failures", entry.get("failures", 0)) > 0
    ]
    historical = [objective_id for objective_id, entry in valid.items() if entry.get("failures", 0) > 0]
    repeated = [
        objective_id for objective_id, entry in valid.items()
        if entry.get("consecutive_failures", entry.get("failures", 0)) >= repeated_failure_count
    ]
    return {
        "coverage_complete": bool(expected) and not missing,
        "missing_objectives": missing,
        "failed_objectives": failed,
        "repeated_failed_objectives": repeated,
        "historically_failed_objectives": historical,
        "expected_objective_ids": list(expected),
    }
