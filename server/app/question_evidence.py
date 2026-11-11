"""Conservative, question-level attribution used by quiz scoring.

The semantic benchmark evaluates whether declarations are *true*.  Tutoring may
only use structurally valid declarations; it must not consult the judge.
"""

from typing import Any


def credited_kcs(question: dict[str, Any], current_kc: str | None, previous_kcs: list[str], scope: str) -> list[str]:
    """Return KCs receiving this answer as evidence.

    Integration is credited to the current KC only: one integrated answer does
    not independently establish mastery of every KC named in the question.
    Legacy saved questions without either declaration retain their old kc_id.
    Incomplete or contradictory new declarations receive no KC credit.
    """
    role = str(question.get("role") or "").strip().lower()
    raw_ids = question.get("assessed_kc_ids") or []
    ids = [str(k).strip() for k in raw_ids] if isinstance(raw_ids, list) else []
    ids = list(dict.fromkeys(k for k in ids if k))
    legacy_kc = str(question.get("kc_id") or "").strip()
    if not role and not ids:
        return [legacy_kc] if legacy_kc else []
    if not role or not ids:
        return []
    if scope != "practice":
        return ids if role == "current" and len(ids) == 1 and ids[0] == legacy_kc else []
    previous = set(previous_kcs)
    if role == "current" and ids == [current_kc]:
        return [current_kc]
    if role == "review" and set(ids) <= previous and current_kc not in ids:
        return ids
    if role == "integration" and current_kc in ids and len(ids) > 1 and set(ids) <= previous | {current_kc}:
        return [current_kc]
    return []
