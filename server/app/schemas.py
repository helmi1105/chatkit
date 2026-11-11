# app/schemas.py
"""Structured outputs for every LLM generation, plus the normalisation that
turns a model answer into something the scorer can trust.

Why this exists (cahier de labo 2026-09-15): the generators used to be free
text parsed with json.loads(), and `_coerce_qcm_answer_letter` searched the
FIRST letter A-D inside whatever the model wrote and fell back to "A". An
answer given as the choice text ("Carré", "Bleu") became "C" or "B" at random
and the learner was then corrected against a wrong key. Here the answer is
resolved against the actual choices, and a question whose key cannot be
resolved is DROPPED, never defaulted."""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

LETTERS = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# Pydantic models (model output shapes)
# ---------------------------------------------------------------------------
class QcmQuestion(BaseModel):
    number: int = 1
    text: str
    choices: List[str] = Field(default_factory=list, description="Exactement 4 propositions courtes, distinctes")
    answer: str = Field(description="Lettre de la bonne réponse : A, B, C ou D")
    explanation: str = Field(default="", description="Une phrase qui justifie la bonne réponse avec la doctrine, page citée")
    kc_id: str = ""
    target_id: str = ""
    role: str = Field(default="", description="current, review ou integration; declarer le role reel de la question")
    assessed_kc_ids: List[str] = Field(default_factory=list, description="Identifiants des notions effectivement evaluees")
    assessed_objective_ids: List[str] = Field(default_factory=list, description="Identifiants des objectifs effectivement evalues, si disponibles")
    integration_attempt: bool = Field(
        default=False,
        description="Vrai si le modele choisit cette question comme integration de plusieurs notions",
    )
    page: Optional[int] = None


class QcmList(BaseModel):
    questions: List[QcmQuestion]


class CoverageItem(BaseModel):
    target_id: str
    target: str = ""
    question_numbers: List[int] = Field(default_factory=list)
    coverage_note: str = ""


class PracticeQuestion(QcmQuestion):
    """Practice output must declare attribution before it can reach scoring."""
    target_id: str = Field(min_length=1, description="Identifiant E1.. de la cible réellement évaluée")
    role: Literal["current", "review", "integration"]
    assessed_kc_ids: List[str] = Field(
        min_length=1,
        description="Identifiants des notions effectivement évaluées",
    )


class PracticePack(BaseModel):
    n_questions: int = Field(
        default=0,
        description="Nombre de questions que le modele juge pedagogiquement necessaire",
    )
    coverage_plan: List[CoverageItem] = Field(default_factory=list)
    questions: List[PracticeQuestion] = Field(default_factory=list)


class LessonPlan(BaseModel):
    lesson_complexity: str = "medium"
    lesson_mode: str = "first_exposure"
    max_words: int = 240
    essential_information_count: int = 4
    reason: str = ""


class LessonBody(BaseModel):
    title: str = ""
    source_basis: List[str] = Field(default_factory=list)
    learning_objective: str = ""
    essential_information: List[str] = Field(default_factory=list)
    rule_to_remember: str = ""
    operational_example: str = ""
    common_mistake: str = ""
    targeted_remediation: str = ""
    self_check: str = ""


class LessonOut(BaseModel):
    lesson_plan: LessonPlan = Field(default_factory=LessonPlan)
    lesson: LessonBody = Field(default_factory=LessonBody)


class EssentialTargets(BaseModel):
    essential_targets: List[str] = Field(default_factory=list)


class Feedback(BaseModel):
    summary: str = Field(description="2 à 4 phrases en français : lecture du score et corrections clés")
    corrections: List[str] = Field(default_factory=list, description="Une correction par erreur importante")
    hints: List[str] = Field(default_factory=list, description="3 indices progressifs, spécifiques aux questions ratées, sans donner la réponse")


class FreeAnswer(BaseModel):
    answer: str
    pages: List[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return text.strip()


def resolve_answer_index(answer: Any, choices: List[str]) -> Optional[int]:
    """Map the model's `answer` to a 0..3 index, or None if it cannot be
    resolved unambiguously. Accepts "B", "b", "B)", "B) Bleu", "Bleu", "2"."""
    raw = str(answer or "").strip()
    if not raw:
        return None
    up = raw.upper()
    if up in LETTERS:
        return LETTERS.index(up)
    m = re.match(r"^\s*([ABCD])\s*[\)\.\:\-]", up)
    if m:
        return LETTERS.index(m.group(1))
    if re.fullmatch(r"[1-4]", raw):
        return int(raw) - 1
    folded = _fold(raw)
    folded_choices = [_fold(c) for c in choices]
    if folded in folded_choices:
        return folded_choices.index(folded)
    # "B) Bleu" -> compare the part after the letter
    m = re.match(r"^\s*[ABCD]\s*[\)\.\:\-]\s*(.+)$", raw, flags=re.I)
    if m:
        f2 = _fold(m.group(1))
        if f2 in folded_choices:
            return folded_choices.index(f2)
    # unique substring containment as a last resort (both sides long enough
    # to mean something: "b" inside "zebre" must not count)
    hits = [i for i, c in enumerate(folded_choices) if len(folded) >= 4 and len(c) >= 4 and (folded in c or c in folded)]
    if len(hits) == 1:
        return hits[0]
    return None


def question_id(kc_id: str, text: str) -> str:
    return f"{kc_id}:{hashlib.sha1(_fold(text).encode('utf-8')).hexdigest()[:10]}"


def normalize_question(
    q: Any,
    kc_id: str,
    *,
    shuffle: bool = True,
    rng: Optional[random.Random] = None,
    max_prompt_chars: int = 240,
    max_choice_chars: int = 90,
) -> Optional[Dict[str, Any]]:
    """Validate one generated question. Returns a clean dict or None (dropped).
    Choices are shuffled (answer letter recomputed) so that the key is not
    biased toward A/B the way LLM output usually is."""
    if isinstance(q, BaseModel):
        q = q.model_dump()
    if not isinstance(q, dict):
        return None
    text = re.sub(r"\s+", " ", str(q.get("text") or "")).strip()
    choices_raw = q.get("choices") or []
    if not text or not isinstance(choices_raw, list):
        return None
    choices = [re.sub(r"\s+", " ", str(c or "")).strip() for c in choices_raw]
    choices = [re.sub(r"^\s*[ABCD]\s*[\)\.\:\-]\s*", "", c) for c in choices]  # strip "A) "
    choices = [c for c in choices if c]
    if len(choices) != 4 or len({_fold(c) for c in choices}) != 4:
        return None
    idx = resolve_answer_index(q.get("answer"), choices)
    if idx is None:
        return None
    if shuffle:
        rng = rng or random.Random(question_id(kc_id, text))
        order = list(range(4))
        rng.shuffle(order)
        choices = [choices[i] for i in order]
        idx = order.index(idx)
    page = q.get("page")
    try:
        page = int(page) if page is not None else None
    except (TypeError, ValueError):
        page = None
    return {
        "id": question_id(kc_id, text),
        "number": int(q.get("number") or 0) or 0,
        "text": text[:max_prompt_chars].rstrip() if len(text) > max_prompt_chars else text,
        "choices": [c if len(c) <= max_choice_chars else c[: max_choice_chars - 1].rstrip() + "…" for c in choices],
        "answer": LETTERS[idx],
        "explanation": re.sub(r"\s+", " ", str(q.get("explanation") or "")).strip(),
        "kc_id": str(q.get("kc_id") or kc_id or "").strip() or kc_id,
        "target_id": str(q.get("target_id") or "").strip(),
        "role": str(q.get("role") or "").strip(),
        "assessed_kc_ids": list(q.get("assessed_kc_ids") or []),
        "assessed_objective_ids": list(q.get("assessed_objective_ids") or []),
        "integration_attempt": bool(q.get("integration_attempt", False)),
        "page": page,
    }


def normalize_questions(items: Any, kc_id: str, *, shuffle: bool = True) -> List[Dict[str, Any]]:
    """Validate, dedupe (by normalised text) and renumber 1..n."""
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for q in (items or []):
        clean = normalize_question(q, kc_id, shuffle=shuffle)
        if not clean:
            continue
        key = _fold(clean["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    for i, q in enumerate(out, start=1):
        q["number"] = i
    return out
