# app/orchestrator.py
"""Intelligent tutoring workflow for the memento GOC (sapeurs-pompiers).

Rewritten 2026-09-15 (cahier de labo, audit of the same day). What changed
and why, in one place:

- French everywhere the learner reads, no "KC"/"module" jargon: « notion » and
  « chapitre ». Commands are French with English aliases, and every step ends
  with action BUTTONS (widgets) so nobody has to type "practice".
- Grounding by prompt injection (content.py): the memento fits in a prompt, so
  no tool loop; one round trip per generation instead of minutes.
- Typed outputs (schemas.py): the answer key is resolved against the choices,
  choices are shuffled, unusable questions are dropped, never defaulted to A.
- Diagnostic samples the WHOLE course (one question per sampled notion across
  every chapter) and builds a queue of weak notions; before it only looked at
  the first eight notions.
- Answer sheet (corrigé) after every quiz, question by question, with the
  justification generated with the question (no extra call), and a
  « Signaler » button per question feeding the trainers.
- Specific hints generated from the actual mistakes; remediation lesson only
  when the score is really low (< 50 %), otherwise feedback + hints.
- Free questions accepted at any time (even with a quiz pending), on the
  whole memento; notions that come later are flagged, not refused.
- Validated question bank served without any model call when available.
- State keyed by learner (userId) and persisted (storage.py), so a trainer
  who comes back the next day resumes where they stopped.
- Honest failure: when the model fails, say so and offer to retry; no fake
  quiz.

Evidence events keep the names and key fields benchmark_runner.py reads."""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import unicodedata
import uuid
from app.visual_retrieval import OBSERVE_INSTRUCTIONS, VisualInventory, retrieve_visual_evidence
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from chatkit.agents import AgentContext

from app.podcast import Podcasts
from app.flashcards import Flashcards
from app.svg_answer import IllustratedAnswer, SVG_INSTRUCTIONS, validate_svg, illustration_widget
from app.qa_retrieval import retrieve_question_context
from app.bank import QuestionBank
from app.decision_policy import build_learner_state, decide_next_action
from app.diagnostic import HierarchicalDiagnostic
from app.content import doctrine
from app.providers import (
    friendly_llm_error,
    provider_label,
    run_structured,
    run_text,
    set_provider_from_context,
    current_provider,
)
from app.schemas import LETTERS, EssentialTargets, Feedback, LessonOut, PracticePack, QcmList, normalize_questions
from app.storage import store
from app.viz.radar_html import build_radar_dashboard_html
from app.widgets import its_widgets as W

# =====================================================
# CONFIG
# =====================================================
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
KC_GRAPH_PATH = os.getenv("KC_GRAPH_PATH", os.path.join(os.path.dirname(__file__), "kc_graph1.json"))
USER_ID_KEY = "userId"

DIAGNOSTIC_Q_NUM = int(os.getenv("DIAGNOSTIC_Q_NUM", "10"))
DIAGNOSTIC_MODE = os.getenv("DIAGNOSTIC_MODE", "hierarchical")
DIAGNOSTIC_MAX_QUESTIONS = int(os.getenv("DIAGNOSTIC_MAX_QUESTIONS", "96"))
THRESHOLD = float(os.getenv("MASTERY_THRESHOLD", "0.7"))
PRACTICE_MIN_Q = int(os.getenv("PRACTICE_MIN_Q", "4"))
PRACTICE_MAX_Q = int(os.getenv("PRACTICE_MAX_Q", "8"))
MODULE_THRESHOLD = float(os.getenv("MODULE_THRESHOLD", "0.7"))
MODULE_MIN_Q = int(os.getenv("MODULE_MIN_Q", "6"))
MODULE_MAX_Q = int(os.getenv("MODULE_MAX_Q", "12"))
DIAGNOSTIC_MIN_MASTERY = float(os.getenv("DIAGNOSTIC_MIN_MASTERY", "0.25"))
DIAGNOSTIC_MAX_MASTERY = float(os.getenv("DIAGNOSTIC_MAX_MASTERY", "0.55"))
REMEDIATION_LESSON_BELOW = float(os.getenv("REMEDIATION_LESSON_BELOW", "0.5"))
MAX_GENERATIONS_PER_DAY = int(os.getenv("MAX_GENERATIONS_PER_DAY", "150"))
EVIDENCE_LOG_PATH = os.getenv("EVIDENCE_LOG_PATH", os.path.join(os.path.dirname(__file__), "evidence_log.jsonl"))
QCM_PROMPT_MAX_CHARS = int(os.getenv("QCM_PROMPT_MAX_CHARS", "240"))
QCM_CHOICE_MAX_CHARS = int(os.getenv("QCM_CHOICE_MAX_CHARS", "90"))

ProgressFn = Callable[[str, str], None]
_progress_cv: ContextVar[Optional[ProgressFn]] = ContextVar("its_progress", default=None)


def _progress(text: str, icon: str = "sparkle") -> None:
    fn = _progress_cv.get()
    if fn:
        try:
            fn(text, icon)
        except Exception:
            pass


# =====================================================
# COMMANDS (French first, English aliases kept)
# =====================================================
def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return text.strip()


COMMANDS: Dict[str, set] = {
    "start": {"commencer le diagnostic", "commencer", "diagnostic", "demarrer le diagnostic", "start diagnostic", "start", "lancer le diagnostic"},
    "practice": {"quiz", "lancer le quiz", "refaire le quiz", "m entrainer", "entrainement", "practice", "practice qcm", "qcm", "un quiz", "nouveau quiz"},
    "hint": {"indice", "un indice", "hint", "next hint", "indice suivant"},
    "next": {"notion suivante", "suivant", "suivante", "continuer", "next", "next kc", "passer a la suite"},
    "progress": {"ma progression", "progression", "radar", "ou en suis je", "show radar", "evaluation radar", "mon avancement"},
    "help": {"aide", "help", "guide", "commands", "commandes", "que puis je faire"},
    "lesson": {"revoir la lecon", "la lecon", "lecon", "relire la lecon", "lesson"},
    "checkpoint": {"controle", "refaire le controle", "controle de chapitre", "checkpoint", "retry", "module quiz"},
    "clear_image": {"oublier l image", "oublier la photo", "clear image", "forget image", "new image"},
    "reset": {"recommencer a zero", "recommencer", "tout effacer", "reset"},
    "reset_confirm": {"oui tout effacer", "reset confirm", "confirmer la remise a zero"},
    "cancel": {"non", "annuler", "cancel", "non garder"},
    "debug": {"debug its", "its debug", "show its state", "debug mastery", "show mastery"},
}
_COMMAND_INDEX: Dict[str, str] = {alias: cmd for cmd, aliases in COMMANDS.items() for alias in aliases}
SLOW_COMMANDS = {"start", "practice", "next", "checkpoint", "lesson"}


def classify_command(text: str) -> Optional[str]:
    return _COMMAND_INDEX.get(_fold(text))


def looks_like_answers(text: str) -> bool:
    return bool(re.search(r"\b\d+\s*[A-D]\b", text.upper())) and len(text) < 120


def parse_answers_from_text(text: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for m in re.finditer(r"\b(\d+)\s*([A-D])\b", text.upper().replace(",", " ")):
        out[int(m.group(1))] = m.group(2)
    return out


def extract_latest_user_text(input_items: Any) -> str:
    if isinstance(input_items, list):
        for msg in reversed(input_items):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            blocks = msg.get("content", [])
            if isinstance(blocks, list):
                for b in blocks:
                    if isinstance(b, dict) and b.get("type") == "input_text":
                        t = (b.get("text") or "").strip()
                        if t:
                            return t
            return ""
    return str(input_items or "").strip()


def extract_latest_user_image_urls(input_items: Any) -> List[str]:
    if not isinstance(input_items, list):
        return []
    for msg in reversed(input_items):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        blocks = msg.get("content", [])
        return [str(b.get("image_url")) for b in blocks if isinstance(b, dict) and b.get("type") == "input_image" and b.get("image_url")] if isinstance(blocks, list) else []
    return []


def fit_text(text: Any, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= max_chars:
        return clean
    cut = clean[: max_chars - 1].rstrip()
    for sep in (". ", "; ", ", ", " - ", " "):
        idx = cut.rfind(sep)
        if idx >= max_chars // 2:
            cut = cut[:idx].rstrip()
            break
    return f"{cut}…"


def qcm_widget_data(title: str, questions: List[dict]) -> Dict[str, Any]:
    # Backend terminal only: the learner widget below receives no answer key.
    lines = [f"[QCM answer key] {title}"]
    for q in questions:
        letter = str(q.get("answer") or "").upper()
        choices = q.get("choices") or []
        if letter in LETTERS and len(choices) == len(LETTERS):
            lines.append(f"  Q{q['number']}: {q['text']}\n"
                         f"    Correct answer: {letter}) {choices[LETTERS.index(letter)]}")
    print("\n".join(lines), flush=True)
    q_out: List[dict] = []
    for q in questions:
        c = q["choices"]
        q_out.append({
            "id": str(q["number"]),
            "prompt": fit_text(q["text"], QCM_PROMPT_MAX_CHARS),
            "choices": [{"label": f"{L}) {fit_text(c[i], QCM_CHOICE_MAX_CHARS)}", "value": L} for i, L in enumerate(LETTERS)],
        })
    return {"title": title, "questions": q_out}


# =====================================================
# KC GRAPH
# =====================================================
@dataclass
class KCNode:
    id: str
    title: str
    kind: str
    outline_path: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)


@dataclass
class KcGraph:
    title: str
    nodes: Dict[str, KCNode]
    source_pdf: str = ""
    next_by_id: Dict[str, str] = field(default_factory=dict)
    children_by_id: Dict[str, List[str]] = field(default_factory=dict)
    parent_by_id: Dict[str, str] = field(default_factory=dict)
    ordered_kc_ids: List[str] = field(default_factory=list)
    index_by_kc: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def load(path: str) -> "KcGraph":
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = json.loads(re.sub(r",(\s*[\]}])", r"\1", raw))
        nodes: Dict[str, KCNode] = {}
        for n in data.get("nodes", []):
            if not isinstance(n, dict):
                continue
            nid = str(n.get("id") or "").strip()
            if nid:
                nodes[nid] = KCNode(
                    id=nid,
                    title=str(n.get("title") or "").strip(),
                    kind=str(n.get("kind") or "").strip(),
                    outline_path=n.get("outline_path") or [],
                    pages=[int(p) for p in (n.get("pages") or []) if isinstance(p, int)],
                )
        g = KcGraph(
            title=str(data.get("title") or data.get("source_title") or "Cours").strip(),
            nodes=nodes,
            source_pdf=str(data.get("source_pdf") or "").strip(),
        )

        def add_contains(src: str, dst: str) -> None:
            if src and dst:
                g.children_by_id.setdefault(src, []).append(dst)
                g.parent_by_id.setdefault(dst, src)

        def add_sequence(src: str, dst: str) -> None:
            if src and dst:
                g.next_by_id[src] = dst

        for e in data.get("edges", []):
            if isinstance(e, dict):
                et = str(e.get("type") or "").lower()
                if et == "sequence":
                    add_sequence(str(e.get("src") or ""), str(e.get("dst") or ""))
                elif et == "contains":
                    add_contains(str(e.get("src") or ""), str(e.get("dst") or ""))
        for pair in data.get("contains", []):
            if isinstance(pair, list) and len(pair) >= 2:
                add_contains(str(pair[0] or "").strip(), str(pair[1] or "").strip())
        for pair in data.get("sequence", []):
            if isinstance(pair, list) and len(pair) >= 2:
                add_sequence(str(pair[0] or "").strip(), str(pair[1] or "").strip())
        g.ordered_kc_ids = g._compute_ordered_teachable_kcs()
        g.index_by_kc = {kc_id: i for i, kc_id in enumerate(g.ordered_kc_ids)}
        return g

    def _is_teachable_kc(self, nid: str) -> bool:
        n = self.nodes.get(nid)
        return bool(n and n.kind.lower() == "kc")

    def _ordered_children(self, parent_id: str) -> List[str]:
        kids = [k for k in self.children_by_id.get(parent_id, []) if k in self.nodes]
        if not kids:
            return []
        kidset = set(kids)
        nxt = {k: self.next_by_id[k] for k in kids if self.next_by_id.get(k) in kidset}
        pointed_to = set(nxt.values())
        ordered: List[str] = []
        visited: set[str] = set()

        def follow(start: str) -> None:
            cur: Optional[str] = start
            while cur and cur not in visited:
                visited.add(cur)
                ordered.append(cur)
                cur = nxt.get(cur)

        for s in [k for k in kids if k not in pointed_to]:
            follow(s)
        for k in kids:
            if k not in visited:
                follow(k)
        return ordered

    def _compute_ordered_teachable_kcs(self) -> List[str]:
        root = next((nid for nid, n in self.nodes.items() if n.kind.lower() == "course"), None)
        if not root:
            roots = [nid for nid in self.nodes if nid not in self.parent_by_id]
            root = roots[0] if roots else None
        if not root:
            return [nid for nid in self.nodes if self._is_teachable_kc(nid)]
        out: List[str] = []
        seen: set[str] = set()

        def dfs(node_id: str) -> None:
            if self._is_teachable_kc(node_id) and node_id not in seen:
                seen.add(node_id)
                out.append(node_id)
            for child in self._ordered_children(node_id):
                dfs(child)

        dfs(root)
        return out

    def kc_ids(self) -> List[str]:
        return list(self.ordered_kc_ids)

    def next_kc(self, kc_id: str) -> Optional[str]:
        i = self.index_by_kc.get(kc_id)
        if i is None or i + 1 >= len(self.ordered_kc_ids):
            return None
        return self.ordered_kc_ids[i + 1]

    def previous_kcs(self, kc_id: str, limit: int = 2) -> List[str]:
        i = self.index_by_kc.get(kc_id)
        if i is None or i <= 0:
            return []
        previous = self.ordered_kc_ids[:i]
        cur_module = self.module_of(kc_id)
        same = [pid for pid in previous if self.module_of(pid) == cur_module]
        picked = same[-limit:]
        for pid in reversed(previous):
            if len(picked) >= limit:
                break
            if pid not in picked:
                picked.insert(0, pid)
        return picked[-limit:]

    def module_of(self, nid: str) -> Optional[str]:
        cur = nid
        while True:
            p = self.parent_by_id.get(cur)
            if not p:
                return None
            pn = self.nodes.get(p)
            if pn and pn.kind.lower() == "module":
                return p
            cur = p

    def module_title(self, nid: str) -> str:
        mid = self.module_of(nid)
        if mid and mid in self.nodes:
            return self.nodes[mid].title
        return "Introduction"

    def module_kcs(self, module_id: str) -> List[str]:
        out: List[str] = []

        def dfs(n: str) -> None:
            if self._is_teachable_kc(n):
                out.append(n)
            for ch in self._ordered_children(n):
                dfs(ch)

        if module_id in self.nodes:
            dfs(module_id)
        out_set = set(out)
        return [kc for kc in self.ordered_kc_ids if kc in out_set]

    def modules_in_order(self) -> List[Tuple[str, str, List[str]]]:
        """(module_id or 'ROOT', title, [kc ids]) in teaching order."""
        groups: Dict[str, List[str]] = {}
        order: List[str] = []
        for kc_id in self.ordered_kc_ids:
            mid = self.module_of(kc_id) or "ROOT"
            if mid not in groups:
                groups[mid] = []
                order.append(mid)
            groups[mid].append(kc_id)
        return [(mid, self.nodes[mid].title if mid in self.nodes else "Introduction", groups[mid]) for mid in order]

    def kc_pages(self, kc: KCNode) -> List[int]:
        pages = sorted({p for p in kc.pages if p > 0})
        if pages:
            return pages
        mid = self.module_of(kc.id)
        if mid and mid in self.nodes:
            return sorted({p for p in self.nodes[mid].pages if p > 0})
        return []


# =====================================================
# SESSION (per learner, persisted)
# =====================================================
@dataclass
class Session:
    doctrine_revision: str = ""
    diagnostic_taught_kc_ids: List[str] = field(default_factory=list)
    diagnostic_groups: List[dict] = field(default_factory=list)
    diagnostic_group_index: int = 0
    diagnostic_scenario: str = ""
    diagnostic_policy_version: int = 1
    diagnostic_learning_queue: List[str] = field(default_factory=list)
    diagnostic_mode: str = "flat"
    practice_evidence_by_kc: Dict[str, List[dict]] = field(default_factory=dict)
    diagnostic_stage: str = ""
    diagnostic_validation_challenge: bool = False
    diagnostic_round: int = 0
    diagnostic_question_limit: int = 12
    diagnostic_kc_ids: List[str] = field(default_factory=list)
    diagnostic_evidence: Dict[str, List[dict]] = field(default_factory=dict)
    diagnostic_status_by_kc: Dict[str, str] = field(default_factory=dict)
    podcast_cache: Dict[str, Any] = field(default_factory=dict)
    flashcard_deck: Dict[str, Any] = field(default_factory=dict)
    user_id: str = ""
    phase: str = "idle"                 # idle | waiting_answers
    scope: str = "diagnostic"           # diagnostic | practice | module_quiz
    current_kc_id: Optional[str] = None
    quiz_id: str = ""
    quiz_questions: Dict[str, dict] = field(default_factory=dict)   # number(str) -> question dict
    mastery: Dict[str, float] = field(default_factory=dict)
    last_score_by_kc: Dict[str, float] = field(default_factory=dict)
    diagnostic_profile: Dict[str, float] = field(default_factory=dict)
    diagnostic_raw_score_by_kc: Dict[str, float] = field(default_factory=dict)
    diagnostic_done: bool = False
    weak_queue: List[str] = field(default_factory=list)
    validated_kc_ids: List[str] = field(default_factory=list)
    kc_essential_targets: Dict[str, List[str]] = field(default_factory=dict)
    current_micro_lesson: str = ""
    last_mistakes_summary: str = ""
    misconceptions: Dict[str, List[str]] = field(default_factory=dict)
    attempts_by_kc: Dict[str, int] = field(default_factory=dict)
    pending_hint_ladder: List[str] = field(default_factory=list)
    hint_index: int = 0
    last_tutor_action: str = ""
    can_advance: bool = False
    validated_kc_id: Optional[str] = None
    pending_next_kc_id: Optional[str] = None
    pending_module_id: Optional[str] = None
    module_gate_locked: bool = False
    module_mastery: Dict[str, float] = field(default_factory=dict)
    pending_module_retry: bool = False
    last_visual_image_urls: List[str] = field(default_factory=list)
    visual_question_history: List[dict] = field(default_factory=list)
    seen_question_ids: List[str] = field(default_factory=list)
    recent_stems: Dict[str, List[str]] = field(default_factory=dict)   # kc_id -> stems already asked
    llm_budget_date: str = ""
    llm_budget_used: int = 0
    pending_reset: bool = False
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


class BudgetExceeded(Exception):
    pass


# =====================================================
# ORCHESTRATOR
# =====================================================
class Orchestrator(HierarchicalDiagnostic, Flashcards, Podcasts):
    def __init__(self) -> None:
        self.graph = KcGraph.load(KC_GRAPH_PATH)
        self.doc = doctrine()
        self.store = store()
        self.bank = QuestionBank(self.store)
        self._sessions: Dict[str, Session] = {}
        print(f"[orchestrator] {len(self.graph.kc_ids())} notions, doctrine source={self.doc.source}, bank={sum(self.bank.stats().values())} questions")

    # ---------------- session persistence ----------------
    @staticmethod
    def _user_id(ctx: Any) -> str:
        rc = getattr(ctx, "request_context", None) or {}
        return str(rc.get(USER_ID_KEY) or getattr(getattr(ctx, "thread", None), "id", None) or "anonymous")

    async def _load_session(self, user_id: str) -> Session:
        if user_id in self._sessions:
            return self._sessions[user_id]
        data = await self.store.aget_json(f"sessions/{user_id}.json", None)
        sess = Session.from_dict(data) if isinstance(data, dict) else Session(user_id=user_id, created_at=datetime.now(timezone.utc).isoformat())
        sess.user_id = user_id
        self._refresh_doctrine_material(sess)
        self._sessions[user_id] = sess
        return sess

    def load_session_sync(self, user_id: str) -> Session:
        if user_id in self._sessions:
            return self._sessions[user_id]
        data = self.store.get_json(f"sessions/{user_id}.json", None)
        sess = Session.from_dict(data) if isinstance(data, dict) else Session(user_id=user_id)
        sess.user_id = user_id
        self._refresh_doctrine_material(sess)
        self._sessions[user_id] = sess
        return sess

    def _refresh_doctrine_material(self, sess: Session) -> None:
        revision = getattr(self.doc, "revision", "legacy")
        if sess.doctrine_revision != revision:
            sess.current_micro_lesson = ""
            sess.kc_essential_targets = {}
            sess.flashcard_deck = {}
            sess.podcast_cache = {}
            sess.pending_hint_ladder = []
            sess.hint_index = 0
            sess.doctrine_revision = revision

    async def _save_session(self, sess: Session) -> None:
        sess.updated_at = datetime.now(timezone.utc).isoformat()
        await self.store.aput_json(f"sessions/{sess.user_id}.json", asdict(sess))

    def _record_event(self, sess: Session, ctx: Any, event: Dict[str, Any]) -> None:
        from app.auth import current_username
        enriched = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thread_id": str(getattr(getattr(ctx, "thread", None), "id", "") or ""),
            "user_id": sess.user_id,
            "provider": current_provider.get().provider,
            **event,
            "username": current_username.get(),
        }
        try:
            os.makedirs(os.path.dirname(EVIDENCE_LOG_PATH), exist_ok=True)
            with open(EVIDENCE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(enriched, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            print(f"Evidence logging failed: {exc}")
        try:
            asyncio.get_running_loop().create_task(self.store.aappend_event(enriched))
        except RuntimeError:
            self.store.append_event(enriched)

    def _charge_budget(self, sess: Session, n: int = 1) -> None:
        today = date.today().isoformat()
        if sess.llm_budget_date != today:
            sess.llm_budget_date = today
            sess.llm_budget_used = 0
        if sess.llm_budget_used + n > MAX_GENERATIONS_PER_DAY:
            raise BudgetExceeded()
        sess.llm_budget_used += n

    # ---------------- helpers ----------------
    def _kc(self, kc_id: Optional[str]) -> Optional[KCNode]:
        return self.graph.nodes.get(kc_id or "")

    def _kc_context(self, kc: KCNode, *, neighbours: bool = True, max_chars: int = 7000) -> str:
        pages = self.graph.kc_pages(kc)
        if neighbours:
            extra: List[int] = []
            for p in pages:
                extra.extend([p - 1, p + 1])
            pages = pages + [p for p in extra if p > 0]
        return self.doc.pages_text(pages, max_chars=max_chars)

    def _source_card(self, kc: KCNode, buttons: List[Tuple[str, str]]) -> Dict[str, Any]:
        if not any(command == "flashcards" for _, command in buttons):
            buttons = [*buttons, ("Cartes de revision", "flashcards"), ("ecouter la lecon", "podcast")]
        pages = self.graph.kc_pages(kc)
        first = pages[0] if pages else None
        return {
            "type": "widget",
            "title": f"Source : {kc.title}",
            "widget": W.source_card(
                kc_title=kc.title,
                pages=pages,
                image_url=self.doc.page_image_url(first) if first else None,
                pdf_url=self.doc.pdf_url(first) if first else None,
                buttons=buttons,
            ),
        }

    @staticmethod
    def _text(text: str) -> Dict[str, Any]:
        return {"type": "text", "text": text}

    @staticmethod
    def _actions(buttons: List[Tuple[str, str]], **kw: Any) -> Dict[str, Any]:
        return {"type": "widget", "title": "Actions", "widget": W.actions_card(buttons, **kw)}

    def _next_actions(self, sess: Session) -> List[Tuple[str, str]]:
        """Contextual buttons: what makes sense from the current state."""
        if sess.phase == "waiting_answers":
            return [("Poser une question", "question")]
        if sess.module_gate_locked and sess.pending_module_retry:
            return [("Refaire le contr?le du chapitre", "controle"), ("Ma progression", "ma progression")]
        if sess.diagnostic_mode == 'hierarchical' and sess.diagnostic_done and not sess.current_kc_id:
            return [("Ma progression", "ma progression"), ("Poser une question", "question")]
        if not sess.diagnostic_done and not sess.current_kc_id:
            return [("Commencer le diagnostic", "commencer le diagnostic"), ("Ma progression", "ma progression")]
        buttons: List[Tuple[str, str]] = []
        if sess.module_gate_locked and sess.pending_module_retry:
            buttons.append(("Refaire le contrôle du chapitre", "controle"))
        elif sess.can_advance and sess.validated_kc_id == sess.current_kc_id:
            buttons.append(("Notion suivante", "notion suivante"))
        else:
            buttons.append(("Lancer le quiz", "quiz"))
            if sess.pending_hint_ladder and sess.hint_index < len(sess.pending_hint_ladder):
                buttons.append(("Un indice", "indice"))
        buttons.append(("Revoir la lecon", "revoir la lecon"))
        buttons.append(("Ma progression", "ma progression"))
        return buttons

    def _kc_label(self, kc: KCNode) -> str:
        i = self.graph.index_by_kc.get(kc.id, 0) + 1
        return f"Notion {i}/{len(self.graph.kc_ids())} · {kc.title}"

    # =====================================================
    # ENTRY POINTS
    # =====================================================
    async def handle(self, user_input: Any, ctx: AgentContext, progress: Optional[ProgressFn] = None) -> List[Dict[str, Any]]:
        token = _progress_cv.set(progress)
        try:
            set_provider_from_context(getattr(ctx, "request_context", None))
            sess = await self._load_session(self._user_id(ctx))
            text = extract_latest_user_text(user_input).strip()
            image_urls = extract_latest_user_image_urls(user_input)
            try:
                if image_urls:
                    blocks = await self._answer_visual_question(sess, text, image_urls, ctx)
                elif looks_like_answers(text) and sess.phase == "waiting_answers":
                    blocks = await self._process_answers(sess, parse_answers_from_text(text), ctx)
                else:
                    blocks = await self._dispatch(sess, classify_command(text), text, ctx)
            except BudgetExceeded:
                blocks = [self._text("Vous avez atteint la limite quotidienne de générations pour ce compte de démonstration. Reprenez demain, ou ajoutez votre propre clé dans les réglages.")]
            except Exception as exc:  # noqa: BLE001
                print(f"[orchestrator] error: {type(exc).__name__}: {exc}")
                blocks = [self._text(friendly_llm_error(exc)), self._actions(self._retry_actions(sess), caption="Vous pouvez réessayer.")]
            await self._save_session(sess)
            return blocks
        finally:
            _progress_cv.reset(token)

    async def handle_command(self, command: str, ctx: AgentContext, progress: Optional[ProgressFn] = None) -> List[Dict[str, Any]]:
        return await self.handle([{"role": "user", "content": [{"type": "input_text", "text": command}]}], ctx, progress)

    async def handle_qcm_submit(self, submitted: Dict[int, str], ctx: AgentContext, progress: Optional[ProgressFn] = None) -> List[Dict[str, Any]]:
        token = _progress_cv.set(progress)
        try:
            set_provider_from_context(getattr(ctx, "request_context", None))
            sess = await self._load_session(self._user_id(ctx))
            try:
                blocks = await self._process_answers(sess, submitted, ctx)
            except BudgetExceeded:
                blocks = [self._text("Limite quotidienne de générations atteinte pour ce compte de démonstration.")]
            except Exception as exc:  # noqa: BLE001
                print(f"[orchestrator] submit error: {type(exc).__name__}: {exc}")
                blocks = [self._text(friendly_llm_error(exc)), self._actions(self._retry_actions(sess))]
            await self._save_session(sess)
            return blocks
        finally:
            _progress_cv.reset(token)

    async def handle_report(self, payload: Dict[str, Any], ctx: AgentContext) -> List[Dict[str, Any]]:
        sess = await self._load_session(self._user_id(ctx))
        self._record_event(sess, ctx, {
            "event": "question_reported",
            "quiz_id": payload.get("quiz_id"),
            "question_id": payload.get("question_id"),
            "number": payload.get("number"),
            "kc_id": payload.get("kc_id"),
        })
        return [self._text("Merci, la question est signalée aux formateurs. Vous pouvez continuer.")]

    def record_feedback(self, user_id: str, thread_id: str, item_ids: List[str], kind: str) -> None:
        from app.auth import current_username
        sess = self.load_session_sync(user_id)
        enriched = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thread_id": thread_id,
            "user_id": user_id,
            "event": "item_feedback",
            "username": current_username.get(),
            "kind": kind,
            "item_ids": item_ids,
            "current_kc_id": sess.current_kc_id,
        }
        self.store.append_event(enriched)

    def peek_transition_message(self, ctx: AgentContext, raw_text: str) -> Optional[str]:
        cmd = classify_command(raw_text or "")
        if cmd == "start":
            return ("Je prépare un diagnostic par chapitre, avec des questions de suivi adaptées."
                    if DIAGNOSTIC_MODE == 'hierarchical' else
                    "Je prépare un diagnostic rapide sur l'ensemble du mémento. Une trentaine de secondes.")
        if cmd == "practice":
            return "Je prépare un quiz adapté à cette notion. Un instant."
        if cmd == "next":
            return "Je prépare la leçon de la notion suivante. Un instant."
        if cmd == "checkpoint":
            return "Je prépare le contrôle du chapitre. Un instant."
        return None

    def progress_summary(self, user_id: str) -> Dict[str, Any]:
        sess = self.load_session_sync(user_id)
        kc = self._kc(sess.current_kc_id)
        total = len(self.graph.kc_ids())
        return {
            "course_title": self.graph.title,
            "current_kc": {"id": kc.id, "title": kc.title, "index": self.graph.index_by_kc.get(kc.id, 0) + 1, "total": total} if kc else None,
            "module_title": self.graph.module_title(kc.id) if kc else None,
            "validated": len(set(sess.validated_kc_ids)),
            "total": total,
            "diagnostic_done": sess.diagnostic_done,
            "quiz_pending": sess.phase == "waiting_answers",
            "diagnostic_status_by_kc": dict(sess.diagnostic_status_by_kc),
            "diagnostically_supported": sum(s == 'supported' for s in sess.diagnostic_status_by_kc.values()),
        }

    # =====================================================
    # DISPATCH
    # =====================================================
    async def _dispatch(self, sess: Session, cmd: Optional[str], text: str, ctx: Any) -> List[Dict[str, Any]]:
        if text.strip().lower() in {'podcast', 'lecon audio', 'lecon audio'}:
            return await self._podcast(sess, ctx)
        if text.strip().lower() in {'flashcards', 'fiches', 'cartes de révision'} or text.strip().lower().startswith('fiche '):
            return await self._flashcards(sess, text.strip().lower(), ctx)
        if (sess.diagnostic_mode == 'hierarchical' and sess.diagnostic_stage not in {'', 'learning'} and not sess.diagnostic_done
                and cmd in {'practice', 'next', 'lesson', 'checkpoint'}):
            return [self._text('Terminez le diagnostic en cours avant de commencer la leçon ou le quiz.'),
                    self._actions([('Reprendre le diagnostic', 'commencer le diagnostic')])]
        if sess.pending_reset and cmd not in {"reset_confirm", "cancel"}:
            sess.pending_reset = False
        if cmd == "start":
            return await self._start_diagnostic(sess, ctx)
        if cmd == "practice":
            if not sess.current_kc_id:
                return [self._text("Commencez par le diagnostic : il choisit la première notion à travailler."), self._actions([("Commencer le diagnostic", "commencer le diagnostic")])]
            return await self._start_practice(sess, ctx)
        if cmd == "next":
            if not sess.current_kc_id:
                return [self._text("Aucune notion en cours. Commencez par le diagnostic."), self._actions([("Commencer le diagnostic", "commencer le diagnostic")])]
            return await self._next_kc(sess, ctx)
        if cmd == "hint":
            return [self._text(self._next_hint_text(sess)), self._actions(self._next_actions(sess))]
        if cmd == "progress":
            return self._show_progress(sess)
        if cmd == "help":
            return self._help(sess)
        if cmd == "lesson":
            return await self._show_lesson(sess, ctx)
        if cmd == "checkpoint":
            if not sess.pending_module_id or not sess.pending_module_retry:
                return [self._text("Aucun contrôle de chapitre à refaire pour le moment."), self._actions(self._next_actions(sess))]
            return await self._start_module_quiz(sess, sess.pending_module_id, ctx)
        if cmd == "clear_image":
            sess.last_visual_image_urls = []
            sess.visual_question_history = []
            return [self._text("Photo oubliée. Envoyez-en une autre pour une nouvelle analyse.")]
        if cmd == "reset":
            sess.pending_reset = True
            return [{"type": "widget", "title": "Confirmation", "widget": W.confirm_card(
                "Effacer toute votre progression (diagnostic, notions validées, quiz en cours) ?",
                ("Oui, tout effacer", "oui, tout effacer"), ("Non, garder", "non"))}]
        if cmd == "reset_confirm":
            if not sess.pending_reset:
                return [self._text("Aucune remise à zéro en attente."), self._actions(self._next_actions(sess))]
            fresh = Session(user_id=sess.user_id, created_at=datetime.now(timezone.utc).isoformat())
            self._sessions[sess.user_id] = fresh
            await self._save_session(fresh)
            self._record_event(fresh, ctx, {"event": "session_reset"})
            sess.__dict__.update(fresh.__dict__)
            return [self._text("Progression effacée. On repart de zéro."), self._actions([("Commencer le diagnostic", "commencer le diagnostic")])]
        if cmd == "cancel":
            sess.pending_reset = False
            return [self._text("D'accord, rien n'est effacé."), self._actions(self._next_actions(sess))]
        if cmd == "debug":
            return [self._text(self._debug_text(sess))]
        if _fold(text) == "question":
            return [self._text("Posez votre question directement dans le champ de message : je réponds avec le mémento.")]
        if not text:
            return self._help(sess)
        if sess.last_visual_image_urls and sess.last_tutor_action == "answer_visual_pdf_question":
            return await self._answer_visual_question(sess, text, [], ctx)
        return await self._answer_free_question(sess, text, ctx)

    def _retry_actions(self, sess: Session) -> List[Tuple[str, str]]:
        if sess.scope == "diagnostic" and not sess.diagnostic_done:
            return [("Réessayer le diagnostic", "commencer le diagnostic")]
        return self._next_actions(sess)

    # =====================================================
    # HELP / PROGRESS / LESSON DISPLAY
    # =====================================================
    def _help(self, sess: Session) -> List[Dict[str, Any]]:
        text = (
            "Comment ça marche\n\n"
            + ("1. Un diagnostic par chapitre repère les notions à travailler. Les réussites étayées permettent de passer les leçons introductives correspondantes.\n"
               if DIAGNOSTIC_MODE == 'hierarchical' else
               "1. Un diagnostic rapide sur le mémento repère les notions à travailler. Il oriente sans valider la maîtrise.\n")
            +
            "2. Pour chaque notion : une leçon courte avec la page du mémento, puis un quiz. À 70 % de bonnes réponses, la notion est validée.\n"
            "3. En cas d'erreur : un corrigé question par question, des indices, et une reprise du quiz.\n"
            "4. À la fin de chaque chapitre, un contrôle regroupe ses notions.\n\n"
            "À tout moment : posez une question libre sur le mémento, envoyez la photo d'un symbole, ou cliquez sur « Ma progression ». "
            "Les boutons sous chaque message proposent la suite ; vous pouvez aussi taper « quiz », « indice », « notion suivante », « ma progression », « aide » ou « recommencer à zéro »."
        )
        return [self._text(text), self._actions(self._next_actions(sess))]

    def _show_progress(self, sess: Session) -> List[Dict[str, Any]]:
        validated = set(sess.validated_kc_ids)
        modules_out: List[Dict[str, Any]] = []
        views: Dict[str, Dict[str, Any]] = {}
        sec_labels: List[str] = []
        sec_vals: List[float] = []
        cur_module = self.graph.module_of(sess.current_kc_id) if sess.current_kc_id else None
        for mid, title, kc_ids in self.graph.modules_in_order():
            v = sum(1 for k in kc_ids if k in validated)
            e = sum(1 for k in kc_ids if k not in validated and sess.diagnostic_raw_score_by_kc.get(k, 0.0) >= 1.0)
            modules_out.append({"title": title, "total": len(kc_ids), "validated": v, "estimated": e, "current": (mid == (cur_module or "ROOT")) if cur_module or mid == "ROOT" else False})
            vals = [self._display_mastery(sess, k) for k in kc_ids]
            sec_labels.append(title)
            sec_vals.append(sum(vals) / max(1, len(vals)))
            views[f"sec::{mid}"] = {"label": f"Notions : {title}", "labels": [self.graph.nodes[k].title for k in kc_ids], "values": vals}
        views = {"modules": {"label": "Chapitres", "labels": sec_labels, "values": sec_vals}, **views}
        radar_html = build_radar_dashboard_html("Progression par chapitre et par notion", views, default_view="modules")
        card = W.progress_card(
            title="Ma progression",
            modules=modules_out,
            validated=len(validated),
            total=len(self.graph.kc_ids()),
            radar_html=radar_html,
            buttons=[b for b in self._next_actions(sess) if b[1] != "ma progression"],
        )
        cur_kc = self._kc(sess.current_kc_id)
        intro = "Diagnostic non fait : commencez par lui pour situer votre niveau." if not sess.diagnostic_done else (
            f"Notion en cours : {cur_kc.title}." if cur_kc else "Parcours terminé.")
        if sess.diagnostic_mode == 'hierarchical' and sess.diagnostic_done:
            supported = [self.graph.nodes[k].title for k, s in sess.diagnostic_status_by_kc.items()
                         if s == 'supported' and k in self.graph.nodes]
            intro = (f"Notion à travailler : {cur_kc.title}." if cur_kc else
                     "Aucune leçon introductive restante selon le diagnostic. Cela ne constitue pas une validation de toutes les notions.")
            intro += '\nLeçons introductives dispensées : ' + (', '.join(supported) or 'aucune') + '.'
            intro += '\nLe radar représente les scores de pratique ; les observations diagnostiques sont distinctes.'
        return [self._text(intro), {"type": "widget", "title": "Ma progression", "widget": card}]

    def _display_mastery(self, sess: Session, kc_id: str) -> float:
        if kc_id in sess.mastery:
            return float(sess.mastery[kc_id])
        return float(sess.diagnostic_profile.get(kc_id, 0.0))

    async def _show_lesson(self, sess: Session, ctx: Any) -> List[Dict[str, Any]]:
        kc = self._kc(sess.current_kc_id)
        if not kc:
            return [self._text("Aucune notion en cours."), self._actions([("Commencer le diagnostic", "commencer le diagnostic")])]
        if not sess.current_micro_lesson.strip():
            sess.current_micro_lesson = await self._build_lesson(sess, kc, ctx)
        return [self._text(f"{self._kc_label(kc)}\n\n{sess.current_micro_lesson}"), self._source_card(kc, self._next_actions(sess))]

    def _debug_text(self, sess: Session) -> str:
        lines = [f"scope={sess.scope} phase={sess.phase} current={sess.current_kc_id} action={sess.last_tutor_action}",
                 f"validated={sess.validated_kc_ids} weak_queue={sess.weak_queue}",
                 f"budget={sess.llm_budget_used}/{MAX_GENERATIONS_PER_DAY} provider={current_provider.get().provider} doctrine={self.doc.source}"]
        for kid, val in sorted(sess.mastery.items()):
            lines.append(f"{kid} | {self.graph.nodes[kid].title if kid in self.graph.nodes else kid} | mastery={val:.2f}")
        return "\n".join(lines)

    # =====================================================
    # DIAGNOSTIC
    # =====================================================
    def _sample_diagnostic_kcs(self, n: int) -> List[KCNode]:
        """Round-robin across chapters in teaching order, so the sample spans
        the whole memento instead of its first pages."""
        groups = [list(kcs) for _, _, kcs in self.graph.modules_in_order()]
        picked: List[str] = []
        while len(picked) < n and any(groups):
            for g in groups:
                if g and len(picked) < n:
                    picked.append(g.pop(0))
        picked.sort(key=lambda k: self.graph.index_by_kc.get(k, 0))
        return [self.graph.nodes[k] for k in picked if k in self.graph.nodes]

    async def _generate_diagnostic_question(self, kc: KCNode, ctx: Any) -> List[dict]:
        bank_q = self.bank.draw(kc.id, 1)
        if bank_q:
            q = dict(bank_q[0])
            q["kc_id"] = kc.id
            return [q]
        pages = self.graph.kc_pages(kc)
        prompt = (
            f"Rédige UNE question à choix multiples de niveau diagnostic sur la notion « {kc.title} » (identifiant {kc.id}), "
            f"à partir des pages suivantes du mémento (pages {pages}). La question doit tester un point de doctrine précis et vérifiable "
            "dans ces pages : signification d'une forme, d'une couleur, d'un symbole, une règle, un ordre.\n\n"
            f"{self.doc.pages_text(pages, max_chars=6000)}\n\n"
            "Contraintes : question de moins de 240 caractères, en français ; 4 propositions courtes (moins de 90 caractères), distinctes, une seule juste ; "
            "answer = lettre A/B/C/D ; explanation = une phrase qui justifie avec la page ; kc_id = l'identifiant fourni ; page = numéro de page source."
        )
        out = await run_structured("Diagnostic-QCM", INSTR_QCM, prompt, QcmList, ctx)
        return normalize_questions(out.questions, kc.id)

    async def _start_diagnostic(self, sess: Session, ctx: Any) -> List[Dict[str, Any]]:
        if DIAGNOSTIC_MODE == "hierarchical":
            return await self._start_hierarchical(sess, ctx)
        if DIAGNOSTIC_MODE != "flat":
            raise ValueError('DIAGNOSTIC_MODE must be flat or hierarchical')
        sess.diagnostic_mode = "flat"
        kcs = self._sample_diagnostic_kcs(DIAGNOSTIC_Q_NUM)
        if not kcs:
            return [self._text("Le plan du cours est vide : impossible de construire le diagnostic.")]
        self._charge_budget(sess, len(kcs))
        done = 0
        results: List[List[dict]] = [[] for _ in kcs]

        async def one(i: int, kc: KCNode) -> None:
            nonlocal done
            try:
                results[i] = await self._generate_diagnostic_question(kc, ctx)
            except Exception as exc:  # noqa: BLE001
                print(f"[diagnostic] {kc.id}: {type(exc).__name__}: {str(exc)[:160]}")
                if "credit" in str(exc).lower() or "401" in str(exc):
                    raise
                results[i] = []
            done += 1
            _progress(f"Question {done}/{len(kcs)} prête", "write")

        _progress(f"Diagnostic : {len(kcs)} notions tirées dans tous les chapitres", "compass")
        await asyncio.gather(*(one(i, kc) for i, kc in enumerate(kcs)))
        questions: List[dict] = []
        for kc, qs in zip(kcs, results):
            for q in qs[:1]:
                q["kc_id"] = kc.id
                questions.append(q)
        if len(questions) < max(4, len(kcs) // 2):
            raise RuntimeError(f"seulement {len(questions)} questions générées sur {len(kcs)}")
        for i, q in enumerate(questions, start=1):
            q["number"] = i
        sess.scope = "diagnostic"
        sess.phase = "waiting_answers"
        sess.quiz_id = uuid.uuid4().hex[:12]
        sess.quiz_questions = {str(q["number"]): q for q in questions}
        sess.last_tutor_action = "start_diagnostic"
        sess.pending_hint_ladder = []
        sess.hint_index = 0
        self._record_event(sess, ctx, {"event": "diagnostic_started", "tutor_action": sess.last_tutor_action, "n_questions": len(questions), "kc_ids": [q["kc_id"] for q in questions], "quiz_id": sess.quiz_id})
        intro = (
            f"Diagnostic : {len(questions)} questions, une par notion, réparties sur tous les chapitres du mémento. "
            "Répondez au mieux, sans tricher : ce n'est pas une note, c'est ce qui me permet de choisir par quoi commencer."
        )
        return [self._text(intro), {"type": "qcm", "data": qcm_widget_data(f"Diagnostic · {len(questions)} questions", questions)}]

    # =====================================================
    # LESSON
    # =====================================================
    def _lesson_mode(self, sess: Session, kc: KCNode, level: float, attempts: int, labels: List[str]) -> Tuple[str, str]:
        if sess.diagnostic_stage == 'learning' and sess.last_mistakes_summary:
            return 'remediation', 'Cible les erreurs observées dans le diagnostic, sans supposer une misconception confirmée.'
        if attempts <= 0:
            return "first_exposure", "Donne la base minimale nécessaire avant un premier quiz."
        if level < 0.4 or labels:
            return "remediation", "Cible précisément les erreurs commises. Ne réexplique pas toute la notion."
        if level < THRESHOLD:
            return "focused_review", "Revois seulement le point faible nécessaire au prochain quiz."
        return "brief_validation", "Confirme brièvement et donne un conseil de transfert."

    @staticmethod
    def _clean_field(text: Any) -> str:
        text = re.sub(r"\*\*|__|`", "", str(text or ""))
        text = re.sub(r"\s*\((?:g_[a-z]\d+|KC[_ ]?\w+)\)", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _render_lesson(cls, kc: KCNode, body: Any, pages: List[int]) -> str:
        for name in ("title", "learning_objective", "rule_to_remember", "operational_example", "common_mistake", "targeted_remediation", "self_check"):
            setattr(body, name, cls._clean_field(getattr(body, name, "")))
        body.essential_information = [cls._clean_field(x) for x in body.essential_information if cls._clean_field(x)]
        lines: List[str] = [f"**{body.title or kc.title}**"]
        if body.learning_objective:
            lines.append(f"Objectif : {body.learning_objective}")
        if body.essential_information:
            lines.append("")
            lines.append("À retenir :")
            lines.extend(f"- {item}" for item in body.essential_information[:6])
        if body.rule_to_remember:
            lines.append("")
            lines.append(f"Règle à mémoriser : {body.rule_to_remember}")
        if body.operational_example:
            lines.append(f"Exemple opérationnel : {body.operational_example}")
        if body.common_mistake:
            lines.append(f"Erreur fréquente : {body.common_mistake}")
        if body.targeted_remediation:
            lines.append(f"Pour corriger : {body.targeted_remediation}")
        if body.self_check:
            lines.append(f"Vérifiez-vous : {body.self_check}")
        if pages:
            lines.append("")
            lines.append("Source : mémento GOC, " + ", ".join(f"p. {p}" for p in pages) + ".")
        return "\n".join(lines)

    async def _build_lesson(self, sess: Session, kc: KCNode, ctx: Any, mistakes_summary: str = "") -> str:
        level = float(sess.mastery.get(kc.id, sess.diagnostic_profile.get(kc.id, 0.0)))
        attempts = int(sess.attempts_by_kc.get(kc.id, 0))
        labels = list(sess.misconceptions.get(kc.id, []))
        mode, focus = self._lesson_mode(sess, kc, level, attempts, labels)
        pages = self.graph.kc_pages(kc)
        summary = mistakes_summary or sess.last_mistakes_summary
        cache_key = f"cache/lesson/{getattr(self.doc, 'revision', 'legacy')}/{current_provider.get().provider}/{kc.id}/{mode}.json" if mode in {"first_exposure", "brief_validation"} else None
        cached = await self.store.aget_json(cache_key, None) if cache_key else None
        if isinstance(cached, dict) and cached.get("text"):
            text = str(cached["text"])
            plan = cached.get("plan") or {}
        else:
            self._charge_budget(sess)
            _progress(f"Leçon sur « {kc.title} »", "book-open")
            prompt = (
                f"Notion à enseigner : « {kc.title} » (identifiant {kc.id}), pages {pages} du mémento.\n"
                f"Niveau estimé de l'apprenant sur cette notion : {level:.2f} (0 = rien, 1 = maîtrise ; une estimation issue du diagnostic n'est pas une validation).\n"
                f"Tentatives de quiz sur cette notion : {attempts}. Mode de leçon : {mode}. Consigne : {focus}\n"
                f"Confusions détectées : {labels or 'aucune'}\n"
                f"Erreurs récentes : {summary or 'aucune'}\n\n"
                f"Extrait du mémento (source unique autorisée) :\n{self._kc_context(kc)}\n\n"
                "Rédige la leçon en français, courte mais complète sur la doctrine de cette notion : essential_information = 3 à 6 points précis "
                "(formes, couleurs, symboles, règles avec leur signification exacte), rule_to_remember = la règle en une phrase, "
                "operational_example = une situation concrète de sapeur-pompier, common_mistake = la confusion typique, self_check = une question "
                "d'auto-vérification. Conserve les termes officiels du mémento tels quels. N'invente aucune règle absente de l'extrait."
            )
            out = await run_structured("Micro-lesson", INSTR_LESSON, prompt, LessonOut, ctx)
            text = self._render_lesson(kc, out.lesson, pages)
            plan = out.lesson_plan.model_dump()
            if cache_key:
                await self.store.aput_json(cache_key, {"text": text, "plan": plan, "created_at": datetime.now(timezone.utc).isoformat()})
        self._record_event(sess, ctx, {
            "event": "micro_lesson_generated", "kc_id": kc.id, "kc_title": kc.title, "mastery": sess.mastery.get(kc.id),
            "diagnostic_level": sess.diagnostic_profile.get(kc.id), "learner_level_for_adaptation": level, "attempts": attempts,
            "misconceptions": labels, "source_pages": pages, "word_estimate": len(text.split()), "lesson_plan": plan,
            "lesson_budget_words": plan.get("max_words"), "lesson_mode": mode, "lesson_preview": text[:1200], "cached": bool(cached),
        })
        return text

    # =====================================================
    # PRACTICE
    # =====================================================
    async def _essential_targets(self, sess: Session, kc: KCNode, ctx: Any) -> List[str]:
        cached = sess.kc_essential_targets.get(kc.id)
        if cached:
            return cached
        pages = self.graph.kc_pages(kc)
        rules = self.doc.rules_for_pages(pages)
        if len(rules) >= 3:
            targets = rules[:10]
        else:
            shared_key = f"cache/targets/{getattr(self.doc, 'revision', 'legacy')}/{kc.id}.json"
            shared = await self.store.aget_json(shared_key, None)
            if isinstance(shared, list) and shared:
                targets = [str(t) for t in shared]
            else:
                self._charge_budget(sess)
                prompt = (
                    f"Notion : « {kc.title} », pages {pages}.\n\nExtrait du mémento :\n{self._kc_context(kc, neighbours=False)}\n\n"
                    "Liste 3 à 8 cibles d'évaluation précises et testables par QCM pour cette notion (une règle ou un fait par cible, "
                    "en français, sans doublon, sans formulation vague du type « comprendre la leçon »)."
                )
                out = await run_structured("KC-targets", INSTR_TARGETS, prompt, EssentialTargets, ctx)
                targets = [re.sub(r"\s+", " ", t).strip() for t in out.essential_targets if t.strip()][:10]
                if targets:
                    await self.store.aput_json(shared_key, targets)
        sess.kc_essential_targets[kc.id] = targets
        return targets

    async def _start_practice(self, sess: Session, ctx: Any) -> List[Dict[str, Any]]:
        kc = self._kc(sess.current_kc_id)
        if not kc:
            return [self._text("Notion introuvable dans le plan du cours.")]
        sess.can_advance = False
        sess.validated_kc_id = None
        challenge = sess.diagnostic_validation_challenge
        if not sess.current_micro_lesson.strip() and not challenge:
            sess.current_micro_lesson = await self._build_lesson(sess, kc, ctx)
            if sess.diagnostic_stage != 'learning':
                sess.last_mistakes_summary = ""
        mastery = float(sess.mastery.get(kc.id, 0.0))
        attempts_before = int(sess.attempts_by_kc.get(kc.id, 0))
        difficulty = "easy" if attempts_before <= 0 or mastery < 0.4 else ("medium" if mastery < THRESHOLD else "hard")
        if challenge:
            difficulty = "medium"
        review_kcs = [self.graph.nodes[i] for i in self.graph.previous_kcs(kc.id, limit=2) if i in self.graph.nodes]
        targets = await self._essential_targets(sess, kc, ctx)
        mistakes = sess.last_mistakes_summary.strip()
        mistake_count = len(re.findall(r"\bQ\d+", mistakes))
        target_n = min(len(targets), 6) + min(2, mistake_count) + (1 if review_kcs and difficulty != "easy" else 0)
        min_q = max(PRACTICE_MIN_Q, min(PRACTICE_MAX_Q, target_n))
        max_q = max(min_q, min(PRACTICE_MAX_Q, min_q + 2))
        if challenge:
            min_q = max(min_q, len(targets))
            max_q = max(max_q, min_q)
        seen = set(sess.seen_question_ids)
        from_bank = False
        questions: List[dict] = []
        if not challenge and self.bank.count(kc.id) >= min_q:
            questions = self.bank.draw(kc.id, min_q, exclude_ids=seen, difficulty=difficulty)
            from_bank = bool(questions)
        coverage_plan: Any = []
        if not questions:
            self._charge_budget(sess)
            _progress(f"Quiz sur « {kc.title} » ({min_q} à {max_q} questions)", "write")
            targets_payload = [{"target_id": f"E{i}", "target": t} for i, t in enumerate(targets, start=1)]
            review_payload = [{"id": r.id, "title": r.title} for r in review_kcs]
            prompt = (
                f"Notion à valider : « {kc.title} » ({kc.id}). Difficulté : {difficulty} "
                "(easy = reconnaissance et définitions ; medium = application ; hard = transfert, comparaison, distracteurs plausibles).\n"
                f"Nombre de questions attendu : entre {min_q} et {max_q}.\n"
                f"Cibles à couvrir (au moins une question par cible) : {targets_payload}\n"
                f"Notions précédentes utilisables pour 1 question d'intégration au plus (la question doit rester centrée sur la notion actuelle) : {review_payload or 'aucune'}\n"
                f"Erreurs récentes de l'apprenant à retravailler : {mistakes or 'aucune (premier essai)'}\n"
                f"Questions déjà posées à cet apprenant (formule des questions DIFFÉRENTES, angle ou exemple nouveau) : {sess.recent_stems.get(kc.id, [])[-12:] or 'aucune'}\n\n"
                f"Leçon reçue par l'apprenant :\n{sess.current_micro_lesson}\n\n"
                f"Extrait du mémento (source unique autorisée) :\n{self._kc_context(kc)}\n\n"
                "Contraintes : questions en français de moins de 240 caractères, 4 propositions courtes (moins de 90 caractères) et distinctes, une seule juste, "
                "answer = lettre, explanation = une phrase de justification avec la page, target_id = E1.. ou M1.. pour une erreur, page = page source, "
                "pas de question dont la réponse est visible dans l'énoncé, pas de doublon, coverage_plan qui liste chaque cible avec ses numéros de question."
            )
            out = await run_structured("Practice-QCM", INSTR_PRACTICE, prompt, PracticePack, ctx)
            questions = normalize_questions(out.questions, kc.id)
            coverage_plan = [c.model_dump() for c in out.coverage_plan]
            if challenge:
                expected_targets = {f"E{i}" for i in range(1, len(targets) + 1)}
                # Prefer one question per objective before filling remaining slots.
                def select_complete(pool):
                    unique = []
                    stems = set()
                    for q in pool:
                        stem = q['text'].strip().casefold()
                        if stem not in stems:
                            unique.append(q)
                            stems.add(stem)
                    selected = []
                    for target in sorted(expected_targets):
                        match = next((q for q in unique if q.get('target_id') == target), None)
                        if match is not None:
                            selected.append(match)
                    selected.extend(q for q in unique if q not in selected)
                    return selected[:max_q]
                questions = select_complete(questions)
                for repair in range(3):
                    missing = expected_targets - {q.get('target_id') for q in questions}
                    if expected_targets and not missing and len(questions) >= min_q:
                        break
                    if repair == 2 or not expected_targets:
                        raise RuntimeError('Quiz de validation incomplet : les objectifs doivent tous ?tre couverts.')
                    self._charge_budget(sess)
                    self._record_event(sess, ctx, {'event': 'validation_quiz_repair', 'kc_id': kc.id,
                        'attempt': repair + 1, 'missing_targets': sorted(missing), 'valid_questions': len(questions)})
                    repair_prompt = prompt + (
                        f"\nCompl?te le quiz avec {max(len(missing), min_q - len(questions))} nouvelles questions. "
                        f"Objectifs manquants obligatoires : {sorted(missing)}. "
                        f"Ne r?p?te pas ces ?nonc?s : {[q['text'] for q in questions]}."
                    )
                    extra = await run_structured('Practice-QCM-repair', INSTR_PRACTICE, repair_prompt, PracticePack, ctx)
                    questions = select_complete(questions + normalize_questions(extra.questions, kc.id))
                coverage_plan = [{'target_id': target, 'question_numbers': [i for i, q in enumerate(questions, 1)
                                  if q.get('target_id') == target]} for target in sorted(expected_targets)]
            else:
                if len(questions) < min(PRACTICE_MIN_Q, 3):
                    raise RuntimeError(f"quiz inexploitable ({len(questions)} questions valides)")
                questions = questions[:max_q]
        for i, q in enumerate(questions, start=1):
            q["number"] = i
            q["kc_id"] = kc.id
        sess.scope = "practice"
        sess.phase = "waiting_answers"
        sess.quiz_id = uuid.uuid4().hex[:12]
        sess.quiz_questions = {str(q["number"]): q for q in questions}
        sess.attempts_by_kc[kc.id] = attempts_before + 1
        sess.last_tutor_action = f"generate_{difficulty}_practice"
        sess.pending_hint_ladder = []
        sess.hint_index = 0
        sess.seen_question_ids = (sess.seen_question_ids + [str(q.get("id")) for q in questions if q.get("id")])[-400:]
        sess.recent_stems[kc.id] = (sess.recent_stems.get(kc.id, []) + [q["text"] for q in questions])[-24:]
        expected = {f"E{i}" for i in range(1, len(targets) + 1)}
        covered = {str(c.get("target_id")) for c in coverage_plan if isinstance(c, dict) and c.get("question_numbers")} if isinstance(coverage_plan, list) else set()
        self._record_event(sess, ctx, {
            "event": "practice_started", "tutor_action": sess.last_tutor_action, "kc_id": kc.id, "kc_title": kc.title, "difficulty": difficulty,
            "attempt": sess.attempts_by_kc[kc.id], "mastery_before": mastery, "n_questions": len(questions), "practice_min_questions": min_q,
            "practice_max_questions": max_q, "lesson_essential_count": len(targets), "lesson_essential_targets": targets,
            "practice_essential_targets": targets, "missing_coverage_target_ids": sorted(expected - covered) if not from_bank else [],
            "coverage_plan": coverage_plan, "review_kc_ids": [r.id for r in review_kcs], "review_kc_titles": [r.title for r in review_kcs],
            "source_pages": self.graph.kc_pages(kc), "from_bank": from_bank, "quiz_id": sess.quiz_id,
            "practice_length_strategy": {"basis": "targets + mistakes + integration", "difficulty": difficulty, "attempts_before": attempts_before, "mistake_count": mistake_count},
        })
        title = f"Quiz · {kc.title} · {len(questions)} questions"
        lead = f"Quiz sur « {kc.title} » : {len(questions)} questions. Validez à partir de {THRESHOLD:.0%} de bonnes réponses."
        return [self._text(lead), {"type": "qcm", "data": qcm_widget_data(title, questions)}]

    # =====================================================
    # MODULE CHECKPOINT
    # =====================================================
    async def _start_module_quiz(self, sess: Session, module_id: str, ctx: Any) -> List[Dict[str, Any]]:
        kc_ids = self.graph.module_kcs(module_id)
        kcs = [self.graph.nodes[i] for i in kc_ids if i in self.graph.nodes]
        title = self.graph.nodes[module_id].title if module_id in self.graph.nodes else "Chapitre"
        if not kcs:
            sess.module_gate_locked = False
            sess.pending_module_id = None
            return [self._text("Ce chapitre n'a pas de notion à contrôler."), self._actions(self._next_actions(sess))]
        n_q = max(MODULE_MIN_Q, min(MODULE_MAX_Q, len(kcs) * 2))
        per_kc = max(1, n_q // len(kcs))
        questions: List[dict] = []
        for kc in kcs:
            drawn = self.bank.draw(kc.id, per_kc, exclude_ids=set(sess.seen_question_ids))
            for q in drawn:
                q["kc_id"] = kc.id
            questions.extend(drawn)
        if len(questions) < n_q:
            self._charge_budget(sess)
            _progress(f"Contrôle du chapitre « {title} »", "check-circle")
            pages: List[int] = []
            for kc in kcs:
                pages.extend(self.graph.kc_pages(kc))
            need = n_q - len(questions)
            prompt = (
                f"Chapitre : « {title} ». Notions à couvrir (utilise leurs identifiants dans kc_id) : {[{'id': k.id, 'title': k.title} for k in kcs]}\n"
                f"Rédige {need} questions de contrôle réparties sur ces notions (au moins une par notion si possible), difficulté moyenne, "
                "en français, chacune vérifiable dans l'extrait ci-dessous.\n\n"
                f"{self.doc.pages_text(pages, max_chars=9000)}\n\n"
                "Contraintes : question de moins de 240 caractères ; 4 propositions courtes et distinctes ; une seule juste ; answer = lettre ; "
                "explanation = une phrase avec la page ; kc_id = identifiant de la notion testée ; page = page source ; pas de doublon."
            )
            out = await run_structured("Module-QCM", INSTR_QCM, prompt, QcmList, ctx)
            generated = normalize_questions(out.questions, kcs[0].id)
            valid_ids = {k.id for k in kcs}
            for q in generated:
                if q["kc_id"] not in valid_ids:
                    q["kc_id"] = kcs[0].id
            questions.extend(generated[:need])
        if len(questions) < max(3, MODULE_MIN_Q // 2):
            raise RuntimeError("contrôle de chapitre inexploitable")
        for i, q in enumerate(questions, start=1):
            q["number"] = i
        sess.scope = "module_quiz"
        sess.phase = "waiting_answers"
        sess.quiz_id = uuid.uuid4().hex[:12]
        sess.quiz_questions = {str(q["number"]): q for q in questions}
        sess.last_tutor_action = "start_module_checkpoint"
        sess.pending_hint_ladder = []
        sess.hint_index = 0
        self._record_event(sess, ctx, {"event": "module_checkpoint_started", "tutor_action": sess.last_tutor_action, "module_id": module_id, "module_title": title, "kc_ids": kc_ids, "n_questions": len(questions), "quiz_id": sess.quiz_id})
        lead = f"Contrôle du chapitre « {title} » : {len(questions)} questions sur ses {len(kcs)} notions. Seuil : {MODULE_THRESHOLD:.0%}."
        return [self._text(lead), {"type": "qcm", "data": qcm_widget_data(f"Contrôle · {title}", questions)}]

    # =====================================================
    # SCORING
    # =====================================================
    def _score(self, sess: Session, answers: Dict[int, str]) -> Tuple[float, Dict[str, Tuple[int, int]], List[dict]]:
        per: Dict[str, Tuple[int, int]] = {}
        items: List[dict] = []
        correct_total = 0
        for num_s, q in sorted(sess.quiz_questions.items(), key=lambda kv: int(kv[0])):
            num = int(num_s)
            learner = str(answers.get(num, "") or "").upper()[:1]
            correct = str(q.get("answer") or "").upper()
            ok = learner == correct
            correct_total += 1 if ok else 0
            c, t = per.get(q["kc_id"], (0, 0))
            per[q["kc_id"]] = (c + (1 if ok else 0), t + 1)
            choices = q.get("choices") or []
            items.append({
                "id": q.get("id"), "number": num, "question": q.get("text", ""), "choices": choices, "kc_id": q["kc_id"],
                "correct": ok, "learner_letter": learner or "-", "correct_letter": correct,
                "learner_choice": choices[LETTERS.index(learner)] if learner in LETTERS and len(choices) == 4 else "",
                "correct_choice": choices[LETTERS.index(correct)] if correct in LETTERS and len(choices) == 4 else "",
                "explanation": q.get("explanation", ""), "page": q.get("page"),
            })
        total = len(sess.quiz_questions) or 1
        return correct_total / total, per, items

    def _correction_block(self, sess: Session, items: List[dict], score: float, passed: Optional[bool], title: str) -> Dict[str, Any]:
        n_ok = sum(1 for it in items if it["correct"])
        return {"type": "widget", "title": title, "widget": W.correction_card(
            title, items, score_label=f"{n_ok}/{len(items)} · {score:.0%}", passed=passed, quiz_id=sess.quiz_id,
            kc_id=sess.current_kc_id or "")}

    def _clear_quiz(self, sess: Session) -> None:
        sess.quiz_questions = {}
        sess.phase = "idle"

    async def _process_answers(self, sess: Session, answers: Dict[int, str], ctx: Any) -> List[Dict[str, Any]]:
        if sess.phase != "waiting_answers" or not sess.quiz_questions:
            return [self._text("Aucun quiz en attente de réponses."), self._actions(self._next_actions(sess))]
        if sess.scope == "diagnostic" and sess.diagnostic_mode == "hierarchical":
            return await self._submit_hierarchical(sess, answers, ctx)
        overall, per_kc, items = self._score(sess, answers)
        mastery_before = dict(sess.mastery)
        wrong_items = [it for it in items if not it["correct"]]
        for kc_id, (c, t) in per_kc.items():
            score = c / t if t else 1.0
            sess.last_score_by_kc[kc_id] = score
            if sess.scope == "diagnostic":
                sess.diagnostic_raw_score_by_kc[kc_id] = score
                sess.diagnostic_profile[kc_id] = DIAGNOSTIC_MIN_MASTERY + score * (DIAGNOSTIC_MAX_MASTERY - DIAGNOSTIC_MIN_MASTERY)
                continue
            alpha = 0.6 if sess.scope == "practice" else 0.4
            sess.mastery[kc_id] = score if kc_id not in sess.mastery else (1 - alpha) * sess.mastery[kc_id] + alpha * score

        if sess.scope == "diagnostic":
            return await self._after_diagnostic(sess, ctx, overall, per_kc, items, mastery_before)
        if sess.scope == "module_quiz":
            return await self._after_module(sess, ctx, overall, items, wrong_items, mastery_before)
        return await self._after_practice(sess, ctx, per_kc, items, wrong_items, mastery_before)

    async def _after_diagnostic(self, sess: Session, ctx: Any, overall: float, per_kc: Dict[str, Tuple[int, int]], items: List[dict], mastery_before: Dict[str, float]) -> List[Dict[str, Any]]:
        weak = [k for k in self.graph.kc_ids() if k in per_kc and per_kc[k][0] < per_kc[k][1]]
        sess.weak_queue = weak
        sess.diagnostic_done = True
        first = weak[0] if weak else self.graph.kc_ids()[0]
        sess.current_kc_id = first
        kc = self.graph.nodes[first]
        self._clear_quiz(sess)
        sess.scope = "practice"
        sess.current_micro_lesson = ""
        sess.last_mistakes_summary = ""
        sess.last_tutor_action = "diagnose_weak_kc_then_micro_lesson"
        correction = self._correction_block(sess, items, overall, None, "Corrigé du diagnostic")
        self._record_event(sess, ctx, {
            "event": "diagnostic_submitted", "tutor_action": sess.last_tutor_action, "evidence_role": "screening_only",
            "mastery_interpretation": "Diagnostic estimates candidate weakness; it does not validate mastery.",
            "diagnostic_mastery_cap": {"min": DIAGNOSTIC_MIN_MASTERY, "max": DIAGNOSTIC_MAX_MASTERY},
            "overall_score": overall, "weakness_kc_id": first, "weakness_kc_title": kc.title, "weak_queue": weak, "per_kc": per_kc,
            "diagnostic_raw_score_by_kc": dict(sess.diagnostic_raw_score_by_kc), "diagnostic_profile": dict(sess.diagnostic_profile),
            "mastery_before": mastery_before, "mastery_after": dict(sess.mastery), "mastery_update": "none_from_diagnostic",
            "source_pages": self.graph.kc_pages(kc), "quiz_id": sess.quiz_id,
        })
        lesson = await self._build_lesson(sess, kc, ctx)
        sess.current_micro_lesson = lesson
        n_ok = sum(1 for it in items if it["correct"])
        if weak:
            names = ", ".join(self.graph.nodes[k].title for k in weak[:4]) + (" et d'autres" if len(weak) > 4 else "")
            summary = (
                f"Diagnostic terminé : {n_ok}/{len(items)} bonnes réponses. Ce n'est qu'une estimation.\n"
                f"Notions à travailler en priorité : {names}.\n\n"
                f"On commence par « {kc.title} ». Lisez la leçon, puis lancez le quiz."
            )
        else:
            summary = (
                f"Diagnostic terminé : {n_ok}/{len(items)}, tout juste. Une question par notion ne prouve pas la maîtrise : "
                f"le parcours reprend depuis le début, en mode rapide. Première notion : « {kc.title} »."
            )
        return [correction, self._text(summary), self._text(f"{self._kc_label(kc)}\n\n{lesson}"), self._source_card(kc, [("Lancer le quiz", "quiz"), ("Ma progression", "ma progression")])]

    async def _after_practice(self, sess: Session, ctx: Any, per_kc: Dict[str, Tuple[int, int]], items: List[dict], wrong_items: List[dict], mastery_before: Dict[str, float]) -> List[Dict[str, Any]]:
        kc = self._kc(sess.current_kc_id)
        if not kc:
            self._clear_quiz(sess)
            return [self._text("Notion en cours introuvable.")]
        c, t = per_kc.get(kc.id, (0, len(sess.quiz_questions)))
        score = c / max(1, t)
        state = build_learner_state(sess, self.graph)
        practice_result = {'score': score, 'threshold': THRESHOLD,
                           'remediation_below': REMEDIATION_LESSON_BELOW}
        decision = decide_next_action(state, [], practice_result=practice_result)
        self._record_event(sess, ctx, {'event': 'pedagogical_decision', 'phase': 'practice',
            'quiz_id': sess.quiz_id, 'learner_state': state, 'practice_result': practice_result,
            'curriculum': [], 'decision': decision})
        sess.practice_evidence_by_kc.setdefault(kc.id, []).append({
            'quiz_id': sess.quiz_id, 'score': score, 'items': items,
            'validation_challenge': sess.diagnostic_validation_challenge,
            'lesson_available': bool(sess.current_micro_lesson),
            'objective_ids_assessed': sorted({q.get('target_id') for q in sess.quiz_questions.values() if q.get('target_id')}),
        })
        self._clear_quiz(sess)
        quiz_id = sess.quiz_id
        if decision['action'] != 'advance':
            sess.diagnostic_validation_challenge = False
            sess.can_advance = False
            sess.validated_kc_id = None
            labels: List[str] = []
            for wi in wrong_items:
                lab = f"confond « {wi.get('learner_choice') or wi.get('learner_letter')} » avec « {wi.get('correct_choice') or wi.get('correct_letter')} »"
                labels.append(lab)
            sess.misconceptions[kc.id] = (sess.misconceptions.get(kc.id, []) + labels)[-6:]
            sess.last_mistakes_summary = "\n".join(f"Q{wi['number']}: répondu {wi['learner_letter']} ({wi.get('learner_choice')}) au lieu de {wi['correct_letter']} ({wi.get('correct_choice')}) | {wi['question'][:160]}" for wi in wrong_items[:8])
            action = "hint_ladder_then_remediate" if labels else "retry_with_micro_lesson"
            sess.last_tutor_action = action
            self._charge_budget(sess)
            _progress("Analyse de vos erreurs", "lightbulb")
            fb = await self._feedback(kc, wrong_items, score, ctx)
            sess.pending_hint_ladder = [h for h in fb.hints if h.strip()][:3] or [
                f"Relisez la page {self.graph.kc_pages(kc)[0] if self.graph.kc_pages(kc) else ''} du mémento en cherchant la règle qui sépare vos réponses des bonnes.",
                "Comparez la forme, la couleur et l'état de chaque symbole : lequel change le sens ?",
                "Reformulez la règle avec vos mots, puis appliquez-la à la question ratée.",
            ]
            sess.hint_index = 0
            blocks: List[Dict[str, Any]] = [self._correction_block(sess, items, score, False, f"Corrigé · {kc.title}")]
            if decision['action'] == 'remediate':
                lesson = await self._build_lesson(sess, kc, ctx, mistakes_summary=sess.last_mistakes_summary)
                sess.current_micro_lesson = lesson
                blocks.append(self._text(f"Notion non validée ({score:.0%}, seuil {THRESHOLD:.0%}).\n\n{fb.summary}\n\nLeçon de reprise :\n\n{lesson}"))
            else:
                blocks.append(self._text(f"Notion non validée ({score:.0%}, seuil {THRESHOLD:.0%}), mais vous n'êtes pas loin.\n\n{fb.summary}"))
            self._record_event(sess, ctx, {
                "event": "practice_submitted", "tutor_action": action, "decision_reason": "score below threshold", "kc_id": kc.id, "kc_title": kc.title,
                "score": score, "threshold": THRESHOLD, "passed": False, "wrong_items": wrong_items, "misconceptions": labels,
                "mastery_before": mastery_before, "mastery_after": dict(sess.mastery), "attempt": sess.attempts_by_kc.get(kc.id, 0),
                "source_pages": self.graph.kc_pages(kc), "quiz_id": quiz_id, "hints": sess.pending_hint_ladder,
            })
            blocks.append(self._source_card(kc, [("Refaire le quiz", "quiz"), ("Un indice", "indice"), ("Revoir la leçon", "revoir la leçon")]))
            return blocks

        sess.can_advance = True
        sess.diagnostic_validation_challenge = False
        sess.validated_kc_id = kc.id
        if kc.id not in sess.validated_kc_ids:
            sess.validated_kc_ids.append(kc.id)
        if sess.diagnostic_mode == 'hierarchical' and sess.diagnostic_stage == 'learning' and not sess.diagnostic_done:
            self._record_event(sess, ctx, {'event': 'practice_submitted', 'tutor_action': 'validate_kc',
                'kc_id': kc.id, 'kc_title': kc.title, 'score': score, 'threshold': THRESHOLD,
                'passed': True, 'mastery_before': mastery_before, 'mastery_after': dict(sess.mastery),
                'quiz_id': quiz_id, 'decision_reason': 'resume_paused_diagnostic'})
            sess.diagnostic_taught_kc_ids.append(kc.id)
            sess.diagnostic_stage = 'resuming'
            sess.scope = 'diagnostic'
            sess.can_advance = False
            sess.validated_kc_id = None
            sess.current_kc_id = None
            sess.current_micro_lesson = ''
            sess.pending_hint_ladder = []
            sess.last_mistakes_summary = ''
            await self._save_session(sess)
            return [self._correction_block(sess, items, score, True, f'Corrigé · {kc.title}'),
                    self._text('Notion validée. Reprenons le diagnostic à la position enregistrée.'),
                    self._actions([('Reprendre le diagnostic', 'commencer le diagnostic')])]
        sess.weak_queue = [k for k in sess.weak_queue if k != kc.id]
        sess.pending_hint_ladder = []
        sess.hint_index = 0
        nxt = self._learning_next(sess, kc.id)
        sess.pending_next_kc_id = nxt if nxt in self.graph.nodes else None
        cur_module = self.graph.module_of(kc.id)
        next_module = self.graph.module_of(nxt) if nxt else None
        blocks = [self._correction_block(sess, items, score, True, f"Corrigé · {kc.title}")]
        if cur_module and (not nxt or next_module != cur_module):
            sess.pending_module_id = cur_module
            sess.module_gate_locked = True
            sess.pending_module_retry = False
            sess.last_tutor_action = "validate_kc_then_module_checkpoint"
            self._record_event(sess, ctx, {
                "event": "practice_submitted", "tutor_action": sess.last_tutor_action, "decision_reason": "KC validated, end of module", "kc_id": kc.id,
                "kc_title": kc.title, "score": score, "threshold": THRESHOLD, "passed": True, "mastery_before": mastery_before,
                "mastery_after": dict(sess.mastery), "attempt": sess.attempts_by_kc.get(kc.id, 0), "source_pages": self.graph.kc_pages(kc),
                "module_id": cur_module, "quiz_id": quiz_id,
            })
            blocks.append(self._text(f"Notion validée : « {kc.title} » ({score:.0%}). C'était la dernière du chapitre « {self.graph.module_title(kc.id)} » : place au contrôle du chapitre."))
            blocks.extend(await self._start_module_quiz(sess, cur_module, ctx))
            return blocks
        sess.last_tutor_action = "validate_kc"
        self._record_event(sess, ctx, {
            "event": "practice_submitted", "tutor_action": "validate_kc", "decision_reason": "score reached threshold", "kc_id": kc.id, "kc_title": kc.title,
            "score": score, "threshold": THRESHOLD, "passed": True, "mastery_before": mastery_before, "mastery_after": dict(sess.mastery),
            "attempt": sess.attempts_by_kc.get(kc.id, 0), "source_pages": self.graph.kc_pages(kc), "next_kc_id": sess.pending_next_kc_id, "quiz_id": quiz_id,
        })
        if sess.pending_next_kc_id:
            next_kc = self.graph.nodes[sess.pending_next_kc_id]
            blocks.append(self._text(f"Notion validée : « {kc.title} » ({score:.0%}). Suivante : « {next_kc.title} »."))
            blocks.append(self._actions([("Notion suivante", "notion suivante"), ("Ma progression", "ma progression"), ("Poser une question", "question")]))
        else:
            blocks.append(self._text(f"Notion validée : « {kc.title} » ({score:.0%}). Vous avez terminé le parcours du mémento."))
            blocks.append(self._actions([("Ma progression", "ma progression")]))
        return blocks

    def completion_status(self, sess):
        missing_kcs = [k for k in self.graph.kc_ids() if k not in sess.validated_kc_ids]
        missing_modules = [mid for mid, _, ids in self.graph.modules_in_order()
                           if mid != 'ROOT' and ids and sess.module_mastery.get(mid, 0) < MODULE_THRESHOLD]
        return {'complete': not missing_kcs and not missing_modules,
                'missing_kc_ids': missing_kcs, 'missing_module_ids': missing_modules}

    async def _completion_checkpoint(self, sess, ctx):
        status = self.completion_status(sess)
        if not status['missing_kc_ids'] and status['missing_module_ids']:
            sess.pending_module_id = status['missing_module_ids'][0]
            sess.module_gate_locked = True
            sess.pending_module_retry = True  # remains retryable if generation fails
            sess.pending_next_kc_id = None
            return await self._start_module_quiz(sess, sess.pending_module_id, ctx)
        return []

    async def _after_module(self, sess: Session, ctx: Any, score: float, items: List[dict], wrong_items: List[dict], mastery_before: Dict[str, float]) -> List[Dict[str, Any]]:
        module_id = sess.pending_module_id
        title = self.graph.nodes[module_id].title if module_id and module_id in self.graph.nodes else "Chapitre"
        self._clear_quiz(sess)
        quiz_id = sess.quiz_id
        if module_id:
            sess.module_mastery[module_id] = score
        blocks = [self._correction_block(sess, items, score, score >= MODULE_THRESHOLD, f"Corrigé · contrôle « {title} »")]
        if score < MODULE_THRESHOLD:
            sess.module_gate_locked = True
            sess.pending_module_retry = True
            sess.last_tutor_action = "retry_module_checkpoint"
            sess.scope = "practice"
            self._record_event(sess, ctx, {"event": "module_checkpoint_submitted", "tutor_action": sess.last_tutor_action, "decision_reason": "score below threshold", "module_id": module_id, "score": score, "threshold": MODULE_THRESHOLD, "wrong_items": wrong_items, "mastery_before": mastery_before, "mastery_after": dict(sess.mastery), "quiz_id": quiz_id})
            weak_titles = sorted({self.graph.nodes[it["kc_id"]].title for it in wrong_items if it["kc_id"] in self.graph.nodes})
            blocks.append(self._text(f"Chapitre non validé ({score:.0%}, seuil {MODULE_THRESHOLD:.0%}). Notions à revoir : {', '.join(weak_titles) or 'voir le corrigé'}. Relisez les pages indiquées, puis refaites le contrôle."))
            blocks.append(self._actions([("Refaire le contrôle du chapitre", "controle"), ("Revoir la leçon", "revoir la leçon"), ("Poser une question", "question")]))
            return blocks
        sess.module_gate_locked = False
        sess.pending_module_id = None
        sess.pending_module_retry = False
        sess.last_tutor_action = "validate_module"
        nxt = sess.pending_next_kc_id
        self._record_event(sess, ctx, {"event": "module_checkpoint_submitted", "tutor_action": sess.last_tutor_action, "decision_reason": "score reached threshold", "module_id": module_id, "score": score, "threshold": MODULE_THRESHOLD, "wrong_items": wrong_items, "mastery_before": mastery_before, "mastery_after": dict(sess.mastery), "next_kc_id": nxt, "quiz_id": quiz_id})
        if not self.completion_status(sess)['missing_kc_ids']:
            pending = await self._completion_checkpoint(sess, ctx)
            if pending:
                return blocks + pending
        if not nxt or nxt not in self.graph.nodes:
            blocks.append(self._text(f"Chapitre « {title} » validé ({score:.0%}). Vous avez terminé le parcours du mémento."))
            blocks.append(self._actions([("Ma progression", "ma progression")]))
            return blocks
        blocks.append(self._text(f"Chapitre « {title} » validé ({score:.0%})."))
        blocks.extend(await self._next_kc(sess, ctx, force=True))
        return blocks

    async def _feedback(self, kc: KCNode, wrong_items: List[dict], score: float, ctx: Any) -> Feedback:
        payload = [{"question": wi["question"], "reponse_donnee": f"{wi['learner_letter']}) {wi.get('learner_choice')}", "bonne_reponse": f"{wi['correct_letter']}) {wi.get('correct_choice')}", "justification": wi.get("explanation", "")} for wi in wrong_items[:6]]
        prompt = (
            f"Notion : « {kc.title} ». Score : {score:.0%}. Erreurs :\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            f"Extrait du mémento :\n{self._kc_context(kc, neighbours=False, max_chars=5000)}\n\n"
            "Rédige en français : summary = 2 à 4 phrases (lecture du score, les corrections essentielles regroupées, une mini-remédiation concrète) ; "
            "corrections = une phrase par erreur importante ; hints = 3 indices progressifs, spécifiques aux questions ratées, qui guident vers la règle "
            "sans donner la réponse (le premier oriente vers la page et le bon critère, le deuxième élimine une confusion, le troisième formule presque la règle)."
        )
        try:
            return await run_structured("Feedback", INSTR_FEEDBACK, prompt, Feedback, ctx)
        except Exception as exc:  # noqa: BLE001
            print(f"[feedback] {type(exc).__name__}: {str(exc)[:160]}")
            return Feedback(summary="Regardez le corrigé ci-dessus : chaque bonne réponse est justifiée avec sa page. Relisez la page, puis refaites le quiz.", corrections=[], hints=[])

    def _next_hint_text(self, sess: Session) -> str:
        if not sess.pending_hint_ladder:
            return "Aucun indice en attente : les indices apparaissent après un quiz non validé."
        idx = min(sess.hint_index, len(sess.pending_hint_ladder) - 1)
        hint = sess.pending_hint_ladder[idx]
        sess.hint_index = min(idx + 1, len(sess.pending_hint_ladder))
        if sess.hint_index >= len(sess.pending_hint_ladder):
            return f"Indice {idx + 1}/{len(sess.pending_hint_ladder)} : {hint}\n\nC'était le dernier indice. Refaites le quiz quand vous êtes prêt."
        return f"Indice {idx + 1}/{len(sess.pending_hint_ladder)} : {hint}"

    # =====================================================
    # NEXT NOTION
    # =====================================================
    async def _next_kc(self, sess: Session, ctx: Any, *, force: bool = False) -> List[Dict[str, Any]]:
        if not force:
            if not sess.can_advance or sess.validated_kc_id != sess.current_kc_id:
                return [self._text("Validez d'abord la notion en cours avec le quiz (70 % de bonnes réponses)."), self._actions([("Lancer le quiz", "quiz"), ("Revoir la leçon", "revoir la leçon")])]
            if sess.module_gate_locked:
                mod = self.graph.nodes[sess.pending_module_id].title if sess.pending_module_id in self.graph.nodes else "chapitre"
                return [self._text(f"Le contrôle du chapitre « {mod} » doit être réussi avant de continuer."), self._actions([("Refaire le contrôle du chapitre", "controle")] if sess.pending_module_retry else [("Poser une question", "question")])]
        nxt = sess.pending_next_kc_id or self._learning_next(sess, sess.current_kc_id)
        if not nxt or nxt not in self.graph.nodes:
            sess.can_advance = False
            sess.validated_kc_id = None
            sess.pending_next_kc_id = None
            return [self._text("Vous avez terminé le parcours du mémento. Bravo."), self._actions([("Ma progression", "ma progression")])]
        sess.current_kc_id = nxt
        sess.pending_next_kc_id = None
        sess.scope = "practice"
        sess.phase = "idle"
        sess.last_mistakes_summary = ""
        sess.current_micro_lesson = ""
        sess.can_advance = False
        sess.validated_kc_id = None
        sess.pending_hint_ladder = []
        sess.hint_index = 0
        kc = self.graph.nodes[nxt]
        lesson = await self._build_lesson(sess, kc, ctx)
        sess.current_micro_lesson = lesson
        known = sess.diagnostic_raw_score_by_kc.get(kc.id, 0.0) >= 1.0
        note = " Vous aviez juste au diagnostic : la leçon est courte, le quiz confirmera." if known else ""
        return [self._text(f"{self._kc_label(kc)} · chapitre « {self.graph.module_title(kc.id)} ».{note}\n\n{lesson}"), self._source_card(kc, [("Lancer le quiz", "quiz"), ("Poser une question", "question")])]

    # =====================================================
    # FREE QUESTIONS AND IMAGES
    # =====================================================
    async def _answer_free_question(self, sess: Session, question: str, ctx: Any) -> List[Dict[str, Any]]:
        self._charge_budget(sess)
        kc = self._kc(sess.current_kc_id)
        pages: List[int] = list(self.graph.kc_pages(kc)) if kc else []
        evidence, pages, retrieval = retrieve_question_context(self.doc, question, pages)
        found = pages
        later: List[str] = []
        if kc:
            cur_idx = self.graph.index_by_kc.get(kc.id, 0)
            for p in found:
                for kid in self.graph.kc_ids():
                    if p in self.graph.kc_pages(self.graph.nodes[kid]) and self.graph.index_by_kc.get(kid, 0) > cur_idx:
                        later.append(self.graph.nodes[kid].title)
        _progress("Je cherche dans le mémento", "search")
        prompt = (
            f"Question de l'apprenant : {question}\n\n"
            + (f"Notion en cours : « {kc.title} ».\n" if kc else "Aucune notion en cours (parcours pas encore commencé).\n")
            + (f"Leçon en cours :\n{sess.current_micro_lesson[:1500]}\n\n" if sess.current_micro_lesson else "")
            + f"Extraits du mémento (pages {pages}) :\n{evidence}\n\n"
            "Réponds en français, de façon concise et pédagogique, uniquement à partir des extraits, en citant la ou les pages. "
            "Si la réponse n'est pas dans le mémento, dis-le clairement et propose la page la plus proche."
        )
        out = await run_structured("Learner-question", INSTR_QA + " " + SVG_INSTRUCTIONS,
                                   prompt, IllustratedAnswer, ctx)
        answer = out.answer
        illustration = []
        if out.svg:
            try:
                if not out.illustration_pages or not set(out.illustration_pages).issubset(pages):
                    raise ValueError('Invalid illustration pages')
                svg = validate_svg(out.svg)
                illustration = [illustration_widget(svg, sorted(set(out.illustration_pages)), self.doc.pdf_url)]
            except Exception as exc:
                print(f"[free-question SVG] {type(exc).__name__}: {exc}")
                self._record_event(sess, ctx, {'event': 'free_question_svg_rejected',
                    'error_type': type(exc).__name__, 'reason': str(exc)[:200]})
                answer += "\n\nLe schéma n’a pas pu être affiché. Consultez les pages citées."

        if later and kc:
            answer += f"\n\nCe point est détaillé plus loin dans le parcours (notion « {later[0]} ») ; vous y reviendrez avec un quiz."
        sess.last_tutor_action = "answer_lesson_question"
        self._record_event(sess, ctx, {"event": "learner_question_answered", "tutor_action": sess.last_tutor_action, "question": question, "current_kc_id": kc.id if kc else None, "source_pages": sorted(set(pages)), "retrieval": retrieval, "context_chars": len(evidence), "answer_preview": answer[:1000]})
        return [self._text(answer), *illustration, self._actions(self._next_actions(sess))]

    async def _answer_visual_question(self, sess: Session, question: str, image_urls: List[str], ctx: Any) -> List[Dict[str, Any]]:
        self._charge_budget(sess)
        is_followup = False
        if image_urls:
            sess.last_visual_image_urls = list(image_urls)
            sess.visual_question_history = []
        else:
            image_urls = list(sess.last_visual_image_urls)
            is_followup = True
        if not image_urls:
            return [self._text("Envoyez d'abord la photo d'un symbole, puis posez votre question.")]
        kc = self._kc(sess.current_kc_id)
        _progress("J'identifie les éléments visibles dans l'image", "images")
        observation_content = [{"type": "input_text", "text": f"Question : {question or 'Décris cette image.'}\nRelève les éléments visibles utiles pour répondre, sans inventer leur signification."}]
        observation_content.extend({"type": "input_image", "image_url": url, "detail": "high"} for url in image_urls)
        inventory = await run_structured(
            "Visual-inventory", OBSERVE_INSTRUCTIONS,
            [{"role": "user", "content": observation_content}], VisualInventory, ctx, vision=True,
        )
        _progress("Je cherche les références de chaque élément dans le cours", "search")
        evidence, pages = await asyncio.to_thread(retrieve_visual_evidence, self.doc, inventory, question)
        self._charge_budget(sess)
        prompt = (
            f"Question : {question or 'Décris cette image.'}\n"
            + (f"Échanges récents sur la même image : {sess.visual_question_history[-4:]}\n" if is_followup and sess.visual_question_history else "")
            + f"\nObservations visuelles (provisoires, à vérifier sur les images) :\n{inventory.model_dump_json()}\n"
            + f"\nPassages candidats recherchés séparément pour chaque élément dans tout le cours :\n{evidence}\n\n"
        )
        content: List[dict] = [{"type": "input_text", "text": prompt}]
        for url in image_urls:
            content.append({"type": "input_image", "image_url": url, "detail": "high"})
        answer = await run_text("Visual-symbol", INSTR_VISUAL, [{"role": "user", "content": content}], ctx, vision=True)
        sess.visual_question_history = (sess.visual_question_history + [{"question": question or "Décris cette image.", "answer": answer[:1200]}])[-8:]
        sess.last_tutor_action = "answer_visual_pdf_question"
        self._record_event(sess, ctx, {"event": "visual_question_answered", "tutor_action": sess.last_tutor_action, "question": question, "n_images": len(image_urls), "image_context_reused": is_followup, "scope": "full_pdf_course", "current_kc_id": kc.id if kc else None, "source_pages": sorted(set(pages)), "answer_preview": answer[:1000]})
        return [self._text(answer), self._actions([("Oublier la photo", "oublier la photo")] + self._next_actions(sess))]


# =====================================================
# INSTRUCTIONS (system prompts)
# =====================================================
INSTR_QCM = (
    "Tu es formateur de sapeurs-pompiers. Tu rédiges des questions à choix multiples en français, uniquement à partir de l'extrait du mémento fourni "
    "dans le message. Chaque question a exactement 4 propositions distinctes et une seule bonne réponse, désignée par sa lettre. "
    "Les distracteurs sont plausibles (autres symboles, couleurs ou règles du mémento), jamais absurdes. La justification cite la page. "
    "Conserve les termes officiels du mémento. N'utilise aucune connaissance extérieure."
)
INSTR_PRACTICE = INSTR_QCM + " Tu construis un quiz d'entraînement adaptatif qui couvre chaque cible d'évaluation fournie et retravaille les erreurs signalées."
INSTR_LESSON = (
    "Tu es formateur de sapeurs-pompiers. Tu rédiges une micro-leçon en français sur une seule notion du mémento GOC, à partir du seul extrait fourni. "
    "Précis, concret, sans bavardage. Conserve les libellés, couleurs, formes et abréviations officiels exactement. N'invente aucune règle. "
    "Texte brut dans chaque champ : pas de Markdown (ni gras, ni titres), pas d'identifiant technique (g_k2, KC) dans le titre."
)
INSTR_TARGETS = "Tu extrais des cibles d'évaluation précises et testables à partir d'un extrait de doctrine, en français."
INSTR_FEEDBACK = (
    "Tu es formateur de sapeurs-pompiers, bienveillant et précis. Tu expliques des erreurs de quiz en français à partir du mémento, "
    "sans jamais exposer d'étiquette interne, et tu formules des indices qui guident sans donner la réponse."
)
INSTR_QA = (
    "Tu es formateur de sapeurs-pompiers. Tu réponds en français aux questions sur le mémento GOC (outils graphiques), uniquement d'après les extraits fournis, "
    "en citant les pages. Concis, pédagogique, sans invention, mise en forme sobre (pas de titres, gras limité aux termes officiels)."
)
INSTR_VISUAL = (
    "Réponds en français directement à la question sur l'image, sans plan imposé. "
    "Utilise les extraits du mémento seulement s'ils sont pertinents et cite les pages qui soutiennent tes explications métier. "
    "N'invente pas de convention graphique ; indique simplement ce qui reste incertain."
)
