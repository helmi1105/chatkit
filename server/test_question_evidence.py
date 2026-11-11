import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.orchestrator import KC_GRAPH_PATH, KcGraph, Orchestrator, Session, PRACTICE_MASTERY_ALPHA
from app.question_evidence import credited_kcs


def question(number, role, ids, answer="A", target_id="E1"):
    return {"id": f"q{number}", "number": number, "text": f"Question {number}",
            "choices": ["one", "two", "three", "four"], "answer": answer,
            "kc_id": "g_k4", "role": role, "assessed_kc_ids": ids,
            "target_id": target_id}


class QuestionEvidenceTests(unittest.TestCase):
    def test_roles_and_legacy(self):
        self.assertEqual(credited_kcs(question(1, "current", ["g_k4"]), "g_k4", ["g_k3"], "practice"), ["g_k4"])
        self.assertEqual(credited_kcs(question(1, "review", ["g_k3"]), "g_k4", ["g_k3"], "practice"), ["g_k3"])
        self.assertEqual(credited_kcs(question(1, "integration", ["g_k3", "g_k4"]), "g_k4", ["g_k3"], "practice"), ["g_k4"])
        self.assertEqual(credited_kcs(question(1, "review", ["g_k4"]), "g_k4", ["g_k3"], "practice"), [])
        self.assertEqual(credited_kcs(question(1, "current", ["g_k3"]), "g_k4", ["g_k3"], "practice"), [])
        self.assertEqual(credited_kcs({"kc_id": "g_k4"}, "g_k4", ["g_k3"], "practice"), ["g_k4"])

    def test_scores_are_per_attributed_kc(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        sess = Session(scope="practice", current_kc_id="g_k4", quiz_questions={
            "1": question(1, "review", ["g_k3"]),
            "2": question(2, "current", ["g_k4"]),
            "3": question(3, "integration", ["g_k3", "g_k4"]),
        })
        total, per, items = orch._score(sess, {1: "A", 2: "B", 3: "A"})
        self.assertAlmostEqual(total, 2 / 3)
        self.assertEqual(per, {"g_k3": (1, 1), "g_k4": (1, 2)})
        self.assertEqual(items[0]["credited_kc_ids"], ["g_k3"])
        self.assertEqual(items[2]["credited_kc_ids"], ["g_k4"])


class PracticeAttributionTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_success_cannot_validate_current_kc(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._record_event = Mock()
        orch._correction_block = Mock(return_value={"type": "text", "text": "correction"})
        orch._source_card = Mock(return_value={"type": "text", "text": "source"})
        orch._feedback = AsyncMock(return_value=SimpleNamespace(summary="Try again", hints=[]))
        orch._build_lesson = AsyncMock(return_value="Lesson")
        orch._charge_budget = Mock()
        sess = Session(scope="practice", phase="waiting_answers", current_kc_id="g_k4",
                       kc_essential_targets={"g_k4": ["state"]},
                       quiz_questions={"1": question(1, "review", ["g_k3"]),
                                       "2": question(2, "current", ["g_k4"])})
        await orch._process_answers(sess, {1: "A", 2: "B"}, None)
        self.assertEqual(sess.mastery["g_k3"], 1)
        self.assertEqual(sess.mastery["g_k4"], 0)
        self.assertNotIn("g_k4", sess.validated_kc_ids)
        events = [call.args[2] for call in orch._record_event.call_args_list]
        provenance = [e for e in events if e.get("event") == "question_update_provenance"]
        self.assertEqual({e["updated_kc_id"] for e in provenance}, {"g_k3", "g_k4"})
        self.assertEqual(provenance[0]["contributions"][0]["question_id"], "q1")
        self.assertEqual(provenance[0]["scope"], "practice")
        self.assertIn("before", provenance[0])
        self.assertEqual(provenance[0]["after"], sess.mastery[provenance[0]["updated_kc_id"]])

    async def test_existing_mastery_uses_exported_production_alpha(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._record_event = Mock()
        orch._correction_block = Mock(return_value={"type": "text", "text": "correction"})
        orch._source_card = Mock(return_value={"type": "text", "text": "source"})
        orch._feedback = AsyncMock(return_value=SimpleNamespace(summary="Try again", hints=[]))
        sess = Session(scope="practice", phase="waiting_answers", current_kc_id="g_k4",
                       mastery={"g_k4": .25}, kc_essential_targets={"g_k4": ["state"]},
                       quiz_questions={"1": question(1, "current", ["g_k4"])})
        await orch._process_answers(sess, {1: "A"}, None)
        self.assertAlmostEqual(sess.mastery["g_k4"], (1-PRACTICE_MASTERY_ALPHA)*.25 + PRACTICE_MASTERY_ALPHA)

    async def test_current_success_validates_despite_wrong_review(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._record_event = Mock()
        orch._correction_block = Mock(return_value={"type": "text", "text": "correction"})
        sess = Session(scope="practice", phase="waiting_answers", current_kc_id="g_k4",
                       kc_essential_targets={"g_k4": ["state"]},
                       quiz_questions={"1": question(1, "review", ["g_k3"]),
                                       "2": question(2, "current", ["g_k4"])})
        await orch._process_answers(sess, {1: "B", 2: "A"}, None)
        self.assertIn("g_k4", sess.validated_kc_ids)
        self.assertEqual(sess.last_score_by_kc["g_k4"], 1)
        self.assertEqual(sess.last_score_by_kc["g_k3"], 0)
