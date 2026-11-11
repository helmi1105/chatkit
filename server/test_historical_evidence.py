import unittest
from types import SimpleNamespace

from app.historical_evidence import (
    historical_error_candidates,
    prompt_history,
    raw_chat_history,
)


class Graph:
    ordered_kc_ids = ["g1", "g2", "g3"]
    index_by_kc = {"g1": 0, "g2": 1, "g3": 2}
    nodes = {kid: SimpleNamespace(title=kid.upper()) for kid in ordered_kc_ids}
    parent_by_id = {"g1": "m1", "g2": "m1", "g3": "m1"}

    def previous_kcs(self, kc_id, limit=2):
        return self.ordered_kc_ids[: self.index_by_kc[kc_id]][-limit:]

    def module_of(self, kc_id):
        return "m1"


class HistoricalEvidenceTests(unittest.TestCase):
    def test_prior_wrong_answer_becomes_stable_candidate(self):
        session = SimpleNamespace(
            kc_essential_targets={"g1": ["Recognise symbol"]},
            objective_evidence_by_kc={}, misconceptions={}, validated_kc_ids=["g1"],
            practice_evidence_by_kc={"g1": [{"items": [{
                "correct": False, "target_id": "E1", "question": "Which symbol?",
                "learner_choice": "Circle", "correct_choice": "Triangle", "page": 2,
            }]}]},
        )
        rows = historical_error_candidates(session, Graph(), "g3")
        self.assertEqual(rows[0]["history_id"], "g1.E1")
        self.assertEqual(rows[0]["objective"], "Recognise symbol")
        self.assertEqual(rows[0]["relation"]["basis"], "ecg")

    def test_no_ecg_prompt_hides_kc_identity_but_keeps_error(self):
        row = {
            "history_id": "g1.E1", "kc_id": "g1", "kc_title": "One",
            "objective": "Recognise symbol", "question": "Which symbol?",
            "learner_answer": "Circle", "correct_answer": "Triangle",
            "resolved": True, "failures": 1, "relation": {"basis": "none"},
        }
        payload = prompt_history([row])[0]
        self.assertEqual(payload["history_id"], "H01")
        self.assertNotIn("kc_id", payload)
        self.assertEqual(payload["question"], "Which symbol?")

    def test_raw_chat_history_contains_dialogue_without_structured_labels(self):
        session = SimpleNamespace(
            practice_evidence_by_kc={"g1": [{"items": [{
                "correct": False,
                "target_id": "E1",
                "question": "Quelle forme faut-il choisir ?",
                "learner_choice": "Un cercle",
                "correct_choice": "Un triangle",
                "explanation": "Le triangle correspond au symbole demandé.",
            }]}]},
        )

        transcript = raw_chat_history(session, Graph(), "g3")

        self.assertIn("Tuteur : Quelle forme faut-il choisir ?", transcript)
        self.assertIn("Apprenant : Un cercle", transcript)
        self.assertIn("Correction : Un triangle", transcript)
        self.assertNotIn("g1", transcript)
        self.assertNotIn("E1", transcript)
        self.assertNotIn("failures", transcript)
        self.assertNotIn("maitrise", transcript.casefold())


if __name__ == "__main__":
    unittest.main()
