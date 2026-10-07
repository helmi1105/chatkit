import itertools
import unittest
from copy import deepcopy
from app.orchestrator import Session, KcGraph, KC_GRAPH_PATH
from app.decision_policy import build_learner_state, decide_next_action, evidence_status
from app.diagnostic import curriculum_groups


class PolicyTests(unittest.TestCase):
    def test_replay_and_expert_alternatives(self):
        from app.eval_decisions import evaluate
        state = {'current_kc_id': 'a', 'validation_challenge': False}
        outcome = {'score': .6, 'threshold': .7, 'remediation_below': .5}
        event = dict(event='pedagogical_decision', learner_state=state, curriculum=[], practice_result=outcome,
                     decision=decide_next_action(state, [], practice_result=outcome))
        self.assertIsNone(evaluate([event])['expert_agreement'])
        event['acceptable_decisions'] = [{'action': 'review', 'target_kc': 'a'}]
        report = evaluate([event])
        self.assertEqual(report['exact_replay_matches'], 1)
        self.assertEqual(report['expert_agreement'], 1)

    def test_legacy_priority_and_no_mutation(self):
        graph = KcGraph.load(KC_GRAPH_PATH)
        ids = graph.kc_ids()[:2]
        cases = [(), (True,), (False,), (True, True), (False, False), (False, True), (False, True, True)]
        for values in itertools.product(cases, repeat=2):
            sess = Session(diagnostic_kc_ids=ids, diagnostic_question_limit=96)
            for kid, answers in zip(ids, values):
                sess.diagnostic_evidence[kid] = [dict(correct=v, evidence_type='recognition' if i == 0 else 'application') for i, v in enumerate(answers)]
            state = build_learner_state(sess, graph)
            before = deepcopy(state)
            decision = decide_next_action(state, curriculum_groups(graph))
            weak = [k for k in ids if evidence_status(sess.diagnostic_evidence[k]) == 'needs_practice' or
                    (evidence_status(sess.diagnostic_evidence[k]) == 'uncertain' and len(sess.diagnostic_evidence[k]) >= 3)]
            supported = [k for k in ids if evidence_status(sess.diagnostic_evidence[k]) == 'supported']
            self.assertEqual(decision['action'], 'teach' if weak else ('validate' if supported else 'probe'))
            if weak or supported:
                self.assertEqual(decision['target_kc'], (weak or supported)[0])
            self.assertEqual(state, before)
            self.assertIsNone(decision['confidence'])

    def test_practice_boundaries(self):
        for challenge, score in itertools.product((False, True), (0, .49, .5, .69, .7, 1)):
            result = decide_next_action({'current_kc_id': 'a', 'validation_challenge': challenge}, [],
                practice_result={'score': score, 'threshold': .7, 'remediation_below': .5})
            self.assertEqual(result['action'], 'advance' if score >= .7 else ('remediate' if challenge or score < .5 else 'review'))

    def test_old_session_has_unknown_history(self):
        sess = Session.from_dict({'user_id': 'old'})
        graph = KcGraph.load(KC_GRAPH_PATH)
        state = build_learner_state(sess, graph)
        kc = state['kcs'][graph.kc_ids()[0]]
        self.assertEqual(kc['practice_evidence'], [])
        self.assertIsNone(kc['practice_score_estimate'])
        kc['diagnostic_evidence'].append({'correct': True})
        self.assertEqual(sess.diagnostic_evidence, {})
