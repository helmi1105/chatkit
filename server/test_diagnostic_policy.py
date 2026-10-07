import unittest
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace
import json
from app.diagnostic import evidence_status, select_followups
from app.diagnostic import DiagnosticRound
from app.orchestrator import Session, Orchestrator, KcGraph, KC_GRAPH_PATH


class DiagnosticPolicyTests(unittest.TestCase):
    def test_statuses(self):
        for values, expected in [([], 'unassessed'), ([True], 'uncertain'),
                                 ([True, True], 'supported'), ([False, False], 'needs_practice'),
                                 ([False, True, True], 'uncertain')]:
            self.assertEqual(evidence_status([{'correct': v, 'evidence_type': 'recognition' if i == 0 else 'application'} for i, v in enumerate(values)]), expected)

    def test_branch_selection_and_budget(self):
        evidence = {'a': [{'correct': True}, {'correct': True}],
                    'b': [{'correct': False}, {'correct': True}],
                    'c': [{'correct': False}, {'correct': False}]}
        evidence['a'][0]['evidence_type'] = 'recognition'
        evidence['a'][1]['evidence_type'] = 'application'
        self.assertEqual(select_followups(['a', 'b', 'c'], evidence, 5), ['b'])
        self.assertEqual(select_followups(['a', 'b', 'c'], evidence, 0), [])
        evidence['b'].append({'correct': True})
        self.assertEqual(select_followups(['a', 'b', 'c'], evidence, 5), [])

    def test_old_session_defaults_to_flat(self):
        self.assertEqual(Session.from_dict({'user_id': 'old'}).diagnostic_mode, 'flat')

    def test_untyped_success_is_not_supported(self):
        self.assertEqual(evidence_status([{'correct': True}, {'correct': True}]), 'uncertain')

    def test_benchmark_accepts_labels_without_fabricated_probabilities(self):
        from app.benchmark_runner import evaluate
        report = evaluate([{'event': 'diagnostic_submitted', 'mode': 'hierarchical',
                           'evidence_role': 'screening_only', 'statuses': {'g_k1': 'supported'},
                           'mastery_before': {}, 'mastery_after': {}}])
        self.assertEqual(report['metrics']['diagnostic_mastery_cap_rate'], 1)


class RoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_repairs_missing_objective_without_discarding_valid_questions(self):
        from app.schemas import PracticePack
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._essential_targets = AsyncMock(return_value=['shape', 'color'])
        orch._kc_context = Mock(return_value='Source')
        orch._record_event = Mock()
        sess = Session(diagnostic_validation_challenge=True, current_kc_id='g_k2')
        def question(i, target):
            return dict(text=f'Question {i}', choices=['a','b','c','d'], answer='A', kc_id='g_k2',
                        target_id=target, explanation='Source', page=3)
        initial = PracticePack(questions=[question(i, 'E1') for i in range(4)], coverage_plan=[])
        repair = PracticePack(questions=[question(4, 'E2')], coverage_plan=[])
        with patch('app.orchestrator.run_structured', new=AsyncMock(side_effect=[initial, repair])) as model:
            await orch._start_practice(sess, None)
            self.assertEqual(model.await_count, 2)
        self.assertEqual({q['target_id'] for q in sess.quiz_questions.values()}, {'E1', 'E2'})
        self.assertEqual(sess.llm_budget_used, 2)
        self.assertEqual(sess.phase, 'waiting_answers')

    async def test_all_kcs_require_all_chapter_checks(self):
        from app.orchestrator import MODULE_THRESHOLD
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._record_event = Mock()
        orch._correction_block = Mock(return_value={})
        orch._start_module_quiz = AsyncMock(return_value=[{'type': 'text', 'text': 'checkpoint'}])
        sess = Session(validated_kc_ids=orch.graph.kc_ids())
        modules = orch.completion_status(sess)['missing_module_ids']
        self.assertTrue(modules)
        await orch._completion_checkpoint(sess, None)
        for mid in modules:
            self.assertEqual(sess.pending_module_id, mid)
            self.assertFalse(orch.completion_status(sess)['complete'])
            await orch._after_module(sess, None, MODULE_THRESHOLD, [], [], {})
        self.assertTrue(orch.completion_status(sess)['complete'])
        self.assertEqual(orch._start_module_quiz.await_count, len(modules))

    async def test_failed_challenge_gives_lesson_even_above_remediation_threshold(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._record_event = Mock()
        orch._correction_block = Mock(return_value={})
        orch._source_card = Mock(return_value={})
        orch._feedback = AsyncMock(return_value=SimpleNamespace(summary='Feedback', hints=['Indice']))
        orch._build_lesson = AsyncMock(return_value='Targeted lesson')
        sess = Session(diagnostic_mode='hierarchical', diagnostic_stage='learning',
                       diagnostic_validation_challenge=True, current_kc_id='g_k2')
        await orch._after_practice(sess, None, {'g_k2': (3, 5)}, [], [], {})
        orch._build_lesson.assert_awaited_once()
        self.assertFalse(sess.diagnostic_validation_challenge)
        self.assertEqual(sess.current_micro_lesson, 'Targeted lesson')
        self.assertNotIn('g_k2', sess.validated_kc_ids)
        self.assertEqual(sess.diagnostic_stage, 'learning')

    async def test_challenge_generation_skips_lesson_and_rejects_missing_coverage(self):
        from app.schemas import PracticePack
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._build_lesson = AsyncMock()
        orch._essential_targets = AsyncMock(return_value=['shape', 'color'])
        orch._kc_context = Mock(return_value='Source')
        orch._record_event = Mock()
        sess = Session(diagnostic_validation_challenge=True, diagnostic_stage='learning', current_kc_id='g_k2')
        pack = PracticePack(questions=[dict(text=f'Question {i}', choices=['a','b','c','d'],
            answer='A', kc_id='g_k2', target_id='E1', explanation='Source', page=3) for i in range(4)], coverage_plan=[])
        with patch('app.orchestrator.run_structured', new=AsyncMock(return_value=pack)):
            with self.assertRaisesRegex(RuntimeError, 'objectifs'):
                await orch._start_practice(sess, None)
        orch._build_lesson.assert_not_awaited()
        self.assertEqual(sess.phase, 'idle')
        pack.questions[-1].target_id = 'E2'
        with patch('app.orchestrator.run_structured', new=AsyncMock(return_value=pack)):
            await orch._start_practice(sess, None)
        orch._build_lesson.assert_not_awaited()
        self.assertEqual(sess.phase, 'waiting_answers')
        self.assertEqual(sess.last_tutor_action, 'generate_medium_practice')

    async def test_practice_success_returns_to_diagnostic_without_advancing_graph(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._save_session = AsyncMock()
        orch._record_event = Mock()
        orch._correction_block = Mock(return_value={'type': 'text', 'text': 'correction'})
        sess = Session(diagnostic_mode='hierarchical', diagnostic_stage='learning',
                       scope='practice', current_kc_id='g_k2', diagnostic_group_index=1)
        await orch._after_practice(sess, None, {'g_k2': (4, 4)}, [], [], {})
        self.assertEqual(sess.diagnostic_stage, 'resuming')
        self.assertEqual(sess.scope, 'diagnostic')
        self.assertEqual(sess.diagnostic_group_index, 1)
        self.assertIn('g_k2', sess.validated_kc_ids)
        self.assertEqual(sess.diagnostic_taught_kc_ids, ['g_k2'])
        self.assertFalse(sess.diagnostic_done)

    async def test_repeated_errors_pause_before_more_questions(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._save_session = AsyncMock()
        orch._record_event = Mock()
        orch._show_lesson = AsyncMock(return_value=[])
        sess = Session(diagnostic_mode='hierarchical', diagnostic_kc_ids=orch.graph.kc_ids())
        sess.diagnostic_evidence = {'g_k2': [dict(correct=False, number=i+1, question='q',
            learner_choice='wrong', correct_choice='right', correct_letter='A', explanation='source') for i in range(2)]}
        with patch('app.diagnostic.run_structured', new_callable=AsyncMock) as model:
            await orch._continue_hierarchical(sess, None)
            model.assert_not_awaited()
        self.assertEqual(sess.diagnostic_stage, 'learning')
        self.assertEqual(sess.scope, 'practice')
        self.assertFalse(sess.diagnostic_done)
        self.assertEqual(sess.current_kc_id, 'g_k2')
        orch._show_lesson.assert_awaited_once()
        self.assertEqual(sess.diagnostic_status_by_kc['g_k3'], 'unassessed')

    async def test_whole_curriculum_preserves_mastery_and_skips_lessons(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch.doc = SimpleNamespace(pages_text=lambda *a, **kw: 'Source p. 3, 4, 5')
        orch._save_session = AsyncMock()
        orch._record_event = Mock()
        sess = Session(mastery={'g_k2': .8}, validated_kc_ids=['g_k2'])
        orch._start_practice = AsyncMock(return_value=[])
        orch._build_lesson = AsyncMock()
        orch._correction_block = Mock(return_value={'type': 'text', 'text': 'correction'})
        orch._start_module_quiz = AsyncMock(return_value=[{'type': 'text', 'text': 'checkpoint'}])
        counter = 0
        async def generate(name, instructions, prompt, schema, ctx):
            nonlocal counter
            counter += 1
            tasks = json.loads(prompt.split('Tâches : ')[1].split('\n')[0])
            return DiagnosticRound(scenario='Situation test', questions=[
                {'text': f'Question {counter} {t["kc_id"]}', 'choices': ['A1','B1','C1','D1'],
                 'answer': 'A', 'kc_id': t['kc_id'], 'page': orch.graph.kc_pages(orch.graph.nodes[t['kc_id']])[0], 'explanation': 'Source'} for t in tasks])
        with patch('app.diagnostic.run_structured', side_effect=generate):
            await orch._start_hierarchical(sess, None)
            for _ in range(100):
                if sess.diagnostic_done:
                    break
                if sess.diagnostic_stage == 'learning':
                    self.assertTrue(sess.diagnostic_validation_challenge)
                    kid = sess.current_kc_id
                    await orch._after_practice(sess, None, {kid: (4, 4)}, [], [], {})
                    await orch._start_hierarchical(sess, None)
                    continue
                await orch._process_answers(sess, {int(n): q['answer'] for n,q in sess.quiz_questions.items()}, None)
        self.assertTrue(sess.diagnostic_done)
        self.assertGreater(counter, 2)
        self.assertEqual(sess.mastery, {'g_k2': .8})
        self.assertEqual(set(sess.validated_kc_ids), set(orch.graph.kc_ids()))
        orch._build_lesson.assert_not_awaited()
        self.assertEqual(orch._start_practice.await_count, len(orch.graph.kc_ids()) - 1)
        self.assertEqual(sess.diagnostic_status_by_kc['g_k3'], 'supported')
        self.assertEqual(sess.diagnostic_status_by_kc['g_k12'], 'supported')
        self.assertEqual(sum(map(len, sess.diagnostic_evidence.values())), 2 * len(orch.graph.kc_ids()))
        self.assertIsNone(sess.current_kc_id)
        self.assertEqual(sess.current_micro_lesson, '')
        self.assertEqual(sess.diagnostic_learning_queue, [])
        orch._start_module_quiz.assert_awaited_once()
        self.assertFalse(orch.completion_status(sess)['complete'])

    async def test_budget_preserves_unassessed_and_learning_queue(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._record_event = Mock()
        sess = Session(diagnostic_mode='hierarchical', diagnostic_policy_version=2,
                       diagnostic_kc_ids=orch.graph.kc_ids(), diagnostic_question_limit=2)
        sess.diagnostic_evidence = {'g_k1': [dict(correct=True, evidence_type=kind, question='q',
             number=1, round=i+1, learner_letter='A', learner_choice='yes', correct_letter='A',
             correct_choice='yes', explanation='source') for i,kind in enumerate(['recognition','application'])]}
        await orch._finish_hierarchical(sess, None)
        self.assertEqual(sess.current_kc_id, 'g_k2')
        self.assertEqual(sess.diagnostic_status_by_kc['g_k2'], 'unassessed')
        self.assertNotIn('g_k1', sess.diagnostic_learning_queue)
        self.assertEqual(orch._learning_next(sess, 'g_k2'), 'g_k3')

    async def test_incomplete_answers_do_not_count_as_errors(self):
        orch = Orchestrator.__new__(Orchestrator)
        sess = Session(quiz_questions={'1': {'answer': 'A'}})
        await orch._submit_hierarchical(sess, {}, None)
        self.assertEqual(sess.diagnostic_evidence, {})

    async def test_failed_followup_can_resume_without_double_scoring(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch._record_event = Mock()
        orch._save_session = AsyncMock()
        sess = Session(diagnostic_mode='hierarchical', diagnostic_stage='screening',
                       scope='diagnostic', phase='waiting_answers', quiz_id='original',
                       quiz_questions={'1': {'number': 1, 'text': 'q', 'kc_id': 'g_k2',
                           'answer': 'A', 'choices': ['yes','no','maybe','other'],
                           'evidence_type': 'recognition', 'selection_reason': 'chapter_screening'}})
        orch._continue_hierarchical = AsyncMock(side_effect=RuntimeError('generation failed'))
        with self.assertRaises(RuntimeError):
            await orch._submit_hierarchical(sess, {1: 'B'}, None)
        self.assertEqual(sess.phase, 'idle')
        self.assertEqual(len(sess.diagnostic_evidence['g_k2']), 1)
        orch._continue_hierarchical = AsyncMock(return_value=[])
        await orch._start_hierarchical(sess, None)
        orch._continue_hierarchical.assert_awaited_once()
        self.assertEqual(len(sess.diagnostic_evidence['g_k2']), 1)


if __name__ == '__main__':
    unittest.main()
