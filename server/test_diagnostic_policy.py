import unittest
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace
import json
from app.diagnostic import evidence_status, select_followups
from app.diagnostic import DiagnosticRound, DiagnosticSingleQuestion
from app.orchestrator import Session, Orchestrator, KcGraph, KC_GRAPH_PATH, compact_keywords
from app.objective_evidence import expected_objectives, summarize_objective_evidence


def successful_objective_state(target='target'):
    definition = expected_objectives([target])['E1']
    return {'E1': {**definition, 'correct': 1, 'total': 1, 'failures': 0,
                   'consecutive_failures': 0}}


class DiagnosticPolicyTests(unittest.TestCase):
    def test_single_diagnostic_schema_requires_a_question(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            DiagnosticSingleQuestion.model_validate({'scenario': 'Situation'})

    def test_previous_kc_context_is_reduced_to_keywords(self):
        keywords = compact_keywords([
            "Définition du langage cartographique",
            "Identifier les formes et les symboles opérationnels standardisés",
        ])
        self.assertIn("langage", keywords)
        self.assertIn("cartographique", keywords)
        self.assertIn("symboles", keywords)
        self.assertLessEqual(len(keywords), 12)

    def test_practice_schema_requires_question_attribution(self):
        from pydantic import ValidationError
        from app.schemas import PracticePack
        with self.assertRaises(ValidationError):
            PracticePack(questions=[{
                'text': 'Question', 'choices': ['a', 'b', 'c', 'd'],
                'answer': 'A', 'target_id': 'E1',
            }])

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
    async def test_single_kc_fallback_stamps_server_owned_attribution(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch.doc = SimpleNamespace(pages_text=lambda *a, **kw: 'Source')
        orch._charge_budget = Mock()
        sess = Session()
        tasks = [
            {'kc_id': kid, 'title': orch.graph.nodes[kid].title,
             'evidence_type': 'recognition', 'selection_reason': 'screening',
             'previous_evidence': []}
            for kid in ('g_k1', 'g_k2')
        ]
        responses = []
        for index, task in enumerate(tasks, start=1):
            page = orch.graph.kc_pages(orch.graph.nodes[task['kc_id']])[0]
            responses.append(DiagnosticRound(
                scenario='Situation test',
                questions=[{
                    'text': f'Question distincte {index}',
                    'choices': ['Choix A', 'Choix B', 'Choix C', 'Choix D'],
                    'answer': 'A',
                    'explanation': 'Source',
                    # Missing page is safe to repair because this request has
                    # exactly one authoritative source page.
                    'page': None if index == 1 else page,
                    'kc_id': 'invented-id',
                    'role': 'review',
                    'assessed_kc_ids': ['invented-id'],
                }],
            ))

        with patch('app.diagnostic.run_structured', new=AsyncMock(side_effect=responses)):
            questions, scenario = await orch._regenerate_diagnostic_questions_individually(
                sess, 'Chapitre', 'screening', tasks, None
            )

        self.assertEqual([q['kc_id'] for q in questions], ['g_k1', 'g_k2'])
        self.assertEqual([q['assessed_kc_ids'] for q in questions], [['g_k1'], ['g_k2']])
        self.assertEqual([q['role'] for q in questions], ['current', 'current'])
        self.assertEqual([q['number'] for q in questions], [1, 2])
        self.assertEqual(
            questions[0]['page'],
            orch.graph.kc_pages(orch.graph.nodes['g_k1'])[0],
        )
        self.assertEqual(scenario, 'Situation test')
        self.assertEqual(orch._charge_budget.call_count, 2)

    async def test_repeated_grouped_question_gets_one_focused_retry(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch.doc = SimpleNamespace(pages_text=lambda *a, **kw: 'Source')
        orch._charge_budget = Mock()
        orch._record_event = Mock()
        kid = 'g_k1'
        page = orch.graph.kc_pages(orch.graph.nodes[kid])[0]
        old_text = 'Question déjà posée'
        sess = Session(
            diagnostic_mode='hierarchical',
            diagnostic_stage='screening',
            diagnostic_policy_version=2,
            diagnostic_kc_ids=[kid],
            diagnostic_groups=[{
                'module_id': 'ROOT', 'title': 'Chapitre', 'kc_ids': [kid],
            }],
            diagnostic_question_limit=5,
            diagnostic_evidence={kid: [{
                'question': old_text, 'correct': True,
                'evidence_type': 'recognition',
            }]},
        )
        grouped = DiagnosticRound(scenario='Situation initiale', questions=[{
            'text': old_text,
            'choices': ['A1', 'B1', 'C1', 'D1'],
            'answer': 'A',
            'explanation': 'Source',
            'page': page,
            'kc_id': kid,
            'role': 'current',
            'assessed_kc_ids': [kid],
        }])
        fresh = orch._server_owned_question({
            'text': 'Nouvelle question ciblée',
            'choices': ['A2', 'B2', 'C2', 'D2'],
            'answer': 'A',
            'explanation': 'Source',
            'page': page,
        }, kid, [page])
        decision = {
            'action': 'probe', 'group_index': 0,
            'completed_group_indices': [],
            'tasks': [{'kc_id': kid}],
        }
        orch._regenerate_diagnostic_questions_individually = AsyncMock(
            return_value=([fresh], 'Situation de remplacement')
        )

        with patch('app.diagnostic.decide_next_action', return_value=decision), \
             patch('app.diagnostic.run_structured', new=AsyncMock(return_value=grouped)):
            await orch._continue_hierarchical(sess, None)

        orch._regenerate_diagnostic_questions_individually.assert_awaited_once()
        retry_tasks = orch._regenerate_diagnostic_questions_individually.await_args.args[3]
        retry_kwargs = orch._regenerate_diagnostic_questions_individually.await_args.kwargs
        self.assertEqual([task['kc_id'] for task in retry_tasks], [kid])
        self.assertEqual(retry_kwargs['max_attempts'], 1)
        self.assertEqual(sess.phase, 'waiting_answers')
        self.assertEqual(sess.quiz_questions['1']['text'], 'Nouvelle question ciblée')

    async def test_single_source_page_is_repaired_without_retry(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch.doc = SimpleNamespace(pages_text=lambda *a, **kw: 'Source')
        orch._charge_budget = Mock()
        orch._record_event = Mock()
        kid = 'g_k1'
        page = orch.graph.kc_pages(orch.graph.nodes[kid])[0]
        sess = Session(
            diagnostic_mode='hierarchical',
            diagnostic_stage='screening',
            diagnostic_policy_version=2,
            diagnostic_kc_ids=[kid],
            diagnostic_groups=[{
                'module_id': 'ROOT', 'title': 'Chapitre', 'kc_ids': [kid],
            }],
            diagnostic_question_limit=5,
        )
        grouped = DiagnosticRound(scenario='Situation', questions=[{
            'text': 'Question nouvelle',
            'choices': ['A', 'B', 'C', 'D'],
            'answer': 'A',
            'explanation': 'Source',
            'page': page + 100,
            'kc_id': kid,
            'role': 'current',
            'assessed_kc_ids': [kid],
        }])
        decision = {
            'action': 'probe', 'group_index': 0,
            'completed_group_indices': [],
            'tasks': [{'kc_id': kid}],
        }
        orch._regenerate_diagnostic_questions_individually = AsyncMock()

        with patch('app.diagnostic.decide_next_action', return_value=decision), \
             patch('app.diagnostic.run_structured', new=AsyncMock(return_value=grouped)):
            await orch._continue_hierarchical(sess, None)

        orch._regenerate_diagnostic_questions_individually.assert_not_awaited()
        self.assertEqual(sess.quiz_questions['1']['page'], page)

    async def test_practice_preserves_invalid_attribution_for_evaluation(self):
        from app.schemas import PracticePack
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._essential_targets = AsyncMock(return_value=['shape'])
        orch._kc_context = Mock(return_value='Source')
        orch._record_event = Mock()
        sess = Session(
            current_kc_id='g_k2', current_micro_lesson='Lesson',
            pending_confirmation_objectives_by_kc={'g_k2': ['E1']},
        )

        def pack():
            return PracticePack(questions=[
                dict(text=f'Question {i}', choices=['a', 'b', 'c', 'd'], answer='A',
                     kc_id='g_k2', target_id='E1',
                     role=('current' if i == 0 else 'review'),
                     assessed_kc_ids=['g_k2'], explanation='Source', page=3)
                for i in range(2)
            ], coverage_plan=[])

        model = AsyncMock(return_value=pack())
        with patch('app.orchestrator.run_structured', new=model):
            await orch._start_practice(sess, None)

        self.assertEqual(model.await_count, 1)
        self.assertEqual(
            {q['role'] for q in sess.quiz_questions.values()},
            {'current', 'review'},
        )
        warning = next(
            call.args[2] for call in orch._record_event.call_args_list
            if call.args[2].get('event') == 'practice_attribution_warning'
        )
        self.assertEqual(warning['invalid_questions'][0]['role'], 'review')

    async def test_confirmation_retry_uses_only_new_focused_questions(self):
        from app.schemas import PracticePack
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._essential_targets = AsyncMock(return_value=['shape', 'color', 'state'])
        orch._kc_context = Mock(return_value='Source')
        orch._record_event = Mock()
        sess = Session(
            current_kc_id='g_k2', current_micro_lesson='Lesson',
            pending_confirmation_objectives_by_kc={'g_k2': ['E2']},
            recent_stems={'g_k2': ['Old E2 question']},
        )

        def question(text, target):
            return dict(text=text, choices=['a', 'b', 'c', 'd'], answer='A',
                        kc_id='g_k2', target_id=target, role='current',
                        assessed_kc_ids=['g_k2'], explanation='Source', page=3)

        pack = PracticePack(questions=[
            question('Unrelated E1', 'E1'),
            question('Old E2 question', 'E2'),
            question('New E2 application', 'E2'),
            question('New E2 transfer', 'E2'),
            question('Unrelated E3', 'E3'),
        ], coverage_plan=[])
        with patch('app.orchestrator.run_structured', new=AsyncMock(return_value=pack)) as model:
            await orch._start_practice(sess, None)

        self.assertEqual(model.await_count, 1)
        self.assertEqual(len(sess.quiz_questions), 2)
        self.assertEqual({q['target_id'] for q in sess.quiz_questions.values()}, {'E2'})
        self.assertEqual(
            {q['text'] for q in sess.quiz_questions.values()},
            {'New E2 application', 'New E2 transfer'},
        )
        self.assertEqual(sess.last_tutor_action, 'generate_targeted_confirmation')

    async def test_validation_accepts_partial_coverage_for_later_followup(self):
        from app.schemas import PracticePack
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._essential_targets = AsyncMock(return_value=['shape', 'color'])
        orch._kc_context = Mock(return_value='Source')
        orch._record_event = Mock()
        sess = Session(diagnostic_validation_challenge=True, current_kc_id='g_k2')
        def question(i, target):
            return dict(text=f'Question {i}', choices=['a','b','c','d'], answer='A', kc_id='g_k2',
                        target_id=target, role='current', assessed_kc_ids=['g_k2'], explanation='Source', page=3)
        initial = PracticePack(questions=[question(i, 'E1') for i in range(4)], coverage_plan=[])
        initial.questions[0].role = 'integration'
        initial.questions[0].assessed_kc_ids = ['g_k2', 'g_k1']
        repair = PracticePack(questions=[question(4, 'E2')], coverage_plan=[])
        with patch('app.orchestrator.run_structured', new=AsyncMock(side_effect=[initial, repair])) as model:
            await orch._start_practice(sess, None)
            self.assertEqual(model.await_count, 1)
        self.assertEqual({q['target_id'] for q in sess.quiz_questions.values()}, {'E1'})
        self.assertEqual(sess.llm_budget_used, 1)
        self.assertEqual(sess.phase, 'waiting_answers')

    async def test_practice_accepts_model_choice_without_integration(self):
        from app.schemas import PracticePack
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._essential_targets = AsyncMock(return_value=['shape'])
        orch._kc_context = Mock(return_value='Source')
        orch._record_event = Mock()
        sess = Session(current_kc_id='g_k2', current_micro_lesson='Lesson')

        def pack(with_integration):
            questions = [dict(
                text=f'Question {i}', choices=['a', 'b', 'c', 'd'], answer='A',
                kc_id='g_k2', target_id='E1', role='current',
                assessed_kc_ids=['g_k2'], explanation='Source', page=3,
            ) for i in range(4)]
            if with_integration:
                questions[0]['role'] = 'integration'
                questions[0]['assessed_kc_ids'] = ['g_k2', 'g_k1']
            return PracticePack(questions=questions, coverage_plan=[])

        model = AsyncMock(side_effect=[pack(False), pack(True)])
        with patch('app.orchestrator.run_structured', new=model):
            await orch._start_practice(sess, None)

        self.assertEqual(model.await_count, 1)
        integrations = [
            q for q in sess.quiz_questions.values() if q['role'] == 'integration'
        ]
        self.assertEqual(len(integrations), 0)

    async def test_current_question_is_not_forced_into_integration(self):
        from app.schemas import PracticePack
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._essential_targets = AsyncMock(return_value=['shape'])
        orch._kc_context = Mock(return_value='Source')
        orch._record_event = Mock()
        sess = Session(current_kc_id='g_k2', current_micro_lesson='Lesson')

        def current_pack(prefix):
            return PracticePack(questions=[dict(
                text=f'{prefix} {i}', choices=['a', 'b', 'c', 'd'], answer='A',
                kc_id='g_k2', target_id='E1', role='current',
                assessed_kc_ids=['g_k2'], explanation='Source', page=3,
            ) for i in range(4)], coverage_plan=[])

        model = AsyncMock(side_effect=[current_pack('base'), current_pack('attempt')])
        with patch('app.orchestrator.run_structured', new=model):
            await orch._start_practice(sess, None)

        attempts = [
            q for q in sess.quiz_questions.values() if q.get('integration_attempt')
        ]
        self.assertEqual(model.await_count, 1)
        self.assertEqual(len(attempts), 0)

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

    async def test_mid_score_challenge_uses_review_policy(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._record_event = Mock()
        orch._correction_block = Mock(return_value={})
        orch._source_card = Mock(return_value={})
        orch._feedback = AsyncMock(return_value=SimpleNamespace(summary='Feedback', hints=['Indice']))
        orch._build_lesson = AsyncMock(return_value='Targeted lesson')
        sess = Session(diagnostic_mode='hierarchical', diagnostic_stage='learning',
                       diagnostic_validation_challenge=True, current_kc_id='g_k2',
                       kc_essential_targets={'g_k2': ['target']})
        await orch._after_practice(sess, None, {'g_k2': (3, 5)}, [], [], {})
        orch._build_lesson.assert_not_awaited()
        self.assertFalse(sess.diagnostic_validation_challenge)
        self.assertEqual(sess.current_micro_lesson, '')
        self.assertNotIn('g_k2', sess.validated_kc_ids)
        self.assertEqual(sess.diagnostic_stage, 'learning')

    async def test_challenge_generation_skips_lesson_and_accepts_partial_coverage(self):
        from app.schemas import PracticePack
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._build_lesson = AsyncMock()
        orch._essential_targets = AsyncMock(return_value=['shape', 'color'])
        orch._kc_context = Mock(return_value='Source')
        orch._record_event = Mock()
        sess = Session(diagnostic_validation_challenge=True, diagnostic_stage='learning', current_kc_id='g_k2')
        pack = PracticePack(questions=[dict(text=f'Question {i}', choices=['a','b','c','d'],
            answer='A', kc_id='g_k2', target_id='E1', role='current', assessed_kc_ids=['g_k2'],
            explanation='Source', page=3) for i in range(4)], coverage_plan=[])
        pack.questions[0].role = 'integration'
        pack.questions[0].assessed_kc_ids = ['g_k2', 'g_k1']
        with patch('app.orchestrator.run_structured', new=AsyncMock(return_value=pack)):
            await orch._start_practice(sess, None)
        orch._build_lesson.assert_not_awaited()
        self.assertEqual(sess.phase, 'waiting_answers')
        self.assertEqual({q['target_id'] for q in sess.quiz_questions.values()}, {'E1'})
        self.assertEqual(sess.last_tutor_action, 'generate_medium_practice')

    async def test_practice_success_returns_to_diagnostic_without_advancing_graph(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._save_session = AsyncMock()
        orch._record_event = Mock()
        orch._correction_block = Mock(return_value={'type': 'text', 'text': 'correction'})
        sess = Session(diagnostic_mode='hierarchical', diagnostic_stage='learning',
                       scope='practice', current_kc_id='g_k2', diagnostic_group_index=1,
                       kc_essential_targets={'g_k2': ['target']},
                       objective_evidence_by_kc={'g_k2': successful_objective_state()})
        await orch._after_practice(sess, None, {'g_k2': (4, 4)}, [], [], {})
        self.assertEqual(sess.diagnostic_stage, 'resuming')
        self.assertEqual(sess.scope, 'diagnostic')
        self.assertEqual(sess.diagnostic_group_index, 1)
        self.assertIn('g_k2', sess.validated_kc_ids)
        self.assertEqual(sess.diagnostic_taught_kc_ids, ['g_k2'])
        self.assertFalse(sess.diagnostic_done)

    async def test_successful_targeted_confirmation_resolves_failure_and_advances(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch._save_session = AsyncMock()
        orch._record_event = Mock()
        orch._correction_block = Mock(return_value={'type': 'text', 'text': 'correction'})
        definitions = expected_objectives(['one', 'two'])
        evidence = {
            'E1': {**definitions['E1'], 'correct': 1, 'total': 1,
                   'failures': 0, 'consecutive_failures': 0},
            'E2': {**definitions['E2'], 'correct': 0, 'total': 1,
                   'failures': 1, 'consecutive_failures': 1},
        }
        sess = Session(
            diagnostic_mode='hierarchical', diagnostic_stage='learning',
            scope='practice', current_kc_id='g_k2', diagnostic_group_index=1,
            kc_essential_targets={'g_k2': ['one', 'two']},
            objective_evidence_by_kc={'g_k2': evidence},
            pending_confirmation_objectives_by_kc={'g_k2': ['E2']},
        )
        items = [
            {'number': 1, 'id': 'q1', 'target_id': 'E2', 'credited_kc_ids': ['g_k2'],
             'correct': True},
            {'number': 2, 'id': 'q2', 'target_id': 'E2', 'credited_kc_ids': ['g_k2'],
             'correct': True},
        ]

        await orch._after_practice(sess, None, {'g_k2': (2, 2)}, items, [], {})

        summary = summarize_objective_evidence(
            sess.kc_essential_targets['g_k2'], sess.objective_evidence_by_kc['g_k2'])
        self.assertEqual(summary['failed_objectives'], [])
        self.assertEqual(summary['historically_failed_objectives'], ['E2'])
        self.assertNotIn('g_k2', sess.pending_confirmation_objectives_by_kc)
        self.assertEqual(sess.diagnostic_taught_kc_ids, ['g_k2'])

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
                 'answer': 'A', 'kc_id': t['kc_id'], 'role': 'current', 'assessed_kc_ids': [t['kc_id']],
                 'page': orch.graph.kc_pages(orch.graph.nodes[t['kc_id']])[0], 'explanation': 'Source'} for t in tasks])
        with patch('app.diagnostic.run_structured', side_effect=generate):
            await orch._start_hierarchical(sess, None)
            for _ in range(100):
                if sess.diagnostic_done:
                    break
                if sess.diagnostic_stage == 'learning':
                    self.assertTrue(sess.diagnostic_validation_challenge)
                    kid = sess.current_kc_id
                    sess.kc_essential_targets[kid] = ['target']
                    sess.objective_evidence_by_kc[kid] = successful_objective_state()
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
