import unittest
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace
from app.orchestrator import Orchestrator, Session, KcGraph, KC_GRAPH_PATH
from app.flashcards import FlashcardPack


class FlashcardTests(unittest.IsolatedAsyncioTestCase):
    async def test_rating_queue_persistence_and_stale_clicks(self):
        from dataclasses import asdict
        orch = self.setup_orch()
        sess = Session(current_kc_id='g_k2', diagnostic_stage='learning', mastery={'g_k2': .3})
        sess.flashcard_deck = dict(id='deck', kc_id='g_k2', index=0, revealed=False, review_indices=[],
            cards=[dict(question=f'Q{i}', answer='A', page=3) for i in range(3)])
        async def action(name):
            return await orch._flashcards(sess, f"fiche {name} deck {sess.flashcard_deck['index']}", None)
        await action('encore')
        self.assertEqual(sess.flashcard_deck['index'], 0)
        await action('reveler')
        await action('encore')
        self.assertEqual(sess.flashcard_deck['queue'], [0, 1, 0, 2])
        await orch._flashcards(sess, 'fiche encore deck 0', None)
        self.assertEqual(len(sess.flashcard_deck['ratings']), 1)
        await action('reveler')
        await action('difficile')
        self.assertEqual(sess.flashcard_deck['queue'], [0, 1, 0, 2, 1])
        sess = Session.from_dict(asdict(sess))
        while sess.flashcard_deck['index'] < len(sess.flashcard_deck['queue']):
            await action('reveler')
            blocks = await action('sais')
        self.assertIn('Revision terminee', str(blocks))
        self.assertEqual(sess.flashcard_deck['review_indices'], [])
        self.assertEqual(sess.mastery, {'g_k2': .3})
        self.assertEqual(sess.validated_kc_ids, [])
        ratings = [c.args[2] for c in orch._record_event.call_args_list if c.args[2]['event'] == 'flashcard_rated']
        self.assertEqual([e['rating'] for e in ratings[:2]], ['again', 'hard'])
        self.assertTrue(all(e['evidence_role'] == 'self_report_only' for e in ratings))

    async def test_svg_visibility_and_reveal_without_additional_generation(self):
        svg = '<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="20"/></svg>'
        for side in ('front', 'back'):
            orch = self.setup_orch()
            sess = Session(current_kc_id='g_k2', diagnostic_stage='learning')
            page = orch.graph.kc_pages(orch.graph.nodes['g_k2'])[0]
            pack = FlashcardPack(cards=[dict(question=f'Q{i}', answer='Secret', page=page,
                                            svg=svg, svg_side=side) for i in range(3)])
            with patch('app.flashcards.run_structured', new=AsyncMock(return_value=pack)) as model:
                blocks = await orch._flashcards(sess, 'flashcards', None)
                self.assertEqual('data:image/png;base64,' in str(blocks), side == 'front')
                self.assertNotIn('report.open', str(blocks))
                self.assertNotIn('Secret', str(blocks))
                self.assertNotIn('/course.pdf', str(blocks))
                from dataclasses import asdict
                sess = Session.from_dict(asdict(sess))
                blocks = await orch._flashcards(sess, f"fiche reveler {sess.flashcard_deck['id']} 0", None)
                self.assertIn('data:image/png;base64,', str(blocks))
                self.assertNotIn('Voir le schema', str(blocks))
                self.assertIn('Secret', str(blocks))
                model.assert_awaited_once()
                self.assertEqual(sess.mastery, {})

    async def test_invalid_front_svg_is_dropped_and_back_keeps_text(self):
        orch = self.setup_orch()
        sess = Session(current_kc_id='g_k2', diagnostic_stage='learning')
        page = orch.graph.kc_pages(orch.graph.nodes['g_k2'])[0]
        cards = [dict(question=f'Q{i}', answer='A', page=page) for i in range(3)]
        cards += [dict(question='Identify missing image', answer='A', page=page, svg='<script/>', svg_side='front'),
                  dict(question='Recall symbol', answer='A', page=page, svg='<script/>', svg_side='back')]
        with patch('app.flashcards.run_structured', new=AsyncMock(return_value=FlashcardPack(cards=cards))):
            await orch._flashcards(sess, 'flashcards', None)
        self.assertEqual(len(sess.flashcard_deck['cards']), 4)
        self.assertTrue(all(not c['svg'] for c in sess.flashcard_deck['cards']))

    def setup_orch(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch.doc = SimpleNamespace(pages_text=lambda *a, **kw: 'source', pdf_url=lambda p: f'/course.pdf#page={p}')
        orch._record_event = Mock()
        orch._actions = lambda buttons, **kw: {'buttons': buttons}
        return orch

    async def test_reveal_review_next_and_persistence_without_mastery_change(self):
        orch = self.setup_orch()
        sess = Session(current_kc_id='g_k2', diagnostic_stage='learning', mastery={'g_k2': .2})
        page = orch.graph.kc_pages(orch.graph.nodes['g_k2'])[0]
        pack = FlashcardPack(cards=[dict(question=f'Question {i}', answer=f'Secret {i}', page=page) for i in range(3)])
        with patch('app.flashcards.run_structured', new=AsyncMock(return_value=pack)) as model:
            blocks = await orch._dispatch(sess, None, 'flashcards', None)
            self.assertNotIn('Secret', str(blocks))
            suffix = f"{sess.flashcard_deck['id']} 0"
            blocks = await orch._flashcards(sess, 'fiche reveler ' + suffix, None)
            self.assertIn('Secret 0', str(blocks))
            self.assertIn('/course.pdf', str(blocks))
            await orch._flashcards(sess, 'fiche revoir ' + suffix, None)
            await orch._flashcards(sess, 'fiche revoir ' + suffix, None)
            self.assertEqual(sess.flashcard_deck['review_indices'], [0])
            from dataclasses import asdict
            restored = Session.from_dict(asdict(sess))
            await orch._flashcards(restored, 'fiche suivante ' + suffix, None)
            self.assertEqual(restored.flashcard_deck['index'], 1)
            await orch._flashcards(restored, 'fiche suivante ' + suffix, None)
            self.assertEqual(restored.flashcard_deck['index'], 1)
            model.assert_awaited_once()
        self.assertEqual(sess.mastery, {'g_k2': .2})
        self.assertEqual(sess.validated_kc_ids, [])
        self.assertEqual(sess.diagnostic_evidence, {})

    async def test_pending_quiz_blocks_generation(self):
        orch = self.setup_orch()
        sess = Session(current_kc_id='g_k2', phase='waiting_answers')
        with patch('app.flashcards.run_structured', new_callable=AsyncMock) as model:
            await orch._flashcards(sess, 'flashcards', None)
            model.assert_not_awaited()

    async def test_invalid_sources_do_not_replace_deck(self):
        orch = self.setup_orch()
        sess = Session(current_kc_id='g_k2', diagnostic_stage='learning')
        pack = FlashcardPack(cards=[dict(question='Q', answer='A', page=999)])
        with patch('app.flashcards.run_structured', new=AsyncMock(return_value=pack)):
            with self.assertRaises(RuntimeError):
                await orch._flashcards(sess, 'flashcards', None)
        self.assertEqual(sess.flashcard_deck, {})
