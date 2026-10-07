import unittest
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace
from app.orchestrator import Orchestrator, Session, KcGraph, KC_GRAPH_PATH
from app.podcast import PodcastScript, player_html


class PodcastTests(unittest.IsolatedAsyncioTestCase):
    def setup_orch(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.graph = KcGraph.load(KC_GRAPH_PATH)
        orch.doc = SimpleNamespace(pages_text=lambda *a, **k: 'source', pdf_url=lambda p: f'https://example.test/pdf#page={p}')
        orch._record_event = Mock()
        orch._actions = Mock(return_value={})
        return orch

    async def test_cache_and_no_assessment_changes(self):
        orch = self.setup_orch()
        sess = Session(current_kc_id='g_k2', diagnostic_stage='learning', mastery={'g_k2': .2})
        script = PodcastScript(transcript='Voici la règle.', pages=orch.graph.kc_pages(orch.graph.nodes['g_k2']))
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}), patch('app.podcast.run_structured', new=AsyncMock(return_value=script)) as generate, patch('app.podcast.synthesize', new=AsyncMock(return_value=b'mp3')) as speech:
            await orch._dispatch(sess, None, 'podcast', None)
            await orch._podcast(sess, None)
            generate.assert_awaited_once()
            speech.assert_awaited_once()
        self.assertEqual(sess.llm_budget_used, 2)
        self.assertEqual(sess.mastery, {'g_k2': .2})
        self.assertEqual(sess.validated_kc_ids, [])

    async def test_audio_failure_preserves_script_for_retry(self):
        orch = self.setup_orch()
        sess = Session(current_kc_id='g_k2', diagnostic_stage='learning')
        script = PodcastScript(transcript='Explication.', pages=orch.graph.kc_pages(orch.graph.nodes['g_k2']))
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}), patch('app.podcast.run_structured', new=AsyncMock(return_value=script)) as generate, patch('app.podcast.synthesize', new=AsyncMock(side_effect=[RuntimeError('failed'), b'mp3'])):
            first = await orch._podcast(sess, None)
            self.assertIn('Explication.', str(first))
            await orch._podcast(sess, None)
            generate.assert_awaited_once()

    async def test_pending_quiz_blocks(self):
        orch = self.setup_orch()
        with patch('app.podcast.run_structured', new_callable=AsyncMock) as generate:
            await orch._podcast(Session(current_kc_id='g_k2', phase='waiting_answers'), None)
            generate.assert_not_awaited()

    def test_html_escapes_model_text(self):
        rendered = player_html('<script>', '<img onerror=bad>', b'mp3', [])
        self.assertNotIn('<script>', rendered)
        self.assertNotIn('<img ', rendered)
        self.assertIn('<audio controls', rendered)
