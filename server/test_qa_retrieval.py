import unittest
from types import SimpleNamespace
from app.qa_retrieval import retrieve_question_context
from app.content import Doctrine


class RetrievalTests(unittest.TestCase):
    def test_requested_range_is_prioritized(self):
        doc = Doctrine()
        text, pages, trace = retrieve_question_context(doc, 'Pages 13 à 16 : reconnaissance et actions offensives et défensives en SVG', [10])
        self.assertEqual(pages[:4], [13, 14, 15, 16])
        self.assertTrue({3, 4, 5}.issubset(pages))
        self.assertLessEqual(len(text), 24000)
        self.assertTrue(all(f'=== Page {p} ===' in text for p in pages))

    def test_long_early_page_does_not_hide_later_match(self):
        content = {1: 'irrelevant ' * 2000 + 'human green triangle ' * 20, 10: 'human green triangle'}
        doc = SimpleNamespace(pages=content, page_text=content.get, search_pages=lambda *a, **kw: [1, 10])
        text, pages, trace = retrieve_question_context(doc, 'human green triangle', max_chars=2500)
        self.assertEqual(set(pages), {1, 10})
        self.assertIn('green triangle', text)
        self.assertLessEqual(len(text), 2500)
        self.assertIn(1, trace['truncated_pages'])

    def test_no_match_does_not_invent_sources(self):
        doc = SimpleNamespace(pages={7: 'actions'}, page_text=lambda p: 'actions', search_pages=lambda *a, **kw: [])
        text, pages, trace = retrieve_question_context(doc, '1500 personnes')
        self.assertEqual(pages, [])
        self.assertEqual(text, '')
