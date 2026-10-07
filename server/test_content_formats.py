import unittest
from unittest.mock import patch
from app.content import Doctrine
from app.visual_retrieval import build_visual_chunks


class ContentFormatTests(unittest.TestCase):
    def test_table_entries_reach_search_context_and_embedding_chunks(self):
        from app.qa_retrieval import retrieve_question_context
        payload = {'pages': [{'page': 14, 'title': 'Actions offensives',
            'tables': [{'title': 'Sigles', 'entries': [{'code': 'EVAC', 'meaning': 'évacuation'}]}]},
            {'page': 10, 'title': 'Sources de danger'},
            {'page': 21, 'title': 'Orientation et vent'}]}
        with patch('app.content._load_json', side_effect=[payload, None]):
            doc = Doctrine()
        self.assertIn('EVAC', doc.page_text(14))
        self.assertTrue(any(p == 14 and 'EVAC' in text and 'évacuation' in text
                            for p, text in build_visual_chunks(doc)))
        context, pages, _ = retrieve_question_context(doc, 'EVAC', [10])
        self.assertIn(14, pages)
        self.assertIn('évacuation', context)

    def test_wrapped_and_legacy_formats_produce_same_retrieval_content(self):
        pages = [{'page': 10, 'title': 'Sources de danger',
                  'sections': [{'text': 'Composante humaine verte'}],
                  'symbols': [], 'rules': ['Triangle vert vers le haut']}]
        docs = []
        for payload in (pages, {'document': {'title': 'Course'}, 'pages': pages}):
            with patch('app.content._load_json', side_effect=[payload, None]):
                doc = Doctrine()
            self.assertEqual(doc.source, 'vision')
            self.assertIn('humaine verte', doc.page_text(10))
            self.assertTrue(build_visual_chunks(doc))
            docs.append(doc)
        self.assertEqual(docs[0].revision, docs[1].revision)
        self.assertEqual(build_visual_chunks(docs[0]), build_visual_chunks(docs[1]))
