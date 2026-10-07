import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from app import embeddings
from app.visual_retrieval import retrieve_visual_evidence
from test_visual_retrieval import course, inventory


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict('os.environ', {
            'OPENAI_EMBEDDING_API_KEY': 'test-only',
            'OPENAI_EMBEDDING_CACHE_DIR': self.temp.name,
            'VISUAL_RETRIEVAL_MODE': 'hybrid',
        })
        self.env.start()
        self.client = MagicMock()
        self.client.__enter__.return_value = self.client
        def create(**kwargs):
            self.assertEqual(kwargs['encoding_format'], 'float')
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1., 0.] if 'rouge' in text else [0., 1.])
                                         for i, text in reversed(list(enumerate(kwargs['input'])))])
        self.client.embeddings.create.side_effect = create
        self.client_patch = patch.object(embeddings, 'OpenAI', return_value=self.client)
        self.client_patch.start()
        embeddings._course_vectors.cache_clear()

    def tearDown(self):
        self.client_patch.stop()
        self.env.stop()
        embeddings._course_vectors.cache_clear()
        self.temp.cleanup()

    def test_cache_reuse_invalidation_and_batched_queries(self):
        chunks = [(12, 'triangle rouge'), (23, 'carré bleu')]
        scores = embeddings.semantic_scores(chunks, ['rouge', 'bleu'])
        self.assertEqual(scores.argmax(axis=1).tolist(), [0, 1])
        self.assertEqual(self.client.embeddings.create.call_count, 2)
        embeddings._course_vectors.cache_clear()
        embeddings.semantic_scores(chunks, ['rouge'])
        self.assertEqual(self.client.embeddings.create.call_count, 3)
        embeddings.semantic_scores(chunks + [(24, 'nouveau')], ['rouge'])
        self.assertEqual(self.client.embeddings.create.call_count, 5)

    def test_model_change_rebuilds_cache(self):
        chunks = [(1, 'rouge')]
        embeddings.prepare_course(chunks)
        with patch.dict('os.environ', {'OPENAI_EMBEDDING_MODEL': 'text-embedding-3-large'}):
            embeddings.prepare_course(chunks)
        self.assertEqual(self.client.embeddings.create.call_count, 2)

    def test_missing_key_is_explicit_even_when_answer_provider_is_mistral(self):
        from app.providers import friendly_llm_error
        with patch.dict('os.environ', {'OPENAI_API_KEY': '', 'OPENAI_EMBEDDING_API_KEY': '', 'MISTRAL_API_KEY': 'not-an-openai-key'}):
            with self.assertRaises(embeddings.EmbeddingError) as error:
                embeddings.semantic_scores([(1, 'rouge')], ['rouge'])
        self.assertIn('OPENAI_EMBEDDING_API_KEY', friendly_llm_error(error.exception))
        self.client.embeddings.create.assert_not_called()

    def test_failed_request_is_not_silently_downgraded(self):
        self.client.embeddings.create.side_effect = RuntimeError('private response')
        with self.assertRaises(embeddings.EmbeddingError) as error:
            embeddings.semantic_scores([(1, 'rouge')], ['rouge'])
        self.assertNotIn('private response', str(error.exception))

    def test_hybrid_path_batches_inventory_and_keeps_page_sources(self):
        with patch('app.visual_retrieval.semantic_scores', wraps=embeddings.semantic_scores) as search:
            text, pages = retrieve_visual_evidence(course(), inventory(), 'Explique')
        self.assertEqual(search.call_count, 1)
        self.assertEqual(len(search.call_args.args[1]), 2)
        self.assertIn('[Page 12]', text)
        self.assertIn(23, pages)
        self.assertEqual(embeddings.hybrid_ranking([0., 10., 0.], [0.9, 0.8, 0.7])[0], 1)


if __name__ == '__main__':
    unittest.main()
