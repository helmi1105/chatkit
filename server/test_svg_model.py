import unittest
from unittest.mock import patch
from app import providers as p


class ModelRoutingTests(unittest.TestCase):
    def test_free_answer_uses_dedicated_openai_model(self):
        token = p.current_provider.set(p.ProviderChoice(provider='mistral', api_key='test-mistral'))
        try:
            with patch.object(p, 'AsyncOpenAI') as client, patch.object(p, 'OpenAIResponsesModel') as model, patch.object(p, 'Agent'):
                p.make_agent('Learner-question', 'test')
                self.assertEqual(model.call_args.kwargs['model'], p.OPENAI_FREE_QUESTION_MODEL)
                self.assertNotEqual(client.call_args.kwargs['api_key'], 'test-mistral')
            with patch.object(p, 'build_model') as model, patch.object(p, 'Agent'):
                p.make_agent('Practice-QCM', 'test')
                model.assert_called_once()
        finally:
            p.current_provider.reset(token)
