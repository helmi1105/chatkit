import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.visual_retrieval import VisualElement, VisualInventory, retrieve_visual_evidence
from app.orchestrator import Orchestrator, Session


def inventory():
    return VisualInventory(elements=[
        VisualElement(image_number=1, position=position, shape=shape, color=color,
                      line_style='', lettering='', details='', uncertainty='')
        for position, shape, color in [('gauche', 'triangle', 'rouge'), ('droite', 'carré', 'bleu')]
    ], relationships=['Le triangle est à gauche du carré.'], limitations='')


def course():
    return SimpleNamespace(structured={
        12: {'symbols': [{'shape': 'triangle', 'color': 'rouge', 'meaning': 'Alpha'}]},
        23: {'symbols': [{'shape': 'carré', 'color': 'bleu', 'meaning': 'Beta'}]},
    }, pages={12: 'triangle rouge Alpha', 23: 'carré bleu Beta', 3: 'Introduction générale'})


@patch.dict('os.environ', {'VISUAL_RETRIEVAL_MODE': 'bm25'})
class RetrievalTests(unittest.TestCase):
    def test_each_symbol_retrieves_its_own_evidence_without_fixed_pages(self):
        observed = inventory()
        observed.relationships = []
        text, pages = retrieve_visual_evidence(course(), observed, 'Que signifie ceci ?')
        self.assertEqual(pages, [12, 23])
        left, right = text.split('Élément 2')
        self.assertIn('[Page 12]', left)
        self.assertNotIn('[Page 23]', left)
        self.assertIn('[Page 23]', right)

    def test_unknown_symbol_has_no_invented_reference(self):
        unknown = inventory()
        unknown.relationships = []
        unknown.elements = [unknown.elements[0].model_copy(update={'shape': 'hexagone', 'color': 'violet'})]
        text, pages = retrieve_visual_evidence(course(), unknown, '')
        self.assertEqual(pages, [])
        self.assertIn('signification non établie', text)


@patch.dict('os.environ', {'VISUAL_RETRIEVAL_MODE': 'bm25'})
class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_inspection_precedes_grounded_answer_and_followup_reuses_image(self):
        tutor = SimpleNamespace(doc=course(), _charge_budget=Mock(), _kc=lambda _: None,
                                _record_event=Mock(), _text=lambda text: {'text': text},
                                _actions=lambda actions: {'actions': actions}, _next_actions=lambda _: [])
        session = Session()
        with patch('app.orchestrator.run_structured', new_callable=AsyncMock, return_value=inventory()) as observe, \
             patch('app.orchestrator.run_text', new_callable=AsyncMock, return_value='Réponse sourcée') as answer:
            await Orchestrator._answer_visual_question(tutor, session, 'Explique', ['data:image/png;base64,test'], None)
            prompt = answer.call_args.args[2][0]['content']
            self.assertIn('[Page 12]', prompt[0]['text'])
            self.assertIn('[Page 23]', prompt[0]['text'])
            self.assertEqual(prompt[1]['image_url'], 'data:image/png;base64,test')
            self.assertTrue(observe.call_args.kwargs['vision'])
            self.assertEqual(tutor._record_event.call_args.args[2]['source_pages'], [12, 23])
            await Orchestrator._answer_visual_question(tutor, session, 'Et la couleur ?', [], None)
            self.assertEqual(observe.call_args.args[2][0]['content'][1]['image_url'], 'data:image/png;base64,test')
            self.assertTrue(tutor._record_event.call_args.args[2]['image_context_reused'])


if __name__ == '__main__':
    unittest.main()
