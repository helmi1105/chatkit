import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from app.content import Doctrine
from app.orchestrator import Orchestrator, Session


class DoctrineAuditTests(unittest.TestCase):
    def test_pages_and_human_symbols(self):
        data = json.loads((Path(__file__).parent / 'app/doctrine_pages.json').read_text(encoding='utf-8'))
        self.assertEqual([p['page'] for p in data], list(range(1, 25)))
        human = [s for s in data[9]['symbols'] if 'humaine' in s['label']]
        self.assertEqual(len(human), 2)
        self.assertTrue(all(s['color'] == 'vert' for s in human))
        self.assertIn('haut', human[0]['shape'])
        self.assertIn('bas', human[1]['shape'])
        self.assertEqual(data[0]['symbols'], [])
        self.assertEqual(data[23]['symbols'], [])
        doc = Doctrine()
        self.assertIn('B biologique', doc.page_text(10))
        self.assertNotIn('Texte OCR de la page', doc.page_text(10))
        self.assertEqual(doc.revision, Doctrine().revision)

    def test_material_refresh_preserves_assessment_history(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.doc = SimpleNamespace(revision='new')
        sess = Session(current_micro_lesson='old', mastery={'k': .8}, validated_kc_ids=['k'],
                       quiz_questions={'1': {'answer': 'A'}}, phase='waiting_answers',
                       flashcard_deck={'id': 'old'}, kc_essential_targets={'k': ['old']})
        orch._refresh_doctrine_material(sess)
        self.assertEqual(sess.current_micro_lesson, '')
        self.assertEqual(sess.flashcard_deck, {})
        self.assertEqual(sess.kc_essential_targets, {})
        self.assertEqual(sess.mastery, {'k': .8})
        self.assertEqual(sess.validated_kc_ids, ['k'])
        self.assertEqual(sess.phase, 'waiting_answers')
        sess.current_micro_lesson = 'new material'
        orch._refresh_doctrine_material(sess)
        self.assertEqual(sess.current_micro_lesson, 'new material')
