import unittest

from app.objective_evidence import (
    expected_objectives,
    summarize_objective_evidence,
    update_objective_evidence,
)


class ObjectiveEvidenceTests(unittest.TestCase):
    def test_records_counts_and_ignores_uncredited_items(self):
        evidence = {}
        items = [
            {'target_id': 'E1', 'credited_kc_ids': ['g_k4'], 'correct': True},
            {'target_id': 'E2', 'credited_kc_ids': ['g_k4'], 'correct': False},
            {'target_id': 'E2', 'credited_kc_ids': ['g_k4'], 'correct': False},
            {'target_id': 'E1', 'credited_kc_ids': ['g_k1'], 'correct': False},
            {'target_id': 'M1', 'credited_kc_ids': ['g_k4'], 'correct': False},
        ]
        current = update_objective_evidence(evidence, 'g_k4', ['one', 'two'], items)
        self.assertEqual(current['E1']['correct'], 1)
        self.assertEqual(current['E1']['total'], 1)
        self.assertEqual(current['E2']['failures'], 2)
        summary = summarize_objective_evidence(['one', 'two'], current)
        self.assertTrue(summary['coverage_complete'])
        self.assertEqual(summary['repeated_failed_objectives'], ['E2'])
        update_objective_evidence(evidence, 'g_k4', ['one', 'two'], [
            {'target_id': 'E2', 'credited_kc_ids': ['g_k4'], 'correct': True},
        ])
        resolved = summarize_objective_evidence(['one', 'two'], evidence['g_k4'])
        self.assertEqual(resolved['failed_objectives'], [])
        self.assertEqual(resolved['repeated_failed_objectives'], [])
        self.assertEqual(resolved['historically_failed_objectives'], ['E2'])
        self.assertEqual(evidence['g_k4']['E2']['failures'], 2)

    def test_changed_positional_target_does_not_merge_history(self):
        evidence = {}
        update_objective_evidence(evidence, 'g_k4', ['old'], [
            {'target_id': 'E1', 'credited_kc_ids': ['g_k4'], 'correct': False},
        ])
        update_objective_evidence(evidence, 'g_k4', ['new'], [
            {'target_id': 'E1', 'credited_kc_ids': ['g_k4'], 'correct': True},
        ])
        self.assertEqual(evidence['g_k4']['E1']['total'], 1)
        self.assertEqual(evidence['g_k4']['E1']['correct'], 1)
        self.assertEqual(evidence['g_k4']['E1']['target_hash'],
                         expected_objectives(['new'])['E1']['target_hash'])


if __name__ == '__main__':
    unittest.main()
