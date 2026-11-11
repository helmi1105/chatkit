import unittest
from app.content import doctrine
from app.practice_targets import scoped_targets


class PracticeTargetTests(unittest.TestCase):
    def test_state_targets_exclude_colour_rules_on_shared_page(self):
        rules = doctrine().rules_for_pages([4])
        targets = scoped_targets('g_k4', rules)
        self.assertEqual(len(targets), 2)
        self.assertTrue(all('pointill' in t and 'continu' in t for t in targets[:1]))
        self.assertNotIn('Noir :', ' '.join(targets))
        self.assertIsNone(scoped_targets('g_k3', rules))

if __name__ == '__main__':
    unittest.main()
