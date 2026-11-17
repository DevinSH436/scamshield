"""Run with:  python -m unittest test_scamshield.py"""
import unittest

from scamshield import NaiveBayes, evaluate, load_data, tokenize


class TestScamShield(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts, cls.labels = load_data()
        cls.model = NaiveBayes().fit(cls.texts, cls.labels)

    def test_tokenize_flags_risky_patterns(self):
        toks = tokenize("Pay $5 now at http://bit.ly/abc, code 123456")
        self.assertIn("__url__", toks)
        self.assertIn("__money__", toks)
        self.assertIn("__long_number__", toks)

    def test_probabilities_sum_to_one(self):
        p = self.model.predict_proba("hello there")
        self.assertAlmostEqual(sum(p.values()), 1.0)

    def test_detects_unseen_scam(self):
        msg = "URGENT: verify your account now or it will be suspended http://x.example"
        self.assertEqual(self.model.predict(msg), "scam")

    def test_detects_unseen_safe(self):
        self.assertEqual(self.model.predict("Are you coming to dinner tonight?"), "safe")

    def test_explanation_points_to_scam_words(self):
        top = dict(self.model.explain("Claim your prize now, click the link"))
        self.assertTrue(any(w > 0 for w in top.values()))

    def test_cross_validation_beats_baseline(self):
        self.assertGreater(evaluate(self.texts, self.labels)["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
