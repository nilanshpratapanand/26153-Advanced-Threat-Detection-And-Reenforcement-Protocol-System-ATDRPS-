"""The out-of-distribution guard.

This exists because of a real result, and the tests encode it. A 145-minute
capture of ordinary hostel wifi -- 52,010 packets, no attack in it -- was
scored CONFIRMED Exfiltration at 0.96 by a model trained only on generated
traffic. 54 of the 103 state dimensions in that capture sat more than three
standard deviations outside the training distribution, and several sat
infinitely outside it: the generator emits exactly zero for TTL inconsistency,
RST injection ratio and inbound flow share, so any real value at all is
somewhere the model has never been.

The guard does not decide whether traffic is malicious. It decides whether
this model is entitled to an opinion about it.
"""

from __future__ import annotations

import unittest

import numpy as np

from atdrps.engine.novelty import (
    IN_DISTRIBUTION, MARGINAL, OUTSIDE, assess_novelty,
)
from atdrps.models.base import Standardiser


def _standardiser(train: np.ndarray) -> Standardiser:
    s = Standardiser()
    s.fit(train)
    return s


def _names(n: int) -> list[str]:
    return [f"feature_{i}" for i in range(n)]


class TestNovelty(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.n_features = 20
        self.train = rng.normal(size=(4000, self.n_features))
        self.std = _standardiser(self.train)

    def test_training_like_traffic_is_in_distribution(self):
        rng = np.random.default_rng(11)
        fresh = rng.normal(size=(300, self.n_features))
        report = assess_novelty(fresh, self.std, _names(self.n_features))
        self.assertEqual(report.verdict, IN_DISTRIBUTION)
        self.assertEqual(report.message(), "")
        self.assertTrue(report.trustworthy)

    def test_shifted_traffic_is_flagged_as_outside(self):
        rng = np.random.default_rng(13)
        shifted = rng.normal(size=(300, self.n_features))
        shifted[:, :8] += 40.0            # eight features far outside the envelope
        report = assess_novelty(shifted, self.std, _names(self.n_features))
        self.assertEqual(report.verdict, OUTSIDE)
        self.assertFalse(report.trustworthy)
        self.assertGreater(report.median_fraction, 0.10)

    def test_the_message_names_the_features_and_their_trained_range(self):
        rng = np.random.default_rng(17)
        shifted = rng.normal(size=(200, self.n_features))
        shifted[:, 3] += 60.0
        shifted[:, 4] += 60.0
        shifted[:, 5] += 60.0
        report = assess_novelty(shifted, self.std, _names(self.n_features))
        message = report.message()
        self.assertIn("feature_3", message)
        self.assertIn("trained range", message)
        # it must say what to do about it, not merely that something is wrong
        self.assertIn("Retrain", message)

    def test_a_feature_the_training_data_never_varied_is_caught(self):
        """The generator emits a constant 0 for several packet-level features.
        Any real traffic with a non-zero value there is outside by definition,
        and a constant column must not divide by zero on the way to saying so.
        """
        train = self.train.copy()
        train[:, 0] = 0.0
        std = _standardiser(train)
        real = np.random.default_rng(19).normal(size=(200, self.n_features))
        real[:, 0] = 0.21                     # what hostel wifi actually showed
        report = assess_novelty(real, std, _names(self.n_features))
        self.assertTrue(np.isfinite(report.median_fraction))
        flagged = [f["feature"] for f in report.worst_features]
        self.assertIn("feature_0", flagged)

    def test_a_model_with_no_envelope_declines_to_guess(self):
        class Bare:
            mean = scale = lo = hi = None

        report = assess_novelty(np.zeros((5, 4)), Bare(), _names(4))
        self.assertEqual(report.verdict, IN_DISTRIBUTION)
        self.assertEqual(report.message(), "")

    def test_report_survives_json(self):
        import json

        report = assess_novelty(self.train[:50], self.std, _names(self.n_features))
        json.loads(json.dumps(report.as_dict()))

    def test_a_single_odd_window_does_not_condemn_the_capture(self):
        """The capture-level call uses the median window. One burst of unusual
        traffic is a burst; a median of 0.20 is a different network."""
        rng = np.random.default_rng(23)
        mostly_normal = rng.normal(size=(200, self.n_features))
        mostly_normal[0, :] += 50.0
        report = assess_novelty(mostly_normal, self.std, _names(self.n_features))
        self.assertEqual(report.verdict, IN_DISTRIBUTION)


class TestVerdictIsWithheld(unittest.TestCase):
    """The headline must change, not just a note somewhere below it."""

    def _result(self, verdict):
        from atdrps.engine.inference import AnalysisResult
        from atdrps.engine.novelty import NoveltyReport

        r = AnalysisResult(source="x", source_kind="pcap", window_size_s=30.0)
        r.timeline = [{
            "window": 3, "infiltration_probability": 0.96, "risk": 0.96,
            "stage": "Exfiltration", "level": "CONFIRMED", "alert": True,
            "observed_score": 0.38, "corroborated": True,
        }]
        r.novelty = NoveltyReport(verdict, 0.204, 0.31, 250, 291, 103)
        return r

    def test_out_of_distribution_withholds_the_verdict(self):
        head = self._result(OUTSIDE).headline()
        self.assertIn("WITHHELD", head)
        self.assertNotIn("CONFIRMED", head)

    def test_in_distribution_leaves_the_verdict_alone(self):
        head = self._result(IN_DISTRIBUTION).headline()
        self.assertIn("CONFIRMED", head)

    def test_marginal_does_not_withhold(self):
        """Marginal is a caveat, not a refusal -- it warns in the notes and
        leaves the verdict standing."""
        head = self._result(MARGINAL).headline()
        self.assertIn("CONFIRMED", head)


if __name__ == "__main__":
    unittest.main()
