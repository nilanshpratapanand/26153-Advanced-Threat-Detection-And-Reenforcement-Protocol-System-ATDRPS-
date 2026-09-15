"""Dual-track scoring: observed evidence versus model forecast.

These tests encode the failure this module was written to fix. An
nmap-faithful capture run through a trained model produced a correct
Reconnaissance call on the scan window and then eight minutes of phantom
LateralMovement / CommandAndControl / Exfiltration alerts on windows whose
traffic was pure benign background. A benign control through the same model
fired once in 45 windows, so the background was never the problem: the model
had learned that recon is followed by a kill chain, and the scan stayed in its
16-window context.
"""

from __future__ import annotations

import unittest

from atdrps.engine.scoring import (
    CLEAR, CONFIRMED, PREDICTED, WATCH, ScoringPolicy, score_window,
)


def _scan_window() -> dict:
    """Features of a window that really does contain a port scan."""
    return {
        "distinct_dst_ports": 100.0,
        "max_ports_per_src_dst": 100.0,
        "mean_payload_zero_ratio": 0.95,
        "mean_flow_packets": 3.0,
        "rst_share": 0.9,
    }


def _quiet_window() -> dict:
    """Features of an ordinary benign window: nothing for any rule to fire on."""
    return {
        "distinct_dst_ports": 3.0,
        "max_ports_per_src_dst": 1.0,
        "mean_payload_zero_ratio": 0.05,
        "mean_flow_packets": 40.0,
        "rst_share": 0.0,
        "beacon_regularity": 0.2,
        "internal_flow_share": 0.1,
        "admin_service_share": 0.0,
        "outbound_bytes_ratio": 0.1,
        "half_open_ratio": 0.0,
    }


class TestVerdictLevels(unittest.TestCase):
    def test_forecast_with_supporting_evidence_is_confirmed(self):
        v = score_window(20, 0.83, "Reconnaissance", _scan_window())
        self.assertEqual(v.level, CONFIRMED)
        self.assertTrue(v.corroborated)
        self.assertGreater(v.observed_score, 0.0)
        self.assertAlmostEqual(v.risk, 0.83, places=3,
                               msg="a corroborated forecast keeps its full risk")

    def test_confident_forecast_with_no_evidence_is_only_predicted(self):
        """The exact phantom-cascade case: model certain, capture shows nothing."""
        v = score_window(38, 0.99, "Exfiltration", _quiet_window())
        self.assertEqual(v.level, PREDICTED)
        self.assertFalse(v.corroborated)
        self.assertLess(v.risk, 0.99,
                        "an uncorroborated forecast must not carry full weight")

    def test_quiet_window_with_a_calm_model_is_clear(self):
        v = score_window(5, 0.06, "Benign", _quiet_window())
        self.assertEqual(v.level, CLEAR)

    def test_evidence_without_a_forecast_still_raises_a_watch(self):
        """The rule engine must be able to speak even when the model is calm."""
        v = score_window(9, 0.05, "Benign", _scan_window())
        self.assertEqual(v.level, WATCH)


class TestEvidenceWeightedThreshold(unittest.TestCase):
    """Corroboration lowers the bar the model must clear on its own."""

    def test_threshold_slides_down_as_evidence_rises(self):
        pol = ScoringPolicy()
        self.assertAlmostEqual(pol.confirm_threshold(0.0), pol.forecast_high, places=6)
        self.assertAlmostEqual(pol.confirm_threshold(pol.evidence_strong),
                               pol.forecast_floor, places=6)
        self.assertLess(pol.confirm_threshold(0.2), pol.confirm_threshold(0.05))

    def test_a_corroborated_scan_just_under_the_flat_bar_still_confirms(self):
        """The measured real case: nmap -T4 scored 0.698 against a 0.70 bar."""
        v = score_window(20, 0.698, "Reconnaissance", _scan_window())
        self.assertEqual(v.level, CONFIRMED)

    def test_an_uncorroborated_forecast_at_the_same_level_does_not_confirm(self):
        v = score_window(20, 0.698, "Exfiltration", _quiet_window())
        self.assertNotEqual(v.level, CONFIRMED)


class TestPolicyValidation(unittest.TestCase):
    def test_rejects_inverted_thresholds(self):
        with self.assertRaises(ValueError):
            ScoringPolicy(forecast_high=0.3, forecast_low=0.8).validate()
        with self.assertRaises(ValueError):
            ScoringPolicy(evidence_strong=0.1, evidence_weak=0.9).validate()

    def test_rejects_a_floor_outside_the_band(self):
        with self.assertRaises(ValueError):
            ScoringPolicy(forecast_floor=0.95).validate()


if __name__ == "__main__":
    unittest.main()
