"""The train-local / forecast workflow, end to end, on a small simulated 'network'."""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atdrps.data.overlay import overlay_campaigns, synthetic_background
from atdrps.data.pcap import write_pcap
from atdrps.forecast.hazard import HazardModel
from atdrps.forecast.local import analyse_capture, train_local
from atdrps.models.safe_pickle import UnsafePickleError


class _Evil:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


class TestLocalWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.bg = synthetic_background(seed=31, duration_s=2700.0)
        write_pcap(t / "benign.pcap", cls.bg.records)
        cap = overlay_campaigns(cls.bg, seed=5, n_campaigns=1, direct_access_prob=1.0,
                                continue_probs=(1, 1, 1, 1))
        assert {"InitialAccess", "LateralMovement"} <= {i.stage for i in cap.timeline}, \
            "the test capture must actually contain the stages it claims to"
        write_pcap(t / "attack.pcap", cap.packets)
        cls.logs = []
        cls.meta = train_local(t / "benign.pcap", t / "model", n_train=16, n_val=30, workers=1,
                               min_train_positives=1, min_val_positives=1, log=cls.logs.append)
        cls.model_dir = t / "model"
        cls.attack = t / "attack.pcap"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artifacts_and_metadata(self):
        for name in ("profile.json", "hazard.pkl", "meta.json"):
            self.assertTrue((self.model_dir / name).exists(), name)
        m = json.loads((self.model_dir / "meta.json").read_text())
        for key in ("threshold", "task", "calibration_windows", "false_alerts_per_hour_upper95",
                    "budget_verified", "assumption", "disclaimer", "sklearn"):
            self.assertIn(key, m)
        self.assertEqual(m["assumption"], "the training capture was benign")

    def test_small_calibration_set_is_not_claimed_to_verify_the_budget(self):
        self.assertFalse(self.meta["budget_verified"])
        self.assertGreater(self.meta["false_alerts_per_hour_upper95"], 1.0)
        self.assertTrue(any("not the requested" in line or "Capture more benign" in line for line in self.logs))

    def test_forecast_returns_a_consistent_report(self):
        r = analyse_capture(self.model_dir, self.attack)
        self.assertGreater(len(r["windows"]), 10)
        alerts = [w for w in r["windows"] if w["alert"]]
        self.assertEqual(sum(e["n"] for e in r["episodes"]), len(alerts))
        for w in r["windows"]:
            self.assertGreaterEqual(w["probability"], 0.0)
            self.assertLessEqual(w["probability"], 1.0)
            self.assertEqual(w["alert"], w["score"] >= self.meta["threshold"])
        self.assertTrue(any("not verified" in x for x in r["warnings"]))
        self.assertIn("not proof", r["disclaimer"])

    def test_threshold_controls_alerts_and_episodes_merge(self):
        meta_path = self.model_dir / "meta.json"
        original = meta_path.read_text()
        try:
            m = json.loads(original)
            m["threshold"] = -1.0
            meta_path.write_text(json.dumps(m))
            r = analyse_capture(self.model_dir, self.attack)
            self.assertEqual(len(r["episodes"]), 1)
            self.assertEqual(r["episodes"][0]["n"], len(r["windows"]))
            m["threshold"] = 2.0
            meta_path.write_text(json.dumps(m))
            self.assertEqual(analyse_capture(self.model_dir, self.attack)["episodes"], [])
        finally:
            meta_path.write_text(original)

    def test_model_round_trip_gives_identical_scores(self):
        a = HazardModel.load(self.model_dir)
        ctx = np.random.default_rng(0).normal(size=(6, 16, len(a.profile.centre))).astype(np.float32)
        np.testing.assert_allclose(a.score(ctx), HazardModel.load(self.model_dir).score(ctx))

    def test_library_version_mismatch_is_reported(self):
        meta_path = self.model_dir / "meta.json"
        original = meta_path.read_text()
        try:
            m = json.loads(original)
            m["sklearn"] = "0.0.1"
            meta_path.write_text(json.dumps(m))
            r = analyse_capture(self.model_dir, self.attack)
            self.assertTrue(any("scikit-learn" in x for x in r["warnings"]))
        finally:
            meta_path.write_text(original)

    def test_tampered_model_file_cannot_run_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "pwned")
            copy = Path(tmp) / "model"
            copy.mkdir()
            for name in ("profile.json", "meta.json"):
                (copy / name).write_bytes((self.model_dir / name).read_bytes())
            (copy / "hazard.pkl").write_bytes(pickle.dumps(_Evil(marker)))
            with self.assertRaises(UnsafePickleError):
                HazardModel.load(copy)
            self.assertFalse(os.path.exists(marker))


class TestGuards(unittest.TestCase):
    def test_too_short_a_benign_capture_is_refused(self):
        bg = synthetic_background(seed=3, duration_s=600.0)
        with tempfile.TemporaryDirectory() as tmp:
            write_pcap(Path(tmp) / "short.pcap", bg.records)
            with self.assertRaises(ValueError) as cm:
                train_local(Path(tmp) / "short.pcap", Path(tmp) / "m", workers=1, log=lambda *_: None)
            self.assertIn("minutes", str(cm.exception))

    def test_bad_task_is_refused(self):
        with self.assertRaises(ValueError):
            train_local("nonexistent.pcap", "out", task="prophecy")


if __name__ == "__main__":
    unittest.main()
