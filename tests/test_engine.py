"""End-to-end tests: file on disk -> forecast, and the offline dashboard.

These are the tests that would catch a regression the unit tests miss -- the
pipeline being wired together wrongly even though every part works.
"""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from atdrps.data.flows import assemble_flows
from atdrps.data.schema import PacketTable
from atdrps.data.synth import generate_capture, write_scenario
from atdrps.data.windows import build_windows
from atdrps.engine.inference import ThreatForecastEngine, load_capture
from atdrps.explain.explainer import ForecastExplainer
from atdrps.models.numpy_dynamics import NumpyDynamicsWorldModel
from atdrps.train.dataset import build_sequences, group_split

CONTEXT, HORIZON, WINDOW = 8, 3, 30.0


def _corpus(seeds=range(6), duration=1500.0):
    caps = []
    for seed in seeds:
        cap = generate_capture(seed=1000 + seed, duration_s=duration, n_campaigns=2,
                               background_intensity=0.12)
        flows = assemble_flows(PacketTable.from_records(cap.packets))
        caps.append(build_windows(flows, window_size_s=WINDOW, stage_timeline=cap.stage_at))
    return caps


class _Fixture:
    """Trained once for the whole module -- generating traffic is the slow part."""

    model = None
    engine = None
    pcap = None
    tmpdir = None

    @classmethod
    def setup(cls):
        if cls.engine is not None:
            return
        caps = _corpus()
        ds = build_sequences(caps, context=CONTEXT, horizon=HORIZON)
        train, val, _ = group_split(ds, 0.2, 0.2)
        cls.model = NumpyDynamicsWorldModel(ds.feature_names, CONTEXT, HORIZON)
        cls.model.fit(train, val, verbose=False)
        explainer = ForecastExplainer(cls.model, train.context, top_k=5,
                                      n_samples=64, n_background=8)
        cls.engine = ThreatForecastEngine(cls.model, explainer, window_size_s=WINDOW)

        cls.tmpdir = tempfile.mkdtemp()
        demo = generate_capture(seed=777, duration_s=1200, n_campaigns=2,
                                background_intensity=0.12)
        info = write_scenario(cls.tmpdir, demo, name="demo")
        cls.pcap = info["pcap"]


class TestLoadCapture(unittest.TestCase):
    def test_reads_a_pcap(self):
        _Fixture.setup()
        states, flows, kind = load_capture(_Fixture.pcap, window_size_s=WINDOW)
        self.assertGreater(len(states), CONTEXT)
        self.assertGreater(len(flows), 10)
        self.assertIn("pcap", kind)
        self.assertEqual(states.label_source, "unlabelled")

    def test_reads_a_flow_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "flows.csv")
            pd.DataFrame({
                "Dst Port": [80, 445, 22] * 20,
                "Protocol": [6] * 60,
                "Timestamp": [1_700_000_000 + i * 10 for i in range(60)],
                "Flow Duration": [1_000_000] * 60,
                "Tot Fwd Pkts": [5] * 60, "Tot Bwd Pkts": [4] * 60,
                "TotLen Fwd Pkts": [600] * 60, "TotLen Bwd Pkts": [900] * 60,
                "Label": ["Benign"] * 60,
            }).to_csv(path, index=False)
            states, flows, kind = load_capture(path, window_size_s=WINDOW)
        self.assertEqual(len(flows), 60)
        self.assertIn("CSV", kind)


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _Fixture.setup()
        cls.result = _Fixture.engine.analyse(_Fixture.pcap, explain=True)

    def test_produces_a_timeline(self):
        self.assertGreater(len(self.result.timeline), 10)
        for row in self.result.timeline:
            self.assertGreaterEqual(row["infiltration_probability"], 0.0)
            self.assertLessEqual(row["infiltration_probability"], 1.0)
            self.assertIn("stage", row)
            self.assertIsInstance(row["alert"], bool)

    def test_timeline_starts_after_the_context(self):
        self.assertEqual(self.result.timeline[0]["window"], CONTEXT - 1)

    def test_forecast_is_bounded_and_the_right_length(self):
        forecast = self.result.forecast
        self.assertIsNotNone(forecast)
        self.assertEqual(len(forecast.timeline()), HORIZON)
        self.assertTrue(((forecast.infiltration >= 0) & (forecast.infiltration <= 1)).all())
        self.assertTrue(np.isfinite(forecast.states).all())

    def test_forward_simulation_does_not_oscillate_wildly(self):
        """An unclamped roll-out swung 1.00 -> 0.00 -> 1.00 across five steps.
        Consecutive steps should not make the maximum possible jump."""
        probs = self.result.forecast.infiltration
        if len(probs) > 1:
            self.assertLess(float(np.abs(np.diff(probs)).max()), 0.999)

    def test_flagged_flows_have_the_expected_shape(self):
        self.assertTrue(self.result.flagged_flows)
        first = self.result.flagged_flows[0]
        for key in ("src", "dst", "protocol", "packets", "bytes", "flags"):
            self.assertIn(key, first)

    def test_explanation_is_attached_and_balanced(self):
        explanation = self.result.explanation
        self.assertIsNotNone(explanation)
        infil = explanation["infiltration_explanation"]
        self.assertLess(infil.efficiency_error, 1e-6)
        self.assertTrue(infil.drivers())
        self.assertIn("infiltration", infil.to_text())

    def test_headline_states_a_verdict_not_a_bare_probability(self):
        """The headline must say what is *confirmed* versus merely forecast.

        It used to read "peak infiltration probability 1.00 (stage: X)" where X
        was whatever window scored highest -- including a window the model had
        only extrapolated to, with nothing in the capture supporting it. On an
        nmap capture whose only attack was a 3-second scan that produced
        "peak infiltration probability 1.00 (stage: Exfiltration)".
        """
        headline = self.result.headline()
        self.assertTrue(
            any(word in headline for word in ("CONFIRMED", "PREDICTED", "no confirmed intrusion")),
            f"headline should carry a verdict, got: {headline!r}",
        )
        self.assertIn("forward simulation peaks at", headline)

    def test_short_capture_reports_a_note_rather_than_crashing(self):
        tiny = generate_capture(seed=5, duration_s=60, n_campaigns=1,
                                background_intensity=0.05)
        with tempfile.TemporaryDirectory() as tmp:
            info = write_scenario(tmp, tiny, name="tiny")
            result = _Fixture.engine.analyse(info["pcap"])
        self.assertEqual(result.timeline, [])
        self.assertTrue(result.notes)
        self.assertIn("context", result.notes[0])

    def test_detects_the_injected_campaign(self):
        """A capture containing a real kill chain must not score flat.

        The bar is on the *model* output (``raw_peak_probability``), which is
        what this assertion has always really been about. ``peak_risk`` is now
        the dual-track credible score, and this fixture deliberately trains a
        very small model on a tiny corpus: its raw peak is around 0.58, and on
        a signal that weak the scoring layer is supposed to decline to confirm.
        Asserting a confirmed verdict here would be asserting that an
        under-trained model sounds certain, which is the opposite of the point.
        The shipped model on a real campaign does reach CONFIRMED at 1.00 --
        that is covered by the benchmark, not by this fixture.
        """
        self.assertGreater(self.result.raw_peak_probability, 0.5)
        levels = {row.get("level") for row in self.result.timeline}
        self.assertTrue(
            levels - {"CLEAR"},
            "a capture with a real kill chain should raise at least a WATCH",
        )


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _Fixture.setup()
        from app.server import create_app

        cls.model_dir = tempfile.mkdtemp()
        _Fixture.model.save(cls.model_dir)
        np.save(Path(cls.model_dir) / "background.npy", _Fixture.engine.explainer.background)
        cls.app = create_app(model_dir=cls.model_dir)
        cls.app.config.update(TESTING=True)
        cls.client = cls.app.test_client()

    def test_health_reports_a_loaded_model(self):
        payload = self.client.get("/api/health").get_json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["offline"])
        self.assertEqual(payload["context"], CONTEXT)

    def test_index_renders_and_loads_nothing_external(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("ATD", html)
        # an air-gapped deployment cannot fetch a CDN
        for marker in ("http://", "https://"):
            self.assertNotIn(marker, html)

    def test_analyse_returns_a_full_payload(self):
        with open(_Fixture.pcap, "rb") as fh:
            data = {"capture": (io.BytesIO(fh.read()), "demo.pcap"),
                    "threshold": "0.5", "horizon": str(HORIZON)}
            response = self.client.post("/api/analyse", data=data,
                                        content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        for key in ("headline", "timeline", "forecast", "flagged_flows", "explanation"):
            self.assertIn(key, payload)
        self.assertGreater(len(payload["timeline"]), 5)
        self.assertEqual(len(payload["forecast"]), HORIZON)
        # the payload has to survive a JSON round trip for the browser
        json.loads(json.dumps(payload))

    def test_rejects_an_unsupported_file_type(self):
        data = {"capture": (io.BytesIO(b"not a capture"), "notes.txt")}
        response = self.client.post("/api/analyse", data=data,
                                    content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported", response.get_json()["error"])

    def test_rejects_an_empty_request(self):
        response = self.client.post("/api/analyse", data={},
                                    content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_a_corrupt_capture_is_a_400_not_a_500(self):
        data = {"capture": (io.BytesIO(b"\x00" * 200), "broken.pcap")}
        response = self.client.post("/api/analyse", data=data,
                                    content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())


class TestCli(unittest.TestCase):
    def test_parser_exposes_every_subcommand(self):
        from atdrps.cli import build_parser

        parser = build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        names = set()
        for action in actions:
            names |= set(action.choices or {})
        for expected in ("synth", "corpus", "train", "benchmark", "predict", "serve"):
            self.assertIn(expected, names)

    def test_synth_writes_a_capture(self):
        from atdrps.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            code = main(["synth", "--out", tmp, "--seed", "3", "--duration", "180",
                         "--campaigns", "1"])
            self.assertEqual(code, 0)
            self.assertTrue(Path(tmp, "capture.pcap").exists())
            self.assertTrue(Path(tmp, "capture.timeline.json").exists())


if __name__ == "__main__":
    unittest.main()
