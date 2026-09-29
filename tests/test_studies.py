"""The two studies: attack-family holdout and the dual-level ablation.

What these tests actually guard is the honesty of both studies, because a
study that quietly leaks is worse than no study at all:

* the feature partition must cover every state dimension exactly once, and
  must put packet-derived dimensions on the packet side -- an ablation whose
  "flow-only" arm still contains TTL variance proves nothing;
* the holdout must refuse to run if the held-out family survives in training;
* narrowing the feature set or the context must narrow the dynamics target and
  the context tensor with it, or the arms are not comparable.
"""

from __future__ import annotations

import unittest

import numpy as np

from atdrps.data.schema import STAGE_INDEX, STAGES
from atdrps.data.windows import state_feature_names
from atdrps.train.dataset import build_sequences
from atdrps.train.studies import (
    PACKET_AGGREGATE_BASES, PACKET_DETECTORS, context_slice, feature_levels,
    group_stages, run_ablation, run_attack_type_holdout, select_features,
    write_ablation_report, write_holdout_report,
)
from atdrps.data.windows import WindowedStates


def _capture(seed: int, stages: list[str], n: int = 40, n_features: int = 103):
    """A tiny synthetic windowed capture with a chosen set of stages in it."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    stage = np.zeros(n, dtype=np.int64)
    infil = np.zeros(n, dtype=np.int64)
    for i, name in enumerate(stages):
        lo = 5 + i * 6
        stage[lo:lo + 5] = STAGE_INDEX[name]
        infil[lo:lo + 5] = 1
        X[lo:lo + 5] += 2.5          # something for a model to key on
    return WindowedStates(
        X=X, feature_names=state_feature_names()[:n_features],
        ts_start=np.arange(n, dtype=np.float64) * 30.0, window_size_s=30.0,
        stage=stage, stage_mask=np.ones(n, dtype=bool), infiltration=infil,
        label_source="test",
    )


class TestFeaturePartition(unittest.TestCase):
    def setUp(self):
        self.names = state_feature_names()
        self.levels = feature_levels(self.names)

    def test_partition_is_exact(self):
        """Every dimension lands on exactly one side, none is lost."""
        both = self.levels["flow"] + self.levels["packet"]
        self.assertEqual(sorted(both), list(range(len(self.names))))
        self.assertEqual(len(set(both)), len(both))

    def test_packet_side_is_packet_derived(self):
        packet_names = [self.names[i] for i in self.levels["packet"]]
        for name in packet_names:
            base = name.split("__", 1)[0]
            self.assertTrue(
                base in PACKET_AGGREGATE_BASES or name in PACKET_DETECTORS,
                f"{name} is on the packet side but is not packet-derived",
            )

    def test_flow_side_holds_no_packet_feature(self):
        """The arm that stands in for a NetFlow export must not contain TTL
        variance, retransmissions or any other thing an export discards."""
        flow_names = [self.names[i] for i in self.levels["flow"]]
        for base in PACKET_AGGREGATE_BASES:
            for stat in ("mean", "std", "max"):
                self.assertNotIn(f"{base}__{stat}", flow_names)
        for name in PACKET_DETECTORS:
            self.assertNotIn(name, flow_names)

    def test_counts_are_the_documented_ones(self):
        self.assertEqual(len(self.names), 103)
        self.assertEqual(len(self.levels["packet"]), 23)
        self.assertEqual(len(self.levels["flow"]), 80)


class TestViews(unittest.TestCase):
    def setUp(self):
        self.ds = build_sequences([_capture(1, ["Reconnaissance", "Exfiltration"])],
                                  context=8, horizon=3)

    def test_select_features_narrows_input_and_target(self):
        idx = [0, 3, 7]
        view = select_features(self.ds, idx)
        self.assertEqual(view.context.shape[2], 3)
        self.assertEqual(view.target_state.shape[1], 3)
        self.assertEqual(len(view.feature_names), 3)
        np.testing.assert_allclose(view.context[:, :, 1], self.ds.context[:, :, 3])
        np.testing.assert_allclose(view.target_state[:, 2], self.ds.target_state[:, 7])

    def test_select_features_leaves_labels_alone(self):
        view = select_features(self.ds, [1, 2])
        np.testing.assert_array_equal(view.target_infil, self.ds.target_infil)
        np.testing.assert_array_equal(view.target_stage, self.ds.target_stage)

    def test_context_slice_keeps_the_most_recent_windows(self):
        view = context_slice(self.ds, 3)
        self.assertEqual(view.context_length, 3)
        np.testing.assert_allclose(view.context, self.ds.context[:, -3:, :])

    def test_context_slice_rejects_impossible_lengths(self):
        with self.assertRaises(ValueError):
            context_slice(self.ds, 0)
        with self.assertRaises(ValueError):
            context_slice(self.ds, self.ds.context_length + 1)


class TestGroupStages(unittest.TestCase):
    def test_reports_the_stages_each_capture_contains(self):
        caps = [_capture(2, ["Reconnaissance"]),
                _capture(3, ["LateralMovement", "Exfiltration"])]
        ds = build_sequences(caps, context=4, horizon=2)
        stages = group_stages(ds)
        self.assertIn(STAGE_INDEX["Reconnaissance"], stages[0])
        self.assertNotIn(STAGE_INDEX["LateralMovement"], stages[0])
        self.assertIn(STAGE_INDEX["LateralMovement"], stages[1])


class TestAttackTypeHoldout(unittest.TestCase):
    def _corpus(self):
        clean = [_capture(10 + i, ["Reconnaissance", "InitialAccess"])
                 for i in range(6)]
        dirty = [_capture(50 + i, ["Reconnaissance", "LateralMovement"])
                 for i in range(3)]
        return clean + dirty

    def test_runs_and_reports_the_unseen_family(self):
        payload = run_attack_type_holdout(
            self._corpus(), holdout_stage="LateralMovement",
            context=4, horizon=2, backend="linear", verbose=False,
        )
        self.assertEqual(payload["captures"]["held_out"], 3)
        self.assertEqual(payload["unseen_family"]["windows"] > 0, True)
        # a class never trained on cannot be emitted: that is the honest zero
        self.assertEqual(payload["unseen_family"]["stage_recall"], 0.0)
        self.assertIn("LateralMovement", payload["stages_absent_from_training"])

    def test_refuses_an_unknown_stage(self):
        with self.assertRaises(ValueError):
            run_attack_type_holdout(self._corpus(), holdout_stage="Nonsense",
                                    context=4, horizon=2, verbose=False)

    def test_refuses_when_the_family_is_everywhere(self):
        """Every capture contains it -> there is nothing to train on, and the
        study must say so rather than silently training on the family."""
        caps = [_capture(70 + i, ["LateralMovement"]) for i in range(5)]
        with self.assertRaises(ValueError):
            run_attack_type_holdout(caps, holdout_stage="LateralMovement",
                                    context=4, horizon=2, verbose=False)

    def test_training_split_never_contains_the_family(self):
        payload = run_attack_type_holdout(
            self._corpus(), holdout_stage="LateralMovement",
            context=4, horizon=2, backend="linear", verbose=False,
        )
        # run_attack_type_holdout raises if the family leaks, so reaching here
        # is the assertion; this pins the guarantee in a named test.
        self.assertEqual(payload["holdout_stage"], "LateralMovement")

    def test_report_renders(self):
        payload = run_attack_type_holdout(
            self._corpus(), holdout_stage="LateralMovement",
            context=4, horizon=2, backend="linear", verbose=False,
        )
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            text = write_holdout_report(payload, pathlib.Path(tmp) / "h.md")
        self.assertIn("LateralMovement", text)
        self.assertIn("Alarm fired", text)
        self.assertIn("In training?", text)


class TestAblation(unittest.TestCase):
    def test_arms_differ_only_in_input(self):
        caps = [_capture(100 + i, ["Reconnaissance", "Exfiltration"]) for i in range(8)]
        payload = run_ablation(caps, context=4, horizon=2, short_context=2,
                               backend="linear", verbose=False)
        variants = {r["Variant"]: r for r in payload["rows"]}
        self.assertIn("flow-only", variants)
        self.assertIn("packet-only", variants)
        self.assertIn("dual (shipped)", variants)
        self.assertEqual(variants["flow-only"]["Dims"]
                         + variants["packet-only"]["Dims"],
                         variants["dual (shipped)"]["Dims"])
        self.assertEqual(variants["dual, context=1"]["Context"], 1)

    def test_report_renders_with_findings(self):
        caps = [_capture(200 + i, ["Reconnaissance", "Exfiltration"]) for i in range(8)]
        payload = run_ablation(caps, context=4, horizon=2, short_context=2,
                               backend="linear", verbose=False)
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            text = write_ablation_report(payload, pathlib.Path(tmp) / "a.md")
        self.assertIn("What the table says", text)
        self.assertIn("flow-only", text)


if __name__ == "__main__":
    unittest.main()
