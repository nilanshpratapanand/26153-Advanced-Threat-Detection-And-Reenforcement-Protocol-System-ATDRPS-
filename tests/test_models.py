import tempfile
import unittest

import numpy as np

from atdrps.data.schema import STAGES
from atdrps.data.windows import WindowedStates
from atdrps.models.base import Forecast, Standardiser
from atdrps.models.baseline import LogisticBaseline
from atdrps.models.numpy_dynamics import NumpyDynamicsWorldModel, PersistenceModel
from atdrps.train.dataset import (
    build_sequences, class_weights, group_split, time_split,
)
from atdrps.train.evaluate import (
    benchmark_table, binary_metrics, dynamics_metrics, find_best_threshold,
    stage_metrics,
)


def make_capture(n_windows=60, n_features=6, seed=0, offset=0.0):
    """A capture whose next state is a genuine function of the last two."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_windows, n_features), dtype=np.float32)
    X[0] = rng.normal(size=n_features)
    X[1] = rng.normal(size=n_features)
    for t in range(2, n_windows):
        X[t] = 0.7 * X[t - 1] + 0.2 * (X[t - 1] - X[t - 2]) + 0.05 * rng.normal(size=n_features)
    stage = np.zeros(n_windows, dtype=np.int64)
    stage[X[:, 0] > 0.3] = 3            # LateralMovement
    infil = (stage > 1).astype(np.int64)
    return WindowedStates(
        X=X, feature_names=[f"f{i}" for i in range(n_features)],
        ts_start=offset + np.arange(n_windows) * 30.0, window_size_s=30.0,
        stage=stage, stage_mask=np.ones(n_windows, dtype=bool),
        infiltration=infil, label_source="ground-truth timeline",
    )


class TestSequenceBuilding(unittest.TestCase):
    def test_target_is_the_future_not_the_present(self):
        cap = make_capture(40, 4, seed=1)
        ds = build_sequences([cap], context=5, horizon=3)
        # the first sample uses windows 0..4 and is supervised on 5..7
        self.assertTrue(np.allclose(ds.context[0], cap.X[0:5]))
        self.assertTrue(np.allclose(ds.target_state[0], cap.X[5]))
        self.assertTrue(np.array_equal(ds.target_stage[0], cap.stage[5:8]))

    def test_the_target_window_is_never_inside_the_context(self):
        cap = make_capture(30, 3, seed=2)
        ds = build_sequences([cap], context=4, horizon=2)
        for i in range(len(ds)):
            for row in ds.context[i]:
                self.assertFalse(np.allclose(row, ds.target_state[i]),
                                 "target state leaked into the context")

    def test_sequences_never_span_two_captures(self):
        caps = [make_capture(30, 3, seed=3), make_capture(30, 3, seed=4, offset=10_000)]
        ds = build_sequences(caps, context=4, horizon=2)
        self.assertEqual(set(np.unique(ds.group).tolist()), {0, 1})
        for gid in (0, 1):
            sel = ds.group == gid
            self.assertEqual(len(np.unique(ds.ts[sel])), int(sel.sum()))

    def test_short_capture_is_skipped(self):
        ds = build_sequences([make_capture(5, 3)], context=8, horizon=3)
        self.assertEqual(len(ds), 0)
        self.assertEqual(ds.context.shape[1:], (8, 3))

    def test_feature_space_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            build_sequences([make_capture(30, 3), make_capture(30, 5)], context=4, horizon=2)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            build_sequences([make_capture(30, 3)], context=0, horizon=2)


class TestSplits(unittest.TestCase):
    def setUp(self):
        caps = [make_capture(80, 4, seed=i, offset=i * 100_000) for i in range(6)]
        self.ds = build_sequences(caps, context=5, horizon=2)

    def test_time_split_is_chronological_within_each_capture(self):
        train, val, test = time_split(self.ds, 0.15, 0.2, gap=5)
        for gid in np.unique(self.ds.group):
            tr = train.ts[train.group == gid]
            va = val.ts[val.group == gid]
            te = test.ts[test.group == gid]
            if tr.size and va.size:
                self.assertLess(tr.max(), va.min())
            if va.size and te.size:
                self.assertLess(va.max(), te.min())

    def test_gap_is_clamped_so_no_block_empties(self):
        train, val, test = time_split(self.ds, 0.15, 0.2, gap=10_000)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

    def test_group_split_shares_no_capture(self):
        train, val, test = group_split(self.ds, 0.2, 0.2)
        tg, vg, sg = (set(np.unique(d.group).tolist()) for d in (train, val, test))
        self.assertEqual(tg & vg, set())
        self.assertEqual(tg & sg, set())
        self.assertEqual(vg & sg, set())
        self.assertEqual(len(tg | vg | sg), 6)

    def test_group_split_is_reproducible(self):
        a = group_split(self.ds, 0.2, 0.2)[2]
        b = group_split(self.ds, 0.2, 0.2)[2]
        self.assertTrue(np.array_equal(np.unique(a.group), np.unique(b.group)))

    def test_group_split_refuses_impossible_request(self):
        with self.assertRaises(ValueError):
            group_split(self.ds, 0.5, 0.5)

    def test_bad_fractions_rejected(self):
        with self.assertRaises(ValueError):
            time_split(self.ds, 0.6, 0.6)


class TestClassWeights(unittest.TestCase):
    def test_rare_classes_weigh_more(self):
        labels = np.array([0] * 90 + [3] * 10)
        w = class_weights(labels, len(STAGES))
        self.assertGreater(w[3], w[0])

    def test_absent_classes_get_unit_weight(self):
        w = class_weights(np.array([0, 0, 1]), len(STAGES))
        self.assertEqual(w[5], 1.0)

    def test_mask_is_respected(self):
        labels = np.array([0, 0, 3, 3])
        mask = np.array([True, True, False, False])
        w = class_weights(labels, len(STAGES), mask)
        self.assertEqual(w[3], 1.0)


class TestStandardiser(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4, 3)).astype(np.float32) * 10 + 5
        s = Standardiser().fit(X)
        self.assertTrue(np.allclose(s.inverse(s.transform(X)), X, atol=1e-3))

    def test_constant_column_does_not_divide_by_zero(self):
        X = np.ones((20, 3, 2), dtype=np.float32)
        X[:, :, 1] = np.arange(20)[:, None]
        out = Standardiser().fit(X).transform(X)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.allclose(out[:, :, 0], 0.0))

    def test_state_dict_roundtrip(self):
        X = np.random.default_rng(1).normal(size=(30, 2, 4)).astype(np.float32)
        s = Standardiser().fit(X)
        s2 = Standardiser.from_state_dict(s.state_dict())
        self.assertTrue(np.allclose(s.transform(X), s2.transform(X)))

    def test_clamp_confines_states_to_the_observed_range(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 4, 3)).astype(np.float32)
        s = Standardiser().fit(X)
        wild = np.full((1, 3), 1e6, dtype=np.float32)
        clamped = s.clamp(wild)
        self.assertTrue(np.isfinite(clamped).all())
        self.assertTrue((np.abs(clamped) < 1e3).all())

    def test_clamp_leaves_ordinary_states_alone(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(400, 4, 3)).astype(np.float32)
        s = Standardiser().fit(X)
        typical = X[0]
        self.assertTrue(np.allclose(s.clamp(typical), typical, atol=1e-3))

    def test_clamp_survives_a_state_dict_roundtrip(self):
        X = np.random.default_rng(4).normal(size=(120, 3, 4)).astype(np.float32)
        s = Standardiser().fit(X)
        s2 = Standardiser.from_state_dict(s.state_dict())
        wild = np.full((1, 4), 500.0, dtype=np.float32)
        self.assertTrue(np.allclose(s.clamp(wild), s2.clamp(wild)))

    def test_use_before_fit_is_an_error(self):
        with self.assertRaises(RuntimeError):
            Standardiser().transform(np.zeros((2, 2)))


class TestNumpyWorldModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        caps = [make_capture(120, 6, seed=i, offset=i * 100_000) for i in range(10)]
        cls.ds = build_sequences(caps, context=6, horizon=3)
        cls.train, cls.val, cls.test = group_split(cls.ds, 0.2, 0.2)
        cls.model = NumpyDynamicsWorldModel(cls.ds.feature_names, 6, 3).fit(
            cls.train, cls.val, verbose=False
        )

    def test_beats_persistence_on_next_state_error(self):
        """The capture generator is a genuine autoregressive process, so a model
        that learns dynamics must beat 'assume nothing changes'."""
        persistence = PersistenceModel(self.ds.feature_names, 6, 3).fit(self.train)
        learned = dynamics_metrics(self.model.predict_next_batch(self.test.context),
                                   self.test.target_state, self.model.standardiser)
        floor = dynamics_metrics(persistence.predict_next_batch(self.test.context),
                                 self.test.target_state, persistence.standardiser)
        self.assertLess(learned["mse"], floor["mse"])

    def test_prediction_shapes(self):
        state = self.model.predict_next(self.test.context[0])
        self.assertEqual(state.shape, (self.ds.n_features,))
        probs, infil = self.model.heads(self.test.context[0])
        self.assertEqual(probs.shape, (len(STAGES),))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)
        self.assertGreaterEqual(infil, 0.0)
        self.assertLessEqual(infil, 1.0)

    def test_batch_and_single_agree(self):
        single = self.model.predict_next(self.test.context[3])
        batch = self.model.predict_next_batch(self.test.context[3:4])[0]
        self.assertTrue(np.allclose(single, batch, atol=1e-4))

    def test_rollout_contract(self):
        forecast = self.model.rollout(self.test.context[0], horizon=4)
        self.assertIsInstance(forecast, Forecast)
        self.assertEqual(forecast.states.shape, (4, self.ds.n_features))
        self.assertEqual(forecast.stage_probs.shape, (4, len(STAGES)))
        self.assertTrue(((forecast.infiltration >= 0) & (forecast.infiltration <= 1)).all())
        self.assertTrue(np.allclose(forecast.stage_probs.sum(axis=1), 1.0, atol=1e-4))
        self.assertEqual(len(forecast.timeline()), 4)
        self.assertIn("peak infiltration", forecast.summary())

    def test_rollout_stays_inside_the_training_distribution(self):
        """Unclamped autoregressive roll-outs leave the data distribution within
        a few steps, and the heads then return saturated nonsense."""
        forecast = self.model.rollout(self.test.context[0], horizon=12)
        train_lo = self.train.context.reshape(-1, self.ds.n_features).min(axis=0)
        train_hi = self.train.context.reshape(-1, self.ds.n_features).max(axis=0)
        span = train_hi - train_lo + 1e-6
        self.assertTrue((forecast.states >= train_lo - 3 * span).all())
        self.assertTrue((forecast.states <= train_hi + 3 * span).all())
        self.assertTrue(np.isfinite(forecast.states).all())

    def test_rollout_pads_a_short_context(self):
        forecast = self.model.rollout(self.test.context[0][-2:], horizon=2)
        self.assertEqual(forecast.states.shape[0], 2)

    def test_rollout_rejects_wrong_feature_count(self):
        with self.assertRaises(ValueError):
            self.model.rollout(np.zeros((6, self.ds.n_features + 1), dtype=np.float32))

    def test_checkpoint_roundtrip(self):
        before = self.model.predict_next(self.test.context[1])
        pb, ib = self.model.heads(self.test.context[1])
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save(tmp)
            reloaded = NumpyDynamicsWorldModel.load(tmp)
        self.assertTrue(np.allclose(before, reloaded.predict_next(self.test.context[1]), atol=1e-5))
        pa, ia = reloaded.heads(self.test.context[1])
        self.assertTrue(np.allclose(pb, pa, atol=1e-6))
        self.assertAlmostEqual(ib, ia, places=6)

    def test_ridge_strength_chosen_on_validation(self):
        model = NumpyDynamicsWorldModel(self.ds.feature_names, 6, 3)
        model.fit(self.train, self.val, alphas=(0.01, 1.0, 100.0), verbose=False)
        self.assertIn(model.ridge_alpha, (0.01, 1.0, 100.0))

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            NumpyDynamicsWorldModel(self.ds.feature_names, 6, 3).fit(self.train.subset(np.array([])))


class TestLogisticBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        caps = [make_capture(120, 6, seed=i, offset=i * 100_000) for i in range(8)]
        cls.ds = build_sequences(caps, context=5, horizon=3)
        cls.train, _, cls.test = group_split(cls.ds, 0.2, 0.2)

    def test_static_and_context_modes_both_predict(self):
        for mode in ("static", "context"):
            with self.subTest(mode=mode):
                base = LogisticBaseline(mode=mode).fit(self.train, n_stages=len(STAGES))
                stage, infil = base.predict(self.test)
                self.assertEqual(stage.shape, (len(self.test), 3, len(STAGES)))
                self.assertEqual(infil.shape, (len(self.test), 3))
                self.assertTrue(((infil >= 0) & (infil <= 1)).all())

    def test_one_model_is_fitted_per_horizon_step(self):
        base = LogisticBaseline().fit(self.train, n_stages=len(STAGES))
        self.assertEqual(len(base.stage_models), 3)
        self.assertEqual(len(base.infil_models), 3)

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            LogisticBaseline(mode="telepathy")


class TestMetrics(unittest.TestCase):
    def test_confusion_counts_by_hand(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7])
        m = binary_metrics(y, s, 0.5)
        self.assertEqual((m.tp, m.fp, m.tn, m.fn), (3, 1, 3, 1))
        self.assertAlmostEqual(m.precision, 0.75)
        self.assertAlmostEqual(m.recall, 0.75)
        self.assertAlmostEqual(m.f1, 0.75)
        self.assertAlmostEqual(m.fpr, 0.25)
        self.assertAlmostEqual(m.accuracy, 0.75)

    def test_fpr_is_zero_when_nothing_is_flagged(self):
        y = np.array([0, 0, 1, 1])
        m = binary_metrics(y, np.array([0.1, 0.1, 0.2, 0.2]), 0.9)
        self.assertEqual(m.fpr, 0.0)
        self.assertEqual(m.recall, 0.0)

    def test_single_class_gives_nan_auc_not_a_crash(self):
        m = binary_metrics(np.zeros(5), np.linspace(0, 1, 5), 0.5)
        self.assertNotEqual(m.roc_auc, m.roc_auc)     # NaN

    def test_threshold_search_respects_an_fpr_budget(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7])
        t = find_best_threshold(y, s, max_fpr=0.0)
        self.assertEqual(binary_metrics(y, s, t).fpr, 0.0)

    def test_stage_metrics_confusion(self):
        sm = stage_metrics(np.array([0, 1, 1, 2]), np.eye(3)[[0, 1, 2, 2]],
                           ["Benign", "Recon", "IA"])
        self.assertAlmostEqual(sm.accuracy, 0.75)
        self.assertEqual(sm.confusion.shape, (3, 3))
        self.assertEqual(int(sm.confusion.sum()), 4)

    def test_dynamics_metrics_zero_for_perfect_prediction(self):
        X = np.random.default_rng(0).normal(size=(10, 4))
        m = dynamics_metrics(X, X)
        self.assertAlmostEqual(m["mse"], 0.0)
        self.assertAlmostEqual(m["mae"], 0.0)

    def test_table_renders_nan_as_not_available(self):
        table = benchmark_table([{"Model": "x", "AUC": float("nan")}])
        self.assertIn("n/a", table)


if __name__ == "__main__":
    unittest.main()
