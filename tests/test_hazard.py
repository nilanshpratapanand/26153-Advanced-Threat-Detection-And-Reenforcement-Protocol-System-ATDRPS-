"""Network profile and hazard model: behaviour that must hold regardless of the data."""

import unittest

import numpy as np

from atdrps.forecast.hazard import HazardModel, NetworkProfile, context_features


class TestProfile(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.benign = rng.normal([5.0, 100.0, 0.0], [1.0, 20.0, 0.01], size=(300, 3))

    def test_benign_windows_score_near_zero(self):
        p = NetworkProfile().fit(self.benign)
        z = p.transform(self.benign)
        self.assertLess(abs(float(np.median(z))), 0.2)
        self.assertLess(float(np.mean(np.abs(z) > 4)), 0.02)

    def test_a_shift_far_from_this_networks_normal_is_large(self):
        p = NetworkProfile().fit(self.benign)
        z = p.transform(np.array([[5.0, 400.0, 0.0]]))
        self.assertGreater(float(z[0, 1]), 8)

    def test_same_traffic_is_normal_on_one_network_and_odd_on_another(self):
        rng = np.random.default_rng(1)
        chatty = rng.normal([50.0, 100.0, 0.0], [5.0, 20.0, 0.01], size=(300, 3))
        a, b = NetworkProfile().fit(self.benign), NetworkProfile().fit(chatty)
        probe = np.array([[50.0, 100.0, 0.0]])
        self.assertGreater(abs(float(a.transform(probe)[0, 0])), 5)
        self.assertLess(abs(float(b.transform(probe)[0, 0])), 1)

    def test_constant_feature_does_not_explode(self):
        x = self.benign.copy()
        x[:, 2] = 0.0
        p = NetworkProfile().fit(x)
        self.assertTrue(np.isfinite(p.transform(x)).all())
        self.assertLessEqual(float(np.abs(p.transform(x)).max()), 10.0)

    def test_needs_enough_windows(self):
        with self.assertRaises(ValueError):
            NetworkProfile().fit(self.benign[:3])

    def test_round_trip(self):
        p = NetworkProfile().fit(self.benign)
        q = NetworkProfile.from_dict(p.to_dict())
        np.testing.assert_allclose(p.transform(self.benign), q.transform(self.benign))


class TestFeatures(unittest.TestCase):
    def test_shape_and_finiteness(self):
        z = np.random.default_rng(0).normal(size=(7, 16, 10))
        f = context_features(z)
        self.assertEqual(f.shape, (7, 10 * 4 + 4))
        self.assertTrue(np.isfinite(f).all())

    def test_a_late_change_raises_breadth_and_slope(self):
        calm = np.zeros((1, 16, 10))
        moved = calm.copy()
        moved[0, -2:, :6] = 6.0
        fc, fm = context_features(calm), context_features(moved)
        self.assertGreater(fm[0, 40], fc[0, 40])          # breadth of the last window
        self.assertGreater(fm[0, 30:36].sum(), fc[0, 30:36].sum())   # slope block


class TestHazardModelGuards(unittest.TestCase):
    def test_refuses_to_score_without_a_profile(self):
        with self.assertRaises(RuntimeError):
            HazardModel().raw_score(np.zeros((1, 4, 3)))

    def test_probability_needs_calibration_first(self):
        m = HazardModel(NetworkProfile().fit(np.random.default_rng(0).normal(size=(50, 3))))
        with self.assertRaises(RuntimeError):
            m.probability(np.zeros((1, 4, 3)))


if __name__ == "__main__":
    unittest.main()
