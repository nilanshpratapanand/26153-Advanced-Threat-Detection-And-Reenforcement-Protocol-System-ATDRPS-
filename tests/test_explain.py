"""Explainability tests.

The SHAP implementation is checked against closed-form answers rather than
against itself: for a linear model the Shapley value of feature i is exactly
``w_i * (x_i - E[background_i])``, so there is a right answer to compare to.
"""

import unittest

import numpy as np

from atdrps.explain.explainer import ForecastExplainer
from atdrps.explain.glossary import describe_feature
from atdrps.explain.kernel_shap import kernel_shap, shap_kernel_weights


class TestKernelWeights(unittest.TestCase):
    def test_endpoints_are_zero(self):
        w = shap_kernel_weights(10, np.array([0, 10]))
        self.assertTrue(np.all(w == 0.0))

    def test_symmetric_in_coalition_size(self):
        w = shap_kernel_weights(10, np.arange(1, 10))
        self.assertTrue(np.allclose(w, w[::-1]))

    def test_mass_concentrates_at_the_extremes(self):
        w = shap_kernel_weights(20, np.arange(1, 20))
        self.assertGreater(w[0], w[len(w) // 2])

    def test_large_m_does_not_overflow(self):
        # C(96, 48) overflows a float; the log-space path must survive it
        w = shap_kernel_weights(96, np.arange(1, 96))
        self.assertTrue(np.isfinite(w).all())
        self.assertGreater(w.sum(), 0.0)


class TestKernelShapCorrectness(unittest.TestCase):
    def test_matches_closed_form_for_a_linear_model(self):
        rng = np.random.default_rng(0)
        d = 8
        w = rng.normal(size=d)
        background = rng.normal(size=(60, d))
        x = rng.normal(size=d)

        result = kernel_shap(lambda X: X @ w, x, background, n_samples=600, seed=1)
        exact = w * (x - background.mean(axis=0))
        self.assertLess(float(np.abs(result.values - exact).max()), 1e-3)

    def test_efficiency_holds_exactly(self):
        rng = np.random.default_rng(2)
        d = 6
        background = rng.normal(size=(40, d))
        x = rng.normal(size=d)

        def f(X):
            return X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + X[:, 3] ** 2

        result = kernel_shap(f, x, background, n_samples=400, seed=3)
        self.assertLess(result.efficiency_error, 1e-8)
        self.assertAlmostEqual(
            float(result.values.sum() + result.base_value), result.prediction, places=6
        )

    def test_irrelevant_features_get_no_credit(self):
        rng = np.random.default_rng(4)
        d = 6
        background = rng.normal(size=(50, d))
        x = rng.normal(size=d)
        result = kernel_shap(lambda X: 3.0 * X[:, 1], x, background, n_samples=500, seed=5)
        relevant = abs(result.values[1])
        others = np.abs(np.delete(result.values, 1)).max()
        self.assertGreater(relevant, 10 * max(others, 1e-9))

    def test_constant_model_gives_zero_attribution(self):
        background = np.random.default_rng(6).normal(size=(30, 5))
        x = np.zeros(5)
        result = kernel_shap(lambda X: np.full(X.shape[0], 2.5), x, background,
                             n_samples=200, seed=7)
        self.assertLess(float(np.abs(result.values).max()), 1e-6)

    def test_top_is_ordered_by_magnitude(self):
        rng = np.random.default_rng(8)
        d = 7
        w = np.array([5.0, -4.0, 0.1, 0.0, 3.0, -0.2, 0.05])
        background = rng.normal(size=(40, d))
        x = np.ones(d)
        result = kernel_shap(lambda X: X @ w, x, background, n_samples=500, seed=9)
        names = [n for n, _ in result.top(3)]
        self.assertEqual(set(names), {"f0", "f1", "f4"})

    def test_rejects_mismatched_background(self):
        with self.assertRaises(ValueError):
            kernel_shap(lambda X: X.sum(1), np.zeros(4), np.zeros((5, 6)))


class TestGlossary(unittest.TestCase):
    def test_aggregate_suffixes_are_expanded(self):
        self.assertEqual(describe_feature("syn_ratio__std"), "variability of SYN flag ratio")
        self.assertEqual(describe_feature("payload_mean__max"), "peak payload size")

    def test_detectors_have_plain_english(self):
        self.assertIn("metronomic", describe_feature("beacon_regularity"))
        self.assertIn("ports", describe_feature("max_ports_per_src_dst"))

    def test_unknown_feature_degrades_gracefully(self):
        self.assertEqual(describe_feature("some_new_thing"), "some new thing")

    def test_every_state_feature_has_a_description(self):
        from atdrps.data.windows import state_feature_names
        for name in state_feature_names():
            described = describe_feature(name)
            self.assertTrue(described)
            # a description that is just the raw name means it was missed
            self.assertNotEqual(described, name)


class _ToyModel:
    """A tiny WorldModel-shaped object: infiltration depends on feature 2 only."""

    feature_names = [f"f{i}" for i in range(5)]
    stage_names = ["Benign", "Reconnaissance"]
    context = 4

    def heads_batch(self, contexts):
        arr = np.asarray(contexts, dtype=np.float64)
        score = 1.0 / (1.0 + np.exp(-arr[:, -1, 2]))
        probs = np.stack([1 - score, score], axis=1)
        return probs.astype(np.float32), score.astype(np.float32)

    def heads(self, context):
        p, s = self.heads_batch(np.asarray(context)[None, ...])
        return p[0], float(s[0])


class TestForecastExplainer(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.background = rng.normal(size=(24, 4, 5)).astype(np.float32)
        self.model = _ToyModel()

    def test_identifies_the_only_feature_that_matters(self):
        ex = ForecastExplainer(self.model, self.background, top_k=3, n_samples=200,
                               n_background=12)
        context = np.zeros((4, 5), dtype=np.float32)
        context[-1, 2] = 4.0
        explanation = ex.explain(context, target="infiltration")
        self.assertEqual(explanation.drivers(1)[0]["feature"], "f2")
        self.assertGreater(explanation.drivers(1)[0]["contribution"], 0.0)

    def test_attributions_satisfy_efficiency(self):
        ex = ForecastExplainer(self.model, self.background, n_samples=200, n_background=12)
        context = np.zeros((4, 5), dtype=np.float32)
        context[-1, 2] = 2.0
        self.assertLess(ex.explain(context).efficiency_error, 1e-8)

    def test_temporal_attribution_is_a_distribution(self):
        ex = ForecastExplainer(self.model, self.background, n_samples=120, n_background=8)
        context = np.zeros((4, 5), dtype=np.float32)
        context[-1, 2] = 3.0
        weights, source = ex.temporal_attribution(context)
        self.assertEqual(weights.shape, (4,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertEqual(source, "occlusion")

    def test_occlusion_finds_the_window_that_matters(self):
        """Only the most recent window feeds this toy model's output."""
        ex = ForecastExplainer(self.model, self.background, n_samples=120, n_background=8)
        context = np.zeros((4, 5), dtype=np.float32)
        context[-1, 2] = 3.0
        weights, _ = ex.temporal_attribution(context)
        self.assertEqual(int(weights.argmax()), 3)

    def test_text_report_is_human_readable(self):
        ex = ForecastExplainer(self.model, self.background, top_k=3, n_samples=150,
                               n_background=8)
        context = np.zeros((4, 5), dtype=np.float32)
        context[-1, 2] = 3.0
        text = ex.explain(context).to_text()
        self.assertIn("infiltration", text)
        self.assertIn("Driven by", text)
        self.assertIn("Most influential time steps", text)

    def test_rejects_bad_background_shape(self):
        with self.assertRaises(ValueError):
            ForecastExplainer(self.model, np.zeros((10, 5)))


if __name__ == "__main__":
    unittest.main()


class TestSaturationLink(unittest.TestCase):
    """Attribution must stay informative when the model is confident.

    Explaining in probability space fails exactly when an analyst most needs
    the answer. Once a model sits at p = 0.99 the sigmoid is flat, so masking
    any single feature barely moves the output, marginal contributions all
    compress toward the same small number, and the efficiency constraint then
    splits the gap almost evenly. A trained transformer on the demo capture
    returned eight "top drivers" every one of which was +0.036 -- arithmetically
    correct, analytically worthless.
    """

    class _SaturatingModel:
        """A model whose probability head is pinned hard against 1.0.

        The logit still carries the ranking; the probability has thrown it away.
        """

        context = 4
        stage_names = ["Benign", "Reconnaissance"]

        def __init__(self, n_features=6):
            self.feature_names = [f"f{i}" for i in range(n_features)]
            # geometrically spaced weights: a clear, checkable ordering
            self._w = np.array([2.0 ** (-i) for i in range(n_features)])

        def heads_batch(self, contexts):
            contexts = np.asarray(contexts, dtype=np.float64)
            z = contexts.mean(axis=1) @ self._w + 12.0      # +12 => deep saturation
            infil = 1.0 / (1.0 + np.exp(-z))
            probs = np.zeros((contexts.shape[0], 2))
            probs[:, 1] = infil
            probs[:, 0] = 1.0 - infil
            return probs, infil

    def _explain(self, link):
        rng = np.random.default_rng(0)
        model = self._SaturatingModel()
        background = rng.normal(size=(12, model.context, len(model.feature_names)))
        context = np.abs(rng.normal(size=(model.context, len(model.feature_names)))) + 1.0
        explainer = ForecastExplainer(model, background, top_k=6, n_samples=256,
                                      n_background=12, link=link)
        return explainer.explain(context, target="infiltration")

    def test_the_model_really_is_saturated(self):
        """Guard the premise: if this stops saturating the test proves nothing."""
        ex = self._explain("probability")
        self.assertGreater(ex.prediction, 0.999)

    def test_both_links_still_separate_drivers_on_this_model(self):
        """Records what was actually measured, not what was hoped for.

        The logit link was added expecting it to rescue a saturated model from
        returning near-identical contributions. On this construction it does
        not: probability space separates the drivers perfectly well at
        p > 0.999, and by relative spread it separates them slightly *better*.
        The real transformer degeneracy could not be reproduced without
        PyTorch, so this asserts the measured fact and leaves the hypothesis
        open rather than encoding a claim that was not demonstrated.
        """
        for link in ("probability", "logit"):
            vals = [abs(d["contribution"]) for d in self._explain(link).drivers()]
            self.assertGreater(len(set(round(v, 6) for v in vals)), 1,
                               f"{link}: contributions collapsed to a single value")

    def test_logit_ranks_the_genuinely_dominant_feature_first(self):
        ex = self._explain("logit")
        self.assertEqual(ex.drivers()[0]["feature"], "f0",
                         "f0 carries the largest weight by construction")

    def test_efficiency_still_holds_in_logit_space(self):
        self.assertLess(self._explain("logit").efficiency_error, 1e-6)

    def test_rejects_an_unknown_link(self):
        with self.assertRaises(ValueError):
            ForecastExplainer(self._SaturatingModel(), np.zeros((4, 4, 6)), link="probit")
