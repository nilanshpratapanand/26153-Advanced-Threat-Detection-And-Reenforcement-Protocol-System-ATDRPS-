"""The evaluation protocol is the thing every later claim rests on, so it is tested
against hand-built timelines whose correct answers are known by construction."""

import unittest

import numpy as np

from atdrps.data.windows import WindowedStates
from atdrps.forecast.protocol import (
    build_onset_samples, evaluate, split_groups, threshold_for_fpr,
)


def _cap(active, valid=None, n_feat=3, window_s=30.0):
    n = len(active)
    X = np.zeros((n, n_feat), dtype=np.float32)
    X[:, 0] = np.arange(n)                       # feature 0 encodes "which window am I"
    return WindowedStates(
        X=X, feature_names=[f"f{i}" for i in range(n_feat)], ts_start=np.arange(n) * window_s,
        window_size_s=window_s, stage=np.zeros(n, dtype=np.int64),
        stage_mask=np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        infiltration=np.asarray(active, dtype=np.int64),
    )


# window index:  0 1 2 3 4 5 6 7 8 9 ...
ACTIVE = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.s = build_onset_samples([_cap(ACTIVE)], context=3, horizon=3, quiet_gap=2)

    def test_no_sample_is_taken_while_an_attack_is_recent(self):
        active = np.array(ACTIVE)
        for t in self.s.t:
            self.assertEqual(active[t - 1:t + 1].sum(), 0, f"window {t} is not quiet")

    def test_label_and_lead_are_correct(self):
        by_t = {int(t): (int(y), int(k)) for t, y, k in zip(self.s.t, self.s.onset_in_k,
                                                            self.s.steps_to_onset)}
        # onset at window 6: samples at t=3,4,5 see it 3,2,1 windows ahead
        self.assertEqual(by_t[5], (1, 1))
        self.assertEqual(by_t[4], (1, 2))
        self.assertEqual(by_t[3], (1, 3))
        self.assertEqual(by_t[2], (0, 0))            # 4 windows away, beyond K=3
        # onset at window 16: samples at t=13,14,15
        self.assertEqual(by_t[15], (1, 1))
        self.assertEqual(by_t[13], (1, 3))

    def test_samples_before_the_same_onset_share_an_id(self):
        m6 = {int(t): int(o) for t, o in zip(self.s.t, self.s.onset_id) if t in (3, 4, 5)}
        self.assertEqual(len(set(m6.values())), 1)
        m16 = {int(o) for t, o in zip(self.s.t, self.s.onset_id) if t in (13, 14, 15)}
        self.assertEqual(len(m16), 1)
        self.assertNotEqual(m6[3], next(iter(m16)))
        self.assertEqual(self.s.n_events, 2)

    def test_context_never_contains_the_future(self):
        for ctx, t in zip(self.s.context, self.s.t):
            self.assertEqual(int(ctx[:, 0].max()), int(t))
            self.assertEqual(ctx.shape, (3, 3))

    def test_invalid_labels_never_become_silent_negatives(self):
        valid = np.ones(len(ACTIVE), dtype=bool)
        valid[10:14] = False
        s = build_onset_samples([_cap(ACTIVE, valid)], context=3, horizon=3, quiet_gap=2)
        for t in s.t:
            self.assertTrue(valid[t - 1:t + 4].all())

    def test_groups_are_kept_apart(self):
        s = build_onset_samples([_cap(ACTIVE), _cap(ACTIVE)], context=3, horizon=3, quiet_gap=2)
        self.assertEqual(set(np.unique(s.group)), {0, 1})
        self.assertEqual(s.n_events, 4)


class TestScoring(unittest.TestCase):
    def setUp(self):
        caps = [_cap(ACTIVE) for _ in range(8)]
        self.s = build_onset_samples(caps, context=3, horizon=3, quiet_gap=2)
        self.y = self.s.onset_in_k.astype(float)

    def test_perfect_scorer(self):
        r = evaluate(self.s, self.y, threshold=0.5, n_boot=50)
        self.assertEqual(r["auc"], 1.0)
        self.assertEqual(r["operating"]["event_recall"], 1.0)
        self.assertEqual(r["operating"]["fpr"], 0.0)
        # earliest warning is 3 windows ahead of each onset
        self.assertEqual(r["operating"]["median_lead_s"], 3 * 30.0)

    def test_uninformative_scorer_is_at_chance(self):
        rng = np.random.default_rng(0)
        big = build_onset_samples([_cap(ACTIVE * 20) for _ in range(12)], context=3, horizon=3, quiet_gap=2)
        r = evaluate(big, rng.random(len(big)), fpr_budget=0.1, n_boot=0)
        self.assertLess(abs(r["auc"] - 0.5), 0.1)

    def test_a_late_scorer_earns_a_short_lead(self):
        late = (self.s.steps_to_onset == 1).astype(float)      # only fires 1 window before onset
        r = evaluate(self.s, late, threshold=0.5, n_boot=0)
        self.assertEqual(r["operating"]["event_recall"], 1.0)
        self.assertEqual(r["operating"]["median_lead_s"], 30.0)

    def test_threshold_respects_the_false_positive_budget(self):
        rng = np.random.default_rng(1)
        scores, labels = rng.random(2000), np.zeros(2000, dtype=int)
        for b in (0.1, 0.01, 0.001):
            thr = threshold_for_fpr(scores, labels, b)
            self.assertLessEqual((scores >= thr).mean(), b + 1e-12)

    def test_unresolvable_budget_is_flagged_not_hidden(self):
        r = evaluate(self.s, np.zeros(len(self.s)), threshold=0.5, fpr_budget=1e-4, n_boot=0)
        self.assertFalse(r["budget_resolvable"])
        self.assertAlmostEqual(r["fpr_resolution"], 3.0 / r["n_neg"])

    def test_precision_is_corrected_for_a_realistic_base_rate(self):
        # TPR 0.9, FPR 0.01 -> at prevalence 1e-3 precision must collapse to ~8%
        tpr, fpr, pi = 0.9, 0.01, 1e-3
        expected = tpr * pi / (tpr * pi + fpr * (1 - pi))
        rng = np.random.default_rng(2)
        n = 20000
        s = build_onset_samples([_cap(ACTIVE * 40) for _ in range(20)], context=3, horizon=3, quiet_gap=2)
        y = s.onset_in_k.astype(bool)
        scores = np.where(y, rng.random(len(s)) < tpr, rng.random(len(s)) < fpr).astype(float)
        r = evaluate(s, scores, threshold=0.5, n_boot=0)
        self.assertAlmostEqual(r["operating"]["ppv@0.001"], expected, delta=0.03)
        self.assertLess(r["operating"]["ppv@0.001"], 0.15)

    def test_validation_threshold_is_used_when_given(self):
        rng = np.random.default_rng(3)
        val = build_onset_samples([_cap(ACTIVE * 5) for _ in range(6)], context=3, horizon=3, quiet_gap=2)
        r = evaluate(self.s, rng.random(len(self.s)), val_scores=rng.random(len(val)),
                     val_samples=val, n_boot=0)
        self.assertEqual(r["threshold_source"], "validation")

    def test_split_never_shares_a_capture(self):
        s = build_onset_samples([_cap(ACTIVE) for _ in range(10)], context=3, horizon=3, quiet_gap=2)
        tr, va, te = split_groups(s, 0.2, 0.3, seed=1)
        gs = [set(np.unique(x.group)) for x in (tr, va, te)]
        self.assertFalse(gs[0] & gs[1] or gs[0] & gs[2] or gs[1] & gs[2])
        self.assertEqual(len(tr) + len(va) + len(te), len(s))

    def test_bootstrap_interval_brackets_the_estimate(self):
        r = evaluate(self.s, self.y * 0.9 + 0.05, threshold=0.5, n_boot=100)
        lo, hi = r["ci95"]["auc"]
        self.assertLessEqual(lo, r["auc"] + 1e-9)
        self.assertGreaterEqual(hi, r["auc"] - 1e-9)


if __name__ == "__main__":
    unittest.main()
