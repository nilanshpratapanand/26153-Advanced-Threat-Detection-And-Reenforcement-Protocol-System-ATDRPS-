"""Tests for the temporal transformer backend.

The heavy tests need PyTorch and skip cleanly without it.  The tensor-shape
tests do not: NumPy's ``reshape``/``transpose`` semantics are identical to
PyTorch's, so the reshape-and-permute logic inside multi-head attention -- by
some distance the easiest thing to get silently wrong in a transformer -- is
verified here in any environment, against attention computed the slow, obvious
way.
"""

import math
import unittest

import numpy as np

from atdrps.models.transformer import (
    TemporalTransformerWorldModel, require_torch, torch_available,
)

HAS_TORCH = torch_available()


class TestAttentionShapeLogic(unittest.TestCase):
    """Mirrors of the exact reshape/permute sequence in MultiHeadSelfAttention."""

    def test_qkv_split_selects_the_right_thirds(self):
        B, L, D, H = 2, 5, 8, 2
        d_head = D // H
        rng = np.random.default_rng(0)
        qkv = rng.normal(size=(B, L, 3 * D))

        # the module does: reshape(B, L, 3, H, d_head).permute(2, 0, 3, 1, 4)
        stacked = qkv.reshape(B, L, 3, H, d_head)
        q, k, v = np.transpose(stacked, (2, 0, 3, 1, 4))

        self.assertEqual(q.shape, (B, H, L, d_head))
        self.assertEqual(k.shape, (B, H, L, d_head))
        self.assertEqual(v.shape, (B, H, L, d_head))

        # q must be the first D projected columns, split across heads
        expected_q = qkv[:, :, :D].reshape(B, L, H, d_head).transpose(0, 2, 1, 3)
        expected_k = qkv[:, :, D:2 * D].reshape(B, L, H, d_head).transpose(0, 2, 1, 3)
        expected_v = qkv[:, :, 2 * D:].reshape(B, L, H, d_head).transpose(0, 2, 1, 3)
        self.assertTrue(np.allclose(q, expected_q))
        self.assertTrue(np.allclose(k, expected_k))
        self.assertTrue(np.allclose(v, expected_v))

    def test_head_merge_is_the_inverse_of_the_split(self):
        B, L, D, H = 3, 4, 12, 3
        d_head = D // H
        rng = np.random.default_rng(1)
        x = rng.normal(size=(B, L, D))
        split = x.reshape(B, L, H, d_head).transpose(0, 2, 1, 3)   # (B, H, L, d_head)
        merged = split.transpose(0, 2, 1, 3).reshape(B, L, D)      # the module's merge
        self.assertTrue(np.allclose(merged, x))

    def test_attention_matches_the_naive_computation(self):
        B, L, D, H = 1, 4, 6, 2
        d_head = D // H
        rng = np.random.default_rng(2)
        qkv = rng.normal(size=(B, L, 3 * D))
        stacked = qkv.reshape(B, L, 3, H, d_head)
        q, k, v = np.transpose(stacked, (2, 0, 3, 1, 4))

        scores = q @ np.swapaxes(k, -2, -1) / math.sqrt(d_head)
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)

        # the same thing, written out position by position
        for b in range(B):
            for h in range(H):
                for i in range(L):
                    raw = [float(np.dot(q[b, h, i], k[b, h, j]) / math.sqrt(d_head))
                           for j in range(L)]
                    mx = max(raw)
                    exp = [math.exp(r - mx) for r in raw]
                    total = sum(exp)
                    naive = [e / total for e in exp]
                    self.assertTrue(np.allclose(weights[b, h, i], naive, atol=1e-10))

        self.assertTrue(np.allclose(weights.sum(-1), 1.0))
        out = weights @ v
        self.assertEqual(out.shape, (B, H, L, d_head))


class TestConstruction(unittest.TestCase):
    def test_config_records_the_architecture(self):
        model = TemporalTransformerWorldModel(
            [f"f{i}" for i in range(7)], context=8, horizon=3,
            d_model=32, n_heads=4, n_layers=2,
        )
        config = model.config_dict()
        self.assertEqual(config["d_model"], 32)
        self.assertEqual(config["n_heads"], 4)
        self.assertEqual(config["n_layers"], 2)
        self.assertEqual(config["context"], 8)
        self.assertEqual(config["name"], "temporal-transformer")
        self.assertEqual(len(config["feature_names"]), 7)

    @unittest.skipIf(HAS_TORCH, "torch is installed, so the guidance path is not taken")
    def test_missing_torch_gives_actionable_guidance(self):
        with self.assertRaises(ImportError) as ctx:
            require_torch()
        message = str(ctx.exception)
        self.assertIn("pip install torch", message)
        self.assertIn("model.backend=numpy", message)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchBackend(unittest.TestCase):
    """Runs on the training machine, where torch is present."""

    @staticmethod
    def _dataset(n=96, L=8, F=12, K=3):
        from atdrps.train.dataset import SequenceDataset
        rng = np.random.default_rng(0)
        # a state whose next value genuinely depends on the trend of the context,
        # so a model that learns nothing cannot score well by accident
        context = rng.normal(size=(n, L, F)).astype(np.float32)
        target_state = (context[:, -1, :] + 0.5 * (context[:, -1, :] - context[:, -2, :]))
        stage = (context[:, -1, 0] > 0).astype(np.int64)
        return SequenceDataset(
            context=context, target_state=target_state.astype(np.float32),
            target_stage=np.repeat(stage[:, None], K, axis=1),
            target_infil=np.repeat(stage[:, None], K, axis=1),
            target_mask=np.ones((n, K), dtype=bool),
            ts=np.arange(n, dtype=float), group=np.zeros(n, dtype=np.int64),
            feature_names=[f"f{i}" for i in range(F)],
            stage_names=["Benign", "Reconnaissance"],
        )

    def test_forward_shapes(self):
        ds = self._dataset()
        model = TemporalTransformerWorldModel(
            ds.feature_names, context=ds.context_length, horizon=ds.horizon,
            stage_names=("Benign", "Reconnaissance"), d_model=32, n_heads=4, n_layers=2,
        )
        model.fit(ds, epochs=2, batch_size=16, verbose=False, patience=0)
        state = model.predict_next(ds.context[0])
        probs, infil = model.heads(ds.context[0])
        self.assertEqual(state.shape, (ds.n_features,))
        self.assertEqual(probs.shape, (2,))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertGreaterEqual(infil, 0.0)
        self.assertLessEqual(infil, 1.0)

    def test_training_reduces_loss(self):
        ds = self._dataset()
        model = TemporalTransformerWorldModel(
            ds.feature_names, context=ds.context_length, horizon=ds.horizon,
            stage_names=("Benign", "Reconnaissance"), d_model=32, n_heads=4, n_layers=2,
        )
        model.fit(ds, epochs=12, batch_size=16, verbose=False, patience=0)
        self.assertLess(model.history[-1]["loss"], model.history[0]["loss"])

    def test_attention_is_a_distribution_over_the_context(self):
        ds = self._dataset()
        model = TemporalTransformerWorldModel(
            ds.feature_names, context=ds.context_length, horizon=ds.horizon,
            stage_names=("Benign", "Reconnaissance"), d_model=32, n_heads=4, n_layers=2,
        )
        model.fit(ds, epochs=2, batch_size=16, verbose=False, patience=0)
        attn = model.attention(ds.context[0])
        self.assertEqual(attn.shape, (ds.context_length,))
        self.assertAlmostEqual(float(attn.sum()), 1.0, places=4)
        self.assertTrue((attn >= 0).all())
        layers = model.attention_all_layers(ds.context[0])
        self.assertEqual(layers.shape, (2, ds.context_length))

    def test_rollout_shapes_and_probabilities(self):
        ds = self._dataset()
        model = TemporalTransformerWorldModel(
            ds.feature_names, context=ds.context_length, horizon=4,
            stage_names=("Benign", "Reconnaissance"), d_model=32, n_heads=4, n_layers=2,
        )
        model.fit(ds, epochs=2, batch_size=16, verbose=False, patience=0)
        forecast = model.rollout(ds.context[0], horizon=4)
        self.assertEqual(forecast.states.shape, (4, ds.n_features))
        self.assertEqual(forecast.stage_probs.shape, (4, 2))
        self.assertEqual(forecast.infiltration.shape, (4,))
        self.assertTrue(((forecast.infiltration >= 0) & (forecast.infiltration <= 1)).all())
        self.assertEqual(len(forecast.timeline()), 4)

    def test_checkpoint_roundtrip(self):
        import tempfile
        ds = self._dataset()
        model = TemporalTransformerWorldModel(
            ds.feature_names, context=ds.context_length, horizon=3,
            stage_names=("Benign", "Reconnaissance"), d_model=32, n_heads=4, n_layers=2,
        )
        model.fit(ds, epochs=2, batch_size=16, verbose=False, patience=0)
        before = model.predict_next(ds.context[0])
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            reloaded = TemporalTransformerWorldModel.load(tmp)
        after = reloaded.predict_next(ds.context[0])
        self.assertTrue(np.allclose(before, after, atol=1e-5))

    def test_wrong_context_length_is_rejected(self):
        ds = self._dataset()
        model = TemporalTransformerWorldModel(
            ds.feature_names, context=ds.context_length, horizon=3,
            stage_names=("Benign", "Reconnaissance"), d_model=32, n_heads=4, n_layers=2,
        )
        model.fit(ds, epochs=1, batch_size=16, verbose=False, patience=0)
        # rollout pads/truncates, but a direct forward must not silently accept
        with self.assertRaises(ValueError):
            model.net(require_torch().zeros(1, ds.context_length + 3, ds.n_features))


if __name__ == "__main__":
    unittest.main()
