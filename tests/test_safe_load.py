"""Model files are code-execution vectors unless loading is restricted."""

import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atdrps.models.safe_pickle import UnsafePickleError, safe_load


class _Evil:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


class TestSafePickle(unittest.TestCase):
    def test_allowlisted_model_objects_round_trip(self):
        from sklearn.linear_model import LogisticRegression, Ridge

        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(40, 3)), np.arange(40) % 2
        blob = {"a": Ridge().fit(X, X[:, 0]), "b": LogisticRegression().fit(X, y),
                "arr": np.arange(6.0).reshape(2, 3), "scalar": np.float64(1.5), "n": 3}
        import io
        out = safe_load(io.BytesIO(pickle.dumps(blob)))
        np.testing.assert_allclose(out["arr"], blob["arr"])
        np.testing.assert_allclose(out["b"].predict_proba(X), blob["b"].predict_proba(X))
        self.assertEqual(out["scalar"], 1.5)

    def test_os_system_payload_is_refused_and_never_runs(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "pwned")
            with self.assertRaises(UnsafePickleError):
                safe_load(io.BytesIO(pickle.dumps(_Evil(marker))))
            self.assertFalse(os.path.exists(marker), "payload executed")

    def test_arbitrary_callables_are_refused(self):
        import io
        for obj in (eval, __import__, open):
            with self.subTest(obj=getattr(obj, "__name__", obj)):
                class R:
                    def __reduce__(self, obj=obj):
                        return (obj, ("1+1",))
                with self.assertRaises(UnsafePickleError):
                    safe_load(io.BytesIO(pickle.dumps(R())))

    def test_model_loaders_use_the_restricted_unpickler(self):
        from atdrps.models.baseline import LogisticBaseline
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "pwned")
            path = Path(tmp) / "baseline.pkl"
            path.write_bytes(pickle.dumps(_Evil(marker)))
            with self.assertRaises(UnsafePickleError):
                LogisticBaseline.load(path)
            self.assertFalse(os.path.exists(marker))

    def test_numpy_dynamics_loader_refuses_a_tampered_model(self):
        from atdrps.models.numpy_dynamics import NumpyDynamicsWorldModel
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "pwned")
            model = NumpyDynamicsWorldModel(["f0", "f1"], context=4, horizon=2)
            model.save(tmp)
            Path(tmp, "model.pkl").write_bytes(pickle.dumps(_Evil(marker)))
            with self.assertRaises(UnsafePickleError):
                NumpyDynamicsWorldModel.load(tmp)
            self.assertFalse(os.path.exists(marker))


if __name__ == "__main__":
    unittest.main()
