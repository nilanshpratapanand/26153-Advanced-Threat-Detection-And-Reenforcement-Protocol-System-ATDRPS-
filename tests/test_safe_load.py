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


class TestAllowlistCoversTheInstalledScikitLearn(unittest.TestCase):
    """The allowlist names scikit-learn internals, which move between releases.  This fits the
    exact objects a saved HazardModel contains under whatever version is installed and checks
    every global they need is allowed, so an upgrade fails here -- loudly, in CI -- and not in a
    user's hands."""

    def test_hazard_model_objects_only_need_allowlisted_globals(self):
        import io
        import warnings
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.isotonic import IsotonicRegression
        from atdrps.models.safe_pickle import HAZARD_ALLOWED, safe_load
        warnings.simplefilter("ignore")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 5))
        y = (X[:, 0] + rng.normal(size=200) > 0.8).astype(int)
        clf = HistGradientBoostingClassifier(max_iter=10, random_state=0, class_weight="balanced").fit(X, y)
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(clf.predict_proba(X)[:, 1], y)
        blob = pickle.dumps({"clf": clf, "calibrator": cal})
        requested = []

        class Recorder(pickle.Unpickler):
            def find_class(self, module, name):
                requested.append((module, name))
                return super().find_class(module, name)

        Recorder(io.BytesIO(blob)).load()
        missing = sorted(set(requested) - HAZARD_ALLOWED)
        self.assertEqual(missing, [], f"add these to HAZARD_ALLOWED after reviewing them: {missing}")
        out = safe_load(io.BytesIO(blob), extra=HAZARD_ALLOWED)      # and it really loads
        np.testing.assert_allclose(out["clf"].predict_proba(X), clf.predict_proba(X))


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
