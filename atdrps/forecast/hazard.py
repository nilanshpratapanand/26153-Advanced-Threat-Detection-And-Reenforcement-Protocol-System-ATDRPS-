"""The v2 forecaster: a calibrated hazard of infiltration onset, relative to *this* network.

Design choices, each traced to something measured or published:

* **Scores are relative to a per-network profile.**  The v1 model saw real traffic as far
  outside its training range (median |z| 1.64 vs 0.34) and saturated.  The profile is the
  robust centre and spread of each feature on that network's own benign windows, so a
  keepalive that is normal *here* stops looking like a beacon.
* **Features describe change, not level:** the newest window, an EWMA, the recent maximum,
  the slope over the context, and the *breadth* of anomaly (how many features are unusual
  at once) -- a scan moves many features a little; a busy afternoon moves a few a lot.
* **Gradient-boosted trees** (scikit-learn ``HistGradientBoostingClassifier``), strongly
  regularised: the training set has tens of independent onsets, not thousands.
* **Calibration on held-out data from the target network.**  Isotonic regression maps scores
  to probabilities; the alert threshold is a benign-score quantile at a stated
  false-alarm budget (a split-conformal rule: valid under exchangeability of the calibration
  and deployment windows, which real traffic only approximately satisfies).
"""

from __future__ import annotations

import numpy as np

__all__ = ["NetworkProfile", "context_features", "HazardModel"]

_Z_CLIP = 10.0


class NetworkProfile:
    """Robust per-feature centre and scale of a network's benign windows."""

    def __init__(self):
        self.centre = None
        self.scale = None
        self.n_windows = 0

    def fit(self, windows: np.ndarray) -> "NetworkProfile":
        x = np.asarray(windows, dtype=np.float64).reshape(-1, windows.shape[-1])
        if len(x) < 8:
            raise ValueError(f"a profile needs at least 8 benign windows, got {len(x)}")
        centre = np.median(x, axis=0)
        mad = 1.4826 * np.median(np.abs(x - centre), axis=0)
        std = x.std(axis=0)
        # floor keeps a feature that is constant on this network from turning any change into z=1e6
        self.scale = np.maximum(np.maximum(mad, 0.5 * std), 1e-3 + 0.01 * np.abs(centre))
        self.centre = centre
        self.n_windows = len(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.centre) / self.scale
        return np.clip(z, -_Z_CLIP, _Z_CLIP)

    def to_dict(self) -> dict:
        return {"centre": self.centre.tolist(), "scale": self.scale.tolist(),
                "n_windows": self.n_windows}

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkProfile":
        p = cls()
        p.centre, p.scale = np.asarray(d["centre"]), np.asarray(d["scale"])
        p.n_windows = int(d["n_windows"])
        return p


def context_features(z: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """(N, L, F) profile-relative windows -> (N, D) features describing the recent change."""
    z = np.asarray(z, dtype=np.float64)
    n, length, f = z.shape
    ewma = np.zeros((n, f))
    for step in range(length):
        ewma = alpha * z[:, step] + (1 - alpha) * ewma
    last = z[:, -1]
    recent_max = z[:, -min(3, length):].max(axis=1)
    ref = z[:, -min(8, length)]
    slope = last - ref
    a_last = (np.abs(last) > 3).sum(axis=1, keepdims=True)
    a_ewma = (np.abs(ewma) > 3).sum(axis=1, keepdims=True)
    a_mean = np.abs(last).mean(axis=1, keepdims=True)
    a_hist = np.abs(z).mean(axis=(1, 2)).reshape(-1, 1)
    return np.hstack([last, ewma, recent_max, slope, a_last, a_ewma, a_mean, a_hist]).astype(np.float32)


class HazardModel:
    name = "ATDRPS v2 hazard (profile-relative GBT)"

    def __init__(self, profile: NetworkProfile | None = None, seed: int = 0,
                 max_iter: int = 150, learning_rate: float = 0.05, max_depth: int = 3,
                 l2: float = 5.0):
        self.profile = profile
        self.seed = seed
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
                           l2_regularization=l2)
        self.clf = None
        self.calibrator = None

    def _features(self, context):
        if self.profile is None:
            raise RuntimeError("HazardModel has no NetworkProfile: fit one on benign windows first")
        return context_features(self.profile.transform(context))

    def fit(self, train, val=None) -> "HazardModel":
        from sklearn.ensemble import HistGradientBoostingClassifier
        x, y = self._features(train.context), train.onset_in_k.astype(int)
        self.clf = HistGradientBoostingClassifier(
            random_state=self.seed, min_samples_leaf=20, class_weight="balanced", **self.params)
        self.clf.fit(x, y)
        if val is not None and len(np.unique(val.onset_in_k)) == 2:
            self.calibrate(val)
        return self

    def raw_score(self, context) -> np.ndarray:
        features = self._features(context)          # raises a clear error if no profile
        if self.clf is None:
            raise RuntimeError("HazardModel is not fitted: call fit() first")
        return self.clf.predict_proba(features)[:, 1]

    def calibrate(self, val) -> "HazardModel":
        from sklearn.isotonic import IsotonicRegression
        s, y = self.raw_score(val.context), val.onset_in_k.astype(int)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
        return self

    def score(self, context) -> np.ndarray:
        """Ranking score (uncalibrated).  Isotonic calibration collapses scores into ties,
        which would blunt AUC and threshold resolution, so ranking uses this."""
        return self.raw_score(context)

    def probability(self, context) -> np.ndarray:
        """Calibrated P(onset within K windows), if a calibrator has been fitted."""
        if self.calibrator is None:
            raise RuntimeError("call calibrate(val) first")
        return self.calibrator.predict(self.raw_score(context))
