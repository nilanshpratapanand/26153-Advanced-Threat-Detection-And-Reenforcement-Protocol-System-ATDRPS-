"""Reference forecasters the hazard model has to beat.

A result that only compares a new model to a weak or missing baseline is one of the
pitfalls Arp et al. (2022) list, so the comparison set is deliberately varied:

* ``PriorBaseline``           -- a constant.  Any real model must beat AUC 0.5.
* ``CusumBaseline``           -- the classic hand-built detector: an EWMA of a few
                                 reconnaissance indicators against a benign baseline.
* ``MahalanobisBaseline``     -- a generic unsupervised anomaly score of the latest window.
* ``IsolationForestBaseline`` -- the other standard unsupervised detector used in the
                                 Kitsune comparison (Mirsky et al., NDSS 2018).
* ``V1Baseline``              -- the deployed ATDRPS v1 world model, unchanged.

Every baseline exposes ``fit(train)`` / ``score(context)`` and returns a higher-is-riskier
score, so the same protocol scores them all.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PriorBaseline", "CusumBaseline", "MahalanobisBaseline",
           "IsolationForestBaseline", "V1Baseline", "PRECURSOR_FEATURES"]

# the signals a human analyst reads as "someone is probing us"
PRECURSOR_FEATURES = (
    "log_distinct_dst_ports", "sequential_port_ratio", "syn_ratio__mean", "half_open_ratio",
    "max_ports_per_src_dst", "new_dst_port_ratio", "log_same_port_fanout", "new_dst_ratio",
    "rst_ratio__mean", "log_max_flows_per_src_dst_port",
)


def _benign(train):
    """Quiet samples that are not followed by an onset: the closest thing to 'normal'."""
    return train.onset_in_k == 0


class PriorBaseline:
    name = "prior (constant)"

    def fit(self, train):
        return self

    def score(self, context):
        return np.zeros(len(context))


class CusumBaseline:
    name = "CUSUM/EWMA on recon indicators"

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def fit(self, train):
        idx = [train.feature_names.index(f) for f in PRECURSOR_FEATURES if f in train.feature_names]
        self.idx = idx
        last = train.context[_benign(train)][:, :, idx].reshape(-1, len(idx))
        self.mu = last.mean(0)
        self.sd = last.std(0) + 1e-6
        return self

    def score(self, context):
        z = (context[:, :, self.idx] - self.mu) / self.sd            # (N, L, k)
        ewma = np.zeros((len(context), len(self.idx)))
        for step in range(context.shape[1]):
            ewma = self.alpha * z[:, step] + (1 - self.alpha) * ewma
        return ewma.max(axis=1)


class MahalanobisBaseline:
    name = "Mahalanobis anomaly (last window)"

    def fit(self, train):
        from sklearn.covariance import LedoitWolf
        x = train.context[_benign(train)][:, -1, :]
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        self.cov = LedoitWolf().fit((x - self.mu) / self.sd)
        return self

    def score(self, context):
        z = (context[:, -1, :] - self.mu) / self.sd
        return self.cov.mahalanobis(z)


class IsolationForestBaseline:
    name = "Isolation Forest (last window)"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, train):
        from sklearn.ensemble import IsolationForest
        x = train.context[_benign(train)][:, -1, :]
        self.model = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=1).fit(x)
        return self

    def score(self, context):
        return -self.model.score_samples(context[:, -1, :])


class V1Baseline:
    """The v1 world model, trained the way v1 trained it (on all windows, next-window label)."""

    name = "ATDRPS v1 world model"

    def __init__(self, context: int, horizon: int):
        self.context, self.horizon = context, horizon

    def fit_captures(self, captures):
        from ..models.numpy_dynamics import NumpyDynamicsWorldModel
        from ..train.dataset import build_sequences
        ds = build_sequences(captures, context=self.context, horizon=self.horizon)
        n = len(ds)
        cut = int(n * 0.85)
        train, val = ds.subset(np.arange(cut)), ds.subset(np.arange(cut, n))
        self.model = NumpyDynamicsWorldModel(ds.feature_names, self.context, self.horizon)
        self.model.fit(train, val_dataset=val, verbose=False)
        return self

    def fit(self, train):                    # pragma: no cover - use fit_captures
        raise NotImplementedError("V1Baseline trains from captures: call fit_captures")

    def score(self, context):
        _, infil = self.model.heads_batch(context)
        return np.asarray(infil, dtype=float)
