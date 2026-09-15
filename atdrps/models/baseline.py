"""The mandated Logistic Regression baseline -- built to be hard to beat.

The problem statement asks for a logistic-regression comparison on the same
features.  A baseline is only worth reporting if it was given a fair chance, so
this one gets two forms:

``static``
    Logistic regression on the **current window only**.  This is the classifier
    the problem statement criticises: it sees a snapshot and has no access to
    how the network got there.

``context``
    Logistic regression on the **flattened L-window context** -- exactly the
    same input tensor the world model receives.  It therefore has the temporal
    information available to it; what it lacks is any mechanism for representing
    dynamics beyond a linear map.

Both are trained *per horizon step*: a separate model for "one window ahead",
"two windows ahead" and so on.  That is the strongest form of a direct
multi-horizon baseline and means the K-step comparison is like for like rather
than a forecaster being compared against something that can only answer for the
next window.

If the world model cannot beat the ``context`` variant, the honest conclusion is
that the architecture is not earning its complexity on this data, and the report
should say so.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

__all__ = ["LogisticBaseline"]


class LogisticBaseline:
    def __init__(self, mode: str = "context", C: float = 0.5, max_iter: int = 500,
                 class_weight: str | None = "balanced") -> None:
        if mode not in ("static", "context"):
            raise ValueError("mode must be 'static' or 'context'")
        self.mode = mode
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.stage_models: list[LogisticRegression | None] = []
        self.infil_models: list[LogisticRegression | None] = []
        self.n_stages = 0
        self.horizon = 0

    @property
    def name(self) -> str:
        return f"logistic-regression ({self.mode})"

    # ------------------------------------------------------------- features
    def _design(self, context: np.ndarray) -> np.ndarray:
        arr = np.asarray(context, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[None, ...]
        block = arr[:, -1, :] if self.mode == "static" else arr.reshape(arr.shape[0], -1)
        if self.mean is None:
            self.mean = block.mean(axis=0)
            std = block.std(axis=0)
            self.scale = np.where(std < 1e-8, 1.0, std)
        return (block - self.mean) / self.scale

    # ------------------------------------------------------------- training
    def fit(self, dataset, n_stages: int) -> "LogisticBaseline":
        self.n_stages = int(n_stages)
        self.horizon = int(dataset.horizon)
        self.mean = None                      # refit scaling on this split
        X = self._design(dataset.context)

        self.stage_models, self.infil_models = [], []
        for k in range(self.horizon):
            mask = dataset.target_mask[:, k]
            stage_y = dataset.target_stage[:, k][mask]
            infil_y = dataset.target_infil[:, k][mask]
            Xk = X[mask]

            if Xk.shape[0] == 0 or np.unique(stage_y).size < 2:
                self.stage_models.append(None)
            else:
                model = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                           class_weight=self.class_weight)
                model.fit(Xk, stage_y)
                self.stage_models.append(model)

            if Xk.shape[0] == 0 or np.unique(infil_y).size < 2:
                self.infil_models.append(None)
            else:
                model = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                           class_weight=self.class_weight)
                model.fit(Xk, infil_y)
                self.infil_models.append(model)
        return self

    # ------------------------------------------------------------ inference
    def predict(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        """-> stage probabilities ``(N, K, S)`` and infiltration ``(N, K)``."""
        X = self._design(dataset.context)
        n = X.shape[0]
        stage = np.zeros((n, self.horizon, self.n_stages), dtype=np.float32)
        infil = np.zeros((n, self.horizon), dtype=np.float32)
        for k in range(self.horizon):
            model = self.stage_models[k] if k < len(self.stage_models) else None
            if model is None:
                stage[:, k, 0] = 1.0
            else:
                raw = model.predict_proba(X)
                for j, cls in enumerate(model.classes_):
                    stage[:, k, int(cls)] = raw[:, j]
            imodel = self.infil_models[k] if k < len(self.infil_models) else None
            if imodel is not None:
                infil[:, k] = imodel.predict_proba(X)[:, 1]
            else:
                infil[:, k] = stage[:, k, 2:].sum(axis=1)
        return stage, infil

    # --------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "LogisticBaseline":
        with open(Path(path), "rb") as fh:
            return pickle.load(fh)
