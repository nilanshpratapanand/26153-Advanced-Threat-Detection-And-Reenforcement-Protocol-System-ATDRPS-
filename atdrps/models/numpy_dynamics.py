"""A linear-dynamics world model: ridge dynamics plus logistic heads.

This is the ablation that keeps the project honest.  If a temporal transformer
cannot beat a well-specified linear model on the same features, the same splits
and the same rollout code, then the transformer is decoration and the report
should say so.  Building the comparison in from the start -- rather than
bolting on a weak baseline at the end to flatter the headline number -- is the
difference between a benchmark and a marketing table.

It earns its place twice over: it has no dependency beyond NumPy and
scikit-learn, so the whole pipeline (windowing, rollout, explainability,
dashboard) stays runnable and testable in environments where PyTorch is not
installable.

Structure, deliberately mirroring the transformer:

* **dynamics** -- ridge regression from the flattened context to the *change*
  in state, learned in standardised space
* **stage head** -- multinomial logistic regression over MITRE stages of the
  *next* window
* **infiltration head** -- binary logistic regression

Forward simulation is inherited from :class:`~atdrps.models.base.WorldModel`,
so both backends roll out through identical code.

Why the dynamics head predicts a *delta*
----------------------------------------
The first version regressed the absolute next state. Ridge shrinks its
coefficients toward zero, so a strongly regularised model predicts something
close to the *mean* state -- and feeding a mean state back into the context
step after step produced forecasts that oscillated 1.00, 0.00, 0.00, 0.99,
1.00 across five steps, with the stage label contradicting the probability.

Predicting the change instead makes shrinkage mean "nothing changes", which is
the correct prior for network state and is exactly the persistence baseline.
The roll-out then degrades gracefully toward "no further change" rather than
lurching between arbitrary points in state space.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from ..data.schema import STAGES
from .base import Forecast, WorldModel

__all__ = ["NumpyDynamicsWorldModel", "PersistenceModel"]


class NumpyDynamicsWorldModel(WorldModel):
    name = "numpy-linear-dynamics"

    def __init__(self, feature_names, context=16, horizon=5,
                 stage_names=STAGES, ridge_alpha: float = 1.0,
                 logistic_C: float = 0.5, max_iter: int = 400) -> None:
        super().__init__(feature_names, context, horizon, stage_names)
        self.ridge_alpha = float(ridge_alpha)
        self.logistic_C = float(logistic_C)
        self.max_iter = int(max_iter)
        self.dynamics: Ridge | None = None
        self.stage_clf: LogisticRegression | None = None
        self.infil_clf: LogisticRegression | None = None
        self._stage_classes: np.ndarray | None = None

    # ------------------------------------------------------------- helpers
    def _flatten(self, context: np.ndarray) -> np.ndarray:
        """(N, L, F) standardised -> (N, L*F).

        Keeping every window separate rather than averaging is the point: the
        model has to be able to see that a quantity *rose* across the context,
        which is what a transition looks like.
        """
        arr = np.asarray(context, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr.reshape(arr.shape[0], -1)

    def _prepare(self, context: np.ndarray) -> np.ndarray:
        return self._flatten(self.standardiser.transform(context))

    # ------------------------------------------------------------ training
    def fit(self, dataset, val_dataset=None, class_weight: str | None = "balanced",
            alphas=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
            logistic_Cs=(0.003, 0.01, 0.03, 0.1, 0.5), verbose: bool = False,
            **kwargs) -> "NumpyDynamicsWorldModel":
        if len(dataset) == 0:
            raise ValueError("cannot fit on an empty dataset")

        self.standardiser.fit(dataset.context)
        X = self._prepare(dataset.context)
        # residual target: how the state *changes*, not where it lands
        last = self.standardiser.transform(dataset.context[:, -1, :])
        y_state = self.standardiser.transform(dataset.target_state) - last

        # The flattened context is L x F wide -- 1536 columns at the default
        # settings -- so the ridge penalty is doing real work, not decoration.
        # Left at alpha=1 on a small corpus this model overfits badly enough to
        # lose to "assume nothing changes". The strength is chosen on the
        # validation split, never on test.
        self.dynamics = self._fit_dynamics(X, y_state, val_dataset, alphas, verbose)

        # heads are supervised on the *next* window (index 0 of the horizon),
        # and only where the label is real
        mask = dataset.target_mask[:, 0]
        stage_y = dataset.target_stage[:, 0][mask]
        infil_y = dataset.target_infil[:, 0][mask]
        Xh = X[mask]

        self._stage_classes = np.unique(stage_y)
        # Regularisation strength is chosen on validation log-loss, not accuracy.
        # At C=0.5 with 1536 columns both heads returned 1.00 confidence for
        # every window, which is not a probability, it is an assertion -- and it
        # made the roll-out swing between extremes.
        self.stage_clf = self._fit_head(
            Xh, stage_y, val_dataset, "stage", logistic_Cs, class_weight, verbose
        ) if (Xh.shape[0] and self._stage_classes.size >= 2) else None
        self.infil_clf = self._fit_head(
            Xh, infil_y, val_dataset, "infiltration", logistic_Cs, class_weight, verbose
        ) if (Xh.shape[0] and np.unique(infil_y).size == 2) else None

        self.fitted = True
        return self

    def _fit_head(self, X, y, val_dataset, kind, candidates, class_weight, verbose):
        from sklearn.metrics import log_loss

        candidates = [float(c) for c in (candidates or (self.logistic_C,))]
        if val_dataset is None or len(val_dataset) == 0 or len(candidates) == 1:
            model = LogisticRegression(C=candidates[0], max_iter=self.max_iter,
                                       class_weight=class_weight)
            model.fit(X, y)
            return model

        v_mask = val_dataset.target_mask[:, 0]
        Xv = self._prepare(val_dataset.context)[v_mask]
        yv = (val_dataset.target_stage if kind == "stage" else val_dataset.target_infil)
        yv = yv[:, 0][v_mask]
        best, best_loss = None, float("inf")
        for C in candidates:
            model = LogisticRegression(C=C, max_iter=self.max_iter,
                                       class_weight=class_weight)
            model.fit(X, y)
            try:
                loss = float(log_loss(yv, model.predict_proba(Xv), labels=model.classes_))
            except ValueError:
                continue
            if verbose:
                print(f"    {kind:<13} C={C:<7g} val log-loss={loss:.4f}", flush=True)
            if loss < best_loss:
                best, best_loss = model, loss
        return best if best is not None else LogisticRegression(
            C=candidates[0], max_iter=self.max_iter, class_weight=class_weight).fit(X, y)

    def _fit_dynamics(self, X, y_state, val_dataset, alphas, verbose):
        candidates = [float(a) for a in (alphas or (self.ridge_alpha,))]
        if val_dataset is None or len(val_dataset) == 0 or len(candidates) == 1:
            model = Ridge(alpha=candidates[0])
            model.fit(X, y_state)
            self.ridge_alpha = candidates[0]
            return model

        Xv = self._prepare(val_dataset.context)
        yv = (self.standardiser.transform(val_dataset.target_state)
              - self.standardiser.transform(val_dataset.context[:, -1, :]))
        best, best_mse, best_alpha = None, float("inf"), candidates[0]
        for alpha in candidates:
            model = Ridge(alpha=alpha)
            model.fit(X, y_state)
            mse = float(np.mean((model.predict(Xv) - yv) ** 2))
            if verbose:
                print(f"    ridge alpha={alpha:<9g} val next-state MSE={mse:.4f}", flush=True)
            if mse < best_mse:
                best, best_mse, best_alpha = model, mse, alpha
        self.ridge_alpha = best_alpha
        return best

    # ----------------------------------------------------------- inference
    def predict_next(self, context: np.ndarray) -> np.ndarray:
        return self.predict_next_batch(np.asarray(context)[None, ...])[0].astype(np.float32)

    def predict_next_batch(self, context: np.ndarray) -> np.ndarray:
        if self.dynamics is None:
            raise RuntimeError("model is not fitted")
        arr = np.asarray(context, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        delta = self.dynamics.predict(self._prepare(arr))
        last = self.standardiser.transform(arr[:, -1, :])
        return self.standardiser.inverse(last + delta)

    def heads(self, context: np.ndarray) -> tuple[np.ndarray, float]:
        X = self._prepare(context)
        probs = np.zeros(self.n_stages, dtype=np.float32)
        if self.stage_clf is None:
            probs[0] = 1.0
        else:
            raw = self.stage_clf.predict_proba(X)[0]
            for cls, value in zip(self.stage_clf.classes_, raw):
                probs[int(cls)] = value
        infil = (
            float(self.infil_clf.predict_proba(X)[0, 1])
            if self.infil_clf is not None
            else float(probs[2:].sum())      # fall back to the stage head
        )
        return probs, infil

    def heads_batch(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised :meth:`heads`, for evaluating a whole test split."""
        X = self._prepare(context)
        n = X.shape[0]
        probs = np.zeros((n, self.n_stages), dtype=np.float32)
        if self.stage_clf is None:
            probs[:, 0] = 1.0
        else:
            raw = self.stage_clf.predict_proba(X)
            for j, cls in enumerate(self.stage_clf.classes_):
                probs[:, int(cls)] = raw[:, j]
        if self.infil_clf is not None:
            infil = self.infil_clf.predict_proba(X)[:, 1].astype(np.float32)
        else:
            infil = probs[:, 2:].sum(axis=1)
        return probs, infil

    # --------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._write_config(path / "config.json")
        with open(path / "model.pkl", "wb") as fh:
            pickle.dump(
                {"dynamics": self.dynamics, "stage_clf": self.stage_clf,
                 "infil_clf": self.infil_clf, "ridge_alpha": self.ridge_alpha,
                 "logistic_C": self.logistic_C, "max_iter": self.max_iter},
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "NumpyDynamicsWorldModel":
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        with open(path / "model.pkl", "rb") as fh:
            blob = pickle.load(fh)
        model = cls(
            feature_names=config["feature_names"], context=config["context"],
            horizon=config["horizon"], stage_names=tuple(config["stage_names"]),
            ridge_alpha=blob["ridge_alpha"], logistic_C=blob["logistic_C"],
            max_iter=blob["max_iter"],
        )
        model.dynamics = blob["dynamics"]
        model.stage_clf = blob["stage_clf"]
        model.infil_clf = blob["infil_clf"]
        if config.get("standardiser"):
            from .base import Standardiser
            model.standardiser = Standardiser.from_state_dict(config["standardiser"])
        model.fitted = True
        return model


class PersistenceModel(WorldModel):
    """"Tomorrow looks like today."

    The floor for the dynamics task.  Network state is strongly autocorrelated,
    so persistence is a surprisingly hard baseline on next-state error -- which
    is exactly why it has to be reported.  A world model that beats a logistic
    regression on stage classification but cannot beat persistence on next-state
    prediction has not learned dynamics at all.
    """

    name = "persistence"

    def fit(self, dataset, val_dataset=None, **kwargs) -> "PersistenceModel":
        self.standardiser.fit(dataset.context)
        self.fitted = True
        return self

    def predict_next(self, context: np.ndarray) -> np.ndarray:
        return np.asarray(context, dtype=np.float32)[-1].copy()

    def predict_next_batch(self, context: np.ndarray) -> np.ndarray:
        return np.asarray(context, dtype=np.float32)[:, -1, :].copy()

    def heads(self, context: np.ndarray) -> tuple[np.ndarray, float]:
        probs = np.zeros(self.n_stages, dtype=np.float32)
        probs[0] = 1.0
        return probs, 0.0

    def save(self, path: str | Path) -> None:
        self._write_config(Path(path) / "config.json")

    @classmethod
    def load(cls, path: str | Path) -> "PersistenceModel":
        config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        model = cls(config["feature_names"], config["context"], config["horizon"],
                    tuple(config["stage_names"]))
        if config.get("standardiser"):
            from .base import Standardiser
            model.standardiser = Standardiser.from_state_dict(config["standardiser"])
        model.fitted = True
        return model
