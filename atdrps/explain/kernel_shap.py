"""KernelSHAP, implemented in-tree.

The problem statement makes explainability mandatory and names SHAP.  ATDRPS
does not depend on the ``shap`` package: the tool has to install inside
air-gapped networks, and the algorithm for the model-agnostic kernel explainer
is short enough to implement exactly and test against closed-form answers.

The method (Lundberg & Lee, 2017): approximate the Shapley values of a
prediction by fitting a weighted linear model to the outputs of the black box
under sampled feature coalitions, with the coalition weights

    pi(z) = (M - 1) / (C(M, |z|) * |z| * (M - |z|))

which is the unique kernel that makes the fitted coefficients Shapley values.
The two trivial coalitions -- nothing present, everything present -- have
infinite weight; they are imposed as the constraint

    sum_i phi_i = f(x) - E[f]

rather than sampled, by eliminating one variable, which is what makes the
result satisfy *efficiency* exactly instead of approximately.

Grouped features
----------------
The world model's input is ``L x F`` -- 1536 numbers at the default settings.
Explaining all of them individually is expensive and unreadable ("window 11 of
``syn_ratio__std`` contributed 0.002"). So attributions are computed over the
``F`` features as groups: masking a feature replaces its entire trace across
the context with the background's. The answer is then at the level an analyst
asks about -- *which measurement* drove this -- and :mod:`atdrps.explain.attention`
answers the complementary question of *which time step*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ShapResult", "kernel_shap", "kernel_shap_masked", "shap_kernel_weights"]


@dataclass
class ShapResult:
    values: np.ndarray          # (d,) contribution per feature group
    base_value: float           # E[f] over the background
    prediction: float           # f(x)
    feature_names: list[str]
    n_coalitions: int = 0

    @property
    def efficiency_error(self) -> float:
        """How far the attributions are from summing to the prediction gap.

        Reported rather than hidden: if this is not tiny, the explanation is
        not an explanation, and the caller deserves to know.
        """
        return float(abs(self.values.sum() - (self.prediction - self.base_value)))

    def top(self, k: int = 8) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.values))[:k]
        return [(self.feature_names[i], float(self.values[i])) for i in order]


def shap_kernel_weights(m: int, sizes: np.ndarray) -> np.ndarray:
    """pi(z) for coalitions of the given sizes, over ``m`` features.

    Computed in log space: ``C(96, 48)`` overflows a float otherwise, and the
    resulting silent ``inf`` would quietly destroy the regression.
    """
    from scipy.special import gammaln

    sizes = np.asarray(sizes, dtype=np.float64)
    log_comb = (gammaln(m + 1) - gammaln(sizes + 1) - gammaln(m - sizes + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_w = np.log(m - 1) - log_comb - np.log(sizes) - np.log(m - sizes)
    weights = np.exp(log_w)
    weights[(sizes <= 0) | (sizes >= m)] = 0.0
    return weights


def kernel_shap(
    predict: callable,
    x: np.ndarray,
    background: np.ndarray,
    feature_names: list[str] | None = None,
    n_samples: int = 256,
    seed: int = 0,
    l2: float = 1e-6,
) -> ShapResult:
    """Shapley attributions for a single prediction.

    ``predict`` maps ``(n, d)`` to ``(n,)``.  ``x`` is the instance, ``background``
    is the reference distribution the explanation is *relative to* -- SHAP values
    answer "why this rather than a typical window", so the background should be
    typical windows, not zeros.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    background = np.atleast_2d(np.asarray(background, dtype=np.float64))
    m = x.size
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(m)]
    if background.shape[1] != m:
        raise ValueError(f"background has {background.shape[1]} features, instance has {m}")
    if m < 2:
        raise ValueError("kernel_shap needs at least two features")

    rng = np.random.default_rng(seed)
    base_value = float(np.mean(predict(background)))
    prediction = float(predict(x.reshape(1, -1))[0])

    def evaluate(Z: np.ndarray) -> np.ndarray:
        out = np.zeros(Z.shape[0], dtype=np.float64)
        for i in range(Z.shape[0]):
            mask = Z[i].astype(bool)
            synthetic = background.copy()
            synthetic[:, mask] = x[mask]
            out[i] = float(np.mean(predict(synthetic)))
        return out

    return kernel_shap_masked(evaluate, m, base_value, prediction, names,
                              n_samples=n_samples, seed=seed, l2=l2, rng=rng)


def kernel_shap_masked(
    evaluate: callable,
    m: int,
    base_value: float,
    prediction: float,
    feature_names: list[str] | None = None,
    n_samples: int = 256,
    seed: int = 0,
    l2: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> ShapResult:
    """KernelSHAP where the caller owns how a coalition is realised.

    ``evaluate`` takes a ``(n, m)`` binary coalition matrix and returns the
    expected model output for each row.  This is the form the world model needs:
    a "feature" there is a whole time series across the context, and masking it
    means substituting a background capture's trace for that measurement --
    something only the caller can do.
    """
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(m)]
    if m < 2:
        raise ValueError("kernel_shap needs at least two features")
    rng = rng or np.random.default_rng(seed)

    # sample coalitions in proportion to the kernel, which puts nearly all of
    # its mass on the very small and very large ones
    sizes = np.arange(1, m)
    size_weights = shap_kernel_weights(m, sizes)
    size_weights = size_weights / size_weights.sum()
    drawn = rng.choice(sizes, size=n_samples, p=size_weights)

    Z = np.zeros((n_samples, m), dtype=np.float64)
    for i, size in enumerate(drawn):
        Z[i, rng.choice(m, size=int(size), replace=False)] = 1.0

    outputs = np.asarray(evaluate(Z), dtype=np.float64)

    weights = shap_kernel_weights(m, Z.sum(axis=1))
    keep = weights > 0
    Z, outputs, weights = Z[keep], outputs[keep], weights[keep]
    if Z.shape[0] < 2:
        return ShapResult(np.zeros(m), base_value, prediction, names, 0)

    # efficiency imposed exactly by eliminating the last variable, rather than
    # left to the regression to approximate
    y = outputs - base_value - Z[:, -1] * (prediction - base_value)
    A = Z[:, :-1] - Z[:, -1:]
    W = weights / weights.sum()

    Aw = A * W[:, None]
    lhs = A.T @ Aw + l2 * np.eye(m - 1)
    rhs = Aw.T @ y
    phi_rest = np.linalg.solve(lhs, rhs)
    phi_last = (prediction - base_value) - phi_rest.sum()
    values = np.concatenate([phi_rest, [phi_last]])

    return ShapResult(values=values, base_value=base_value, prediction=prediction,
                      feature_names=names, n_coalitions=int(Z.shape[0]))
