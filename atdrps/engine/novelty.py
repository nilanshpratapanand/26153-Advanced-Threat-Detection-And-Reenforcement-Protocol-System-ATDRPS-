"""Is this capture anything like the traffic the model was trained on?

A world model fitted on one traffic distribution will happily score a capture
from another one.  It will not fail; it will produce confident numbers.  This
module exists because that happened: a 145-minute capture of ordinary hostel
wifi was scored **CONFIRMED Exfiltration at 0.96**, and the reason was not a
bug in the model.  Fifty-four of the 103 state dimensions in that capture sat
more than three standard deviations outside the training distribution, and
several sat infinitely outside it -- the synthetic generator emits *exactly
zero* for TTL inconsistency, RST injection ratio and inbound flow share, so
real traffic containing any of them is in a region the model has never seen.
A linear model asked to extrapolate there extrapolates without limit.

The guard is cheap because the material is already on disk.  Every model ships
a :class:`~atdrps.models.base.Standardiser` carrying the per-feature mean, scale
and the ``lo``/``hi`` z-score envelope (roughly the 0.5th and 99.5th percentiles
of the training data, widened by 1).  A window's novelty is simply the fraction
of its dimensions that fall outside that envelope.

**Calibration.**  Measured, not chosen.  Over the 8,640 windows of the shipped
training corpus and the 291 windows of the real capture above:

| fraction outside | training corpus | real wifi capture |
|---|---|---|
| median           | 0.0000          | 0.2039            |
| p95              | 0.0097          | 0.3107            |
| windows > 0.10   | 0.02 %          | 99.3 %            |
| windows > 0.05   | 0.49 %          | 100 %             |

So the thresholds below separate the two populations almost perfectly, and the
capture-level call uses the *median* window rather than the worst one: a single
odd window is a burst of traffic, a median of 0.20 is a different network.

What this does **not** do is decide whether a capture is malicious.  It decides
whether this model is entitled to an opinion about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["NoveltyReport", "assess_novelty", "IN_DISTRIBUTION", "MARGINAL",
           "OUTSIDE", "WINDOW_OUTSIDE", "WINDOW_MARGINAL"]

IN_DISTRIBUTION = "in-distribution"
MARGINAL = "marginal"
OUTSIDE = "outside"

# per-window fraction-outside cutoffs (see the calibration table above)
WINDOW_MARGINAL = 0.05
WINDOW_OUTSIDE = 0.10

# capture-level cutoffs, applied to the median window
_CAPTURE_MARGINAL = 0.03
_CAPTURE_OUTSIDE = 0.10


@dataclass
class NoveltyReport:
    verdict: str                      # IN_DISTRIBUTION | MARGINAL | OUTSIDE
    median_fraction: float
    p95_fraction: float
    windows_outside: int
    n_windows: int
    n_features: int
    per_window: np.ndarray = field(default_factory=lambda: np.zeros(0))
    worst_features: list = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return self.verdict == IN_DISTRIBUTION

    def message(self) -> str:
        if self.verdict == IN_DISTRIBUTION:
            return ""
        share = 100.0 * self.median_fraction
        lead = (
            "This capture is outside the distribution this model was trained on"
            if self.verdict == OUTSIDE else
            "This capture sits at the edge of the distribution this model was "
            "trained on"
        )
        drivers = ", ".join(
            f"{f['feature']} ({f['value']:.3g} vs a trained range of "
            f"{f['train_low']:.3g}-{f['train_high']:.3g})"
            for f in self.worst_features[:3]
        )
        tail = (
            " Treat every number below as unverified: the model is "
            "extrapolating, and a model extrapolating past its training data "
            "can be confidently wrong. Retrain on traffic from this network "
            "before acting on it."
            if self.verdict == OUTSIDE else
            " The scores below are usable but should be read with that in mind."
        )
        return (f"{lead}: in a typical window {share:.0f}% of the "
                f"{self.n_features} state features fall outside the range the "
                f"training data covered. Furthest out: {drivers}.{tail}")

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "median_fraction": float(self.median_fraction),
            "p95_fraction": float(self.p95_fraction),
            "windows_outside": int(self.windows_outside),
            "n_windows": int(self.n_windows),
            "n_features": int(self.n_features),
            "message": self.message(),
            "worst_features": self.worst_features,
        }


def assess_novelty(X: np.ndarray, standardiser, feature_names: list[str],
                   top_k: int = 6) -> NoveltyReport:
    """Score a capture's windows against the envelope the model was fitted on.

    Returns an in-distribution verdict when the standardiser carries no
    envelope: a model that never recorded where its training data lived cannot
    be asked this question, and inventing an answer would be worse than
    declining one.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n_features = X.shape[1]
    lo = getattr(standardiser, "lo", None)
    hi = getattr(standardiser, "hi", None)
    mean = getattr(standardiser, "mean", None)
    scale = getattr(standardiser, "scale", None)
    if lo is None or hi is None or mean is None or scale is None:
        return NoveltyReport(IN_DISTRIBUTION, 0.0, 0.0, 0, len(X), n_features)

    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    mean, scale = np.asarray(mean), np.asarray(scale)
    z = (X - mean) / scale
    outside = (z < lo) | (z > hi)

    # A feature the training data never varied is the sharpest signal there is,
    # and the plain envelope misses it completely: a constant column gets unit
    # scale, so its widened percentile bounds are +/-1 *raw unit* -- for
    # ttl_inconsistency, whose whole range is 0 to 1, that declares every
    # possible value in-distribution.  The training data says nothing about any
    # value other than the constant, so anything else is outside.
    constant = getattr(standardiser, "constant", None)
    if constant is not None:
        constant = np.asarray(constant, dtype=bool)
        if constant.shape[-1] == X.shape[1]:
            differs = ~np.isclose(X, mean, rtol=0.0, atol=1e-8)
            outside = outside | (differs & constant)
    per_window = outside.mean(axis=1)

    median = float(np.median(per_window)) if len(per_window) else 0.0
    p95 = float(np.percentile(per_window, 95)) if len(per_window) else 0.0
    verdict = (OUTSIDE if median > _CAPTURE_OUTSIDE else
               MARGINAL if median > _CAPTURE_MARGINAL else IN_DISTRIBUTION)

    # which features are furthest out, and by how much, in original units
    share_outside = outside.mean(axis=0)
    excess = np.maximum(z.mean(axis=0) - hi, lo - z.mean(axis=0))
    rank = np.argsort(-(share_outside + np.clip(excess, 0, None) / 1e3))
    worst = []
    for i in rank[:top_k]:
        if not share_outside[i]:
            continue
        worst.append({
            "feature": feature_names[i] if i < len(feature_names) else str(i),
            "value": float(X[:, i].mean()),
            "train_low": float(lo[i] * scale[i] + mean[i]),
            "train_high": float(hi[i] * scale[i] + mean[i]),
            "windows_outside": float(share_outside[i]),
        })

    return NoveltyReport(
        verdict=verdict, median_fraction=median, p95_fraction=p95,
        windows_outside=int((per_window > WINDOW_OUTSIDE).sum()),
        n_windows=len(per_window), n_features=n_features,
        per_window=per_window, worst_features=worst,
    )
