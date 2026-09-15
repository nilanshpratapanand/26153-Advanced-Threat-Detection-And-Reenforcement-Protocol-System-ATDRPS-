"""Turning a forecast into an answer an analyst can argue with.

Two complementary questions, answered separately because they are different
questions:

*What* drove the forecast
    Grouped KernelSHAP over the 96 state features.  Masking a feature swaps its
    whole trace across the context for a background capture's, so the
    attribution is per *measurement*, not per (measurement, time step) cell.

*When* did it come from
    For the transformer, the attention weights of the final block, which are
    exactly "how much of this forecast came from each of the last L windows".
    For any other backend -- and as a cross-check on the transformer -- temporal
    occlusion: replace one window with the background average and measure how
    much the forecast moves.  Model-agnostic, and it means the linear backend is
    not a second-class citizen at explanation time.

Both are reported against a **background distribution of real windows**, not
against zeros.  A SHAP value answers "why this rather than a typical window",
and picking an unrealistic reference makes the answer unrealistic too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.mitre import describe_stage, explain_rules, score_stages
from .glossary import describe_feature
from .kernel_shap import ShapResult, kernel_shap_masked

__all__ = ["Explanation", "ForecastExplainer"]


@dataclass
class Explanation:
    target: str
    prediction: float
    base_value: float
    feature_values: np.ndarray
    feature_names: list[str]
    temporal: np.ndarray
    temporal_source: str
    top_k: int = 8
    rule_reasons: list[str] = field(default_factory=list)
    efficiency_error: float = 0.0

    def drivers(self, k: int | None = None) -> list[dict]:
        k = k or self.top_k
        order = np.argsort(-np.abs(self.feature_values))[:k]
        return [
            {
                "feature": self.feature_names[i],
                "description": describe_feature(self.feature_names[i]),
                "contribution": float(self.feature_values[i]),
                "direction": "raises" if self.feature_values[i] > 0 else "lowers",
            }
            for i in order
        ]

    def key_windows(self, k: int = 3) -> list[dict]:
        """Which of the last L windows mattered most, most recent = index L-1."""
        order = np.argsort(-np.abs(self.temporal))[:k]
        L = len(self.temporal)
        return [
            {"window": int(i), "windows_ago": int(L - 1 - i),
             "weight": float(self.temporal[i])}
            for i in sorted(order, key=lambda j: -abs(self.temporal[j]))
        ]

    def to_text(self, k: int | None = None) -> str:
        lines = [
            f"{self.target}: {self.prediction:.3f} "
            f"(typical window: {self.base_value:.3f})",
            "Driven by:",
        ]
        for d in self.drivers(k):
            arrow = "+" if d["contribution"] > 0 else "-"
            lines.append(f"  {arrow} {d['description']} ({d['contribution']:+.3f})")
        windows = self.key_windows()
        if windows:
            spans = ", ".join(
                f"{w['windows_ago']} window(s) ago ({w['weight']:.2f})" for w in windows
            )
            lines.append(f"Most influential time steps [{self.temporal_source}]: {spans}")
        if self.rule_reasons:
            lines.append("Corroborating rules:")
            lines.extend(f"  - {r}" for r in self.rule_reasons)
        return "\n".join(lines)


class ForecastExplainer:
    def __init__(self, model, background: np.ndarray, top_k: int = 8,
                 n_samples: int = 256, n_background: int = 32, seed: int = 0) -> None:
        """``background`` is ``(N, L, F)`` -- real observed contexts."""
        bg = np.asarray(background, dtype=np.float32)
        if bg.ndim != 3:
            raise ValueError(f"background must be (N, L, F), got {bg.shape}")
        if bg.shape[0] > n_background:
            idx = np.random.default_rng(seed).choice(bg.shape[0], n_background, replace=False)
            bg = bg[idx]
        self.model = model
        self.background = bg
        self.top_k = int(top_k)
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.feature_names = list(model.feature_names)

    # ------------------------------------------------------------- scoring
    def _score(self, contexts: np.ndarray, target: str, stage_index: int) -> np.ndarray:
        probs, infil = self.model.heads_batch(contexts)
        if target == "infiltration":
            return np.asarray(infil, dtype=np.float64).ravel()
        return np.asarray(probs, dtype=np.float64)[:, stage_index]

    # ------------------------------------------------------------ what/why
    def explain(self, context: np.ndarray, target: str = "infiltration",
                stage: str | None = None, summary: dict | None = None) -> Explanation:
        context = np.asarray(context, dtype=np.float32)[-self.model.context:]
        stage_index = self.model.stage_names.index(stage) if stage else 0
        label = target if target == "infiltration" else f"P({stage})"

        base_value = float(np.mean(self._score(self.background, target, stage_index)))
        prediction = float(self._score(context[None, ...], target, stage_index)[0])

        n_bg, L, F = self.background.shape

        def evaluate(Z: np.ndarray) -> np.ndarray:
            out = np.zeros(Z.shape[0], dtype=np.float64)
            for i in range(Z.shape[0]):
                mask = Z[i].astype(bool)
                # start from the background, paste in the observed trace for
                # every feature present in this coalition
                synthetic = self.background.copy()
                synthetic[:, :, mask] = context[None, :, mask]
                out[i] = float(np.mean(self._score(synthetic, target, stage_index)))
            return out

        shap: ShapResult = kernel_shap_masked(
            evaluate, F, base_value, prediction, self.feature_names,
            n_samples=self.n_samples, seed=self.seed,
        )

        temporal, source = self.temporal_attribution(context, target, stage_index)
        reasons = []
        if summary and stage:
            reasons = explain_rules(summary, stage)

        return Explanation(
            target=label, prediction=prediction, base_value=base_value,
            feature_values=shap.values, feature_names=self.feature_names,
            temporal=temporal, temporal_source=source, top_k=self.top_k,
            rule_reasons=reasons, efficiency_error=shap.efficiency_error,
        )

    # ---------------------------------------------------------------- when
    def temporal_attribution(self, context: np.ndarray, target: str = "infiltration",
                             stage_index: int = 0) -> tuple[np.ndarray, str]:
        context = np.asarray(context, dtype=np.float32)[-self.model.context:]
        attention = getattr(self.model, "attention", None)
        if callable(attention):
            try:
                weights = np.asarray(attention(context), dtype=np.float64)
                if weights.size == context.shape[0] and np.isfinite(weights).all():
                    return weights, "attention"
            except Exception:  # pragma: no cover - fall back rather than fail
                pass
        return self._occlusion(context, target, stage_index), "occlusion"

    def _occlusion(self, context: np.ndarray, target: str, stage_index: int) -> np.ndarray:
        """Replace one window with the background average and see what moves."""
        L = context.shape[0]
        reference = self.background.mean(axis=0)
        baseline = float(self._score(context[None, ...], target, stage_index)[0])
        variants = np.repeat(context[None, ...], L, axis=0)
        for t in range(L):
            variants[t, t, :] = reference[t]
        scores = self._score(variants, target, stage_index)
        deltas = np.abs(baseline - np.asarray(scores, dtype=np.float64))
        total = deltas.sum()
        return deltas / total if total > 0 else np.full(L, 1.0 / L)

    # -------------------------------------------------------------- report
    def report(self, context: np.ndarray, forecast, summary: dict | None = None) -> dict:
        """A complete explanation package for one forecast."""
        stage = forecast.stage_names[0] if forecast.stage_names else None
        infil = self.explain(context, target="infiltration", summary=summary)
        stage_exp = self.explain(context, target="stage", stage=stage, summary=summary)
        rule_scores = score_stages(summary) if summary else {}
        return {
            "forecast": forecast.timeline(),
            "peak_infiltration": forecast.peak_infiltration,
            "predicted_stage": stage,
            "stage_description": describe_stage(stage) if stage else "",
            "infiltration_explanation": infil,
            "stage_explanation": stage_exp,
            "rule_scores": rule_scores,
        }
