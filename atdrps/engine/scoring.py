"""Dual-track scoring: what is *observed* versus what is *forecast*.

Why this module exists
----------------------
The world model answers "what comes next?".  That is the point of the project,
but on its own it produces a specific, embarrassing failure mode that was found
by testing an nmap-faithful capture against a trained model:

    window 20   0.83  Reconnaissance   <- the real scan. Correct.
    window 23   0.99  LateralMovement  <- never happened
    window 30   0.89  CommandAndControl<- never happened
    window 38   0.65  Exfiltration     <- never happened

A benign control capture through the same model fired one alert in 45 windows,
so the background is not the problem: the *scan* is.  The model has learned,
correctly, that reconnaissance is usually followed by the rest of a kill chain,
and the scan stays inside its 16-window context for eight minutes.  So it keeps
forecasting the campaign continuing.

That is defensible behaviour for a forecaster and indefensible behaviour for a
dashboard, because the two were being rendered identically.  An analyst reading
"Exfiltration 0.65" reasonably believes data is leaving right now.

The fix is not to weaken the model.  It is to stop conflating two questions:

    observed  -- is there evidence *in this window* for that stage?
                 (the auditable rule engine, on this window's real features)
    forecast  -- does the model expect infiltration in the *next* window?
                 (the learned dynamics)

Both are reported, and the verdict is graded by how they agree.  A forecast
with no supporting evidence is still shown -- suppressing it would throw away
the early warning that justifies the whole approach -- but it is labelled
PREDICTED, not CONFIRMED, and its contribution to the headline risk is damped.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..data.mitre import explain_rules, score_stages

__all__ = ["ScoringPolicy", "WindowVerdict", "score_window", "score_timeline",
           "LEVELS", "CONFIRMED", "PREDICTED", "WATCH", "CLEAR"]

CONFIRMED = "CONFIRMED"   # model forecasts it AND this window shows evidence
PREDICTED = "PREDICTED"   # model forecasts it, no corroborating evidence yet
WATCH = "WATCH"           # partial signal on either track
CLEAR = "CLEAR"           # neither track is elevated
LEVELS = (CLEAR, WATCH, PREDICTED, CONFIRMED)

# how much of the headline risk an uncorroborated forecast keeps.  Not zero --
# an early warning before the evidence arrives is the product -- but not full
# weight either, or the cascade above comes straight back.
_UNCORROBORATED_RETENTION = 0.55


@dataclass
class ScoringPolicy:
    """Thresholds for the dual-track verdict.

    Separate thresholds for the two tracks, rather than one global cutoff, is
    what lets "the model is confident but nothing supports it" be a different
    outcome from "the model is confident and the evidence agrees".
    """

    forecast_high: float = 0.70    # model alone is confident infiltration is coming
    forecast_floor: float = 0.55   # lowest forecast that fully-corroborated evidence can confirm
    forecast_low: float = 0.40     # model is mildly elevated
    evidence_strong: float = 0.35  # rule engine clearly supports that stage
    evidence_weak: float = 0.18    # rule engine partially supports it

    def confirm_threshold(self, evidence: float) -> float:
        """The forecast level required to CONFIRM, given this much evidence.

        Two independent signals agreeing is stronger than either alone, so the
        bar the model has to clear slides down as the rule engine -- which never
        sees the model's output -- corroborates the same stage. With no evidence
        the model must reach ``forecast_high`` on its own; with evidence at or
        above ``evidence_strong`` it need only reach ``forecast_floor``.

        This is why it matters in practice: a real nmap -T4 scan produced a
        model forecast of 0.698 against a flat 0.70 bar, and the auditable rule
        engine independently scored Reconnaissance at 0.368 on the same window.
        Missing a corroborated scan by two thousandths is a threshold artefact,
        not a judgement.
        """
        if self.evidence_strong <= 0:
            return self.forecast_high
        ratio = float(np.clip(evidence / self.evidence_strong, 0.0, 1.0))
        return self.forecast_high - (self.forecast_high - self.forecast_floor) * ratio

    def validate(self) -> "ScoringPolicy":
        if not 0.0 <= self.forecast_low <= self.forecast_high <= 1.0:
            raise ValueError("require 0 <= forecast_low <= forecast_high <= 1")
        if not self.forecast_low <= self.forecast_floor <= self.forecast_high:
            raise ValueError("require forecast_low <= forecast_floor <= forecast_high")
        if not 0.0 <= self.evidence_weak <= self.evidence_strong <= 1.0:
            raise ValueError("require 0 <= evidence_weak <= evidence_strong <= 1")
        return self


@dataclass
class WindowVerdict:
    window: int
    forecast_probability: float   # P(infiltration next window), from the model
    forecast_stage: str           # the model's predicted stage
    observed_score: float         # rule support for the FORECAST stage, this window
    observed_stage: str           # the stage this window's evidence best supports
    observed_best_score: float    # rule support for observed_stage
    corroborated: bool
    level: str
    risk: float                   # headline number, damped when uncorroborated
    reasons: list[str]            # the rules that actually fired, verbatim

    def to_dict(self) -> dict:
        return asdict(self)


def _observed(window_features: dict[str, float], forecast_stage: str
              ) -> tuple[float, str, float, list[str]]:
    """Rule-engine evidence for this window, independent of the model."""
    scores = score_stages(window_features)
    if not scores:
        return 0.0, "Benign", 0.0, []
    best_stage = max(scores, key=lambda s: scores[s])
    best_score = float(scores[best_stage])
    # support specifically for what the model is predicting -- this is the
    # corroboration question, and it is NOT the same as "the top stage"
    stage_score = float(scores.get(forecast_stage, 0.0))
    reasons = explain_rules(window_features, forecast_stage) if stage_score > 0 else []
    if not reasons and best_score > 0:
        reasons = explain_rules(window_features, best_stage)
    return stage_score, best_stage, best_score, reasons


def score_window(window_index: int, forecast_probability: float, forecast_stage: str,
                 window_features: dict[str, float],
                 policy: ScoringPolicy | None = None) -> WindowVerdict:
    """Grade one window on both tracks."""
    policy = (policy or ScoringPolicy()).validate()
    p = float(forecast_probability)
    stage_score, best_stage, best_score, reasons = _observed(window_features, forecast_stage)

    # "Benign" is not a thing the rule engine can evidence -- there is no rule
    # for the absence of attack -- so a benign forecast is never "corroborated"
    # and never needs to be: it raises nothing.
    benign_forecast = forecast_stage == "Benign"
    corroborated = (not benign_forecast) and stage_score >= policy.evidence_strong
    # evidence lowers the bar the model must clear on its own -- see
    # ScoringPolicy.confirm_threshold
    confirm_at = policy.confirm_threshold(stage_score)

    if benign_forecast:
        # the model predicts nothing is coming; evidence can still raise a WATCH
        level = WATCH if best_score >= policy.evidence_strong else CLEAR
    elif corroborated and p >= confirm_at:
        level = CONFIRMED
    elif p >= policy.forecast_high:
        level = PREDICTED
    elif p >= policy.forecast_low or best_score >= policy.evidence_strong:
        level = WATCH
    else:
        level = CLEAR

    if benign_forecast:
        risk = p * _UNCORROBORATED_RETENTION
    elif corroborated:
        risk = p
    else:
        # partial credit scales with partial evidence, so a forecast backed by
        # *some* evidence is not damped as hard as one backed by none
        partial = 0.0
        if policy.evidence_strong > policy.evidence_weak:
            partial = np.clip(
                (stage_score - policy.evidence_weak)
                / (policy.evidence_strong - policy.evidence_weak), 0.0, 1.0)
        risk = p * (_UNCORROBORATED_RETENTION
                    + (1.0 - _UNCORROBORATED_RETENTION) * float(partial))

    return WindowVerdict(
        window=int(window_index),
        forecast_probability=p,
        forecast_stage=forecast_stage,
        observed_score=round(stage_score, 4),
        observed_stage=best_stage,
        observed_best_score=round(best_score, 4),
        corroborated=bool(corroborated),
        level=level,
        risk=round(float(risk), 4),
        reasons=reasons[:4],
    )


def score_timeline(timeline: list[dict], states, policy: ScoringPolicy | None = None
                   ) -> list[WindowVerdict]:
    """Grade every row of an :class:`AnalysisResult` timeline.

    ``states`` is the :class:`~atdrps.data.windows.WindowedStates` the timeline
    was produced from; its raw feature rows are what the rule engine reads, so
    the observed track never touches the model.
    """
    policy = (policy or ScoringPolicy()).validate()
    names = list(states.feature_names)
    out: list[WindowVerdict] = []
    for row in timeline:
        t = int(row["window"])
        if t < 0 or t >= states.X.shape[0]:
            continue
        features = dict(zip(names, states.X[t].tolist()))
        out.append(score_window(
            t, row["infiltration_probability"], row.get("stage", "Benign"),
            features, policy,
        ))
    return out
