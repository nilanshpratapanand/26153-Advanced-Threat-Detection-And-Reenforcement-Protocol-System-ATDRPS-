"""End-to-end inference: a capture goes in, a forecast comes out.

This is the object the dashboard and the CLI both drive, and it is deliberately
the only place that knows how to go from a file on disk to an answer.  Having
one path means the number in the demo is the number the benchmark measured.

What it produces for a capture:

* a **risk timeline** -- for every window, the model's one-step-ahead
  infiltration probability and predicted MITRE stage, so an analyst can see
  where the network started drifting rather than only the final verdict
* a **K-step forward simulation** from the end of the capture -- the actual
  forecast
* the **flows behind the peak window**, so "the network is heading for lateral
  movement" comes with the conversations that say so
* an **explanation** for the peak: driving features, the time steps that
  mattered, and any interpretable rules that agree

Everything runs locally.  No network calls, no model downloads, nothing that
would stop it working inside an air-gapped network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.datasets import load_flow_csv
from ..data.flows import assemble_flows
from ..data.mitre import describe_stage, score_stages
from ..data.pcap import read_pcap
from ..data.schema import STAGES, flags_to_str
from ..data.windows import build_windows
from ..explain.explainer import ForecastExplainer
from ..models.base import Forecast
from .scoring import ScoringPolicy, score_timeline

__all__ = ["AnalysisResult", "ThreatForecastEngine", "load_capture", "serialise_result"]

PCAP_SUFFIXES = {".pcap", ".pcapng", ".cap", ".dmp"}


def load_capture(path: str | Path, window_size_s: float = 30.0,
                 history_s: float = 600.0, max_packets: int | None = None):
    """Read a pcap or a flow CSV and return ``(states, flows, source_kind)``."""
    path = Path(path)
    if path.suffix.lower() in PCAP_SUFFIXES:
        table = read_pcap(path, max_packets=max_packets)
        flows = assemble_flows(table)
        kind = f"pcap ({len(table)} packets)"
        states = build_windows(flows, window_size_s=window_size_s,
                               history_s=history_s, use_labels=False)
    else:
        loaded = load_flow_csv(path)
        flows = loaded.frame
        kind = f"{loaded.dataset} CSV ({loaded.summary()})"
        states = build_windows(flows, window_size_s=window_size_s, history_s=history_s)
    return states, flows, kind


@dataclass
class AnalysisResult:
    source: str
    source_kind: str
    window_size_s: float
    timeline: list[dict] = field(default_factory=list)
    forecast: Forecast | None = None
    peak_window: int = -1
    flagged_flows: list[dict] = field(default_factory=list)
    explanation: dict | None = None
    n_windows: int = 0
    n_flows: int = 0
    notes: list[str] = field(default_factory=list)
    verdicts: list[dict] = field(default_factory=list)

    @property
    def peak_risk(self) -> float:
        """Highest *credible* risk -- the damped score, not the raw probability.

        Reporting the raw maximum is how an uncorroborated forecast ends up as
        the headline: an nmap scan with nothing after it produced "peak
        infiltration probability 1.00 (stage: LateralMovement)" on a capture
        whose only attack was the scan.
        """
        return max((row.get("risk", row["infiltration_probability"])
                    for row in self.timeline), default=0.0)

    @property
    def raw_peak_probability(self) -> float:
        """The undamped model maximum, kept for benchmarking and transparency."""
        return max((row["infiltration_probability"] for row in self.timeline), default=0.0)

    @property
    def confirmed(self) -> list[dict]:
        return [r for r in self.timeline if r.get("level") == "CONFIRMED"]

    @property
    def predicted(self) -> list[dict]:
        return [r for r in self.timeline if r.get("level") == "PREDICTED"]

    def headline(self) -> str:
        """One line that distinguishes what is seen from what is expected."""
        if not self.timeline:
            return "not enough traffic to forecast"

        confirmed, predicted = self.confirmed, self.predicted
        if confirmed:
            top = max(confirmed, key=lambda r: r.get("risk", 0.0))
            head = (f"CONFIRMED {top['stage']} in window {top['window']} "
                    f"at {top.get('risk', 0.0):.2f} "
                    f"(corroborated by {top.get('observed_score', 0.0):.2f} rule evidence)")
        elif predicted:
            top = max(predicted, key=lambda r: r.get("risk", 0.0))
            head = (f"PREDICTED {top['stage']} at {top.get('risk', 0.0):.2f} "
                    f"-- forecast only, no corroborating evidence in the capture yet")
        else:
            head = f"no confirmed intrusion; peak credible risk {self.peak_risk:.2f}"

        extra = ""
        if confirmed and predicted:
            nxt = max(predicted, key=lambda r: r.get("risk", 0.0))
            extra = f"; model forecasts {nxt['stage']} next (not yet corroborated)"
        forward = (f"; forward simulation peaks at {self.forecast.peak_infiltration:.2f} "
                   f"in {self.forecast.peak_step} window(s)") if self.forecast else ""
        return head + extra + forward


class ThreatForecastEngine:
    def __init__(self, model, explainer: ForecastExplainer | None = None,
                 window_size_s: float = 30.0, history_s: float = 600.0,
                 threshold: float = 0.5,
                 policy: ScoringPolicy | None = None) -> None:
        self.model = model
        self.explainer = explainer
        self.window_size_s = float(window_size_s)
        self.history_s = float(history_s)
        self.threshold = float(threshold)
        # dual-track scoring: see atdrps/engine/scoring.py for why
        self.policy = (policy or ScoringPolicy()).validate()

    # -------------------------------------------------------------- loading
    @classmethod
    def load(cls, model_dir: str | Path, background: np.ndarray | None = None,
             window_size_s: float = 30.0, threshold: float = 0.5,
             top_k: int = 8, shap_samples: int = 192) -> "ThreatForecastEngine":
        """Load whichever backend was saved in ``model_dir``."""
        import json

        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        name = config.get("name", "")
        if name == "temporal-transformer":
            from ..models.transformer import TemporalTransformerWorldModel as Backend
        elif name == "persistence":
            from ..models.numpy_dynamics import PersistenceModel as Backend
        else:
            from ..models.numpy_dynamics import NumpyDynamicsWorldModel as Backend
        model = Backend.load(model_dir)

        explainer = None
        if background is not None and len(background):
            explainer = ForecastExplainer(model, background, top_k=top_k,
                                          n_samples=shap_samples)
        return cls(model, explainer, window_size_s=window_size_s, threshold=threshold)

    # ------------------------------------------------------------- analysis
    def analyse(self, path: str | Path, horizon: int | None = None,
                explain: bool = True, max_flagged: int = 25,
                max_packets: int | None = None) -> AnalysisResult:
        states, flows, kind = load_capture(path, self.window_size_s, self.history_s,
                                           max_packets=max_packets)
        result = AnalysisResult(
            source=str(path), source_kind=kind, window_size_s=self.window_size_s,
            n_windows=len(states), n_flows=len(flows),
        )
        L = self.model.context
        if len(states) < L:
            result.notes.append(
                f"capture covers {len(states)} windows but the model needs {L} "
                f"of context; extend the capture or reduce window size"
            )
            return result
        return self._score(states, flows, result, horizon, explain, max_flagged)

    def analyse_states(self, states, flows, source: str = "in-memory",
                       horizon: int | None = None, explain: bool = True,
                       max_flagged: int = 25) -> AnalysisResult:
        result = AnalysisResult(source=source, source_kind="pre-windowed states",
                                window_size_s=states.window_size_s,
                                n_windows=len(states), n_flows=len(flows))
        if len(states) < self.model.context:
            result.notes.append("not enough context windows")
            return result
        return self._score(states, flows, result, horizon, explain, max_flagged)

    def _score(self, states, flows, result, horizon, explain, max_flagged):
        L = self.model.context
        X = states.X
        contexts = np.stack([X[t - L + 1:t + 1] for t in range(L - 1, len(states))])
        probs, infil = self.model.heads_batch(contexts)

        for i, t in enumerate(range(L - 1, len(states))):
            stage_idx = int(np.asarray(probs[i]).argmax())
            result.timeline.append({
                "window": t,
                "window_start": float(states.ts_start[t]),
                "infiltration_probability": float(infil[i]),
                "stage": STAGES[stage_idx] if stage_idx < len(STAGES) else "Benign",
                "stage_confidence": float(np.asarray(probs[i]).max()),
                "alert": bool(infil[i] >= self.threshold),
            })

        # --- dual-track grading -------------------------------------------
        # The rule engine reads this window's real features; the model forecasts
        # the next one. Grading them against each other is what stops a single
        # scan from producing eight minutes of phantom kill-chain alerts.
        verdicts = score_timeline(result.timeline, states, self.policy)
        by_window = {v.window: v for v in verdicts}
        for row in result.timeline:
            v = by_window.get(int(row["window"]))
            if v is None:
                continue
            row["level"] = v.level
            row["risk"] = v.risk
            row["observed_score"] = v.observed_score
            row["observed_stage"] = v.observed_stage
            row["corroborated"] = v.corroborated
            row["reasons"] = v.reasons
            # an alert now requires the two tracks to agree, not just a high
            # forecast -- PREDICTED still surfaces, but as a forecast, not a fact
            row["alert"] = v.level == "CONFIRMED"
        result.verdicts = [v.to_dict() for v in verdicts]

        # peak = most *credible* window, not merely the highest raw probability
        peak = int(np.argmax([row.get("risk", row["infiltration_probability"])
                              for row in result.timeline]))
        result.peak_window = peak
        peak_window_index = result.timeline[peak]["window"]

        result.forecast = self.model.rollout(
            X[-L:], horizon=horizon,
            ts_start=float(states.ts_start[-1]), window_size_s=states.window_size_s,
        )

        result.flagged_flows = self._flows_for_window(states, flows, peak_window_index,
                                                      max_flagged)
        summary = states.summaries[peak_window_index] if states.summaries else None
        if summary:
            result.notes.append(
                "rule engine agrees: "
                + ", ".join(f"{k}={v:.2f}" for k, v in
                            sorted(score_stages(summary).items(), key=lambda x: -x[1])[:3])
            )
        if explain and self.explainer is not None:
            # The "why" panel explains the peak *observed* window, so it must be
            # keyed on that window's own predicted stage -- not on the first
            # step of a forward simulation that starts from the end of the
            # capture and may well be describing something else entirely.
            result.explanation = self.explainer.report(
                contexts[peak], result.forecast, summary,
                stage=result.timeline[peak]["stage"],
            )
        return result

    @staticmethod
    def _flows_for_window(states, flows, window_index: int, limit: int) -> list[dict]:
        """The conversations inside a window, worst first.

        Ordered by a blend of the signals that make a flow worth a human's
        attention rather than by raw byte count, so a noisy backup does not bury
        a 300-byte beacon.
        """
        if not states.flow_index or window_index >= len(states.flow_index):
            return []
        idx = states.flow_index[window_index]
        if len(idx) == 0 or flows is None or len(flows) == 0:
            return []
        subset = flows.iloc[idx].copy()
        score = (
            subset.get("syn_ratio", 0) * 2.0
            + subset.get("rst_ratio", 0) * 1.5
            + subset.get("payload_zero_ratio", 0) * 1.0
            + np.log1p(subset.get("total_bytes", 0)) / 20.0
            + subset.get("retransmission_ratio", 0)
        )
        subset["_score"] = score
        subset = subset.sort_values("_score", ascending=False).head(limit)

        out = []
        for _, row in subset.iterrows():
            out.append({
                "src": f"{row.get('src_ip', '')}:{int(row.get('src_port_raw', 0))}",
                "dst": f"{row.get('dst_ip', '')}:{int(row.get('dst_port_raw', 0))}",
                "protocol": int(row.get("protocol_raw", 0)),
                "packets": int(row.get("total_packets", 0)),
                "bytes": int(row.get("total_bytes", 0)),
                "duration": round(float(row.get("flow_duration", 0.0)), 3),
                "flags": _flag_summary(row),
                "payload_zero_ratio": round(float(row.get("payload_zero_ratio", 0.0)), 3),
                "retransmissions": int(row.get("retransmission_count", 0)),
            })
        return out


def serialise_result(result: AnalysisResult) -> dict:
    """The wire format both the batch dashboard and live mode send to the browser.

    One function so a window analysed from an uploaded file and a window
    analysed from a live capture look identical on the frontend -- the UI
    never needs to know which one it's looking at.
    """
    payload = {
        "source": Path(result.source).name if "/" in str(result.source) or "\\" in str(result.source)
                  else str(result.source),
        "source_kind": result.source_kind,
        "headline": result.headline(),
        "n_windows": result.n_windows,
        "n_flows": result.n_flows,
        "window_size_s": result.window_size_s,
        "peak_window": result.peak_window,
        "peak_risk": result.peak_risk,
        "notes": result.notes,
        "timeline": result.timeline,
        "forecast": result.forecast.timeline() if result.forecast else [],
        "flagged_flows": result.flagged_flows,
    }
    explanation = result.explanation
    if explanation:
        infil = explanation["infiltration_explanation"]
        payload["explanation"] = {
            "stage": explanation["predicted_stage"],
            "stage_description": explanation["stage_description"],
            "prediction": infil.prediction,
            "base_value": infil.base_value,
            "drivers": infil.drivers(),
            "key_windows": infil.key_windows(),
            "temporal_source": infil.temporal_source,
            "temporal": [float(v) for v in infil.temporal],
            "rules": infil.rule_reasons,
            "efficiency_error": infil.efficiency_error,
            "rule_scores": {k: float(v) for k, v in explanation["rule_scores"].items()},
        }
    else:
        payload["explanation"] = None
    return payload


def _flag_summary(row: pd.Series) -> str:
    bits = 0
    from ..data.schema import (TCP_ACK, TCP_FIN, TCP_PSH, TCP_RST, TCP_SYN, TCP_URG)
    for name, bit in (("fin_count", TCP_FIN), ("syn_count", TCP_SYN),
                      ("rst_count", TCP_RST), ("psh_count", TCP_PSH),
                      ("ack_count", TCP_ACK), ("urg_count", TCP_URG)):
        if float(row.get(name, 0)) > 0:
            bits |= bit
    return flags_to_str(bits)
