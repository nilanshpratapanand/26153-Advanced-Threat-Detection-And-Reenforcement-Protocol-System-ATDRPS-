"""The offline dashboard.

Flask rather than Streamlit, for one reason that matters here: Streamlit pulls
a large dependency tree and expects to phone home for usage statistics unless
told otherwise.  This has to run inside an air-gapped network, so the app is a
single Flask process serving one page, with every stylesheet and script inlined
and no request to any external host anywhere in it.

The page does what the problem statement asks a demo interface to do: take a
PCAP or a flow CSV, and show the infiltration probability timeline, the
predicted ATT&CK stage, the flows behind the decision, and the features that
drove it.
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from atdrps.config import Config
from atdrps.data.mitre import MITRE_TACTICS, describe_stage
from atdrps.engine.inference import ThreatForecastEngine

__all__ = ["create_app"]

ALLOWED = {".pcap", ".pcapng", ".cap", ".dmp", ".csv"}


def create_app(model_dir: str | Path = "artifacts/model-linear",
               config: Config | None = None) -> Flask:
    cfg = config or Config.load()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = int(cfg.server.max_upload_mb) * 1024 * 1024

    model_dir = Path(model_dir)
    state: dict = {"engine": None, "error": None, "model_dir": str(model_dir)}

    def engine() -> ThreatForecastEngine:
        if state["engine"] is None and state["error"] is None:
            try:
                background = None
                for candidate in (model_dir / "background.npy",
                                  model_dir.parent / "background.npy"):
                    if candidate.exists():
                        background = np.load(candidate)
                        break
                state["engine"] = ThreatForecastEngine.load(
                    model_dir, background=background,
                    window_size_s=cfg.window.size_s,
                    top_k=cfg.explain.top_k, shap_samples=cfg.explain.shap_samples,
                )
            except Exception as exc:            # surfaced in the UI, not swallowed
                state["error"] = (
                    f"could not load a model from {model_dir}: {exc}. "
                    "Run `atdrps benchmark` or `atdrps train` first."
                )
        if state["error"]:
            raise RuntimeError(state["error"])
        return state["engine"]

    @app.route("/")
    def index():
        ready, message = True, ""
        try:
            engine()
        except RuntimeError as exc:
            ready, message = False, str(exc)
        return render_template(
            "index.html", ready=ready, message=message,
            model_dir=str(model_dir), window_size=cfg.window.size_s,
            horizon=cfg.model.horizon,
            stages=[(name, MITRE_TACTICS.get(name, "-")) for name in
                    ("Benign", "Reconnaissance", "InitialAccess",
                     "LateralMovement", "CommandAndControl", "Exfiltration")],
        )

    @app.route("/api/health")
    def health():
        try:
            eng = engine()
            return jsonify({
                "ok": True, "model": eng.model.name, "context": eng.model.context,
                "horizon": eng.model.horizon, "features": eng.model.n_features,
                "offline": True,
            })
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 503

    @app.route("/api/analyse", methods=["POST"])
    def analyse():
        upload = request.files.get("capture")
        if upload is None or not upload.filename:
            return jsonify({"error": "no file was uploaded"}), 400
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED:
            return jsonify({
                "error": f"unsupported file type {suffix!r}; "
                         f"expected one of {', '.join(sorted(ALLOWED))}"
            }), 400

        threshold = float(request.form.get("threshold", 0.5))
        horizon = request.form.get("horizon")
        explain = request.form.get("explain", "1") != "0"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / Path(upload.filename).name
            upload.save(path)
            try:
                eng = engine()
                eng.threshold = threshold
                result = eng.analyse(path, horizon=int(horizon) if horizon else None,
                                     explain=explain)
            except RuntimeError as exc:
                return jsonify({"error": str(exc)}), 503
            except Exception as exc:            # a bad capture must not 500 silently
                return jsonify({
                    "error": f"could not analyse this capture: {exc}",
                    "detail": traceback.format_exc(limit=3),
                }), 400

        return jsonify(_serialise(result))

    return app


def _serialise(result) -> dict:
    payload = {
        "source": Path(result.source).name,
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
