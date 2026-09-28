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

import copy
import math
import secrets
import tempfile
import threading
from pathlib import Path

import numpy as np
from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from atdrps.config import Config
from atdrps.data.mitre import MITRE_TACTICS
from atdrps.engine.inference import ThreatForecastEngine, serialise_result

__all__ = ["create_app"]

ALLOWED = {".pcap", ".pcapng", ".cap", ".dmp", ".csv"}
MAX_HORIZON = 64

# Every inline <script> carries a per-response nonce, so an injected script
# without it will not run.  Styles stay 'unsafe-inline' because the page sets
# style attributes; styles cannot execute code.
_CSP = ("default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'")


class _BadParameter(ValueError):
    """A form field the client got wrong -- reported as a 400, never a 500."""


def _parse_threshold(raw: str | None) -> float:
    if raw is None:
        return 0.5
    try:
        value = float(raw)
    except ValueError:
        raise _BadParameter(f"threshold must be a number between 0 and 1, got {raw!r}") from None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise _BadParameter(f"threshold must be between 0 and 1, got {raw!r}")
    return value


def _parse_horizon(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise _BadParameter(f"horizon must be a whole number, got {raw!r}") from None
    if not 1 <= value <= MAX_HORIZON:
        raise _BadParameter(f"horizon must be between 1 and {MAX_HORIZON}, got {value}")
    return value


def create_app(model_dir: str | Path = "artifacts/model-linear",
               config: Config | None = None) -> Flask:
    cfg = config or Config.load()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = int(cfg.server.max_upload_mb) * 1024 * 1024

    model_dir = Path(model_dir)
    state: dict = {"engine": None, "error": None, "model_dir": str(model_dir)}
    # Analyses are serialised: the model and explainer are shared, and one
    # upload already uses every core, so concurrency buys nothing but a way for
    # a few large uploads to starve the machine.
    analysis_lock = threading.Lock()

    @app.before_request
    def _assign_nonce():
        g.csp_nonce = secrets.token_urlsafe(16)

    @app.after_request
    def _security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = _CSP.format(nonce=g.get("csp_nonce", ""))
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_exc):
        return jsonify({"error": f"upload exceeds the {cfg.server.max_upload_mb} MB limit"}), 413

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
            model_dir=str(model_dir), window_size=cfg.window.size_s, nonce=g.csp_nonce,
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

        try:
            threshold = _parse_threshold(request.form.get("threshold"))
            horizon = _parse_horizon(request.form.get("horizon"))
        except _BadParameter as exc:
            return jsonify({"error": str(exc)}), 400
        explain = request.form.get("explain", "1") != "0"

        with tempfile.TemporaryDirectory() as tmp:
            # never trust the client's filename for a path: only its suffix is used
            path = Path(tmp) / f"upload{suffix}"
            upload.save(path)
            try:
                with analysis_lock:
                    # a per-request copy, so one caller's threshold cannot
                    # bleed into the next caller's analysis
                    eng = copy.copy(engine())
                    eng.threshold = threshold
                    result = eng.analyse(path, horizon=horizon, explain=explain)
            except RuntimeError as exc:
                return jsonify({"error": str(exc)}), 503
            except Exception as exc:            # a bad capture must not 500 silently
                app.logger.exception("analysis failed for %r", upload.filename)
                return jsonify({
                    "error": f"could not analyse this capture: {exc}",
                }), 400

        payload = serialise_result(result)
        payload["source"] = Path(upload.filename).name
        return jsonify(payload)

    return app
