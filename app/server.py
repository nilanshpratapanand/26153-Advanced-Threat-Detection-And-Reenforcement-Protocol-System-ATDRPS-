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
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, g, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from atdrps.config import Config
from atdrps.data.mitre import MITRE_TACTICS
from atdrps.engine.inference import ThreatForecastEngine, serialise_result
from atdrps.live import dumpcap as dumpcap_mod

__all__ = ["create_app", "WORKSPACE"]

ALLOWED = {".pcap", ".pcapng", ".cap", ".dmp", ".csv"}
MAX_HORIZON = 64

# Every inline <script> carries a per-response nonce, so an injected script
# without it will not run.  Styles stay 'unsafe-inline' because the page sets
# style attributes; styles cannot execute code.
_CSP = ("default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'")


def _hostname(host_header: str) -> str:
    """'localhost:8501' -> 'localhost';  '[::1]:8501' -> '::1'."""
    h = (host_header or "").strip().lower()
    if h.startswith("["):
        return h[1:h.index("]")] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


# an interface name or number as `dumpcap -D` prints it; a leading '-' would read as an option
_INTERFACE_OK = re.compile(r"^(?!-)[\w .:/\\{}()\[\]@#-]{1,256}$")


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

# Captures and generated files land here rather than in a temporary directory:
# the whole point of the buttons is that the person keeps the file, opens it in
# Wireshark, re-runs it later, or hands it to a judge.
WORKSPACE = Path("data/captures")


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

    loopback_only = str(cfg.server.host) in {"127.0.0.1", "localhost", "::1"}
    allowed_hosts = {"localhost", "127.0.0.1", "::1"} | {
        _hostname(h) for h in os.environ.get("ATDRPS_ALLOWED_HOSTS", "").split(",") if h.strip()}

    @app.before_request
    def _assign_nonce():
        g.csp_nonce = secrets.token_urlsafe(16)

    @app.before_request
    def _reject_foreign_host():
        """DNS-rebinding guard: a page on another origin can point its own hostname at
        127.0.0.1 and then talk to this server as if it were same-origin.  When the dashboard
        is bound to loopback, only requests addressed to a loopback name are served
        (add names with ATDRPS_ALLOWED_HOSTS=a.example,b.example)."""
        if loopback_only and _hostname(request.host) not in allowed_hosts:
            return jsonify({"error": "unexpected Host header"}), 400

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

    # ------------------------------------------------------------------
    # Getting a capture in the first place.
    #
    # The upload box assumes a person already has a pcap. Two things produce
    # one: Wireshark's dumpcap, for real traffic off a real interface, and the
    # in-tree generator, for a labelled capture that works with no network and
    # no privilege at all -- which is what a demo needs as its fallback.
    #
    # A capture is a long-running job (minutes), so it runs on a thread and the
    # page polls. The alternative, a request held open for five minutes, is a
    # request that any proxy or browser is entitled to drop halfway through.
    # ------------------------------------------------------------------
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()

    def _trim_jobs(keep: int = 50) -> None:
        """Forget the oldest finished jobs so a long-running dashboard cannot grow without bound."""
        finished = [k for k, v in jobs.items() if v.get("state") in {"done", "error"}]
        for k in finished[:max(0, len(jobs) - keep)]:
            jobs.pop(k, None)

    def _analyse_path(path: Path, threshold: float, horizon, explain: bool) -> dict:
        with analysis_lock:
            eng = copy.copy(engine())            # never mutate the shared engine
            eng.threshold = threshold
            result = eng.analyse(path, horizon=horizon, explain=explain)
        payload = serialise_result(result)
        payload["source"] = Path(result.source).name
        return payload

    def _job_options(body: dict) -> tuple[float, object, bool]:
        """Validated like the upload form: a bad value is a 400 (``_BadParameter``), not a 500."""
        raw_t, raw_h = body.get("threshold"), body.get("horizon")
        threshold = _parse_threshold(None if raw_t is None else str(raw_t))
        horizon = _parse_horizon(None if raw_h in (None, "") else str(raw_h))
        return threshold, horizon, bool(body.get("explain", True))

    @app.route("/api/capture/interfaces")
    def capture_interfaces():
        try:
            binary = dumpcap_mod.find_dumpcap()          # never a path taken from the request
            interfaces = dumpcap_mod.list_interfaces(binary)
        except dumpcap_mod.DumpcapError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 503
        return jsonify({
            "ok": True, "dumpcap": str(binary),
            "interfaces": [i.as_dict() for i in interfaces],
            "min_seconds": dumpcap_mod.MIN_SECONDS,
            "max_seconds": dumpcap_mod.MAX_SECONDS,
        })

    @app.route("/api/capture/start", methods=["POST"])
    def capture_start():
        body = request.get_json(silent=True) or {}
        interface = str(body.get("interface", "")).strip()
        try:
            seconds = int(body.get("seconds", 60))
        except (TypeError, ValueError):
            return jsonify({"error": "seconds must be a number"}), 400
        if not interface:
            return jsonify({"error": "pick a capture interface first"}), 400
        if not _INTERFACE_OK.match(interface):
            return jsonify({"error": "that is not a valid capture interface name"}), 400
        if not dumpcap_mod.MIN_SECONDS <= seconds <= dumpcap_mod.MAX_SECONDS:
            return jsonify({
                "error": f"duration must be {dumpcap_mod.MIN_SECONDS}-"
                         f"{dumpcap_mod.MAX_SECONDS} seconds"
            }), 400
        try:
            binary = dumpcap_mod.find_dumpcap()      # never a path taken from the request
        except dumpcap_mod.DumpcapError as exc:
            return jsonify({"error": str(exc)}), 503

        try:
            threshold, horizon, explain = _job_options(body)
        except _BadParameter as exc:
            return jsonify({"error": str(exc)}), 400
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = WORKSPACE / f"live-{stamp}.pcapng"
        job_id = uuid.uuid4().hex

        with jobs_lock:
            _trim_jobs()
            jobs[job_id] = {"state": "capturing", "seconds": seconds,
                            "started": time.time(), "file": str(out),
                            "kind": "capture"}

        def work():
            try:
                outcome = dumpcap_mod.capture(interface, seconds, out, dumpcap=binary)
                with jobs_lock:
                    jobs[job_id].update(state="analysing", capture=outcome.as_dict())
                payload = _analyse_path(out, threshold, horizon, explain)
                with jobs_lock:
                    jobs[job_id].update(state="done", result=payload)
            except Exception as exc:
                app.logger.exception("live capture failed")
                with jobs_lock:
                    jobs[job_id].update(state="error", error=str(exc))

        threading.Thread(target=work, daemon=True, name=f"capture-{job_id}").start()
        return jsonify({"job": job_id, "seconds": seconds, "file": str(out)})

    @app.route("/api/capture/generate", methods=["POST"])
    def capture_generate():
        """A labelled synthetic capture, as .pcap or as a flow CSV.

        Clearly labelled as synthetic everywhere it surfaces: it is development
        and demo data, and presenting generated traffic as a real capture is
        the one thing this project must never do.
        """
        body = request.get_json(silent=True) or {}
        kind = str(body.get("kind", "pcap")).lower()
        if kind not in {"pcap", "pcapng", "csv"}:
            return jsonify({"error": "kind must be pcap, pcapng or csv"}), 400
        try:
            seconds = int(body.get("seconds", 1800))
            campaigns = int(body.get("campaigns", 2))
            seed = int(body.get("seed", 4242))
        except (TypeError, ValueError):
            return jsonify({"error": "seconds, campaigns and seed must be numbers"}), 400
        if not 120 <= seconds <= 7200:
            return jsonify({"error": "duration must be 120-7200 seconds"}), 400
        if not 0 <= campaigns <= 10:
            return jsonify({"error": "campaigns must be 0-10"}), 400

        try:
            threshold, horizon, explain = _job_options(body)
        except _BadParameter as exc:
            return jsonify({"error": str(exc)}), 400
        job_id = uuid.uuid4().hex
        with jobs_lock:
            _trim_jobs()
            jobs[job_id] = {"state": "generating", "seconds": seconds,
                            "started": time.time(), "kind": "generate"}

        def work():
            try:
                from atdrps.data.flows import assemble_flows
                from atdrps.data.pcap import write_pcap
                from atdrps.data.schema import PacketTable
                from atdrps.data.synth import generate_capture

                WORKSPACE.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                cap = generate_capture(seed=seed, duration_s=float(seconds),
                                       n_campaigns=campaigns)
                if kind == "csv":
                    out = WORKSPACE / f"SYNTHETIC-{stamp}.csv"
                    flows = assemble_flows(PacketTable.from_records(cap.packets))
                    flows.to_csv(out, index=False)
                else:
                    out = WORKSPACE / f"SYNTHETIC-{stamp}.pcap"
                    write_pcap(out, cap.sorted_packets())
                with jobs_lock:
                    jobs[job_id].update(state="analysing", file=str(out),
                                        synthetic=True,
                                        packets=len(cap.packets))
                payload = _analyse_path(out, threshold, horizon, explain)
                payload["synthetic"] = True
                payload["ground_truth"] = [
                    {"stage": iv.stage, "start": iv.start, "end": iv.end,
                     "note": iv.note}
                    for iv in cap.timeline
                ]
                with jobs_lock:
                    jobs[job_id].update(state="done", result=payload)
            except Exception as exc:
                app.logger.exception("capture generation failed")
                with jobs_lock:
                    jobs[job_id].update(state="error", error=str(exc))

        threading.Thread(target=work, daemon=True, name=f"generate-{job_id}").start()
        return jsonify({"job": job_id, "kind": kind, "seconds": seconds})

    @app.route("/api/capture/job/<job_id>")
    def capture_job(job_id: str):
        with jobs_lock:
            job = jobs.get(job_id)
            job = dict(job) if job else None
        if job is None:
            return jsonify({"error": "no such job"}), 404
        if job["state"] in {"capturing", "generating"}:
            job["elapsed"] = round(time.time() - job["started"], 1)
        return jsonify(job)

    @app.route("/api/capture/download/<path:name>")
    def capture_download(name: str):
        """Hand back a file this dashboard produced, and only one of those.

        The name is resolved inside the workspace and the result is checked to
        be under it, so ``..`` segments cannot walk out of the directory.
        """
        try:
            root = WORKSPACE.resolve()
            target = (root / name).resolve()
            target.relative_to(root)
        except (ValueError, OSError):
            return jsonify({"error": "not a file this dashboard produced"}), 400
        if not target.is_file():
            return jsonify({"error": "no such file"}), 404
        return send_file(target, as_attachment=True, download_name=target.name)

    return app
