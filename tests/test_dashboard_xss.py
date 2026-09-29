"""Drive the real dashboard in a real browser with hostile server responses.

Skipped automatically when Playwright or a Chromium build is not installed,
so the standard-library-only test run still works.  The escaping is checked
with the CSP switched off as well as on: the CSP is a second layer, and the
page must be safe without leaning on it.
"""

import glob
import os
import tempfile
import threading
import unittest

try:
    from playwright.sync_api import sync_playwright
except ImportError:                                   # pragma: no cover
    sync_playwright = None

EVIL = '<img src=x onerror="window.__pwn=(window.__pwn||0)+1">'


def _chromium():
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")]
    for root in roots:
        for pattern in ("chromium-*/chrome-linux/chrome", "chromium/chrome-linux/chrome",
                        "chromium-*/chrome-win/chrome.exe", "chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium"):
            hits = sorted(glob.glob(os.path.join(root, pattern)))
            if hits:
                return hits[-1]
    return None


def _payload(stage):
    timeline = [{"window": i, "infiltration_probability": (0.1 * i) % 1, "stage": "Benign",
                 "stage_confidence": .9, "alert": False, "level": "CLEAR", "risk": .1}
                for i in range(10)]
    timeline[3]["stage"] = stage
    return {"source": EVIL, "source_kind": EVIL, "headline": EVIL, "n_windows": 10, "n_flows": 3,
            "window_size_s": 30, "peak_window": 3, "peak_risk": .5, "notes": [EVIL],
            "timeline": timeline,
            "forecast": [{"step": 1, "infiltration_probability": .5, "stage": stage,
                          "stage_confidence": .5}],
            "flagged_flows": [{"src": EVIL, "dst": EVIL, "protocol": EVIL, "packets": 1,
                               "bytes": 5, "duration": 1, "flags": EVIL,
                               "payload_zero_ratio": 0, "retransmissions": 0}],
            "novelty": {"verdict": "outside", "message": EVIL, "median_fraction": 0.9,
                        "worst_features": []},
            "explanation": {"stage": stage, "stage_description": EVIL, "prediction": .5,
                            "base_value": .1,
                            "drivers": [{"description": EVIL, "contribution": .3}],
                            "key_windows": [{"windows_ago": 1, "weight": .5}],
                            "temporal_source": EVIL, "temporal": [0.1], "rules": [EVIL],
                            "efficiency_error": 1e-9, "rule_scores": {}}}


@unittest.skipIf(sync_playwright is None or _chromium() is None,
                 "playwright and a chromium build are required")
class TestDashboardEscaping(unittest.TestCase):
    def _run(self, csp_off):
        import app.server as srv
        from werkzeug.serving import make_server

        class _Model:
            name = "dummy"; context = 8; horizon = 3; n_features = 1

        class _Engine:
            model = _Model()

        original_load, original_csp = srv.ThreatForecastEngine.load, srv._CSP
        srv.ThreatForecastEngine.load = staticmethod(lambda *a, **k: _Engine())
        if csp_off:
            srv._CSP = "default-src * 'unsafe-inline'"
        server = None
        try:
            app = srv.create_app(model_dir=tempfile.mkdtemp())
            server = make_server("127.0.0.1", 0, app, threaded=True)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return self._drive(server.server_port)
        finally:
            srv.ThreatForecastEngine.load, srv._CSP = original_load, original_csp
            if server is not None:
                server.shutdown()

    def _drive(self, port):
        out = {}
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=_chromium(), args=["--no-sandbox"])
            try:
                for mode in ("payload", "error", "filename", "benign", "iface"):
                    page = browser.new_page()
                    page.route("**/api/health", lambda r, _q: r.fulfill(json={"ok": True, "model": "d"}))

                    def make(mode):
                        def handler(route, _request):
                            if mode == "error":
                                route.fulfill(status=400, json={"error": "bad column " + EVIL})
                            else:
                                route.fulfill(json=_payload("Reconnaissance" if mode == "benign" else EVIL))
                        return handler

                    page.route("**/api/analyse", make(mode))
                    page.route("**/api/capture/interfaces", lambda r, _q: r.fulfill(
                        status=503, json={"ok": False, "error": "no dumpcap " + EVIL}))
                    page.goto(f"http://127.0.0.1:{port}/")
                    name = ("x" + EVIL + ".csv") if mode == "filename" else "a.csv"
                    path = os.path.join(tempfile.mkdtemp(), name)
                    with open(path, "w") as fh:
                        fh.write("a,b\n1,2\n")
                    if mode == "iface":
                        page.click("#ifacesBtn")
                        page.wait_for_timeout(600)
                    else:
                        page.set_input_files("#capture", path)
                        page.click("#go")
                        page.wait_for_function(
                            "document.querySelector('#go').disabled === false", timeout=5000)
                    out[mode] = {
                        "executed": page.evaluate("window.__pwn || 0"),
                        "injected": page.evaluate(
                            "document.querySelectorAll('#results img, #status img, #sourceStatus img, "
                            "#oodBox img, #verdictText img').length"),
                        "visible": page.evaluate(
                            "!document.querySelector('#results').classList.contains('hidden')"),
                        "status": page.evaluate("document.querySelector('#status').innerText"),
                        "source_status": page.evaluate("document.querySelector('#sourceStatus').innerText"),
                    }
                    page.close()
            finally:
                browser.close()
        return out

    def _check(self, csp_off):
        out = self._run(csp_off)
        for mode, r in out.items():
            self.assertEqual(r["executed"], 0, f"{mode}: injected script ran")
            self.assertEqual(r["injected"], 0, f"{mode}: attacker markup reached the DOM")
        self.assertTrue(out["benign"]["visible"], "benign results did not render")
        self.assertIn("no dumpcap <img", out["iface"]["source_status"])   # shown literally
        self.assertTrue(out["payload"]["visible"])
        self.assertFalse(out["error"]["visible"])
        self.assertIn("bad column <img", out["error"]["status"])   # shown literally

    def test_escaping_alone_blocks_injection(self):
        self._check(csp_off=True)

    def test_escaping_with_csp(self):
        self._check(csp_off=False)


if __name__ == "__main__":
    unittest.main()
