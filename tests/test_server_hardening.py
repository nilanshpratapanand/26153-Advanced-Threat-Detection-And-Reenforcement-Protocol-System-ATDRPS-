"""The upload endpoint takes untrusted input; these pin down how it fails."""

import io
import struct
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from tests.test_engine import HORIZON, _Fixture


class TestServerHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _Fixture.setup()
        from app.server import create_app

        cls.model_dir = tempfile.mkdtemp()
        _Fixture.model.save(cls.model_dir)
        np.save(Path(cls.model_dir) / "background.npy", _Fixture.engine.explainer.background)
        cls.app = create_app(model_dir=cls.model_dir)
        cls.app.config.update(TESTING=True)
        cls.client = cls.app.test_client()

    def _post(self, filename="demo.pcap", payload=None, **form):
        if payload is None:
            with open(_Fixture.pcap, "rb") as fh:
                payload = fh.read()
        data = {"capture": (io.BytesIO(payload), filename), **form}
        return self.client.post("/api/analyse", data=data, content_type="multipart/form-data")

    def test_non_numeric_threshold_is_a_400_not_a_500(self):
        for bad in ("abc", "nan", "inf", "-0.1", "1.5", ""):
            with self.subTest(threshold=bad):
                r = self._post(threshold=bad)
                self.assertEqual(r.status_code, 400)
                self.assertIn("threshold", r.get_json()["error"])

    def test_bad_horizon_is_a_400(self):
        for bad in ("x", "0", "-3", "1.5", "100000"):
            with self.subTest(horizon=bad):
                r = self._post(horizon=bad)
                self.assertEqual(r.status_code, 400)
                self.assertIn("horizon", r.get_json()["error"])

    def test_valid_parameters_still_work(self):
        r = self._post(threshold="0.6", horizon=str(HORIZON))
        self.assertEqual(r.status_code, 200)

    def test_request_threshold_does_not_leak_into_shared_engine(self):
        first = self.client.get("/api/health")
        self.assertEqual(first.status_code, 200)
        state_engine = self._engine()
        before = state_engine.threshold
        self._post(threshold="0.93")
        self.assertEqual(state_engine.threshold, before)

    def _engine(self):
        # the closure holds the engine; reach it through the health route's side effect
        for cell in self.app.view_functions["health"].__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value) and getattr(value, "__name__", "") == "engine":
                return value()
        self.fail("could not locate the shared engine")

    def test_errors_do_not_leak_a_traceback(self):
        r = self._post(filename="junk.pcap", payload=b"\x00" * 64)
        self.assertEqual(r.status_code, 400)
        body = r.get_json()
        self.assertNotIn("detail", body)
        self.assertNotIn("Traceback", r.get_data(as_text=True))

    def test_hostile_filenames_do_not_break_saving(self):
        for name in ("..", "../../etc/passwd.csv", "a/b\\c.pcap", "x" * 300 + ".pcap"):
            with self.subTest(name=name[:20]):
                r = self._post(filename=name, payload=b"\x00" * 64)
                # a junk .csv is accepted with a "not enough data" note, a junk
                # capture is rejected; what must never happen is a 5xx
                self.assertIn(r.status_code, (200, 400))
                self.assertEqual(Path(r.get_json().get("source", "x")).name,
                                 r.get_json().get("source", "x"))

    def test_zero_length_pcapng_section_returns_400_promptly(self):
        hostile = struct.pack("<II", 0x0A0D0D0A, 0) + struct.pack("<I", 0x1A2B3C4D) + bytes(20)
        result = {}

        def run():
            result["r"] = self._post(filename="evil.pcapng", payload=hostile)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=10)
        self.assertFalse(t.is_alive(), "server hung on a malformed pcapng")
        self.assertEqual(result["r"].status_code, 400)

    def test_oversize_upload_is_a_json_413(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 1024
        try:
            r = self._post(payload=b"x" * 5000)
        finally:
            self.app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
        self.assertEqual(r.status_code, 413)
        self.assertIn("error", r.get_json())

    def test_security_headers_are_set(self):
        for path in ("/", "/api/health"):
            h = self.client.get(path).headers
            self.assertEqual(h.get("X-Content-Type-Options"), "nosniff")
            self.assertEqual(h.get("X-Frame-Options"), "DENY")
            self.assertEqual(h.get("Referrer-Policy"), "no-referrer")
            self.assertIn("frame-ancestors 'none'", h.get("Content-Security-Policy", ""))


if __name__ == "__main__":
    unittest.main()
