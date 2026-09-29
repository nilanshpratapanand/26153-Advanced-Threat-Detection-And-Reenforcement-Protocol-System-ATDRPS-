"""The dashboard's capture endpoints start real processes and write files, so their inputs
are hostile until proven otherwise.  Nothing here needs Wireshark: the calls that would
launch it are replaced, and the tests check *what the server would have run and with what*."""

import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import app.server as server
from atdrps.live import dumpcap as dc


class TestFindDumpcap(unittest.TestCase):
    def test_a_hint_naming_another_program_is_never_returned(self):
        for hint in ("/usr/bin/env", "/bin/sh", "/usr/bin/python3", "C:\\Windows\\System32\\cmd.exe"):
            with self.subTest(hint=hint):
                try:
                    found = dc.find_dumpcap(hint)
                except dc.DumpcapError:
                    continue                                  # nothing legitimate installed: fine
                self.assertIn(found.name.lower(), {"dumpcap", "dumpcap.exe"})

    def test_the_environment_variable_gets_the_same_treatment(self):
        with mock.patch.dict(os.environ, {"ATDRPS_DUMPCAP": "/bin/sh"}):
            try:
                found = dc.find_dumpcap()
            except dc.DumpcapError:
                return
            self.assertIn(found.name.lower(), {"dumpcap", "dumpcap.exe"})

    def test_a_real_looking_dumpcap_hint_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "dumpcap"
            fake.write_text("#!/bin/sh\n")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            self.assertEqual(dc.find_dumpcap(fake), fake)
            self.assertEqual(dc.find_dumpcap(tmp), fake)         # the folder form people paste


class _AppCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._ws = mock.patch.object(server, "WORKSPACE", Path(self._tmp.name) / "captures")
        self._ws.start()
        self.addCleanup(self._ws.stop)
        self.app = server.create_app(model_dir=Path(self._tmp.name) / "no-model")
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()


class TestHintNeverComesFromTheRequest(_AppCase):
    def test_interfaces_ignores_a_dumpcap_query_parameter(self):
        seen = []
        with mock.patch.object(dc, "find_dumpcap", side_effect=lambda hint=None: seen.append(hint) or Path("dumpcap")), \
             mock.patch.object(dc, "list_interfaces", return_value=[]):
            r = self.client.get("/api/capture/interfaces?dumpcap=/usr/bin/env")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(seen, [None])

    def test_start_ignores_a_dumpcap_field_in_the_body(self):
        seen = []
        with mock.patch.object(dc, "find_dumpcap", side_effect=lambda hint=None: seen.append(hint) or Path("dumpcap")), \
             mock.patch.object(dc, "capture", side_effect=dc.DumpcapError("stopped for the test")):
            r = self.client.post("/api/capture/start",
                                 json={"interface": "1", "seconds": 10, "dumpcap": "/usr/bin/env"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(seen, [None])


class TestInputValidation(_AppCase):
    def _start(self, **body):
        base = {"interface": "1", "seconds": 10}
        base.update(body)
        with mock.patch.object(dc, "find_dumpcap", return_value=Path("dumpcap")), \
             mock.patch.object(dc, "capture", side_effect=dc.DumpcapError("stopped for the test")):
            return self.client.post("/api/capture/start", json=base)

    def test_dangerous_interface_names_are_refused(self):
        for bad in ("-w/tmp/x", "--help", "1; rm -rf /", "eth0 && id", "a\nb", "$(id)", "`id`", "x" * 300):
            with self.subTest(interface=bad[:20]):
                self.assertEqual(self._start(interface=bad).status_code, 400)

    def test_ordinary_interface_names_are_accepted(self):
        for ok in ("1", "eth0", "en0", "wlan0", "\\Device\\NPF_{3A5F-01}", "Wi-Fi", "any"):
            with self.subTest(interface=ok):
                self.assertEqual(self._start(interface=ok).status_code, 200)

    def test_bad_threshold_and_horizon_are_400_not_500(self):
        for body in ({"threshold": "abc"}, {"threshold": 5}, {"threshold": float("nan")},
                     {"horizon": "x"}, {"horizon": 0}, {"horizon": 10**6}):
            with self.subTest(body=body):
                self.assertEqual(self._start(**body).status_code, 400)
        gen = self.client.post("/api/capture/generate", json={"threshold": "abc"})
        self.assertEqual(gen.status_code, 400)

    def test_seconds_out_of_range_is_400(self):
        self.assertEqual(self._start(seconds=1).status_code, 400)
        self.assertEqual(self._start(seconds="ten").status_code, 400)


class TestNoInformationLeaks(_AppCase):
    def test_a_failed_job_reports_the_error_but_no_traceback(self):
        with mock.patch("atdrps.data.synth.generate_capture", side_effect=RuntimeError("boom-marker")):
            r = self.client.post("/api/capture/generate", json={"seconds": 120})
            self.assertEqual(r.status_code, 200)
            job = r.get_json()["job"]
            for _ in range(100):
                d = self.client.get(f"/api/capture/job/{job}").get_json()
                if d["state"] == "error":
                    break
                time.sleep(0.05)
        self.assertEqual(d["state"], "error")
        self.assertIn("boom-marker", d["error"])
        self.assertNotIn("detail", d)
        self.assertNotIn("Traceback", str(d))


class TestHostAndPathGuards(_AppCase):
    def test_foreign_host_header_is_refused_on_a_loopback_server(self):
        self.assertEqual(self.client.get("/api/health", headers={"Host": "evil.example"}).status_code, 400)
        self.assertEqual(self.client.get("/api/health", headers={"Host": "evil.example:8501"}).status_code, 400)

    def test_loopback_names_are_served(self):
        for host in ("localhost", "localhost:8501", "127.0.0.1:8501", "[::1]:8501"):
            with self.subTest(host=host):
                self.assertNotEqual(self.client.get("/api/health", headers={"Host": host}).status_code, 400)

    def test_extra_hosts_can_be_allowed_explicitly(self):
        with mock.patch.dict(os.environ, {"ATDRPS_ALLOWED_HOSTS": "soc.internal.example"}):
            app = server.create_app(model_dir=Path(self._tmp.name) / "no-model")
        c = app.test_client()
        self.assertNotEqual(c.get("/api/health", headers={"Host": "soc.internal.example"}).status_code, 400)
        self.assertEqual(c.get("/api/health", headers={"Host": "other.example"}).status_code, 400)

    def test_download_cannot_walk_out_of_the_workspace(self):
        secret = Path(self._tmp.name) / "secret.txt"
        secret.write_text("TOP-SECRET-MARKER")
        for name in ("../secret.txt", "..%2fsecret.txt", "%2e%2e/secret.txt", "../../../../etc/passwd",
                     "captures/../../secret.txt"):
            with self.subTest(name=name):
                r = self.client.get(f"/api/capture/download/{name}")
                self.assertNotIn(b"TOP-SECRET-MARKER", r.data)
                self.assertNotIn(b"root:", r.data)
                self.assertIn(r.status_code, (400, 404))

    def test_download_serves_a_file_the_dashboard_produced(self):
        server.WORKSPACE.mkdir(parents=True, exist_ok=True)
        (server.WORKSPACE / "SYNTHETIC-x.pcap").write_bytes(b"pcap-bytes")
        r = self.client.get("/api/capture/download/SYNTHETIC-x.pcap")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data, b"pcap-bytes")


if __name__ == "__main__":
    unittest.main()
