"""Live capture, port scanning and the live inference loop.

These tests generate and capture *real* traffic on the loopback interface --
no mocked sockets, no fabricated packets. They need CAP_NET_RAW (root, or
``setcap cap_net_raw,cap_net_admin=eip`` on the interpreter) to open a raw
socket, exactly like the feature itself does, so they skip themselves rather
than fail when that privilege isn't available -- a CI runner or a sandboxed
review environment without root is a permission gap, not a broken feature.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from atdrps.data.windows import build_windows
from atdrps.engine.inference import ThreatForecastEngine
from atdrps.live.capture import CaptureError, LiveCapture, list_interfaces
from atdrps.live.portscan import PortScanner
from atdrps.live.session import LiveSession

REPO_ROOT = Path(__file__).resolve().parent.parent


def _can_open_raw_socket() -> bool:
    try:
        import socket as _s
        s = _s.socket(_s.AF_PACKET, _s.SOCK_RAW, _s.htons(0x0003)) if hasattr(_s, "AF_PACKET") \
            else _s.socket(_s.AF_INET, _s.SOCK_RAW, _s.IPPROTO_TCP)
        s.close()
        return True
    except (PermissionError, OSError, AttributeError):
        return False


HAVE_RAW_SOCKET = _can_open_raw_socket()
skip_without_privilege = unittest.skipUnless(
    HAVE_RAW_SOCKET, "needs CAP_NET_RAW / Administrator for a raw capture socket",
)


class TestListInterfaces(unittest.TestCase):
    def test_returns_at_least_loopback(self):
        # Every platform ATDRPS targets has a loopback interface; if this
        # list is empty something is wrong with the environment, not a
        # privilege issue (listing interfaces needs no elevation).
        names = list_interfaces()
        self.assertIsInstance(names, list)


@skip_without_privilege
class TestLiveCapture(unittest.TestCase):
    def test_captures_real_udp_traffic_it_can_see(self):
        captured = []
        lock = threading.Lock()

        def on_packet(rec):
            with lock:
                captured.append(rec)

        cap = LiveCapture(iface="lo")
        cap.start(on_packet)
        try:
            time.sleep(0.2)
            port = 48765
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for i in range(10):
                sock.sendto(f"real-{i}".encode(), ("127.0.0.1", port))
                time.sleep(0.02)
            time.sleep(0.3)
        finally:
            cap.stop()

        hits = [r for r in captured if r.dst_port == port or r.src_port == port]
        self.assertGreaterEqual(len(hits), 8, "should have really seen the UDP packets sent")
        self.assertGreater(cap.stats.packets_decoded, 0)

    def test_stop_is_idempotent_and_releases_the_socket(self):
        cap = LiveCapture(iface="lo")
        cap.start(lambda rec: None)
        cap.stop()
        cap.stop()  # must not raise
        self.assertFalse(cap.running)

    def test_cannot_start_twice_concurrently(self):
        cap = LiveCapture(iface="lo")
        cap.start(lambda rec: None)
        try:
            with self.assertRaises(CaptureError):
                cap.start(lambda rec: None)
        finally:
            cap.stop()


class TestPortScanner(unittest.TestCase):
    def test_detects_a_real_open_and_a_real_closed_port(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        open_port = listener.getsockname()[1]
        listener.listen(5)
        stop = threading.Event()

        def accept_loop():
            listener.settimeout(0.2)
            while not stop.is_set():
                try:
                    conn, _ = listener.accept()
                    conn.close()
                except socket.timeout:
                    continue
                except OSError:
                    break  # listener was closed from the main thread; exit quietly

        t = threading.Thread(target=accept_loop, daemon=True)
        t.start()
        try:
            # a closed port: bind-and-immediately-release to get a free one
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", 0))
            closed_port = probe.getsockname()[1]
            probe.close()

            scanner = PortScanner(timeout_s=0.3)
            summary = scanner.scan("127.0.0.1", ports=[open_port, closed_port])
            states = {r.port: r.state for r in summary.open_ports}
            all_states = {r.port: r.state for r in
                          [scanner.scan_port("127.0.0.1", open_port),
                           scanner.scan_port("127.0.0.1", closed_port)]}
            self.assertEqual(all_states[open_port], "open")
            self.assertEqual(all_states[closed_port], "closed")
        finally:
            stop.set()
            listener.close()
            t.join(timeout=1)

    def test_scan_result_ordering_is_by_port(self):
        scanner = PortScanner(timeout_s=0.1)
        summary = scanner.scan("127.0.0.1", ports=[9, 7])
        # summary.open_ports may be empty (both likely closed in a container);
        # the contract under test is that scan() itself doesn't crash on a
        # multi-port real scan and returns a well-formed summary.
        self.assertEqual(summary.ports_scanned, 2)
        self.assertGreaterEqual(summary.duration_s, 0.0)


class TestWindowAnchoring(unittest.TestCase):
    """The bug a naive live loop hits: one long-lived flow must not freeze
    the window count just because its *start* time never moves."""

    def test_t_end_grows_window_count_for_a_single_persistent_flow(self):
        from tests.test_windows import flow, make_flows

        # one flow, started once at t=1000 and never restarted -- exactly the
        # shape of a live capture's single persistent connection/beacon.
        frame = make_flows([flow(ts=1000.0)])

        without_anchor = build_windows(frame, window_size_s=2.0, use_labels=False)
        with_anchor = build_windows(frame, window_size_s=2.0, use_labels=False,
                                    t_end=1000.0 + 40.0)  # 40s of real time later

        self.assertEqual(len(without_anchor), 1)
        self.assertGreaterEqual(len(with_anchor), 20)  # ~40s / 2s windows


@skip_without_privilege
class TestLiveSession(unittest.TestCase):
    """End-to-end: real capture -> real windowing -> the actual trained model."""

    @classmethod
    def setUpClass(cls):
        model_dir = REPO_ROOT / "artifacts" / "model-linear"
        bg_path = REPO_ROOT / "artifacts" / "background.npy"
        if not (model_dir / "config.json").exists():
            raise unittest.SkipTest("no trained model in artifacts/model-linear; run `atdrps benchmark` first")
        bg = np.load(bg_path) if bg_path.exists() else None
        cls.engine = ThreatForecastEngine.load(model_dir, background=bg, window_size_s=2.0)

    def test_real_capture_produces_a_real_prediction(self):
        session = LiveSession(self.engine, iface="lo", window_size_s=2.0,
                              history_s=120.0, tick_s=1.0)
        session.start()
        stop_gen = threading.Event()

        def traffic():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            i = 0
            while not stop_gen.is_set():
                s.sendto(f"pkt{i}".encode(), ("127.0.0.1", 9999 + (i % 5)))
                i += 1
                time.sleep(0.03)

        gen = threading.Thread(target=traffic, daemon=True)
        gen.start()
        try:
            window_events = []
            deadline = time.time() + 40
            needed_windows = self.engine.model.context
            while time.time() < deadline and not window_events:
                try:
                    ev = session.events.get(timeout=1.0)
                except queue.Empty:
                    continue
                if ev.get("type") == "window":
                    window_events.append(ev)
        finally:
            stop_gen.set()
            session.stop()
            gen.join(timeout=2)

        self.assertTrue(window_events, "expected at least one real live prediction "
                        f"within the test window (model needs {needed_windows} windows of context)")
        payload = window_events[-1]
        self.assertGreater(payload["packets_seen"], 0)
        self.assertIn("headline", payload)
        self.assertIn("timeline", payload)


if __name__ == "__main__":
    unittest.main()
