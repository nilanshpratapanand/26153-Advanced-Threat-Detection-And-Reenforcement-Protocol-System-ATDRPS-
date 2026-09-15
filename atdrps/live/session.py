"""Ties live capture to the same windowing/inference pipeline offline mode uses.

A :class:`LiveSession` owns:

* a :class:`~atdrps.live.capture.LiveCapture` -- packets land here in real time;
* a rolling buffer of the last ``history_s`` seconds of those packets;
* a background clock that, every ``window_size_s`` seconds, re-assembles that
  buffer into flows and windows (:func:`atdrps.data.flows.assemble_flows`,
  :func:`atdrps.data.windows.build_windows` -- unchanged from batch mode) and
  runs one forecast through the already-trained model;
* a queue of JSON-serialisable events a web layer can stream out (SSE or
  otherwise) without needing to know anything about sockets or the model.

Nothing here is simulated: every window scored is built from packets this
process actually captured, and every prediction comes from the same
``ThreatForecastEngine`` the batch dashboard and CLI use.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from ..data.flows import assemble_flows
from ..data.schema import PacketRecord, PacketTable
from ..data.windows import build_windows
from ..engine.inference import ThreatForecastEngine, serialise_result
from .capture import CaptureError, LiveCapture

__all__ = ["LiveSession"]


@dataclass
class _State:
    status: str = "idle"        # idle | starting | live | stopped | error
    iface: Optional[str] = None
    started_at: Optional[float] = None
    error: Optional[str] = None
    windows_scored: int = 0


class LiveSession:
    """One live-capture-and-forecast run. Not shared across requests."""

    def __init__(self, engine: ThreatForecastEngine, iface: Optional[str] = None,
                window_size_s: float = 30.0, history_s: float = 300.0,
                tick_s: float = 5.0):
        self.engine = engine
        self.window_size_s = window_size_s
        self.history_s = history_s
        self.tick_s = tick_s
        self.capture = LiveCapture(iface=iface)
        self.state = _State(iface=iface)

        self._packets: deque[PacketRecord] = deque()
        self._lock = threading.Lock()
        self.events: "queue.Queue[dict]" = queue.Queue(maxsize=500)
        self._analysis_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------- control
    def start(self) -> None:
        self.state.status = "starting"
        try:
            self.capture.start(self._on_packet)
        except CaptureError as exc:
            self.state.status = "error"
            self.state.error = str(exc)
            self._emit({"type": "error", "message": str(exc)})
            raise
        self.state.status = "live"
        self.state.started_at = time.time()
        self._stop.clear()
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop, name="atdrps-live-analysis", daemon=True,
        )
        self._analysis_thread.start()
        self._emit({"type": "status", "status": "live", "iface": self.capture.iface,
                    "window_size_s": self.window_size_s, "started_at": self.state.started_at})

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        self.state.status = "stopped"
        self._emit({"type": "status", "status": "stopped"})
        if self._analysis_thread is not None:
            self._analysis_thread.join(timeout=2)

    # --------------------------------------------------------------- feed
    def _on_packet(self, rec: PacketRecord) -> None:
        with self._lock:
            self._packets.append(rec)
            cutoff = rec.ts - self.history_s
            while self._packets and self._packets[0].ts < cutoff:
                self._packets.popleft()

    def _emit(self, event: dict) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()  # drop the oldest rather than block capture
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(event)
            except queue.Full:
                pass

    # ---------------------------------------------------------------- loop
    def _analysis_loop(self) -> None:
        next_tick = time.time() + self.tick_s
        while not self._stop.is_set():
            time.sleep(0.2)
            now = time.time()
            if now < next_tick:
                continue
            next_tick = now + self.tick_s
            try:
                self._run_one_pass()
            except Exception as exc:  # the live loop must survive a bad window
                self._emit({"type": "error", "message": f"analysis error: {exc}"})

    def _run_one_pass(self) -> None:
        with self._lock:
            records = list(self._packets)

        heartbeat = {
            "type": "heartbeat",
            "packets_seen": self.capture.stats.packets_seen,
            "packets_decoded": self.capture.stats.packets_decoded,
            "bytes_seen": self.capture.stats.bytes_seen,
            "buffered_packets": len(records),
            "uptime_s": round(time.time() - (self.state.started_at or time.time()), 1),
        }
        if self.capture.stats.error:
            heartbeat["capture_warning"] = self.capture.stats.error

        if len(records) < 2:
            self._emit(heartbeat)
            return

        table = PacketTable.from_records(records).sort_by_time()
        flows = assemble_flows(table)
        # t_end anchors window edges to wall-clock "now" rather than the last
        # flow's start time -- see build_windows' docstring. Without this a
        # single long-lived flow (one persistent connection) freezes the
        # window count forever even as real time and real packets keep coming.
        states = build_windows(flows, window_size_s=self.window_size_s,
                               history_s=self.history_s, use_labels=False,
                               t_end=time.time())

        context = self.engine.model.context
        if len(states) < context:
            heartbeat["need_windows"] = context
            heartbeat["have_windows"] = len(states)
            self._emit(heartbeat)
            return

        result = self.engine.analyse_states(states, flows, source="live capture",
                                            explain=True, max_flagged=20)
        payload = serialise_result(result)
        payload["type"] = "window"
        payload["packets_seen"] = self.capture.stats.packets_seen
        payload["bytes_seen"] = self.capture.stats.bytes_seen
        payload["uptime_s"] = heartbeat["uptime_s"]
        self.state.windows_scored += 1
        self._emit(payload)
