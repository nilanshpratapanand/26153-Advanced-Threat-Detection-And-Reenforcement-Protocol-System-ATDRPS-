"""Live packet capture, no third-party packet library.

``atdrps/data/pcap.py`` already refuses to depend on scapy or pyshark, for an
air-gapped-CII reason documented right there: every extra package is another
thing to vet, ship and patch.  Live capture keeps that promise -- both
backends below use nothing but the standard library socket module.

Two platform-native paths:

* **Linux** -- an ``AF_PACKET``/``SOCK_RAW`` socket bound to an interface
  receives full Ethernet frames.  Needs ``CAP_NET_RAW`` (root, or
  ``setcap cap_net_raw,cap_net_admin=eip $(readlink -f $(which python3))``
  once, so the interpreter itself carries the capability instead of the
  user needing ``sudo`` every run).
* **Windows** -- raw Ethernet capture is not available without a third-party
  driver (Npcap).  What *is* available in plain Python is a raw IP socket
  bound to a local adapter address with promiscuous mode turned on via the
  ``SIO_RCVALL`` ioctl -- this is the standard "no-Npcap" technique and it
  needs an elevated (Administrator) process, nothing else.  It hands back
  bare IP datagrams with no Ethernet header, which is exactly the
  ``LINKTYPE_RAW`` case the offline pcap reader already handles.

Both paths converge on :func:`atdrps.data.pcap.decode_frame`, so a live
packet is parsed by the identical code a saved ``.pcap`` is.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..data.pcap import LINKTYPE_ETHERNET, LINKTYPE_RAW, decode_frame
from ..data.schema import PacketRecord

__all__ = ["CaptureError", "LiveCapture", "list_interfaces", "CaptureStats"]

_ETH_P_ALL = 0x0003


class CaptureError(RuntimeError):
    """Live capture could not start -- missing privilege or bad interface."""


def _elevation_hint() -> str:
    if sys.platform == "win32":
        return ("live capture needs an elevated process. Close this window and "
                "re-open PowerShell/cmd with \"Run as administrator\", then run "
                "the same command again.")
    return ("live capture needs CAP_NET_RAW. Run with sudo, or once run: "
            "sudo setcap cap_net_raw,cap_net_admin=eip "
            "$(readlink -f $(which python3)) — after that no sudo is needed.")


def list_interfaces() -> list[str]:
    """Best-effort list of local network interface names."""
    try:
        return sorted(name for _, name in socket.if_nameindex())
    except (AttributeError, OSError):
        # if_nameindex isn't available on this platform/build; fall back to
        # "whatever address this host resolves to", which is what the
        # Windows raw-IP backend binds to anyway.
        try:
            return [socket.gethostbyname(socket.gethostname())]
        except OSError:
            return []


@dataclass
class CaptureStats:
    packets_seen: int = 0
    bytes_seen: int = 0
    packets_decoded: int = 0
    started_at: Optional[float] = None
    error: Optional[str] = None


class LiveCapture:
    """Sniffs packets on a background thread, calling ``on_packet`` for each one.

    Every packet handed to the callback is a real :class:`PacketRecord`
    decoded from bytes that were actually on the wire a moment ago -- this
    class never fabricates or replays traffic.
    """

    def __init__(self, iface: Optional[str] = None, snaplen: int = 65535,
                read_timeout_s: float = 1.0):
        self.iface = iface
        self.snaplen = snaplen
        self.read_timeout_s = read_timeout_s
        self.stats = CaptureStats()
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------- control
    def start(self, on_packet: Callable[[PacketRecord], None]) -> None:
        if self.running:
            raise CaptureError("capture is already running")
        self._sock = self._open_socket()
        self.stats = CaptureStats(started_at=time.time())
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(on_packet,), name="atdrps-live-capture", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                if sys.platform == "win32":
                    self._sock.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)  # type: ignore[attr-defined]
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    # -------------------------------------------------------------- socket
    def _open_socket(self) -> socket.socket:
        try:
            return self._open_windows_socket() if sys.platform == "win32" else self._open_linux_socket()
        except PermissionError as exc:
            raise CaptureError(_elevation_hint()) from exc
        except OSError as exc:
            # AF_PACKET is Linux-only; anything else unsupported (e.g. no
            # if_nameindex entry for the requested interface) surfaces here too.
            raise CaptureError(f"could not open a capture socket: {exc}") from exc

    def _open_linux_socket(self) -> socket.socket:
        s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(_ETH_P_ALL))
        if self.iface:
            s.bind((self.iface, 0))
        s.settimeout(self.read_timeout_s)
        return s

    def _open_windows_socket(self) -> socket.socket:
        host = self.iface or socket.gethostbyname(socket.gethostname())
        s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
        s.bind((host, 0))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
        s.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)  # type: ignore[attr-defined]
        s.settimeout(self.read_timeout_s)
        return s

    # ---------------------------------------------------------------- loop
    def _loop(self, on_packet: Callable[[PacketRecord], None]) -> None:
        linktype = LINKTYPE_RAW if sys.platform == "win32" else LINKTYPE_ETHERNET
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data = self._sock.recv(self.snaplen)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self.stats.error = str(exc)
                break
            if not data:
                continue
            ts = time.time()
            self.stats.packets_seen += 1
            self.stats.bytes_seen += len(data)
            try:
                rec = decode_frame(linktype, data, ts)
            except Exception as exc:  # a malformed frame must not kill the thread
                self.stats.error = f"decode error (continuing): {exc}"
                continue
            if rec is not None:
                self.stats.packets_decoded += 1
                on_packet(rec)
