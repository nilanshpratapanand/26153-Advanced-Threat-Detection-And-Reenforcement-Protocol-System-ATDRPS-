"""A real TCP connect-scan port scanner.

This deliberately does a *connect* scan (a full three-way handshake per
port, then an immediate close) rather than a SYN/half-open scan. The
trade-off is honest: a connect scan is slightly noisier on the wire and a
little slower, but it needs no raw socket and no elevated privilege at all,
so it works even when live packet *capture* can't (no Administrator, no
Npcap, no CAP_NET_RAW). It is also the same technique ``nmap -sT`` uses.

Every result here comes from an actual ``connect()`` call against the real
target -- nothing is guessed or simulated.
"""

from __future__ import annotations

import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

__all__ = ["PortScanner", "PortScanResult"]

# A compact, well-known set -- enough to be informative in the seconds a
# live demo has, without turning into a 65535-port sweep.
COMMON_PORTS: tuple[int, ...] = (
    21, 22, 23, 25, 53, 80, 110, 111, 123, 135, 139, 143, 161, 389, 443,
    445, 465, 587, 631, 993, 995, 1433, 1521, 1723, 2049, 3000, 3306, 3389,
    5432, 5900, 5984, 6379, 8000, 8080, 8443, 8888, 9000, 9090, 9200, 27017,
)

_WELL_KNOWN_NAMES: dict[int, str] = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns", 80: "http",
    110: "pop3", 111: "rpcbind", 123: "ntp", 135: "msrpc", 139: "netbios-ssn",
    143: "imap", 161: "snmp", 389: "ldap", 443: "https", 445: "smb",
    465: "smtps", 587: "submission", 631: "ipp", 993: "imaps", 995: "pop3s",
    1433: "mssql", 1521: "oracle", 1723: "pptp", 2049: "nfs", 3000: "dev-http",
    3306: "mysql", 3389: "rdp", 5432: "postgres", 5900: "vnc", 5984: "couchdb",
    6379: "redis", 8000: "http-alt", 8080: "http-proxy", 8443: "https-alt",
    8888: "http-alt", 9000: "http-alt", 9090: "http-alt", 9200: "elasticsearch",
    27017: "mongodb",
}


@dataclass
class PortScanResult:
    target: str
    port: int
    state: str            # "open" | "closed" | "filtered"
    service: str
    latency_ms: float


@dataclass
class ScanSummary:
    target: str
    started_at: float
    finished_at: float
    ports_scanned: int
    open_ports: list[PortScanResult] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.finished_at - self.started_at


class PortScanner:
    """A concurrent TCP connect scanner against one target."""

    def __init__(self, timeout_s: float = 0.6, max_workers: int = 100):
        self.timeout_s = timeout_s
        self.max_workers = max_workers

    def scan_port(self, target: str, port: int) -> PortScanResult:
        start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        try:
            result = sock.connect_ex((target, port))
            state = "open" if result == 0 else "closed"
        except socket.timeout:
            state = "filtered"
        except OSError:
            state = "filtered"
        finally:
            sock.close()
        return PortScanResult(
            target=target, port=port, state=state,
            service=_WELL_KNOWN_NAMES.get(port, "-"),
            latency_ms=(time.time() - start) * 1000,
        )

    def scan(self, target: str, ports: Iterable[int] = COMMON_PORTS,
             on_result: Optional[Callable[[PortScanResult], None]] = None) -> ScanSummary:
        """Scan ``ports`` on ``target``, streaming each real result as it lands."""
        target = socket.gethostbyname(target) if not _looks_like_ip(target) else target
        started = time.time()
        results: list[PortScanResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.scan_port, target, p): p for p in ports}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if on_result is not None:
                    on_result(res)
        results.sort(key=lambda r: r.port)
        return ScanSummary(
            target=target, started_at=started, finished_at=time.time(),
            ports_scanned=len(results),
            open_ports=[r for r in results if r.state == "open"],
        )


def _looks_like_ip(host: str) -> bool:
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        return False
