"""Capture traffic by driving Wireshark's ``dumpcap``, and nothing else.

``atdrps/live/capture.py`` captures with the standard library alone, which is
the right default for an air-gapped box with nothing installed on it.  It has
two limits on Windows that matter for a live demonstration: it needs an
elevated process, and without Npcap it sees bare IP datagrams rather than
Ethernet frames, so anything that depends on the link layer is simply absent.

Where Wireshark *is* installed, ``dumpcap`` is the better sensor.  It is the
small privileged helper Wireshark itself shells out to -- it writes a pcapng
and does nothing else, which is exactly the amount of program that should be
handed capture privilege.  Nothing here parses what it writes: the file goes
through the same in-tree reader an uploaded capture does.

Two rules this module keeps, because the dashboard exposes it over HTTP:

* **No shell, ever.**  Every invocation is an argument list.  A capture
  filter, an interface name or a path containing a quote is then data, not
  syntax, and cannot become another command.
* **The caller never names the binary or the output file.**  ``dumpcap`` is
  located here and the output path is chosen by the caller from its own
  temporary directory -- a request body supplies an interface and a duration
  and nothing else.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DumpcapError", "DumpcapMissing", "Interface", "CaptureOutcome",
    "find_dumpcap", "list_interfaces", "capture", "MIN_SECONDS", "MAX_SECONDS",
]

MIN_SECONDS = 5
MAX_SECONDS = 3600

# Where Wireshark installs itself, plus the location this project's own
# bundled copy lives in.  Searched only after PATH and the env var.
_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Wireshark\dumpcap.exe",
    r"C:\Program Files (x86)\Wireshark\dumpcap.exe",
)
_POSIX_CANDIDATES = ("/usr/bin/dumpcap", "/usr/local/bin/dumpcap",
                     "/opt/wireshark/bin/dumpcap")

# "1. \Device\NPF_{2B0D...} (Wi-Fi)"  /  "3. eth0"
_IFACE_RE = re.compile(r"^\s*(\d+)\.\s+(\S+)(?:\s+\((.*)\))?\s*$")


class DumpcapError(RuntimeError):
    """dumpcap ran and failed, or could not be run."""


class DumpcapMissing(DumpcapError):
    """dumpcap is not installed anywhere this module knows to look."""


@dataclass(frozen=True)
class Interface:
    number: int
    device: str
    description: str = ""

    @property
    def label(self) -> str:
        return self.description or self.device

    def as_dict(self) -> dict:
        return {"number": self.number, "device": self.device,
                "description": self.description, "label": self.label}


@dataclass
class CaptureOutcome:
    path: Path
    seconds: float
    bytes_written: int
    interface: str
    message: str = ""

    def as_dict(self) -> dict:
        return {"path": str(self.path), "seconds": round(self.seconds, 2),
                "bytes": self.bytes_written, "interface": self.interface,
                "message": self.message}


def find_dumpcap(hint: str | os.PathLike | None = None) -> Path:
    """Locate ``dumpcap``.

    Order: an explicit hint, then ``ATDRPS_DUMPCAP``, then ``PATH``, then the
    usual install directories.  A hint that points at a *directory* (the
    Wireshark folder rather than the binary) is accepted, because that is what
    a person pastes.
    """
    candidates: list[Path] = []
    for raw in (hint, os.environ.get("ATDRPS_DUMPCAP")):
        if not raw:
            continue
        p = Path(str(raw)).expanduser()
        if p.is_dir():
            candidates += [p / "dumpcap.exe", p / "dumpcap"]
        else:
            candidates.append(p)

    found = shutil.which("dumpcap")
    if found:
        candidates.append(Path(found))
    candidates += [Path(c) for c in
                   (_WINDOWS_CANDIDATES if sys.platform == "win32" else _POSIX_CANDIDATES)]

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise DumpcapMissing(
        "dumpcap was not found. It ships with Wireshark "
        "(https://www.wireshark.org/download.html). If it is installed "
        "somewhere unusual, set the ATDRPS_DUMPCAP environment variable to the "
        "full path of dumpcap.exe, or pass --dumpcap."
    )


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout, shell=False, check=False)
    except subprocess.TimeoutExpired as exc:
        raise DumpcapError(f"dumpcap did not finish within {timeout:.0f}s") from exc
    except OSError as exc:
        raise DumpcapError(f"could not run dumpcap: {exc}") from exc


def list_interfaces(dumpcap: str | os.PathLike | None = None,
                    timeout: float = 20.0) -> list[Interface]:
    """``dumpcap -D`` -- the capture devices this machine offers.

    An empty list on a machine that has interfaces almost always means
    privilege: on Windows the Npcap driver refuses a non-elevated process, on
    Linux the socket needs ``CAP_NET_RAW``.  That is reported as an error with
    the fix in it rather than as "no interfaces", which sends a person looking
    at their network card.
    """
    binary = find_dumpcap(dumpcap)
    proc = _run([str(binary), "-D"], timeout)
    interfaces: list[Interface] = []
    for line in proc.stdout.splitlines():
        m = _IFACE_RE.match(line)
        if m:
            interfaces.append(Interface(int(m.group(1)), m.group(2),
                                        (m.group(3) or "").strip()))
    if not interfaces:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        raise DumpcapError(
            f"dumpcap listed no capture interfaces ({tail}). This is normally a "
            "permission problem: " + _privilege_hint()
        )
    return interfaces


def _privilege_hint() -> str:
    if sys.platform == "win32":
        return ("re-open the terminal with \"Run as administrator\" and start "
                "the dashboard again, and check that Npcap was installed with "
                "Wireshark.")
    return ("run as root, or once: sudo setcap cap_net_raw,cap_net_admin=eip "
            "$(readlink -f $(which dumpcap))")


def capture(interface: str | int, seconds: int, out_path: str | os.PathLike,
            dumpcap: str | os.PathLike | None = None,
            extra_timeout: float = 30.0,
            snaplen: int | None = None) -> CaptureOutcome:
    """Capture for ``seconds`` into ``out_path``, then return what was written.

    ``interface`` is the number or device name ``dumpcap -D`` printed. It is
    passed as a single argument and never interpolated into a string.
    """
    seconds = int(seconds)
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ValueError(
            f"capture duration must be {MIN_SECONDS}-{MAX_SECONDS}s, got {seconds}"
        )
    iface = str(interface).strip()
    if not iface:
        raise ValueError("no capture interface was given")

    binary = find_dumpcap(dumpcap)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    args = [str(binary), "-i", iface, "-a", f"duration:{seconds}", "-w", str(out)]
    if snaplen:
        args += ["-s", str(int(snaplen))]

    t0 = time.time()
    proc = _run(args, timeout=seconds + extra_timeout)
    elapsed = time.time() - t0

    if not out.exists() or out.stat().st_size == 0:
        detail = (proc.stderr or proc.stdout or "").strip() or "no output"
        raise DumpcapError(
            f"dumpcap wrote nothing (exit {proc.returncode}): {detail}. "
            + (_privilege_hint() if proc.returncode != 0 else
               "The interface may simply have been idle -- pick the one your "
               "traffic is actually on.")
        )

    # dumpcap reports "Packets captured: N" on stderr when it finishes
    message = ""
    for line in (proc.stderr or "").splitlines():
        if "Packets" in line:
            message = line.strip()
    return CaptureOutcome(path=out, seconds=elapsed,
                          bytes_written=out.stat().st_size,
                          interface=iface, message=message)
