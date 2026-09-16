"""Driving Wireshark's dumpcap.

Verified against a stub binary rather than a real capture, because the thing
worth testing here is not whether Wireshark works -- it is whether this module
builds the right argument list, refuses the wrong input, and reports a failure
instead of returning an empty capture as a success.

The stub is a Python script that records the argv it was handed and writes
whatever the test asks it to, so every test below is an assertion about
ATDRPS's behaviour and none of them needs capture privilege.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from atdrps.live.dumpcap import (
    DumpcapError, DumpcapMissing, MAX_SECONDS, MIN_SECONDS, capture,
    find_dumpcap, list_interfaces,
)

_STUB = '''#!{python}
import json, sys, pathlib
argv = sys.argv[1:]
pathlib.Path({argv_log!r}).write_text(json.dumps(argv))
mode = {mode!r}
if "-D" in argv:
    sys.stdout.write({listing!r})
    sys.exit(0)
if mode == "writes":
    out = argv[argv.index("-w") + 1]
    pathlib.Path(out).write_bytes(b"\\xd4\\xc3\\xb2\\xa1" + b"\\0" * 60)
    sys.stderr.write("Packets captured: 42\\n")
    sys.exit(0)
if mode == "silent-failure":
    sys.stderr.write("The capture session could not be initiated\\n")
    sys.exit(2)
sys.exit(0)
'''


class _Stub:
    """A fake dumpcap on disk."""

    def __init__(self, tmp: Path, mode: str = "writes", listing: str = ""):
        self.dir = tmp
        self.argv_log = tmp / "argv.json"
        self.path = tmp / ("dumpcap.py" if sys.platform == "win32" else "dumpcap")
        self.path.write_text(_STUB.format(
            python=sys.executable, argv_log=str(self.argv_log), mode=mode,
            listing=listing,
        ))
        self.path.chmod(self.path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)

    @property
    def argv(self) -> list[str]:
        return json.loads(self.argv_log.read_text())


_LISTING = (
    "1. \\Device\\NPF_{2B0D1234-5678} (Wi-Fi)\n"
    "2. \\Device\\NPF_{9ABCDEF0-1111} (Ethernet)\n"
    "3. \\Device\\NPF_Loopback (Adapter for loopback traffic capture)\n"
)


@unittest.skipIf(sys.platform == "win32",
                 "the stub relies on a POSIX shebang; the module itself is "
                 "platform-independent and its Windows path is exercised on Windows")
class TestListInterfaces(unittest.TestCase):
    def test_parses_the_dumpcap_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(Path(tmp), listing=_LISTING)
            ifaces = list_interfaces(stub.path)
        self.assertEqual([i.number for i in ifaces], [1, 2, 3])
        self.assertEqual(ifaces[0].description, "Wi-Fi")
        self.assertEqual(ifaces[0].device, "\\Device\\NPF_{2B0D1234-5678}")
        self.assertEqual(ifaces[0].label, "Wi-Fi")

    def test_a_device_with_no_description_still_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(Path(tmp), listing="1. eth0\n2. any\n")
            ifaces = list_interfaces(stub.path)
        self.assertEqual([i.device for i in ifaces], ["eth0", "any"])
        self.assertEqual(ifaces[0].label, "eth0")

    def test_empty_listing_is_reported_as_a_permission_problem(self):
        """An empty list is almost never "you have no network card"."""
        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(Path(tmp), listing="")
            with self.assertRaises(DumpcapError) as ctx:
                list_interfaces(stub.path)
        self.assertIn("permission", str(ctx.exception).lower())


@unittest.skipIf(sys.platform == "win32", "see above")
class TestCapture(unittest.TestCase):
    def test_builds_the_expected_argument_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(Path(tmp))
            out = Path(tmp) / "live.pcapng"
            result = capture("4", 30, out, dumpcap=stub.path)
            argv = stub.argv
        self.assertEqual(argv[:5], ["-i", "4", "-a", "duration:30", "-w"])
        self.assertEqual(argv[5], str(out))
        self.assertEqual(result.bytes_written, out_size := 64)
        self.assertIn("42", result.message)

    def test_never_uses_a_shell(self):
        """An interface name with shell syntax in it must stay one argument."""
        nasty = 'eth0" & calc.exe & "'
        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(Path(tmp))
            capture(nasty, 5, Path(tmp) / "x.pcapng", dumpcap=stub.path)
            argv = stub.argv
        self.assertEqual(argv[1], nasty)
        self.assertEqual(len(argv), 6)

    def test_rejects_durations_outside_the_allowed_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(Path(tmp))
            for bad in (0, MIN_SECONDS - 1, MAX_SECONDS + 1):
                with self.assertRaises(ValueError):
                    capture("1", bad, Path(tmp) / "x.pcapng", dumpcap=stub.path)

    def test_rejects_an_empty_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(Path(tmp))
            with self.assertRaises(ValueError):
                capture("   ", 10, Path(tmp) / "x.pcapng", dumpcap=stub.path)

    def test_a_failed_capture_raises_rather_than_returning_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(Path(tmp), mode="silent-failure")
            with self.assertRaises(DumpcapError) as ctx:
                capture("1", 10, Path(tmp) / "x.pcapng", dumpcap=stub.path)
        self.assertIn("could not be initiated", str(ctx.exception))

    def test_passes_a_snaplen_through_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = _Stub(Path(tmp))
            capture("1", 10, Path(tmp) / "x.pcapng", dumpcap=stub.path, snaplen=128)
            self.assertEqual(stub.argv[-2:], ["-s", "128"])


class TestFindDumpcap(unittest.TestCase):
    def test_accepts_a_directory_hint(self):
        """People paste the Wireshark folder, not the binary inside it."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "dumpcap").write_text("#!/bin/sh\n")
            self.assertEqual(find_dumpcap(folder), folder / "dumpcap")

    def test_env_var_is_consulted(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "dumpcap"
            binary.write_text("#!/bin/sh\n")
            old = os.environ.get("ATDRPS_DUMPCAP")
            os.environ["ATDRPS_DUMPCAP"] = str(binary)
            try:
                self.assertEqual(find_dumpcap(), binary)
            finally:
                if old is None:
                    os.environ.pop("ATDRPS_DUMPCAP", None)
                else:
                    os.environ["ATDRPS_DUMPCAP"] = old

    def test_missing_dumpcap_says_where_to_get_it(self):
        """PATH and the install directories are stubbed out, so this test says
        the same thing on a machine that happens to have Wireshark installed."""
        from unittest import mock

        import atdrps.live.dumpcap as mod

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(mod.shutil, "which", return_value=None), \
                mock.patch.object(mod, "_WINDOWS_CANDIDATES", ()), \
                mock.patch.object(mod, "_POSIX_CANDIDATES", ()), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATDRPS_DUMPCAP", None)
            with self.assertRaises(DumpcapMissing) as ctx:
                find_dumpcap(Path(tmp) / "nope" / "dumpcap.exe")
        self.assertIn("wireshark.org", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
