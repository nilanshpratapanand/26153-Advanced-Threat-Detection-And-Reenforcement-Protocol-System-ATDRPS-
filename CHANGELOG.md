# Changelog

## 0.2.0 — security hardening and honest forecasting

**Read this first.** v1's headline F1 (0.93) mostly measured detecting an attack that was
already under way, and on a real benign laptop capture the v1 model scored most windows as an
attack. This release fixes the security problems found in an audit, replaces the evaluation with
one that measures forecasting properly, and adds a per-network model. It does **not** make
attack prediction accurate: see `docs/RESEARCH_AND_PLAN_V2.md` for every result, its command
and its limits.

### Security
* **Dashboard could be told which program to run.** `find_dumpcap` accepted any existing file
  from a request parameter, so a web request could make the server execute an unrelated
  executable (for example `env`, whose arguments include the client-supplied interface name).
  The request can no longer name a binary, and a hint or `ATDRPS_DUMPCAP` is honoured only if the
  file is actually called `dumpcap`/`dumpcap.exe`.
* pcap/pcapng reader: a zero-length pcapng section header looped forever; huge length fields
  allocated gigabytes. Block lengths are now validated and capped.
* Upload/capture endpoints: bad `threshold`/`horizon` returned HTTP 500; tracebacks were returned
  to clients; the shared engine's threshold was mutated per request (a race). Now validated,
  logged server-side only, and analysed on a per-request copy under a lock.
* Dashboard: cross-site scripting through CSV addresses, filenames, error text, capture errors and
  the novelty message. All server- and upload-derived text is escaped; a per-response CSP nonce and
  `X-Frame-Options`/`nosniff`/`Referrer-Policy` headers are sent; exported CSV cells that start
  with `=`, `+`, `-` or `@` are defused.
* DNS-rebinding guard: a loopback-bound dashboard only answers requests addressed to a loopback
  name (`ATDRPS_ALLOWED_HOSTS` adds more). Capture interface names are validated; the job table is
  bounded.
* Model files: pickled models load through an allowlist unpickler; PyTorch checkpoints load with
  `weights_only=True`.

### Added
* `atdrps/forecast/`: quiet-state onset/escalation evaluation protocol (event recall, lead time,
  base-rate-corrected precision, capture-level bootstrap CIs), baselines, a per-network profile,
  a gradient-boosted hazard model with calibration.
* `atdrps/data/overlay.py`: inject labelled campaigns and benign look-alikes into a real
  background capture.
* `atdrps train-local --benign normal.pcap` and `atdrps forecast capture.pcap`: learn one network's
  normal and score captures, reporting the false-alarm rate the calibration data can actually
  support.
* `.gitattributes` so `.bat` files are checked out with CRLF.

### Known limits
* Forecasting an onset from silence is not achieved (AUC 0.52-0.61 on the hard benchmark).
* Escalation forecasting flags about half to three quarters of simulated escalations ~150 s ahead
  at 1-4 false alerts/hour; precision at a realistic base rate is under 5 %. 15 test events, one
  background network, simulated attacks.
* The dashboard still serves the v1 engine (with the novelty gate); v2 is command-line only.
* ARP and IPv6 are ignored by the flow pipeline.
* Self-updating launchers are not included.

## 0.1.0
Initial submission build.
