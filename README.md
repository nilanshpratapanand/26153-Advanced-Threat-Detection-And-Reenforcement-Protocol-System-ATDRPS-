# ATDRPS — Advanced Threat Detection and Reinforcement Protocol System

**A world model for network attack forecasting.**

| | |
|---|---|
| Event | Smart India Hackathon 2026 |
| Problem Statement | **26153** — AI-based Network Attack Forecasting from Network Traffic Data |
| Organisation | National Technical Research Organisation (NTRO) |
| Theme / Category | Blockchain & Cybersecurity / Software |
| Team | Bell Labs — Lloyd Institute of Engineering & Technology |
| Idea | PreCog-NetDefense — a predictive world model for network intrusion |

---

## What this is

A conventional IDS answers **"is this flow malicious?"** — one flow at a time, with the
temporal and causal structure thrown away.

ATDRPS answers a different question: **"where is this network heading?"**

It learns the *transition dynamics* of network state,

$$P(S_{t+1} \mid S_{\le t})$$

then rolls that model forward **K steps** from the current traffic snapshot to produce:

1. an **infiltration probability timeline** over the next K time windows,
2. the **predicted MITRE ATT&CK stage** (Reconnaissance → Initial Access → Lateral Movement → Command & Control → Exfiltration),
3. the **driving features** behind every prediction (SHAP values + attention weights) — never a black box.

A slow port scan that stays under every per-flow threshold, or a SYN flood that resolves
into lateral movement, is invisible flow-by-flow but plainly visible as a *trajectory*.
That trajectory is what ATDRPS models.

## Design constraints we took seriously

* **Fully offline.** No cloud APIs, no telemetry, no model downloads at runtime.
  Critical Information Infrastructure is frequently air-gapped, so the tool has to
  work there.
* **Minimal dependency surface.** ATDRPS ships **its own pcap/pcapng parser** and
  **its own KernelSHAP implementation**. `scapy`, `pyshark` and `shap` are *not*
  required — one fewer supply-chain surface inside a defence network.
* **Explainable by construction.** Every prediction carries a ranked list of the
  flags, ports, timing statistics and payload patterns that produced it.
* **Reproducible.** A run is defined by `(config, dataset, seed)`. The resolved config
  is written next to every checkpoint.

## Status

Under active construction. See `docs/PLAN.md` for the phase plan and what is done.

## Quick start

```bash
git clone https://github.com/nilanshpratapanand/26153-Advanced-Threat-Detection-And-Reenforcement-Protocol-System-ATDRPS-.git
cd 26153-Advanced-Threat-Detection-And-Reenforcement-Protocol-System-ATDRPS-

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python tests/run_tests.py            # test suite (standard library only, no pytest)
```

## Repository layout

```
atdrps/
  config.py         resolved YAML configuration, dotted access, seeding
  data/             pcap parsing, flow assembly, feature extraction, windowing, MITRE mapping
  models/           world model backends (PyTorch temporal transformer, NumPy dynamics), rollout
  train/            training loops, time-based splits, evaluation harness
  explain/          KernelSHAP, attention attribution, human-readable driver reports
  engine/           end-to-end inference: capture in -> forecast out
app/                offline Flask dashboard
configs/            YAML configuration
docs/               architecture document, plan, benchmarks
tests/              standard-library test suite
```

## Licence

MIT — see `LICENSE`.
