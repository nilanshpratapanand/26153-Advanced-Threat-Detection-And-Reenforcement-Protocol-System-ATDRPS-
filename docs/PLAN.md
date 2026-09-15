# ATDRPS — Build Plan

Every phase ends with a **verification gate**. Nothing moves to the next phase until the
gate passes, and each passed gate is a commit. This is the plan of record; it is updated
as phases land.

## Ground rules

1. **Build → verify → commit → next.** No phase is "done" on the strength of the code
   reading correctly; it is done when a test exercises it and passes.
2. **Time-based splits only.** Random splits leak the future into the past and would make
   every number in this repo a lie.
3. **No cloud calls, ever.** Anywhere in the runtime path.
4. **Minimal dependency surface.** Where a dependency would be a supply-chain risk inside
   an air-gapped network and the replacement is tractable, we write the replacement
   (pcap parser, KernelSHAP).

## Phases

| # | Phase | Gate |
|---|-------|------|
| P0 | Repo skeleton, config system, test runner | Config round-trips, overrides typed, suite green |
| P1 | Synthetic attack-chain traffic generator | Generated capture replays to the exact flows/stages injected |
| P2 | Stdlib pcap/pcapng parser + packet features | Byte-exact round-trip on generated captures; feature sanity bounds |
| P3 | Flow assembly, flow features, dataset adapters | Flow counts/durations match ground truth; CIC-IDS2018 columns map cleanly |
| P4 | Windowing, state vectors, MITRE stage labelling | Window boundaries exact; stage labels match injected timeline |
| P5 | NumPy world model + K-step rollout | Beats persistence baseline on next-state MSE; rollout shapes/probabilities valid |
| P6 | PyTorch temporal transformer | Gradient flows, loss decreases, attention extractable, checkpoint reloads |
| P7 | Logistic Regression baseline + evaluation | Metrics verified against hand-computed confusion matrices |
| P8 | KernelSHAP + attention attribution | SHAP values sum to (prediction − base value) within tolerance |
| P9 | Inference engine + CLI | End-to-end: pcap in → timeline/stage/drivers out, offline |
| P10 | Offline Flask dashboard | Server boots, upload → render verified over HTTP |
| P11 | Deliverable docs, benchmarks, deck, video script | SIH checklist complete |

## SIH deliverable checklist

- [x] Source code (open source, public repo)
- [x] README with setup instructions — plus `install.bat`/`install.sh`, `run.bat`/`run.sh`
- [x] Architecture document (max 2 pages) — `docs/ARCHITECTURE.md`
- [ ] Demo video (max 2 minutes) — script ready in `docs/DEMO_SCRIPT.md`, needs recording
- [x] Technical presentation (max 5 slides) — `docs/ATDRPS_SIH26153.pptx`
- [x] Feature extraction pipeline — CSV **and** PCAP
- [x] Trained world model + training scripts + reproducible config + weights
- [x] Infiltration prediction engine with K-step forward simulation
- [x] Explainability output (SHAP / attention)
- [x] Working offline interface
- [x] Benchmark results vs Logistic Regression baseline (F1, Precision, Recall, FPR)
- [x] Attack coverage analysis — `docs/ATTACK_COVERAGE.md` (added; not required, but it is
      the question a reviewer asks first)

## Environment note

The development sandbox used to author this repository has no access to PyPI, so
PyTorch could not be installed there. Consequences, handled deliberately:

* The **NumPy dynamics backend** (P5) implements the same `WorldModel` interface and is
  fully exercised by the test suite in any environment. It also serves as a legitimate
  ablation in the benchmark table ("does the transformer actually buy us anything over
  linear dynamics?").
* The **PyTorch backend** (P6) is verified on the target training machine.
* Nothing else in the pipeline depends on an uninstallable package — that is precisely
  why the pcap parser and KernelSHAP are written in-tree.
