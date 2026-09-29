# ATDRPS v2 — research digest and plan for reliable attack forecasting

Status: living document. Only items marked **[done]** were built and measured in this
repository; everything else is planned. Numbers here are only ever copied from a
command that was actually run (see the `Evidence` column).

## 1. Why v1 was not reliable (measured, not assumed)

Diagnostic run on the v1 pipeline (16 synthetic captures, group split, 180 test sequences):

| Question | Result |
|---|---|
| F1 from copying the *current* label forward (oracle persistence, uses the answer) | 0.898 (agreement 0.928) |
| Share of test windows that are attack windows | 35.6 % |
| Onset-only subset (no attack in the 16-window context): onsets available | **5** of 70 windows |
| Model AUC when no attack in the *last* window | 0.461 (chance), 7 onsets |

Reading: the headline F1 mostly measures detecting an attack that is already under way;
the genuine forecasting cases are too few to support any claim; and the data are far more
attack-dense than any real network. The sample is small, so this is a warning, not a proof
that forecasting skill is zero — but it is enough to say the v1 numbers do not establish
forecasting skill.

## 2. What the literature says (sources verified by search in this session)

| Finding | Source | Consequence for the design |
|---|---|---|
| Ten recurring evaluation pitfalls in security ML: sampling bias, label inaccuracy, data snooping, inappropriate baselines/measures, **base-rate neglect**, lab-only evaluation | Arp et al., *Dos and Don'ts of ML in Computer Security*, USENIX Sec 2022 | Evaluate at realistic prevalence, with strong baselines, on held-out real data |
| Temporal bias (test data older than training data) and spatial bias inflate results; propose time-aware constraints and the AUT robustness metric | Pendlebury et al., *TESSERACT*, USENIX Sec 2019 | Train on earlier captures, test on later; report performance decay |
| The false alarm rate, not the detection rate, limits an IDS because attacks are rare | Axelsson, *The base-rate fallacy…*, ACM TISSEC 2000 | Report precision at assumed real prevalence and alerts/day, not just F1 |
| Six reasons ML is hard in intrusion detection: cost of errors, semantic gap, variability of normal, hard evaluation, adversaries | Sommer & Paxson, *Outside the Closed World*, IEEE S&P 2010 | Prefer detecting *changes in behaviour with explanations*; treat anomaly scores as evidence, not verdicts |
| CIC-IDS2017 has labelling errors and artefact flows (>25 % of flows meaningless in some classes; ~10 % mislabelled) and packet misordering | Engelen et al. 2021; Lanvin et al., CRiSIS 2022 | If real data is used, use the corrected versions and re-derive flows from PCAP |
| Attack prediction is a distinct task family (projection, intrusion prediction, situation forecasting); methods span Markov/Bayesian, time-series, ML | Husák et al., IEEE Commun. Surv. Tutor. 21(1), 2019 | Frame the task as *hazard of onset*, and include Markov and time-series baselines |
| Online, unsupervised ensembles of autoencoders reach detection comparable to offline detectors on CPU-only hardware | Mirsky et al., *Kitsune*, NDSS 2018 | Use a Kitsune-style anomaly baseline |
| Conformal risk control can map an alert budget to a false-positive-bounded threshold under exchangeability | CALIBURN and related 2025–26 preprints (arXiv) | Calibrate the alert threshold on held-out benign data; state the exchangeability assumption |
| LANL multi-source dataset: 58 days, 1.65 B events, red-team labels with ~0.00007 % malicious | csr.lanl.gov | Real prevalence is many orders of magnitude below the synthetic 35 % |
| CALDERA/LADEMU generate labelled data from real adversary emulation; sim-to-real gap is a known limit of synthetic traffic | MITRE CALDERA; LADEMU (FFI); TempoNet, GAN-based generators | Add a real-kernel-stack lab generator and *measure* the sim-to-real gap |

## 3. Design

**What can honestly be promised.** No system can say with certainty that an attack "is going
to happen". What can be built and *measured* is a calibrated **hazard**: P(infiltration onset
within the next K windows | recent behaviour), reported together with the false-alarm rate
and the warning lead time achieved at a stated alert budget.

1. **Evaluation protocol first [done — Phase A].** Onset-conditional task (only windows with no
   attack in the recent past), event-level recall and lead time, prevalence-adjusted
   precision, cluster-bootstrap confidence intervals, time/family-aware splits, and baselines
   including the deployable v1 model.
2. **Harder, more honest simulation [Phase B].** Benign look-alikes that never escalate
   (vulnerability scanners, backup bursts, admin sweeps), aborted campaigns, heavy-tailed
   dwell times between stages, low-and-slow/distributed tempo, per-capture domain
   randomisation, benign-only captures to estimate false-alarm rates precisely.
3. **Real-kernel lab [Phase D].** Real client/server processes and attacker tools on loopback
   addresses, captured with `AF_PACKET`, so the packets come from a real TCP stack. Used as a
   *held-out realism test* (train on simulator, test on lab) to measure the sim-to-real gap.
4. **Hazard model [Phase C].** Multi-scale features of the recent windows, gradient-boosted
   trees, isotonic calibration, conformal alert threshold; compared against persistence,
   generic anomaly detectors (Mahalanobis, isolation forest), and v1.
5. **Real data [Phase E].** Adapters and an `eval-real` command so the same protocol runs on
   CIC-IDS (corrected), UNSW-NB15, CTU-13 and LANL flows as soon as the files are supplied.

## 4. Hard limits (stated up front)

* The build sandbox cannot reach any real-dataset host (CIC, UNSW, CTU, Zenodo, Kaggle,
  HuggingFace all refused). Until real files are provided, **no accuracy claim on real
  networks can be made**, and this document will not make one.
* Simulator results measure the simulator. They are useful for finding bugs and comparing
  methods under controlled difficulty, not for predicting field performance.
* A calibrated hazard at a 1-false-alert-per-day budget will miss a large share of
  campaigns; the protocol reports that trade-off instead of hiding it.

## 5. Evidence log

(Each row names the command that produced it.)

### Phase A — v1 vs simple detectors on the quiet-state onset task

Command: `python -m atdrps.cli corpus --captures 64 --duration 3600 --campaigns 3 --out DIR/corpus_v1sim.npz --workers 4`
then `python scripts/forecast_baselines_v1sim.py DIR`. Context 16 windows of 30 s, horizon K=5,
quiet gap 3, group split by capture (train 31 / val 13 / test 19 captures). Test set: 637 quiet
samples, 165 positive (25.9 %), **35 distinct onsets**, 472 negatives. Thresholds chosen on
validation at a 1 % FPR budget. CIs: 200 capture-level bootstrap draws.

| model | AUC [95 % CI] | AP (chance 0.259) | TPR @ 1 % FPR | event recall | median lead | precision @ onset rate 1e-3 |
|---|---|---|---|---|---|---|
| constant | 0.500 | 0.259 | 0.000 | 0.000 | – | – |
| CUSUM/EWMA on recon indicators | 0.765 [0.694, 0.847] | 0.642 | 0.224 | 0.600 | 90 s | 0.018 |
| Mahalanobis (last window) | 0.685 [0.621, 0.757] | 0.460 | 0.067 | 0.257 | 90 s | 0.005 |
| Isolation Forest (last window) | 0.714 [0.651, 0.785] | 0.528 | 0.073 | 0.314 | 90 s | 0.008 |
| ATDRPS v1 world model | 0.663 [0.579, 0.743] | 0.405 | 0.030 | 0.343 | 90 s | 0.003 |

Reading, with caveats:
* On genuine forecasting samples the deployed v1 model is **worse than a hand-built
  reconnaissance detector** and worse than a stock Isolation Forest. The v1 and CUSUM
  intervals overlap slightly (35 onsets is a small sample), so this is strong evidence,
  not proof, that v1's learned dynamics add nothing here.
* At a realistic onset rate of 1 in 1000 windows every model's precision is below 2 %:
  more than 98 % of alerts would be false. This is the base-rate problem (Axelsson 2000)
  and it is why later phases report alerts/day and precision at assumed prevalence.
* Median lead is 90 s for all models because the simulator's reconnaissance-to-access gap is
  5–60 s: the simulator itself caps the warning time. Phase B replaces that.
* The corpus is 52 % attack windows. This is not a realistic prevalence (Phase B).
