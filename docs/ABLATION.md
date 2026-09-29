# Ablation: what each half of the state buys

One split (held-out captures), one model class (`linear`), one threshold rule (chosen on validation). Only the input changes.

- state dimensions: **103** = 80 flow-derived + 23 packet-derived
- sequences: 4960 train / 1120 val / 1600 test

Flow-derived dimensions are what a NetFlow or CICFlowMeter export carries. Packet-derived dimensions (TTL spread, TCP window trace, payload distribution, retransmissions, fragmentation, injected RSTs) exist only because ATDRPS parses the packets itself.

| Variant         | Dims | Context | F1     | Precision | Recall | FPR    | StageAcc | StageMacroF1 | NextStateMSE | PersistenceMSE |
|-----------------|------|---------|--------|-----------|--------|--------|----------|--------------|--------------|----------------|
| flow-only       | 80   | 16      | 0.8952 | 0.9508    | 0.8457 | 0.0463 | 0.8375   | 0.7842       | 0.5880       | 0.9691         |
| packet-only     | 23   | 16      | 0.8661 | 0.8417    | 0.8919 | 0.1776 | 0.7738   | 0.7202       | 0.6600       | 1.1380         |
| dual (shipped)  | 103  | 16      | 0.9294 | 0.9306    | 0.9283 | 0.0734 | 0.8400   | 0.7807       | 0.6114       | 1.0068         |
| dual, context=1 | 103  | 1       | 0.8459 | 0.8287    | 0.8639 | 0.1892 | 0.7600   | 0.7034       | 0.5837       | 0.9934         |
| dual, context=4 | 103  | 4       | 0.9096 | 0.9204    | 0.8991 | 0.0824 | 0.8450   | 0.7780       | 0.5776       | 0.9979         |

## What the table says

**Fusion.** Neither half alone matches the pair: flow-only F1 0.8952, packet-only 0.8661, both 0.9294 (+0.0343 over the better half). The gain is in recall (0.9283 vs 0.8457 flow-only), which is the direction that matters for forecasting: the packet features carry evidence of attacks whose flow counters look ordinary.

**Where it loses.** Flow-only has the lower false-positive rate (0.0463 vs 0.0734) and the higher precision (0.9508 vs 0.9306). Fusion buys recall and pays for it in precision. A deployment that cares more about analyst time than about missed attacks should know that and can pick the operating point accordingly.

**Temporal context.** The same features with no history score 0.8459 F1 and 0.1892 FPR, against 0.9294 and 0.0734 with 16 windows -- +0.0835 F1 for looking backwards. This is the world-model claim measured on its own: identical features, identical model class, only the history removed.

Most of that is bought early: 4 windows of history already reach 0.9096. Going to 16 adds +0.0198.

Next-state error runs the other way (0.5837 vs 0.6114): for a *linear* model, a long context is extra parameters to fit and the immediately preceding window is most of the signal for one-step prediction. Context earns its place on the forecasting and stage heads, not on next-state regression.

`PersistenceMSE` is the dynamics floor for that feature subset -- copy the current state forward. It differs per row because the feature set differs, so next-state error is only comparable against the floor beside it, never across rows.
