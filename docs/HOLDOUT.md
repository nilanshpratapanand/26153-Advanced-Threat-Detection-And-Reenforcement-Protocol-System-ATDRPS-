# Generalisation to an unseen attack family: LateralMovement

Every capture containing **LateralMovement** was removed from training *and* from validation, so the operating threshold was also chosen without ever seeing the family. Those captures are the test set. The training half was generated so the family is impossible in it, not merely absent by luck of the seed (`scripts/make_corpus.py --holdout-stage`).

- captures: 24 train, 6 validation, 10 held out (of 40)
- sequences: 3840 / 960 / 1600
- model: `linear`, threshold 0.775 chosen on validation

## Detection rate per attack family, in the held-out captures

| Family            | In training? | Windows | Alarm fired | Mean p | AUC vs benign |
|-------------------|--------------|---------|-------------|--------|---------------|
| CommandAndControl | **no**       | 410     | 0.2098      | 0.3213 | 0.7716        |
| LateralMovement   | **no**       | 257     | 0.2957      | 0.3642 | 0.7293        |
| InitialAccess     | yes          | 223     | 0.8117      | 0.8474 | 0.9414        |
| Exfiltration      | yes          | 214     | 0.1729      | 0.1979 | 0.3782        |
| Reconnaissance    | yes          | 146     | 0.0479      | 0.1866 | 0.7138        |
| Impact            | yes          | 8       | 0.0000      | 0.0005 | 0.0716        |

False-alarm rate on the benign windows of the same captures: **0.0088**. Infiltration F1 over every held-out window (all families, benign included): 0.5060 at FPR 0.0205.

## What this actually shows

At the shipped operating point the alarm fires on **29.6%** of LateralMovement windows it was never trained on, at a 0.9% false-alarm rate on benign traffic in the same captures (threshold 0.775, chosen on validation captures that also lacked the family).

The trained-on families in the same captures average 25.8%, which looks like the unseen family does *better*. It does not: the two halves of this corpus were generated differently, because forcing a family out of the training half means truncating the campaigns that lead to it.

That shift is visible: Impact (AUC 0.0716 on 8 windows), Exfiltration (AUC 0.3782 on 214 windows) rank at or below chance against benign traffic despite being present in training. Those families appear in the training half only in the shapes the truncated generator could produce -- exfiltration as DNS tunnelling rather than bulk outbound transfer, for instance -- so they are closer to unseen families than the label suggests. The trained-on column is therefore a weak control, not a clean one, and the honest reading of this study is the absolute number for the held-out family, not the comparison.

AUC against benign traffic is 0.7293, so this is not just a mistuned threshold: the model ranks the unseen family only modestly above background. Transfer to a genuinely novel family is partial, not free.

**Do not read this as LateralMovement detection.** What this study supports is the narrow claim: a model that never saw the family still recovers a minority of it from trajectory alone, at a low false-alarm rate. A signature-based detector recovers none of an unseen family by construction. It does not support any claim about detecting novel attacks in general, and the same pipeline trained on a corpus containing every family scores far higher (`docs/BENCHMARKS.md`, a different corpus and split, so the two numbers are not directly comparable). The operational conclusion is to keep the training corpus current rather than to rely on transfer.

## What the stage head calls it

The LateralMovement class was never in the training labels, so the stage head **cannot** emit it: recall on those windows is 0.0000 by construction. That is a property of closed-set classification, not a result. What it emits instead: Benign 117, InitialAccess 88, Reconnaissance 42, Impact 8, Exfiltration 2.

Excluding one family upstream can strand the chain stages that follow it, so every stage the training split never contained is listed, not just the one named:

| Stage never trained on | Held-out windows | Alarm fired |
|------------------------|------------------|-------------|
| CommandAndControl      | 410              | 0.2098      |
| LateralMovement        | 257              | 0.2957      |

Stage accuracy over all held-out windows (benign included) is 0.3956, macro-F1 0.2708; both are floored by the missing classes and are reported that way on purpose.
