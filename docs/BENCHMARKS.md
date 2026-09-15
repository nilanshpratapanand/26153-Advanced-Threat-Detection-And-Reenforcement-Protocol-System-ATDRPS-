# ATDRPS benchmark results

All models see identical sequences, an identical chronological split and identical features. Operating thresholds are chosen on the validation split and applied unchanged to test.

## Setup

- sequences: **7680** (train 4224 / val 384 / test 1536)
- context: **16 windows** of 30s; forecast horizon **K = 5**
- state dimension: **96**
- split: chronological within capture, with a 16-sample gap at each boundary so that train and test never share a window

## Infiltration forecast, one window ahead

| Model           | Step | F1     | Precision | Recall | FPR    | ROC-AUC | StageAcc | StageMacroF1 |
|-----------------|------|--------|-----------|--------|--------|---------|----------|--------------|
| logreg-static   | 1    | 0.8000 | 0.6667    | 1.0000 | 0.0007 | 1.0000  | 0.9857   | 0.4982       |
| logreg-context  | 1    | 0.4444 | 0.2857    | 1.0000 | 0.0033 | 1.0000  | 0.9447   | 0.4095       |
| linear-dynamics | 1    | 0.2500 | 0.1429    | 1.0000 | 0.0078 | 1.0000  | 0.9440   | 0.4928       |

## Every horizon step

| Model           | Step | F1     | Precision | Recall | FPR    | ROC-AUC | StageAcc | StageMacroF1 |
|-----------------|------|--------|-----------|--------|--------|---------|----------|--------------|
| logreg-static   | 1    | 0.8000 | 0.6667    | 1.0000 | 0.0007 | 1.0000  | 0.9857   | 0.4982       |
| logreg-static   | 2    | 0.2222 | 0.1250    | 1.0000 | 0.0046 | 1.0000  | 0.9785   | 0.3315       |
| logreg-static   | 3    | 0.0000 | 0.0000    | 0.0000 | 0.0026 | n/a     | 0.9720   | 0.1643       |
| logreg-static   | 4    | 0.0000 | 0.0000    | 0.0000 | 0.0026 | n/a     | 0.9629   | 0.1635       |
| logreg-static   | 5    | 0.0000 | 0.0000    | 0.0000 | 0.0013 | n/a     | 0.9251   | 0.1602       |
| logreg-context  | 1    | 0.4444 | 0.2857    | 1.0000 | 0.0033 | 1.0000  | 0.9447   | 0.4095       |
| logreg-context  | 2    | 0.0357 | 0.0182    | 1.0000 | 0.0352 | 1.0000  | 0.8639   | 0.4817       |
| logreg-context  | 3    | 0.0000 | 0.0000    | 0.0000 | 0.0117 | n/a     | 0.7910   | 0.1472       |
| logreg-context  | 4    | 0.0000 | 0.0000    | 0.0000 | 0.0326 | n/a     | 0.7129   | 0.2081       |
| logreg-context  | 5    | 0.0000 | 0.0000    | 0.0000 | 0.0215 | n/a     | 0.6745   | 0.1611       |
| linear-dynamics | 1    | 0.2500 | 0.1429    | 1.0000 | 0.0078 | 1.0000  | 0.9440   | 0.4928       |
| linear-dynamics | 2    | 0.0741 | 0.0385    | 1.0000 | 0.0163 | 1.0000  | 0.9030   | 0.1898       |
| linear-dynamics | 3    | 0.0000 | 0.0000    | 0.0000 | 0.0020 | n/a     | 0.8815   | 0.1874       |
| linear-dynamics | 4    | 0.0000 | 0.0000    | 0.0000 | 0.0085 | n/a     | 0.8535   | 0.2302       |
| linear-dynamics | 5    | 0.0000 | 0.0000    | 0.0000 | 0.0007 | n/a     | 0.8307   | 0.1815       |

## Dynamics: can the model predict the next state at all?

Next-state error in standardised units. `persistence` is the floor -- it copies the current state forward. A model that cannot beat it has not learned dynamics, whatever its classification scores say.

| Model           | Next-state MSE | MAE    | RMSE   |
|-----------------|----------------|--------|--------|
| persistence     | 0.9812         | 0.5133 | 0.9906 |
| linear-dynamics | 0.5260         | 0.4207 | 0.7253 |

> **Note.** PyTorch was not installed in the environment that produced this report, so the temporal transformer row is absent. Install the requirements and re-run `atdrps benchmark` to fill it in.
