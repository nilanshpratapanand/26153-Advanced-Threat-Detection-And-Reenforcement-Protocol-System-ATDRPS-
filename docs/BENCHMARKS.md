# ATDRPS benchmark results

All models see identical sequences, an identical split and identical features. Operating thresholds are chosen on the validation split and applied unchanged to test -- tuning a threshold on the data you then report turns a benchmark into a sales pitch.

## Setup

- sequences: **7680** (train 4960 / val 1120 / test 1600)
- context: **16 windows** of 30s; forecast horizon **K = 5**
- state dimension: **103**
- split: held-out captures (train and test share no window or campaign)

## Infiltration forecast, one window ahead

| Model           | Step | F1     | Precision | Recall | FPR    | ROC-AUC | StageAcc | StageMacroF1 |
|-----------------|------|--------|-----------|--------|--------|---------|----------|--------------|
| logreg-static   | 1    | 0.8477 | 0.8275    | 0.8688 | 0.1918 | 0.9192  | 0.7619   | 0.7071       |
| logreg-context  | 1    | 0.8946 | 0.9075    | 0.8821 | 0.0952 | 0.9598  | 0.7956   | 0.7298       |
| linear-dynamics | 1    | 0.9294 | 0.9306    | 0.9283 | 0.0734 | 0.9734  | 0.8400   | 0.7807       |

## Every horizon step

| Model           | Step | F1     | Precision | Recall | FPR    | ROC-AUC | StageAcc | StageMacroF1 |
|-----------------|------|--------|-----------|--------|--------|---------|----------|--------------|
| logreg-static   | 1    | 0.8477 | 0.8275    | 0.8688 | 0.1918 | 0.9192  | 0.7619   | 0.7071       |
| logreg-static   | 2    | 0.8207 | 0.8325    | 0.8092 | 0.1725 | 0.8968  | 0.6956   | 0.6319       |
| logreg-static   | 3    | 0.8194 | 0.7894    | 0.8518 | 0.2407 | 0.8803  | 0.6412   | 0.5717       |
| logreg-static   | 4    | 0.8073 | 0.7644    | 0.8554 | 0.2793 | 0.8677  | 0.5938   | 0.5247       |
| logreg-static   | 5    | 0.7866 | 0.7097    | 0.8821 | 0.3822 | 0.8556  | 0.5537   | 0.4848       |
| logreg-context  | 1    | 0.8946 | 0.9075    | 0.8821 | 0.0952 | 0.9598  | 0.7956   | 0.7298       |
| logreg-context  | 2    | 0.8580 | 0.8720    | 0.8445 | 0.1313 | 0.9261  | 0.7056   | 0.6221       |
| logreg-context  | 3    | 0.8342 | 0.7952    | 0.8773 | 0.2394 | 0.8989  | 0.6381   | 0.5410       |
| logreg-context  | 4    | 0.8173 | 0.7508    | 0.8967 | 0.3153 | 0.8828  | 0.5819   | 0.4754       |
| logreg-context  | 5    | 0.8152 | 0.7836    | 0.8493 | 0.2484 | 0.8681  | 0.5312   | 0.4245       |
| linear-dynamics | 1    | 0.9294 | 0.9306    | 0.9283 | 0.0734 | 0.9734  | 0.8400   | 0.7807       |
| linear-dynamics | 2    | 0.9006 | 0.9095    | 0.8919 | 0.0940 | 0.9525  | 0.7913   | 0.7038       |
| linear-dynamics | 3    | 0.8645 | 0.8164    | 0.9186 | 0.2188 | 0.9362  | 0.7369   | 0.6352       |
| linear-dynamics | 4    | 0.8490 | 0.8102    | 0.8919 | 0.2214 | 0.9230  | 0.6919   | 0.5715       |
| linear-dynamics | 5    | 0.8437 | 0.8004    | 0.8919 | 0.2355 | 0.9096  | 0.6600   | 0.5321       |

## Dynamics: can the model predict the next state at all?

Next-state error in standardised units. `persistence` is the floor -- it copies the current state forward. A model that cannot beat it has not learned dynamics, whatever its classification scores say.

| Model           | Next-state MSE | MAE    | RMSE   |
|-----------------|----------------|--------|--------|
| persistence     | 1.0068         | 0.4907 | 1.0034 |
| linear-dynamics | 0.6114         | 0.4411 | 0.7819 |

> **Note.** PyTorch was not installed in the environment that produced this report, so the temporal transformer row is absent. Install the requirements and re-run `atdrps benchmark` to fill it in.
