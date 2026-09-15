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
| logreg-static   | 1    | 0.8376 | 0.8417    | 0.8335 | 0.1660 | 0.9188  | 0.7556   | 0.6987       |
| logreg-context  | 1    | 0.8947 | 0.9128    | 0.8773 | 0.0888 | 0.9587  | 0.8013   | 0.7353       |
| linear-dynamics | 1    | 0.9214 | 0.9317    | 0.9113 | 0.0708 | 0.9733  | 0.8400   | 0.7821       |

## Every horizon step

| Model           | Step | F1     | Precision | Recall | FPR    | ROC-AUC | StageAcc | StageMacroF1 |
|-----------------|------|--------|-----------|--------|--------|---------|----------|--------------|
| logreg-static   | 1    | 0.8376 | 0.8417    | 0.8335 | 0.1660 | 0.9188  | 0.7556   | 0.6987       |
| logreg-static   | 2    | 0.8242 | 0.7984    | 0.8518 | 0.2278 | 0.8953  | 0.6956   | 0.6324       |
| logreg-static   | 3    | 0.8076 | 0.7944    | 0.8214 | 0.2252 | 0.8768  | 0.6419   | 0.5713       |
| logreg-static   | 4    | 0.7928 | 0.7160    | 0.8882 | 0.3732 | 0.8637  | 0.5869   | 0.5178       |
| logreg-static   | 5    | 0.7768 | 0.6917    | 0.8858 | 0.4183 | 0.8514  | 0.5419   | 0.4721       |
| logreg-context  | 1    | 0.8947 | 0.9128    | 0.8773 | 0.0888 | 0.9587  | 0.8013   | 0.7353       |
| logreg-context  | 2    | 0.8671 | 0.8470    | 0.8882 | 0.1699 | 0.9248  | 0.7081   | 0.6277       |
| logreg-context  | 3    | 0.8353 | 0.8043    | 0.8688 | 0.2239 | 0.8975  | 0.6394   | 0.5414       |
| logreg-context  | 4    | 0.8191 | 0.7618    | 0.8858 | 0.2934 | 0.8813  | 0.5844   | 0.4879       |
| logreg-context  | 5    | 0.8113 | 0.7725    | 0.8542 | 0.2664 | 0.8671  | 0.5369   | 0.4337       |
| linear-dynamics | 1    | 0.9214 | 0.9317    | 0.9113 | 0.0708 | 0.9733  | 0.8400   | 0.7821       |
| linear-dynamics | 2    | 0.9009 | 0.9127    | 0.8894 | 0.0901 | 0.9520  | 0.7850   | 0.6940       |
| linear-dynamics | 3    | 0.8631 | 0.8257    | 0.9040 | 0.2021 | 0.9354  | 0.7362   | 0.6322       |
| linear-dynamics | 4    | 0.8499 | 0.8068    | 0.8979 | 0.2278 | 0.9221  | 0.6950   | 0.5730       |
| linear-dynamics | 5    | 0.8419 | 0.7963    | 0.8931 | 0.2420 | 0.9083  | 0.6562   | 0.5282       |

## Dynamics: can the model predict the next state at all?

Next-state error in standardised units. `persistence` is the floor -- it copies the current state forward. A model that cannot beat it has not learned dynamics, whatever its classification scores say.

| Model           | Next-state MSE | MAE    | RMSE   |
|-----------------|----------------|--------|--------|
| persistence     | 1.0064         | 0.4906 | 1.0032 |
| linear-dynamics | 0.6107         | 0.4409 | 0.7815 |

> **Note.** PyTorch was not installed in the environment that produced this report, so the temporal transformer row is absent. Install the requirements and re-run `atdrps benchmark` to fill it in.
