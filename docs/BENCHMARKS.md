# ATDRPS benchmark results

All models see identical sequences, an identical split and identical features. Operating thresholds are chosen on the validation split and applied unchanged to test -- tuning a threshold on the data you then report turns a benchmark into a sales pitch.

## Setup

- sequences: **7753** (train 5022 / val 1123 / test 1608)
- context: **16 windows** of 30s; forecast horizon **K = 5**
- state dimension: **96**
- split: held-out captures (train and test share no window or campaign)

## Infiltration forecast, one window ahead

| Model           | Step | F1     | Precision | Recall | FPR    | ROC-AUC | StageAcc | StageMacroF1 |
|-----------------|------|--------|-----------|--------|--------|---------|----------|--------------|
| logreg-static   | 1    | 0.9023 | 0.8757    | 0.9305 | 0.1589 | 0.9546  | 0.7618   | 0.7281       |
| logreg-context  | 1    | 0.9375 | 0.9117    | 0.9647 | 0.1123 | 0.9804  | 0.8209   | 0.7981       |
| linear-dynamics | 1    | 0.9562 | 0.9425    | 0.9704 | 0.0712 | 0.9873  | 0.8514   | 0.8247       |

## Every horizon step

| Model           | Step | F1     | Precision | Recall | FPR    | ROC-AUC | StageAcc | StageMacroF1 |
|-----------------|------|--------|-----------|--------|--------|---------|----------|--------------|
| logreg-static   | 1    | 0.9023 | 0.8757    | 0.9305 | 0.1589 | 0.9546  | 0.7618   | 0.7281       |
| logreg-static   | 2    | 0.8989 | 0.8765    | 0.9225 | 0.1560 | 0.9487  | 0.7382   | 0.6986       |
| logreg-static   | 3    | 0.8934 | 0.8573    | 0.9326 | 0.1858 | 0.9417  | 0.6984   | 0.6515       |
| logreg-static   | 4    | 0.8811 | 0.8614    | 0.9017 | 0.1733 | 0.9311  | 0.6474   | 0.5893       |
| logreg-static   | 5    | 0.8678 | 0.8538    | 0.8822 | 0.1798 | 0.9213  | 0.6051   | 0.5449       |
| logreg-context  | 1    | 0.9375 | 0.9117    | 0.9647 | 0.1123 | 0.9804  | 0.8209   | 0.7981       |
| logreg-context  | 2    | 0.9179 | 0.8830    | 0.9555 | 0.1518 | 0.9664  | 0.7637   | 0.7322       |
| logreg-context  | 3    | 0.9013 | 0.8585    | 0.9486 | 0.1872 | 0.9504  | 0.6723   | 0.6278       |
| logreg-context  | 4    | 0.8842 | 0.8434    | 0.9291 | 0.2060 | 0.9347  | 0.6150   | 0.5636       |
| logreg-context  | 5    | 0.8684 | 0.8243    | 0.9176 | 0.2330 | 0.9246  | 0.5790   | 0.5312       |
| linear-dynamics | 1    | 0.9562 | 0.9425    | 0.9704 | 0.0712 | 0.9873  | 0.8514   | 0.8247       |
| linear-dynamics | 2    | 0.9435 | 0.9254    | 0.9624 | 0.0930 | 0.9789  | 0.8197   | 0.7856       |
| linear-dynamics | 3    | 0.9312 | 0.9124    | 0.9509 | 0.1093 | 0.9693  | 0.7606   | 0.7145       |
| linear-dynamics | 4    | 0.9175 | 0.9073    | 0.9280 | 0.1132 | 0.9583  | 0.7233   | 0.6633       |
| linear-dynamics | 5    | 0.9049 | 0.8614    | 0.9531 | 0.1826 | 0.9501  | 0.6866   | 0.6237       |

## Dynamics: can the model predict the next state at all?

Next-state error in standardised units. `persistence` is the floor -- it copies the current state forward. A model that cannot beat it has not learned dynamics, whatever its classification scores say.

| Model           | Next-state MSE | MAE    | RMSE   |
|-----------------|----------------|--------|--------|
| persistence     | 1.5205         | 0.6094 | 1.2331 |
| linear-dynamics | 0.8677         | 0.5319 | 0.9315 |

> **Note.** PyTorch was not installed in the environment that produced this report, so the temporal transformer row is absent. Install the requirements and re-run `atdrps benchmark` to fill it in.
