# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.3210 | 0.0053 | 0.0024 | 0.3144 | 0.3276 |
| MILO_LW | 0.3021 | 0.0042 | 0.0019 | 0.2970 | 0.3073 |
| MUON | 0.3136 | 0.0052 | 0.0023 | 0.3071 | 0.3201 |
| ADALAYER | 0.4311 | 0.0364 | 0.0163 | 0.3860 | 0.4763 |
| ADAM_MINI | 0.5188 | 0.0310 | 0.0139 | 0.4803 | 0.5573 |
| SGD | 0.3094 | 0.0018 | 0.0008 | 0.3072 | 0.3115 |
| ADAMW | 0.4600 | 0.0094 | 0.0042 | 0.4483 | 0.4716 |
| ADAGRAD | 0.3877 | 0.0032 | 0.0014 | 0.3837 | 0.3918 |
| NOVOGRAD | 0.3128 | 0.0109 | 0.0049 | 0.2993 | 0.3263 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.321005 | 0.302149 | MILO_LW  | 0.000311763 | ***           | final_validation_loss |
| MILO          | MUON          | 0.321005 | 0.313568 | MUON     | 0.0565791   |               | final_validation_loss |
| MILO          | ADALAYER      | 0.321005 | 0.431122 | MILO     | 0.00221998  | **            | final_validation_loss |
| MILO          | ADAM_MINI     | 0.321005 | 0.518803 | MILO     | 0.000103165 | ***           | final_validation_loss |
| MILO          | SGD           | 0.321005 | 0.309374 | SGD      | 0.00603195  | **            | final_validation_loss |
| MILO          | ADAMW         | 0.321005 | 0.45997  | MILO     | 5.91628e-08 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.321005 | 0.38774  | MILO     | 1.13589e-07 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.321005 | 0.312832 | NOVOGRAD | 0.183227    |               | final_validation_loss |
| MILO_LW       | MUON          | 0.302149 | 0.313568 | MILO_LW  | 0.00560919  | **            | final_validation_loss |
| MILO_LW       | ADALAYER      | 0.302149 | 0.431122 | MILO_LW  | 0.00125971  | **            | final_validation_loss |
| MILO_LW       | ADAM_MINI     | 0.302149 | 0.518803 | MILO_LW  | 8.01915e-05 | ***           | final_validation_loss |
| MILO_LW       | SGD           | 0.302149 | 0.309374 | MILO_LW  | 0.0139281   | *             | final_validation_loss |
| MILO_LW       | ADAMW         | 0.302149 | 0.45997  | MILO_LW  | 1.21454e-07 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.302149 | 0.38774  | MILO_LW  | 9.33441e-10 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.302149 | 0.312832 | MILO_LW  | 0.0936666   |               | final_validation_loss |
| MUON          | ADALAYER      | 0.313568 | 0.431122 | MUON     | 0.00172755  | **            | final_validation_loss |
| MUON          | ADAM_MINI     | 0.313568 | 0.518803 | MUON     | 8.92798e-05 | ***           | final_validation_loss |
| MUON          | SGD           | 0.313568 | 0.309374 | SGD      | 0.15182     |               | final_validation_loss |
| MUON          | ADAMW         | 0.313568 | 0.45997  | MUON     | 4.67917e-08 | ***           | final_validation_loss |
| MUON          | ADAGRAD       | 0.313568 | 0.38774  | MUON     | 4.73329e-08 | ***           | final_validation_loss |
| MUON          | NOVOGRAD      | 0.313568 | 0.312832 | NOVOGRAD | 0.896087    |               | final_validation_loss |
| ADALAYER      | ADAM_MINI     | 0.431122 | 0.518803 | ADALAYER | 0.0036136   | **            | final_validation_loss |
| ADALAYER      | SGD           | 0.431122 | 0.309374 | SGD      | 0.00167863  | **            | final_validation_loss |
| ADALAYER      | ADAMW         | 0.431122 | 0.45997  | ADALAYER | 0.152553    |               | final_validation_loss |
| ADALAYER      | ADAGRAD       | 0.431122 | 0.38774  | ADAGRAD  | 0.0556392   |               | final_validation_loss |
| ADALAYER      | NOVOGRAD      | 0.431122 | 0.312832 | NOVOGRAD | 0.00119064  | **            | final_validation_loss |
| ADAM_MINI     | SGD           | 0.518803 | 0.309374 | SGD      | 0.00010834  | ***           | final_validation_loss |
| ADAM_MINI     | ADAMW         | 0.518803 | 0.45997  | ADAMW    | 0.0109438   | *             | final_validation_loss |
| ADAM_MINI     | ADAGRAD       | 0.518803 | 0.38774  | ADAGRAD  | 0.000644711 | ***           | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      | 0.518803 | 0.312832 | NOVOGRAD | 3.49453e-05 | ***           | final_validation_loss |
| SGD           | ADAMW         | 0.309374 | 0.45997  | SGD      | 1.93894e-06 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 0.309374 | 0.38774  | SGD      | 3.86817e-09 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 0.309374 | 0.312832 | SGD      | 0.519302    |               | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.45997  | 0.38774  | ADAGRAD  | 1.75971e-05 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.45997  | 0.312832 | NOVOGRAD | 1.84222e-08 | ***           | final_validation_loss |
| ADAGRAD       | NOVOGRAD      | 0.38774  | 0.312832 | NOVOGRAD | 4.0208e-05  | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 91.5037 | 0.1819 | 0.0814 | 91.2778 | 91.7296 |
| MILO_LW | 91.7827 | 0.1273 | 0.0569 | 91.6247 | 91.9408 |
| MUON | 91.4914 | 0.2007 | 0.0897 | 91.2422 | 91.7405 |
| ADALAYER | 88.6469 | 1.2192 | 0.5452 | 87.1331 | 90.1607 |
| ADAM_MINI | 89.2815 | 0.2849 | 0.1274 | 88.9277 | 89.6353 |
| SGD | 91.4963 | 0.1123 | 0.0502 | 91.3569 | 91.6357 |
| ADAMW | 90.2741 | 0.3701 | 0.1655 | 89.8145 | 90.7336 |
| ADAGRAD | 89.6198 | 0.0835 | 0.0373 | 89.5161 | 89.7234 |
| NOVOGRAD | 91.3333 | 0.2161 | 0.0967 | 91.0650 | 91.6017 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  91.5037 |  91.7827 | MILO_LW   | 0.0255619   | *             | final_validation_accuracy |
| MILO          | MUON          |  91.5037 |  91.4914 | MILO      | 0.921357    |               | final_validation_accuracy |
| MILO          | ADALAYER      |  91.5037 |  88.6469 | MILO      | 0.00585305  | **            | final_validation_accuracy |
| MILO          | ADAM_MINI     |  91.5037 |  89.2815 | MILO      | 2.10949e-06 | ***           | final_validation_accuracy |
| MILO          | SGD           |  91.5037 |  91.4963 | MILO      | 0.940523    |               | final_validation_accuracy |
| MILO          | ADAMW         |  91.5037 |  90.2741 | MILO      | 0.000621796 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  91.5037 |  89.6198 | MILO      | 1.48399e-06 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  91.5037 |  91.3333 | MILO      | 0.21549     |               | final_validation_accuracy |
| MILO_LW       | MUON          |  91.7827 |  91.4914 | MILO_LW   | 0.0298145   | *             | final_validation_accuracy |
| MILO_LW       | ADALAYER      |  91.7827 |  88.6469 | MILO_LW   | 0.00432733  | **            | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  91.7827 |  89.2815 | MILO_LW   | 4.0928e-06  | ***           | final_validation_accuracy |
| MILO_LW       | SGD           |  91.7827 |  91.4963 | MILO_LW   | 0.00559453  | **            | final_validation_accuracy |
| MILO_LW       | ADAMW         |  91.7827 |  90.2741 | MILO_LW   | 0.000370944 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  91.7827 |  89.6198 | MILO_LW   | 9.65642e-09 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  91.7827 |  91.3333 | MILO_LW   | 0.0060423   | **            | final_validation_accuracy |
| MUON          | ADALAYER      |  91.4914 |  88.6469 | MUON      | 0.0058505   | **            | final_validation_accuracy |
| MUON          | ADAM_MINI     |  91.4914 |  89.2815 | MUON      | 1.63063e-06 | ***           | final_validation_accuracy |
| MUON          | SGD           |  91.4914 |  91.4963 | SGD       | 0.963189    |               | final_validation_accuracy |
| MUON          | ADAMW         |  91.4914 |  90.2741 | MUON      | 0.000582502 | ***           | final_validation_accuracy |
| MUON          | ADAGRAD       |  91.4914 |  89.6198 | MUON      | 3.83579e-06 | ***           | final_validation_accuracy |
| MUON          | NOVOGRAD      |  91.4914 |  91.3333 | MUON      | 0.265349    |               | final_validation_accuracy |
| ADALAYER      | ADAM_MINI     |  88.6469 |  89.2815 | ADAM_MINI | 0.314599    |               | final_validation_accuracy |
| ADALAYER      | SGD           |  88.6469 |  91.4963 | SGD       | 0.00620508  | **            | final_validation_accuracy |
| ADALAYER      | ADAMW         |  88.6469 |  90.2741 | ADAMW     | 0.0379272   | *             | final_validation_accuracy |
| ADALAYER      | ADAGRAD       |  88.6469 |  89.6198 | ADAGRAD   | 0.149002    |               | final_validation_accuracy |
| ADALAYER      | NOVOGRAD      |  88.6469 |  91.3333 | NOVOGRAD  | 0.00713486  | **            | final_validation_accuracy |
| ADAM_MINI     | SGD           |  89.2815 |  91.4963 | SGD       | 1.17923e-05 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAMW         |  89.2815 |  90.2741 | ADAMW     | 0.00171601  | **            | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  89.2815 |  89.6198 | ADAGRAD   | 0.0546631   |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  89.2815 |  91.3333 | NOVOGRAD  | 2.37688e-06 | ***           | final_validation_accuracy |
| SGD           | ADAMW         |  91.4963 |  90.2741 | SGD       | 0.00110103  | **            | final_validation_accuracy |
| SGD           | ADAGRAD       |  91.4963 |  89.6198 | SGD       | 5.47511e-09 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  91.4963 |  91.3333 | SGD       | 0.185155    |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  90.2741 |  89.6198 | ADAMW     | 0.0151739   | *             | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  90.2741 |  91.3333 | NOVOGRAD  | 0.00116593  | **            | final_validation_accuracy |
| ADAGRAD       | NOVOGRAD      |  89.6198 |  91.3333 | NOVOGRAD  | 1.13034e-05 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9143 | 0.0018 | 0.0008 | 0.9120 | 0.9165 |
| MILO_LW | 0.9172 | 0.0013 | 0.0006 | 0.9156 | 0.9188 |
| MUON | 0.9143 | 0.0020 | 0.0009 | 0.9118 | 0.9168 |
| ADALAYER | 0.8856 | 0.0119 | 0.0053 | 0.8709 | 0.9004 |
| ADAM_MINI | 0.8916 | 0.0029 | 0.0013 | 0.8880 | 0.8952 |
| SGD | 0.9141 | 0.0011 | 0.0005 | 0.9127 | 0.9155 |
| ADAMW | 0.9018 | 0.0038 | 0.0017 | 0.8971 | 0.9065 |
| ADAGRAD | 0.8953 | 0.0009 | 0.0004 | 0.8942 | 0.8964 |
| NOVOGRAD | 0.9125 | 0.0025 | 0.0011 | 0.9095 | 0.9156 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.914294 | 0.917191 | MILO_LW   | 0.0216883   | *             | final_validation_f1_score |
| MILO          | MUON          | 0.914294 | 0.914292 | MILO      | 0.998359    |               | final_validation_f1_score |
| MILO          | ADALAYER      | 0.914294 | 0.885643 | MILO      | 0.00522961  | **            | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.914294 | 0.891574 | MILO      | 2.20207e-06 | ***           | final_validation_f1_score |
| MILO          | SGD           | 0.914294 | 0.914116 | MILO      | 0.858471    |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.914294 | 0.9018   | MILO      | 0.000686733 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.914294 | 0.895334 | MILO      | 1.13606e-06 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.914294 | 0.912536 | MILO      | 0.239576    |               | final_validation_f1_score |
| MILO_LW       | MUON          | 0.917191 | 0.914292 | MILO_LW   | 0.0312602   | *             | final_validation_f1_score |
| MILO_LW       | ADALAYER      | 0.917191 | 0.885643 | MILO_LW   | 0.00382824  | **            | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.917191 | 0.891574 | MILO_LW   | 4.2866e-06  | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.917191 | 0.914116 | MILO_LW   | 0.00392737  | **            | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.917191 | 0.9018   | MILO_LW   | 0.000400345 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.917191 | 0.895334 | MILO_LW   | 6.47706e-09 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.917191 | 0.912536 | MILO_LW   | 0.00972957  | **            | final_validation_f1_score |
| MUON          | ADALAYER      | 0.914292 | 0.885643 | MUON      | 0.00512558  | **            | final_validation_f1_score |
| MUON          | ADAM_MINI     | 0.914292 | 0.891574 | MUON      | 1.56435e-06 | ***           | final_validation_f1_score |
| MUON          | SGD           | 0.914292 | 0.914116 | MUON      | 0.87144     |               | final_validation_f1_score |
| MUON          | ADAMW         | 0.914292 | 0.9018   | MUON      | 0.000601894 | ***           | final_validation_f1_score |
| MUON          | ADAGRAD       | 0.914292 | 0.895334 | MUON      | 3.25405e-06 | ***           | final_validation_f1_score |
| MUON          | NOVOGRAD      | 0.914292 | 0.912536 | MUON      | 0.256064    |               | final_validation_f1_score |
| ADALAYER      | ADAM_MINI     | 0.885643 | 0.891574 | ADAM_MINI | 0.332927    |               | final_validation_f1_score |
| ADALAYER      | SGD           | 0.885643 | 0.914116 | SGD       | 0.00563651  | **            | final_validation_f1_score |
| ADALAYER      | ADAMW         | 0.885643 | 0.9018   | ADAMW     | 0.035461    | *             | final_validation_f1_score |
| ADALAYER      | ADAGRAD       | 0.885643 | 0.895334 | ADAGRAD   | 0.142135    |               | final_validation_f1_score |
| ADALAYER      | NOVOGRAD      | 0.885643 | 0.912536 | NOVOGRAD  | 0.00620813  | **            | final_validation_f1_score |
| ADAM_MINI     | SGD           | 0.891574 | 0.914116 | SGD       | 1.17112e-05 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAMW         | 0.891574 | 0.9018   | ADAMW     | 0.00168174  | **            | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.891574 | 0.895334 | ADAGRAD   | 0.041704    | *             | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.891574 | 0.912536 | NOVOGRAD  | 2.20592e-06 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.914116 | 0.9018   | SGD       | 0.00120914  | **            | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.914116 | 0.895334 | SGD       | 5.69228e-09 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.914116 | 0.912536 | SGD       | 0.24602     |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.9018   | 0.895334 | ADAMW     | 0.0173897   | *             | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.9018   | 0.912536 | NOVOGRAD  | 0.00120792  | **            | final_validation_f1_score |
| ADAGRAD       | NOVOGRAD      | 0.895334 | 0.912536 | NOVOGRAD  | 2.74716e-05 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9926 | 0.0001 | 0.0001 | 0.9924 | 0.9927 |
| MILO_LW | 0.9932 | 0.0001 | 0.0001 | 0.9931 | 0.9934 |
| MUON | 0.9927 | 0.0001 | 0.0000 | 0.9926 | 0.9928 |
| ADALAYER | 0.9904 | 0.0007 | 0.0003 | 0.9895 | 0.9912 |
| ADAM_MINI | 0.9907 | 0.0004 | 0.0002 | 0.9902 | 0.9912 |
| SGD | 0.9925 | 0.0001 | 0.0000 | 0.9924 | 0.9926 |
| ADAMW | 0.9912 | 0.0002 | 0.0001 | 0.9910 | 0.9914 |
| ADAGRAD | 0.9905 | 0.0001 | 0.0000 | 0.9904 | 0.9906 |
| NOVOGRAD | 0.9925 | 0.0002 | 0.0001 | 0.9923 | 0.9927 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.992553 | 0.993239 | MILO_LW   | 5.30034e-05 | ***           | final_validation_auc |
| MILO          | MUON          | 0.992553 | 0.992711 | MUON      | 0.0939506   |               | final_validation_auc |
| MILO          | ADALAYER      | 0.992553 | 0.990363 | MILO      | 0.00142456  | **            | final_validation_auc |
| MILO          | ADAM_MINI     | 0.992553 | 0.990704 | MILO      | 0.000159622 | ***           | final_validation_auc |
| MILO          | SGD           | 0.992553 | 0.992512 | MILO      | 0.585467    |               | final_validation_auc |
| MILO          | ADAMW         | 0.992553 | 0.991161 | MILO      | 5.57517e-07 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.992553 | 0.990505 | MILO      | 5.12495e-07 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.992553 | 0.99249  | MILO      | 0.530891    |               | final_validation_auc |
| MILO_LW       | MUON          | 0.993239 | 0.992711 | MILO_LW   | 9.93239e-05 | ***           | final_validation_auc |
| MILO_LW       | ADALAYER      | 0.993239 | 0.990363 | MILO_LW   | 0.00051178  | ***           | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.993239 | 0.990704 | MILO_LW   | 4.9486e-05  | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.993239 | 0.992512 | MILO_LW   | 3.61649e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.993239 | 0.991161 | MILO_LW   | 2.77871e-08 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.993239 | 0.990505 | MILO_LW   | 1.27309e-08 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.993239 | 0.99249  | MILO_LW   | 4.20035e-05 | ***           | final_validation_auc |
| MUON          | ADALAYER      | 0.992711 | 0.990363 | MUON      | 0.00121558  | **            | final_validation_auc |
| MUON          | ADAM_MINI     | 0.992711 | 0.990704 | MUON      | 0.000170726 | ***           | final_validation_auc |
| MUON          | SGD           | 0.992711 | 0.992512 | MUON      | 0.0112957   | *             | final_validation_auc |
| MUON          | ADAMW         | 0.992711 | 0.991161 | MUON      | 3.36465e-07 | ***           | final_validation_auc |
| MUON          | ADAGRAD       | 0.992711 | 0.990505 | MUON      | 1.04262e-08 | ***           | final_validation_auc |
| MUON          | NOVOGRAD      | 0.992711 | 0.99249  | MUON      | 0.0354431   | *             | final_validation_auc |
| ADALAYER      | ADAM_MINI     | 0.990363 | 0.990704 | ADAM_MINI | 0.361323    |               | final_validation_auc |
| ADALAYER      | SGD           | 0.990363 | 0.992512 | SGD       | 0.00192988  | **            | final_validation_auc |
| ADALAYER      | ADAMW         | 0.990363 | 0.991161 | ADAMW     | 0.054567    |               | final_validation_auc |
| ADALAYER      | ADAGRAD       | 0.990363 | 0.990505 | ADAGRAD   | 0.660541    |               | final_validation_auc |
| ADALAYER      | NOVOGRAD      | 0.990363 | 0.99249  | NOVOGRAD  | 0.0015576   | **            | final_validation_auc |
| ADAM_MINI     | SGD           | 0.990704 | 0.992512 | SGD       | 0.000426686 | ***           | final_validation_auc |
| ADAM_MINI     | ADAMW         | 0.990704 | 0.991161 | ADAMW     | 0.0575607   |               | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.990704 | 0.990505 | ADAM_MINI | 0.322365    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.990704 | 0.99249  | NOVOGRAD  | 0.000170564 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.992512 | 0.991161 | SGD       | 1.11541e-05 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.992512 | 0.990505 | SGD       | 1.58667e-11 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.992512 | 0.99249  | SGD       | 0.776892    |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.991161 | 0.990505 | ADAMW     | 0.000294725 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.991161 | 0.99249  | NOVOGRAD  | 9.48359e-07 | ***           | final_validation_auc |
| ADAGRAD       | NOVOGRAD      | 0.990505 | 0.99249  | NOVOGRAD  | 1.04454e-06 | ***           | final_validation_auc |

