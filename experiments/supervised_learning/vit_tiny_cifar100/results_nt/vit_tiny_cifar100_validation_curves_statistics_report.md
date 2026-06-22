# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 3.7058 | 0.0073 | 0.0033 | 3.6967 | 3.7150 |
| MILO_LW | 3.7044 | 0.0086 | 0.0038 | 3.6938 | 3.7151 |
| SGD | 3.7608 | 0.0202 | 0.0090 | 3.7357 | 3.7860 |
| ADAMW | 2.9230 | 0.0311 | 0.0139 | 2.8843 | 2.9616 |
| ADAGRAD | 3.5907 | 0.0132 | 0.0059 | 3.5743 | 3.6070 |
| ADEMAMIX | 2.9166 | 0.0367 | 0.0164 | 2.8711 | 2.9622 |
| SOAP | 2.2634 | 0.0183 | 0.0082 | 2.2408 | 2.2861 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  3.70585 |  3.70443 | MILO_LW  | 0.785641    |               | final_validation_loss |
| MILO          | SGD           |  3.70585 |  3.76083 | MILO     | 0.00224403  | **            | final_validation_loss |
| MILO          | ADAMW         |  3.70585 |  2.92298 | ADAMW    | 1.85112e-07 | ***           | final_validation_loss |
| MILO          | ADAGRAD       |  3.70585 |  3.59066 | ADAGRAD  | 1.74333e-06 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      |  3.70585 |  2.91663 | ADEMAMIX | 4.99612e-07 | ***           | final_validation_loss |
| MILO          | SOAP          |  3.70585 |  2.26343 | SOAP     | 5.88598e-11 | ***           | final_validation_loss |
| MILO_LW       | SGD           |  3.70443 |  3.76083 | MILO_LW  | 0.00175096  | **            | final_validation_loss |
| MILO_LW       | ADAMW         |  3.70443 |  2.92298 | ADAMW    | 1.23993e-07 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       |  3.70443 |  3.59066 | ADAGRAD  | 1.0175e-06  | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  3.70443 |  2.91663 | ADEMAMIX | 3.79575e-07 | ***           | final_validation_loss |
| MILO_LW       | SOAP          |  3.70443 |  2.26343 | SOAP     | 1.3654e-11  | ***           | final_validation_loss |
| SGD           | ADAMW         |  3.76083 |  2.92298 | ADAMW    | 4.39037e-10 | ***           | final_validation_loss |
| SGD           | ADAGRAD       |  3.76083 |  3.59066 | ADAGRAD  | 1.18487e-06 | ***           | final_validation_loss |
| SGD           | ADEMAMIX      |  3.76083 |  2.91663 | ADEMAMIX | 4.52015e-09 | ***           | final_validation_loss |
| SGD           | SOAP          |  3.76083 |  2.26343 | SOAP     | 2.83377e-14 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       |  2.92298 |  3.59066 | ADAMW    | 4.13113e-08 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      |  2.92298 |  2.91663 | ADEMAMIX | 0.775632    |               | final_validation_loss |
| ADAMW         | SOAP          |  2.92298 |  2.26343 | SOAP     | 4.78968e-09 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  3.59066 |  2.91663 | ADEMAMIX | 2.07875e-07 | ***           | final_validation_loss |
| ADAGRAD       | SOAP          |  3.59066 |  2.26343 | SOAP     | 1.45461e-13 | ***           | final_validation_loss |
| ADEMAMIX      | SOAP          |  2.91663 |  2.26343 | SOAP     | 4.34659e-08 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 16.8385 | 0.1723 | 0.0771 | 16.6246 | 17.0525 |
| MILO_LW | 16.7467 | 0.4311 | 0.1928 | 16.2114 | 17.2820 |
| SGD | 13.1170 | 0.4615 | 0.2064 | 12.5440 | 13.6900 |
| ADAMW | 26.2815 | 0.5297 | 0.2369 | 25.6237 | 26.9392 |
| ADAGRAD | 18.0356 | 0.2471 | 0.1105 | 17.7287 | 18.3424 |
| ADEMAMIX | 26.3644 | 0.6445 | 0.2882 | 25.5643 | 27.1646 |
| SOAP | 40.5837 | 0.1908 | 0.0853 | 40.3468 | 40.8206 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  16.8385 |  16.7467 | MILO     | 0.675857    |               | final_validation_accuracy |
| MILO          | SGD           |  16.8385 |  13.117  | MILO     | 1.14072e-05 | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  16.8385 |  26.2815 | ADAMW    | 3.57725e-07 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  16.8385 |  18.0356 | ADAGRAD  | 4.09989e-05 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  16.8385 |  26.3644 | ADEMAMIX | 1.51092e-06 | ***           | final_validation_accuracy |
| MILO          | SOAP          |  16.8385 |  40.5837 | SOAP     | 4.62162e-16 | ***           | final_validation_accuracy |
| MILO_LW       | SGD           |  16.7467 |  13.117  | MILO_LW  | 1.32251e-06 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  16.7467 |  26.2815 | ADAMW    | 2.25783e-09 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  16.7467 |  18.0356 | ADAGRAD  | 0.000930688 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  16.7467 |  26.3644 | ADEMAMIX | 2.10182e-08 | ***           | final_validation_accuracy |
| MILO_LW       | SOAP          |  16.7467 |  40.5837 | SOAP     | 1.74415e-10 | ***           | final_validation_accuracy |
| SGD           | ADAMW         |  13.117  |  26.2815 | ADAMW    | 1.61948e-10 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  13.117  |  18.0356 | ADAGRAD  | 6.1525e-07  | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  13.117  |  26.3644 | ADEMAMIX | 1.47641e-09 | ***           | final_validation_accuracy |
| SGD           | SOAP          |  13.117  |  40.5837 | SOAP     | 2.08141e-10 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  26.2815 |  18.0356 | ADAMW    | 1.39486e-07 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  26.2815 |  26.3644 | ADEMAMIX | 0.829797    |               | final_validation_accuracy |
| ADAMW         | SOAP          |  26.2815 |  40.5837 | SOAP     | 3.02085e-08 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  18.0356 |  26.3644 | ADEMAMIX | 9.54179e-07 | ***           | final_validation_accuracy |
| ADAGRAD       | SOAP          |  18.0356 |  40.5837 | SOAP     | 1.38692e-14 | ***           | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  26.3644 |  40.5837 | SOAP     | 1.79266e-07 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.1321 | 0.0028 | 0.0013 | 0.1286 | 0.1356 |
| MILO_LW | 0.1300 | 0.0041 | 0.0019 | 0.1249 | 0.1351 |
| SGD | 0.0990 | 0.0060 | 0.0027 | 0.0914 | 0.1065 |
| ADAMW | 0.2462 | 0.0094 | 0.0042 | 0.2345 | 0.2580 |
| ADAGRAD | 0.1470 | 0.0033 | 0.0015 | 0.1429 | 0.1511 |
| ADEMAMIX | 0.2494 | 0.0057 | 0.0026 | 0.2423 | 0.2566 |
| SOAP | 0.3998 | 0.0029 | 0.0013 | 0.3962 | 0.4034 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |    Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|----------:|----------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.132143  | 0.12999   | MILO     | 0.368003    |               | final_validation_f1_score |
| MILO          | SGD           | 0.132143  | 0.0989512 | MILO     | 4.63525e-05 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.132143  | 0.246239  | ADAMW    | 2.93932e-06 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.132143  | 0.146993  | ADAGRAD  | 7.12227e-05 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.132143  | 0.24943   | ADEMAMIX | 2.17813e-08 | ***           | final_validation_f1_score |
| MILO          | SOAP          | 0.132143  | 0.399767  | SOAP     | 4.83785e-15 | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.12999   | 0.0989512 | MILO_LW  | 2.85001e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.12999   | 0.246239  | ADAMW    | 7.01129e-07 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.12999   | 0.146993  | ADAGRAD  | 0.000119679 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.12999   | 0.24943   | ADEMAMIX | 1.31382e-09 | ***           | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.12999   | 0.399767  | SOAP     | 4.57776e-13 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.0989512 | 0.246239  | ADAMW    | 2.0011e-08  | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.0989512 | 0.146993  | ADAGRAD  | 3.24529e-06 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.0989512 | 0.24943   | ADEMAMIX | 1.63791e-10 | ***           | final_validation_f1_score |
| SGD           | SOAP          | 0.0989512 | 0.399767  | SOAP     | 1.58213e-10 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.246239  | 0.146993  | ADAMW    | 3.62264e-06 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.246239  | 0.24943   | ADEMAMIX | 0.540228    |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.246239  | 0.399767  | SOAP     | 6.8065e-07  | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.146993  | 0.24943   | ADEMAMIX | 1.59664e-08 | ***           | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.146993  | 0.399767  | SOAP     | 2.55527e-14 | ***           | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.24943   | 0.399767  | SOAP     | 4.29258e-09 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.8579 | 0.0017 | 0.0008 | 0.8558 | 0.8600 |
| MILO_LW | 0.8567 | 0.0008 | 0.0004 | 0.8557 | 0.8577 |
| SGD | 0.8272 | 0.0037 | 0.0017 | 0.8226 | 0.8318 |
| ADAMW | 0.9166 | 0.0017 | 0.0007 | 0.9145 | 0.9186 |
| ADAGRAD | 0.8642 | 0.0031 | 0.0014 | 0.8604 | 0.8681 |
| ADEMAMIX | 0.9158 | 0.0020 | 0.0009 | 0.9133 | 0.9183 |
| SOAP | 0.9538 | 0.0009 | 0.0004 | 0.9527 | 0.9549 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.857937 | 0.85673  | MILO     | 0.201699    |               | final_validation_auc |
| MILO          | SGD           | 0.857937 | 0.827238 | MILO     | 5.30029e-06 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.857937 | 0.916564 | ADAMW    | 1.25002e-11 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.857937 | 0.864238 | ADAGRAD  | 0.00660713  | **            | final_validation_auc |
| MILO          | ADEMAMIX      | 0.857937 | 0.915778 | ADEMAMIX | 5.56645e-11 | ***           | final_validation_auc |
| MILO          | SOAP          | 0.857937 | 0.953806 | SOAP     | 3.19887e-11 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.85673  | 0.827238 | MILO_LW  | 3.27854e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.85673  | 0.916564 | ADAMW    | 7.46921e-10 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.85673  | 0.864238 | ADAGRAD  | 0.00431525  | **            | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.85673  | 0.915778 | ADEMAMIX | 9.85492e-09 | ***           | final_validation_auc |
| MILO_LW       | SOAP          | 0.85673  | 0.953806 | SOAP     | 1.10389e-15 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.827238 | 0.916564 | ADAMW    | 1.60602e-08 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.827238 | 0.864238 | ADAGRAD  | 1.95182e-07 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.827238 | 0.915778 | ADEMAMIX | 4.20064e-09 | ***           | final_validation_auc |
| SGD           | SOAP          | 0.827238 | 0.953806 | SOAP     | 4.78385e-08 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.916564 | 0.864238 | ADAMW    | 3.6075e-08  | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.916564 | 0.915778 | ADAMW    | 0.519367    |               | final_validation_auc |
| ADAMW         | SOAP          | 0.916564 | 0.953806 | SOAP     | 7.20347e-09 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.864238 | 0.915778 | ADEMAMIX | 1.12931e-08 | ***           | final_validation_auc |
| ADAGRAD       | SOAP          | 0.864238 | 0.953806 | SOAP     | 5.73858e-08 | ***           | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.915778 | 0.953806 | SOAP     | 6.93154e-08 | ***           | final_validation_auc |

