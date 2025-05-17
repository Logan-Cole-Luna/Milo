# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_weight_decay_0.0001 | 0.5355 | 0.1503 | 0.1063 | -0.8148 | 1.8858 |
| MILO_LW_weight_decay_0.001 | 0.4878 | 0.0499 | 0.0353 | 0.0398 | 0.9357 |
| MILO_LW_weight_decay_0.0 | 0.5809 | 0.0802 | 0.0567 | -0.1396 | 1.3014 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A                 | Optimizer B                |   Mean A |   Mean B | Better                      |   p-value | Significant   | Metric                |
|:----------------------------|:---------------------------|---------:|---------:|:----------------------------|----------:|:--------------|:----------------------|
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.001 | 0.535486 | 0.487755 | MILO_LW_weight_decay_0.001  |  0.732978 |               | final_validation_loss |
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.0   | 0.535486 | 0.58087  | MILO_LW_weight_decay_0.0001 |  0.752077 |               | final_validation_loss |
| MILO_LW_weight_decay_0.001  | MILO_LW_weight_decay_0.0   | 0.487755 | 0.58087  | MILO_LW_weight_decay_0.001  |  0.319622 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_weight_decay_0.0001 | 94.5260 | 1.0607 | 0.7500 | 84.9963 | 104.0557 |
| MILO_LW_weight_decay_0.001 | 94.9050 | 0.5020 | 0.3550 | 90.3943 | 99.4157 |
| MILO_LW_weight_decay_0.0 | 94.1930 | 0.3833 | 0.2710 | 90.7496 | 97.6364 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A                 | Optimizer B                |   Mean A |   Mean B | Better                      |   p-value | Significant   | Metric                    |
|:----------------------------|:---------------------------|---------:|---------:|:----------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.001 |   94.526 |   94.905 | MILO_LW_weight_decay_0.001  |  0.707694 |               | final_validation_accuracy |
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.0   |   94.526 |   94.193 | MILO_LW_weight_decay_0.0001 |  0.736375 |               | final_validation_accuracy |
| MILO_LW_weight_decay_0.001  | MILO_LW_weight_decay_0.0   |   94.905 |   94.193 | MILO_LW_weight_decay_0.001  |  0.260302 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_weight_decay_0.0001 | 0.9449 | 0.0115 | 0.0081 | 0.8413 | 1.0484 |
| MILO_LW_weight_decay_0.001 | 0.9488 | 0.0049 | 0.0035 | 0.9045 | 0.9931 |
| MILO_LW_weight_decay_0.0 | 0.9421 | 0.0037 | 0.0026 | 0.9093 | 0.9750 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A                 | Optimizer B                |   Mean A |   Mean B | Better                      |   p-value | Significant   | Metric                    |
|:----------------------------|:---------------------------|---------:|---------:|:----------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.001 | 0.944873 | 0.948827 | MILO_LW_weight_decay_0.001  |  0.716405 |               | final_validation_f1_score |
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.0   | 0.944873 | 0.942135 | MILO_LW_weight_decay_0.0001 |  0.795566 |               | final_validation_f1_score |
| MILO_LW_weight_decay_0.001  | MILO_LW_weight_decay_0.0   | 0.948827 | 0.942135 | MILO_LW_weight_decay_0.001  |  0.273218 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_weight_decay_0.0001 | 0.9978 | 0.0005 | 0.0004 | 0.9931 | 1.0025 |
| MILO_LW_weight_decay_0.001 | 0.9982 | 0.0001 | 0.0001 | 0.9975 | 0.9988 |
| MILO_LW_weight_decay_0.0 | 0.9980 | 0.0001 | 0.0001 | 0.9971 | 0.9990 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A                 | Optimizer B                |   Mean A |   Mean B | Better                     |   p-value | Significant   | Metric               |
|:----------------------------|:---------------------------|---------:|---------:|:---------------------------|----------:|:--------------|:---------------------|
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.001 | 0.997783 | 0.998157 | MILO_LW_weight_decay_0.001 |  0.493204 |               | final_validation_auc |
| MILO_LW_weight_decay_0.0001 | MILO_LW_weight_decay_0.0   | 0.997783 | 0.998045 | MILO_LW_weight_decay_0.0   |  0.60501  |               | final_validation_auc |
| MILO_LW_weight_decay_0.001  | MILO_LW_weight_decay_0.0   | 0.998157 | 0.998045 | MILO_LW_weight_decay_0.001 |  0.372334 |               | final_validation_auc |

