# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_factor_0.2 | 0.3790 | 0.1476 | 0.1044 | -0.9473 | 1.7052 |
| MILO_LW_scale_factor_0.1 | 0.1794 | 0.0214 | 0.0151 | -0.0131 | 0.3719 |
| MILO_LW_scale_factor_0.5 | 0.1334 | 0.0187 | 0.0132 | -0.0347 | 0.3016 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric                |
|:-------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:----------------------|
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.1 | 0.378956 | 0.179372 | MILO_LW_scale_factor_0.1 |  0.301702 |               | final_validation_loss |
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.5 | 0.378956 | 0.133414 | MILO_LW_scale_factor_0.5 |  0.251471 |               | final_validation_loss |
| MILO_LW_scale_factor_0.1 | MILO_LW_scale_factor_0.5 | 0.179372 | 0.133414 | MILO_LW_scale_factor_0.5 |  0.151987 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_factor_0.2 | 86.9540 | 4.9299 | 3.4860 | 42.6602 | 131.2478 |
| MILO_LW_scale_factor_0.1 | 93.7150 | 0.7085 | 0.5010 | 87.3492 | 100.0808 |
| MILO_LW_scale_factor_0.5 | 95.3480 | 0.7410 | 0.5240 | 88.6899 | 102.0061 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric                    |
|:-------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.1 |   86.954 |   93.715 | MILO_LW_scale_factor_0.1 |  0.29804  |               | final_validation_accuracy |
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.5 |   86.954 |   95.348 | MILO_LW_scale_factor_0.5 |  0.244391 |               | final_validation_accuracy |
| MILO_LW_scale_factor_0.1 | MILO_LW_scale_factor_0.5 |   93.715 |   95.348 | MILO_LW_scale_factor_0.5 |  0.153338 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_factor_0.2 | 0.8706 | 0.0484 | 0.0342 | 0.4355 | 1.3058 |
| MILO_LW_scale_factor_0.1 | 0.9374 | 0.0068 | 0.0048 | 0.8758 | 0.9989 |
| MILO_LW_scale_factor_0.5 | 0.9535 | 0.0073 | 0.0052 | 0.8878 | 1.0192 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric                    |
|:-------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.1 | 0.870643 | 0.937367 | MILO_LW_scale_factor_0.1 |  0.296987 |               | final_validation_f1_score |
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.5 | 0.870643 | 0.953507 | MILO_LW_scale_factor_0.5 |  0.243233 |               | final_validation_f1_score |
| MILO_LW_scale_factor_0.1 | MILO_LW_scale_factor_0.5 | 0.937367 | 0.953507 | MILO_LW_scale_factor_0.5 |  0.150854 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_factor_0.2 | 0.9910 | 0.0061 | 0.0043 | 0.9364 | 1.0456 |
| MILO_LW_scale_factor_0.1 | 0.9978 | 0.0004 | 0.0003 | 0.9940 | 1.0016 |
| MILO_LW_scale_factor_0.5 | 0.9987 | 0.0003 | 0.0002 | 0.9957 | 1.0018 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric               |
|:-------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:---------------------|
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.1 | 0.990998 |  0.99777 | MILO_LW_scale_factor_0.1 |  0.358764 |               | final_validation_auc |
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.5 | 0.990998 |  0.99875 | MILO_LW_scale_factor_0.5 |  0.321347 |               | final_validation_auc |
| MILO_LW_scale_factor_0.1 | MILO_LW_scale_factor_0.5 | 0.99777  |  0.99875 | MILO_LW_scale_factor_0.5 |  0.131159 |               | final_validation_auc |

