# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_factor_0.2 | 1.4448 | 0.0249 | 0.0176 | 1.2209 | 1.6687 |
| MILO_LW_scale_factor_0.1 | 1.3811 | 0.0178 | 0.0126 | 1.2215 | 1.5407 |
| MILO_LW_scale_factor_0.5 | 1.3329 | 0.0132 | 0.0093 | 1.2143 | 1.4515 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric                |
|:-------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:----------------------|
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.1 |  1.4448  |  1.38114 | MILO_LW_scale_factor_0.1 | 0.111103  |               | final_validation_loss |
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.5 |  1.4448  |  1.33291 | MILO_LW_scale_factor_0.5 | 0.0539478 |               | final_validation_loss |
| MILO_LW_scale_factor_0.1 | MILO_LW_scale_factor_0.5 |  1.38114 |  1.33291 | MILO_LW_scale_factor_0.5 | 0.10057   |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_factor_0.2 | 46.2950 | 0.7802 | 0.5517 | 39.2854 | 53.3046 |
| MILO_LW_scale_factor_0.1 | 48.3800 | 0.6388 | 0.4517 | 42.6410 | 54.1190 |
| MILO_LW_scale_factor_0.5 | 49.8475 | 0.3948 | 0.2792 | 46.3004 | 53.3946 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric                    |
|:-------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.1 |   46.295 |  48.38   | MILO_LW_scale_factor_0.1 | 0.104283  |               | final_validation_accuracy |
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.5 |   46.295 |  49.8475 | MILO_LW_scale_factor_0.5 | 0.0548524 |               | final_validation_accuracy |
| MILO_LW_scale_factor_0.1 | MILO_LW_scale_factor_0.5 |   48.38  |  49.8475 | MILO_LW_scale_factor_0.5 | 0.133478  |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_factor_0.2 | 0.4593 | 0.0102 | 0.0072 | 0.3675 | 0.5511 |
| MILO_LW_scale_factor_0.1 | 0.4829 | 0.0065 | 0.0046 | 0.4240 | 0.5417 |
| MILO_LW_scale_factor_0.5 | 0.4972 | 0.0037 | 0.0026 | 0.4643 | 0.5301 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric                    |
|:-------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.1 | 0.459304 | 0.48285  | MILO_LW_scale_factor_0.1 | 0.13192   |               | final_validation_f1_score |
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.5 | 0.459304 | 0.497186 | MILO_LW_scale_factor_0.5 | 0.0903512 |               | final_validation_f1_score |
| MILO_LW_scale_factor_0.1 | MILO_LW_scale_factor_0.5 | 0.48285  | 0.497186 | MILO_LW_scale_factor_0.5 | 0.14658   |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_factor_0.2 | 0.7995 | 0.0052 | 0.0037 | 0.7525 | 0.8465 |
| MILO_LW_scale_factor_0.1 | 0.8134 | 0.0043 | 0.0030 | 0.7749 | 0.8519 |
| MILO_LW_scale_factor_0.5 | 0.8232 | 0.0026 | 0.0019 | 0.7996 | 0.8468 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric               |
|:-------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:---------------------|
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.1 | 0.799497 | 0.813382 | MILO_LW_scale_factor_0.1 |  0.105348 |               | final_validation_auc |
| MILO_LW_scale_factor_0.2 | MILO_LW_scale_factor_0.5 | 0.799497 | 0.823218 | MILO_LW_scale_factor_0.5 |  0.055435 |               | final_validation_auc |
| MILO_LW_scale_factor_0.1 | MILO_LW_scale_factor_0.5 | 0.813382 | 0.823218 | MILO_LW_scale_factor_0.5 |  0.13399  |               | final_validation_auc |

