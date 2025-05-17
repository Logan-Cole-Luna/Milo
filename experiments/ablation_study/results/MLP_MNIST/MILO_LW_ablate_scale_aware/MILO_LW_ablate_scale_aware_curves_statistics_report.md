# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_aware_True | 1.5828 | 0.0299 | 0.0212 | 1.3140 | 1.8516 |
| MILO_LW_scale_aware_False | 1.5236 | 0.0153 | 0.0108 | 1.3861 | 1.6612 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric                |
|:-------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:----------------------|
| MILO_LW_scale_aware_True | MILO_LW_scale_aware_False |  1.58279 |  1.52362 | MILO_LW_scale_aware_False |  0.171726 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_aware_True | 42.1925 | 0.5268 | 0.3725 | 37.4594 | 46.9256 |
| MILO_LW_scale_aware_False | 43.4992 | 0.3429 | 0.2425 | 40.4179 | 46.5804 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric                    |
|:-------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_scale_aware_True | MILO_LW_scale_aware_False |  42.1925 |  43.4992 | MILO_LW_scale_aware_False |  0.117832 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_aware_True | 0.4042 | 0.0049 | 0.0035 | 0.3598 | 0.4486 |
| MILO_LW_scale_aware_False | 0.4155 | 0.0027 | 0.0019 | 0.3913 | 0.4397 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric                    |
|:-------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_scale_aware_True | MILO_LW_scale_aware_False | 0.404223 | 0.415472 | MILO_LW_scale_aware_False |  0.140125 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_aware_True | 0.7705 | 0.0037 | 0.0027 | 0.7368 | 0.8042 |
| MILO_LW_scale_aware_False | 0.7798 | 0.0024 | 0.0017 | 0.7581 | 0.8015 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric               |
|:-------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:---------------------|
| MILO_LW_scale_aware_True | MILO_LW_scale_aware_False | 0.770518 | 0.779791 | MILO_LW_scale_aware_False |  0.118651 |               | final_validation_auc |

