# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_adaptive_True | 0.4978 | 0.1057 | 0.0748 | -0.4520 | 1.4476 |
| MILO_LW_adaptive_False | 227.4893 | 122.1627 | 86.3821 | -870.0992 | 1325.0778 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric                |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|----------:|:--------------|:----------------------|
| MILO_LW_adaptive_True | MILO_LW_adaptive_False | 0.497807 |  227.489 | MILO_LW_adaptive_True |  0.231493 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_adaptive_True | 94.8520 | 0.3705 | 0.2620 | 91.5230 | 98.1810 |
| MILO_LW_adaptive_False | 71.8920 | 5.3797 | 3.8040 | 23.5576 | 120.2264 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric                    |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|----------:|:--------------|:--------------------------|
| MILO_LW_adaptive_True | MILO_LW_adaptive_False |   94.852 |   71.892 | MILO_LW_adaptive_True |  0.103172 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_adaptive_True | 0.9484 | 0.0037 | 0.0026 | 0.9153 | 0.9816 |
| MILO_LW_adaptive_False | 0.7230 | 0.0592 | 0.0419 | 0.1907 | 1.2554 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric                    |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|----------:|:--------------|:--------------------------|
| MILO_LW_adaptive_True | MILO_LW_adaptive_False | 0.948421 | 0.723044 | MILO_LW_adaptive_True |  0.115858 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_adaptive_True | 0.9980 | 0.0003 | 0.0002 | 0.9949 | 1.0011 |
| MILO_LW_adaptive_False | 0.8567 | 0.0315 | 0.0222 | 0.5741 | 1.1393 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric               |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|----------:|:--------------|:---------------------|
| MILO_LW_adaptive_True | MILO_LW_adaptive_False | 0.997989 | 0.856683 | MILO_LW_adaptive_True | 0.0993461 |               | final_validation_auc |

