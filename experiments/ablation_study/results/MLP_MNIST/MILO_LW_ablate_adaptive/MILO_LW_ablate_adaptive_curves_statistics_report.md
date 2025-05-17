# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_adaptive_True | 0.0822 | 0.0093 | 0.0066 | -0.0018 | 0.1662 |
| MILO_LW_adaptive_False | 1.8881 | 0.1725 | 0.1220 | 0.3380 | 3.4382 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A           | Optimizer B            |    Mean A |   Mean B | Better                |   p-value | Significant   | Metric                |
|:----------------------|:-----------------------|----------:|---------:|:----------------------|----------:|:--------------|:----------------------|
| MILO_LW_adaptive_True | MILO_LW_adaptive_False | 0.0822256 |   1.8881 | MILO_LW_adaptive_True | 0.0423762 | *             | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_adaptive_True | 97.2408 | 0.3477 | 0.2458 | 94.1172 | 100.3644 |
| MILO_LW_adaptive_False | 38.1367 | 2.9298 | 2.0717 | 11.8136 | 64.4597 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric                    |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|----------:|:--------------|:--------------------------|
| MILO_LW_adaptive_True | MILO_LW_adaptive_False |  97.2408 |  38.1367 | MILO_LW_adaptive_True | 0.0205595 | *             | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_adaptive_True | 0.9730 | 0.0032 | 0.0022 | 0.9445 | 1.0015 |
| MILO_LW_adaptive_False | 0.3667 | 0.0290 | 0.0205 | 0.1059 | 0.6274 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric                    |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|----------:|:--------------|:--------------------------|
| MILO_LW_adaptive_True | MILO_LW_adaptive_False | 0.973006 | 0.366678 | MILO_LW_adaptive_True | 0.0200856 | *             | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_adaptive_True | 0.9994 | 0.0002 | 0.0001 | 0.9980 | 1.0008 |
| MILO_LW_adaptive_False | 0.7430 | 0.0220 | 0.0155 | 0.5455 | 0.9404 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric               |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|----------:|:--------------|:---------------------|
| MILO_LW_adaptive_True | MILO_LW_adaptive_False | 0.999401 | 0.742958 | MILO_LW_adaptive_True | 0.0385172 | *             | final_validation_auc |

