# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_normalize_True | 0.4036 | 0.0560 | 0.0396 | -0.0999 | 0.9071 |
| MILO_LW_normalize_False | 1.2754 | 0.7008 | 0.4956 | -5.0214 | 7.5721 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A            | Optimizer B             |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric                |
|:-----------------------|:------------------------|---------:|---------:|:-----------------------|----------:|:--------------|:----------------------|
| MILO_LW_normalize_True | MILO_LW_normalize_False | 0.403607 |  1.27536 | MILO_LW_normalize_True |   0.32759 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_normalize_True | 95.6930 | 0.7170 | 0.5070 | 89.2510 | 102.1350 |
| MILO_LW_normalize_False | 91.7570 | 3.7576 | 2.6570 | 57.9966 | 125.5174 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A            | Optimizer B             |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric                    |
|:-----------------------|:------------------------|---------:|---------:|:-----------------------|----------:|:--------------|:--------------------------|
| MILO_LW_normalize_True | MILO_LW_normalize_False |   95.693 |   91.757 | MILO_LW_normalize_True |  0.371459 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_normalize_True | 0.9573 | 0.0068 | 0.0048 | 0.8961 | 1.0184 |
| MILO_LW_normalize_False | 0.9145 | 0.0421 | 0.0297 | 0.5366 | 1.2925 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A            | Optimizer B             |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric                    |
|:-----------------------|:------------------------|---------:|---------:|:-----------------------|----------:|:--------------|:--------------------------|
| MILO_LW_normalize_True | MILO_LW_normalize_False | 0.957253 |  0.91453 | MILO_LW_normalize_True |   0.38253 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_normalize_True | 0.9986 | 0.0001 | 0.0001 | 0.9978 | 0.9995 |
| MILO_LW_normalize_False | 0.9965 | 0.0015 | 0.0011 | 0.9831 | 1.0099 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A            | Optimizer B             |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric               |
|:-----------------------|:------------------------|---------:|---------:|:-----------------------|----------:|:--------------|:---------------------|
| MILO_LW_normalize_True | MILO_LW_normalize_False | 0.998646 | 0.996505 | MILO_LW_normalize_True |  0.290015 |               | final_validation_auc |

