# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_normalize_True | 0.0969 | 0.0020 | 0.0014 | 0.0787 | 0.1151 |
| MILO_LW_normalize_False | 0.1036 | 0.0039 | 0.0028 | 0.0685 | 0.1386 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A            | Optimizer B             |    Mean A |   Mean B | Better                 |   p-value | Significant   | Metric                |
|:-----------------------|:------------------------|----------:|---------:|:-----------------------|----------:|:--------------|:----------------------|
| MILO_LW_normalize_True | MILO_LW_normalize_False | 0.0969152 | 0.103563 | MILO_LW_normalize_True |   0.20625 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_normalize_True | 96.6458 | 0.0625 | 0.0442 | 96.0846 | 97.2070 |
| MILO_LW_normalize_False | 96.5575 | 0.0436 | 0.0308 | 96.1657 | 96.9493 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A            | Optimizer B             |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric                    |
|:-----------------------|:------------------------|---------:|---------:|:-----------------------|----------:|:--------------|:--------------------------|
| MILO_LW_normalize_True | MILO_LW_normalize_False |  96.6458 |  96.5575 | MILO_LW_normalize_True |  0.257055 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_normalize_True | 0.9678 | 0.0005 | 0.0004 | 0.9629 | 0.9727 |
| MILO_LW_normalize_False | 0.9666 | 0.0006 | 0.0004 | 0.9611 | 0.9722 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A            | Optimizer B             |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric                    |
|:-----------------------|:------------------------|---------:|---------:|:-----------------------|----------:|:--------------|:--------------------------|
| MILO_LW_normalize_True | MILO_LW_normalize_False | 0.967793 | 0.966643 | MILO_LW_normalize_True |  0.189379 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_normalize_True | 0.9991 | 0.0000 | 0.0000 | 0.9987 | 0.9994 |
| MILO_LW_normalize_False | 0.9990 | 0.0000 | 0.0000 | 0.9987 | 0.9993 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A            | Optimizer B             |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric               |
|:-----------------------|:------------------------|---------:|---------:|:-----------------------|----------:|:--------------|:---------------------|
| MILO_LW_normalize_True | MILO_LW_normalize_False | 0.999087 | 0.999009 | MILO_LW_normalize_True |  0.167984 |               | final_validation_auc |

