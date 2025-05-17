# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_adaptive_True | 0.0682 | 0.0208 | 0.0147 | -0.1189 | 0.2554 |
| MILO_adaptive_False | 94.9110 | 73.2626 | 51.8045 | -563.3275 | 753.1494 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric                |
|:-------------------|:--------------------|---------:|---------:|:-------------------|----------:|:--------------|:----------------------|
| MILO_adaptive_True | MILO_adaptive_False | 0.068231 |   94.911 | MILO_adaptive_True |  0.318267 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_adaptive_True | 97.5910 | 0.7283 | 0.5150 | 91.0473 | 104.1347 |
| MILO_adaptive_False | 71.0940 | 0.8598 | 0.6080 | 63.3686 | 78.8194 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better             |    p-value | Significant   | Metric                    |
|:-------------------|:--------------------|---------:|---------:|:-------------------|-----------:|:--------------|:--------------------------|
| MILO_adaptive_True | MILO_adaptive_False |   97.591 |   71.094 | MILO_adaptive_True | 0.00104999 | **            | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_adaptive_True | 0.9760 | 0.0073 | 0.0051 | 0.9107 | 1.0413 |
| MILO_adaptive_False | 0.7122 | 0.0082 | 0.0058 | 0.6385 | 0.7858 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better             |    p-value | Significant   | Metric                    |
|:-------------------|:--------------------|---------:|---------:|:-------------------|-----------:|:--------------|:--------------------------|
| MILO_adaptive_True | MILO_adaptive_False | 0.975968 | 0.712153 | MILO_adaptive_True | 0.00093356 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_adaptive_True | 0.9996 | 0.0002 | 0.0001 | 0.9980 | 1.0013 |
| MILO_adaptive_False | 0.8797 | 0.0250 | 0.0177 | 0.6549 | 1.1044 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric               |
|:-------------------|:--------------------|---------:|---------:|:-------------------|----------:|:--------------|:---------------------|
| MILO_adaptive_True | MILO_adaptive_False | 0.999641 | 0.879672 | MILO_adaptive_True | 0.0931675 |               | final_validation_auc |

