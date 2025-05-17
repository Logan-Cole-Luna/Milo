# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_adaptive_True | 0.0003 | 0.0004 | 0.0003 | -0.0036 | 0.0043 |
| MILO_adaptive_False | 0.4414 | 0.1454 | 0.1028 | -0.8653 | 1.7481 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A        | Optimizer B         |      Mean A |   Mean B | Better             |   p-value | Significant   | Metric                |
|:-------------------|:--------------------|------------:|---------:|:-------------------|----------:|:--------------|:----------------------|
| MILO_adaptive_True | MILO_adaptive_False | 0.000341147 | 0.441383 | MILO_adaptive_True |  0.145832 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_adaptive_True | 99.9900 | 0.0141 | 0.0100 | 99.8629 | 100.1171 |
| MILO_adaptive_False | 95.0517 | 1.2044 | 0.8517 | 84.2302 | 105.8731 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric                    |
|:-------------------|:--------------------|---------:|---------:|:-------------------|----------:|:--------------|:--------------------------|
| MILO_adaptive_True | MILO_adaptive_False |    99.99 |  95.0517 | MILO_adaptive_True |  0.108682 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_adaptive_True | 0.9999 | 0.0001 | 0.0001 | 0.9986 | 1.0012 |
| MILO_adaptive_False | 0.9526 | 0.0114 | 0.0080 | 0.8505 | 1.0546 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric                    |
|:-------------------|:--------------------|---------:|---------:|:-------------------|----------:|:--------------|:--------------------------|
| MILO_adaptive_True | MILO_adaptive_False |   0.9999 | 0.952574 | MILO_adaptive_True |  0.106948 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_adaptive_True | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| MILO_adaptive_False | 0.9972 | 0.0011 | 0.0008 | 0.9873 | 1.0071 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric               |
|:-------------------|:--------------------|---------:|---------:|:-------------------|----------:|:--------------|:---------------------|
| MILO_adaptive_True | MILO_adaptive_False |        1 | 0.997244 | MILO_adaptive_True |  0.175368 |               | final_validation_auc |

