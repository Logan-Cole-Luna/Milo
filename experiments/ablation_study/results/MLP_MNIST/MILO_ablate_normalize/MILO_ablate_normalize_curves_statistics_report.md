# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_normalize_True | 0.0000 | 0.0001 | 0.0000 | -0.0004 | 0.0005 |
| MILO_normalize_False | 0.0017 | 0.0011 | 0.0008 | -0.0081 | 0.0115 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A         | Optimizer B          |      Mean A |     Mean B | Better              |   p-value | Significant   | Metric                |
|:--------------------|:---------------------|------------:|-----------:|:--------------------|----------:|:--------------|:----------------------|
| MILO_normalize_True | MILO_normalize_False | 4.31775e-05 | 0.00167463 | MILO_normalize_True |  0.281039 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_normalize_True | 99.9983 | 0.0024 | 0.0017 | 99.9772 | 100.0195 |
| MILO_normalize_False | 99.9967 | 0.0024 | 0.0017 | 99.9755 | 100.0178 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better              |   p-value | Significant   | Metric                    |
|:--------------------|:---------------------|---------:|---------:|:--------------------|----------:|:--------------|:--------------------------|
| MILO_normalize_True | MILO_normalize_False |  99.9983 |  99.9967 | MILO_normalize_True |  0.552786 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_normalize_True | 1.0000 | 0.0000 | 0.0000 | 0.9998 | 1.0002 |
| MILO_normalize_False | 1.0000 | 0.0000 | 0.0000 | 0.9998 | 1.0002 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better              |   p-value | Significant   | Metric                    |
|:--------------------|:---------------------|---------:|---------:|:--------------------|----------:|:--------------|:--------------------------|
| MILO_normalize_True | MILO_normalize_False | 0.999983 | 0.999967 | MILO_normalize_True |  0.553181 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_normalize_True | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| MILO_normalize_False | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric               |
|:--------------------|:---------------------|---------:|---------:|:---------------------|----------:|:--------------|:---------------------|
| MILO_normalize_True | MILO_normalize_False |        1 |        1 | MILO_normalize_False |       nan |               | final_validation_auc |

