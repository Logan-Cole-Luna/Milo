# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_normalize_True | 0.0252 | 0.0020 | 0.0014 | 0.0072 | 0.0432 |
| MILO_normalize_False | 0.1359 | 0.0630 | 0.0446 | -0.4305 | 0.7024 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A         | Optimizer B          |    Mean A |   Mean B | Better              |   p-value | Significant   | Metric                |
|:--------------------|:---------------------|----------:|---------:|:--------------------|----------:|:--------------|:----------------------|
| MILO_normalize_True | MILO_normalize_False | 0.0251539 | 0.135944 | MILO_normalize_True |  0.243257 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_normalize_True | 99.1030 | 0.1089 | 0.0770 | 98.1246 | 100.0814 |
| MILO_normalize_False | 95.4000 | 2.2373 | 1.5820 | 75.2988 | 115.5012 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better              |   p-value | Significant   | Metric                    |
|:--------------------|:---------------------|---------:|---------:|:--------------------|----------:|:--------------|:--------------------------|
| MILO_normalize_True | MILO_normalize_False |   99.103 |     95.4 | MILO_normalize_True |  0.256367 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_normalize_True | 0.9910 | 0.0011 | 0.0008 | 0.9812 | 1.0008 |
| MILO_normalize_False | 0.9541 | 0.0222 | 0.0157 | 0.7546 | 1.1537 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better              |   p-value | Significant   | Metric                    |
|:--------------------|:---------------------|---------:|---------:|:--------------------|----------:|:--------------|:--------------------------|
| MILO_normalize_True | MILO_normalize_False | 0.991033 | 0.954149 | MILO_normalize_True |  0.255571 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_normalize_True | 1.0000 | 0.0000 | 0.0000 | 0.9998 | 1.0001 |
| MILO_normalize_False | 0.9990 | 0.0007 | 0.0005 | 0.9923 | 1.0057 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better              |   p-value | Significant   | Metric               |
|:--------------------|:---------------------|---------:|---------:|:--------------------|----------:|:--------------|:---------------------|
| MILO_normalize_True | MILO_normalize_False | 0.999951 | 0.999001 | MILO_normalize_True |  0.321432 |               | final_validation_auc |

