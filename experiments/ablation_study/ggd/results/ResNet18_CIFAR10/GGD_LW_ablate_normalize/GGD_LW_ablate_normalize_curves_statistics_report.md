# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_normalize_True | 0.0099 | 0.0026 | 0.0015 | 0.0033 | 0.0164 |
| milo_LW_normalize_False | 0.0977 | 0.0116 | 0.0067 | 0.0688 | 0.1266 |

#### Pairwise Significance Tests

| Optimizer A           | Optimizer B            |     Mean A |    Mean B | Better                |    p-value | Significant   | Metric                |
|:----------------------|:-----------------------|-----------:|----------:|:----------------------|-----------:|:--------------|:----------------------|
| milo_LW_normalize_True | milo_LW_normalize_False | 0.00986511 | 0.0976995 | milo_LW_normalize_True | 0.00414968 | **            | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_normalize_True | 99.6553 | 0.1091 | 0.0630 | 99.3844 | 99.9263 |
| milo_LW_normalize_False | 96.6173 | 0.4306 | 0.2486 | 95.5478 | 97.6869 |

#### Pairwise Significance Tests

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |    p-value | Significant   | Metric                    |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|-----------:|:--------------|:--------------------------|
| milo_LW_normalize_True | milo_LW_normalize_False |  99.6553 |  96.6173 | milo_LW_normalize_True | 0.00445917 | **            | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_normalize_True | 0.9966 | 0.0011 | 0.0006 | 0.9938 | 0.9993 |
| milo_LW_normalize_False | 0.9662 | 0.0043 | 0.0025 | 0.9555 | 0.9768 |

#### Pairwise Significance Tests

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |    p-value | Significant   | Metric                    |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|-----------:|:--------------|:--------------------------|
| milo_LW_normalize_True | milo_LW_normalize_False | 0.996554 |  0.96617 | milo_LW_normalize_True | 0.00440651 | **            | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_normalize_True | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_LW_normalize_False | 0.9995 | 0.0001 | 0.0000 | 0.9994 | 0.9996 |

#### Pairwise Significance Tests

| Optimizer A           | Optimizer B            |   Mean A |   Mean B | Better                |    p-value | Significant   | Metric               |
|:----------------------|:-----------------------|---------:|---------:|:----------------------|-----------:|:--------------|:---------------------|
| milo_LW_normalize_True | milo_LW_normalize_False | 0.999993 | 0.999499 | milo_LW_normalize_True | 0.00460795 | **            | final_validation_auc |

