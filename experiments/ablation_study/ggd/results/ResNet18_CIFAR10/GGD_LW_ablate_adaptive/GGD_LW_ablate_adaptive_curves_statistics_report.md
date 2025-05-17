# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_adaptive_True | 0.2393 | 0.1588 | 0.0917 | -0.1551 | 0.6337 |
| milo_LW_adaptive_False | 0.5920 | 0.1946 | 0.1123 | 0.1087 | 1.0753 |

#### Pairwise Significance Tests

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                |
|:---------------------|:----------------------|---------:|---------:|:---------------------|----------:|:--------------|:----------------------|
| milo_LW_adaptive_True | milo_LW_adaptive_False | 0.239267 | 0.591987 | milo_LW_adaptive_True | 0.0743225 |               | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_adaptive_True | 91.5513 | 5.5307 | 3.1931 | 77.8124 | 105.2902 |
| milo_LW_adaptive_False | 82.6387 | 7.2456 | 4.1833 | 64.6395 | 100.6378 |

#### Pairwise Significance Tests

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                    |
|:---------------------|:----------------------|---------:|---------:|:---------------------|----------:|:--------------|:--------------------------|
| milo_LW_adaptive_True | milo_LW_adaptive_False |  91.5513 |  82.6387 | milo_LW_adaptive_True |  0.170551 |               | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_adaptive_True | 0.9156 | 0.0553 | 0.0319 | 0.7782 | 1.0531 |
| milo_LW_adaptive_False | 0.8272 | 0.0712 | 0.0411 | 0.6503 | 1.0041 |

#### Pairwise Significance Tests

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                    |
|:---------------------|:----------------------|---------:|---------:|:---------------------|----------:|:--------------|:--------------------------|
| milo_LW_adaptive_True | milo_LW_adaptive_False | 0.915623 | 0.827188 | milo_LW_adaptive_True |  0.168957 |               | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_adaptive_True | 0.9950 | 0.0056 | 0.0032 | 0.9812 | 1.0088 |
| milo_LW_adaptive_False | 0.9834 | 0.0092 | 0.0053 | 0.9604 | 1.0064 |

#### Pairwise Significance Tests

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric               |
|:---------------------|:----------------------|---------:|---------:|:---------------------|----------:|:--------------|:---------------------|
| milo_LW_adaptive_True | milo_LW_adaptive_False | 0.995031 | 0.983404 | milo_LW_adaptive_True |   0.15077 |               | final_validation_auc |

