# Statistical Analysis Report

## Statistical Methodology

This report documents the statistical methods used to analyze the experimental results.

### Error Calculation

- **Standard Deviation**: Calculated using the unbiased estimator (n-1 denominator)
- **Standard Error of the Mean (SEM)**: SD/√n where n is the number of runs
- **95% Confidence Intervals**: Using t-distribution (t-critical × SEM)
- **Error Bands in Plots**: Represent ±1 SEM around the mean

### Statistical Assumptions

- Errors are assumed to follow a t-distribution due to small sample sizes
- Runs are treated as independent samples with the same underlying distribution
- Variances between different optimizers may be unequal (Welch's t-test used)

### Sources of Variability

- Random initialization of model parameters
- Stochastic gradient descent optimization
- Batch sampling during training

## Statistical Results

### Loss Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_adaptive_True | 0.1482 | 0.0124 | 0.0072 | 0.1174 | 0.1790 |
| milo_LW_adaptive_False | 9.4888 | 0.5623 | 0.3246 | 8.0920 | 10.8855 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |    p-value | Significant   | Metric     |
|:---------------------|:----------------------|---------:|---------:|:---------------------|-----------:|:--------------|:-----------|
| milo_LW_adaptive_True | milo_LW_adaptive_False | 0.148247 |  9.48875 | milo_LW_adaptive_True | 0.00120001 | **            | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_adaptive_True | 94.8680 | 0.4436 | 0.2561 | 93.7660 | 95.9700 |
| milo_LW_adaptive_False | 10.0133 | 0.1640 | 0.0947 | 9.6060 | 10.4207 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |     p-value | Significant   | Metric         |
|:---------------------|:----------------------|---------:|---------:|:---------------------|------------:|:--------------|:---------------|
| milo_LW_adaptive_True | milo_LW_adaptive_False |   94.868 |  10.0133 | milo_LW_adaptive_True | 7.05505e-07 | ***           | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_adaptive_True | 0.9487 | 0.0044 | 0.0026 | 0.9376 | 0.9597 |
| milo_LW_adaptive_False | 0.1001 | 0.0017 | 0.0010 | 0.0958 | 0.1043 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |     p-value | Significant   | Metric         |
|:---------------------|:----------------------|---------:|---------:|:---------------------|------------:|:--------------|:---------------|
| milo_LW_adaptive_True | milo_LW_adaptive_False | 0.948687 | 0.100053 | milo_LW_adaptive_True | 5.93405e-07 | ***           | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

