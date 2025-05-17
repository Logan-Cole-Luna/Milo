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
| milo_momentum_0.9 | 0.5191 | 0.0920 | 0.0531 | 0.2906 | 0.7475 |
| milo_momentum_0.0 | 0.3516 | 0.0129 | 0.0074 | 0.3196 | 0.3837 |
| milo_momentum_0.45 | 0.3218 | 0.0099 | 0.0057 | 0.2973 | 0.3464 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A      | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric     |
|:-----------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:-----------|
| milo_momentum_0.9 | milo_momentum_0.0  | 0.51906  | 0.351614 | milo_momentum_0.0  | 0.0847656 |               | final_loss |
| milo_momentum_0.9 | milo_momentum_0.45 | 0.51906  | 0.321814 | milo_momentum_0.45 | 0.0638261 |               | final_loss |
| milo_momentum_0.0 | milo_momentum_0.45 | 0.351614 | 0.321814 | milo_momentum_0.45 | 0.0369114 | *             | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_momentum_0.9 | 83.6440 | 3.4479 | 1.9906 | 75.0790 | 92.2090 |
| milo_momentum_0.0 | 89.0133 | 0.5494 | 0.3172 | 87.6485 | 90.3781 |
| milo_momentum_0.45 | 90.0093 | 0.4521 | 0.2610 | 88.8864 | 91.1323 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A      | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric         |
|:-----------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:---------------|
| milo_momentum_0.9 | milo_momentum_0.0  |  83.644  |  89.0133 | milo_momentum_0.0  | 0.110895  |               | final_accuracy |
| milo_momentum_0.9 | milo_momentum_0.45 |  83.644  |  90.0093 | milo_momentum_0.45 | 0.0830315 |               | final_accuracy |
| milo_momentum_0.0 | milo_momentum_0.45 |  89.0133 |  90.0093 | milo_momentum_0.45 | 0.0747604 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_momentum_0.9 | 0.8364 | 0.0346 | 0.0199 | 0.7505 | 0.9222 |
| milo_momentum_0.0 | 0.8901 | 0.0055 | 0.0032 | 0.8765 | 0.9037 |
| milo_momentum_0.45 | 0.9001 | 0.0045 | 0.0026 | 0.8889 | 0.9113 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A      | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric         |
|:-----------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:---------------|
| milo_momentum_0.9 | milo_momentum_0.0  |  0.83636 |  0.89009 | milo_momentum_0.0  | 0.111209  |               | final_f1_score |
| milo_momentum_0.9 | milo_momentum_0.45 |  0.83636 |  0.90008 | milo_momentum_0.45 | 0.0832019 |               | final_f1_score |
| milo_momentum_0.0 | milo_momentum_0.45 |  0.89009 |  0.90008 | milo_momentum_0.45 | 0.0734185 |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

