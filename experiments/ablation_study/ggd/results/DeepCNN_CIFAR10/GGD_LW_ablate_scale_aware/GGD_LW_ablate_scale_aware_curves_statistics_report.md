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
| milo_LW_scale_aware_True | 3.8316 | 1.0852 | 0.6266 | 1.1357 | 6.5274 |
| milo_LW_scale_aware_False | 1.9415 | 0.5659 | 0.3267 | 0.5358 | 3.3473 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric     |
|:------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:-----------|
| milo_LW_scale_aware_True | milo_LW_scale_aware_False |  3.83157 |  1.94154 | milo_LW_scale_aware_False | 0.0750457 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_scale_aware_True | 18.6627 | 2.8307 | 1.6343 | 11.6308 | 25.6945 |
| milo_LW_scale_aware_False | 36.0360 | 13.8834 | 8.0156 | 1.5477 | 70.5243 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric         |
|:------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:---------------|
| milo_LW_scale_aware_True | milo_LW_scale_aware_False |  18.6627 |   36.036 | milo_LW_scale_aware_False |  0.157815 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_scale_aware_True | 0.1504 | 0.0288 | 0.0166 | 0.0789 | 0.2219 |
| milo_LW_scale_aware_False | 0.3296 | 0.1498 | 0.0865 | -0.0425 | 0.7018 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A             | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric         |
|:------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:---------------|
| milo_LW_scale_aware_True | milo_LW_scale_aware_False |  0.15041 | 0.329645 | milo_LW_scale_aware_False |  0.169979 |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

