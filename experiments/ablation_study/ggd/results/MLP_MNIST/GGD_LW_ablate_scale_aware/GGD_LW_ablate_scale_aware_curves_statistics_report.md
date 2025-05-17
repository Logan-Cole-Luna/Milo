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
| milo_LW_baseline_scale_aware | 0.4027 | 0.2391 | 0.1380 | -0.1912 | 0.9966 |
| milo_LW_scale_aware_False | 0.1367 | 0.0257 | 0.0149 | 0.0728 | 0.2006 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                 | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric     |
|:----------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:-----------|
| milo_LW_baseline_scale_aware | milo_LW_scale_aware_False | 0.402688 | 0.136727 | milo_LW_scale_aware_False |  0.192595 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_baseline_scale_aware | 88.4283 | 7.1534 | 4.1300 | 70.6584 | 106.1982 |
| milo_LW_scale_aware_False | 96.1006 | 0.7767 | 0.4484 | 94.1711 | 98.0301 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                 | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric         |
|:----------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:---------------|
| milo_LW_baseline_scale_aware | milo_LW_scale_aware_False |  88.4283 |  96.1006 | milo_LW_scale_aware_False |  0.203119 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_baseline_scale_aware | 0.8863 | 0.0669 | 0.0387 | 0.7200 | 1.0527 |
| milo_LW_scale_aware_False | 0.9607 | 0.0078 | 0.0045 | 0.9413 | 0.9802 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                 | Optimizer B              |   Mean A |   Mean B | Better                   |   p-value | Significant   | Metric         |
|:----------------------------|:-------------------------|---------:|---------:|:-------------------------|----------:|:--------------|:---------------|
| milo_LW_baseline_scale_aware | milo_LW_scale_aware_False | 0.886347 | 0.960717 | milo_LW_scale_aware_False |  0.192768 |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

