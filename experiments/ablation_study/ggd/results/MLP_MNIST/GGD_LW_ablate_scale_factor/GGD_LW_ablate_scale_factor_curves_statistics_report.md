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
| milo_LW_baseline_scale_factor | 0.0834 | 0.0129 | 0.0074 | 0.0513 | 0.1154 |
| milo_LW_scale_factor_0.1 | 0.0561 | 0.0065 | 0.0038 | 0.0399 | 0.0724 |
| milo_LW_scale_factor_0.5 | 0.0394 | 0.0044 | 0.0025 | 0.0284 | 0.0503 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                  | Optimizer B             |    Mean A |    Mean B | Better                  |   p-value | Significant   | Metric     |
|:-----------------------------|:------------------------|----------:|----------:|:------------------------|----------:|:--------------|:-----------|
| milo_LW_baseline_scale_factor | milo_LW_scale_factor_0.1 | 0.0833519 | 0.0561375 | milo_LW_scale_factor_0.1 | 0.0479609 | *             | final_loss |
| milo_LW_baseline_scale_factor | milo_LW_scale_factor_0.5 | 0.0833519 | 0.03937   | milo_LW_scale_factor_0.5 | 0.0187763 | *             | final_loss |
| milo_LW_scale_factor_0.1      | milo_LW_scale_factor_0.5 | 0.0561375 | 0.03937   | milo_LW_scale_factor_0.5 | 0.0266536 | *             | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_baseline_scale_factor | 97.6406 | 0.3962 | 0.2287 | 96.6564 | 98.6247 |
| milo_LW_scale_factor_0.1 | 98.4772 | 0.1985 | 0.1146 | 97.9842 | 98.9703 |
| milo_LW_scale_factor_0.5 | 98.9711 | 0.1379 | 0.0796 | 98.6286 | 99.3136 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                  | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric         |
|:-----------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:---------------|
| milo_LW_baseline_scale_factor | milo_LW_scale_factor_0.1 |  97.6406 |  98.4772 | milo_LW_scale_factor_0.1 | 0.0480384 | *             | final_accuracy |
| milo_LW_baseline_scale_factor | milo_LW_scale_factor_0.5 |  97.6406 |  98.9711 | milo_LW_scale_factor_0.5 | 0.0192382 | *             | final_accuracy |
| milo_LW_scale_factor_0.1      | milo_LW_scale_factor_0.5 |  98.4772 |  98.9711 | milo_LW_scale_factor_0.5 | 0.0290223 | *             | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_baseline_scale_factor | 0.9763 | 0.0040 | 0.0023 | 0.9663 | 0.9862 |
| milo_LW_scale_factor_0.1 | 0.9847 | 0.0020 | 0.0012 | 0.9797 | 0.9897 |
| milo_LW_scale_factor_0.5 | 0.9897 | 0.0014 | 0.0008 | 0.9862 | 0.9931 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                  | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric         |
|:-----------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:---------------|
| milo_LW_baseline_scale_factor | milo_LW_scale_factor_0.1 | 0.97627  | 0.984699 | milo_LW_scale_factor_0.1 | 0.0482407 | *             | final_f1_score |
| milo_LW_baseline_scale_factor | milo_LW_scale_factor_0.5 | 0.97627  | 0.989682 | milo_LW_scale_factor_0.5 | 0.0193607 | *             | final_f1_score |
| milo_LW_scale_factor_0.1      | milo_LW_scale_factor_0.5 | 0.984699 | 0.989682 | milo_LW_scale_factor_0.5 | 0.0289442 | *             | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

