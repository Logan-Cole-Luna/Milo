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
| milo_LW_weight_decay_0.0001 | 0.4850 | 0.0625 | 0.0361 | 0.3298 | 0.6402 |
| milo_LW_weight_decay_0.001 | 0.3241 | 0.0494 | 0.0285 | 0.2013 | 0.4468 |
| milo_LW_weight_decay_0.0 | 0.2351 | 0.0131 | 0.0076 | 0.2026 | 0.2676 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric     |
|:---------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:-----------|
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.001 | 0.485024 | 0.324052 | milo_LW_weight_decay_0.001 | 0.0270575 | *             | final_loss |
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.0   | 0.485024 | 0.235106 | milo_LW_weight_decay_0.0   | 0.0168497 | *             | final_loss |
| milo_LW_weight_decay_0.001  | milo_LW_weight_decay_0.0   | 0.324052 | 0.235106 | milo_LW_weight_decay_0.0   | 0.0806373 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_weight_decay_0.0001 | 83.8353 | 2.1569 | 1.2453 | 78.4774 | 89.1933 |
| milo_LW_weight_decay_0.001 | 88.8613 | 1.4871 | 0.8586 | 85.1671 | 92.5556 |
| milo_LW_weight_decay_0.0 | 91.7347 | 0.4474 | 0.2583 | 90.6234 | 92.8460 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric         |
|:---------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:---------------|
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.001 |  83.8353 |  88.8613 | milo_LW_weight_decay_0.001 | 0.0350242 | *             | final_accuracy |
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.0   |  83.8353 |  91.7347 | milo_LW_weight_decay_0.0   | 0.0203455 | *             | final_accuracy |
| milo_LW_weight_decay_0.001  | milo_LW_weight_decay_0.0   |  88.8613 |  91.7347 | milo_LW_weight_decay_0.0   | 0.0683879 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_weight_decay_0.0001 | 0.8384 | 0.0216 | 0.0124 | 0.7849 | 0.8920 |
| milo_LW_weight_decay_0.001 | 0.8886 | 0.0149 | 0.0086 | 0.8517 | 0.9255 |
| milo_LW_weight_decay_0.0 | 0.9173 | 0.0045 | 0.0026 | 0.9062 | 0.9285 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A                | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric         |
|:---------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:---------------|
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.001 | 0.83843  | 0.888576 | milo_LW_weight_decay_0.001 | 0.0351739 | *             | final_f1_score |
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.0   | 0.83843  | 0.917345 | milo_LW_weight_decay_0.0   | 0.0203498 | *             | final_f1_score |
| milo_LW_weight_decay_0.001  | milo_LW_weight_decay_0.0   | 0.888576 | 0.917345 | milo_LW_weight_decay_0.0   | 0.0680476 |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

