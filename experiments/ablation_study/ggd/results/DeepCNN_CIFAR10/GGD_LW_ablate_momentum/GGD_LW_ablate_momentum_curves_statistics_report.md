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
| milo_LW_momentum_0.9 | 1.9579 | 0.5972 | 0.3448 | 0.4743 | 3.4414 |
| milo_LW_momentum_0.0 | 0.9014 | 0.0909 | 0.0525 | 0.6757 | 1.1272 |
| milo_LW_momentum_0.45 | 0.6656 | 0.0652 | 0.0376 | 0.5036 | 0.8276 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric     |
|:--------------------|:---------------------|---------:|---------:|:---------------------|----------:|:--------------|:-----------|
| milo_LW_momentum_0.9 | milo_LW_momentum_0.0  | 1.95785  | 0.901419 | milo_LW_momentum_0.0  | 0.0888313 |               | final_loss |
| milo_LW_momentum_0.9 | milo_LW_momentum_0.45 | 1.95785  | 0.665607 | milo_LW_momentum_0.45 | 0.062764  |               | final_loss |
| milo_LW_momentum_0.0 | milo_LW_momentum_0.45 | 0.901419 | 0.665607 | milo_LW_momentum_0.45 | 0.0257161 | *             | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_momentum_0.9 | 24.5340 | 25.6604 | 14.8150 | -39.2099 | 88.2779 |
| milo_LW_momentum_0.0 | 68.4353 | 3.5566 | 2.0534 | 59.6001 | 77.2705 |
| milo_LW_momentum_0.45 | 77.3973 | 2.3953 | 1.3829 | 71.4471 | 83.3475 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric         |
|:--------------------|:---------------------|---------:|---------:|:---------------------|----------:|:--------------|:---------------|
| milo_LW_momentum_0.9 | milo_LW_momentum_0.0  |  24.534  |  68.4353 | milo_LW_momentum_0.0  | 0.0948097 |               | final_accuracy |
| milo_LW_momentum_0.9 | milo_LW_momentum_0.45 |  24.534  |  77.3973 | milo_LW_momentum_0.45 | 0.0691306 |               | final_accuracy |
| milo_LW_momentum_0.0 | milo_LW_momentum_0.45 |  68.4353 |  77.3973 | milo_LW_momentum_0.45 | 0.0279821 | *             | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_momentum_0.9 | 0.2309 | 0.2649 | 0.1529 | -0.4271 | 0.8889 |
| milo_LW_momentum_0.0 | 0.6837 | 0.0361 | 0.0208 | 0.5941 | 0.7732 |
| milo_LW_momentum_0.45 | 0.7742 | 0.0238 | 0.0137 | 0.7151 | 0.8334 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric         |
|:--------------------|:---------------------|---------:|---------:|:---------------------|----------:|:--------------|:---------------|
| milo_LW_momentum_0.9 | milo_LW_momentum_0.0  | 0.230866 | 0.683665 | milo_LW_momentum_0.0  | 0.0950375 |               | final_f1_score |
| milo_LW_momentum_0.9 | milo_LW_momentum_0.45 | 0.230866 | 0.774239 | milo_LW_momentum_0.45 | 0.0697516 |               | final_f1_score |
| milo_LW_momentum_0.0 | milo_LW_momentum_0.45 | 0.683665 | 0.774239 | milo_LW_momentum_0.45 | 0.0282613 | *             | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

