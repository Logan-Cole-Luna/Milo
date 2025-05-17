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
| milo_LW_momentum_0.9 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 0.0000 |
| milo_LW_momentum_0.0 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 0.0000 |
| milo_LW_momentum_0.45 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A         | Optimizer B          |      Mean A |      Mean B | Better               |   p-value | Significant   | Metric     |
|:--------------------|:---------------------|------------:|------------:|:---------------------|----------:|:--------------|:-----------|
| milo_LW_momentum_0.9 | milo_LW_momentum_0.0  | 7.93143e-07 | 4.75159e-07 | milo_LW_momentum_0.0  |  0.303287 |               | final_loss |
| milo_LW_momentum_0.9 | milo_LW_momentum_0.45 | 7.93143e-07 | 2.89682e-07 | milo_LW_momentum_0.45 |  0.131823 |               | final_loss |
| milo_LW_momentum_0.0 | milo_LW_momentum_0.45 | 4.75159e-07 | 2.89682e-07 | milo_LW_momentum_0.45 |  0.400623 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_momentum_0.9 | 100.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 |
| milo_LW_momentum_0.0 | 100.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 |
| milo_LW_momentum_0.45 | 100.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric         |
|:--------------------|:---------------------|---------:|---------:|:---------------------|----------:|:--------------|:---------------|
| milo_LW_momentum_0.9 | milo_LW_momentum_0.0  |      100 |      100 | milo_LW_momentum_0.0  |       nan |               | final_accuracy |
| milo_LW_momentum_0.9 | milo_LW_momentum_0.45 |      100 |      100 | milo_LW_momentum_0.45 |       nan |               | final_accuracy |
| milo_LW_momentum_0.0 | milo_LW_momentum_0.45 |      100 |      100 | milo_LW_momentum_0.45 |       nan |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_momentum_0.9 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_LW_momentum_0.0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_LW_momentum_0.45 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric         |
|:--------------------|:---------------------|---------:|---------:|:---------------------|----------:|:--------------|:---------------|
| milo_LW_momentum_0.9 | milo_LW_momentum_0.0  |        1 |        1 | milo_LW_momentum_0.0  |       nan |               | final_f1_score |
| milo_LW_momentum_0.9 | milo_LW_momentum_0.45 |        1 |        1 | milo_LW_momentum_0.45 |       nan |               | final_f1_score |
| milo_LW_momentum_0.0 | milo_LW_momentum_0.45 |        1 |        1 | milo_LW_momentum_0.45 |       nan |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

