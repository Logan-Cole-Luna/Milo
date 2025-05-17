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
| milo_momentum_0.9 | 0.0073 | 0.0091 | 0.0052 | -0.0152 | 0.0298 |
| milo_momentum_0.0 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 0.0001 |
| milo_momentum_0.45 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A      | Optimizer B       |      Mean A |      Mean B | Better            |   p-value | Significant   | Metric     |
|:-----------------|:------------------|------------:|------------:|:------------------|----------:|:--------------|:-----------|
| milo_momentum_0.9 | milo_momentum_0.0  | 0.00728006  | 1.44801e-05 | milo_momentum_0.0  |  0.299178 |               | final_loss |
| milo_momentum_0.9 | milo_momentum_0.45 | 0.00728006  | 7.60105e-07 | milo_momentum_0.45 |  0.298506 |               | final_loss |
| milo_momentum_0.0 | milo_momentum_0.45 | 1.44801e-05 | 7.60105e-07 | milo_momentum_0.45 |  0.367621 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_momentum_0.9 | 99.8206 | 0.2457 | 0.1418 | 99.2102 | 100.4309 |
| milo_momentum_0.0 | 99.9994 | 0.0010 | 0.0006 | 99.9971 | 100.0018 |
| milo_momentum_0.45 | 100.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A      | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric         |
|:-----------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:---------------|
| milo_momentum_0.9 | milo_momentum_0.0  |  99.8206 |  99.9994 | milo_momentum_0.0  |  0.33444  |               | final_accuracy |
| milo_momentum_0.9 | milo_momentum_0.45 |  99.8206 | 100      | milo_momentum_0.45 |  0.333292 |               | final_accuracy |
| milo_momentum_0.0 | milo_momentum_0.45 |  99.9994 | 100      | milo_momentum_0.45 |  0.42265  |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_momentum_0.9 | 0.9982 | 0.0025 | 0.0014 | 0.9921 | 1.0043 |
| milo_momentum_0.0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_momentum_0.45 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A      | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric         |
|:-----------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:---------------|
| milo_momentum_0.9 | milo_momentum_0.0  | 0.998207 | 0.999995 | milo_momentum_0.0  |  0.336282 |               | final_f1_score |
| milo_momentum_0.9 | milo_momentum_0.45 | 0.998207 | 1        | milo_momentum_0.45 |  0.335218 |               | final_f1_score |
| milo_momentum_0.0 | milo_momentum_0.45 | 0.999995 | 1        | milo_momentum_0.45 |  0.42265  |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

