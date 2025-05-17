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
| milo_LW_eps_1e-05 | 0.1969 | 0.0112 | 0.0065 | 0.1690 | 0.2247 |
| milo_LW_eps_0.005 | 0.1636 | 0.0063 | 0.0036 | 0.1479 | 0.1792 |
| milo_LW_eps_0.05 | 0.1421 | 0.0057 | 0.0033 | 0.1279 | 0.1564 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A      | Optimizer B      |   Mean A |   Mean B | Better           |    p-value | Significant   | Metric     |
|:-----------------|:-----------------|---------:|---------:|:-----------------|-----------:|:--------------|:-----------|
| milo_LW_eps_1e-05 | milo_LW_eps_0.005 | 0.19688  | 0.163567 | milo_LW_eps_0.005 | 0.0185954  | *             | final_loss |
| milo_LW_eps_1e-05 | milo_LW_eps_0.05  | 0.19688  | 0.142135 | milo_LW_eps_0.05  | 0.00495306 | **            | final_loss |
| milo_LW_eps_0.005 | milo_LW_eps_0.05  | 0.163567 | 0.142135 | milo_LW_eps_0.05  | 0.0123677  | *             | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_eps_1e-05 | 93.0980 | 0.3034 | 0.1752 | 92.3443 | 93.8517 |
| milo_LW_eps_0.005 | 94.1807 | 0.1989 | 0.1148 | 93.6866 | 94.6748 |
| milo_LW_eps_0.05 | 94.9133 | 0.1754 | 0.1013 | 94.4776 | 95.3490 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A      | Optimizer B      |   Mean A |   Mean B | Better           |    p-value | Significant   | Metric         |
|:-----------------|:-----------------|---------:|---------:|:-----------------|-----------:|:--------------|:---------------|
| milo_LW_eps_1e-05 | milo_LW_eps_0.005 |  93.098  |  94.1807 | milo_LW_eps_0.005 | 0.00985361 | **            | final_accuracy |
| milo_LW_eps_1e-05 | milo_LW_eps_0.05  |  93.098  |  94.9133 | milo_LW_eps_0.05  | 0.00224735 | **            | final_accuracy |
| milo_LW_eps_0.005 | milo_LW_eps_0.05  |  94.1807 |  94.9133 | milo_LW_eps_0.05  | 0.00908388 | **            | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_eps_1e-05 | 0.9310 | 0.0030 | 0.0018 | 0.9235 | 0.9385 |
| milo_LW_eps_0.005 | 0.9418 | 0.0020 | 0.0012 | 0.9368 | 0.9468 |
| milo_LW_eps_0.05 | 0.9491 | 0.0018 | 0.0010 | 0.9448 | 0.9535 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A      | Optimizer B      |   Mean A |   Mean B | Better           |    p-value | Significant   | Metric         |
|:-----------------|:-----------------|---------:|---------:|:-----------------|-----------:|:--------------|:---------------|
| milo_LW_eps_1e-05 | milo_LW_eps_0.005 | 0.930995 | 0.941803 | milo_LW_eps_0.005 | 0.00986137 | **            | final_f1_score |
| milo_LW_eps_1e-05 | milo_LW_eps_0.05  | 0.930995 | 0.949146 | milo_LW_eps_0.05  | 0.00225095 | **            | final_f1_score |
| milo_LW_eps_0.005 | milo_LW_eps_0.05  | 0.941803 | 0.949146 | milo_LW_eps_0.05  | 0.00916691 | **            | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

