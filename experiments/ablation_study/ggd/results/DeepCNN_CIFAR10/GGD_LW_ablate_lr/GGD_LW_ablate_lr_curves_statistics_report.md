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
| milo_LW_lr_0.01 | 2.3027 | 0.0000 | 0.0000 | 2.3026 | 2.3027 |
| milo_LW_lr_0.1 | 2.3030 | 0.0001 | 0.0001 | 2.3028 | 2.3033 |
| milo_LW_lr_0.001 | 2.2042 | 0.0937 | 0.0541 | 1.9714 | 2.4370 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A    | Optimizer B     |   Mean A |   Mean B | Better          |   p-value | Significant   | Metric     |
|:---------------|:----------------|---------:|---------:|:----------------|----------:|:--------------|:-----------|
| milo_LW_lr_0.01 | milo_LW_lr_0.1   |  2.30265 |  2.30305 | milo_LW_lr_0.01  |  0.020474 | *             | final_loss |
| milo_LW_lr_0.01 | milo_LW_lr_0.001 |  2.30265 |  2.20422 | milo_LW_lr_0.001 |  0.210477 |               | final_loss |
| milo_LW_lr_0.1  | milo_LW_lr_0.001 |  2.30305 |  2.20422 | milo_LW_lr_0.001 |  0.209298 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_lr_0.01 | 9.7473 | 0.1084 | 0.0626 | 9.4781 | 10.0166 |
| milo_LW_lr_0.1 | 9.9900 | 0.1673 | 0.0966 | 9.5744 | 10.4056 |
| milo_LW_lr_0.001 | 14.7733 | 4.1644 | 2.4043 | 4.4283 | 25.1184 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A    | Optimizer B     |   Mean A |   Mean B | Better          |   p-value | Significant   | Metric         |
|:---------------|:----------------|---------:|---------:|:----------------|----------:|:--------------|:---------------|
| milo_LW_lr_0.01 | milo_LW_lr_0.1   |  9.74733 |   9.99   | milo_LW_lr_0.1   |  0.114182 |               | final_accuracy |
| milo_LW_lr_0.01 | milo_LW_lr_0.001 |  9.74733 |  14.7733 | milo_LW_lr_0.001 |  0.171656 |               | final_accuracy |
| milo_LW_lr_0.1  | milo_LW_lr_0.001 |  9.99    |  14.7733 | milo_LW_lr_0.001 |  0.184756 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_lr_0.01 | 0.0533 | 0.0087 | 0.0050 | 0.0316 | 0.0749 |
| milo_LW_lr_0.1 | 0.0847 | 0.0014 | 0.0008 | 0.0813 | 0.0881 |
| milo_LW_lr_0.001 | 0.0680 | 0.0446 | 0.0258 | -0.0428 | 0.1789 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A    | Optimizer B     |    Mean A |    Mean B | Better          |   p-value | Significant   | Metric         |
|:---------------|:----------------|----------:|----------:|:----------------|----------:|:--------------|:---------------|
| milo_LW_lr_0.01 | milo_LW_lr_0.1   | 0.0532784 | 0.084687  | milo_LW_lr_0.1   | 0.0224662 | *             | final_f1_score |
| milo_LW_lr_0.01 | milo_LW_lr_0.001 | 0.0532784 | 0.0680311 | milo_LW_lr_0.001 | 0.627201  |               | final_f1_score |
| milo_LW_lr_0.1  | milo_LW_lr_0.001 | 0.084687  | 0.0680311 | milo_LW_lr_0.1   | 0.584354  |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

