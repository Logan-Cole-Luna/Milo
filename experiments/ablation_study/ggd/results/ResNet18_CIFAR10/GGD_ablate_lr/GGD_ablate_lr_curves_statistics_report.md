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
| milo_lr_0.01 | 0.1939 | 0.1232 | 0.0711 | -0.1122 | 0.4999 |
| milo_lr_0.1 | 0.3192 | 0.1305 | 0.0753 | -0.0049 | 0.6434 |
| milo_lr_0.001 | 0.1214 | 0.0068 | 0.0039 | 0.1045 | 0.1383 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better       |   p-value | Significant   | Metric     |
|:--------------|:--------------|---------:|---------:|:-------------|----------:|:--------------|:-----------|
| milo_lr_0.01   | milo_lr_0.1    | 0.193861 | 0.319225 | milo_lr_0.01  |  0.293108 |               | final_loss |
| milo_lr_0.01   | milo_lr_0.001  | 0.193861 | 0.121415 | milo_lr_0.001 |  0.415623 |               | final_loss |
| milo_lr_0.1    | milo_lr_0.001  | 0.319225 | 0.121415 | milo_lr_0.001 |  0.119193 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_lr_0.01 | 93.2120 | 4.2284 | 2.4413 | 82.7081 | 103.7159 |
| milo_lr_0.1 | 89.0927 | 4.5121 | 2.6051 | 77.8839 | 100.3014 |
| milo_lr_0.001 | 95.8327 | 0.3236 | 0.1868 | 95.0288 | 96.6366 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better       |   p-value | Significant   | Metric         |
|:--------------|:--------------|---------:|---------:|:-------------|----------:|:--------------|:---------------|
| milo_lr_0.01   | milo_lr_0.1    |  93.212  |  89.0927 | milo_lr_0.01  |  0.313074 |               | final_accuracy |
| milo_lr_0.01   | milo_lr_0.001  |  93.212  |  95.8327 | milo_lr_0.001 |  0.395399 |               | final_accuracy |
| milo_lr_0.1    | milo_lr_0.001  |  89.0927 |  95.8327 | milo_lr_0.001 |  0.121799 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_lr_0.01 | 0.9321 | 0.0423 | 0.0244 | 0.8269 | 1.0372 |
| milo_lr_0.1 | 0.8908 | 0.0452 | 0.0261 | 0.7786 | 1.0031 |
| milo_lr_0.001 | 0.9583 | 0.0032 | 0.0019 | 0.9503 | 0.9663 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better       |   p-value | Significant   | Metric         |
|:--------------|:--------------|---------:|---------:|:-------------|----------:|:--------------|:---------------|
| milo_lr_0.01   | milo_lr_0.1    | 0.932075 | 0.890812 | milo_lr_0.01  |  0.312932 |               | final_f1_score |
| milo_lr_0.01   | milo_lr_0.001  | 0.932075 | 0.958303 | milo_lr_0.001 |  0.395525 |               | final_f1_score |
| milo_lr_0.1    | milo_lr_0.001  | 0.890812 | 0.958303 | milo_lr_0.001 |  0.121833 |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

