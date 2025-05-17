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
| milo_baseline_lr | 0.0067 | 0.0089 | 0.0052 | -0.0155 | 0.0289 |
| milo_lr_0.1 | 0.0024 | 0.0011 | 0.0006 | -0.0003 | 0.0051 |
| milo_lr_0.001 | 0.0011 | 0.0002 | 0.0001 | 0.0006 | 0.0017 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A     | Optimizer B   |     Mean A |     Mean B | Better       |   p-value | Significant   | Metric     |
|:----------------|:--------------|-----------:|-----------:|:-------------|----------:|:--------------|:-----------|
| milo_baseline_lr | milo_lr_0.1    | 0.00673344 | 0.00237905 | milo_lr_0.1   |  0.487999 |               | final_loss |
| milo_baseline_lr | milo_lr_0.001  | 0.00673344 | 0.00113806 | milo_lr_0.001 |  0.391359 |               | final_loss |
| milo_lr_0.1      | milo_lr_0.001  | 0.00237905 | 0.00113806 | milo_lr_0.001 |  0.181136 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_baseline_lr | 99.8356 | 0.2313 | 0.1336 | 99.2609 | 100.4102 |
| milo_lr_0.1 | 99.9544 | 0.0199 | 0.0115 | 99.9051 | 100.0038 |
| milo_lr_0.001 | 99.9844 | 0.0010 | 0.0006 | 99.9821 | 99.9868 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A     | Optimizer B   |   Mean A |   Mean B | Better       |   p-value | Significant   | Metric         |
|:----------------|:--------------|---------:|---------:|:-------------|----------:|:--------------|:---------------|
| milo_baseline_lr | milo_lr_0.1    |  99.8356 |  99.9544 | milo_lr_0.1   |  0.467535 |               | final_accuracy |
| milo_baseline_lr | milo_lr_0.001  |  99.8356 |  99.9844 | milo_lr_0.001 |  0.380942 |               | final_accuracy |
| milo_lr_0.1      | milo_lr_0.001  |  99.9544 |  99.9844 | milo_lr_0.001 |  0.120189 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_baseline_lr | 0.9984 | 0.0023 | 0.0013 | 0.9926 | 1.0041 |
| milo_lr_0.1 | 0.9995 | 0.0002 | 0.0001 | 0.9990 | 1.0000 |
| milo_lr_0.001 | 0.9998 | 0.0000 | 0.0000 | 0.9998 | 0.9999 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A     | Optimizer B   |   Mean A |   Mean B | Better       |   p-value | Significant   | Metric         |
|:----------------|:--------------|---------:|---------:|:-------------|----------:|:--------------|:---------------|
| milo_baseline_lr | milo_lr_0.1    | 0.998359 | 0.999543 | milo_lr_0.1   |  0.469889 |               | final_f1_score |
| milo_baseline_lr | milo_lr_0.001  | 0.998359 | 0.999843 | milo_lr_0.001 |  0.383031 |               | final_f1_score |
| milo_lr_0.1      | milo_lr_0.001  | 0.999543 | 0.999843 | milo_lr_0.001 |  0.125047 |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

