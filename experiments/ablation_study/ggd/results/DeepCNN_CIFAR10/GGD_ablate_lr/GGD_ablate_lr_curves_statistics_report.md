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
| milo_baseline_lr | 0.5339 | 0.0762 | 0.0440 | 0.3447 | 0.7231 |
| milo_lr_0.1 | 2.3051 | 0.0015 | 0.0009 | 2.3014 | 2.3088 |
| milo_lr_0.001 | 2.3039 | 0.0000 | 0.0000 | 2.3039 | 2.3040 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A     | Optimizer B   |   Mean A |   Mean B | Better          |     p-value | Significant   | Metric     |
|:----------------|:--------------|---------:|---------:|:----------------|------------:|:--------------|:-----------|
| milo_baseline_lr | milo_lr_0.1    | 0.533887 |  2.30507 | milo_baseline_lr | 0.000613242 | ***           | final_loss |
| milo_baseline_lr | milo_lr_0.001  | 0.533887 |  2.30394 | milo_baseline_lr | 0.000616678 | ***           | final_loss |
| milo_lr_0.1      | milo_lr_0.001  | 2.30507  |  2.30394 | milo_lr_0.001    | 0.322177    |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_baseline_lr | 83.4193 | 3.3558 | 1.9375 | 75.0830 | 91.7556 |
| milo_lr_0.1 | 9.9047 | 0.1244 | 0.0718 | 9.5957 | 10.2136 |
| milo_lr_0.001 | 9.9273 | 0.0555 | 0.0320 | 9.7895 | 10.0651 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A     | Optimizer B   |   Mean A |   Mean B | Better          |     p-value | Significant   | Metric         |
|:----------------|:--------------|---------:|---------:|:----------------|------------:|:--------------|:---------------|
| milo_baseline_lr | milo_lr_0.1    | 83.4193  |  9.90467 | milo_baseline_lr | 0.000683524 | ***           | final_accuracy |
| milo_baseline_lr | milo_lr_0.001  | 83.4193  |  9.92733 | milo_baseline_lr | 0.000692218 | ***           | final_accuracy |
| milo_lr_0.1      | milo_lr_0.001  |  9.90467 |  9.92733 | milo_lr_0.001    | 0.793348    |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_baseline_lr | 0.8341 | 0.0335 | 0.0194 | 0.7509 | 0.9174 |
| milo_lr_0.1 | 0.0537 | 0.0004 | 0.0002 | 0.0526 | 0.0547 |
| milo_lr_0.001 | 0.0326 | 0.0002 | 0.0001 | 0.0321 | 0.0331 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A     | Optimizer B   |    Mean A |    Mean B | Better          |     p-value | Significant   | Metric         |
|:----------------|:--------------|----------:|----------:|:----------------|------------:|:--------------|:---------------|
| milo_baseline_lr | milo_lr_0.1    | 0.834136  | 0.0536932 | milo_baseline_lr | 0.000613447 | ***           | final_f1_score |
| milo_baseline_lr | milo_lr_0.001  | 0.834136  | 0.0326337 | milo_baseline_lr | 0.000582463 | ***           | final_f1_score |
| milo_lr_0.1      | milo_lr_0.001  | 0.0536932 | 0.0326337 | milo_lr_0.1      | 7.49308e-06 | ***           | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

