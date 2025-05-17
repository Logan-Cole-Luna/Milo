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
| milo_LW_baseline_lr | 0.0936 | 0.0247 | 0.0143 | 0.0322 | 0.1550 |
| milo_LW_lr_0.1 | 0.1030 | 0.0579 | 0.0334 | -0.0408 | 0.2469 |
| milo_LW_lr_0.001 | 0.0462 | 0.0006 | 0.0003 | 0.0447 | 0.0476 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A        | Optimizer B     |    Mean A |    Mean B | Better             |   p-value | Significant   | Metric     |
|:-------------------|:----------------|----------:|----------:|:-------------------|----------:|:--------------|:-----------|
| milo_LW_baseline_lr | milo_LW_lr_0.1   | 0.0935839 | 0.103021  | milo_LW_baseline_lr | 0.813666  |               | final_loss |
| milo_LW_baseline_lr | milo_LW_lr_0.001 | 0.0935839 | 0.0461728 | milo_LW_lr_0.001    | 0.0798563 |               | final_loss |
| milo_LW_lr_0.1      | milo_LW_lr_0.001 | 0.103021  | 0.0461728 | milo_LW_lr_0.001    | 0.231198  |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_baseline_lr | 97.3244 | 0.7259 | 0.4191 | 95.5213 | 99.1276 |
| milo_LW_lr_0.1 | 97.0350 | 1.8486 | 1.0673 | 92.4429 | 101.6271 |
| milo_LW_lr_0.001 | 98.7650 | 0.0219 | 0.0126 | 98.7107 | 98.8193 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A        | Optimizer B     |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric         |
|:-------------------|:----------------|---------:|---------:|:-------------------|----------:|:--------------|:---------------|
| milo_LW_baseline_lr | milo_LW_lr_0.1   |  97.3244 |   97.035 | milo_LW_baseline_lr |  0.819299 |               | final_accuracy |
| milo_LW_baseline_lr | milo_LW_lr_0.001 |  97.3244 |   98.765 | milo_LW_lr_0.001    |  0.075076 |               | final_accuracy |
| milo_LW_lr_0.1      | milo_LW_lr_0.001 |  97.035  |   98.765 | milo_LW_lr_0.001    |  0.246465 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_baseline_lr | 0.9742 | 0.0067 | 0.0039 | 0.9576 | 0.9909 |
| milo_LW_lr_0.1 | 0.9710 | 0.0176 | 0.0101 | 0.9274 | 1.0146 |
| milo_LW_lr_0.001 | 0.9877 | 0.0002 | 0.0001 | 0.9871 | 0.9882 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A        | Optimizer B     |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric         |
|:-------------------|:----------------|---------:|---------:|:-------------------|----------:|:--------------|:---------------|
| milo_LW_baseline_lr | milo_LW_lr_0.1   | 0.974227 | 0.971023 | milo_LW_baseline_lr | 0.790004  |               | final_f1_score |
| milo_LW_baseline_lr | milo_LW_lr_0.001 | 0.974227 | 0.987677 | milo_LW_lr_0.001    | 0.0736649 |               | final_f1_score |
| milo_LW_lr_0.1      | milo_LW_lr_0.001 | 0.971023 | 0.987677 | milo_LW_lr_0.001    | 0.2421    |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

