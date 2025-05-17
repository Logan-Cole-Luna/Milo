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
| milo_LW_baseline_normalize | 0.0352 | 0.0035 | 0.0020 | 0.0265 | 0.0440 |
| milo_LW_normalize_False | 0.0338 | 0.0058 | 0.0034 | 0.0193 | 0.0483 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A               | Optimizer B            |    Mean A |   Mean B | Better                 |   p-value | Significant   | Metric     |
|:--------------------------|:-----------------------|----------:|---------:|:-----------------------|----------:|:--------------|:-----------|
| milo_LW_baseline_normalize | milo_LW_normalize_False | 0.0352498 | 0.033826 | milo_LW_normalize_False |  0.739556 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_baseline_normalize | 99.1256 | 0.1268 | 0.0732 | 98.8105 | 99.4407 |
| milo_LW_normalize_False | 99.0067 | 0.1679 | 0.0969 | 98.5896 | 99.4237 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A               | Optimizer B            |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric         |
|:--------------------------|:-----------------------|---------:|---------:|:--------------------------|----------:|:--------------|:---------------|
| milo_LW_baseline_normalize | milo_LW_normalize_False |  99.1256 |  99.0067 | milo_LW_baseline_normalize |  0.387001 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_LW_baseline_normalize | 0.9913 | 0.0013 | 0.0007 | 0.9881 | 0.9944 |
| milo_LW_normalize_False | 0.9900 | 0.0017 | 0.0010 | 0.9859 | 0.9942 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A               | Optimizer B            |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric         |
|:--------------------------|:-----------------------|---------:|---------:|:--------------------------|----------:|:--------------|:---------------|
| milo_LW_baseline_normalize | milo_LW_normalize_False | 0.991257 | 0.990035 | milo_LW_baseline_normalize |  0.374222 |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

