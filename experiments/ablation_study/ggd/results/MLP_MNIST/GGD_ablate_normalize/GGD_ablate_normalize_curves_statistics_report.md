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
| milo_baseline_normalize | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| milo_normalize_False | 0.0029 | 0.0021 | 0.0012 | -0.0023 | 0.0081 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A            | Optimizer B         |      Mean A |     Mean B | Better                 |   p-value | Significant   | Metric     |
|:-----------------------|:--------------------|------------:|-----------:|:-----------------------|----------:|:--------------|:-----------|
| milo_baseline_normalize | milo_normalize_False | 6.55444e-05 | 0.00292368 | milo_baseline_normalize |  0.141793 |               | final_loss |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Accuracy Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_baseline_normalize | 99.9978 | 0.0010 | 0.0006 | 99.9954 | 100.0002 |
| milo_normalize_False | 99.9928 | 0.0067 | 0.0039 | 99.9760 | 100.0095 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A            | Optimizer B         |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric         |
|:-----------------------|:--------------------|---------:|---------:|:-----------------------|----------:|:--------------|:---------------|
| milo_baseline_normalize | milo_normalize_False |  99.9978 |  99.9928 | milo_baseline_normalize |  0.326749 |               | final_accuracy |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### F1 Score Statistics

Number of runs: 3

#### Final Values (last epoch)

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|-------------|
| milo_baseline_normalize | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_normalize_False | 0.9999 | 0.0001 | 0.0000 | 0.9998 | 1.0001 |

#### Pairwise Significance Tests

The following table shows pairwise statistical significance tests between optimizers:

| Optimizer A            | Optimizer B         |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric         |
|:-----------------------|:--------------------|---------:|---------:|:-----------------------|----------:|:--------------|:---------------|
| milo_baseline_normalize | milo_normalize_False | 0.999978 | 0.999928 | milo_baseline_normalize |  0.327568 |               | final_f1_score |

Significance levels: * p<0.05, ** p<0.01, *** p<0.001

